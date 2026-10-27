"""即時控制模式：控制器在 MuJoCo 視窗裡即時接球、拍球、平衡，可以用滑鼠推球或把球甩過去。

視窗裡的疊加文字只能用英數字（MuJoCo 內建字型沒有中文），完整說明印在終端機。
按鍵用數字 6～9：MuJoCo 視窗已經把 A–Z 和 0–5 都綁成顯示選項了。

視窗由 MuJoCo 的繪圖執行緒負責，它和主迴圈共用一把視窗鎖（v.lock()，sync 內部也會拿）。
set_texts／set_figures 會等繪圖執行緒收走上一份，而它收之前要先拿這把鎖，
所以這兩個呼叫絕對不能放在 `with v.lock():` 裡，不然兩邊互等，視窗「沒有回應」而且關不掉。
保險：任何一幀卡超過 HANG_EXIT_S 秒，faulthandler 會把各執行緒卡在哪一行印到終端機並結束行程。
"""
import contextlib
import faulthandler
import math
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

from .params import Config
from .sim import Runner, Scenario, toss

KEY_SLOW, KEY_DROP, KEY_TOSS, KEY_JUGGLE = ord('6'), ord('7'), ord('8'), ord('9')
KEYPAD = {320 + i: ord('0') + i for i in range(10)}     # GLFW 的數字鍵盤 0–9 當成數字列
SPEEDS = (1.0, 0.5, 0.25)
HANG_EXIT_S = 20.0
MODE_EN ={'JUGGLE': 'JUGGLE', 'CUSHION': 'CATCH (cushion)', 'SETTLE': 'SETTLE', 'BALANCE': 'BALANCE'}
HELP = '''
即時控制模式（MuJoCo 視窗）
  6  慢動作：1× → 0.5× → 0.25×
  7  從上方丟一顆球
  8  從旁邊隨機方向拋弧線過來
  9  拍球開關（開：空中的球改成一直拍；關：這一跳之後緩衝接住、平衡）
  滑鼠：在球上雙擊選取，按住 Ctrl＋右鍵拖曳可以推球或把球甩出去
  視窗裡：橘點＝球剛走過的軌跡、綠圈＝控制器預測的落點、藍色半透明球＝控制器以為球在哪
  結束：關掉視窗，或在這裡按 Ctrl+C（萬一卡住超過 20 秒，程式會自己結束並在這裡印出卡在哪）
'''


@contextlib.contextmanager
def _hang_guard():
    """主迴圈每幀重新計時（dump_traceback_later）；離開時一定取消，免得之後誤殺。"""
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _drop(cfg, rng):
    xy = rng.uniform(-0.03, 0.03, 2)
    p = [xy[0], xy[1], cfg.geom.z_surface0 + cfg.ball.radius + rng.uniform(0.25, 0.35)]
    return p, [*rng.uniform(-0.05, 0.05, 2), 0.0]


def _toss(cfg, rng):
    """隨機方向的側拋：水平速度約 0.6–1.05 m/s（模擬裡大多接得住；再快就要靠運氣）"""
    sc = toss(cfg, rng.uniform(-0.02, 0.02, 2), rng.uniform(0, 360), rng.uniform(0.25, 0.42),
              apex=rng.uniform(0.3, 0.4), launch_h=0.25)
    return list(sc.start_p), list(sc.start_v)


class Overlay:
    """視窗上的軌跡殘影、落點標記、估測位置，以及右下角的高度曲線。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.trail = deque(maxlen=60)          # 最近 1.2 秒（每 20 ms 一點）
        self.hist = deque(maxlen=300)          # 最近 3 秒（每 10 ms 一點）：(t, 球高, 平台升降)
        self._next_trail = self._next_hist = 0.0
        fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(fig)
        fig.title = 'height above plate (mm)'
        fig.flg_legend = 1
        fig.linename[0] = 'ball'
        fig.linename[1] = 'plate'
        fig.linergb[0] = [1.0, 0.55, 0.1]
        fig.linergb[1] = [0.45, 0.65, 1.0]
        fig.range[1] = [-30, 420]
        fig.gridsize = [4, 5]
        self.fig = fig

    def sample(self, run):
        t, w = run.t, run.world
        if t >= self._next_trail:
            self.trail.append(w.ball_p.copy())
            self._next_trail = t + 0.02
        if t >= self._next_hist:
            gap = float((w.ball_p - w.point) @ w.n) - self.cfg.ball.radius
            self.hist.append((t, gap * 1e3, (w.c[2] - self.cfg.geom.z_joint0) * 1e3))
            self._next_hist = t + 0.01

    def draw(self, v, run):
        """改 user_scn 要拿視窗鎖；set_figures 一定要在鎖外呼叫（原因見模組說明）。"""
        x, _ = run.est.predict(run.t)
        with v.lock():
            self._geoms(v.user_scn, run, x)
        if self.hist:
            t_now = self.hist[-1][0]
            h = np.array(self.hist)
            for j, col in ((0, 1), (1, 2)):
                self.fig.linepnt[j] = len(h)
                self.fig.linedata[j][:2 * len(h)] = np.column_stack([h[:, 0] - t_now, h[:, col]]).ravel()
            vp = v.viewport
            if vp is not None:
                width = int(min(520, vp.width * 0.36))
                rect = mujoco.MjrRect(vp.width - width, 0, width, int(width * 0.62))
                v.set_figures([(rect, self.fig)])

    def _geoms(self, scn, run, x):
        w = run.world
        scn.ngeom = 0
        eye = np.eye(3).flatten()

        def add(gtype, size, pos, mat, rgba):
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                                    np.asarray(mat, float), np.asarray(rgba, np.float32))
                scn.ngeom += 1

        n = len(self.trail)
        for i, p in enumerate(self.trail):
            a = 0.15 + 0.6 * (i + 1) / max(n, 1)
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], p, eye, [1.0, 0.5, 0.1, a])
        hits = [h for h in run.ctrl.hits if h['t_hit'] > run.t]
        if hits and run.ctrl.mode in ('JUGGLE', 'CUSHION'):
            ph = hits[-1]['p_hit']
            pos = w.point + (ph[0] - w.point[0]) * w.R[:, 0] + (ph[1] - w.point[1]) * w.R[:, 1] + 0.001 * w.n
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.0008, 0], pos, w.R.flatten(), [0.2, 0.85, 0.3, 0.8])
        if x is not None:
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [self.cfg.ball.radius * 1.05, 0, 0], x[:3], eye, [0.3, 0.55, 1.0, 0.3])


def run_live(cfg=None, juggle=False, speed=1.0, seed=None, max_wall=None):
    """max_wall：跑幾秒自動關閉（測試用）"""
    cfg = cfg or Config()
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    p0, v0 = _drop(cfg, rng)
    sc = Scenario('live', duration=math.inf, start_p=tuple(p0), start_v=tuple(v0),
                  juggle_until=math.inf if juggle else 0.0, seed=int(rng.integers(1 << 30)))
    run = Runner(cfg, sc, plant='mj')
    m, d = run.world.m, run.world.d
    pressed = []                        # 繪圖執行緒在按鍵按下時 append，主迴圈每幀取走
    ov = Overlay(cfg)
    speed_i = SPEEDS.index(speed) if speed in SPEEDS else 0
    print(HELP)
    with _hang_guard(), mujoco.viewer.launch_passive(m, d, key_callback=pressed.append,
                                                     show_left_ui=False, show_right_ui=False) as v:
        with v.lock():
            v.cam.lookat[:] = [0.0, 0.0, 0.22]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.95, 130.0, -22.0
        wall_ref, sim_ref = time.perf_counter(), run.t
        rtf, last = 1.0, (time.perf_counter(), run.t)
        lost_at = None
        while v.is_running():
            faulthandler.dump_traceback_later(HANG_EXIT_S, exit=True)
            f0 = time.perf_counter()
            while pressed:
                k = pressed.pop(0)
                k = KEYPAD.get(k, k)
                if k == KEY_SLOW:
                    speed_i = (speed_i + 1) % len(SPEEDS)
                    wall_ref, sim_ref = f0, run.t
                elif k in (KEY_DROP, KEY_TOSS):
                    p, vel = (_drop if k == KEY_DROP else _toss)(cfg, rng)
                    run.new_ball(p, vel, juggle)
                    ov.trail.clear()
                    lost_at = None
                elif k == KEY_JUGGLE:
                    juggle = not juggle
                    run.ctrl.set_juggle(juggle, run.t)
            target = sim_ref + (f0 - wall_ref) * SPEEDS[speed_i]
            n = 0
            while run.t < target and n < 120:
                run.step()
                ov.sample(run)
                n += 1
            if run.t < target - 0.05:           # 電腦跟不上：接受變慢，不要一直追
                wall_ref, sim_ref = time.perf_counter(), run.t
            now = time.perf_counter()
            if now - last[0] > 0.5:
                rtf = (run.t - last[1]) / (now - last[0])
                last = (now, run.t)
            w = run.world
            on_plate = float(np.hypot(*(w.ball_p - w.point)[:2])) < cfg.geom.plate_diameter / 2
            if not on_plate and w.ball_p[2] < w.point[2] - 0.02:
                lost_at = lost_at or run.t
            fl = run.est.flights
            status = [('mode', MODE_EN.get(run.ctrl.mode, run.ctrl.mode)),
                      ('juggle', 'ON (target %.0f mm)' % (cfg.control.apex * 1e3) if juggle else 'off'),
                      ('last flight', '%.0f mm high' % (fl[-1][2] * 1e3) if fl else '-'),
                      ('e (learned)', '%.3f' % run.est.e_hat),
                      ('auto-level', '%.2f deg' % math.degrees(math.atan(float(np.linalg.norm(run.ctrl.tilt_trim))))),
                      ('time', '%.1f s  x%.2f  (sim/real %.2f)' % (run.t, SPEEDS[speed_i], rtf))]
            if lost_at is not None:
                status.append(('BALL LOST', 'press 7 or 8'))
            ov.draw(v, run)
            v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          '\n'.join(a for a, _ in status), '\n'.join(b for _, b in status)),
                         (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                          '[6] slow-mo\n[7] drop from above\n[8] toss from the side\n[9] juggle on/off\n'
                          '[dbl-click ball + Ctrl+right-drag] push / fling', '')])
            v.sync()
            if max_wall is not None and time.perf_counter() - t_start > max_wall:
                break
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - f0)))
    return run
