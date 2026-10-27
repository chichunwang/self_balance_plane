import numpy as np
import pytest

pytest.importorskip('mujoco')

from ppsim import mjcheck  # noqa: E402
from ppsim.mjplant import calibrate_contact  # noqa: E402
from ppsim.params import Config  # noqa: E402
from ppsim.sim import Scenario, simulate  # noqa: E402


def test_contact_restitution_matches_target():
    _, table = calibrate_contact(Config())
    assert abs(table[2.0] - 0.87) < 0.005
    assert all(abs(e - 0.87) < 0.02 for e in table.values())


def test_mujoco_closed_chain_matches_our_kinematics():
    dp_mm, dn_deg = mjcheck.geometry_check(n=5)
    assert dp_mm < 0.5 and dn_deg < 0.05


def test_mujoco_rolling_matches_hollow_ball_theory():
    assert abs(mjcheck.rolling_check() - 1) < 0.03


def test_live_mode_loop_runs_with_keys_and_overlays(monkeypatch):
    """用假視窗跑一小段即時模式：按鍵、疊加圖層、曲線圖都要走得通（不開真的視窗）。"""
    import contextlib

    import mujoco
    import numpy as np

    from ppsim import live

    class FakeViewer:
        def __init__(self, m, d, key_callback=None, **kw):
            self.kc, self.frames = key_callback, 0
            self.user_scn = mujoco.MjvScene(m, maxgeom=1000)
            self.cam = type('Cam', (), {'lookat': np.zeros(3), 'distance': 1, 'azimuth': 0, 'elevation': 0})()
            self.viewport = mujoco.MjrRect(0, 0, 1280, 800)
            self.texts = self.figs = None
            self.locked = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @contextlib.contextmanager
        def lock(self):
            self.locked = True
            try:
                yield
            finally:
                self.locked = False

        # 真的 MuJoCo 會在這兩個呼叫裡等繪圖執行緒收走上一份；繪圖執行緒收之前要先拿同一把鎖，
        # 所以拿著 lock() 呼叫它們 = 兩邊互等，視窗「沒有回應」而且關不掉
        def set_texts(self, t):
            assert not self.locked, 'set_texts 不能在 lock() 裡呼叫'
            self.texts = t

        def set_figures(self, f):
            assert not self.locked, 'set_figures 不能在 lock() 裡呼叫'
            self.figs = f

        def sync(self):
            self.frames += 1
            if self.frames in script:                   # 真的 MuJoCo 是繪圖執行緒在按鍵按下時呼叫
                self.kc(script.pop(self.frames))

        def is_running(self):
            return self.frames < 130

    holder, calls = {}, []
    script = {20: 328, 40: ord('9'), 80: ord('6'), 100: ord('7')}      # 328＝數字鍵盤的 8

    def launch(m, d, **kw):
        assert callable(kw.get('key_callback'))
        holder['v'] = FakeViewer(m, d, **kw)
        return holder['v']

    def counted(name, f):
        return lambda cfg, rng: calls.append(name) or f(cfg, rng)

    monkeypatch.setattr(live.mujoco.viewer, 'launch_passive', launch)
    monkeypatch.setattr(live, '_toss', counted('toss', live._toss))
    monkeypatch.setattr(live, '_drop', counted('drop', live._drop))
    run = live.run_live(seed=3)
    v = holder['v']
    status = v.texts[0][3]
    assert not script and calls == ['drop', 'toss', 'drop']       # 開場一顆、8 側拋、7 上方丟
    assert 'ON' in status and 'x0.50' in status                   # 9 開了拍球、6 切到半速
    assert run.t > 1.0 and v.user_scn.ngeom > 0 and v.figs is not None and 'mode' in v.texts[0][2]


def test_mujoco_plant_balances_and_recovers_a_hand_push():
    """期中目標在 MuJoCo 上也要成立，標準跟純 Python 版（test_control）一樣：球靜止放偏 3.6 cm，4 秒內拉回中心 1 cm 以內；
    5 秒時被推成 0.25 m/s 滾出去，3 秒內回到 1 cm 以內。"""
    cfg = Config()
    z = cfg.geom.z_surface0 + cfg.ball.radius
    sc = Scenario('push', duration=9.0, seed=1, start_p=(0.03, -0.02, z), start_v=(0.0, 0.0, 0.0), push=(5.0, 0.0, 0.25))
    res = simulate(cfg, sc, plant='mj')
    t, p = res.log['t'], res.log['p']
    r = np.hypot(p[:, 0], p[:, 1])
    assert res.fell is None
    assert r[(t > 4.0) & (t < 5.0)].max() < 0.01        # 推之前已經穩在中心
    assert r[t >= 8.0].max() < 0.01                      # 推完 3 秒內回來


def test_same_controller_catches_on_mujoco_plant():
    m = simulate(Config(), Scenario('catch', duration=3.5, seed=1), plant='mj').metrics
    assert m['fell'] is None and m['t_balance'] < 3.0 and m['final_r'] < 0.003
    assert m['torque_peak'] < 0.6 and m['duty_sat'] == 0.0
