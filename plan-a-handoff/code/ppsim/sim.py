"""模擬主迴圈：真實世界（伺服→連桿→平台→球）與控制迴路（相機、壓電片→估測→控制）接在一起。

真實世界可以是純 Python 版（plant.PyPlant）或 MuJoCo 版（mjplant.MjPlant），控制迴路完全相同。
Runner 一次推進一步，批次模擬（simulate）和即時模式（live）共用。
"""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .ball import Surface
from .controller import MODES, Controller
from .estimator import Estimator
from .kinematics import Platform
from .plant import PyPlant
from .sensors import Camera, Piezo


@dataclass
class Scenario:
    name: str = 'catch'
    duration: float = 5.0
    drop_height: float = 0.30          # 球底離板面
    drop_xy: tuple = (0.035, -0.020)
    drop_vxy: tuple = (0.05, 0.03)
    juggle_until: float = 0.0          # >0：先拍球到這個時間，再緩衝接住
    seed: int = 1
    start_p: tuple | None = None       # 有給就蓋過上面的丟球設定（側拋用）
    start_v: tuple | None = None
    push: tuple | None = None          # (時間, vx, vy)：這時候用手推球，球變成這個速度貼著板子滾（期中的「手推回復」）


def toss(cfg, land_xy, azimuth_deg, dist, apex, launch_h=0.25, **kw):
    """從旁邊拋弧線：在離落點水平 dist 的地方、板面上方 launch_h 出手，弧線最高點離板面 apex，
    落在 land_xy。azimuth 是「從落點看出手位置」的方向。考慮空氣阻力，用打靶法把初速修到剛好落在目標。"""
    b = cfg.ball
    z0 = cfg.geom.z_surface0 + b.radius
    d = np.array([math.cos(math.radians(azimuth_deg)), math.sin(math.radians(azimuth_deg))])

    def fly(vh, vz):
        """回傳（落回板面高度時飛了多遠、最高點離板面多高）"""
        x, z, vx, vzz, top, h = 0.0, launch_h, vh, vz, launch_h, 0.001
        for _ in range(3000):
            s = math.hypot(vx, vzz)
            vx -= b.beta * s * vx * h
            vzz -= (b.g_eff + b.beta * s * vzz) * h
            x, z = x + vx * h, z + vzz * h
            top = max(top, z)
            if vzz < 0 and z <= 0:
                return x, top
        return x, top

    g = b.g_eff
    t_up, t_dn = math.sqrt(2 * (apex - launch_h) / g), math.sqrt(2 * apex / g)
    q = np.array([dist / (t_up + t_dn), g * t_up])
    for _ in range(8):                  # 牛頓法修正水平與垂直初速
        f = np.array(fly(*q)) - [dist, apex]
        if np.max(np.abs(f)) < 1e-4:
            break
        J = np.empty((2, 2))
        for j in range(2):
            dq = np.zeros(2)
            dq[j] = 1e-4
            J[:, j] = (np.array(fly(*(q + dq))) - [dist, apex] - f) / 1e-4
        q = q - np.linalg.solve(J, f)
    p0 = np.asarray(land_xy) + d * dist
    kw.setdefault('name', 'toss')
    return Scenario(start_p=(p0[0], p0[1], z0 + launch_h), start_v=(-d[0] * q[0], -d[1] * q[0], q[1]), **kw)


@dataclass
class Result:
    cfg: object
    sc: Scenario
    log: dict
    impacts: list
    hits: list
    events: list
    est_track: list
    fell: float | None
    e_log: list = field(default_factory=list)
    trim_log: list = field(default_factory=list)
    plant: str = ''
    frames: list = field(default_factory=list)     # (時間, MuJoCo qpos)，做 3D 動畫用
    metrics: dict = field(default_factory=dict)


class Runner:
    """感測器→估測器→控制器→真實世界，一次推進一步。"""

    def __init__(self, cfg, sc, plant='py'):
        self.cfg, self.sc = cfg, sc
        rng = np.random.default_rng(sc.seed)
        gm = cfg.geom
        if plant == 'mj':
            from .mjplant import MjPlant
            self.world = MjPlant(cfg, sc, rng)
        else:
            self.world = PyPlant(cfg, sc, rng)
        self.cam = Camera(cfg.camera, gm.z_surface0 + cfg.camera.height, cfg.ball.radius, rng)
        self.piezo = Piezo(cfg.piezo, rng)
        self.ctrl = Controller(cfg, Platform(gm), sc.juggle_until)
        self.est = None
        self._new_estimator()
        self.dt = cfg.dt
        self.every = max(1, int(round(1 / (cfg.control.rate_hz * cfg.dt))))
        self.queue, self.seq, self.k = [], 0, 0
        self.impacts, self.est_track = [], []
        self.th_cmd = np.zeros(3)

    @property
    def t(self):
        return self.k * self.dt

    def _new_estimator(self):
        e_hat = None if self.est is None else self.est.e_hat
        self.est = Estimator(self.cfg, self.cam, self.ctrl.plate_at)
        if e_hat is not None:
            self.est.e_hat = e_hat              # 學到的恢復係數留著
        self.ctrl.est = self.est

    def _push(self, t_deliver, kind, item):
        heapq.heappush(self.queue, (t_deliver, self.seq, kind, item))
        self.seq += 1

    def step(self):
        t = self.t
        # 感測器：拍照（結果晚一點才送到）、送達到期的資料
        shot = self.cam.capture(t, self.world.ball_p)
        if shot is not None:
            self._push(shot[0], 'cam', shot[1:])
        while self.queue and self.queue[0][0] <= t:
            _, _, kind, item = heapq.heappop(self.queue)
            if kind == 'cam':
                self.est.on_camera(*item, t)
                if self.est.t is not None:          # 還在鎖定（只看到一張）時沒有估計值
                    self.est_track.append((self.est.t, list(self.est.x), self.est.mode))
            else:
                self.est.on_piezo(item)
        # 控制器
        if self.k % self.every == 0:
            self.th_cmd = self.ctrl.update(t)
            self.world.command(self.th_cmd)
        # 真實世界往前一步
        for imp in self.world.step(t, self.dt):
            self.impacts.append(imp)
            ev = self.piezo.hit(imp['t'], -imp['vn'])
            if ev is not None:
                self._push(ev[0], 'piezo', ev[1])
        self.k += 1

    def push_ball(self, vxy):
        """手推一下：球留在原地，速度改成貼著板面的 vxy，自轉也改成剛好純滾動（推完很快就會變成這樣）。"""
        n = self.world.n
        v = np.array([vxy[0], vxy[1], 0.0])
        v = v - (v @ n) * n
        w = np.cross(n, v) / self.cfg.ball.radius
        self.world.reset_ball(list(self.world.ball_p), list(v), list(w))

    def new_ball(self, p, v, juggle):
        """即時模式重新丟一顆球：感測資料清空、估測器重來；學到的 e、自動水平修正留著。"""
        self.world.reset_ball(p, v)
        self.queue = []
        self._new_estimator()
        self.ctrl.new_ball(self.t, juggle)


def simulate(cfg, sc, plant='py', frames_every=0.0):
    """plant：'py' 純 Python、'mj' MuJoCo。frames_every>0 時每隔這麼久存一份 MuJoCo 狀態（做 3D 動畫用）。"""
    run = Runner(cfg, sc, plant)
    world, ctrl = run.world, run.ctrl
    rb, radius = cfg.ball.radius, cfg.geom.plate_diameter / 2
    n_steps = int(round(sc.duration / cfg.dt))
    cols = [('p', 3), ('v', 3), ('c', 3), ('n', 3), ('th', 3), ('th_cmd', 3), ('om', 3)]
    if world.extras() is not None:
        cols += [('torque', 3), ('duty', 3), ('rod_force', 3)]
    log = {k: np.zeros((n_steps, w)) for k, w in cols}
    log.update({k: np.zeros(n_steps) for k in ('t', 'gap', 'mode', 'plan_z')})
    mode_code = {m: i for i, m in enumerate(MODES)}
    frames, next_frame = [], 0.0
    fell, k_end = None, n_steps

    pushed = sc.push is None
    for k in range(n_steps):
        run.step()
        if not pushed and run.t >= sc.push[0]:
            run.push_ball(sc.push[1:])
            pushed = True
        t, p = run.t, world.ball_p
        surf = Surface(world.point, world.n, None, None, radius)
        log['t'][k] = t
        log['p'][k], log['v'][k] = p, world.ball_v
        log['c'][k], log['n'][k] = world.c, world.n
        log['th'][k], log['th_cmd'][k] = world.th, run.th_cmd
        log['om'][k] = world.om
        log['gap'][k] = surf.gap(p, rb)
        log['mode'][k] = mode_code[ctrl.mode]
        log['plan_z'][k] = ctrl.plate_at(t)[0]
        ex = world.extras()
        if ex is not None:
            for key, val in ex.items():
                log[key][k] = val
        if frames_every and t >= next_frame:
            frames.append((t, world.d.qpos.copy()))
            next_frame += frames_every
        if not surf.inside(p) and p[2] < world.point[2] - 0.02:
            fell, k_end = t, k + 1
            break

    log = {key: arr[:k_end] for key, arr in log.items()}
    res = Result(cfg, sc, log, run.impacts, ctrl.hits, ctrl.events, run.est_track, fell, run.est.e_log,
                 ctrl.trim_log, world.name, frames)
    res.metrics = metrics(res)
    return res


def metrics(res):
    L, sc, cfg = res.log, res.sc, res.cfg
    t = L['t']
    strong = [i for i in res.impacts if -i['vn'] > 0.1]
    t_imp = [i['t'] for i in strong]
    apex = []                            # (撞擊時間, 下一次撞擊前的最高點)
    for a, b in zip(t_imp, t_imp[1:] + [t[-1]]):
        sel = (t > a) & (t < b)
        if sel.any():
            apex.append((a, float(L['gap'][sel].max())))
    out = dict(bounces=len(strong), fell=res.fell)
    # 拍球階段（跳過前 1.5 秒的收斂期）
    if sc.juggle_until > 0:
        hs = [h for a, h in apex if 1.5 < a < sc.juggle_until - 0.3]
        rs = [float(np.hypot(*i['p'][:2])) for i in strong if 1.5 < i['t'] < sc.juggle_until]
        if hs:
            out.update(apex_mean=float(np.mean(hs)), apex_std=float(np.std(hs)))
        if rs:
            out.update(land_r_mean=float(np.mean(rs)), land_r_max=float(np.max(rs)))
        late = [i['p'][:2] for i in strong if sc.juggle_until / 2 < i['t'] < sc.juggle_until]
        if late:                         # 後半段落點的平均位置：有系統性偏移（例如平台沒調水平）就看得出來
            out['land_offset'] = float(np.linalg.norm(np.mean(late, axis=0)))
    bal = np.nonzero(L['mode'] == MODES.index('BALANCE'))[0]
    if len(bal):
        out['t_balance'] = float(t[bal[0]])
        tail = t > t[-1] - 1.0
        out['final_r'] = float(np.hypot(L['p'][tail, 0], L['p'][tail, 1]).mean())
    # 撞擊瞬間平台速度：規劃值 vs 實際值
    errs, dts = [], []
    for h in res.hits:
        cand = [i for i in strong if abs(i['t'] - h['t_hit']) < 0.05]
        if cand:
            errs.append(float(cand[0]['v_plate'][2] - h['w']))
            dts.append(cand[0]['t'] - h['t_hit'])
    if errs:
        out['hit_w_err_rms'] = float(np.sqrt(np.mean(np.square(errs))))
        out['hit_dt_rms_ms'] = float(np.sqrt(np.mean(np.square(dts))) * 1e3)
        out['hit_dt_max_ms'] = float(np.max(np.abs(dts)) * 1e3)
    # 伺服轉速峰值：簡化模型對它自己的上限（打過折），MuJoCo 馬達模型對空載轉速
    w_lim = cfg.servo.w_noload if 'torque' in L else cfg.servo.w_max
    out['servo_peak'] = float(np.abs(L['om']).max() / w_lim)
    if 'torque' in L:
        out['torque_peak'] = float(np.abs(L['torque']).max() / cfg.servo.stall_torque)
        out['torque_rms'] = float(np.sqrt(np.mean(np.square(L['torque']))) / cfg.servo.stall_torque)
        out['duty_sat'] = float(np.mean(np.abs(L['duty']).max(axis=1) > 0.99))
        out['rod_force_max'] = float(L['rod_force'].max())
    heave = L['c'][:, 2] - cfg.geom.z_joint0
    out['heave_range_mm'] = (float(heave.min() * 1e3), float(heave.max() * 1e3))
    return out
