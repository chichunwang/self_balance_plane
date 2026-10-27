"""MuJoCo 驗證：幾何、接觸、閉迴路對照、機構負載、敏感度，以及 3D 動畫與互動模型匯出。"""
import math

import mujoco
import numpy as np

from .kinematics import Platform
from .mjplant import CONTACT_WN, build_mjcf, calibrate_contact
from .params import Config


def geometry_check(cfg=None, n=20, seed=0):
    """把三顆伺服鎖在 IK 算出的角度，讓 MuJoCo 的閉鏈自己穩下來，再和我們的正解比。
    回傳 (平台位置最大差 mm, 平台法向量最大差 °)"""
    cfg = cfg or Config()
    plat = Platform(cfg.geom)
    m = mujoco.MjModel.from_xml_string(build_mjcf(cfg, ball_pos=(0.5, 0.5, 0.02)))
    d = mujoco.MjData(m)
    qa = [m.jnt_qposadr[m.joint(f'servo{i}').id] for i in range(3)]
    va = [m.jnt_dofadr[m.joint(f'servo{i}').id] for i in range(3)]
    pid = m.body('plate').id
    rng = np.random.default_rng(seed)
    c0, R0 = plat.neutral()
    dp, dn = [], []
    for _ in range(n):
        tilt = math.tan(math.radians(rng.uniform(0, 7)))
        ang = rng.uniform(0, 2 * math.pi)
        c, R, _ = plat.pose(rng.uniform(-0.015, 0.015), tilt * math.cos(ang), tilt * math.sin(ang))
        th, _ = plat.ik(c, R)
        mujoco.mj_resetData(m, d)
        for _ in range(1200):                   # 0.3 s，讓約束收斂
            for a, v, x in zip(qa, va, th):
                d.qpos[a], d.qvel[v] = x, 0.0
            mujoco.mj_step(m, d)
        c_fk, R_fk = plat.fk(th, c0, R0, iters=30)
        dp.append(np.linalg.norm(d.xpos[pid] - c_fk) * 1e3)
        n_mj = d.xmat[pid].reshape(3, 3)[:, 2]
        dn.append(math.degrees(math.acos(min(1.0, float(n_mj @ R_fk[:, 2])))))
    return max(dp), max(dn)


def rolling_check(cfg=None, slope_deg=5.0, T=0.4):
    """球放在傾斜 5° 的固定板上，量沿斜面的加速度，和薄殼球理論值 0.6·g·sinθ 比。"""
    cfg = cfg or Config()
    b, s = cfg.ball, cfg.surface
    damping, _ = calibrate_contact(cfg)
    th = math.radians(slope_deg)
    inert = b.k_inertia * b.mass * b.radius ** 2
    n = np.array([math.sin(th), 0.0, math.cos(th)])
    p0 = n * (b.radius + 1e-5)
    xml = f'''<mujoco><compiler angle="radian"/>
<option timestep="{cfg.mech.mj_dt}" gravity="0 0 {-b.g_eff}" cone="elliptic" impratio="10"/>
<worldbody><geom name="plate" type="box" size="0.4 0.2 0.003" pos="{-n[0] * 0.003} 0 {-n[2] * 0.003}"
 euler="0 {th} 0" contype="0" conaffinity="0"/>
<body name="ball" pos="{p0[0]} 0 {p0[2]}"><freejoint/>
<inertial pos="0 0 0" mass="{b.mass}" diaginertia="{inert} {inert} {inert}"/>
<geom name="ball" size="{b.radius}" contype="0" conaffinity="0"/></body></worldbody>
<contact><pair geom1="ball" geom2="plate" condim="6" friction="{s.mu} {s.mu} 1e-4 4e-5 4e-5"
solref="{-CONTACT_WN ** 2} {-damping}" solimp="0.99 0.99 0.001 0.5 2"/></contact></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(int(T / m.opt.timestep)):
        mujoco.mj_step(m, d)
    along = np.array([math.cos(th), 0.0, -math.sin(th)])
    return float(d.qvel[:3] @ along) / T / (0.6 * b.g_eff * math.sin(th))


def export_viewer_model(path, cfg=None):
    """給 MuJoCo 互動視窗用：伺服改成位置型致動器，可以在 Control 面板拉桿讓平台動。"""
    cfg = cfg or Config()
    g = cfg.geom
    path.write_text(build_mjcf(cfg, ball_pos=(0.03, -0.02, g.z_surface0 + cfg.ball.radius + 0.30), viewer=True),
                    encoding='utf-8')


def render_gif(res, path, camera='iso', size=(560, 420), fps=20):
    """把模擬存下的 MuJoCo 狀態重播成 3D 動畫（左上角標時間與模式）。"""
    from PIL import Image, ImageDraw, ImageFont

    from .controller import MODES
    from .viz import MODE_NAME
    try:
        font = ImageFont.truetype('msjh.ttc', 20)
    except OSError:
        font = ImageFont.load_default()
    m = mujoco.MjModel.from_xml_string(build_mjcf(res.cfg))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, size[1], size[0])
    t_log, modes = res.log['t'], res.log['mode']
    imgs = []
    for t, qpos in res.frames:
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=camera)
        img = Image.fromarray(r.render())
        mode = MODES[int(modes[min(len(modes) - 1, np.searchsorted(t_log, t))])]
        ImageDraw.Draw(img).text((14, 10), f'{t:4.2f} s　{MODE_NAME[mode]}', font=font, fill=(40, 40, 40))
        imgs.append(img.convert('P', palette=Image.ADAPTIVE, colors=256))
    r.close()
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
