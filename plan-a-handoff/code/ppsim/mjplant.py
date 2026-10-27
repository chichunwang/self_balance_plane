"""「真實世界」的 MuJoCo 版：多體動力學（閉鏈 3-RRS）＋馬達伺服＋軟接觸。

- 機構：底座上三個伺服輸出軸（轉軸）→ 臂 → 肘（轉軸）→ 連桿，連桿頂端用 connect 約束（球頭）
  接到自由浮動的平台；平台與球都是自由剛體，質量取自 params.MechSpec。
- 伺服：servo.MotorServo 算齒輪扭力，用 qfrc_applied 加在輸出關節上（電位器量的是輸出軸）。
- 球–板接觸：用「剛性、阻尼」直接指定（solref 取負值）；阻尼由恢復係數 e 反推（calibrate_contact）。
  MuJoCo 預設的（時間常數, 阻尼比）寫法在 e≈0.87 時剛性會大到數值發散。
- 空氣阻力與浮力：每步用 qfrc_applied 加在球上（跟純 Python 版同一個公式）；
  xfrc_applied 留給互動視窗的滑鼠推力用。
"""
import functools
import math

import mujoco
import numpy as np

from .ball import cross
from .kinematics import Platform
from .params import G
from .plant import ball_start
from .servo import MotorServo

CONTACT_WN = 1200.0          # 球–板接觸的固有頻率（rad/s）：每步 ωn·dt≈0.3，穩定


def _v(a):
    return ' '.join(f'{x:.6g}' for x in a)


def _contact_xml(cfg, damping):
    s = cfg.surface
    return (f'<pair name="ball_plate" geom1="ball" geom2="plate" condim="6" '
            f'friction="{s.mu} {s.mu} 1e-4 4e-5 4e-5" solref="{-CONTACT_WN ** 2:.6g} {-damping:.6g}" '
            f'solimp="0.99 0.99 0.001 0.5 2"/>')


@functools.lru_cache(maxsize=8)
def _calibrate(e, mu, radius, mass, k_inertia, dt):
    """在固定平板上用不同速度撞，找出讓恢復係數＝e 的接觸阻尼（二分法）。回傳 (阻尼, {速度: e})"""
    inert = k_inertia * mass * radius ** 2
    xml = f'''<mujoco><option timestep="{dt}" gravity="0 0 0" cone="elliptic" impratio="10"/>
<worldbody><geom name="plate" type="box" size="0.2 0.2 0.003" pos="0 0 -0.003" contype="0" conaffinity="0"/>
<body name="ball" pos="0 0 {radius + 0.004}"><freejoint/>
<inertial pos="0 0 0" mass="{mass}" diaginertia="{inert} {inert} {inert}"/>
<geom name="ball" size="{radius}" contype="0" conaffinity="0"/></body></worldbody>
<contact><pair geom1="ball" geom2="plate" condim="6" friction="{mu} {mu} 1e-4 4e-5 4e-5"
solref="{-CONTACT_WN ** 2} -1" solimp="0.99 0.99 0.001 0.5 2"/></contact></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    def rebound(b, v):
        m.pair_solref[0] = [-CONTACT_WN ** 2, -b]
        mujoco.mj_resetData(m, d)
        d.qvel[2] = -v
        touched = False
        for _ in range(4000):
            mujoco.mj_step(m, d)
            if d.ncon:
                touched = True
            elif touched:
                return d.qvel[2] / v
        return float('nan')

    lo, hi = 0.0, 2 * CONTACT_WN
    for _ in range(40):
        mid = (lo + hi) / 2
        if rebound(mid, 2.0) > e:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    return b, {v: round(rebound(b, v), 4) for v in (0.3, 1.0, 2.0, 3.0)}


def calibrate_contact(cfg):
    b = cfg.ball
    return _calibrate(cfg.surface.e, cfg.surface.mu, b.radius, b.mass, b.k_inertia, cfg.mech.mj_dt)


def build_mjcf(cfg, ball_pos=(0.0, 0.0, 0.5), viewer=False):
    """由 params 產生 MJCF。viewer=True 時另外加位置型致動器，給 MuJoCo 互動視窗拉桿玩。"""
    g, mech, b = cfg.geom, cfg.mech, cfg.ball
    plat = Platform(g)
    R, t, j2s = g.plate_diameter / 2, g.plate_thickness, g.joint_to_surface
    m_plate = mech.plate_density * math.pi * R ** 2 * t
    damping, _ = calibrate_contact(cfg)
    inert = b.k_inertia * b.mass * b.radius ** 2
    legs, eqs, acts = [], [], []
    s = g.arm_sign                      # 臂朝內 +1、朝外 −1；轉軸跟著翻，θ 往上為正才不變
    for i in range(3):
        er, et = plat.er[i], plat.et[i]
        shaft = plat.shaft[i]
        body_c = shaft - s * et * 0.02325 - np.array([0, 0, 0.010])
        legs.append(f'''
    <geom type="box" size="0.010 0.02025 0.020" pos="{_v(body_c)}" xyaxes="{_v(er)} {_v(et)}" material="servo"
          contype="0" conaffinity="0"/>
    <body name="horn{i}" pos="{_v(shaft)}">
      <joint name="servo{i}" type="hinge" axis="{_v(s * et)}" range="-1.05 1.05" armature="1e-5"/>
      <geom type="capsule" fromto="0 0 0 {_v(-s * g.arm * er)}" size="0.004" mass="{mech.arm_mass}" material="arm"
            contype="0" conaffinity="0"/>
      <body name="rod{i}" pos="{_v(-s * g.arm * er)}">
        <joint name="elbow{i}" type="hinge" axis="{_v(s * et)}" damping="1e-4"/>
        <geom type="capsule" fromto="0 0 0 0 0 {g.rod}" size="0.0025" mass="{mech.rod_mass}" material="rod"
              contype="0" conaffinity="0"/>
      </body>
    </body>''')
        eqs.append(f'<connect name="joint{i}" body1="rod{i}" body2="plate" anchor="0 0 {g.rod}" '
                   f'solref="0.002 1" solimp="0.99 0.999 0.0001"/>')
        if viewer:
            acts.append(f'<position name="servo{i}" joint="servo{i}" kp="40" kv="0.3" ctrlrange="-1.05 1.05" '
                        f'forcerange="-{cfg.servo.stall_torque} {cfg.servo.stall_torque}"/>')
    brackets = ''.join(
        f'<geom type="box" size="0.005 0.005 {(j2s - t) / 2:.6g}" pos="{_v(g.r_p * plat.er[i] + [0, 0, (j2s - t) / 2])}" '
        f'mass="0.005" material="rod" contype="0" conaffinity="0"/>' for i in range(3))
    return f'''<mujoco model="pingpong_3rrs">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{mech.mj_dt}" gravity="0 0 {-G}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="1280" offheight="960"/><headlight ambient=".35 .35 .35" diffuse=".55 .55 .55"/>
    <quality shadowsize="4096"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".93 .94 .96" rgb2=".72 .76 .82" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".92 .91 .88" rgb2=".86 .85 .82" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.04"/>
    <material name="wood" rgba=".85 .76 .6 1"/>
    <material name="servo" rgba=".17 .19 .21 1"/>
    <material name="arm" rgba=".9 .55 .18 1" specular=".5"/>
    <material name="rod" rgba=".55 .6 .66 1" specular=".8"/>
    <material name="acrylic" rgba=".6 .82 .95 .45" specular=".7"/>
    <material name="ball" rgba="1 .5 .08 1" specular=".3" emission=".35"/>
    <material name="piezo" rgba=".83 .66 .17 1"/>
  </asset>
  <worldbody>
    <light pos="0.25 -0.3 1.2" dir="-0.2 0.25 -1" diffuse=".7 .7 .7" castshadow="true"/>
    <geom name="floor" type="plane" size="0.8 0.8 0.01" material="floor" contype="2" conaffinity="2"/>
    <geom type="box" size="0.16 0.16 0.004" pos="0 0 0.004" material="wood" contype="0" conaffinity="0"/>
    <body name="focus" pos="0 0 0.24"/>
    <camera name="iso" pos="0.34 -0.46 0.42" mode="targetbody" target="focus" fovy="58"/>
    <camera name="side" pos="0 -0.66 0.26" mode="targetbody" target="focus" fovy="58"/>
    {''.join(legs)}
    <body name="plate" pos="0 0 {g.z_joint0}">
      <freejoint name="plate"/>
      <geom name="plate" type="cylinder" size="{R} {t / 2}" pos="0 0 {j2s - t / 2}" mass="{m_plate:.6g}"
            material="acrylic" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="{cfg.piezo.radius} 0.0006" pos="0 0 {j2s - t - 0.0006}" mass="{mech.plate_extra - 0.015:.6g}"
            material="piezo" contype="0" conaffinity="0"/>
      {brackets}
    </body>
    <body name="ball" pos="{_v(ball_pos)}">
      <freejoint name="ball"/>
      <inertial pos="0 0 0" mass="{b.mass}" diaginertia="{inert:.6g} {inert:.6g} {inert:.6g}"/>
      <geom name="ball" size="{b.radius}" material="ball" contype="2" conaffinity="2"/>
    </body>
  </worldbody>
  <contact>{_contact_xml(cfg, damping)}</contact>
  <equality>{''.join(eqs)}</equality>
  {'<actuator>' + ''.join(acts) + '</actuator>' if viewer else ''}
</mujoco>'''


class MjPlant:
    name = 'MuJoCo'

    def __init__(self, cfg, sc, rng):
        self.cfg = cfg
        p0, v0 = ball_start(cfg, sc)
        self.m = m = mujoco.MjModel.from_xml_string(build_mjcf(cfg, ball_pos=p0))
        self.d = d = mujoco.MjData(m)
        self.qadr = [m.jnt_qposadr[m.joint(f'servo{i}').id] for i in range(3)]
        self.vadr = [m.jnt_dofadr[m.joint(f'servo{i}').id] for i in range(3)]
        self.plate_id = m.body('plate').id
        self.ball_id = m.body('ball').id
        self.bdof = m.jnt_dofadr[m.joint('ball').id]
        self.g_ball, self.g_plate = m.geom('ball').id, m.geom('plate').id
        self.eq_ids = [m.equality(f'joint{i}').id for i in range(3)]
        d.qvel[self.bdof:self.bdof + 3] = v0
        mujoco.mj_forward(m, d)
        gm, mech = cfg.geom, cfg.mech
        m_plate = mech.plate_density * math.pi * (gm.plate_diameter / 2) ** 2 * gm.plate_thickness
        j_load = (mech.arm_mass / 3 + mech.rod_mass + (m_plate + mech.plate_extra) / 3) * gm.arm ** 2
        lim = math.radians(gm.servo_limit_deg)
        er = cfg.errors
        self.servos = [MotorServo(cfg.servo, mech, j_load, 0.0, rng.uniform(0, 1 / cfg.servo.pwm_hz), lim,
                                  math.radians(er.trim_deg[i]), er.gain_err[i]) for i in range(3)]
        self.n_sub = max(1, int(round(cfg.dt / m.opt.timestep)))
        self.m_ball = cfg.ball.mass
        self.beta = cfg.ball.beta
        self.buoy = cfg.ball.mass * (G - cfg.ball.g_eff)
        self._touch = False
        self._last_imp = None
        self._vel = np.zeros(6)
        self._sync()

    # ---------- 狀態 ----------
    def _sync(self):
        d = self.d
        self.c = d.xpos[self.plate_id].copy()
        self.R = d.xmat[self.plate_id].reshape(3, 3).copy()
        self.n = self.R[:, 2].copy()
        self.point = self.c + self.cfg.geom.joint_to_surface * self.n

    @property
    def ball_p(self):
        return self.d.xpos[self.ball_id].copy()

    @property
    def ball_v(self):
        return self.d.qvel[self.bdof:self.bdof + 3].copy()

    @property
    def th(self):
        return np.array([self.d.qpos[a] for a in self.qadr])

    @property
    def om(self):
        return np.array([self.d.qvel[a] for a in self.vadr])

    def _plate_vel_at(self, x):
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.plate_id, self._vel, 0)
        return self._vel[3:] + cross(self._vel[:3], x - self.d.xpos[self.plate_id])

    def _touching(self):
        d = self.d
        if not d.ncon:
            return False
        c = d.contact
        g1, g2 = c.geom1[:d.ncon], c.geom2[:d.ncon]
        return bool(np.any(((g1 == self.g_ball) & (g2 == self.g_plate)) | ((g1 == self.g_plate) & (g2 == self.g_ball))))

    def extras(self):
        """馬達扭力、duty、三個球頭受力（N）"""
        d = self.d
        rod = np.zeros(3)
        if d.nefc:
            eq = d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            for i, e in enumerate(self.eq_ids):
                rows = np.nonzero(eq & (d.efc_id[:d.nefc] == e))[0]
                rod[i] = float(np.linalg.norm(d.efc_force[rows]))
        return dict(torque=np.array([s.torque for s in self.servos]),
                    duty=np.array([s.u for s in self.servos]), rod_force=rod)

    # ---------- 推進 ----------
    def command(self, th):
        for s, x in zip(self.servos, th):
            s.command(x)

    def reset_ball(self, p, v, w=(0.0, 0.0, 0.0)):
        adr = self.m.jnt_qposadr[self.m.joint('ball').id]
        self.d.qpos[adr:adr + 7] = [*p, 1.0, 0.0, 0.0, 0.0]
        self.d.qvel[self.bdof:self.bdof + 6] = [*v, *w]       # 姿態歸零，所以角速度的本體座標＝世界座標
        self._touch, self._last_imp = False, None
        mujoco.mj_forward(self.m, self.d)

    def step(self, t, dt):
        m, d = self.m, self.d
        h = m.opt.timestep
        R = self.cfg.ball.radius
        out = []
        for k in range(self.n_sub):
            tk = t + k * h
            for s, qa, va in zip(self.servos, self.qadr, self.vadr):
                d.qfrc_applied[va] = s.update(tk, h, d.qpos[qa], d.qvel[va])
            v = d.qvel[self.bdof:self.bdof + 3]
            f = -self.m_ball * self.beta * math.sqrt(float(v @ v)) * v
            f[2] += self.buoy
            # 用 qfrc_applied 而不是 xfrc_applied：互動視窗的滑鼠推力寫在 xfrc_applied，不能蓋掉
            d.qfrc_applied[self.bdof:self.bdof + 3] = f
            pre = None
            if not self._touch:             # 撞擊前的相對法向速度（壓電片、統計用）
                n = d.xmat[self.plate_id].reshape(3, 3)[:, 2]
                pb = d.xpos[self.ball_id]
                if (pb - d.xpos[self.plate_id]) @ n < self.cfg.geom.joint_to_surface + R + 0.006:
                    v_pl = self._plate_vel_at(pb - R * n)
                    pre = (pb.copy(), v.copy(), v_pl, n.copy())
            mujoco.mj_step(m, d)
            touch = self._touching()
            if touch and not self._touch and pre is not None:
                pb, vb, v_pl, n = pre
                vn = float((vb - v_pl) @ n)
                if vn < -self.cfg.surface.v_rest:
                    self._last_imp = dict(t=tk + h, p=pb, vn=vn, v_in=vb, v_out=None, v_plate=v_pl, n=n)
                    out.append(self._last_imp)
            if self._touch and not touch and self._last_imp is not None and self._last_imp['v_out'] is None:
                self._last_imp['v_out'] = self.ball_v
            self._touch = touch
        self._sync()
        return out
