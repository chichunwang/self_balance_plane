"""乒乓球物理：飛行（重力＋浮力＋空氣阻力）與撞擊（恢復係數＋庫侖摩擦＋自旋）。

撞擊用衝量法：法向衝量 J_n = (1+e)·m·|v_n|；切向先算「剛好讓接觸點不滑」要多少衝量，
超過 μ·J_n 就改成滑動摩擦。同一套公式也處理靜止接觸和滾動：
相對法向速度小於 v_rest 時用 e=0，球就會貼著板面滾（薄殼球滾動加速度＝0.6·g·sinθ）。
"""
import math
from dataclasses import dataclass

import numpy as np


def cross(a, b):
    """三維外積（np.cross 對小向量很慢，這個快十倍）"""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class Ball:
    def __init__(self, spec, p, v, w=(0.0, 0.0, 0.0)):
        self.p = np.array(p, float)
        self.v = np.array(v, float)
        self.w = np.array(w, float)
        self.m = spec.mass
        self.R = spec.radius
        k = spec.k_inertia
        self.I = k * self.m * self.R ** 2
        self.m_t = self.m * k / (1 + k)     # 切向等效質量（薄殼球＝0.4 m）
        self.g = spec.g_eff
        self.beta = spec.beta

    def fly(self, dt):
        a = -self.beta * math.sqrt(float(self.v @ self.v)) * self.v
        a[2] -= self.g
        self.v = self.v + a * dt
        self.p = self.p + self.v * dt


@dataclass
class Surface:
    point: np.ndarray       # 上表面中心點
    n: np.ndarray           # 法向量
    vel: np.ndarray         # 上表面中心點速度
    omega: np.ndarray       # 角速度
    radius: float

    def gap(self, p, r):
        return float((p - self.point) @ self.n) - r

    def inside(self, p):
        d = p - self.point
        d = d - (d @ self.n) * self.n
        return float(d @ d) <= self.radius ** 2

    def vel_at(self, x):
        return self.vel + cross(self.omega, x - self.point)


def collide(ball, surf, e, mu, v_rest):
    """球已壓進板面時呼叫：施加衝量。回傳 (撞擊前相對法向速度, 接觸點的平台速度)"""
    n = surf.n
    rc = -ball.R * n
    v_pl = surf.vel_at(ball.p + rc)
    v_rel = ball.v + cross(ball.w, rc) - v_pl
    vn = float(v_rel @ n)
    if vn >= 0:
        return vn, v_pl
    Jn = -(1 + (e if -vn > v_rest else 0.0)) * ball.m * vn
    J = Jn * n
    vt = v_rel - vn * n
    vt_mag = math.sqrt(float(vt @ vt))
    if vt_mag > 1e-12:
        Jt = -min(ball.m_t * vt_mag, mu * Jn) / vt_mag * vt
        J = J + Jt
        ball.w = ball.w + cross(rc, Jt) / ball.I
    ball.v = ball.v + J / ball.m
    return vn, v_pl


def step(ball, surf, sspec, dt):
    """推進一步。真的撞擊（不是靜止接觸）時回傳一筆紀錄，否則 None。"""
    p0 = ball.p
    ball.fly(dt)
    if not surf.inside(ball.p):
        return None
    g1 = surf.gap(ball.p, ball.R)
    if g1 >= 0:
        return None
    g0 = surf.gap(p0, ball.R)
    t_rem = dt * (-g1) / (g0 - g1) if g0 > 0 else 0.0   # 本步裡接觸之後剩下的時間
    ball.p = ball.p - ball.v * t_rem
    v_in = ball.v.copy()
    p_hit = ball.p.copy()
    vn, v_pl = collide(ball, surf, sspec.e, sspec.mu, sspec.v_rest)
    ball.p = ball.p + ball.v * t_rem
    g2 = surf.gap(ball.p, ball.R)
    if g2 < 0:
        ball.p = ball.p - g2 * surf.n
    if vn < -sspec.v_rest:
        return dict(t_rem=t_rem, p=p_hit, vn=vn, v_in=v_in, v_out=ball.v.copy(),
                    v_plate=v_pl, n=surf.n.copy())
    return None


def impact_model(v_in, w_in, n, v_plate, e, mu, radius, k_inertia):
    """控制器與估測器用的撞擊模型（和上面的真實物理同一套，只是參數是「以為的」）。

    法向乘 −e；切向：摩擦讓接觸點不滑（薄殼球會吃掉 40% 的滑動速度），但不超過 μ×法向衝量。
    還沒旋轉的球第一次撞擊時水平速度會少很多；已經在滾的球摩擦幾乎不作用。回傳 (v_out, w_out)"""
    rc = -radius * n
    v_rel = v_in - v_plate
    vn = float(v_rel @ n)
    vt = v_rel - vn * n
    c = vt + cross(w_in, rc)                 # 接觸點滑動速度
    c = c - float(c @ n) * n
    dv_t = -(k_inertia / (1 + k_inertia)) * c
    lim = mu * (1 + e) * abs(vn)
    mag = math.sqrt(float(dv_t @ dv_t))
    if mag > lim:
        dv_t = dv_t * (lim / mag)
    v_out = v_plate - e * vn * n + vt + dv_t
    w_out = w_in + cross(rc, dv_t) / (k_inertia * radius ** 2)
    return v_out, w_out
