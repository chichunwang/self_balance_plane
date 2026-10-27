"""3-RRS 平台運動學：指令位姿 ↔ 三顆伺服角。

座標：底板中心為原點、z 朝上。第 i 隻腳在方位角 φ_i，徑向單位向量 er_i、切向 et_i。
伺服軸平行 et_i，臂和連桿都在「過中心軸的徑向平面」裡擺動（肘部是轉軸），
平台端是球頭。所以平台只有 升降＋兩軸傾斜 三個自由度，
另外三個（x、y 平移與扭轉）會被連帶決定，叫寄生運動，這裡用牛頓法解。
"""
import math

import numpy as np

EZ = np.array([0.0, 0.0, 1.0])


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rodrigues(w):
    th = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
    if th < 1e-15:
        return np.eye(3) + skew(w)
    K = skew(np.asarray(w) / th)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def tilt_normal(tx, ty):
    """傾斜向量 τ=(tx, ty) → 單位法向量。τ 指向下坡，球會往 τ 的方向滾。"""
    n = np.array([tx, ty, 1.0])
    return n / np.linalg.norm(n)


def rot_to(n):
    """把 z 軸轉到 n 的最小旋轉（不含扭轉）"""
    axis = np.array([-n[1], n[0], 0.0])   # EZ × n
    s = math.hypot(axis[0], axis[1])
    if s < 1e-12:
        return np.eye(3)
    return rodrigues(axis / s * math.atan2(s, n[2]))


def rot_z(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rowdot(a, b):
    return np.einsum('ij,ij->i', a, b)


class Platform:
    def __init__(self, geom):
        self.g = geom
        phi = np.radians(geom.legs_deg)
        self.er = np.column_stack([np.cos(phi), np.sin(phi), np.zeros(3)])
        self.et = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(3)])
        self.p_loc = geom.r_p * self.er                  # 球頭（平台座標）
        self.shaft = geom.r_b * self.er + geom.z_shaft * EZ
        self.s = geom.arm_sign                           # 臂朝內 +1、朝外 −1
        self._ez_x_p = np.cross(EZ, self.p_loc)
        self.theta_lim = math.radians(geom.servo_limit_deg)

    def neutral(self):
        return np.array([0.0, 0.0, self.g.z_joint0]), np.eye(3)

    def pose(self, dz, tx, ty, guess=None):
        """指令（升降 dz、傾斜 τ）→ 完整位姿 (c, R)，同時解出寄生的 x、y、扭轉。

        回傳 (c, R, guess)，guess 丟回下一次呼叫可以少跑幾圈牛頓法。
        """
        Rt = rot_to(tilt_normal(tx, ty))
        z = self.g.z_joint0 + dz
        x, y, psi = guess if guess is not None else (0.0, 0.0, 0.0)
        for _ in range(10):
            R = Rt @ rot_z(psi)
            c = np.array([x, y, z])
            r = _rowdot(c + self.p_loc @ R.T, self.et)    # 球頭要落在各自的徑向平面
            if np.max(np.abs(r)) < 1e-13:
                break
            dpsi = self._ez_x_p @ R.T                     # d(R·p)/dψ
            J = np.column_stack([self.et[:, 0], self.et[:, 1], _rowdot(dpsi, self.et)])
            dx, dy, dp = np.linalg.solve(J, -r)
            x, y, psi = x + dx, y + dy, psi + dp
        return c, R, (x, y, psi)

    def ik(self, c, R):
        """位姿 → 三顆伺服角（臂水平＝0，往上為正）。回傳 (theta, 是否可達)"""
        P = c + self.p_loc @ R.T
        du = self.s * (_rowdot(P, self.er) - self.g.r_b)  # 換成「沿臂的方向」量，朝內朝外同一條公式
        dv = P[:, 2] - self.g.z_shaft
        a, d = self.g.arm, self.g.rod
        # 連桿長度條件整理成 du·cosθ − dv·sinθ = K
        K = (d * d - du * du - dv * dv - a * a) / (2 * a)
        ratio = K / np.hypot(du, dv)
        ok = bool(np.all(np.abs(ratio) <= 1.0))
        th = np.arccos(np.clip(ratio, -1.0, 1.0)) - np.arctan2(dv, du)
        th = (th + np.pi) % (2 * np.pi) - np.pi
        ok = ok and bool(np.all(np.abs(th) <= self.theta_lim))
        return th, ok

    def arm_tips(self, th):
        a = self.g.arm
        return self.shaft + a * (-self.s * np.cos(th)[:, None] * self.er + np.sin(th)[:, None] * EZ)

    def fk(self, th, c0, R0, iters=3, tol=1e-12):
        """伺服角 → 平台位姿。牛頓法解 6 條件（3 個徑向平面＋3 根連桿長），用上一步當初值。"""
        A = self.arm_tips(th)
        d2 = self.g.rod ** 2
        c, R = c0.copy(), R0.copy()
        J = np.zeros((6, 6))
        J[:3, :3] = self.et
        for _ in range(iters):
            rloc = self.p_loc @ R.T
            P = c + rloc
            L = P - A
            res = np.concatenate([_rowdot(P, self.et), _rowdot(L, L) - d2])
            if np.max(np.abs(res)) < tol:
                break
            J[:3, 3:] = np.cross(rloc, self.et)
            J[3:, :3] = 2 * L
            J[3:, 3:] = 2 * np.cross(rloc, L)
            dq = np.linalg.solve(J, -res)
            c = c + dq[:3]
            R = rodrigues(dq[3:]) @ R
        return c, R

    def surface(self, c, R):
        """平台上表面中心點與法向量"""
        n = R[:, 2]
        return c + self.g.joint_to_surface * n, n
