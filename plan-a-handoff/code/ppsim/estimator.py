"""球的狀態估測（卡爾曼濾波，狀態＝位置＋速度）。

- 預測步用控制器「以為」的物理：重力＋阻力＋撞擊模型（e_hat、mu_model，含自旋）。
  自旋不直接量，靠撞擊模型推：剛看到的球當成沒旋轉，每次撞擊後更新。
- 撞擊時間優先用壓電片時間戳；沒有壓電片（或撞太輕）才用模型算的穿越時間。
- 球貼在板上時切成 ROLLING：高度鎖在板面，只估 x、y（高度量測雜訊太大，沒用）。
- 相機有延遲，所以濾波器停在「最後一張影像的拍攝時間」，控制器要用時再往前推到現在。
"""
import bisect
import math

import numpy as np

from .ball import impact_model
from .kinematics import tilt_normal

FLIGHT, ROLLING = 'flight', 'rolling'
H_STEP = 0.002


class Estimator:
    def __init__(self, cfg, camera, plate_fn):
        self.cfg = cfg
        self.cam = camera
        self.plate_fn = plate_fn            # t -> (z_off, zd, (tx, ty))：控制器以為的平台
        self.R = cfg.ball.radius
        self.beta = cfg.ball.beta
        self.g = cfg.ball.g_eff
        self.z_s0 = cfg.geom.z_surface0
        self.piezo_on = cfg.piezo.enabled
        self.piezo_min = cfg.piezo.min_speed
        self.v_roll = 0.08                  # 反彈法向速度低於此值就當成貼板
        self.mode = None
        self.t = None
        self.x = [0.0] * 6
        self.P = np.eye(6)
        self.piezo = []                     # 收到但還沒用掉的撞擊時間
        self.version = 0                    # 每次有新資訊就 +1，控制器據此決定要不要重新規劃
        self.last_piezo = -math.inf
        self._first = None                  # 鎖定前的第一張影像
        self._bad = 0                       # 連續幾張不合物理
        self.piezo_seen = []                # 最近收過的所有撞擊時間（判斷「該響沒響」用）
        self.e_hat = cfg.control.e_model    # 恢復係數估計值（壓電片校正）
        self.e_log = []                     # (時間, 這次觀測到的 e, 更新後的 e_hat)
        self.flights = []                   # (起跳時間, 實際起跳速度, 實際上升高度)：由飛行時間反推
        self._last_hit = None               # 上一次壓電片撞擊：(時間, 板面高, 平台速度, 起跳速度)
        self.now = 0.0                      # 目前時間：壓電片只可能回報到 now − latency 為止
        self.spin = np.zeros(3)             # 以為的球自旋（濾波器那條線）
        self._pred_spin = np.zeros(3)       # 往前預測時用的暫存
        self._popup = 0
        self._rest = 0

    # ---------- 平台（控制器以為的） ----------
    def plane(self, t):
        z_off, zd, tau = self.plate_fn(t)
        n = tilt_normal(tau[0], tau[1])
        return np.array([0.0, 0.0, self.z_s0 + z_off]), n, np.array([0.0, 0.0, zd])

    # ---------- 均值推進 ----------
    def _fly(self, x, dt):
        """飛行一步（中點法；純量運算，比 numpy 小陣列快）"""
        px, py, pz, vx, vy, vz = x
        b, g = self.beta, self.g
        s = math.sqrt(vx * vx + vy * vy + vz * vz)
        mx, my, mz = vx - b * s * vx * dt / 2, vy - b * s * vy * dt / 2, vz - (g + b * s * vz) * dt / 2
        s = math.sqrt(mx * mx + my * my + mz * mz)
        return [px + mx * dt, py + my * dt, pz + mz * dt,
                vx - b * s * mx * dt, vy - b * s * my * dt, vz - (g + b * s * mz) * dt]

    def _roll(self, x, t, dt):
        point, n, vpl = self.plane(t)
        k = self.cfg.control.roll_factor * self.g * n[2]
        ax, ay = k * n[0], k * n[1]
        px = x[0] + x[3] * dt + 0.5 * ax * dt * dt
        py = x[1] + x[4] * dt + 0.5 * ay * dt * dt
        pz = point[2] + (self.R - n[0] * px - n[1] * py) / n[2]
        return [px, py, pz, x[3] + ax * dt, x[4] + ay * dt, vpl[2]]

    def _cov(self, P, dt, mode):
        c = self.cfg.control
        q = c.q_flight if mode == FLIGHT else c.q_roll
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        Q = np.zeros((6, 6))
        Q[:3, :3] = dt ** 3 / 3 * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = dt ** 2 / 2 * np.eye(3)
        Q[3:, 3:] = dt * np.eye(3)
        P = F @ P @ F.T + q * q * Q
        if mode == ROLLING:                 # 高度被板面鎖住
            P[2, :] = P[:, 2] = P[5, :] = P[:, 5] = 0.0
            P[2, 2], P[5, 5] = 1e-6, 1e-4
        return P

    def _model_hit(self, x, t):
        point, n, vpl = self.plane(t)
        p, v = np.array(x[:3]), np.array(x[3:])
        gap = float((p - point) @ n) - self.R
        if gap >= 0:
            return False
        vn = float((v - vpl) @ n)
        if vn >= 0:
            return False
        if not self.piezo_on or -vn < self.piezo_min * 1.2:
            return True                     # 壓電片抓不到的輕撞，只能靠模型
        # 模型說會重重撞上，那就看壓電片：附近有回報 → 等它強制撞擊；
        # 已經過了回報期限卻沒響 → 球其實沒彈起來，是貼著板面（當成非彈性接觸）
        if any(abs(tp - t) < 0.03 for tp in self.piezo_seen):
            return gap < -0.015             # 保險：模型跟實際差太多
        return t < self.now - self.cfg.piezo.latency - 0.005

    def _impact(self, x, P, mode, t, forced, with_cov):
        point, n, vpl = self.plane(t)
        p, v = np.array(x[:3]), np.array(x[3:])
        if mode == ROLLING:
            if forced:                      # 貼板時卻收到撞擊 → 球其實跳起來過
                v[2] = vpl[2] + 0.3
                if with_cov:
                    P = P.copy()
                    P[2, 2] += 0.005 ** 2
                    P[5, 5] += 0.3 ** 2
                return list(np.concatenate([p, v])), P, FLIGHT
            return x, P, mode
        if forced and with_cov:
            # 壓電片說「此刻球正好在板面上」：當成一次很準的高度量測，
            # 透過位置-速度的相關性，連落下速度一起修正
            # 差太多（>6 mm）代表估測已經飄掉，這時只修位置，免得把大誤差換算成離譜的速度
            innov = float(point @ n) + self.R - float(p @ n)
            if abs(innov) < 0.006:
                h = np.concatenate([n, np.zeros(3)])
                k = P @ h / (h @ P @ h + 0.0005 ** 2)
                xa = np.concatenate([p, v]) + k * innov
                P = P - np.outer(k, h @ P)
                p, v = xa[:3], xa[3:]
        p = p - (float((p - point) @ n) - self.R) * n     # 撞擊時球一定貼在板面上
        vr = v - vpl
        vn = float(vr @ n)
        if vn >= 0:
            return list(np.concatenate([p, v])), P, mode
        c = self.cfg.control
        if forced and with_cov:
            self._learn_e(t, point[2], vpl[2], v, vn)
        elif with_cov:
            self._last_hit = None           # 中間夾了模型推的撞擊，下一段飛行時間不能拿來校正
        e = self.e_hat
        if not forced and self.piezo_on and -vn >= self.piezo_min * 1.2:
            e = 0.0                         # 夠重卻沒觸發壓電片 → 沒有真的反彈
        new_mode = FLIGHT
        if -e * vn < self.v_roll:
            e, new_mode = 0.0, ROLLING
        spin = self.spin if with_cov else self._pred_spin
        v_new, spin = impact_model(v, spin, n, vpl, e, c.mu_model, self.R, self.cfg.ball.k_inertia)
        if with_cov:
            self.spin = spin
            Nn = np.outer(n, n)
            vt_in = vr - vn * n
            vt_out = (v_new - vpl) - float((v_new - vpl) @ n) * n
            kap = min(1.0, max(0.5, float(np.linalg.norm(vt_out)) / max(float(np.linalg.norm(vt_in)), 1e-6)))
            A = np.zeros((6, 6))
            A[:3, :3] = np.eye(3) - Nn
            A[3:, 3:] = kap * (np.eye(3) - Nn) - e * Nn
            P = A @ P @ A.T
            P[:3, :3] += 0.001 ** 2 * Nn
            P[3:, 3:] += (0.02 + 0.03 * abs(vn)) ** 2 * np.eye(3)
        else:
            self._pred_spin = spin
        v = v_new
        if forced and with_cov and self._last_hit is not None:
            self._last_hit = self._last_hit[:3] + (float(v[2]), self._last_hit[4])
        return list(np.concatenate([p, v])), P, new_mode

    def _learn_e(self, t, z_plate, w_plate, v_in, vn_in):
        """用「上一次撞擊→這一次撞擊」的飛行時間反推上一跳真正的起跳速度，校正 e。

        飛行時間由壓電片量得非常準，比相機量高度可靠得多。"""
        prev = self._last_hit
        self._last_hit = (t, z_plate, w_plate, None, -vn_in)
        if prev is None or prev[3] is None:
            return
        t0, z0, w0, v_up_model, s_in0 = prev
        T = t - t0
        if not 0.05 < T < 1.5:
            return
        # 找起跳速度 v0：從板面高 z0 出發，T 秒後剛好落到 z_plate（含空氣阻力，割線法）
        def land(v0):
            x = [0.0, 0.0, z0, 0.0, 0.0, v0]
            n = max(1, int(T / H_STEP))
            for _ in range(n):
                x = self._fly(x, T / n)
            return x[2] - z_plate
        a, b = v_up_model, v_up_model * 1.05 + 0.01
        fa, fb = land(a), land(b)
        for _ in range(6):
            if abs(fb - fa) < 1e-12:
                break
            a, b, fa = b, b - fb * (b - a) / (fb - fa), fb
            fb = land(b)
        if b > 0:
            beta, g = self.beta, self.g
            self.flights.append((t0, b, math.log(1 + beta * b * b / g) / (2 * beta)))
        e_obs = (b - w0) / s_in0 if s_in0 > 0.2 else None
        if e_obs is not None and 0.5 < e_obs < 1.0 and self.cfg.control.e_adapt:
            self.e_hat += 0.2 * (e_obs - self.e_hat)
            self.e_log.append((t, e_obs, self.e_hat))

    def _propagate(self, x, P, mode, t0, t1, with_cov=True):
        forced = [tp for tp in self.piezo if t0 < tp <= t1]
        t = t0
        while t1 - t > 1e-9:
            tn = min(t + H_STEP, t1)
            if forced and forced[0] < tn:
                tn = forced[0]
            dt = tn - t
            x = self._fly(x, dt) if mode == FLIGHT else self._roll(x, tn, dt)
            if with_cov:
                P = self._cov(P, dt, mode)
            t = tn
            if forced and t >= forced[0] - 1e-12:
                forced.pop(0)
                x, P, mode = self._impact(x, P, mode, t, True, with_cov)
            elif mode == FLIGHT and self._model_hit(x, t):
                x, P, mode = self._impact(x, P, mode, t, False, with_cov)
        return x, P, mode

    # ---------- 對外介面 ----------
    def on_piezo(self, t_imp):
        if self.t is not None and t_imp <= self.t:
            return
        bisect.insort(self.piezo, t_imp)
        self.piezo_seen = [tp for tp in self.piezo_seen if tp > t_imp - 1.0] + [t_imp]
        self.last_piezo = max(self.last_piezo, t_imp)
        self.version += 1

    def _acquire(self, t_m, m, Rm):
        """剛看到球（例如從旁邊拋進畫面）：拿前兩張影像直接差分出初速，不假設它靜止。"""
        first = self._first
        if first is None or not 0 < t_m - first[0] < 0.05:
            self._first = (t_m, m, Rm)
            return
        dt = t_m - first[0]
        v = (m - first[1]) / dt
        v[2] -= 0.5 * self.g * dt               # 差分是區間中點的速度，換算到最後一張
        self.x = [*m, *v]
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = Rm
        self.P[:3, 3:] = self.P[3:, :3] = Rm / dt
        self.P[3:, 3:] = (Rm + first[2]) / dt ** 2 + (0.3 * dt) ** 2 * np.eye(3)
        self.mode, self.t = FLIGHT, t_m
        self._first, self._bad = None, 0
        self.piezo = []
        self.spin = np.zeros(3)                 # 剛出手的球當成沒旋轉
        self.version += 1

    def on_camera(self, t_m, m, now):
        self.now = now
        m = np.asarray(m, float)
        Rm = self.cam.meas_cov(m)
        if self.t is None:
            self._acquire(t_m, m, Rm)
            return
        if t_m <= self.t:
            return
        x, P, mode = self._propagate(self.x, self.P, self.mode, self.t, t_m)
        x = np.array(x)
        if mode == ROLLING:
            point, n, _ = self.plane(t_m)
            z_exp = point[2] + (self.R - n[0] * x[0] - n[1] * x[1]) / n[2]
            if m[2] - z_exp > max(4 * math.sqrt(Rm[2, 2]), 0.008):
                self._popup += 1
            else:
                self._popup = 0
            if self._popup >= 2:            # 連兩張都明顯高於板面 → 球跳起來了
                mode, self._popup = FLIGHT, 0
                P[2, 2] += (m[2] - z_exp) ** 2
                P[5, 5] += 0.5 ** 2
        if mode == FLIGHT:
            H, z, Rz = np.hstack([np.eye(3), np.zeros((3, 3))]), m, Rm
        else:
            H, z, Rz = np.eye(6)[:2], m[:2], Rm[:2, :2]
        S = H @ P @ H.T + Rz
        y = z - H @ x
        if float(y @ np.linalg.solve(S, y)) > (30.0 if mode == FLIGHT else 25.0):
            # 量到的位置和物理預測差太多：先當雜訊丟掉；連三張都這樣（球被手拿著走、追丟）就重新鎖定
            self._bad += 1
            if self._bad >= 3:
                self.t, self.mode, self._first = None, None, None
                self._acquire(t_m, m, Rm)
            return
        self._bad = 0
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        IKH = np.eye(6) - K @ H
        P = IKH @ P @ IKH.T + K @ Rz @ K.T
        if mode == FLIGHT and self._at_rest(x, P, t_m):
            point, n, vpl = self.plane(t_m)
            x[:3] -= (float((x[:3] - point) @ n) - self.R) * n
            x[5] = vpl[2]
            P[2, :] = P[:, 2] = P[5, :] = P[:, 5] = 0.0
            P[2, 2], P[5, 5] = 1e-6, 1e-4
            mode = ROLLING
        self.x, self.P, self.mode, self.t = list(x), P, mode, t_m
        self.piezo = [tp for tp in self.piezo if tp > t_m]
        self.version += 1

    def _at_rest(self, x, P, t_m):
        """靜止偵測：壓電片安靜夠久（有裝的話）、估計高度貼著板面、法向速度很小，連續幾張就算貼板。
        相機高度雜訊會讓濾波器以為球還在小跳，沒有這一步會一直卡在飛行模式。"""
        point, n, vpl = self.plane(t_m)
        gap = float((x[:3] - point) @ n) - self.R
        vn = float((x[3:] - vpl) @ n)
        s_n = math.sqrt(max(float(n @ P[:3, :3] @ n), 0.0))
        quiet = not self.piezo_on or t_m - self.last_piezo > 0.12
        self._rest = self._rest + 1 if quiet and gap < max(0.004, 2 * s_n) and abs(vn) < 0.25 else 0
        if self._rest >= 6:
            self._rest = 0
            return True
        return False

    def predict(self, t):
        """往前推到時間 t（只推均值）。回傳 (狀態, 模式)"""
        if self.t is None:
            return None, None
        if t - self.t > 0.3:                    # 太久沒看到球（掉出盤外、出了畫面）：追蹤丟失，等重新鎖定
            self.t, self.mode, self._first = None, None, None
            return None, None
        self.now = max(self.now, t)
        self._pred_spin = self.spin.copy()
        x, _, mode = self._propagate(self.x, None, self.mode, self.t, max(t, self.t), with_cov=False)
        return x, mode

    def predict_crossing(self, t_now, z_off):
        """預測球心下一次往下穿過「板面高度 z_off 的水平面」的時間與狀態。

        回傳 (t_hit, 穿越時的狀態 [p, v], 當時以為的自旋) 或 None。
        """
        x, mode = self.predict(t_now)
        if mode != FLIGHT:
            return None
        spin = self._pred_spin.copy()
        h = self.z_s0 + z_off + self.R
        t = max(t_now, self.t)
        if x[2] <= h and x[5] < 0:
            return t, x, spin
        for _ in range(int(1.5 / H_STEP)):
            xn = self._fly(x, H_STEP)
            if xn[5] < 0 and xn[2] <= h:
                if x[2] <= h:
                    return None             # 球的最高點低於這個平面，到不了
                f = (x[2] - h) / (x[2] - xn[2])
                return t + f * H_STEP, [a + f * (b - a) for a, b in zip(x, xn)], spin
            x, t = xn, t + H_STEP
        return None
