"""控制器：狀態機＋每一跳的「鏡射律」規劃＋平衡 PD。

JUGGLE  拍球：預測落點與落下速度，算出撞擊瞬間平台要多快、斜幾度，
        讓球彈回目標高度，同時往中心修正。
CUSHION 接球緩衝：同一套規劃，但目標高度每跳減半。
SETTLE  最後幾跳太低、來不及揮拍：平台慢慢往下沉吸收能量，傾斜繼續導向中心。
BALANCE 球貼在板上滾：PD 控制傾斜，把球推回中心。

所有命令都下給「現在＋lead」的時間點，用來抵銷伺服的反應延遲。
"""
import bisect
import math
from dataclasses import dataclass

import numpy as np

from .ball import impact_model
from .estimator import FLIGHT, ROLLING
from .kinematics import tilt_normal

JUGGLE, CUSHION, SETTLE, BALANCE = 'JUGGLE', 'CUSHION', 'SETTLE', 'BALANCE'
MODES = (JUGGLE, CUSHION, SETTLE, BALANCE)


def hermite(p0, m0, p1, m1, T, s):
    s2, s3 = s * s, s * s * s
    p = (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * T * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * T * m1
    v = ((6 * s2 - 6 * s) * p0 + (-6 * s2 + 6 * s) * p1) / T + (3 * s2 - 4 * s + 1) * m0 + (3 * s2 - 2 * s) * m1
    return p, v


@dataclass
class Key:
    t: float
    z: float
    zd: float
    tau: np.ndarray
    taud: np.ndarray


class Plan:
    """升降 z 與傾斜 τ 的關鍵影格，中間用三次 Hermite 內插（位置、速度都連續）。"""

    def __init__(self, keys, window=None):
        self.keys = keys
        self.window = window            # (視窗開始, 預定撞擊, 視窗結束)
        self._ts = [k.t for k in keys]

    def eval(self, t):
        ks = self.keys
        if t <= ks[0].t:
            k = ks[0]
            return k.z, k.zd, k.tau.copy(), k.taud.copy()
        if t >= ks[-1].t:
            k = ks[-1]
            return k.z, 0.0, k.tau.copy(), np.zeros(2)
        i = bisect.bisect_right(self._ts, t) - 1
        a, b = ks[i], ks[i + 1]
        T = b.t - a.t
        s = (t - a.t) / T
        z, zd = hermite(a.z, a.zd, b.z, b.zd, T, s)
        tau, taud = hermite(a.tau, a.taud, b.tau, b.taud, T, s)
        return z, zd, tau, taud


def launch_speed(h, beta, g):
    """要升高 h 需要的起跳速度（含二次方空氣阻力）"""
    if beta < 1e-12:
        return math.sqrt(2 * g * h)
    return math.sqrt(g / beta * (math.exp(2 * beta * h) - 1))


def mirror_solve(v_in, v_out, e, bounce=None):
    """鏡射律：要讓球以 v_out 彈走，平台傾斜 τ 與撞擊瞬間升降速度 w 要多少。

    當成無摩擦時，速度變化 Δv 一定和平台法向量平行，可以直接解；
    有給 bounce(n, v_plate)→v_out（含摩擦與自旋的撞擊模型）就再用牛頓法修正。
    """
    dv = v_out - v_in
    n = dv / np.linalg.norm(dv)
    w = (np.linalg.norm(dv) / (1 + e) + v_in @ n) / n[2]
    x = np.array([n[0] / n[2], n[1] / n[2], w])
    if bounce is None:
        return x[:2], x[2]

    def f(q):
        return bounce(tilt_normal(q[0], q[1]), np.array([0.0, 0.0, q[2]])) - v_out

    for _ in range(20):
        r = f(x)
        if np.max(np.abs(r)) < 1e-10:
            break
        J = np.empty((3, 3))
        for j in range(3):
            d = np.zeros(3)
            d[j] = 1e-7
            J[:, j] = (f(x + d) - r) / 1e-7
        x = x - np.linalg.solve(J, r)
    return x[:2], x[2]


class Controller:
    def __init__(self, cfg, platform, juggle_until):
        self.cfg = cfg
        self.plat = platform
        c, s = cfg.control, cfg.servo
        self.lead = c.lead if c.lead is not None else 1 / s.kp + s.deadtime + 0.5 / s.pwm_hz
        self.zd_max = c.heave_speed_frac * s.w_max * cfg.geom.arm
        self.a_leg = c.leg_accel_frac * s.a_max * cfg.geom.arm     # 規劃時每隻腳的加速度上限
        self.tilt_max = math.tan(math.radians(c.tilt_max_deg))
        self.hit_tilt_max = math.tan(math.radians(c.hit_tilt_max_deg))
        self.bal_tilt_max = math.tan(math.radians(c.bal_tilt_max_deg))
        self.g = cfg.ball.g_eff
        g_roll = c.roll_factor * self.g
        self.kp = c.bal_wn ** 2 / g_roll
        self.kd = 2 * c.bal_zeta * c.bal_wn / g_roll
        self.dt = 1 / c.rate_hz
        self.juggle_until = juggle_until
        self.mode = JUGGLE if juggle_until > 0 else CUSHION
        self.est = None
        self.plan = Plan([Key(0.0, 0.0, 0.0, np.zeros(2), np.zeros(2))])
        self.hits = []                  # 每一跳的規劃紀錄
        self.events = [(0.0, self.mode)]
        self._ht, self._hs = [], []     # 已下達命令：生效時間、(z, zd, tx, ty)
        self._ver = -1
        self._guess = None
        self._z, self._zd, self._tau = 0.0, 0.0, np.zeros(2)
        self._z_hit = 0.0
        # 自我校正：平台「命令水平」時其實歪多少（伺服 trim 沒調準造成）、拍球實際飛多高
        self.tilt_trim = np.zeros(2)
        self.apex_bias = 0.0
        self.trim_log = []              # (時間, 傾斜 trim, 高度修正)
        self._trim_lim = math.tan(math.radians(c.trim_max_deg))
        self._learned = set()
        self._flights_used = 0

    # ---------- 給估測器用：控制器以為平台在 t 時刻的狀態 ----------
    def plate_at(self, t):
        ht = self._ht
        if ht and t < ht[-1]:
            if t <= ht[0]:
                s = self._hs[0]
            else:
                i = bisect.bisect_right(ht, t)
                a, b = self._hs[i - 1], self._hs[i]
                f = (t - ht[i - 1]) / (ht[i] - ht[i - 1])
                s = tuple(a[j] + f * (b[j] - a[j]) for j in range(4))
            return s[0], s[1], (s[2], s[3])
        if self.mode in (JUGGLE, CUSHION):
            z, zd, tau, _ = self.plan.eval(t)
            return z, zd, (tau[0], tau[1])
        if self._hs:
            s = self._hs[-1]
            return s[0], s[1], (s[2], s[3])
        return 0.0, 0.0, (0.0, 0.0)

    def _record(self, te, z, zd, tau):
        self._ht.append(te)
        self._hs.append((z, zd, tau[0], tau[1]))
        if len(self._ht) > 4000:
            del self._ht[:2000], self._hs[:2000]

    # ---------- 主迴圈 ----------
    def update(self, t):
        c, gm = self.cfg.control, self.cfg.geom
        te = t + self.lead
        if self.mode == JUGGLE and t >= self.juggle_until and not self._committed(te):
            self._set_mode(CUSHION, t)
        if self.mode in (CUSHION, SETTLE) and self.est.mode == ROLLING:
            self._enter_pd(BALANCE, t, te)
        if self.mode in (SETTLE, BALANCE) and self.est.mode == FLIGHT:
            x, _ = self.est.predict(t)
            if x is not None and x[2] > gm.z_surface0 + self.cfg.ball.radius + 0.04:
                self.new_ball(t, self.juggle_until == math.inf)     # 有人把球丟過來，或球被打飛：改回接球
        if self.mode in (JUGGLE, CUSHION) and self.est.version != self._ver:
            self._ver = self.est.version
            self._learn(t)
            self._replan(t, te)

        if self.mode in (JUGGLE, CUSHION):
            z, zd, tau, _ = self.plan.eval(te)
            lim = self.hit_tilt_max
        else:
            if self.mode == SETTLE:
                zd = -c.sink_speed if self._z > -c.sink_depth else 0.0
                lim = self.tilt_max
            else:
                zd = -math.copysign(min(c.home_speed, abs(self._z) / self.dt), self._z)
                lim = self.bal_tilt_max
            # 切換模式時速度與傾斜不跳變：升降限加速度、傾斜限角速度
            dz_lim = c.plan_accel * self.dt
            zd = self._zd + min(dz_lim, max(-dz_lim, zd - self._zd))
            z = self._z + zd * self.dt
            tau_pd, pos = self._pd_tilt(t, te)
            dtau = tau_pd - self._tau
            dn, dt_lim = float(np.linalg.norm(dtau)), c.tilt_rate * self.dt
            tau = self._tau + (dtau if dn <= dt_lim else dtau * (dt_lim / dn))
            if self.mode == BALANCE and pos is not None:
                # 球長期停在某一側＝平台其實歪向那一側，把偏差慢慢累積成修正量
                self.tilt_trim = self._clip_trim(self.tilt_trim - c.bal_ki * pos * self.dt)
        tn = float(np.linalg.norm(tau))
        if tn > lim:
            tau = tau * (lim / tn)
        z = min(gm.heave_limit, max(-gm.heave_limit, z))
        self._z, self._zd, self._tau = z, zd, tau
        self._record(te, z, zd, tau)                # 估測器用「有效傾斜」（假設 trim 已補掉硬體誤差）
        cmd = tau + self.tilt_trim
        cp, R, self._guess = self.plat.pose(z, cmd[0], cmd[1], self._guess)
        th, _ = self.plat.ik(cp, R)
        return th

    def _clip_trim(self, v):
        n = float(np.linalg.norm(v))
        return v if n <= self._trim_lim else v * (self._trim_lim / n)

    def _learn(self, t):
        """每跳之後的自我校正（拍球時）。

        高度：壓電片的飛行時間反推這一跳實際飛多高，和目標差多少就把下一跳的目標補一點。
        水平：球飛 0.15 秒後相機已把水平速度估準，和鏡射律預期的比；平台多歪 δ，
              水平速度就多 (1+e)·|v_in|·δ，反推出 δ 直接修正（不是積分落點，所以不會越修越晃）。"""
        c, est = self.cfg.control, self.est
        while self._flights_used < len(est.flights):
            t0, _, apex_obs = est.flights[self._flights_used]
            self._flights_used += 1
            rec = next((h for h in self.hits if abs(h['t_hit'] - t0) < 0.03), None)
            if rec is not None and rec['mode'] == JUGGLE:
                self.apex_bias = min(0.03, max(-0.03, self.apex_bias + c.apex_learn * (c.apex - apex_obs)))
        if est.mode != FLIGHT:
            return
        done = [h for h in self.hits if h['t_hit'] < t - 0.15]
        if not done:
            return
        h = done[-1]
        key = round(h['t_hit'], 4)
        later = [x for x in self.hits if x['t_hit'] > h['t_hit'] + 0.06]
        if key in self._learned or h['apex'] < 0.05 or (later and later[0]['t_hit'] < t):
            return
        self._learned.add(key)
        x, _ = est.predict(t)
        delta = (np.array(x[3:5]) - h['v_out'][:2]) / ((1 + est.e_hat) * abs(h['v_in'][2]))
        self.tilt_trim = self._clip_trim(self.tilt_trim - c.tilt_learn * delta)

    def _set_mode(self, mode, t):
        self.mode = mode
        self.events.append((t, mode))

    # ---------- 即時模式用 ----------
    def new_ball(self, t, juggle):
        """重新丟球：模式重設；學到的自動水平與高度修正留著。"""
        self.juggle_until = math.inf if juggle else t
        self.plan = Plan([Key(t + self.lead, self._z, self._zd, self._tau.copy(), np.zeros(2))])
        self._ver = -1
        self._set_mode(JUGGLE if juggle else CUSHION, t)

    def set_juggle(self, on, t):
        """拍球開關：開＝空中的球改成一直拍（球已貼板就等下一顆）；關＝這一跳之後緩衝接住。"""
        self.juggle_until = math.inf if on else t
        if on and self.mode == CUSHION:
            self._set_mode(JUGGLE, t)

    def _enter_pd(self, mode, t, te):
        if self.mode in (JUGGLE, CUSHION):
            self._z, self._zd, self._tau, _ = self.plan.eval(te)
        self._set_mode(mode, t)

    def _committed(self, te):
        w = self.plan.window
        return w is not None and w[0] - self.cfg.control.commit <= te <= w[2]

    def _prev_hit(self, t_hit):
        for h in reversed(self.hits):
            if h['t_hit'] < t_hit - 0.06:
                return h['t_hit']
        return max(self.est.last_piezo, t_hit - 0.4)

    def _planned(self, t_hit):
        w = self.plan.window
        return w is not None and abs(w[1] - t_hit) < 0.06

    def _feasible(self, keys):
        """軌跡有沒有在行程內、伺服跟不跟得上。

        每隻腳的上下速度≈升降速度＋球頭半徑×傾斜變化率，兩者加起來才是伺服要出的力；
        升降速度另外限制在 zd_max，加速度留四成餘量。"""
        tip_v = 0.95 * self.cfg.servo.w_max * self.cfg.geom.arm
        z_lim = self.cfg.geom.heave_limit
        rp, leg_max = self.cfg.geom.r_p, self.cfg.control.leg_travel
        s = np.linspace(0.0, 1.0, 11)
        s2 = s[:, None]
        for a, b in zip(keys[:-1], keys[1:]):
            T = b.t - a.t
            if T <= 0:
                return False
            z, v = hermite(a.z, a.zd, b.z, b.zd, T, s)
            acc = ((12 * s - 6) * (a.z - b.z) + (6 * s - 4) * T * a.zd + (6 * s - 2) * T * b.zd) / T ** 2
            tau, taud = hermite(a.tau, a.taud, b.tau, b.taud, T, s2)
            tacc = ((12 * s2 - 6) * (a.tau - b.tau) + (6 * s2 - 4) * T * a.taud + (6 * s2 - 2) * T * b.taud) / T ** 2
            az, av = np.abs(z), np.abs(v)
            if (np.any(az > z_lim) or np.any(av > self.zd_max)
                    or np.any(az + rp * np.hypot(tau[:, 0], tau[:, 1]) > leg_max)
                    or np.any(av + rp * np.hypot(taud[:, 0], taud[:, 1]) > tip_v)
                    or np.any(np.abs(acc) + rp * np.hypot(tacc[:, 0], tacc[:, 1]) > self.a_leg)):
                return False
        return True

    def _pd_tilt(self, t, te):
        x, _ = self.est.predict(t)
        if x is None:
            return np.zeros(2), None
        lead = te - t
        pos = np.array([x[0] + x[3] * lead, x[1] + x[4] * lead])
        return -(self.kp * pos + self.kd * np.array([x[3], x[4]])), pos

    # ---------- 每一跳的規劃 ----------
    def _solve(self, p_hit, v_in, spin, apex, w_lo, w_hi, scale=(1.0, 1.0), tau_now=None, t_av=None):
        """scale＝(撞擊速度打幾折, 水平修正打幾折)。spin＝估測器以為的球自旋（決定摩擦會吃掉多少水平速度）。
        給了 tau_now、t_av 時，傾斜最多只改到「剩下的時間來得及」的量（側拋進來常常時間很趕）。"""
        c, b = self.cfg.control, self.cfg.ball
        e = self.est.e_hat

        def bounce(n, v_pl):
            return impact_model(v_in, spin, n, v_pl, e, c.mu_model, b.radius, b.k_inertia)[0]

        if apex <= 1e-4:
            v_out, v_up = np.zeros(3), 0.0      # 吸收：希望球撞完就停住
            v_out[:2] = v_in[:2] * (1 - scale[1])
        else:
            v_up = launch_speed(apex, b.beta, self.g)
            v_h = -c.center_gain * scale[1] * p_hit[:2] / (2 * v_up / self.g)
            v_h += v_in[:2] * (1 - scale[1])    # 不修正的部分就讓水平速度照舊
            v_out = np.array([v_h[0], v_h[1], v_up])
        tau, w = mirror_solve(v_in, v_out, e, bounce)
        capped = False
        tn = float(np.linalg.norm(tau))
        if tn > self.hit_tilt_max:              # 斜不到那麼多：傾角用到頂
            tau, capped = tau * (self.hit_tilt_max / tn), True
        if tau_now is not None:
            d = tau - tau_now
            d_max = 0.8 * self.a_leg * max(t_av, 0.0) ** 2 / (6 * self.cfg.geom.r_p)
            dn = float(np.linalg.norm(d))
            if dn > d_max:                      # 時間只夠把傾斜改這麼多
                tau, capped = tau_now + d * (d_max / dn), True
        if capped:                              # 傾斜不夠：至少保證往上的速度
            n = tilt_normal(tau[0], tau[1])
            vz0 = bounce(n, np.zeros(3))[2]
            vz1 = bounce(n, np.array([0.0, 0.0, 1.0]))[2]
            w = (v_up - vz0) / (vz1 - vz0)
        w = min(w_hi, max(w_lo, w * scale[0]))
        v_pred = bounce(tilt_normal(tau[0], tau[1]), np.array([0.0, 0.0, w]))
        return tau, w, v_pred

    def _target(self, t, te, z_hit, state, scale, cache):
        """預測球穿過「板面高 z_hit」的時間，算出這一跳要的傾斜與撞擊速度（速度再乘 scale）。
        回傳 dict，或 'same'（還是剛剛那一跳）/'settle'（小跳不必揮拍）/None（預測不到）。"""
        c, a = self.cfg.control, self.cfg.control.plan_accel
        z0, zd0 = state
        key = round(z_hit, 4)
        if key not in cache:
            cache[key] = self.est.predict_crossing(t, z_hit)
        pred = cache[key]
        if pred is None:
            return None
        t_hit, xs, spin = pred
        win = self.plan.window
        if win is not None and te > win[2] and t_hit < win[2] + 0.05:
            return 'same'               # 估測器還沒看到反彈
        v_in = np.array(xs[3:])
        apex_in = v_in[2] ** 2 / (2 * self.g)
        if self.mode == JUGGLE:
            apex = c.apex + self.apex_bias
        elif apex_in < c.rest_apex:
            return 'settle'
        elif apex_in < c.absorb_apex:
            apex = 0.0
        else:
            apex = c.cushion_ratio * apex_in
        # 等速視窗依這一跳的飛行時間決定（重新規劃時不會跟著縮）
        tw = min(c.hit_window, max(0.008, 0.15 * (t_hit - self._prev_hit(t_hit))))
        t_av = max(0.0, t_hit - tw - te)            # 離視窗開始還有多久
        # 撞擊速度受限於「平台目前規劃中的速度」還來得及改變多少
        w_lo = max(-self.zd_max, zd0 - a * t_av)
        w_hi = min(self.zd_max, zd0 + a * t_av)
        tau, w, v_pred = self._solve(np.array(xs[:3]), v_in, spin, apex, w_lo, w_hi, scale,
                                     tau_now=self.plan.eval(te)[2], t_av=t_av)
        return dict(t_hit=t_hit, p_hit=np.array(xs[:3]), v_in=v_in, apex_in=apex_in, apex=apex,
                    tw=tw, t_av=t_av, tau=tau, w=w, v_out=v_pred, z_hit=z_hit)

    def _keys(self, te, now_key, h):
        a, tw, w, z_hit = self.cfg.control.plan_accel, h['tw'], h['w'], h['z_hit']
        t_a, t_b = h['t_hit'] - tw, h['t_hit'] + tw
        t_stop = t_b + max(abs(w) / a, 0.02)        # 撞完以規劃加速度煞停，再慢慢回中立
        zero = np.zeros(2)
        return [now_key,
                Key(t_a, z_hit - w * tw, w, h['tau'], zero),
                Key(t_b, z_hit + w * tw, w, h['tau'].copy(), zero),
                Key(t_stop, z_hit + w * tw + w * abs(w) / (2 * a), 0.0, h['tau'].copy(), zero),
                Key(t_stop + self.cfg.control.ret_time, 0.0, 0.0, zero, zero)]

    def _replan(self, t, te):
        c, L = self.cfg.control, self.cfg.geom.heave_limit
        if self._committed(te) or self.est.mode != FLIGHT:
            return
        a = c.plan_accel
        z0, zd0, tau0, taud0 = self.plan.eval(te)
        now_key = Key(te, z0, zd0, tau0, taud0)
        chosen, last, cache = None, None, {}
        # 伺服跟不上時逐步讓步：先犧牲升降（接球少退一點／拍球少打一點高度），
        # 最後才犧牲水平修正——球掉出盤外比彈高一點嚴重得多（側拋進來時尤其重要）
        if self.mode == CUSHION:
            scales = ((1, 1), (0.7, 1), (0.45, 1), (0.2, 1), (0, 1), (0.45, 0.5), (0, 0.5), (0, 0))
        else:
            scales = ((1, 1), (0.8, 1), (0.6, 1), (0.4, 1), (0.6, 0.5), (0.4, 0))
        # 先試「撞擊點在中立」（揮拍前後行程對稱、不會越打越偏）；伺服跟不上再試「從目前位置直接接上」
        for scale in scales:
            for strategy in ('center', 'direct'):
                z_hit = 0.0 if strategy == 'center' else z0
                for _ in range(3):      # 撞擊高度會改變撞擊時間，來回算幾次讓兩者一致
                    h = self._target(t, te, z_hit, (z0, zd0), scale, cache)
                    if h == 'settle':
                        self._enter_pd(SETTLE, t, te)
                        return
                    if h == 'same':
                        return
                    if h is None or strategy == 'center':
                        break
                    w, tw = h['w'], h['tw']
                    z_a = z0 + (zd0 + w) / 2 * h['t_av']     # 等加速度從目前狀態接到 w
                    margin = 2 * abs(w) * tw + w * w / (2 * a) + 0.001
                    if w >= 0:
                        z_a = max(-L + 0.001, min(L - margin, z_a))
                    else:
                        z_a = min(L - 0.001, max(-L + margin, z_a))
                    if abs(z_a + w * tw - z_hit) < 0.0005:
                        break
                    z_hit = z_a + w * tw
                if h is None:
                    continue
                last = h
                keys = self._keys(te, now_key, h)
                if h['t_av'] >= 0.03 and self._feasible(keys):
                    chosen = h
                    break
            if chosen is not None:
                break
        if chosen is None:
            # 伺服跟不上：同一跳已有規劃就沿用；很低的來球就改成慢慢下沉
            if (self.mode == CUSHION and last is not None and last['apex_in'] < c.absorb_apex
                    and not self._planned(last['t_hit'])):
                self._enter_pd(SETTLE, t, te)
            return
        h = chosen
        self._z_hit = h['z_hit']
        t_hit, tw = h['t_hit'], h['tw']
        self.plan = Plan(keys, window=(t_hit - tw, t_hit, t_hit + tw))
        rec = dict(t_plan=t, t_hit=t_hit, p_hit=h['p_hit'], v_in=h['v_in'], v_out=h['v_out'],
                   tau=h['tau'], w=h['w'], z_hit=h['z_hit'], apex=h['apex'], mode=self.mode)
        if self.hits and abs(self.hits[-1]['t_hit'] - t_hit) < 0.06:
            self.hits[-1] = rec
        else:
            self.hits.append(rec)
            self.trim_log.append((t, self.tilt_trim.copy(), self.apex_bias))
