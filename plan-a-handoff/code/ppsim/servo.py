"""業餘伺服模型。

共同的命令管線（ServoCommand）：主機每個 PWM 週期把最新命令送出一次（PCA9685 解析度有限，會量化），
伺服收到後經過內部延遲才換目標；另外可以設定「沒校正好」的誤差：中位偏 trim、每度 µs 比例偏 gain。

Servo      簡化版（純 Python 模擬用）：限速、限加速的位置迴路。
MotorServo 馬達版（MuJoCo 用）：直流馬達扭力隨轉速下降（反電動勢）、轉子慣量、齒輪剛性與齒隙；
           位置感測（電位器）在輸出軸上，跟真的伺服一樣。
"""
import math
from collections import deque


class ServoCommand:
    def __init__(self, spec, theta0=0.0, phase=0.0, limit=math.radians(60), trim=0.0, gain_err=0.0):
        self.s = spec
        self.target = theta0
        self._cmd = theta0
        self._queue = deque()
        self._next_frame = phase
        self._frame = 1.0 / spec.pwm_hz
        self._limit = limit
        self._trim = trim
        self._gain = 1.0 + gain_err

    def command(self, theta):
        """主機端：角度→脈寬（量化）。伺服實際會去的角度＝命令×(1+比例誤差)＋中位誤差。"""
        q = self.s.quantum
        theta = max(-self._limit, min(self._limit, theta))
        self._cmd = round(theta / q) * q * self._gain + self._trim

    def _sample(self, t):
        while t >= self._next_frame:
            self._queue.append((self._next_frame + self.s.deadtime, self._cmd))
            self._next_frame += self._frame
        while self._queue and self._queue[0][0] <= t:
            self.target = self._queue.popleft()[1]


class Servo(ServoCommand):
    def __init__(self, spec, theta0=0.0, phase=0.0, limit=math.radians(60), trim=0.0, gain_err=0.0):
        super().__init__(spec, theta0, phase, limit, trim, gain_err)
        self.theta = theta0
        self.omega = 0.0

    def step(self, t, dt):
        s = self.s
        self._sample(t)
        err = self.target - self.theta
        mag = abs(err)
        if mag <= s.deadband / 2:
            w_des = 0.0
        else:
            # 比例控制，但保證來得及煞車、不超過最高轉速
            w_des = math.copysign(min(s.kp * mag, math.sqrt(2 * s.a_max * mag), s.w_max), err)
        dw = max(-s.a_max * dt, min(s.a_max * dt, w_des - self.omega))
        self.omega += dw
        self.theta += self.omega * dt


class MotorServo(ServoCommand):
    """兩慣量模型：馬達側（轉子慣量折算到輸出軸）經齒輪（彈簧＋齒隙）帶動輸出軸（MuJoCo 的關節）。

    驅動：duty u = clip(Kp·誤差 − Kd·輸出轉速, ±1)，誤差在死區內 u=0；
    馬達扭力 = 堵轉扭力 ×(u − 馬達轉速／空載轉速)，所以轉越快出力越小，最快就是空載轉速。
    """

    def __init__(self, spec, mech, j_load, theta0=0.0, phase=0.0, limit=math.radians(60), trim=0.0,
                 gain_err=0.0):
        super().__init__(spec, theta0, phase, limit, trim, gain_err)
        self.w_nl = spec.w_noload
        self.tau_st = spec.stall_torque
        self.J = spec.stall_torque * spec.t_mech / self.w_nl       # 轉子慣量（折算到輸出軸）
        self.kpu = spec.kp / self.w_nl                              # 小訊號時與簡化版同樣的 kp
        j_tot = self.J + j_load
        wn = math.sqrt(self.tau_st * self.kpu / j_tot)
        self.kdu = max(0.0, 2 * 0.8 * wn * j_tot / self.tau_st - 1 / self.w_nl)   # 阻尼比約 0.8
        self.k_gear = mech.gear_stiffness
        self.c_gear = mech.gear_damping
        self.half_bl = math.radians(mech.backlash_deg) / 2
        self.fric = mech.motor_friction
        self.th_m = theta0
        self.w_m = 0.0
        self.u = 0.0
        self.torque = 0.0

    def update(self, t, h, th_out, w_out):
        """推進馬達側一步，回傳要加在輸出關節上的齒輪扭力。"""
        self._sample(t)
        err = self.target - th_out
        u = 0.0 if abs(err) <= self.s.deadband / 2 else max(-1.0, min(1.0, self.kpu * err - self.kdu * w_out))
        tau_m = self.tau_st * (u - self.w_m / self.w_nl) - self.fric * math.tanh(self.w_m / 0.05)
        d = self.th_m - th_out
        if d > self.half_bl:
            tau_c = self.k_gear * (d - self.half_bl) + self.c_gear * (self.w_m - w_out)
        elif d < -self.half_bl:
            tau_c = self.k_gear * (d + self.half_bl) + self.c_gear * (self.w_m - w_out)
        else:
            tau_c = 0.0                     # 在齒隙裡，齒輪沒咬到
        self.w_m += (tau_m - tau_c) / self.J * h
        self.th_m += self.w_m * h
        self.u, self.torque = u, tau_m
        return tau_c
