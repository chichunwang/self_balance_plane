"""模擬參數：規格全部集中在這裡，改規格只改這個檔。

單位一律 SI（公尺、秒、弧度），註解裡的 mm、° 只是方便看。
標「假設」的數字沒有查到可信來源，實機要自己量或測。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

G = 9.81


@dataclass
class Geometry:
    """3-RRS 平台幾何。中立姿態：伺服臂水平、連桿垂直，臂端剛好在球頭正下方。"""
    plate_diameter: float = 0.198     # 收合要在 20 cm 內（2026-10-26 改版，原本 0.220）
    plate_thickness: float = 0.008
    r_p: float = 0.080                # 平台三個球頭所在圓的半徑
    arm: float = 0.045                # 伺服臂長
    rod: float = 0.100                # 連桿中心距
    z_shaft: float = 0.050            # 伺服軸離底板高度（40 時肘關節的軸承座會碰底板，見 docs/bom.md）
    joint_to_surface: float = 0.018   # 球頭中心到平台上表面（板厚 8＋球心到板底 10：球頭座 3＋球頭的頭半徑 6＋間隙 1）
    legs_deg: tuple = (90.0, 210.0, 330.0)
    servo_limit_deg: float = 60.0
    heave_limit: float = 0.022        # 控制器允許的升降行程（±）
    arm_outward: bool = True          # 臂朝外、伺服靠中間（改版）；False＝第一版的臂朝內

    @property
    def arm_sign(self) -> float:
        """臂朝內 +1、朝外 −1。臂端＝伺服軸＋arm·(−s·cosθ·徑向＋sinθ·上)"""
        return -1.0 if self.arm_outward else 1.0

    @property
    def r_b(self) -> float:
        """伺服軸所在圓的半徑"""
        return self.r_p + self.arm_sign * self.arm

    @property
    def z_joint0(self) -> float:
        return self.z_shaft + self.rod

    @property
    def z_surface0(self) -> float:
        return self.z_joint0 + self.joint_to_surface


@dataclass
class ServoSpec:
    name: str
    sec_per_60: float                 # 規格表空載速度
    deadband_us: float
    pwm_hz: float                     # 命令更新率（PCA9685 的 PWM 頻率）
    stall_torque: float = 2.0         # 堵轉扭力 N·m（MuJoCo 馬達模型用）
    t_mech: float = 0.015             # 馬達機械時間常數（假設；決定轉子慣量）
    load_derate: float = 0.85         # 帶載速度打折（假設；簡化模型與控制器規劃用）
    t_accel: float = 0.030            # 靜止到全速所需時間（假設；簡化模型用）
    kp: float = 60.0                  # 內部位置迴路增益 1/s（假設）
    deadtime: float = 0.004           # 內部處理延遲（假設）
    us_per_rad: float = 2000 / math.pi  # 500–2500 µs 對應 180°
    pwm_counts: int = 4096            # PCA9685 為 12 bit

    @property
    def w_noload(self) -> float:
        return math.radians(60) / self.sec_per_60

    @property
    def w_max(self) -> float:
        return self.w_noload * self.load_derate

    @property
    def a_max(self) -> float:
        return self.w_max / self.t_accel

    @property
    def deadband(self) -> float:
        return self.deadband_us / self.us_per_rad

    @property
    def quantum(self) -> float:
        """PCA9685 一格代表的角度"""
        return 1e6 / (self.pwm_hz * self.pwm_counts) / self.us_per_rad


SERVOS = {
    # 規格表：0.16 s/60°、19 kg·cm @5V；0.14 s/60°、21.5 kg·cm @6.8V；死區 3 µs。6 V 取內插
    "DS3218": ServoSpec("DS3218 @6V", sec_per_60=0.15, deadband_us=3.0, pwm_hz=333.0, stall_torque=2.0),
    # 規格表：0.14 s/60°、11 kg·cm @6V；多數專案用 50 Hz 驅動；死區 5 µs 是常見標示值；仿品多，實際可能更慢
    "MG996R": ServoSpec("MG996R @6V", sec_per_60=0.14, deadband_us=5.0, pwm_hz=50.0, stall_torque=1.08,
                        deadtime=0.008),
}


@dataclass
class MechSpec:
    """MuJoCo 才用到的機構細節（質量、齒輪、接觸）。"""
    plate_density: float = 1190.0     # 壓克力 kg/m³
    # 下面三個是 CAD 估的（列印件 PETG、螺絲和球頭的重量是估計值），見 out/cad/report.md；做出來秤過再改
    plate_extra: float = 0.025        # 三個球頭座（黏的）＋M3 套筒＋M3×30＋壓電片＋GY-521 與座子（2026-10-27）
    arm_mass: float = 0.008           # 列印的伺服臂＋四顆螺絲＋金屬套筒
    rod_mass: float = 0.021           # 列印的連桿＋金屬魚眼 SA3B/K＋兩顆 F684ZZ＋M4×25＋M3 螺帽（2026-10-27）
    gear_stiffness: float = 50.0      # 伺服輸出軸剛性 N·m/rad（假設：1 N·m 約偏 1.1°）
    gear_damping: float = 0.05
    backlash_deg: float = 0.0         # 齒隙（假設；便宜伺服常見 0.5–1°）
    motor_friction: float = 0.03      # 齒輪箱摩擦 N·m（假設）
    mj_dt: float = 0.00025            # MuJoCo 步長


@dataclass
class PlantErrors:
    """真實世界沒校正好的部分（控制器不知道）：每顆伺服的中位誤差與比例誤差。"""
    trim_deg: tuple = (0.0, 0.0, 0.0)
    gain_err: tuple = (0.0, 0.0, 0.0)


@dataclass
class BallSpec:
    mass: float = 0.0027              # 40 mm 比賽球 2.7 g
    radius: float = 0.020
    k_inertia: float = 2 / 3          # 薄殼球 I = 2/3·m·r²
    cd: float = 0.45                  # 阻力係數（Re≈5000 球體的典型值）
    rho_air: float = 1.20

    @property
    def beta(self) -> float:
        """空氣阻力加速度 = beta·|v|·v"""
        return 0.5 * self.rho_air * self.cd * math.pi * self.radius ** 2 / self.mass

    @property
    def g_eff(self) -> float:
        """扣掉浮力後的重力加速度"""
        vol = 4 / 3 * math.pi * self.radius ** 3
        return G * (1 - self.rho_air * vol / self.mass)


@dataclass
class SurfaceSpec:
    e: float = 0.87                   # 恢復係數（假設；ITTF 球桌 30 cm 落下彈約 23 cm，含空氣阻力換算約 0.9；要自己量）
    mu: float = 0.25                  # 球與壓克力的摩擦係數（假設）
    v_rest: float = 0.02              # 相對法向速度低於此值當成非彈性（數值穩定用）


@dataclass
class CameraSpec:
    fps: float = 120.0
    latency: float = 0.020            # 曝光中點到結果可用（假設）
    height: float = 0.60              # 鏡頭在中立板面上方多高（往下拍）
    width_px: int = 1280              # OV9281 原生 1280×800@120 fps
    height_px: int = 800
    hfov_deg: float = 100.0           # 廣角 M12 鏡頭：側拋進來的球才看得早（60° 時只剩最後 0.1 秒）
    sigma_center_px: float = 0.15     # 球心定位雜訊（假設）
    sigma_radius_px: float = 0.25     # 半徑量測雜訊（假設），決定高度誤差

    @property
    def focal_px(self) -> float:
        return self.width_px / 2 / math.tan(math.radians(self.hfov_deg / 2))


@dataclass
class PiezoSpec:
    enabled: bool = True
    latency: float = 0.001
    jitter: float = 0.0002
    min_speed: float = 0.10           # 比這更輕的撞擊偵測不到（假設）
    radius: float = 0.0175            # 只用在畫圖和 MuJoCo 的外觀；要跟 cad/hw.py 的 PIEZO_D 一致


@dataclass
class ControlSpec:
    rate_hz: float = 500.0
    lead: float | None = None         # 命令提前量；None＝依伺服參數自動估
    tilt_max_deg: float = 6.0         # 下沉時的傾角上限
    hit_tilt_max_deg: float = 10.0    # 拍球／接球撞擊時的傾角上限（側拋進來要靠大傾角吃掉水平速度）。
                                      # 2026-10-27 從 12 改 10：金屬魚眼 SA3B/K 只能偏 13°，12° 時魚眼要偏 12°、餘量只剩 1°
    leg_travel: float = 0.034         # 每隻腳上下最多走多少（升降＋傾斜合計；臂 45 mm 約 ±50°）
    heave_speed_frac: float = 0.8     # 規劃的升降速度上限＝伺服極限×此比例
    leg_accel_frac: float = 0.8       # 規劃的腳加速度上限＝伺服極限×此比例（側拋時傾斜要快）
    # 拍球
    apex: float = 0.15                # 目標彈跳高度（離開撞擊點後上升多少）
    center_gain: float = 0.6          # 每跳修正幾成的位置誤差
    hit_window: float = 0.015         # 撞擊前後各保持等速的時間（飛行時間短時會自動縮短）
    plan_accel: float = 4.0           # 規劃揮拍時假設的平台加速度（m/s²）
    commit: float = 0.030             # 視窗開始前多久鎖定計畫
    ret_time: float = 0.15            # 撞完回到中立的時間
    # 緩衝
    cushion_ratio: float = 0.5        # 每跳高度乘上這個比例
    absorb_apex: float = 0.02         # 來球低於此高度：這一跳整個吸掉（平台撞擊瞬間往下退）
    rest_apex: float = 0.003          # 來球低於此高度就不揮拍了，等球自己停
    sink_speed: float = 0.04          # 來不及揮拍時，平台慢慢下沉吸收能量
    sink_depth: float = 0.012         # 最多沉到中立下方多少
    # 平衡
    bal_wn: float = 3.0
    bal_zeta: float = 0.8
    bal_tilt_max_deg: float = 8.0     # 平衡時的傾角上限。2026-10-26 從 5 改 8：手推 0.30 m/s 也救得回，平衡精度不變
    bal_ki: float = 2.0               # 自動水平：平衡時把長期偏差累積成傾斜修正（1/(m·s)）
    tilt_learn: float = 0.15          # 自動水平：拍球時每跳用「實際水平反彈－預期」反推平台歪多少，修正這個比例
    apex_learn: float = 0.05          # 高度外迴路：壓電片量到的實際高度，很慢地修正長期偏差（太快會把雜訊修進去）
    trim_max_deg: float = 2.0         # 自動水平修正的上限
    tilt_rate: float = 1.5            # 平衡／下沉時傾斜變化率上限（tan 值/秒，≈86°/s）
    home_speed: float = 0.02
    # 控制器「以為」的模型：刻意和真實值有落差，看控制夠不夠強健
    e_model: float = 0.85             # 起始值；有壓電片時會用兩次撞擊的間隔自動校正
    e_adapt: bool = True
    mu_model: float = 0.25            # 控制器以為的球–板摩擦係數（撞擊模型含自旋，側拋時很重要）
    roll_factor: float = 0.6          # 薄殼球滾動：a = 0.6·g·sinθ
    q_flight: float = 0.3             # 飛行過程雜訊（m/s²）
    q_roll: float = 0.08              # 太大會讓平衡時傾斜跟著相機雜訊抖


@dataclass
class Config:
    geom: Geometry = field(default_factory=Geometry)
    servo: ServoSpec = field(default_factory=lambda: replace(SERVOS["DS3218"]))
    ball: BallSpec = field(default_factory=BallSpec)
    surface: SurfaceSpec = field(default_factory=SurfaceSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    piezo: PiezoSpec = field(default_factory=PiezoSpec)
    control: ControlSpec = field(default_factory=ControlSpec)
    mech: MechSpec = field(default_factory=MechSpec)
    errors: PlantErrors = field(default_factory=PlantErrors)
    dt: float = 0.0005
