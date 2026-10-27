"""感測器：上方相機（針孔模型，用球的半徑推距離）與板下壓電片。"""
import math

import numpy as np


class Camera:
    """鏡頭在 (0, 0, z_cam) 往下拍。球心水平位置很準，高度要靠「球看起來多大」推，
    離鏡頭越遠誤差越大（∝ 距離²）。"""

    def __init__(self, spec, z_cam, ball_radius, rng):
        self.s = spec
        self.z_cam = z_cam
        self.R = ball_radius
        self.rng = rng
        self.f = spec.focal_px
        self.tan_h = math.tan(math.radians(spec.hfov_deg / 2))
        self.tan_v = self.tan_h * spec.height_px / spec.width_px
        self.next_t = rng.uniform(0, 1 / spec.fps)

    def _sig(self, p):
        D = self.z_cam - p[2]
        s_lat = self.s.sigma_center_px * D / self.f
        s_dep = self.s.sigma_radius_px * D * D / (self.f * self.R)
        return D, s_lat, s_dep

    def meas_cov(self, p):
        """量測共變異：深度誤差會順著視線方向把 x、y 一起帶偏"""
        D, s_lat, s_dep = self._sig(p)
        j = np.array([p[0] / D, p[1] / D, -1.0])
        return s_lat ** 2 * np.diag([1.0, 1.0, 0.0]) + s_dep ** 2 * np.outer(j, j)

    def capture(self, t, p):
        """到了拍攝時間就回傳 (送達時間, 拍攝時間, 量到的位置)；球不在畫面內回傳 None"""
        if t < self.next_t:
            return None
        self.next_t += 1 / self.s.fps
        D, s_lat, s_dep = self._sig(p)
        if D < 0.05 or abs(p[0]) > D * self.tan_h or abs(p[1]) > D * self.tan_v:
            return None
        rng = self.rng
        dD = rng.normal(0, s_dep)
        m = np.array([p[0] + rng.normal(0, s_lat) + p[0] / D * dD,
                      p[1] + rng.normal(0, s_lat) + p[1] / D * dD,
                      p[2] - dD])
        return t + self.s.latency, t, m


class Piezo:
    """板下壓電片：撞擊夠重才觸發，微控制器用中斷記下時間戳。"""

    def __init__(self, spec, rng):
        self.s = spec
        self.rng = rng

    def hit(self, t, speed):
        if not self.s.enabled or speed < self.s.min_speed:
            return None
        return t + self.s.latency, t + self.rng.normal(0, self.s.jitter)
