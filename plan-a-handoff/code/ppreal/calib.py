"""校正檔、相機像素換平台座標（mm）、量視角算架高。格式見設計文件第 9 節。"""
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .runlog import PROJECT, write_json

PLATFORM_D_MM = 198.0
BALL_D_MM = 40.0


def calib_path() -> Path:
    if sys.platform.startswith("linux"):
        return Path.home() / "pp" / "calib.json"
    return PROJECT / "out" / "real" / "calib.json"


def load(path: Path | None = None) -> dict:
    p = Path(path) if path else calib_path()
    if not p.exists():
        return {"version": 1, "history": []}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def update(section: str, values: dict, what: str, path: Path | None = None) -> dict:
    p = Path(path) if path else calib_path()
    data = load(p)
    data.setdefault(section, {}).update(values)
    now = datetime.now().isoformat(timespec="seconds")
    data["updated"] = now
    data.setdefault("history", []).append({"time": now, "what": what})
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, data)
    return data


def fov_report(d_cm: float, w_cm: float, h_cm: float | None = None, need_cm: float = 25.0,
               width_px: int = 640, height_px: int = 480, mount_cm: float | None = None) -> dict:
    """離牆 d_cm 時畫面在牆上寬 w_cm、高 h_cm（沒量高就照畫面比例推）。
    min_height_cm：鏡頭到平台面至少多高，畫面短邊才涵蓋得到 need_cm。"""
    if h_cm is None:
        h_cm = w_cm * height_px / width_px
    out = {
        "d_cm": d_cm, "w_cm": w_cm, "h_cm": h_cm,
        "hfov_deg": math.degrees(2 * math.atan(w_cm / (2 * d_cm))),
        "vfov_deg": math.degrees(2 * math.atan(h_cm / (2 * d_cm))),
        "min_height_cm": need_cm * d_cm / min(w_cm, h_cm),
    }
    if mount_cm:
        mm_per_px = w_cm * mount_cm / d_cm * 10 / width_px
        out.update(mount_cm=mount_cm, mm_per_px=mm_per_px,
                   platform_px=PLATFORM_D_MM / mm_per_px, ball_px=BALL_D_MM / mm_per_px)
    return out


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """最小平方法擬合圓：x²+y²+Dx+Ey+F=0。"""
    p = np.asarray(points, float)
    if len(p) < 3:
        raise ValueError("至少要 3 個點才定得出圓")
    a = np.c_[p[:, 0], p[:, 1], np.ones(len(p))]
    b = -(p[:, 0] ** 2 + p[:, 1] ** 2)
    (d, e, f), *_ = np.linalg.lstsq(a, b, rcond=None)
    u, v = -d / 2, -e / 2
    return float(u), float(v), float(math.sqrt(max(u * u + v * v - f, 0.0)))


@dataclass
class CamMap:
    """畫面像素 → 平台座標（mm，x 右、y 前）。front_deg：平台 +y 在畫面裡的方向，
    從畫面右方逆時針量（畫面 v 朝下，算的時候先翻成朝上）。mirror：畫面左右翻了。"""
    u0: float
    v0: float
    mm_per_px: float
    front_deg: float
    mirror: bool = False

    def to_mm(self, u: float, v: float) -> tuple[float, float]:
        mx, my = u - self.u0, -(v - self.v0)
        if self.mirror:
            mx = -mx
        c, s = math.cos(math.radians(self.front_deg)), math.sin(math.radians(self.front_deg))
        return (mx * s - my * c) * self.mm_per_px, (mx * c + my * s) * self.mm_per_px

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CamMap":
        return cls(float(d["u0"]), float(d["v0"]), float(d["mm_per_px"]), float(d["front_deg"]),
                   bool(d.get("mirror", False)))


def map_from_points(center_uv, front_uv, right_uv, mm_per_px: float) -> CamMap:
    """球依序放在平台中心、正面、右邊時量到的三個像素位置 → CamMap。"""
    cu, cv = map(float, center_uv)
    fx, fy = front_uv[0] - cu, -(front_uv[1] - cv)
    rx, ry = right_uv[0] - cu, -(right_uv[1] - cv)
    if math.hypot(fx, fy) < 1 or math.hypot(rx, ry) < 1:
        raise ValueError("正面或右邊那點離中心太近，球沒有移過去？")
    mirror = rx * fy - ry * fx < 0          # 正常的話「右」在「前」順時針 90° 的地方
    if mirror:
        fx, rx = -fx, -rx
    front = math.degrees(math.atan2(fy, fx))
    right = math.degrees(math.atan2(ry, rx))
    off = (right - (front - 90) + 180) % 360 - 180
    if abs(off) > 45:   # 右邊那點只用來看畫面有沒有左右翻，放偏一點沒關係
        between = abs((right - front + 180) % 360 - 180)
        raise ValueError(f"從中心看，正面和右邊兩點差了 {between:.0f}°，應該差 90° 左右；"
                         "右邊那次要放在從中心往右、跟正面成直角的地方")
    return CamMap(cu, cv, float(mm_per_px), front, mirror)
