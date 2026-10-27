"""黑色霧面平台上找白色乒乓球。"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

NAN = float("nan")


@dataclass
class Ball:
    found: bool
    u: float = NAN          # 球心，像素
    v: float = NAN
    r: float = NAN          # 等效半徑 sqrt(面積/π)
    area: float = 0.0
    score: float = 0.0      # 圓度 4π面積/周長²，正圓是 1


@dataclass
class DetectParams:
    thresh: int | None = None       # None＝ROI 裡用 Otsu 自己找
    min_contrast: float = 40.0      # 門檻至少比背景中位數亮這麼多，沒有球的畫面才不會把雜訊當球
    r_min: float = 6.0
    r_max: float = 90.0
    min_circularity: float = 0.6
    roi: tuple[float, float, float] | None = None   # 只在這個圓裡找（u, v, 半徑），通常是平台
    r_expect: float | None = None   # 校正過就知道球該多大（像素）：只收 0.6–1.5 倍、挑最接近的


_KERNEL = np.ones((3, 3), np.uint8)


def detect_ball(gray: np.ndarray, p: DetectParams = DetectParams()) -> Ball:
    img = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = None
    if p.roi is not None:
        mask = np.zeros(img.shape, np.uint8)
        cv2.circle(mask, (round(p.roi[0]), round(p.roi[1])), round(p.roi[2]), 255, -1)
        img = cv2.bitwise_and(img, mask)
    if p.thresh is not None:
        t = p.thresh
    else:
        inside = img[mask > 0] if mask is not None else img.ravel()
        if inside.size == 0:
            return Ball(False)
        otsu, _ = cv2.threshold(inside.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = max(otsu, float(np.median(inside[::16])) + p.min_contrast)
    _, bw = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, _KERNEL)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lo, hi = p.r_min, p.r_max
    if p.r_expect:
        lo, hi = max(lo, 0.6 * p.r_expect), min(hi, 1.5 * p.r_expect)
    cands = []
    for c in contours:
        m = cv2.moments(c)
        area = m["m00"]
        if area <= 0:
            continue
        r = math.sqrt(area / math.pi)
        if not lo <= r <= hi:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * math.pi * area / (per * per) if per > 0 else 0.0
        if circ < p.min_circularity:
            continue
        cands.append(Ball(True, m["m10"] / area, m["m01"] / area, r, area, min(circ, 1.0)))
    if not cands:
        return Ball(False)
    if p.r_expect:   # 大小最像球的；一樣像就挑比較圓的
        return min(cands, key=lambda b: (abs(b.r - p.r_expect) / p.r_expect, -b.score))
    return max(cands, key=lambda b: (b.score, b.area))   # 不知道球多大：挑最圓的，不挑最大的
