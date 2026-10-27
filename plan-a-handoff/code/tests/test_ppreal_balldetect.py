import math

import numpy as np

from ppreal.balldetect import DetectParams, detect_ball

H, W = 480, 640


def background(seed=0, level=20.0, noise=5.0):
    rng = np.random.default_rng(seed)
    return np.full((H, W), level) + rng.normal(0.0, noise, (H, W))


def coverage(u, v, r, ss=4):
    """每個像素被圓蓋住的比例（4×4 超取樣），邊緣才會有次像素資訊。"""
    cov = np.zeros((H, W))
    x0, x1 = max(0, int(u - r - 2)), min(W, int(u + r + 3))
    y0, y1 = max(0, int(v - r - 2)), min(H, int(v + r + 3))
    if x0 >= x1 or y0 >= y1:
        return cov
    off = (np.arange(ss) + 0.5) / ss - 0.5
    ys, xs = np.mgrid[y0:y1, x0:x1]
    acc = np.zeros(ys.shape)
    for dy in off:
        for dx in off:
            acc += ((xs + dx - u) ** 2 + (ys + dy - v) ** 2) <= r * r
    cov[y0:y1, x0:x1] = acc / ss / ss
    return cov


def paint(img, cov, value):
    return img * (1 - cov) + value * cov


def to_u8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def test_finds_disc_subpixel():
    img = to_u8(paint(background(), coverage(320.4, 240.7, 25), 230))
    b = detect_ball(img)
    assert b.found
    assert abs(b.u - 320.4) < 0.3 and abs(b.v - 240.7) < 0.3
    assert abs(b.r - 25) < 1.0


def test_blank_frame_not_found():
    b = detect_ball(to_u8(background()))
    assert not b.found
    assert math.isnan(b.u)


def test_picks_round_blob_over_bright_rectangle():
    img = paint(background(), coverage(200, 200, 25), 230)
    img[335:365, 390:510] = 230          # 120×30 的白色長方形
    b = detect_ball(to_u8(img))
    assert b.found
    assert abs(b.u - 200) < 0.5 and abs(b.v - 200) < 0.5


def test_prefers_rounder_blob_over_bigger_ellipse():
    # 比球大的圓亮東西（握拳的手、杯子）不能贏過球
    import cv2
    img = paint(background(), coverage(200, 200, 25), 230)
    ell = np.zeros((H, W), np.uint8)
    cv2.ellipse(ell, (470, 330), (55, 42), 0, 0, 360, 255, -1)
    img = img * (1 - ell / 255) + 170 * (ell / 255)
    b = detect_ball(to_u8(img))
    assert abs(b.u - 200) < 1 and abs(b.v - 200) < 1


def test_expected_radius_rejects_wrong_size():
    big = to_u8(paint(background(), coverage(470, 330, 48), 230))
    assert not detect_ball(big, DetectParams(r_expect=25)).found
    ok = to_u8(paint(background(), coverage(200, 200, 25), 230))
    assert detect_ball(ok, DetectParams(r_expect=25)).found


def test_roi_excludes_outside():
    img = to_u8(paint(background(), coverage(600, 60, 25), 230))
    b = detect_ball(img, DetectParams(roi=(320, 240, 150)))
    assert not b.found


def test_tiny_speck_ignored():
    img = to_u8(paint(background(), coverage(300, 300, 3), 255))
    assert not detect_ball(img).found


def test_shaded_ball_center_close():
    cov = coverage(330.0, 250.0, 25)
    xs = np.arange(W)[None, :].repeat(H, 0)
    shade = 255 - (xs - (330 - 25)) / 50.0 * 105      # 左邊 255、右邊 150
    img = background() * (1 - cov) + np.clip(shade, 150, 255) * cov
    b = detect_ball(to_u8(img))
    assert b.found
    assert abs(b.u - 330.0) < 1.5 and abs(b.v - 250.0) < 1.5


def test_half_ball_at_frame_edge_found():
    img = to_u8(paint(background(), coverage(5, 240, 25), 230))
    b = detect_ball(img)
    assert b.found
    assert b.score < 0.9
