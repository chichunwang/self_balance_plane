import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from ppreal import calib
from ppreal.calib import CamMap, fit_circle, fov_report, map_from_points


def test_fov_report_example():
    r = fov_report(50, 52)
    assert r["hfov_deg"] == pytest.approx(54.95, abs=0.01)
    assert r["vfov_deg"] == pytest.approx(42.6, abs=0.05)
    assert r["min_height_cm"] == pytest.approx(32.05, abs=0.01)
    r = fov_report(50, 52, mount_cm=50)
    assert r["mm_per_px"] == pytest.approx(0.8125, abs=1e-4)
    assert r["platform_px"] == pytest.approx(243.7, abs=0.1)
    assert r["ball_px"] == pytest.approx(49.2, abs=0.1)


def test_fov_report_uses_short_side_when_rotated():
    # 相機轉 90° 裝：量到的「寬」比「高」短，要用短的那邊算高度
    r = fov_report(50, 39, h_cm=52)
    assert r["min_height_cm"] == pytest.approx(32.05, abs=0.01)


def test_fit_circle():
    rng = np.random.default_rng(1)
    a = np.radians([10, 80, 150, 220, 300])
    pts = np.c_[321.5 + 120 * np.cos(a), 238.0 + 120 * np.sin(a)] + rng.normal(0, 0.3, (5, 2))
    u, v, r = fit_circle(pts)
    assert abs(u - 321.5) < 0.5 and abs(v - 238.0) < 0.5 and abs(r - 120) < 0.5


def test_fit_circle_needs_three_points():
    with pytest.raises(ValueError):
        fit_circle(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_map_normal_orientation():
    m = map_from_points((320, 240), (320, 140), (420, 240), 1.0)
    assert not m.mirror
    assert m.to_mm(320, 140) == pytest.approx((0, 100), abs=1e-6)
    assert m.to_mm(420, 240) == pytest.approx((100, 0), abs=1e-6)


def test_map_rotated_90():
    m = map_from_points((320, 240), (420, 240), (320, 340), 1.0)
    assert not m.mirror
    assert m.to_mm(420, 240) == pytest.approx((0, 100), abs=1e-6)
    assert m.to_mm(320, 340) == pytest.approx((100, 0), abs=1e-6)


def test_map_detects_mirror():
    m = map_from_points((320, 240), (320, 140), (220, 240), 1.0)
    assert m.mirror
    assert m.to_mm(220, 240) == pytest.approx((100, 0), abs=1e-6)
    assert m.to_mm(320, 140) == pytest.approx((0, 100), abs=1e-6)


def test_map_accepts_right_point_off_by_30deg():
    # 球放在 3 號腳旁邊（右後 30°）也要接受：右邊那點只用來看畫面有沒有左右翻
    a = math.radians(-30)
    right = (320 + 100 * math.cos(a), 240 - 100 * math.sin(a))
    m = map_from_points((320, 240), (320, 140), right, 1.0)
    assert not m.mirror
    assert m.to_mm(320, 140) == pytest.approx((0, 100), abs=1e-6)


def test_map_rejection_message_says_actual_angle():
    a = math.radians(60)                        # 跟正面只差 30°
    right = (320 + 100 * math.cos(a), 240 - 100 * math.sin(a))
    with pytest.raises(ValueError) as e:
        map_from_points((320, 240), (320, 140), right, 1.0)
    assert "差了 30°" in str(e.value)


def test_map_rejects_bad_points():
    with pytest.raises(ValueError):
        map_from_points((320, 240), (320, 140), (320, 190), 1.0)    # 右邊那點跟正面同方向
    with pytest.raises(ValueError):
        map_from_points((320, 240), (320, 240), (420, 240), 1.0)    # 正面那點沒移動


def test_cammap_dict_roundtrip():
    m = CamMap(321.5, 238.0, 0.81, 90.0, False)
    assert CamMap.from_dict(m.to_dict()) == m


def test_update_keeps_history(tmp_path):
    p = tmp_path / "calib.json"
    calib.update("camera", {"mm_per_px": 0.8}, "第一次", path=p)
    calib.update("table", {"tilt_fb_deg": 0.1}, "量桌子", path=p)
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["camera"]["mm_per_px"] == 0.8 and d["table"]["tilt_fb_deg"] == 0.1
    assert [h["what"] for h in d["history"]] == ["第一次", "量桌子"]
    assert calib.load(p)["version"] == 1


def test_load_missing_file(tmp_path):
    assert calib.load(tmp_path / "nope.json") == {"version": 1, "history": []}


def test_calib_path_by_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert calib.calib_path() == Path.home() / "pp" / "calib.json"
    monkeypatch.setattr(sys, "platform", "win32")
    assert calib.calib_path().name == "calib.json" and calib.calib_path().parent.name == "real"
