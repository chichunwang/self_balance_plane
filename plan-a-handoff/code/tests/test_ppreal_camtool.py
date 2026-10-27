import csv
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from ppreal import camtool
from ppreal.calib import CamMap
from ppreal.camera import FakeCamera
from ppreal.camtool import CAMERA_COLUMNS, check_criteria, find_platform_circle, frame_stats, record
from ppreal.runlog import Run, read_meta

PROJECT = Path(__file__).resolve().parent.parent


def ball_pos(k):
    a = 2 * math.pi * k / 60
    return 320 + 100 * math.cos(a), 240 + 100 * math.sin(a)


def moving_ball(k):
    img = np.full((480, 640), 20, np.uint8)
    u, v = ball_pos(k)
    cv2.circle(img, (round(u * 16), round(v * 16)), 20 * 16, 230, -1, cv2.LINE_AA, shift=4)
    return img


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_record_writes_mjpeg_and_csv(tmp_path):
    run = Run("cam", base=tmp_path)
    stats = record(FakeCamera(moving_ball, fps=30), run, max_frames=60)
    assert stats["frames"] == 60 and stats["recorded"] == 60 and stats["status"] == "done"
    with open(run.path / "camera.csv", encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == CAMERA_COLUMNS
    rows = read_csv(run.path / "camera.csv")
    assert len(rows) == 60
    data = (run.path / "cam.mjpeg").read_bytes()
    for k, r in enumerate(rows):
        buf = np.frombuffer(data[int(r["jpg_off"]):int(r["jpg_off"]) + int(r["jpg_len"])], np.uint8)
        assert cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).shape == (480, 640)
        assert r["found"] == "1"
        u, v = ball_pos(int(r["frame"]))
        assert abs(float(r["u_px"]) - u) < 0.5 and abs(float(r["v_px"]) - v) < 0.5
    meta = read_meta(run.path)
    assert meta["status"] == "done" and meta["summary"]["frames"] == 60
    assert meta["camera"]["backend"] == "fake"


def test_record_maps_to_mm_when_calibrated(tmp_path):
    run = Run("cam", base=tmp_path)
    record(FakeCamera(moving_ball), run, max_frames=10, cam_map=CamMap(320, 240, 0.5, 90.0))
    for r in read_csv(run.path / "camera.csv"):
        assert math.hypot(float(r["x_mm"]), float(r["y_mm"])) == pytest.approx(50.0, abs=0.5)


def test_record_without_detection_leaves_blank(tmp_path):
    run = Run("cam", base=tmp_path)
    record(FakeCamera(moving_ball), run, max_frames=5, detect=None)
    rows = read_csv(run.path / "camera.csv")
    assert all(r["found"] == "" and r["u_px"] == "" for r in rows)


def test_record_drops_frames_instead_of_blocking(tmp_path, monkeypatch):
    real = camtool.encode_jpeg

    def slow(gray, quality):
        time.sleep(0.02)
        return real(gray, quality)

    monkeypatch.setattr(camtool, "encode_jpeg", slow)
    run = Run("cam", base=tmp_path)
    stats = record(FakeCamera(moving_ball), run, max_frames=120, max_queue=5, detect=None)
    assert stats["recorded"] + stats["dropped"] == 120
    assert stats["dropped"] > 0
    rows = read_csv(run.path / "camera.csv")
    assert len(rows) == 120
    skipped = [r for r in rows if r["rec"] == "0"]
    assert len(skipped) == stats["dropped"] and all(r["jpg_off"] == "" for r in skipped)


def test_record_stops_cleanly_when_camera_dies(tmp_path):
    run = Run("cam", base=tmp_path)
    stats = record(FakeCamera(moving_ball, fail_after=40), run, seconds=60, fail_limit=5)
    assert stats["status"] == "camera_lost" and stats["frames"] == 40
    assert len(read_csv(run.path / "camera.csv")) == 40
    assert read_meta(run.path)["status"] == "camera_lost"


def test_record_stops_at_seconds(tmp_path):
    run = Run("cam", base=tmp_path)
    stats = record(FakeCamera(moving_ball, fps=30), run, seconds=2.0, detect=None)
    assert stats["frames"] == 60


def test_record_marks_crash_and_reraises(tmp_path):
    def boom(k):
        if k == 20:
            raise RuntimeError("壞了")
        return moving_ball(k)

    run = Run("cam", base=tmp_path)
    with pytest.raises(RuntimeError):
        record(FakeCamera(boom), run, max_frames=60)
    assert read_meta(run.path)["status"] == "crashed"
    assert len(read_csv(run.path / "camera.csv")) == 20


def test_recorder_error_is_reported(tmp_path, monkeypatch):
    real = camtool.encode_jpeg
    calls = {"n": 0}

    def flaky(gray, quality):
        calls["n"] += 1
        if calls["n"] > 10:
            raise OSError("磁碟滿了")
        return real(gray, quality)

    monkeypatch.setattr(camtool, "encode_jpeg", flaky)
    run = Run("cam", base=tmp_path)
    st = record(FakeCamera(moving_ball), run, max_frames=100, detect=None, max_queue=1000)
    assert st["status"] == "rec_error" and st["recorded"] == 10 and st["rec_failed"] == 90
    assert st["recorded"] + st["dropped"] + st["rec_failed"] == st["frames"]
    rows = read_csv(run.path / "camera.csv")
    assert len(rows) == 100 and sum(r["rec"] == "1" for r in rows) == 10
    assert read_meta(run.path)["status"] == "rec_error"
    assert any("磁碟滿了" in e["detail"] for e in read_csv(run.path / "events_tool.csv"))


class StallCam:
    """一直讀不到、每次要等 0.2 秒的相機（像 DSHOW 卡住時）。"""
    settings = {"backend": "stall", "name": "stall"}

    def read(self):
        time.sleep(0.2)
        return None


def test_record_stops_when_camera_stalls(tmp_path):
    run = Run("cam", base=tmp_path)
    t0 = time.monotonic()
    st = record(StallCam(), run, seconds=60, fail_limit=1000, stall_s=1.0)
    assert st["status"] == "camera_lost" and st["frames"] == 0
    assert time.monotonic() - t0 < 2.0


def test_record_copies_calib_into_meta(tmp_path):
    run = Run("cam", base=tmp_path)
    record(FakeCamera(moving_ball), run, max_frames=3, calib_snapshot={"camera": {"platform_r_px": 120}})
    assert read_meta(run.path)["calib"]["camera"]["platform_r_px"] == 120


def test_avg_ball_gives_up_without_ball():
    from ppreal.balldetect import DetectParams
    cam = FakeCamera(lambda k: np.zeros((480, 640), np.uint8), n=500)
    assert camtool._avg_ball(cam, DetectParams()) is None
    assert cam.index <= 30


def test_preview_lines_show_position():
    from ppreal.balldetect import Ball
    b = Ball(True, 100.0, 50.0, 20.0, 1256.0, 0.95)
    assert any("x +12.3 mm" in s and "y -4.5 mm" in s for s in camtool.preview_lines(30.0, 1.5, 45, b, (12.34, -4.5)))
    assert any("u 100.0" in s for s in camtool.preview_lines(30.0, 1.5, 45, b, None))
    assert any("no ball" in s for s in camtool.preview_lines(30.0, 1.5, 45, Ball(False), None))


def test_cli_view_help():
    r = subprocess.run([sys.executable, "-m", "ppreal", "cam", "view", "--help"],
                       cwd=PROJECT, capture_output=True, timeout=60)
    assert r.returncode == 0 and "不存檔" in r.stdout.decode("utf-8")


def test_frame_stats_exact():
    t = (np.arange(90) * 1e9 / 30).astype(np.int64)
    s = frame_stats(t, 30)
    assert s["fps"] == pytest.approx(30.0, abs=0.01) and s["gaps"] == 0 and s["n"] == 90
    s = frame_stats(np.delete(t, 50), 30)
    assert s["gaps"] == 1 and s["est_dropped"] == 1
    assert s["dt_max_ms"] == pytest.approx(66.67, abs=0.01)


def test_check_criteria():
    t = (np.arange(300) * 1e9 / 30).astype(np.int64)
    assert all(ok for _, ok, _ in check_criteria(frame_stats(t, 30), 30))
    t[10:] += 100_000_000                   # 第 10 張起卡了 100 ms
    res = dict((name, ok) for name, ok, _ in check_criteria(frame_stats(t, 30), 30))
    assert res["沒有掉張"] is False


def test_find_platform_circle():
    img = np.full((480, 640), 120, np.uint8)
    cv2.circle(img, (330, 235), 150, 20, -1, cv2.LINE_AA)
    found = find_platform_circle(img)
    assert found is not None
    u, v, r = found
    assert abs(u - 330) < 3 and abs(v - 235) < 3 and abs(r - 150) < 4


def test_find_platform_circle_none_on_blank():
    assert find_platform_circle(np.full((480, 640), 60, np.uint8)) is None


def test_cli_fov_prints_height():
    r = subprocess.run([sys.executable, "-m", "ppreal", "cam", "fov", "--d", "50", "--w", "52"],
                       cwd=PROJECT, capture_output=True, timeout=60)
    out = r.stdout.decode("utf-8")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "32" in out and "55" in out
