import shutil

import cv2
import numpy as np
import pytest

from ppreal.camera import FakeCamera
from ppreal.camtool import record
from ppreal.report import iter_mjpeg, load_camera, report
from ppreal.runlog import Run

from test_ppreal_camtool import ball_pos, moving_ball


@pytest.fixture
def fake_run(tmp_path):
    run = Run("cam", note="測試", base=tmp_path)
    run.event("mark", "我推了")
    record(FakeCamera(moving_ball, fps=30), run, max_frames=60)
    return run.path


def test_load_camera_relative_times(fake_run):
    d = load_camera(fake_run)
    assert len(d["frame"]) == 60
    assert d["t_read_s"][1] - d["t_read_s"][0] == pytest.approx(1 / 30, abs=1e-6)
    assert np.isnan(d["t_cap_s"]).all()          # 假相機跟 Windows 一樣沒有相機時間戳
    assert np.nanmax(np.abs(d["u_px"] - [ball_pos(k)[0] for k in range(60)])) < 0.5


def test_times_are_relative_to_run_start(fake_run):
    # 報告的時間軸＝開始後幾秒；假相機的時鐘也要跟紀錄的時鐘對得上
    d = load_camera(fake_run)
    assert 0 <= d["t_read_s"][0] < 5


def test_iter_mjpeg_all_frames(fake_run):
    frames = list(iter_mjpeg(fake_run))
    assert [k for k, _ in frames] == list(range(60))
    assert frames[0][1].shape == (480, 640)


def test_report_files(fake_run):
    md = report(fake_run, video=True)
    text = md.read_text(encoding="utf-8")
    assert "每秒" in text and "60" in text and "我推了" in text
    assert (fake_run / "report" / "timeline.png").stat().st_size > 1000
    cap = cv2.VideoCapture(str(fake_run / "report" / "overlay.mp4"))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n >= 54


def test_report_with_no_frames(tmp_path):
    run = Run("cam", base=tmp_path)
    record(FakeCamera(moving_ball, fail_after=0), run, seconds=60, fail_limit=5)
    text = report(run.path, video=True).read_text(encoding="utf-8")
    assert "camera_lost" in text and "張數：0" in text


def test_report_handles_partial_last_row(fake_run):
    p = fake_run / "camera.csv"
    lines = p.read_text(encoding="utf-8").splitlines()
    cells = lines[-1].split(",")
    cells[-1] = ""                       # jpg_len 還沒寫到就斷電
    lines[-1] = ",".join(cells)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert len(list(iter_mjpeg(fake_run))) == 59
    assert report(fake_run, video=False).exists()


def test_report_ignores_truncated_tail(fake_run):
    p = fake_run / "cam.mjpeg"
    data = p.read_bytes()
    p.write_bytes(data[:-1000])
    assert len(list(iter_mjpeg(fake_run))) == 59
    assert report(fake_run, video=False).exists()
