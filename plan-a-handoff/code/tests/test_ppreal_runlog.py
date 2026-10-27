import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from ppreal import runlog
from ppreal.runlog import Run, CsvWriter, next_seq, run_name, read_meta


def test_run_name_format():
    assert run_name(7, "cam", datetime(2026, 10, 28, 15, 30, 12)) == "0007_20261028-153012_cam"


def test_next_seq_increments_and_persists(tmp_path):
    assert next_seq(tmp_path) == 1
    assert next_seq(tmp_path) == 2
    assert (tmp_path / ".seq").read_text().strip() == "2"


def test_next_seq_skips_past_existing_folders(tmp_path):
    # .seq 不見了也不能跟舊資料夾撞號
    (tmp_path / "0041_20261028-101010_cam").mkdir()
    assert next_seq(tmp_path) == 42


def test_run_lifecycle(tmp_path):
    run = Run("cam", note="x", base=tmp_path)
    meta = read_meta(run.path)
    assert run.path.parent == tmp_path
    assert run.path.name == run.run_id
    assert meta["status"] == "running"
    assert meta["note"] == "x"
    assert meta["t0_ns"] == run.t0_ns
    assert meta["clock"] in ("CLOCK_MONOTONIC", "perf_counter")
    assert meta["wall_start"]
    assert len(meta["code"]["short"]) == 10
    assert "numpy" in meta["versions"]
    run.close(summary={"fps": 30})
    meta = read_meta(run.path)
    assert meta["status"] == "done"
    assert meta["summary"]["fps"] == 30
    assert meta["wall_end"] and meta["t_end_ns"] >= meta["t0_ns"]


def test_update_merges_into_meta(tmp_path):
    run = Run("cam", base=tmp_path)
    run.update(camera={"width": 640})
    assert read_meta(run.path)["camera"]["width"] == 640
    run.close()


def test_csv_writer_blank_for_none_and_nan(tmp_path):
    p = tmp_path / "a.csv"
    w = CsvWriter(p, ["a", "b", "c", "d"])
    w.row(1, None, float("nan"), 0.1 + 0.2)
    w.close()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines == ["a,b,c,d", "1,,,0.3"]


def test_events_go_to_per_source_files(tmp_path):
    run = Run("cam", base=tmp_path)
    run.event("mark", "我推了")
    run.event("warn", "x, y", src="vision")
    run.close()
    with open(run.path / "events_tool.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["kind"] == "mark" and rows[0]["detail"] == "我推了" and rows[0]["src"] == "tool"
    assert int(rows[0]["t_ns"]) >= run.t0_ns
    with open(run.path / "events_vision.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["detail"] == "x, y"


def test_crash_marks_status(tmp_path):
    with pytest.raises(RuntimeError):
        with Run("cam", base=tmp_path) as run:
            raise RuntimeError("boom")
    assert read_meta(run.path)["status"] == "crashed"
    with pytest.raises(KeyboardInterrupt):
        with Run("cam", base=tmp_path) as run2:
            raise KeyboardInterrupt
    assert read_meta(run2.path)["status"] == "stopped"


def test_exit_keeps_status_set_by_close(tmp_path):
    with Run("cam", base=tmp_path) as run:
        run.close(status="camera_lost")
    assert read_meta(run.path)["status"] == "camera_lost"


def test_default_base_by_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert runlog.default_base() == Path.home() / "pp" / "runs"
    monkeypatch.setattr(sys, "platform", "win32")
    assert runlog.default_base() == Path(runlog.__file__).resolve().parent.parent / "out" / "real"
