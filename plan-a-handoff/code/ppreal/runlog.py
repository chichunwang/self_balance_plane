"""一次測試＝一個資料夾：meta.json、各種 CSV、事件檔。格式見 docs/superpowers/specs/2026-10-27-測試紀錄與監控-design.md 第 4 節。"""
import csv
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

# Linux 的 monotonic 就是 CLOCK_MONOTONIC，跟相機、GPIO 的時間戳同一個鐘；Windows 的 monotonic 只有約 16 ms，改用 perf_counter
if sys.platform.startswith("linux"):
    now_ns = time.monotonic_ns
    CLOCK_NAME = "CLOCK_MONOTONIC"
else:
    now_ns = time.perf_counter_ns
    CLOCK_NAME = "perf_counter"


def default_base() -> Path:
    if sys.platform.startswith("linux"):
        return Path.home() / "pp" / "runs"
    return PROJECT / "out" / "real"


def next_seq(base: Path) -> int:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    f = base / ".seq"
    n = 0
    if f.exists():
        txt = f.read_text().strip()
        n = int(txt) if txt.isdigit() else 0
    for d in base.iterdir():   # .seq 不見了也不要跟舊資料夾撞號
        m = re.match(r"(\d{4,})_", d.name)
        if m and d.is_dir():
            n = max(n, int(m.group(1)))
    n += 1
    f.write_text(str(n))
    return n


def run_name(seq: int, kind: str, when: datetime) -> str:
    return f"{seq:04d}_{when:%Y%m%d-%H%M%S}_{kind}"


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if math.isnan(v) else format(float(v), ".9g")
    return str(v)


class CsvWriter:
    """一列一列寫，每 200 列或 0.5 秒 flush 一次。"""

    def __init__(self, path: Path, columns: list[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self._f = open(self.path, "w", encoding="utf-8", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.columns)
        self._pending = 0
        self._last = time.monotonic()

    def row(self, *values):
        self._w.writerow([_cell(v) for v in values])
        self._pending += 1
        if self._pending >= 200 or time.monotonic() - self._last > 0.5:
            self.flush()

    def flush(self):
        if not self._f.closed:
            self._f.flush()
        self._pending = 0
        self._last = time.monotonic()

    def close(self):
        if not self._f.closed:
            self._f.close()


def code_hash(root: Path = PROJECT, dirs=("ppsim", "ppreal")) -> dict:
    files, whole = {}, hashlib.sha256()
    for d in dirs:
        for p in sorted((Path(root) / d).rglob("*.py")):
            data = p.read_bytes()
            rel = p.relative_to(root).as_posix()
            files[rel] = hashlib.sha256(data).hexdigest()[:12]
            whole.update(rel.encode() + data)
    return {"files": files, "short": whole.hexdigest()[:10]}


def versions() -> dict:
    v = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import cv2
        v["cv2"] = cv2.__version__
        v["cv2_file"] = cv2.__file__
    except ImportError:
        v["cv2"] = None
    return v


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o)} 不能存成 JSON")


def write_json(path: Path, data: dict):
    # 先寫暫存檔再換名字，寫到一半當掉也不會留下空檔
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=_jsonable)
    try:
        os.replace(tmp, path)
    except PermissionError:      # Windows 上檔案被別的程式開著
        path.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
        tmp.unlink()


def read_meta(run_dir: Path) -> dict:
    with open(Path(run_dir) / "meta.json", encoding="utf-8") as f:
        return json.load(f)


class Run:
    def __init__(self, kind: str, note: str = "", base: Path | None = None):
        base = Path(base) if base else default_base()
        seq = next_seq(base)
        start = datetime.now()
        self.run_id = run_name(seq, kind, start)
        self.path = base / self.run_id
        self.path.mkdir(parents=True)
        self.t0_ns = now_ns()
        self.closed = False
        self._writers: dict[str, CsvWriter] = {}
        self.meta = {
            "run_id": self.run_id, "kind": kind, "note": note, "result": "", "status": "running",
            "wall_start": start.isoformat(timespec="seconds"), "t0_ns": self.t0_ns, "clock": CLOCK_NAME,
            "host": socket.gethostname(), "code": code_hash(), "versions": versions(),
        }
        self._save()

    def _save(self):
        write_json(self.path / "meta.json", self.meta)

    def csv(self, name: str, columns: list[str]) -> CsvWriter:
        if name not in self._writers:
            self._writers[name] = CsvWriter(self.path / name, columns)
        return self._writers[name]

    def event(self, kind: str, detail: str = "", src: str = "tool", t_ns: int | None = None):
        w = self.csv(f"events_{src}.csv", ["t_ns", "src", "kind", "detail"])
        w.row(now_ns() if t_ns is None else t_ns, src, kind, detail)
        w.flush()

    def update(self, **fields):
        self.meta.update(fields)
        self._save()

    def close(self, status: str = "done", summary: dict | None = None, result: str = ""):
        if summary is not None:
            self.meta["summary"] = summary
        if result:
            self.meta["result"] = result
        self.meta["status"] = status
        if not self.closed:
            self.meta["wall_end"] = datetime.now().isoformat(timespec="seconds")
            self.meta["t_end_ns"] = now_ns()
            for w in self._writers.values():
                w.close()
            self.closed = True
        self._save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.closed:
            if exc_type is None:
                self.close("done")
            elif issubclass(exc_type, KeyboardInterrupt):
                self.close("stopped")
            else:
                self.close("crashed")
        return False
