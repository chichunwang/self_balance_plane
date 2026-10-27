"""讀一次測試的資料夾：成績（summary.md）、時間曲線（timeline.png）、疊圖影片（overlay.mp4）。"""
import csv
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .balldetect import Ball
from .camtool import draw_view, frame_stats
from .runlog import read_meta

NAN = float("nan")
_TIME_COLS = ("t_cap_ns", "t_read_ns", "t_done_ns")


def load_camera(run_dir: Path) -> dict[str, np.ndarray]:
    """camera.csv → 每欄一個陣列。時間換成「開始後幾秒」（先用整數減掉 t0，才不會掉精度），空欄是 NaN。"""
    run_dir = Path(run_dir)
    t0 = int(read_meta(run_dir)["t0_ns"])
    with open(run_dir / "camera.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    out: dict[str, np.ndarray] = {}
    for col in cols:   # 一張都沒有時也回每欄的空陣列；寫到一半斷電的最後一列，缺的欄位當空的
        if col in _TIME_COLS:
            out[col[:-3] + "_s"] = np.array([(int(r[col]) - t0) / 1e9 if r.get(col) else NAN for r in rows], float)
        else:
            out[col] = np.array([float(r[col]) if r.get(col) else NAN for r in rows], float)
    if "frame" in out:
        out["frame"] = np.nan_to_num(out["frame"], nan=-1).astype(int)
    return out


def iter_mjpeg(run_dir: Path):
    """照 camera.csv 記的位置一張一張讀；檔尾不完整（程式當掉）的那張略過。"""
    run_dir = Path(run_dir)
    d = load_camera(run_dir)
    path = run_dir / "cam.mjpeg"
    size = path.stat().st_size
    with open(path, "rb") as f:
        for k, off, ln in zip(d.get("frame", []), d.get("jpg_off", []), d.get("jpg_len", [])):
            if math.isnan(off) or math.isnan(ln) or off + ln > size:
                continue
            f.seek(int(off))
            buf = np.frombuffer(f.read(int(ln)), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                yield int(k), img


def make_overlay(run_dir: Path, out: Path, fps: float = 30.0) -> Path | None:
    run_dir = Path(run_dir)
    d = load_camera(run_dir)
    idx = {int(k): i for i, k in enumerate(d["frame"])}
    writer, proc = None, None
    exe = shutil.which("ffmpeg")
    for k, gray in iter_mjpeg(run_dir):
        i = idx[k]
        found = d["found"][i] == 1
        ball = Ball(True, d["u_px"][i], d["v_px"][i], d["r_px"][i]) if found else None
        t = d["t_read_s"][i]
        lines = [f"t {t:6.2f} s   frame {k}"]
        if found and not math.isnan(d["x_mm"][i]):
            lines.append(f"ball x {d['x_mm'][i]:+6.1f} mm  y {d['y_mm'][i]:+6.1f} mm")
        elif not found:
            lines.append("no ball")
        img = draw_view(gray, ball, lines, grid=False)
        h, w = img.shape[:2]
        if writer is None and proc is None:
            if exe:   # 有 ffmpeg 就壓 H.264，Windows 內建播放器也播得出來
                proc = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                         "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264",
                                         "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast", str(out)],
                                        stdin=subprocess.PIPE)
            else:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if proc is not None:
            proc.stdin.write(img.tobytes())
        else:
            writer.write(img)
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            return None
    elif writer is not None:
        writer.release()
    else:
        return None
    return out


def _timeline(d: dict, out: Path, title: str, nominal_fps: float = 30.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    t = d["t_cap_s"] if not np.isnan(d["t_cap_s"]).all() else d["t_read_s"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    dt = np.diff(t) * 1e3
    ax[0].plot(t[1:], dt, lw=0.8)
    ax[0].set_ylim(0, max(2e3 / nominal_fps, float(np.nanmax(dt)) * 1.1 if dt.size else 0))
    ax[0].axhline(1e3 / nominal_fps, color="gray", lw=0.6, ls="--")
    ax[0].set_ylabel("間隔（ms）")
    ax[0].set_title(title)
    if not np.isnan(d["x_mm"]).all():
        ax[1].plot(t, d["x_mm"], label="x（右）")
        ax[1].plot(t, d["y_mm"], label="y（前）")
        ax[1].set_ylabel("球的位置（mm）")
    else:
        ax[1].plot(t, d["u_px"], label="u")
        ax[1].plot(t, d["v_px"], label="v")
        ax[1].set_ylabel("球在畫面（像素）")
    ax[1].legend(loc="upper right")
    ax[2].plot(t, np.nan_to_num(d["found"]), drawstyle="steps-post", lw=0.8, label="找到球")
    ax[2].plot(t, np.nan_to_num(d["rec"]), drawstyle="steps-post", lw=0.8, alpha=0.7, label="有存圖")
    ax[2].set_ylim(-0.1, 1.1)
    ax[2].set_xlabel("開始後幾秒")
    ax[2].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _events(run_dir: Path, t0: int) -> list[str]:
    rows = []
    for p in sorted(Path(run_dir).glob("events_*.csv")):
        with open(p, encoding="utf-8", newline="") as f:
            rows += list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["t_ns"]))
    return [f"  - {(int(r['t_ns']) - t0) / 1e9:7.2f} s　{r['src']}／{r['kind']}　{r['detail']}" for r in rows]


def report(run_dir: Path, video: bool = True) -> Path:
    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    d = load_camera(run_dir)
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    cam = meta.get("camera", {})
    nominal = float(cam.get("requested", {}).get("fps") or cam.get("fps_reported") or 30.0)
    t = d["t_cap_s"] if not np.isnan(d["t_cap_s"]).all() else d["t_read_s"]
    fs = frame_stats(t * 1e9, nominal)
    n = len(d["frame"])
    found = int(np.nansum(d["found"] == 1))
    looked = int(np.sum(~np.isnan(d["found"])))
    stored = int(np.nansum(d["rec"] == 1))
    s = meta.get("summary", {})
    lines = [f"# {meta['run_id']}", "",
             f"- 種類：{meta['kind']}　狀態：{meta['status']}" + (f"　備註：{meta['note']}" if meta.get("note") else "")
             + (f"　結果：{meta['result']}" if meta.get("result") else ""),
             f"- 開始：{meta['wall_start']}　長度：{fs.get('duration_s', 0):.1f} 秒　時鐘：{meta['clock']}"]
    if cam:
        lines.append(f"- 相機：{cam.get('name')}（{cam.get('backend')}）{cam.get('width')}×{cam.get('height')} "
                     f"{cam.get('fourcc')}")
    lines.append(f"- 張數：{n}（存了圖 {stored}、跳過 {n - stored}、讀取失敗 {s.get('read_fail', 0)}）")
    if fs["n"] >= 2:
        lines.append(f"- 每秒張數：{fs['fps']:.2f}（間隔平均 {fs['dt_mean_ms']:.1f} ms、中位 {fs['dt_p50_ms']:.1f}、"
                     f"99% {fs['dt_p99_ms']:.1f}、最長 {fs['dt_max_ms']:.1f}；超過 {1.5e3 / nominal:.0f} ms 的間隔 "
                     f"{fs['gaps']} 次、估計掉 {fs['est_dropped']} 張）")
    if looked:
        lines.append(f"- 找到球：{found}/{looked} 張（{100 * found / looked:.0f}%）")
        if found:
            lines.append(f"- 球在畫面：u {np.nanmin(d['u_px']):.0f}–{np.nanmax(d['u_px']):.0f}、"
                         f"v {np.nanmin(d['v_px']):.0f}–{np.nanmax(d['v_px']):.0f} 像素")
        if not np.isnan(d["x_mm"]).all():
            r = np.hypot(d["x_mm"], d["y_mm"])
            lines.append(f"- 球離中心：平均 {np.nanmean(r):.1f} mm、最遠 {np.nanmax(r):.1f} mm")
    ev = _events(run_dir, int(meta["t0_ns"]))
    if ev:
        lines += ["- 事件："] + ev
    lines.append("")
    if n:
        _timeline(d, out / "timeline.png", meta["run_id"], nominal)
        lines.append("![時間曲線](timeline.png)")
    if video and n:
        mp4 = make_overlay(run_dir, out / "overlay.mp4", fps=fs.get("fps") or nominal)
        lines.append("" if mp4 else "（疊圖影片沒做出來：沒有 ffmpeg 也沒有可用的 mp4 編碼器）")
    md = out / "summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def cmd_report(args) -> int:
    md = report(Path(args.run_dir), video=not args.no_video)
    print(md.read_text(encoding="utf-8"))
    print(f"報告在 {md.parent}")
    return 0
