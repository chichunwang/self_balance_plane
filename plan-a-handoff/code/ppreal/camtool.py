"""相機工具：錄影管線（record）和 cam list／check／rec／fov／calib 指令。

錄影時主執行緒讀相機、找球；另一條執行緒壓 JPEG、寫檔。寫檔跟不上就跳過那張不存（只記一列），絕不讓找球等錄影。
"""
import math
import os
import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from . import calib
from .balldetect import Ball, DetectParams, detect_ball
from .calib import PLATFORM_D_MM, BALL_D_MM, CamMap, fit_circle, fov_report, map_from_points
from .camera import Camera, CamConfig, CameraNotFound, default_device, list_devices
from .runlog import Run, now_ns

CAMERA_COLUMNS = ["frame", "t_cap_ns", "t_read_ns", "t_done_ns", "found", "u_px", "v_px", "r_px",
                  "x_mm", "y_mm", "area_px", "rec", "jpg_off", "jpg_len"]
_DEFAULT = object()


def encode_jpeg(gray: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes()


class _Recorder(threading.Thread):
    """收（一列, 那張 Frame 或 None）：有 Frame 就壓縮寫進 cam.mjpeg，沒有就只記一列（rec＝0）。
    存圖出錯（磁碟滿、檔案被鎖）就記下錯誤，之後只記 csv、不再存圖。"""

    def __init__(self, run: Run, quality: int):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue()
        self.quality = quality
        self.csv = run.csv("camera.csv", CAMERA_COLUMNS)
        self.f = open(run.path / "cam.mjpeg", "wb")
        self.off = 0
        self.recorded = 0
        self.failed = 0
        self.error: str | None = None

    def _row(self, row, *tail):
        try:
            self.csv.row(*row, *tail)
        except Exception as e:      # 連 csv 都寫不進去：記下來，佇列照樣清空，主執行緒才不會卡住
            self.error = self.error or repr(e)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            row, fr = item
            if fr is None:
                self._row(row, 0, None, None)
                continue
            if self.error is not None:
                self.failed += 1
                self._row(row, 0, None, None)
                continue
            try:
                data = fr.jpeg if fr.jpeg is not None else encode_jpeg(fr.gray, self.quality)
                self.f.write(data)
            except Exception as e:
                self.error = repr(e)
                self.failed += 1
                self._row(row, 0, None, None)
                continue
            self._row(row, 1, self.off, len(data))
            self.off += len(data)
            self.recorded += 1
        try:
            self.f.close()
            self.csv.flush()
        except Exception as e:
            self.error = self.error or repr(e)


def draw_view(gray, ball: Ball | None = None, lines=(), grid: bool = True, roi=None, points=()):
    """預覽和疊圖影片共用。cv2.putText 只會畫英文，所以畫面上的字都是英文。"""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape[:2]
    if grid:   # 正中十字＋四邊每 10% 一個刻度，量視角時對捲尺用
        cv2.line(img, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), (0, 200, 255), 1)
        cv2.line(img, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), (0, 200, 255), 1)
        for i in range(1, 10):
            x, y = w * i // 10, h * i // 10
            cv2.line(img, (x, 0), (x, 8), (0, 200, 255), 1)
            cv2.line(img, (x, h - 9), (x, h - 1), (0, 200, 255), 1)
            cv2.line(img, (0, y), (8, y), (0, 200, 255), 1)
            cv2.line(img, (w - 9, y), (w - 1, y), (0, 200, 255), 1)
    if roi is not None:
        cv2.circle(img, (round(roi[0]), round(roi[1])), round(roi[2]), (255, 160, 0), 1)
    for p in points:
        cv2.circle(img, (round(p[0]), round(p[1])), 4, (0, 0, 255), -1)
    if ball is not None and ball.found:
        c = (round(ball.u), round(ball.v))
        cv2.circle(img, c, max(2, round(ball.r)), (0, 255, 0), 2)
        cv2.drawMarker(img, c, (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def preview_lines(fps: float, elapsed_s: float, frames: int, ball: Ball | None, xy) -> list[str]:
    """預覽畫面左上角的字（只能英文）。xy：換好的平台座標 mm，沒校正過是 None。"""
    lines = [f"{fps:5.1f} fps  {elapsed_s:5.1f} s  frame {frames}"]
    if ball is None:
        lines.append("detect off")
    elif not ball.found:
        lines.append("no ball")
    elif xy is not None:
        lines.append(f"ball x {xy[0]:+.1f} mm  y {xy[1]:+.1f} mm")
    else:
        lines.append(f"ball u {ball.u:.1f}  v {ball.v:.1f}  r {ball.r:.1f} px")
    lines.append("q: stop")
    return lines


def _ui_quit(window: str, shown: bool) -> bool:
    """處理視窗事件；按 q／Esc 或視窗被關掉回 True。視窗還沒顯示過就不檢查（不然一開始就會判成被關掉）。"""
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        return True
    return shown and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1


def record(cam, run: Run, seconds: float | None = None, max_frames: int | None = None, detect=_DEFAULT,
           cam_map: CamMap | None = None, quality: int = 90, view: bool = False, max_queue: int = 60,
           fail_limit: int = 30, stall_s: float = 3.0, calib_snapshot: dict | None = None,
           window: str = "ppreal", platform_roi=None) -> dict:
    """錄到 seconds 秒（照相機給的時間）或 max_frames 張。結束時自己把 run 收尾（meta 的 status、summary）；
    程式出錯時 status 記 crashed，例外照樣往外丟。"""
    if detect is _DEFAULT:
        detect = DetectParams()
    if platform_roi is None and cam_map is not None:
        platform_roi = (cam_map.u0, cam_map.v0, PLATFORM_D_MM / 2 / cam_map.mm_per_px)
    run.update(camera=cam.settings)
    if calib_snapshot is not None:
        run.update(calib=calib_snapshot)
    rec = _Recorder(run, quality)
    rec.start()
    st = {"frames": 0, "dropped": 0, "found": 0, "read_fail": 0, "status": "done"}
    ts: list[int] = []
    fails, t_first, shown, err = 0, None, False, None
    last_ok = now_ns()
    try:
        while max_frames is None or st["frames"] < max_frames:
            fr = cam.read()
            t_got = now_ns()
            if fr is None:
                fails += 1
                st["read_fail"] += 1
                idle = (t_got - last_ok) / 1e9
                if fails >= fail_limit or idle > stall_s:   # DSHOW 讀不到時每次要等，不能只數次數
                    st["status"] = "camera_lost"
                    run.event("error", f"相機 {idle:.1f} 秒沒給畫面（連續 {fails} 次讀不到）")
                    break
                if view and _ui_quit(window, shown):
                    st["status"] = "stopped"
                    break
                continue
            fails, last_ok = 0, t_got
            if t_first is None:
                t_first = fr.t_read_ns
            if seconds is not None and fr.t_read_ns - t_first >= seconds * 1e9:
                break
            ball = detect_ball(fr.gray, detect) if detect is not None else None
            t_done = fr.t_read_ns + (now_ns() - t_got)
            x = y = None
            if ball is not None and ball.found:
                st["found"] += 1
                if cam_map is not None:
                    x, y = cam_map.to_mm(ball.u, ball.v)
            row = [fr.index, fr.t_cap_ns, fr.t_read_ns, t_done,
                   None if ball is None else int(ball.found),
                   None if ball is None else ball.u, None if ball is None else ball.v,
                   None if ball is None else ball.r, x, y, None if ball is None else ball.area]
            keep = rec.q.qsize() < max_queue
            if not keep:
                st["dropped"] += 1
            rec.q.put((row, fr if keep else None))
            st["frames"] += 1
            ts.append(fr.t_cap_ns if fr.t_cap_ns is not None else fr.t_read_ns)
            if view:
                lines = preview_lines(_recent_fps(ts), (fr.t_read_ns - t_first) / 1e9, st["frames"], ball,
                                      (x, y) if x is not None else None)
                cv2.imshow(window, draw_view(fr.gray, ball, lines, roi=platform_roi))
                shown = True
                if _ui_quit(window, shown):
                    st["status"] = "stopped"
                    break
    except KeyboardInterrupt:
        st["status"] = "stopped"
    except BaseException as e:
        st["status"] = "crashed"
        err = e
    finally:
        rec.q.put(None)
        rec.join()
        if view:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    st["recorded"] = rec.recorded
    st["rec_failed"] = rec.failed
    if rec.error:
        run.event("error", f"錄影寫檔出錯，之後只記 csv、沒存圖：{rec.error}")
        if st["status"] == "done":
            st["status"] = "rec_error"
    if err is not None:
        run.event("error", f"程式出錯：{err!r}")
    if len(ts) > 1 and ts[-1] > ts[0]:
        st["duration_s"] = (ts[-1] - ts[0]) / 1e9
        st["fps"] = (len(ts) - 1) / st["duration_s"]
    run.close(status=st["status"], summary=dict(st))
    if err is not None:
        raise err
    st["t_ns"] = ts
    return st


def _recent_fps(ts: list[int]) -> float:
    recent = ts[-31:]
    if len(recent) > 1 and recent[-1] > recent[0]:
        return (len(recent) - 1) / ((recent[-1] - recent[0]) / 1e9)
    return 0.0


def frame_stats(t_ns, nominal_fps: float) -> dict:
    t = np.asarray(t_ns, dtype=np.float64) / 1e9
    n = len(t)
    out = {"n": n, "nominal_fps": nominal_fps}
    if n < 2:
        return out | {"duration_s": 0.0, "fps": 0.0, "gaps": 0, "est_dropped": 0}
    dt = np.diff(t)
    big = dt[dt > 1.5 / nominal_fps]
    dur = float(t[-1] - t[0])
    return out | {
        "duration_s": dur, "fps": (n - 1) / dur if dur > 0 else 0.0,
        "dt_mean_ms": float(dt.mean() * 1e3), "dt_p50_ms": float(np.median(dt) * 1e3),
        "dt_p99_ms": float(np.percentile(dt, 99) * 1e3), "dt_max_ms": float(dt.max() * 1e3),
        "gaps": int(big.size), "est_dropped": int(sum(round(d * nominal_fps) - 1 for d in big)),
    }


def check_criteria(stats: dict, nominal_fps: float) -> list[tuple[str, bool, str]]:
    need = nominal_fps - 1
    return [
        ("每秒張數", stats.get("fps", 0.0) >= need, f"{stats.get('fps', 0.0):.2f} 張/秒（要 ≥ {need:g}）"),
        ("沒有掉張", stats.get("gaps", 0) == 0,
         f"間隔超過 {1.5e3 / nominal_fps:.0f} ms 的有 {stats.get('gaps', 0)} 次，估計掉 {stats.get('est_dropped', 0)} 張"),
    ]


def find_platform_circle(gray: np.ndarray, r_min_frac: float = 0.15, r_max_frac: float = 0.5):
    """在畫面裡找平台的圓邊，回 (u, v, r) 像素或 None。"""
    h, w = gray.shape[:2]
    s = min(h, w)
    img = cv2.GaussianBlur(gray, (9, 9), 2)
    c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.5, minDist=s, param1=80, param2=40,
                         minRadius=int(r_min_frac * s), maxRadius=int(r_max_frac * s))
    if c is None:
        return None
    u, v, r = c[0][0]
    return float(u), float(v), float(r)


# ---------------- 指令 ----------------

def _has_display() -> bool:
    if not sys.platform.startswith("linux"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open(args) -> Camera | None:
    cfg = CamConfig(device=args.device or "", width=args.width, height=args.height, fps=args.fps,
                    fourcc=args.fourcc, exposure=args.exposure, allow_any=getattr(args, "any_camera", False))
    try:
        return Camera(cfg)
    except CameraNotFound as e:
        print(e)
        return None


def camera_geometry(cam: dict):
    """Keep the calibrated plate outline separate from the padded detection mask."""
    if "map" not in cam or "platform_r_px" not in cam:
        return None, DetectParams(), None
    m = CamMap.from_dict(cam["map"])
    r_ball = (BALL_D_MM / 2) / m.mm_per_px
    plate = (m.u0, m.v0, float(cam["platform_r_px"]))
    det = DetectParams(roi=(m.u0, m.v0, plate[2] + r_ball), r_expect=r_ball)
    return m, det, plate


def _saved_map():
    m, det, _ = camera_geometry(calib.load().get("camera", {}))
    return m, det


def _print_settings(s: dict, nominal_fps: float):
    req = s["requested"]
    print(f"相機：{s['name']}（{s['backend']}），讀回 {s['width']}×{s['height']} {s['fourcc'] or '?'}，"
          f"驅動說每秒 {s['fps_reported']:g} 張（只是要求值，實際看下面）")
    if (s["width"], s["height"]) != (req["width"], req["height"]):
        print(f"  ⚠ 要求 {req['width']}×{req['height']}，相機給的是 {s['width']}×{s['height']}")


def cmd_list(args) -> int:
    names = list_devices()
    if not names:
        print("一顆相機都沒看到（Windows 要有 ffmpeg 才列得出名字）。")
        return 1
    want = default_device().lower()
    for i, n in enumerate(names):
        print(f"{i}: {n}" + ("    ← 預設用這顆" if want in n.lower() else ""))
    return 0


def _run_camera(args, kind: str, seconds: float, check: bool) -> int:
    cam = _open(args)
    if cam is None:
        return 2
    if getattr(args, "settings", False):
        cam.open_settings_dialog()
    _print_settings(cam.settings, args.fps)
    snapshot = calib.load()
    cam_map, det, platform_roi = camera_geometry(snapshot.get("camera", {}))
    if getattr(args, "no_detect", False):
        det = None
    view = _has_display() and not args.no_view
    print(f"錄 {seconds:g} 秒" + ("，預覽視窗按 q 可以提早停" if view else "") + " …")
    with Run(kind, note=args.note or "") as run:
        try:
            st = record(cam, run, seconds=seconds, detect=det, cam_map=cam_map, view=view,
                        quality=getattr(args, "quality", 90), calib_snapshot=snapshot,
                        platform_roi=platform_roi)
        finally:
            cam.close()
    if st["frames"] == 0:
        print("一張都沒讀到：相機是不是被 Windows 的「相機」App、Teams 或瀏覽器的視訊開著？關掉再試。")
        print(f"紀錄在 {run.path}")
        return 1
    fs = frame_stats(st["t_ns"], args.fps)
    ok_all = True
    if fs["n"] >= 2:
        print(f"實際：{fs['n']} 張、每秒 {fs['fps']:.2f} 張；間隔平均 {fs['dt_mean_ms']:.1f} ms、"
              f"99% 在 {fs['dt_p99_ms']:.1f} ms 內、最長 {fs['dt_max_ms']:.1f} ms")
    if check:
        crit = check_criteria(fs, args.fps)
        for name, ok, msg in crit:
            print(f"  {'✔' if ok else '✘'} {name}：{msg}")
        ok_all = all(ok for _, ok, _ in crit)
        run.update(check={"frame_stats": fs, "criteria": [{"item": n, "ok": o, "detail": m} for n, o, m in crit]})
    else:
        run.update(frame_stats=fs)
    if det is not None:
        print(f"找到球：{st['found']}/{st['frames']} 張" + ("（畫面裡沒放球的話是 0，正常）" if st["found"] == 0 else ""))
    if st["dropped"]:
        print(f"  ⚠ 錄影來不及，跳過 {st['dropped']} 張沒存圖（csv 還是有記）")
    if st.get("rec_failed"):
        print(f"  ⚠ 存圖出錯，有 {st['rec_failed']} 張沒存到（磁碟滿了或檔案被別的程式鎖住？）")
    status_text = {"done": "完成", "stopped": "提早停了", "camera_lost": "相機斷掉了",
                   "rec_error": "錄完了，但存圖中途出錯"}.get(st["status"], st["status"])
    print(f"{status_text}。紀錄在 {run.path}")
    print(f"看報告：.venv/Scripts/python -m ppreal report \"{run.path}\"")
    return 0 if ok_all and st["status"] != "camera_lost" else 1


def cmd_check(args) -> int:
    return _run_camera(args, "camcheck", args.seconds, check=True)


def cmd_rec(args) -> int:
    return _run_camera(args, "cam", args.seconds, check=False)


def cmd_view(args) -> int:
    """只看畫面、不存檔：量視角、架相機、看找球準不準時用。"""
    if not _has_display():
        print("cam view 要看畫面：在筆電上做，或樹莓派接螢幕。")
        return 2
    cam = _open(args)
    if cam is None:
        return 2
    cam_map, det, platform_roi = camera_geometry(calib.load().get("camera", {}))
    win, shown, ts, n = "ppreal view", False, [], 0
    last_ok = now_ns()
    print("只看不存檔。按 q 或關掉視窗結束。")
    try:
        while True:
            fr = cam.read()
            if fr is None:
                if (now_ns() - last_ok) / 1e9 > 3.0:
                    print("相機 3 秒沒給畫面，結束。被別的程式開著的話先關掉。")
                    return 1
                if _ui_quit(win, shown):
                    break
                continue
            last_ok = now_ns()
            n += 1
            ts.append(fr.t_read_ns)
            b = detect_ball(fr.gray, det)
            xy = cam_map.to_mm(b.u, b.v) if (cam_map is not None and b.found) else None
            lines = preview_lines(_recent_fps(ts), (ts[-1] - ts[0]) / 1e9, n, b, xy)
            cv2.imshow(win, draw_view(fr.gray, b, lines, roi=platform_roi))
            shown = True
            if _ui_quit(win, shown):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
    return 0


def cmd_fov(args) -> int:
    r = fov_report(args.d, args.w, args.h, mount_cm=args.mount)
    guess = "（高是照 4:3 推的）" if args.h is None else ""
    print(f"離牆 {r['d_cm']:g} cm 時，畫面在牆上寬 {r['w_cm']:g} cm、高 {r['h_cm']:g} cm{guess}")
    print(f"水平視角約 {r['hfov_deg']:.0f}°、垂直視角約 {r['vfov_deg']:.0f}°（捲尺讀差 1 cm 大約差 1°）")
    print(f"鏡頭到平台面至少 {r['min_height_cm']:.1f} cm，畫面短邊才涵蓋得到 25 cm；實際再加 5–10 cm。")
    if args.mount:
        print(f"架在 {args.mount:g} cm：每像素 {r['mm_per_px']:.2f} mm，平台約 {r['platform_px']:.0f} 像素寬、"
              f"球約 {r['ball_px']:.0f} 像素")
    return 0


def _avg_ball(cam, det: DetectParams, n: int = 10, max_reads: int = 30):
    """最多讀 max_reads 張（約 1 秒），找到 n 張以上才取平均，不然回 None（不會一直等）。"""
    us, vs = [], []
    for _ in range(max_reads):
        fr = cam.read()
        if fr is None:
            continue
        b = detect_ball(fr.gray, det)
        if b.found:
            us.append(b.u)
            vs.append(b.v)
    if len(us) < n:
        return None
    return float(np.mean(us)), float(np.mean(vs))


def cmd_calib(args) -> int:
    cam = _open(args)
    if cam is None:
        return 2
    try:
        if args.check:
            return _calib_check(cam)
        if not _has_display():
            print("cam calib 要看畫面跟點滑鼠：在筆電上做，或樹莓派接螢幕。每天開工的檢查用 cam calib --check。")
            return 2
        return _calib_interactive(cam)
    finally:
        cam.close()


def _calib_check(cam) -> int:
    saved = calib.load().get("camera", {})
    fr = cam.read()
    if fr is None:
        print("讀不到相機畫面。")
        return 2
    found = find_platform_circle(fr.gray)
    if found is None:
        print("畫面裡找不到平台的圓邊：燈光、背景可能不一樣了，改用 cam calib 手動點。")
        return 1
    if "map" not in saved:
        print(f"找到平台：圓心 ({found[0]:.1f}, {found[1]:.1f})、半徑 {found[2]:.1f} 像素；校正檔還沒有相機資料，先做一次 cam calib。")
        return 1
    m = CamMap.from_dict(saved["map"])
    du, dv, dr = found[0] - m.u0, found[1] - m.v0, found[2] - saved["platform_r_px"]
    print(f"跟上次比：圓心移了 ({du:+.1f}, {dv:+.1f}) 像素、半徑差 {dr:+.1f} 像素")
    if max(abs(du), abs(dv), abs(dr)) <= 3:
        print("相機沒動，照舊。")
        return 0
    m.u0, m.v0, m.mm_per_px = found[0], found[1], (PLATFORM_D_MM / 2) / found[2]
    calib.update("camera", {"map": m.to_dict(), "platform_r_px": found[2]}, "cam calib --check：相機被碰歪，重算平台位置")
    print("相機被碰歪了，已經照新的位置重算（方向沿用上次的）。")
    return 0


def _calib_interactive(cam) -> int:
    win = "ppreal calib"
    clicks: list[tuple[int, int]] = []
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    print("第 1 步：平台停在水平、先不要放球。按 c 讓程式自己找平台的圓邊；"
          "找得不準就用滑鼠在平台邊緣點 3 下以上（點錯按 r 重來），對了按 Enter。q 離開。")
    circle, last = None, None
    while True:
        fr = cam.read()
        if fr is None:
            print("讀不到相機畫面。")
            return 2
        last = fr.gray
        if len(clicks) >= 3:
            circle = fit_circle(np.array(clicks, float))
        img = draw_view(fr.gray, None, ["c: auto  click edge x3  r: reset  Enter: ok  q: quit"],
                        roi=circle, points=clicks)
        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return 1
        if k == ord("r"):
            clicks.clear()
            circle = None
        if k == ord("c"):
            circle = find_platform_circle(fr.gray)
            if circle is None:
                print("  找不到，改用滑鼠點。")
        if k in (13, 10) and circle is not None:
            break
    u0, v0, rpx = circle
    mm_per_px = (PLATFORM_D_MM / 2) / rpx
    det = DetectParams(roi=(u0, v0, rpx + (BALL_D_MM / 2) / mm_per_px))
    pos = {}
    for key, msg in (("center", "把球放在平台中心"), ("front", "把球放在正面（1 號那側）邊緣，離邊大約一顆球"),
                     ("right", "把球放在正右邊：從中心往右、跟正面成直角（1 號和 3 號之間、靠 3 號那邊），離邊大約一顆球")):
        print(f"{msg}，放好、球停了按 Enter（q 離開）。")
        while True:
            fr = cam.read()
            if fr is None:
                return 2
            b = detect_ball(fr.gray, det)
            cv2.imshow(win, draw_view(fr.gray, b, [f"{key}: Enter when the ball is still   q: quit"], roi=(u0, v0, rpx)))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                return 1
            if k in (13, 10):
                p = _avg_ball(cam, det)
                if p is None:
                    print("  這一秒裡沒穩定看到球（畫面上要有綠圈），手拿開、球放進藍圈裡再按一次 Enter。")
                    continue
                pos[key] = p
                last = fr.gray
                break
    cv2.destroyWindow(win)
    try:
        m = map_from_points((u0, v0), pos["front"], pos["right"], mm_per_px)
    except ValueError as e:
        print(e)
        return 1
    cx, cy = m.to_mm(*pos["center"])
    print(f"平台圓心 ({u0:.1f}, {v0:.1f})、半徑 {rpx:.1f} 像素 → 每像素 {mm_per_px:.3f} mm")
    print(f"正面在畫面的 {m.front_deg:.0f}° 方向（90°＝畫面上方）" + ("；畫面是左右翻的，已經自動翻回來" if m.mirror else ""))
    print(f"球放中心時量到 ({cx:+.1f}, {cy:+.1f}) mm（手放的，差 1 cm 內都正常）")
    with Run("camcalib") as run:
        img = draw_view(last, None, grid=False, roi=(u0, v0, rpx))
        for (dx, dy), color in (((60, 0), (0, 0, 255)), ((0, 60), (0, 255, 0))):   # +x 紅、+y 綠
            ex = _mm_to_px(m, dx, dy)
            cv2.arrowedLine(img, (round(u0), round(v0)), (round(ex[0]), round(ex[1])), color, 2, tipLength=0.15)
        cv2.imwrite(str(run.path / "calib.png"), img)
        run.update(calib={"map": m.to_dict(), "platform_r_px": rpx, "balls_px": pos})
        print(f"檢查圖（紅＝右 +x、綠＝前 +y）：{run.path / 'calib.png'}")
    calib.update("camera", {"map": m.to_dict(), "platform_r_px": rpx, "width": cam.settings["width"],
                            "height": cam.settings["height"], "device": cam.settings["name"]}, "cam calib")
    print("已存進校正檔。")
    return 0


def _mm_to_px(m: CamMap, x: float, y: float):
    """CamMap.to_mm 的反函數，只用來畫檢查圖。"""
    th = math.radians(m.front_deg)
    c, s = math.cos(th), math.sin(th)
    mx = (x * s + y * c) / m.mm_per_px
    my = (-x * c + y * s) / m.mm_per_px
    if m.mirror:
        mx = -mx
    return m.u0 + mx, m.v0 - my
