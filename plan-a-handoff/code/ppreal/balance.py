"""Teensy 實機平衡：啟動、找球、控制分成三個行程。"""
from __future__ import annotations

from collections import deque
import ctypes
import math
import multiprocessing as mp
import json
from pathlib import Path
import queue
import shutil
import signal
import struct
import sys
import time

import numpy as np

from . import calib
from .hwtool import high_priority, servo_rows, positive
from .runlog import Run, CsvWriter, now_ns, write_json
from .teensy import (Teensy, ClockMap, STATUS, IMU, PONG, PING,
                     angles_to_pulses)

CONTROL_DTYPE = np.dtype([
    ("t_ns", "<u8"), ("seq", "<u4"), ("late_ns", "<i8"),
    ("cam_frame", "<i8"), ("cam_t_ns", "<i8"), ("xy_m", "<f8", (2,)),
    ("p_est_m", "<f8", (3,)), ("v_est_mps", "<f8", (3,)),
    ("pulses_us", "<f4", (3,)), ("mode", "u1"),
])
USB_DTYPE = np.dtype([("t_ns", "<u8"), ("kind", "u1"), ("seq", "<u4"), ("payload", "u1", (28,))])
IMU_DTYPE = np.dtype([("t_ns", "<u8"), ("micros", "<u4"), ("raw", "<i2", (7,)), ("errors", "<u4")])


class BallRecovery:
    """Keep sending commands while returning to center and waiting for a fresh ball."""
    def __init__(self, centers, start_ns):
        self.centers = np.asarray(centers, dtype=float)
        self.last_seen_ns = start_ns
        self.waiting = False
        self.return_start_ns = 0
        self.return_end_ns = 0
        self.from_pulses = self.centers.copy()

    def update(self, tick_ns, read_ns, shot_ns, detected, last_pulses):
        transition = None
        if not self.waiting:
            if detected:
                self.last_seen_ns = read_ns
            if tick_ns - self.last_seen_ns > 300000000:
                self.waiting = True
                self.return_start_ns = tick_ns
                self.return_end_ns = tick_ns + 1000000000
                self.from_pulses = np.asarray(last_pulses, dtype=float).copy()
                transition = "lost"
        if self.waiting:
            u = min(1., max(0., (tick_ns - self.return_start_ns) / 1e9))
            u = u * u * (3 - 2 * u)
            pulses = self.from_pulses + (self.centers - self.from_pulses) * u
            # Reacquire only from a frame captured after the return completed.
            if tick_ns >= self.return_end_ns and detected and shot_ns >= self.return_end_ns:
                self.waiting = False
                self.last_seen_ns = read_ns
                return False, None, "resumed"
            return True, pulses, transition
        return False, None, transition


class Ring:
    """Single producer, single consumer; the producer never waits for the logger."""
    def __init__(self, ctx, dtype, capacity):
        self.dtype, self.capacity = dtype, capacity
        self.data = ctx.RawArray(ctypes.c_ubyte, dtype.itemsize * capacity)
        self.read = ctx.RawValue(ctypes.c_uint64, 0)
        self.write = ctx.RawValue(ctypes.c_uint64, 0)
        self.dropped = ctx.RawValue(ctypes.c_uint64, 0)
        self.lock = ctx.Lock()

    def put(self, row):
        if not self.lock.acquire(False):
            self.dropped.value += 1
            return
        try:
            if self.write.value - self.read.value >= self.capacity:
                self.dropped.value += 1
                return
            a = np.frombuffer(self.data, dtype=self.dtype)
            a[self.write.value % self.capacity] = row
            self.write.value += 1
        finally:
            self.lock.release()

    def drain(self):
        with self.lock:
            a = np.frombuffer(self.data, dtype=self.dtype)
            n = self.write.value - self.read.value
            start = self.read.value % self.capacity
            tail = min(n, self.capacity - start)
            out = a[start:start + tail].copy()
            if tail < n:
                out = np.concatenate((out, a[:n - tail].copy()))
            self.read.value += n
        return out


class CameraSink:
    def __init__(self, path):
        self.path, self.writers = Path(path), {}

    def csv(self, name, columns):
        w = CsvWriter(self.path / name, columns)
        self.writers[name] = w
        return w

    def close(self):
        for w in self.writers.values():
            w.close()


def _vision(options, snapshot, path, shared, stop, events):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import cv2
    from .camera import Camera, CamConfig
    from .balldetect import detect_ball
    from .camtool import _Recorder, draw_view, camera_geometry
    cam, rec, sink = None, None, CameraSink(path)
    count, recorded_drops = 0, 0
    try:
        saved = snapshot["camera"]
        mapping, det, platform_roi = camera_geometry(saved)
        cam = Camera(CamConfig(device=options["device"], width=options["width"], height=options["height"],
                               fps=options["fps"], fourcc=options["fourcc"], exposure=options["exposure"]))
        if (cam.settings["width"], cam.settings["height"]) != (saved["width"], saved["height"]):
            raise RuntimeError("實際相機解析度與校正檔不一致，請重新 cam calib")
        write_json(Path(path) / "camera_settings.json", cam.settings)
        rec = _Recorder(sink, 90)
        rec.start()
        times, ready, last_ok = deque(maxlen=61), False, time.perf_counter()
        while not stop.is_set():
            fr = cam.read()
            if fr is None:
                if time.perf_counter() - last_ok > 2:
                    raise RuntimeError("相機連續 2 秒沒有畫面")
                continue
            last_ok = time.perf_counter()
            ball = detect_ball(fr.gray, det)
            t_done = now_ns()
            x, y = mapping.to_mm(ball.u, ball.v) if ball.found else (float("nan"), float("nan"))
            # Windows has no capture timestamp; this latency is a configurable estimate.
            shot_ns = (fr.t_cap_ns if fr.t_cap_ns is not None else fr.t_read_ns) - round(options["latency_ms"] * 1e6)
            times.append(fr.t_read_ns)
            fps = (len(times) - 1) * 1e9 / (times[-1] - times[0]) if len(times) > 1 else 0
            with shared.get_lock():
                shared[:] = (fr.index, shot_ns, fr.t_read_ns, int(ball.found), x / 1000, y / 1000, fps, t_done)
            row = [fr.index, fr.t_cap_ns, fr.t_read_ns, t_done, int(ball.found),
                   ball.u if ball.found else None, ball.v if ball.found else None,
                   ball.r if ball.found else None, x if ball.found else None,
                   y if ball.found else None, ball.area if ball.found else None]
            keep = rec.q.qsize() < 60
            rec.q.put((row, fr if keep else None))
            recorded_drops += int(not keep)
            count += 1
            if not ready and len(times) >= 60:
                if fps < options["min_fps"]:
                    raise RuntimeError(f"相機只有 {fps:.1f} fps，低於 {options['min_fps']:.1f}；請檢查曝光設定")
                events.put(("ready", "vision", f"相機 {fps:.1f} fps"))
                ready = True
            if not options["no_view"]:
                cv2.imshow("balance", draw_view(fr.gray, ball, [f"{fps:.1f} fps  x {x:+.1f} y {y:+.1f} mm", "q: stop"], roi=platform_roi))
                key = cv2.waitKey(1) & 0xff
                if key == ord("q") or cv2.getWindowProperty("balance", cv2.WND_PROP_VISIBLE) < 1:
                    stop.set()
                elif key == ord(" "):
                    events.put(("push", "vision", str(now_ns())))
                elif key in (10, 13):
                    events.put(("arm", "vision", ""))
            if rec.error is not None:
                raise RuntimeError(f"錄影寫入失敗：{rec.error}")
    except BaseException as e:
        events.put(("error", "vision", str(e)))
        stop.set()
    finally:
        if cam is not None:
            cam.close()
        if rec is not None:
            rec.q.put(None)
            rec.join(timeout=5)
            if rec.is_alive():
                events.put(("error", "vision", "錄影執行緒未能正常結束"))
            else:
                sink.close()
        else:
            sink.close()
        cv2.destroyAllWindows()
        write_json(Path(path) / "vision_summary.json", {"frames": count, "recording_dropped": recorded_drops,
                                                        "recording_error": rec.error if rec else None})


def _control(options, snapshot, shared, display, control_ring, usb_ring, origin, arm, stop, events):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from ppsim.params import Config
    from ppsim.controller import Controller, MODES
    from ppsim.estimator import Estimator, ROLLING
    from ppsim.kinematics import Platform
    from ppsim.sensors import Camera as CameraModel

    link, started = None, False
    try:
        rows = snapshot["servo"]
        cfg = Config()
        cfg.servo.pwm_hz, cfg.servo.pwm_counts = 300, 32768
        cfg.piezo.enabled = False
        cfg.camera.fps, cfg.camera.latency = options["fps"], options["latency_ms"] / 1000
        cfg.camera.width_px, cfg.camera.height_px = options["width"], options["height"]
        cfg.camera.height = options["mount_m"]
        mmpp = snapshot["camera"]["map"]["mm_per_px"]
        cfg.camera.hfov_deg = math.degrees(2 * math.atan(cfg.camera.width_px * mmpp / (2000 * cfg.camera.height)))
        cfg.camera.sigma_center_px = .5
        cfg.control.lead = 1 / cfg.servo.kp + cfg.servo.deadtime + .5 / 300 + snapshot["usb"]["one_way_ms"] / 1000
        platform = Platform(cfg.geom)
        model = CameraModel(cfg.camera, cfg.geom.z_surface0 + cfg.camera.height, cfg.ball.radius, np.random.default_rng(1))

        def new_tracking():
            c = Controller(cfg, platform, juggle_until=0.0)
            e = Estimator(cfg, model, c.plate_at)
            c.est = e
            return c, e

        controller, estimator = new_tracking()
        link = Teensy(options["port"])
        link.configure(rows, 100, 0, options["i2c_khz"])
        deadline = time.perf_counter() + 5
        while not stop.is_set():
            if time.perf_counter() >= deadline:
                raise RuntimeError("Teensy / IMU 無法就緒")
            if STATUS in link.latest and IMU in link.latest:
                s = link.get_latest(STATUS).status()
                link.get_latest(IMU)
                if not s["homing"]:
                    break
            time.sleep(.02)
        if stop.is_set():
            return
        events.put(("ready", "control", f"Teensy {link.port}、IMU WHO_AM_I=0x{s['who_am_i']:02X}"))

        def drain_usb():
            for m in link.drain():
                usb_ring.put((m.received_ns, m.kind, m.seq, np.frombuffer(m.payload, dtype="u1")))

        while not stop.is_set() and not arm.wait(.02):
            drain_usb()
            if link.error:
                raise RuntimeError(str(link.error))
        if stop.is_set():
            return
        # The operator confirms that the mechanism is already unfolded.
        centers = np.array([r["center_us"] for r in rows])
        s = link.get_latest(STATUS).status()
        initial = np.array(s["pulses_us"]) if s["enabled"] else centers
        started = True
        ramp_start = time.perf_counter()
        while not stop.is_set():
            u = min(1.0, time.perf_counter() - ramp_start)
            link.set_pulses(initial + (centers - initial) * u)
            drain_usb()
            if u >= 1:
                break
            time.sleep(.02)
        if stop.is_set():
            return
        origin.value = now_ns()
        last_frame, last_ping = -1, 0
        recovery = BallRecovery(centers, origin.value)
        pulses = centers.copy()
        frame, shot_ns, read_ns, found, x, y, fps, heartbeat = [-1, 0, 0, 0, 0, 0, 0, 0]
        step = 0
        with high_priority():
            events.put(("running", "control", "500 Hz 平衡控制已開始"))
            while not stop.is_set():
                target_ns = origin.value + step * 2000000
                while now_ns() < target_ns:
                    if stop.is_set():
                        break
                if stop.is_set():
                    break
                tick = now_ns()
                if tick - target_ns > 100000000:
                    raise RuntimeError("控制迴圈停頓超過 100 ms，回中位")
                t = (tick - origin.value) / 1e9
                if shared.get_lock().acquire(False):
                    try:
                        frame, shot_ns, read_ns, found, x, y, fps, heartbeat = shared[:]
                    finally:
                        shared.get_lock().release()
                if tick - heartbeat > 200000000:
                    raise RuntimeError("相機資料超過 200 ms 沒更新，回中位")
                s = link.get_latest(STATUS, age=.2).status()
                imu = link.get_latest(IMU, age=.2).imu()
                if not s["configured"] or s["failsafe"] or not s["imu_ready"]:
                    raise RuntimeError("Teensy 已重新啟動、進入斷線保護或 IMU 失聯")
                if step == 0:
                    options["imu_errors_at_start"] = imu["errors"]
                elif imu["errors"] > options["imu_errors_at_start"]:
                    raise RuntimeError("IMU 讀取新增錯誤，回中位")
                new_frame = frame != last_frame
                t_m = (shot_ns - origin.value) / 1e9
                valid_ball = bool(new_frame and found and t_m >= 0 and
                                  math.isfinite(x) and math.isfinite(y) and
                                  math.hypot(x, y) <= cfg.geom.plate_diameter / 2)
                if new_frame:
                    last_frame = frame
                waiting, fallback, transition = recovery.update(tick, read_ns, shot_ns, valid_ball, pulses)
                if transition == "lost":
                    events.put(("tracking_lost", "control", "找不到盤內球，慢慢回中位並持續等待；重新找到球後自動繼續。"))
                elif transition == "resumed":
                    controller, estimator = new_tracking()
                    events.put(("tracking_resumed", "control", "已重新找到球，重新建立追蹤並繼續平衡。"))
                if waiting:
                    pulses = fallback
                    state = None
                    mode = len(MODES)  # WAIT_BALL
                else:
                    if valid_ball:
                        point, n, vpl = estimator.plane(t_m)
                        z = point[2] + (cfg.ball.radius - n[0] * x - n[1] * y) / n[2]
                        acquiring = estimator.t is None
                        estimator.on_camera(t_m, np.array([x, y, z]), t)
                        if acquiring and estimator.t is not None:
                            # This entrypoint handles a ball resting on the plate.
                            estimator.mode = ROLLING
                            estimator.x[5] = vpl[2]
                    angles = controller.update(t)
                    pulses = angles_to_pulses(angles, rows)
                    state, _ = estimator.predict(t)
                    mode = MODES.index(controller.mode)
                seq = link.set_pulses(pulses)
                if tick - last_ping >= 100000000:
                    token = now_ns()
                    link.send(PING, struct.pack("<Q", token))
                    last_ping = tick
                drain_usb()
                pos = state[:3] if state is not None else [float("nan")] * 3
                vel = state[3:] if state is not None else [float("nan")] * 3
                control_ring.put((tick, seq, tick - target_ns, int(frame), int(shot_ns), (x, y), pos, vel, pulses, mode))
                if display.get_lock().acquire(False):
                    try:
                        display[:] = (tick, mode, pos[0], pos[1], math.hypot(vel[0], vel[1]),
                                      (tick - target_ns) / 1e6, *pulses)
                    finally:
                        display.get_lock().release()
                step += 1
                # Skip missed deadlines; never send a burst of obsolete commands.
                step = max(step, (now_ns() - origin.value) // 2000000)
    except BaseException as e:
        events.put(("error", "control", str(e)))
        stop.set()
    finally:
        if link is not None:
            if started:
                try:
                    link.home()
                    time.sleep(1.05)
                except Exception:
                    pass
            for m in link.drain():
                usb_ring.put((m.received_ns, m.kind, m.seq, np.frombuffer(m.payload, dtype="u1")))
            link.close()
        events.put(("finished", "control", ""))


def cmd_balance(args):
    from ppsim.controller import MODES
    snapshot = calib.load()
    snapshot["servo"] = servo_rows(require=True)
    camcal = snapshot.get("camera", {})
    if not all(k in camcal for k in ("map", "platform_r_px", "width", "height")):
        raise RuntimeError("請先執行 python -m ppreal cam calib --exposure -7")
    one_way = snapshot.get("usb", {}).get("one_way_ms")
    if one_way is None or not math.isfinite(one_way) or not 0 <= one_way < 20:
        raise RuntimeError("請先執行 python -m ppreal usb，儲存 USB 延遲")
    if args.latency_ms < 0 or not math.isfinite(args.latency_ms) or args.latency_ms > 200:
        raise ValueError("相機延遲必須在 0–200 ms")
    if (args.width, args.height) != (camcal["width"], camcal["height"]):
        raise RuntimeError("要求解析度與相機校正檔不同，請重新校正")
    ctx = mp.get_context("spawn")
    shared = ctx.Array("d", [-1., 0, 0, 0, 0, 0, 0, 0])
    display = ctx.Array("d", [0.] * 9)
    controls, usb = Ring(ctx, CONTROL_DTYPE, 8192), Ring(ctx, USB_DTYPE, 16384)
    origin = ctx.RawValue(ctypes.c_uint64, 0)
    arm, stop, events = ctx.Event(), ctx.Event(), ctx.Queue()
    options = dict(vars(args))
    options.pop("func", None)
    options["imu_errors_at_start"] = 0
    status, failure, processes = "stopped", None, []
    with Run("balance", args.note) as run:
        if shutil.disk_usage(run.path).free < 1024 ** 3:
            raise RuntimeError("硬碟可用空間不足 1 GB")
        run.update(calib=snapshot, backend="teensy", protocol=1, options=options,
                   control_dtype=CONTROL_DTYPE.descr, usb_dtype=USB_DTYPE.descr, imu_dtype=IMU_DTYPE.descr,
                   control_modes=list(MODES) + ["WAIT_BALL"],
                   camera_latency_assumed_ms=args.latency_ms)
        cfile = open(run.path / "control.bin", "wb")
        ufile = open(run.path / "usb.bin", "wb")
        ifile = open(run.path / "imu.bin", "wb")
        clock, usb_count, control_count = ClockMap(), 0, 0
        ping_csv = run.csv("ping.csv", ["before_ns", "after_ns", "micros"])
        console = open(run.path / "console.log", "w", encoding="utf-8")

        def say(message):
            print(message)
            console.write(message + "\n")
            console.flush()

        def collect():
            nonlocal usb_count, control_count
            rows = controls.drain()
            cfile.write(rows.tobytes())
            control_count += len(rows)
            rows = usb.drain()
            ufile.write(rows.tobytes())
            usb_count += len(rows)
            for row in rows:
                payload = row["payload"].tobytes()
                if row["kind"] == IMU:
                    raw = struct.unpack_from("<I7hI", payload)
                    sample = np.array([(int(row["t_ns"]), raw[0], raw[1:8], raw[8])], dtype=IMU_DTYPE)
                    ifile.write(sample.tobytes())
                elif row["kind"] == PONG:
                    micros = struct.unpack_from("<I", payload)[0]
                    before = struct.unpack_from("<Q", payload, 4)[0]
                    try:
                        clock.add(before, int(row["t_ns"]), micros)
                        ping_csv.row(before, int(row["t_ns"]), micros)
                    except RuntimeError as e:
                        say(str(e))

        try:
            vision = ctx.Process(target=_vision, args=(options, snapshot, str(run.path), shared, stop, events), name="pp-vision")
            control = ctx.Process(target=_control, args=(options, snapshot, shared, display, controls, usb, origin, arm, stop, events), name="pp-control")
            for process in (vision, control):
                process.start()
                processes.append(process)
            say(f"紀錄：{run.path}")
            ready, prompted, last_print = set(), False, 0
            deadline = time.perf_counter() + 30
            editing, note = False, ""
            while not stop.is_set():
                collect()
                while True:
                    try:
                        kind, source, detail = events.get_nowait()
                    except queue.Empty:
                        break
                    run.event(kind, detail, src=source)
                    if kind == "ready":
                        ready.add(source)
                        say(detail)
                        if source == "vision":
                            with open(run.path / "camera_settings.json", encoding="utf-8") as f:
                                run.update(camera=json.load(f))
                    elif kind == "error":
                        failure = f"{source}：{detail}"
                        stop.set()
                    elif kind == "running":
                        say(detail)
                    elif kind in ("tracking_lost", "tracking_resumed"):
                        say(detail)
                    elif kind == "push":
                        say("已記錄手推。")
                    elif kind == "arm" and prompted:
                        with shared.get_lock():
                            has_ball = bool(shared[3]) and now_ns() - shared[2] < 200000000
                        if has_ball:
                            arm.set()
                            prompted = False
                for process in processes:
                    if not process.is_alive() and not stop.is_set():
                        failure = f"{process.name} 提早結束（{process.exitcode}）"
                        stop.set()
                if ready != {"vision", "control"} and time.perf_counter() > deadline:
                    raise RuntimeError("30 秒內未能完成相機和 Teensy 的啟動檢查")
                if ready == {"vision", "control"} and not prompted and not arm.is_set():
                    say("請確認平台已展開到校正中位、伺服電源開啟，並把球放在平台中心。")
                    say("按 Enter 開始；q 停止，空白鍵記手推，m 輸入備註。")
                    prompted = True
                key = _key()
                if editing:
                    if key in ("\r", "\n"):
                        run.event("note", note, src="launcher")
                        say(f"備註：{note}")
                        note, editing = "", False
                    elif key == "\b":
                        note = note[:-1]
                    elif key:
                        note += key
                elif key.lower() == "q":
                    stop.set()
                elif key in ("\r", "\n") and prompted:
                    with shared.get_lock():
                        has_ball = bool(shared[3]) and now_ns() - shared[2] < 200000000
                    if has_ball:
                        arm.set()
                        prompted = False
                        run.event("arm", "使用者確認已展開", src="launcher")
                    else:
                        say("目前找不到球，請先把球放到平台中央。")
                elif key == " ":
                    run.event("push", "手推", src="launcher")
                    say("已記錄手推。")
                elif key.lower() == "m":
                    editing = True
                    say("請輸入備註，按 Enter 儲存：")
                if origin.value:
                    elapsed = (now_ns() - origin.value) / 1e9
                    if args.seconds is not None and elapsed >= args.seconds:
                        status = "done"
                        stop.set()
                    if time.perf_counter() - last_print >= .5:
                        with display.get_lock():
                            d = display[:]
                        if d[0]:
                            if int(d[1]) == len(MODES):
                                say(f"{elapsed:6.1f}s  等待重新找到球；暫停球位置修正  晚 {d[5]:.2f} ms")
                            else:
                                say(f"{elapsed:6.1f}s  球 x {d[2]*1000:+6.1f} y {d[3]*1000:+6.1f} mm  "
                                    f"速度 {d[4]*100:.1f} cm/s  晚 {d[5]:.2f} ms")
                        last_print = time.perf_counter()
                time.sleep(.01)
        except KeyboardInterrupt:
            stop.set()
        except BaseException as e:
            failure = str(e)
            stop.set()
        finally:
            stop.set()
            for process in reversed(processes):
                process.join(timeout=7)
                if process.is_alive():
                    failure = failure or f"{process.name} 未能正常結束"
                    process.terminate()
                    process.join(timeout=2)
            collect()
            while True:
                try:
                    kind, source, detail = events.get_nowait()
                except queue.Empty:
                    break
                run.event(kind, detail, src=source)
                if kind == "error":
                    failure = failure or f"{source}：{detail}"
            for f in (cfile, ufile, ifile):
                f.close()
            summary = {"control_rows": control_count, "usb_rows": usb_count,
                       "control_dropped": controls.dropped.value, "usb_dropped": usb.dropped.value}
            if len(clock.samples) >= 20:
                try:
                    summary["clock_map"] = {**clock.fit(), "origin_ns": clock.origin_ns,
                                             "slope": clock.slope, "offset_s": clock.offset,
                                             "last_raw_micros": clock.last_raw, "unwrapped_micros": clock.unwrapped}
                except ValueError as e:
                    summary["clock_map_error"] = str(e)
            if origin.value:
                run.update(control_t0_ns=origin.value)
            if failure:
                say(f"停止原因：{failure}")
                run.event("error", failure, src="launcher")
            say(f"已結束。紀錄：{run.path}")
            console.close()
            run.close("crashed" if failure else status, summary)
            events.close()
    return 2 if failure else 0


def _key():
    if sys.platform == "win32":
        import msvcrt
        if msvcrt.kbhit():
            c = msvcrt.getwch()
            if c in ("\x00", "\xe0"):
                if msvcrt.kbhit():
                    msvcrt.getwch()
                return ""
            return c
        return ""
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().rstrip("\n") or "\n"
    return ""


def register(sub):
    run = sub.add_parser("run", help="實機控制").add_subparsers(dest="mode", required=True)
    p = run.add_parser("balance", help="筆電＋Teensy 的 500 Hz 球平衡")
    p.add_argument("--port", default="")
    p.add_argument("--note", default="")
    p.add_argument("--seconds", type=positive, default=None, help="開始平衡後的執行秒數；預設持續至 q")
    p.add_argument("--device", default="PC CAMERA")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=positive, default=30)
    p.add_argument("--min-fps", type=positive, default=25,
                   help="啟動時最低可接受的實測 FPS（預設 25）")
    p.add_argument("--fourcc", choices=("YUYV", "MJPG"), default="YUYV")
    p.add_argument("--exposure", type=float, default=-7)
    p.add_argument("--latency-ms", type=float, default=60, help="相機到讀到畫面的估計延遲；之後改成實測值")
    p.add_argument("--mount-m", type=positive, default=.45, help="鏡頭到平台面高度，公尺")
    p.add_argument("--i2c-khz", type=int, choices=(100, 400), default=400)
    p.add_argument("--no-view", action="store_true")
    p.set_defaults(func=cmd_balance)
