"""筆電端手動操作：python -m ppreal usb / servo / motion / imu。"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import math
import sys
import threading
import time

import numpy as np

from . import calib
from .runlog import Run, now_ns, write_json
from .teensy import (Teensy, ClockMap, STATUS, IMU, PIEZO, LED, LED_EVENT, DIAG, FW_EVENT, SET,
                     default_servos, validate_servos, angles_to_pulses)


def positive(value):
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError("必須是大於 0 的有限數字")
    return v


def servo_rows(require=False):
    rows = calib.load().get("servo")
    if require and (rows is None or not all(r.get("calibrated", False) for r in rows)):
        raise RuntimeError("請先用 servo calibrate 存下三顆伺服實測的方向、中位與比例")
    return validate_servos(rows if rows is not None else default_servos())


def save_servos(rows, what):
    data = calib.load()
    data["servo"] = validate_servos(rows)
    data.setdefault("history", []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "what": what})
    write_json(calib.calib_path(), data)


def configure(link, args, rows=None):
    edge = {None: 0, "off": 0, "rising": 1, "falling": 2}[getattr(args, "edge", None)]
    link.configure(rows or servo_rows(), int(getattr(args, "imu_hz", 100)), edge,
                   int(getattr(args, "i2c_khz", 400)))


@contextlib.contextmanager
def high_priority():
    if sys.platform != "win32":
        yield
        return
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetPriorityClass.argtypes = (ctypes.c_void_p,)
    k.GetPriorityClass.restype = ctypes.c_uint32
    k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    h = k.GetCurrentProcess()
    old = k.GetPriorityClass(h)
    if not old or not k.SetPriorityClass(h, 0x80) or k.GetPriorityClass(h) != 0x80:
        raise OSError("無法設定並確認高優先權")
    try:
        yield
    finally:
        k.SetPriorityClass(h, old)


def statistics_ms(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        raise RuntimeError("沒有成功樣本，無法統計")
    return {"median_ms": float(np.median(a)), "p99_ms": float(np.quantile(a, .99)),
            "p999_ms": float(np.quantile(a, .999)), "max_ms": float(a.max()),
            "over_1ms": int((a > 1).sum()), "over_2ms": int((a > 2).sum())}


class USBLog:
    """Disk writes stay on this thread, away from motion keepalive."""
    def __init__(self, link, run):
        self.link, self.run = link, run
        self.stop = threading.Event()
        self.error = None
        self.previous_status = None
        self.previous_diag = None
        self.summary = {"homing_entries": 0, "watchdog_events": 0, "host_gaps_over_100ms": 0,
                        "host_gaps_over_200ms": 0, "firmware_diagnostics": False}
        self.usb = run.csv("usb.csv", ["t_ns", "type", "seq", "payload_hex"])
        self.imu = run.csv("imu.csv", ["t_ns", "micros", "ax_raw", "ay_raw", "az_raw", "temp_raw",
                                       "gx_raw", "gy_raw", "gz_raw", "errors"])
        self.piezo = run.csv("piezo.csv", ["t_ns", "micros", "seq", "overflow"])
        self.status = run.csv("status.csv", ["t_ns", "micros", "last_seq", "pwm_enabled", "homing",
                                             "failsafe", "configured", "pwm1_us", "pwm2_us", "pwm3_us",
                                             "bad_frames", "imu_errors", "tx_dropped"])
        self.sent = run.csv("host_tx.csv", ["before_ns", "after_ns", "type", "seq", "payload_hex",
                                            "set_gap_ms", "write_ms"])
        self.diag = run.csv("teensy_diag.csv", ["t_ns", "uptime_ms", "set_count", "watchdog_count",
                                               "max_set_gap_us", "pwm_register_writes", "event_dropped",
                                               "reset_reason_raw"])
        self.events = run.csv("events_teensy.csv", ["t_ns", "micros", "event", "detail"])
        self.link.trace_tx = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _drain(self):
        import struct
        sent, events = self.link.drain_trace()
        for before, after, kind, seq, payload, gap in sent:
            self.sent.row(before, after, kind, seq, payload, gap, (after - before) / 1e6)
            if kind == SET and gap is not None and gap > 100:
                self.summary["host_gaps_over_100ms"] += 1
                self.summary["host_gaps_over_200ms"] += int(gap >= 200)
                self.events.row(before, None, "host_set_gap", f"送出間隔 {gap:.3f} ms；序號 {seq}")
        for t, kind, detail in events:
            self.events.row(t, None, kind, detail)
        for m in self.link.drain():
            self.usb.row(m.received_ns, m.kind, m.seq, m.payload.hex())
            if m.kind == IMU:
                self.imu.row(m.received_ns, *struct.unpack_from("<I7hI", m.payload))
            elif m.kind == PIEZO:
                self.piezo.row(m.received_ns, *struct.unpack_from("<III", m.payload))
            elif m.kind == STATUS:
                s = m.status()
                self.status.row(m.received_ns, s["micros"], s["last_seq"], int(s["enabled"]), int(s["homing"]),
                                int(s["failsafe"]), int(s["configured"]), *s["pulses_us"],
                                s["bad_frames"], s["imu_errors"], s["tx_dropped"])
                old = self.previous_status
                if old is not None:
                    if s["homing"] and not old["homing"]:
                        self.summary["homing_entries"] += 1
                        self.events.row(m.received_ns, s["micros"], "homing_enter",
                                        f"開始回中位；最後命令 {s['last_seq']}；脈寬 {s['pulses_us']}")
                    if s["enabled"] != old["enabled"]:
                        self.events.row(m.received_ns, s["micros"], "pwm_enabled_changed", str(s["enabled"]))
                    if s["configured"] != old["configured"]:
                        self.events.row(m.received_ns, s["micros"], "configured_changed", str(s["configured"]))
                self.previous_status = s
            elif m.kind == DIAG:
                d = m.diagnostic()
                self.summary["firmware_diagnostics"] = True
                self.diag.row(m.received_ns, *d.values())
                if self.previous_diag is not None and d["watchdog_count"] > self.previous_diag["watchdog_count"]:
                    self.events.row(m.received_ns, None, "watchdog_counter_increased",
                                    f"累計 {d['watchdog_count']}；最長收命令間隔 {d['max_set_gap_us']} us")
                if (self.previous_diag is not None and d["uptime_ms"] < self.previous_diag["uptime_ms"] and
                        self.previous_diag["uptime_ms"] - d["uptime_ms"] < 0x80000000):
                    self.events.row(m.received_ns, None, "device_restart_suspected",
                                    f"運行時間倒退；reset_reason_raw=0x{d['reset_reason_raw']:08X}")
                self.previous_diag = d
            elif m.kind == FW_EVENT:
                e = m.firmware_event()
                self.summary["watchdog_events"] += int(e["event"] == "watchdog_trip")
                self.events.row(m.received_ns, e["micros"], e["event"], json.dumps(e, ensure_ascii=False))

    def _run(self):
        try:
            while not self.stop.wait(.02):
                self._drain()
            self._drain()
        except Exception as e:
            self.error = str(e)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.link.trace_tx = False
        self.summary.update(max_host_set_gap_ms=self.link.max_set_send_gap_ms,
                            tx_trace_dropped=self.link.tx_trace_dropped,
                            last_firmware_diagnostic=self.previous_diag)
        self.run.update(hold_debug=self.summary)
        if self.error:
            raise RuntimeError(f"紀錄寫入失敗：{self.error}")


class Hold:
    def __init__(self, link, pulses, finish="home"):
        self.link, self.pulses, self.finish = link, tuple(pulses), finish
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            while not self.stop.is_set():
                self.link.set_pulses(self.pulses)
                self.stop.wait(.02)
        except Exception as e:
            self.error = e
            self.link.note("hold_send_error", str(e))
            self.stop.set()

    def check(self):
        if self.error:
            raise RuntimeError(f"控制命令送出失敗：{self.error}")

    def ramp(self, target, seconds=3):
        start = np.asarray(self.pulses, dtype=float)
        end = np.asarray(target, dtype=float)
        before = time.perf_counter()
        while True:
            self.check()
            u = min(1.0, (time.perf_counter() - before) / seconds)
            u = u * u * (3 - 2 * u)
            self.pulses = tuple(start + (end - start) * u)
            if u >= 1:
                break
            time.sleep(.01)

    def __enter__(self):
        self.link.set_pulses(self.pulses)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=.2)
        try:
            self.link.off() if self.finish == "off" else self.link.home()
            if self.finish != "off":
                time.sleep(1.05)
        except (Exception, KeyboardInterrupt):
            pass  # Firmware watchdog takes over if the USB connection is lost.


def pose_pulses(rows, front_deg=0, right_deg=0, heave_mm=0):
    from ppsim.params import Config
    from ppsim.kinematics import Platform
    if not all(math.isfinite(x) for x in (front_deg, right_deg, heave_mm)):
        raise ValueError("姿態參數必須是有限數字")
    if math.hypot(front_deg, right_deg) > 8 or abs(heave_mm) > 22:
        raise ValueError("手動姿態限制：合成傾角 ≤8°、升降 ±22 mm")
    plat = Platform(Config().geom)
    c, r, _ = plat.pose(heave_mm / 1000, -math.tan(math.radians(right_deg)), -math.tan(math.radians(front_deg)))
    angles, ok = plat.ik(c, r)
    if not ok:
        raise ValueError("機構無法到達這個姿態")
    return angles_to_pulses(angles, rows)


def _wait_key(hold, debug=False):
    from .keyboard import poll_key
    print("持續送出命令。q：結束；空白鍵：標記抖動；Ctrl+C 也可停止。")
    next_print = 0
    while True:
        hold.check()
        before = time.perf_counter_ns()
        key = poll_key()
        elapsed_ms = (time.perf_counter_ns() - before) / 1e6
        if elapsed_ms > 100:
            hold.link.note("keyboard_poll_slow", f"鍵盤輪詢耗時 {elapsed_ms:.3f} ms")
        if key.lower() == "q":
            hold.link.note("user_stop", "使用者按 q，要求回中位或停止 PWM")
            return
        if key == " ":
            hold.link.note("user_twitch", "使用者按空白鍵標記抖動")
            print("已標記抖動時間。")
        if debug and time.perf_counter() >= next_print:
            status = hold.link.latest.get(STATUS)
            diagnostic = hold.link.latest.get(DIAG)
            if status is not None:
                s = status.status()
                age_ms = (time.perf_counter_ns() - status.received_ns) / 1e6
                extra = (f"累計保護 {diagnostic.diagnostic()['watchdog_count']} 次"
                         if diagnostic else "尚無新版韌體診斷；已記錄基本狀態")
                print(f"PWM={int(s['enabled'])} 回中位={int(s['homing'])} 保護={int(s['failsafe'])} "
                      f"脈寬={s['pulses_us']} 最新狀態距今={age_ms:.0f} ms "
                      f"PC最長送出間隔={hold.link.max_set_send_gap_ms:.1f} ms；{extra}")
            next_print = time.perf_counter() + 1
        time.sleep(.02)


def cmd_usb(args, link, run):
    n = max(20, round(args.seconds * 500))
    clock, samples, lost, consecutive = ClockMap(), [], 0, 0
    w = run.csv("ping.csv", ["before_ns", "after_ns", "micros", "rtt_ms", "late_ms"])
    with high_priority():
        start = time.perf_counter() + .05
        for i in range(n):
            target = start + i / 500
            while time.perf_counter() < target:
                pass
            late_ms = max(0, (time.perf_counter() - target) * 1000)
            try:
                a, b, t = link.ping(timeout=.05)
            except TimeoutError:
                lost += 1
                consecutive += 1
                if consecutive >= 5:
                    raise RuntimeError("連續 5 包沒有回應，停止量測。請確認韌體與 USB 連線。")
                continue
            consecutive = 0
            rtt = (b - a) / 1e6
            clock.add(a, b, t)
            samples.append(rtt)
            w.row(a, b, t, rtt, late_ms)
    st = {**statistics_ms(samples), **clock.fit(), "success": len(samples), "lost": lost}
    print(json.dumps(st, ensure_ascii=False, indent=2))
    run.update(usb=st)
    if lost == 0:
        calib.update("usb", {"rtt_median_ms": st["median_ms"], "rtt_p99_ms": st["p99_ms"],
                             "rtt_max_ms": st["max_ms"], "one_way_ms": st["median_ms"] / 2}, "USB 延遲量測")
    return 0 if lost == 0 else 1


def cmd_loop(args):
    # Scheduling benchmark, started explicitly by the user.
    late = []
    with Run("loop", args.note) as run, high_priority():
        start = time.perf_counter() + .05
        for i in range(max(1, round(args.seconds * 500))):
            target = start + i / 500
            while time.perf_counter() < target:
                pass
            before = time.perf_counter()
            late.append((before - target) * 1000)
            while time.perf_counter() - before < .0006:
                pass
        st = statistics_ms(late)
        run.update(loop=st)
        np.save(run.path / "late_ms.npy", late)
        print(json.dumps(st, ensure_ascii=False, indent=2))
        print(f"紀錄：{run.path}")
    return int(st["over_2ms"] > 0)


def _current_or(link, fallback):
    st = link.get_latest(STATUS).status()
    return st["pulses_us"] if st["enabled"] else fallback


def cmd_motion(args, link, run):
    rows = servo_rows(require=True)
    center = [r["center_us"] for r in rows]
    fold = angles_to_pulses(np.radians([-60.] * 3), rows)
    status = link.get_latest(STATUS).status()
    if args.cmd != "unfold" and not status["enabled"]:
        raise RuntimeError("目前沒有輸出 PWM，無法判定機構位置。請先用 motion unfold --from-fold 從已收合的位置展開。")
    if args.cmd == "fold":
        initial = _current_or(link, fold)
        target, finish = fold, "off"
        print("收合後會持續保持。先關伺服電源，再按 q；結束會停止 PWM。")
    elif args.cmd == "unfold":
        initial = _current_or(link, fold)
        target, finish = center, "home"
        if not link.get_latest(STATUS).status()["enabled"] and not args.from_fold:
            raise RuntimeError("請先用 servo hold fold 設好收合位置。關伺服電後結束該指令，"
                               "再以 motion unfold --from-fold 啟動。")
    else:
        initial = _current_or(link, center)
        target = pose_pulses(rows, args.front_deg, args.right_deg, args.heave_mm)
        finish = "home"
    with Hold(link, initial, finish) as hold:
        if args.cmd == "unfold" and args.from_fold:
            input("目前正在送收合脈寬。開啟伺服電源後，按 Enter 慢慢展開：")
            hold.check()
        hold.ramp(target, args.ramp)
        print("脈寬 µs：" + " / ".join(f"{x:.1f}" for x in target))
        _wait_key(hold)


def cmd_servo(args, link, run):
    rows = servo_rows(require=args.value == "fold")
    if args.value == "fold":
        pulses = angles_to_pulses(np.radians([-60.] * 3), rows)
    elif args.value == "center":
        pulses = [r["center_us"] for r in rows]
    else:
        pulses = [r["center_us"] for r in rows]
        v = float(args.value)
        for i in range(3):
            if args.leg is None or i == args.leg - 1:
                pulses[i] = v
    print("三路脈寬 µs：" + " / ".join(f"{x:.1f}" for x in pulses))
    finish = "off" if args.value == "fold" or args.on_exit == "off" else "home"
    if args.value == "fold":
        print("保持送出收合位置；先關伺服電源再按 q，結束時停止 PWM。")
    with Hold(link, pulses, finish) as hold:
        _wait_key(hold, debug=args.debug)


def cmd_calibrate(args):
    rows = servo_rows()
    r = rows[args.leg - 1]
    r.update(leg=args.leg, pin=args.leg + 1, dir=args.direction, center_us=args.center,
             center_us_jig=args.center, us_per_deg_up=args.up, us_per_deg_down=args.down,
             min_us=args.minimum, max_us=args.maximum, calibrated=True)
    save_servos(rows, f"手動輸入 {args.leg} 號伺服校正")
    print(f"已儲存：{calib.calib_path()}（沒有送出伺服命令）")


def cmd_read(args, link, run):
    kind = IMU if args.group == "imu" else STATUS
    start = time.perf_counter()
    until = start + args.seconds
    seen = None
    while time.perf_counter() < until:
        if link.error:
            raise RuntimeError(str(link.error))
        try:
            m = link.get_latest(kind)
        except RuntimeError:
            if time.perf_counter() - start > 2:
                raise RuntimeError("收不到 IMU/狀態資料，請檢查接線與韌體")
            time.sleep(.1)
            continue
        if m.seq != seen:
            v = m.imu() if kind == IMU else m.status()
            print(json.dumps(v, ensure_ascii=False))
            seen = m.seq
        time.sleep(.2)
    if seen is None:
        raise RuntimeError("量測期間沒有收到資料")


def cmd_hardware(args):
    if args.group == "loop":
        return cmd_loop(args)
    if args.group == "servo" and args.cmd == "calibrate":
        return cmd_calibrate(args)
    with Run(args.group, args.note) as run:
        run.update(calib=calib.load(), backend="teensy", protocol=1)
        with Teensy(args.port) as link:
            configure(link, args)
            # Configuration does not start PWM. Wait only for telemetry.
            deadline = time.perf_counter() + 1
            while STATUS not in link.latest and time.perf_counter() < deadline:
                time.sleep(.01)
            log = USBLog(link, run)
            try:
                run.update(port=link.port)
                if args.group == "usb":
                    return cmd_usb(args, link, run)
                if args.group == "servo":
                    return cmd_servo(args, link, run)
                if args.group == "motion":
                    return cmd_motion(args, link, run)
                if args.group == "home":
                    link.home()
                    time.sleep(1.05)
                elif args.group == "off":
                    link.off()
                elif args.group == "led":
                    link.request(LED, bytes([int(args.on)]), LED_EVENT)
                elif args.group == "piezo":
                    print("撞擊事件會寫入 piezo.csv；Ctrl+C 結束。")
                    until = time.perf_counter() + args.seconds
                    last = None
                    while time.perf_counter() < until:
                        if link.error:
                            raise RuntimeError(str(link.error))
                        m = link.latest.get(PIEZO)
                        if m is not None and m.seq != last:
                            import struct
                            print("撞擊 micros / 序號 / 溢位：", struct.unpack_from("<III", m.payload))
                            last = m.seq
                        time.sleep(.02)
                else:
                    return cmd_read(args, link, run)
            finally:
                log.close()
                run.update(host_bad_frames=link.decoder.bad, host_dropped=link.host_dropped)
                print(f"紀錄：{run.path}")


def register(sub):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--port", default="", help="例如 COM5；預設自動找唯一的 Teensy")
    common.add_argument("--note", default="")
    common.add_argument("--imu-hz", type=int, choices=range(0, 251), default=100, metavar="0–250")
    common.add_argument("--i2c-khz", type=int, choices=(100, 400), default=400)

    def leaf(parent, name, help):
        p = parent.add_parser(name, help=help, parents=[common])
        p.set_defaults(func=cmd_hardware)
        return p

    p = leaf(sub, "usb", "500 Hz USB 延遲與對時量測（需 teensy_fw）")
    p.add_argument("--seconds", type=positive, default=10)
    p = leaf(sub, "loop", "手動啟動筆電 500 Hz 排程量測，不連接硬體")
    p.add_argument("--seconds", type=positive, default=60)
    for name in ("status", "imu", "piezo"):
        p = leaf(sub, name, {"status": "讀 Teensy 狀態", "imu": "讀 IMU 原始值", "piezo": "讀撞擊事件"}[name])
        if name == "imu":
            p.add_argument("cmd", choices=("raw",), nargs="?", default="raw")
        p.add_argument("--seconds", type=positive, default=60 if name != "status" else 10)
        if name == "piezo":
            p.add_argument("--edge", choices=("rising", "falling"), required=True)
    leaf(sub, "home", "在 1 秒內慢慢回到校正中位")
    leaf(sub, "off", "停止三路 PWM；伺服電源仍須手動關閉")
    p = leaf(sub, "led", "設定板上 LED")
    p.add_argument("on", type=int, choices=(0, 1))

    s = sub.add_parser("servo", help="手動伺服命令與校正輸入").add_subparsers(dest="cmd", required=True)
    p = leaf(s, "hold", "持續送出脈寬（µs）、center 或 fold；按 q 結束")
    p.add_argument("value")
    p.add_argument("--leg", type=int, choices=(1, 2, 3), help="只改此路，其餘兩路送中位")
    p.add_argument("--on-exit", choices=("home", "off"), default="home")
    p.add_argument("--debug", action="store_true", help="每秒顯示 PWM 與斷線保護狀態；空白鍵標記抖動")
    p = leaf(s, "calibrate", "儲存你手動量到的校正值，不連接硬體")
    p.add_argument("leg", type=int, choices=(1, 2, 3))
    p.add_argument("--direction", type=int, choices=(-1, 1), required=True)
    p.add_argument("--center", type=float, required=True)
    p.add_argument("--up", type=positive, required=True, help="往上每度幾 µs")
    p.add_argument("--down", type=positive, required=True, help="往下每度幾 µs")
    p.add_argument("--min", dest="minimum", type=float, default=500)
    p.add_argument("--max", dest="maximum", type=float, default=2500)

    s = sub.add_parser("motion", help="收合、展開、指定平台姿態").add_subparsers(dest="cmd", required=True)
    for name in ("fold", "unfold", "pose"):
        p = leaf(s, name, {"fold": "慢慢收合至 −60°", "unfold": "慢慢展開至中位", "pose": "手動指定姿態"}[name])
        p.add_argument("--ramp", type=positive, default=3)
        if name == "unfold":
            p.add_argument("--from-fold", action="store_true", help="確認機構目前已收合；先送收合脈寬，按 Enter 才展開")
        if name == "pose":
            p.add_argument("--front-deg", type=float, default=0, help="正值＝前方抬高")
            p.add_argument("--right-deg", type=float, default=0, help="正值＝右方抬高")
            p.add_argument("--heave-mm", type=float, default=0)
