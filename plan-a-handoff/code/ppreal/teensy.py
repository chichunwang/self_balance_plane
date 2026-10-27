"""Teensy USB protocol v1. No motion commands are sent by opening a connection."""
from __future__ import annotations

import binascii
from collections import deque
from dataclasses import dataclass
import math
import struct
import threading
import time

import numpy as np
try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

VERSION = 1
# Firmware layout: header(8) + payload(28) + reserved(2) + CRC(2).
FRAME_BODY = struct.Struct("<2sBBI28s2x")
CRC = struct.Struct("<H")
SIZE = FRAME_BODY.size + CRC.size
SET, CONFIG, PING, HOME, OFF, LED = range(1, 7)
STATUS, IMU, PIEZO, PONG, ACK, LED_EVENT = range(0x80, 0x86)
DIAG, FW_EVENT = 0x86, 0x87
FW_EVENT_NAMES = {
    1: "boot", 2: "pwm_start", 3: "watchdog_trip", 4: "home_request",
    5: "home_complete", 6: "command_resumed", 7: "pwm_off", 8: "configured",
}


def encode(kind: int, seq: int, payload: bytes = b"") -> bytes:
    if len(payload) > 28:
        raise ValueError("封包資料超過 28 bytes")
    raw = FRAME_BODY.pack(b"\xaa\x55", VERSION, kind, seq & 0xffffffff, payload)
    return raw + CRC.pack(binascii.crc_hqx(raw, 0xffff))


class Decoder:
    def __init__(self):
        self.buffer = bytearray()
        self.bad = 0

    def feed(self, data):
        self.buffer.extend(data)
        out = []
        while len(self.buffer) >= 2:
            start = self.buffer.find(b"\xaa\x55")
            if start < 0:
                self.buffer[:] = b"\xaa" if self.buffer[-1] == 0xaa else b""
                break
            del self.buffer[:start]
            if len(self.buffer) < SIZE:
                break
            raw = bytes(self.buffer[:SIZE])
            if (raw[2] != VERSION or
                    binascii.crc_hqx(raw[:FRAME_BODY.size], 0xffff) != CRC.unpack_from(raw, FRAME_BODY.size)[0]):
                del self.buffer[0]
                self.bad += 1
                continue
            _, _, kind, seq, payload = FRAME_BODY.unpack_from(raw)
            out.append((kind, seq, payload))
            del self.buffer[:SIZE]
        return out


@dataclass(frozen=True)
class Message:
    kind: int
    seq: int
    payload: bytes
    received_ns: int

    def status(self):
        t, last, a, b, c, flags, who, bad, errors, drops = struct.unpack("<II3HBBIII", self.payload)
        return {"micros": t, "last_seq": last, "pulses_us": [a / 10, b / 10, c / 10],
                "enabled": bool(flags & 1), "homing": bool(flags & 2), "imu_ready": bool(flags & 4),
                "configured": bool(flags & 8), "failsafe": bool(flags & 16), "who_am_i": who,
                "bad_frames": bad, "imu_errors": errors, "tx_dropped": drops,
                "received_ns": self.received_ns}

    def imu(self):
        v = struct.unpack_from("<I7hI", self.payload)
        return {"micros": v[0], "accel_raw": list(v[1:4]), "temp_raw": v[4], "gyro_raw": list(v[5:8]),
                "accel_g": [x / 16384 for x in v[1:4]], "temp_c": v[4] / 340 + 36.53,
                "gyro_dps": [x / 131 for x in v[5:8]], "errors": v[8], "received_ns": self.received_ns}

    def diagnostic(self):
        v = struct.unpack("<7I", self.payload)
        return dict(zip(("uptime_ms", "set_count", "watchdog_count", "max_set_gap_us",
                         "pwm_register_writes", "event_dropped", "reset_reason_raw"), v))

    def firmware_event(self):
        t, code, flags, _, seq, age, a, b, c, number = struct.unpack("<IBBHII3H2xI", self.payload)
        return {"micros": t, "code": code, "event": FW_EVENT_NAMES.get(code, f"unknown_{code}"),
                "flags": flags, "last_seq": seq, "command_age_us": age,
                "pulses_us": [a / 10, b / 10, c / 10], "event_number": number}


def find_port(port=""):
    if list_ports is None:
        raise RuntimeError("找不到 pyserial 套件。請使用已安裝 pyserial 的 Python 環境。")
    if port:
        return port
    found = [p.device for p in list_ports.comports() if p.vid == 0x16c0]
    if len(found) != 1:
        raise RuntimeError("找不到唯一的 Teensy。請插好 USB、選 Serial 韌體，或用 --port COM埠 指定。"
                           + (f" 找到：{', '.join(found)}" if found else ""))
    return found[0]


class Teensy:
    def __init__(self, port=""):
        self.port = find_port(port)
        self.serial = serial.Serial(self.port, 115200, timeout=0.02, write_timeout=0.01)
        self.decoder = Decoder()
        self.stop = threading.Event()
        self.cv = threading.Condition()
        self.write_lock = threading.Lock()
        self.pending = {}
        self.latest = {}
        self.messages = deque(maxlen=32768)
        self.host_dropped = 0
        self.received_bytes = 0
        self.received_frames = 0
        self.trace_tx = False
        self.tx_trace = deque(maxlen=32768)
        self.host_events = deque(maxlen=4096)
        self.tx_trace_dropped = 0
        self.last_set_sent_ns = None
        self.max_set_send_gap_ms = 0.0
        self.seq = 0
        self.error = None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        try:
            self._handshake()
        except BaseException:
            self.close()
            raise

    def _handshake(self):
        # Teensy Serial becomes ready 15 ms after DTR. Allow for Windows
        # scheduling and board startup, then retry only the harmless ping.
        self.serial.dtr = True
        time.sleep(.3)
        deadline = time.perf_counter() + 3.0
        last_timeout = None
        while time.perf_counter() < deadline:
            try:
                self.ping(timeout=min(.25, deadline - time.perf_counter()))
                return
            except TimeoutError as e:
                last_timeout = e
                time.sleep(.05)
        kinds = ", ".join(f"0x{k:02X}" for k in sorted(self.latest)) or "無"
        raise TimeoutError(
            f"{self.port} 連線握手逾時：收到 {self.received_bytes} bytes、"
            f"{self.received_frames} 個有效封包、{self.decoder.bad} 個錯誤封包；"
            f"封包類型：{kinds}。請確認板子正在執行 teensy_fw。"
        ) from last_timeout

    def _reader(self):
        try:
            while not self.stop.is_set():
                data = self.serial.read(max(1, min(self.serial.in_waiting, 4096)))
                self.received_bytes += len(data)
                for kind, seq, payload in self.decoder.feed(data):
                    msg = Message(kind, seq, payload, time.perf_counter_ns())
                    with self.cv:
                        self.received_frames += 1
                        self.latest[kind] = msg
                        if len(self.messages) == self.messages.maxlen:
                            self.host_dropped += 1
                        self.messages.append(msg)
                        key = (kind, seq)
                        if key in self.pending:
                            self.pending[key] = msg
                        self.cv.notify_all()
        except Exception as e:
            if not self.stop.is_set():
                with self.cv:
                    self.error = RuntimeError(f"USB 讀取失敗（{type(e).__name__}）：{e}")
                    self.host_events.append((time.perf_counter_ns(), "usb_reader_error", str(self.error)))
                    self.cv.notify_all()

    def _send_locked(self, kind, payload, reply=None):
        seq = self.seq
        self.seq = (seq + 1) & 0xffffffff
        if reply is not None:
            with self.cv:
                self.pending[(reply, seq)] = None
        before = time.perf_counter_ns()
        try:
            raw = encode(kind, seq, payload)
            if self.serial.write(raw) != len(raw):
                raise RuntimeError("USB 封包只寫入一部分，請重新連線")
        except BaseException:
            with self.cv:
                self.pending.pop((reply, seq), None)
            raise
        after = time.perf_counter_ns()
        if self.trace_tx:
            gap_ms = ((before - self.last_set_sent_ns) / 1e6
                      if kind == SET and self.last_set_sent_ns is not None else None)
            with self.cv:
                if len(self.tx_trace) == self.tx_trace.maxlen:
                    self.tx_trace_dropped += 1
                self.tx_trace.append((before, after, kind, seq, payload.hex(), gap_ms))
            if kind == SET:
                self.last_set_sent_ns = before
                if gap_ms is not None:
                    self.max_set_send_gap_ms = max(self.max_set_send_gap_ms, gap_ms)
        return seq

    def send(self, kind, payload=b""):
        if self.error:
            raise RuntimeError(f"Teensy USB 中斷：{self.error}")
        with self.write_lock:
            return self._send_locked(kind, payload)

    def request(self, kind, payload=b"", reply=ACK, timeout=0.5):
        if self.error:
            raise RuntimeError(f"Teensy USB 中斷：{self.error}")
        with self.write_lock:
            seq = self._send_locked(kind, payload, reply)
        key = (reply, seq)
        deadline = time.perf_counter() + timeout
        with self.cv:
            try:
                while self.pending[key] is None:
                    if self.error:
                        raise RuntimeError(f"Teensy USB 中斷：{self.error}")
                    left = deadline - time.perf_counter()
                    if left <= 0:
                        raise TimeoutError(
                            f"{self.port} 等待回覆 0x{reply:02X}（序號 {seq}）逾時；"
                            f"累計收到 {self.received_bytes} bytes、{self.received_frames} 個有效封包。"
                        )
                    self.cv.wait(left)
                msg = self.pending[key]
            finally:
                self.pending.pop(key, None)
        if reply == ACK and (msg.payload[0] != kind or msg.payload[1]):
            raise RuntimeError(f"Teensy 拒絕命令 {kind}，錯誤碼 {msg.payload[1]}")
        return msg

    def ping(self, timeout=0.5):
        token = struct.pack("<Q", time.perf_counter_ns())
        before = time.perf_counter_ns()
        msg = self.request(PING, token, PONG, timeout)
        if msg.payload[4:12] != token:
            raise RuntimeError("Teensy 回覆的 ping 內容錯誤")
        return before, msg.received_ns, struct.unpack_from("<I", msg.payload)[0]

    def configure(self, servos, imu_hz=100, edge=0, i2c_khz=400):
        rows = validate_servos(servos)
        if not 0 <= imu_hz <= 250 or edge not in (0, 1, 2) or i2c_khz not in (100, 400):
            raise ValueError("IMU 頻率 0–250 Hz；邊緣 0/1/2；I²C 100/400 kHz")
        values = [round(r[k] * 10) for k in ("center_us", "min_us", "max_us") for r in rows]
        self.request(CONFIG, struct.pack("<10HBH", *values, int(imu_hz), edge, i2c_khz))
        self.servos = rows

    def set_pulses(self, pulses):
        v = [float(x) for x in pulses]
        if len(v) != 3 or not all(math.isfinite(x) and 500 <= x <= 2500 for x in v):
            raise ValueError("三路脈寬都必須是 500–2500 µs 的有限數字")
        for x, row in zip(v, self.servos):
            if not row["min_us"] <= x <= row["max_us"]:
                raise ValueError(f"脈寬 {x:.1f} 超過這顆伺服的校正範圍")
        return self.send(SET, struct.pack("<3H", *(round(x * 10) for x in v)))

    def home(self):
        return self.request(HOME)

    def off(self):
        return self.request(OFF)

    def get_latest(self, kind, age=0.5):
        with self.cv:
            msg = self.latest.get(kind)
        if msg is None or (time.perf_counter_ns() - msg.received_ns) / 1e9 > age:
            raise RuntimeError("尚未收到資料，或 Teensy 資料已逾時")
        return msg

    def drain(self):
        with self.cv:
            result = list(self.messages)
            self.messages.clear()
        return result

    def note(self, kind, detail):
        with self.cv:
            self.host_events.append((time.perf_counter_ns(), kind, detail))

    def drain_trace(self):
        with self.cv:
            sent, events = list(self.tx_trace), list(self.host_events)
            self.tx_trace.clear()
            self.host_events.clear()
        return sent, events

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.2)
        self.serial.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def default_servos():
    return [{"dir": 1, "center_us": 1500.0, "us_per_deg_up": 2000 / 180,
             "us_per_deg_down": 2000 / 180, "min_us": 500.0, "max_us": 2500.0}
            for _ in range(3)]


def validate_servos(rows):
    if len(rows) != 3:
        raise ValueError("校正檔必須有三顆伺服")
    out = []
    for row in rows:
        r = {**default_servos()[0], **row}
        if not all(math.isfinite(float(r[k])) for k in default_servos()[0]):
            raise ValueError("伺服校正值必須是有限數字")
        if r["dir"] not in (-1, 1) or not 500 <= r["min_us"] < r["max_us"] <= 2500:
            raise ValueError("伺服方向必須是 ±1，範圍必須在 500–2500 µs")
        if not r["min_us"] <= r["center_us"] <= r["max_us"] or min(r["us_per_deg_up"], r["us_per_deg_down"]) <= 0:
            raise ValueError("伺服中位或比例設定錯誤")
        out.append(r)
    return out


def angles_to_pulses(angles_rad, rows):
    deg = np.degrees(angles_rad)
    if np.shape(deg) != (3,) or not np.isfinite(deg).all() or (np.abs(deg) > 60.001).any():
        raise ValueError("伺服角超過 ±60° 或不是有限數字")
    pulses = []
    for angle, r in zip(deg, rows):
        gain = r["us_per_deg_up"] if angle >= 0 else r["us_per_deg_down"]
        us = r["center_us"] + r["dir"] * float(angle) * gain
        if not r["min_us"] <= us <= r["max_us"]:
            raise ValueError("要求的角度超過伺服校正脈寬範圍")
        pulses.append(us)
    return pulses


class ClockMap:
    """Unwrap micros(), then fit the fastest 10% of ping midpoints."""
    def __init__(self):
        self.samples = deque(maxlen=3000)
        self.last_raw = None
        self.unwrapped = 0
        self.origin_ns = None
        self.slope = 1.0
        self.offset = 0.0

    def add(self, before_ns, after_ns, micros):
        if self.last_raw is None:
            self.unwrapped = micros
            self.origin_ns = before_ns
        else:
            step = (micros - self.last_raw) & 0xffffffff
            if step >= 0x80000000:
                raise RuntimeError("Teensy 時鐘倒退，板子可能重新開機，請重新連線")
            self.unwrapped += step
        self.last_raw = micros
        self.samples.append(((before_ns + after_ns) / 2 - self.origin_ns, self.unwrapped * 1000.0, after_ns - before_ns))

    def fit(self):
        a = np.asarray(self.samples, dtype=float)
        if len(a) < 20:
            raise ValueError("對時至少需要 20 個樣本")
        fast = a[:, 2] <= np.quantile(a[:, 2], 0.1)
        if fast.sum() < 2 or np.ptp(a[fast, 0]) < 1e6:
            raise ValueError("對時的樣本時間跨度不足")
        self.slope, self.offset = np.polyfit(a[fast, 0] / 1e9, a[fast, 1] / 1e9, 1)
        res_us = (a[:, 1] / 1e9 - (self.slope * a[:, 0] / 1e9 + self.offset)) * 1e6
        return {"drift_ppm": (self.slope - 1) * 1e6, "fast_residual_std_us": float(res_us[fast].std()),
                "fast_residual_max_us": float(abs(res_us[fast]).max()),
                "residual_p99_us": float(np.quantile(abs(res_us), .99))}

    def host_ns(self, micros):
        if self.last_raw is None:
            raise ValueError("尚未對時")
        delta = ((micros - self.last_raw + 0x80000000) & 0xffffffff) - 0x80000000
        return self.origin_ns + round(((self.unwrapped + delta) / 1e6 - self.offset) / self.slope * 1e9)
