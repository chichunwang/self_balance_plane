"""Offline protocol/logging checks. Never opens a serial port."""
from collections import deque
import binascii
import struct
import threading
import unittest

from ppreal.hwtool import USBLog
from ppreal.teensy import Decoder, Message, Teensy, STATUS, DIAG, FW_EVENT, SET, encode


class Writer:
    def __init__(self):
        self.rows = []

    def row(self, *values):
        self.rows.append(values)


class MemoryRun:
    def __init__(self):
        self.files, self.metadata = {}, {}

    def csv(self, name, columns):
        self.files[name] = Writer()
        return self.files[name]

    def update(self, **fields):
        self.metadata.update(fields)


class MemoryLink:
    def __init__(self, messages):
        self.messages = messages
        self.max_set_send_gap_ms = 210.
        self.tx_trace_dropped = 0
        self.trace_tx = False

    def drain(self):
        messages, self.messages = self.messages, []
        return messages

    def drain_trace(self):
        if getattr(self, "traced", False):
            return [], []
        self.traced = True
        return [(1000, 2000, SET, 42, "983a983a983a", 210.)], [(1500, "user_twitch", "標記")]


def status(flags, stamp, seq):
    return Message(STATUS, seq, struct.pack("<II3HBBIII", stamp, seq, 15000, 15000, 15000,
                                          flags, 0x68, 0, 0, 0), stamp * 1000)


class DebugTests(unittest.TestCase):
    def test_firmware_event_uses_fixed_wire_offsets(self):
        # Use the firmware offsets, independently of encode()/Decoder's struct.
        raw = bytearray(40)
        raw[:4] = b"\xaa\x55\x01\x87"
        raw[4:8] = (19).to_bytes(4, "little")
        raw[8:12] = (123456).to_bytes(4, "little")
        raw[12:14] = bytes((3, 27))
        raw[16:20] = (42).to_bytes(4, "little")
        raw[20:24] = (210000).to_bytes(4, "little")
        for offset in (24, 26, 28):
            raw[offset:offset+2] = (15000).to_bytes(2, "little")
        raw[32:36] = (7).to_bytes(4, "little")
        raw[38:] = binascii.crc_hqx(raw[:38], 0xffff).to_bytes(2, "little")
        decoder = Decoder()
        self.assertEqual(decoder.feed(raw[:17]), [])
        kind, seq, payload = decoder.feed(raw[17:])[0]
        event = Message(kind, seq, payload, 0).firmware_event()
        self.assertEqual(event["event"], "watchdog_trip")
        self.assertEqual(event["command_age_us"], 210000)
        self.assertEqual(event["last_seq"], 42)
        self.assertEqual(event["event_number"], 7)
        self.assertEqual(event["pulses_us"], [1500., 1500., 1500.])

    def test_log_correlates_host_gap_and_device_watchdog(self):
        event = Message(FW_EVENT, 10, struct.pack("<IBBHII3H2xI", 210000, 3, 27, 0, 42,
                                                210000, 15000, 15000, 15000, 7), 210000000)
        diag = Message(DIAG, 11, struct.pack("<7I", 1000, 100, 1, 210000, 3, 0, 1), 220000000)
        link = MemoryLink([status(9, 10000, 42), status(27, 210000, 42), event, diag,
                           status(9, 220000, 43)])
        run = MemoryRun()
        log = USBLog(link, run)
        log.close()
        summary = run.metadata["hold_debug"]
        self.assertEqual(summary["watchdog_events"], 1)
        self.assertEqual(summary["homing_entries"], 1)
        self.assertEqual(summary["host_gaps_over_200ms"], 1)
        self.assertTrue(summary["firmware_diagnostics"])
        self.assertEqual(len(run.files["status.csv"].rows), 3)
        names = [r[2] for r in run.files["events_teensy.csv"].rows]
        self.assertIn("host_set_gap", names)
        self.assertIn("watchdog_trip", names)
        self.assertIn("user_twitch", names)
        self.assertFalse(link.trace_tx)

    def test_legacy_firmware_still_logs_without_diagnostic_frames(self):
        run = MemoryRun()
        log = USBLog(MemoryLink([status(9, 10000, 1), status(27, 210000, 1)]), run)
        log.close()
        self.assertFalse(run.metadata["hold_debug"]["firmware_diagnostics"])
        self.assertEqual(run.metadata["hold_debug"]["homing_entries"], 1)

    def test_transmit_trace_preserves_original_wire_frame(self):
        class Port:
            def write(self, raw):
                self.raw = raw
                return len(raw)
        link = Teensy.__new__(Teensy)
        link.serial, link.seq, link.pending = Port(), 42, {}
        link.cv, link.trace_tx = threading.Condition(), True
        link.tx_trace, link.tx_trace_dropped = deque(maxlen=10), 0
        link.last_set_sent_ns, link.max_set_send_gap_ms = None, 0.
        payload = struct.pack("<3H", 15000, 15000, 15000)
        self.assertEqual(link._send_locked(SET, payload), 42)
        self.assertEqual(link.serial.raw, encode(SET, 42, payload))
        self.assertEqual(len(link.serial.raw), 40)
        self.assertEqual(len(link.tx_trace), 1)


if __name__ == "__main__":
    unittest.main()
