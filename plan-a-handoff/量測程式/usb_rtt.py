"""量筆電 ↔ Teensy 的 USB 來回延遲和對時誤差（暫存量測）。先把 latency_test.ino 燒進 Teensy 4.1。

用法：python usb_rtt.py [COM 埠]　不給就自動找 Teensy（USB VID 16C0）。照控制迴圈的節奏每 2 ms 送一包，送 10 秒。
"""
import ctypes
import struct
import sys
import time

import numpy as np
import serial
from serial.tools import list_ports

N = 5000
HZ = 500.0


def find_port():
    for p in list_ports.comports():
        if p.vid == 0x16C0:
            return p.device
    sys.exit('找不到 Teensy（USB VID 16C0），插上了嗎？')


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_port()
    k32 = ctypes.WinDLL('kernel32')                           # 跟實際控制迴圈一樣用高優先權；64 位元要先設型別，不然會靜靜失敗
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    k32.GetPriorityClass.argtypes = (ctypes.c_void_p,)
    k32.GetPriorityClass.restype = ctypes.c_uint32
    h = k32.GetCurrentProcess()
    if not k32.SetPriorityClass(h, 0x80) or k32.GetPriorityClass(h) != 0x80:
        sys.exit('設不到高優先權')
    s = serial.Serial(port, timeout=0.5)
    time.sleep(0.3)
    s.reset_input_buffer()
    rtt, mid, tt, lost = [], [], [], 0
    t0 = time.perf_counter() + 0.05
    for k in range(N):
        target = t0 + k / HZ
        while time.perf_counter() < target:
            pass
        a = time.perf_counter()
        s.write(struct.pack('<IIII', k, 0, 0, 0))
        r = s.read(20)
        b = time.perf_counter()
        if len(r) != 20 or struct.unpack('<I', r[:4])[0] != k:
            lost += 1
            s.reset_input_buffer()
            continue
        rtt.append(b - a)
        mid.append((a + b) / 2)
        tt.append(struct.unpack('<I', r[16:20])[0])
    s.close()

    rtt = np.array(rtt) * 1e3
    q = lambda p: np.quantile(rtt, p)
    print(f'{port}：{len(rtt)} 包成功、{lost} 包掉或錯位')
    print(f'來回延遲 ms：中位 {q(0.5):.3f}  p99 {q(0.99):.3f}  p99.9 {q(0.999):.3f}  最大 {rtt.max():.3f}  '
          f'超過 1 ms {int((rtt > 1).sum())} 次、超過 2 ms {int((rtt > 2).sum())} 次')

    # 對時：Teensy 的 micros() 對筆電 perf_counter 做直線擬合，只用來回最快的 10%（延遲最對稱、誤差最小）
    tt = np.unwrap(np.array(tt, dtype=float) * 1e-6, period=2 ** 32 * 1e-6)
    mid = np.array(mid)
    fast = rtt <= np.quantile(rtt, 0.10)
    b1, b0 = np.polyfit(mid[fast], tt[fast], 1)
    res = (tt - (b0 + b1 * mid)) * 1e6
    print(f'時鐘漂移 {(b1 - 1) * 1e6:+.1f} ppm；最快 10% 的殘差 µs：標準差 {res[fast].std():.1f}、最大 {abs(res[fast]).max():.1f}')
    print(f'全部樣本的殘差 µs：標準差 {res.std():.1f}、p99 {np.quantile(abs(res), 0.99):.1f}'
          f'（每個樣本的真實誤差最多是它的來回延遲的一半）')


if __name__ == '__main__':
    main()
