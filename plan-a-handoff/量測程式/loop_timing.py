"""這台筆電上 Python 能不能準時跑 500 Hz 的迴圈（暫存的量測腳本）。

每一拍先空轉 0.6 ms 假裝在算控制（這台筆電上控制＋估測約吃 30% CPU，500 Hz 時每拍約 0.6 ms），
再等到下一拍。記每一拍實際開始的時間比排定的晚多少。
"""
import ctypes
import statistics
import sys
import time

HZ = 500.0
DT = 1.0 / HZ
WORK = 0.0006
SECONDS = 15.0


def high_priority():
    """把這個程式設成高優先權。64 位元要先設好 ctypes 的型別，不然 SetPriorityClass 會靜靜失敗、還是一般優先權。"""
    k32 = ctypes.WinDLL('kernel32')
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    k32.GetPriorityClass.argtypes = (ctypes.c_void_p,)
    k32.GetPriorityClass.restype = ctypes.c_uint32
    h = k32.GetCurrentProcess()
    if not k32.SetPriorityClass(h, 0x80) or k32.GetPriorityClass(h) != 0x80:
        raise OSError('設不到高優先權')


def busy(t_end):
    while time.perf_counter() < t_end:
        pass


def run(mode):
    n = int(SECONDS * HZ)
    late = []
    t0 = time.perf_counter() + 0.05
    for k in range(n):
        target = t0 + k * DT
        now = time.perf_counter()
        if mode == 'sleep':
            if target > now:
                time.sleep(target - now)
        elif mode == 'hybrid':                    # 睡到前 1 ms，剩下空轉
            if target - now > 0.001:
                time.sleep(target - now - 0.001)
            busy(target)
        elif mode == 'spin':
            busy(target)
        start = time.perf_counter()
        late.append((start - target) * 1e3)
        busy(start + WORK)
    late.sort()
    q = lambda p: late[min(len(late) - 1, int(p * len(late)))]
    over = sum(1 for x in late if x > 2.0)
    return (f'{mode:7s} 中位 {q(0.5):6.3f} ms  p99 {q(0.99):6.3f}  p99.9 {q(0.999):6.3f}  最大 {late[-1]:7.3f}  '
            f'晚超過一拍(2 ms) {over} 次／{n}')


def sleep_res():
    xs = []
    for _ in range(300):
        a = time.perf_counter()
        time.sleep(0.001)
        xs.append((time.perf_counter() - a) * 1e3)
    xs.sort()
    return f'time.sleep(0.001) 實際：中位 {statistics.median(xs):.3f} ms、p99 {xs[int(0.99 * len(xs))]:.3f}、最大 {xs[-1]:.3f}'


if __name__ == '__main__':
    print(sys.version.split()[0], sys.platform)
    print(sleep_res())
    for m in ('sleep', 'hybrid', 'spin'):
        print(run(m))
    winmm = ctypes.WinDLL('winmm')
    winmm.timeBeginPeriod(1)                   # 把 Windows 的計時器解析度調到 1 ms 再量一次 sleep 版
    try:
        print('timeBeginPeriod(1) 之後：')
        print(sleep_res())
        print(run('sleep'))
    finally:
        winmm.timeEndPeriod(1)
