"""高優先權＋空轉的 500 Hz 迴圈，旁邊同時有「影像處理」在跑：放同一個行程的另一條執行緒 vs. 放另一個行程。暫存量測。

2026-10-26 第一次跑的版本沒設好 ctypes 型別，高優先權沒生效，那組數字是一般優先權下量的。
"""
import multiprocessing as mp
import threading
import time

import loop_timing as L


def vision(stop_at):
    """假裝在找球：純 Python 運算，一直算到時間到。"""
    x = 0
    while time.perf_counter() < stop_at:
        for i in range(2000):
            x += i * i


def vision_proc(seconds):
    vision(time.perf_counter() + seconds)


if __name__ == '__main__':
    L.high_priority()
    t = threading.Thread(target=vision, args=(time.perf_counter() + L.SECONDS + 1,), daemon=True)
    t.start()
    print('同一個行程（影像在另一條執行緒）', L.run('spin'))
    t.join()
    p = mp.Process(target=vision_proc, args=(L.SECONDS + 1,))
    p.start()
    print('分開的行程（影像在另一個行程）  ', L.run('spin'))
    p.join()
