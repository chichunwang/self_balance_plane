"""相機：Windows 用 DSHOW、樹莓派用 V4L2，另外有測試用的假相機。三個都是 read() 回一張 Frame。

找相機時只開名字對得上的那一顆，對不上就報錯、列出清單，絕不改開別顆（筆電內建鏡頭會拍到人）。
"""
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .runlog import now_ns


class CameraNotFound(Exception):
    pass


@dataclass
class CamConfig:
    device: str = ""            # 空＝預設：Windows 名字含 PC CAMERA、Linux /dev/video0；也可以給編號或名字的一部分
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fourcc: str = "YUYV"        # YUYV 或 MJPG
    exposure: float | None = None   # Windows：2 的次方秒（-7 約 8 ms）；Linux：100 µs 為單位
    allow_any: bool = False     # 用編號指定時，名字不含 PC CAMERA 也照開（--any-camera）


@dataclass
class Frame:
    index: int
    gray: np.ndarray
    jpeg: bytes | None          # 相機本來就給 JPEG 時的原始位元組
    t_cap_ns: int | None        # V4L2 的時間戳（收到這張第一個封包的時刻）；Windows 沒有
    t_read_ns: int              # 程式拿到這張的時刻


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def default_device() -> str:
    return "/dev/video0" if _is_linux() else "PC CAMERA"


def parse_dshow_list(text: str) -> list[str]:
    """ffmpeg -list_devices 的輸出 → 影像裝置名稱，順序就是 DSHOW 的編號。"""
    names, section = [], None
    for line in text.splitlines():
        if "Alternative name" in line:
            continue
        low = line.lower()
        if "directshow video devices" in low:
            section = "video"
            continue
        if "directshow audio devices" in low:
            section = "audio"
            continue
        m = re.search(r'\]\s+"(.+)"\s*(?:\((\w+)\))?\s*$', line)
        if not m:
            continue
        kind = m.group(2)
        if kind == "audio" or (kind is None and section == "audio"):
            break                   # 影像裝置都列在聲音裝置前面
        names.append(m.group(1))
    return names


def list_devices() -> list[str]:
    if _is_linux():
        out = []
        for p in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:] or 0)):
            try:
                name = (p / "name").read_text().strip()
            except OSError:
                name = ""
            out.append(f"/dev/{p.name} {name}".strip())
        return out
    exe = shutil.which("ffmpeg")
    if not exe:
        return []
    r = subprocess.run([exe, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    return parse_dshow_list(r.stderr + r.stdout)


def _listing(names: list[str]) -> str:
    if not names:
        return "  （一個都沒看到；Windows 要裝 ffmpeg 才列得出名字，不然用編號指定）"
    return "\n".join(f"  {i}: {n}" for i, n in enumerate(names))


def resolve_device(spec: str, names: list[str]) -> int | str:
    spec = spec.strip()
    if spec.isdigit():
        return int(spec)
    if spec.startswith("/dev/"):
        return spec
    hits = [i for i, n in enumerate(names) if spec.lower() in n.lower()]
    if len(hits) == 1:
        name = names[hits[0]]
        return name.split()[0] if name.startswith("/dev/") else hits[0]
    if not hits:
        raise CameraNotFound(f"找不到名字含「{spec}」的相機。看得到的有：\n{_listing(names)}\n"
                             "把相機插好再試（拔掉重插、換一個 USB 孔）。")
    raise CameraNotFound(f"名字含「{spec}」的相機有好幾顆：\n{_listing(names)}\n用 --device 打更完整的名字。")


def guard_device(spec: str, dev, names: list[str], expected: str = "PC CAMERA", allow_any: bool = False):
    """用編號指定時，確認那顆的名字是我們要的相機，免得開到筆電內建鏡頭把人錄下來。打名字的不擋。"""
    if not spec.strip().isdigit() or allow_any:
        return
    name = names[dev] if isinstance(dev, int) and dev < len(names) else "（不知道名字）"
    if expected.lower() not in name.lower():
        raise CameraNotFound(f"第 {dev} 顆是「{name}」，不是 {expected}。為了不要開到筆電內建的鏡頭，"
                             f"用編號時只開名字含 {expected} 的那顆；確定要開這顆的話加 --any-camera。")


def decode_fourcc(v: float) -> str:
    n = int(v)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")


class Camera:
    def __init__(self, cfg: CamConfig, clock: Callable[[], int] = now_ns):
        self.cfg, self.clock = cfg, clock
        self.linux = _is_linux()
        spec = cfg.device or default_device()
        names = list_devices()
        dev = resolve_device(spec, names)
        if not self.linux:
            guard_device(spec, dev, names, allow_any=cfg.allow_any)
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2 if self.linux else cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise CameraNotFound(f"打不開相機「{spec}」（{dev}）。可能被別的程式開著（Windows 的「相機」App、Teams、"
                                 f"瀏覽器的視訊），關掉再試。看得到的有：\n{_listing(names)}")
        self._setup()
        self.index = 0
        c = self.cap
        self.settings = {
            "device": spec, "opened": dev,
            "name": names[dev] if isinstance(dev, int) and dev < len(names) else str(dev),
            "backend": c.getBackendName(),
            "requested": {"width": cfg.width, "height": cfg.height, "fps": cfg.fps,
                          "fourcc": cfg.fourcc, "exposure": cfg.exposure},
            "width": int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fourcc": decode_fourcc(c.get(cv2.CAP_PROP_FOURCC)),
            "fps_reported": c.get(cv2.CAP_PROP_FPS),       # DSHOW 讀回的只是要求的值，實際要自己數
            "exposure": c.get(cv2.CAP_PROP_EXPOSURE),
        }

    def _setup(self):
        c, cfg = self.cap, self.cfg
        if self.linux:
            # V4L2：先格式和大小、最後每秒張數；不轉彩色，拿原始 YUYV／JPEG
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            c.set(cv2.CAP_PROP_FPS, cfg.fps)
            c.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        else:
            # DSHOW：設寬高或張數會重開相機，格式要最後設（README「硬體接法」）
            c.set(cv2.CAP_PROP_FPS, cfg.fps)
            c.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*("YUY2" if cfg.fourcc == "YUYV" else cfg.fourcc)))
        if cfg.exposure is not None:
            c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if self.linux else 0.25)   # 兩邊「手動」的值不一樣
            c.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)

    def read(self) -> Frame | None:
        ok, img = self.cap.read()
        t = self.clock()
        if not ok or img is None:
            return None
        jpeg, t_cap = None, None
        if self.linux:
            ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            t_cap = int(ms * 1e6) if ms > 0 else None
            if img.ndim == 2 and img.shape[0] == 1:            # MJPEG：1×N 的壓縮位元組
                jpeg = img.tobytes()
                gray = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    return None
            elif img.ndim == 3 and img.shape[2] == 2:          # YUYV：第 0 層就是亮度
                gray = np.ascontiguousarray(img[:, :, 0])
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        f = Frame(self.index, gray, jpeg, t_cap, t)
        self.index += 1
        return f

    def open_settings_dialog(self):
        if not self.linux:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def close(self):
        self.cap.release()


class FakeCamera:
    """測試用：make_frame(k) 給第 k 張；時間用假時鐘（建立當下＋k/fps 秒），不會真的等。
    起點用真的時鐘，報告算「開始後幾秒」才對得上紀錄的開始時間。"""

    def __init__(self, make_frame: Callable[[int], np.ndarray], fps: float = 30.0, n: int | None = None,
                 fail_after: int | None = None, jpeg: bool = False, t0_ns: int | None = None):
        self.make_frame, self.fps, self.n, self.fail_after, self.jpeg = make_frame, fps, n, fail_after, jpeg
        self.t0_ns = now_ns() if t0_ns is None else t0_ns
        self.index = 0
        h, w = make_frame(0).shape[:2]
        self.settings = {"device": "fake", "opened": "fake", "name": "fake", "backend": "fake",
                         "requested": {"width": w, "height": h, "fps": fps, "fourcc": "MJPG" if jpeg else "GREY",
                                       "exposure": None},
                         "width": w, "height": h, "fourcc": "MJPG" if jpeg else "GREY", "fps_reported": fps,
                         "exposure": None}

    def read(self) -> Frame | None:
        k = self.index
        if (self.n is not None and k >= self.n) or (self.fail_after is not None and k >= self.fail_after):
            return None
        gray = self.make_frame(k)
        jpeg = cv2.imencode(".jpg", gray)[1].tobytes() if self.jpeg else None
        self.index += 1
        return Frame(k, gray, jpeg, None, self.t0_ns + int(k * 1e9 / self.fps))

    def open_settings_dialog(self):
        pass

    def close(self):
        pass
