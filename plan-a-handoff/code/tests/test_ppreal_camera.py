import cv2
import numpy as np
import pytest

from ppreal.camera import CameraNotFound, FakeCamera, guard_device, parse_dshow_list, resolve_device

# 2026-10-27 在這台筆電上跑 ffmpeg -list_devices 的實際輸出
DSHOW_OUT = r"""[in#0 @ 0000024eb6a26b80] "USB2.0 HD UVC WebCam" (video)
[in#0 @ 0000024eb6a26b80]   Alternative name "@device_pnp_\\?\usb#vid_13d3&pid_56a2&mi_00#7&c91c3a9&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
[in#0 @ 0000024eb6a26b80] "OBS Virtual Camera" (none)
[in#0 @ 0000024eb6a26b80]   Alternative name "@device_sw_{860BB310-5D01-11D0-BD3B-00A0C911CE86}\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"
[in#0 @ 0000024eb6a26b80] "Microphone Array (Realtek(R) Audio)" (audio)
[in#0 @ 0000024eb6a26b80]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{BAC7A0E0-8509-402A-8F6C-FCE669DDF9B3}"
[in#0 @ 0000024eb6a26b80] "Stereo Mix (Realtek(R) Audio)" (audio)
[in#0 @ 0000024eb6a26b80]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{26EC3154-73B9-4BFA-B438-68A1261D8754}"
"""

# 舊版 ffmpeg 的格式：有標題、沒有 (video) 標記
DSHOW_OLD = """[dshow @ 000001] DirectShow video devices (some may be both video and audio devices)
[dshow @ 000001]  "USB2.0 PC CAMERA"
[dshow @ 000001]     Alternative name "@device_pnp_xxx"
[dshow @ 000001] DirectShow audio devices
[dshow @ 000001]  "Microphone (USB2.0 PC CAMERA)"
"""


def test_parse_dshow_list():
    assert parse_dshow_list(DSHOW_OUT) == ["USB2.0 HD UVC WebCam", "OBS Virtual Camera"]


def test_parse_dshow_list_old_format():
    assert parse_dshow_list(DSHOW_OLD) == ["USB2.0 PC CAMERA"]


def test_resolve_device_by_name_and_index():
    names = ["USB2.0 HD UVC WebCam", "USB2.0 PC CAMERA"]
    assert resolve_device("pc camera", names) == 1
    assert resolve_device("0", names) == 0
    assert resolve_device("/dev/video2", names) == "/dev/video2"


def test_resolve_device_linux_names_map_to_path():
    names = ["/dev/video0 USB2.0 PC CAMERA", "/dev/video2 Waveshare OV9281"]
    assert resolve_device("ov9281", names) == "/dev/video2"


def test_resolve_device_never_falls_back():
    names = ["USB2.0 HD UVC WebCam", "OBS Virtual Camera"]
    with pytest.raises(CameraNotFound) as e:
        resolve_device("PC CAMERA", names)
    msg = str(e.value)
    assert "USB2.0 HD UVC WebCam" in msg and "OBS Virtual Camera" in msg and "插好" in msg
    assert "--device" not in msg            # 不要把人帶去用編號開到內建鏡頭
    with pytest.raises(CameraNotFound):
        resolve_device("USB2.0", ["USB2.0 HD UVC WebCam", "USB2.0 PC CAMERA"])


def test_numeric_device_refuses_other_cameras():
    names = ["USB2.0 HD UVC WebCam", "USB2.0 PC CAMERA"]
    with pytest.raises(CameraNotFound) as e:
        guard_device("0", 0, names)
    assert "USB2.0 HD UVC WebCam" in str(e.value) and "--any-camera" in str(e.value)
    guard_device("1", 1, names)                     # 名字對得上就放行
    guard_device("0", 0, names, allow_any=True)     # 使用者明講要開別顆才放行
    guard_device("webcam", 0, names)                # 自己打名字的不擋


def frame_k(k):
    img = np.zeros((48, 64), np.uint8)
    img[k % 48, :] = 255
    return img


def test_fake_camera_clock_and_end():
    cam = FakeCamera(frame_k, fps=30, n=3, t0_ns=1000)
    frames = [cam.read() for _ in range(3)]
    assert [f.t_read_ns for f in frames] == [1000, 1000 + 33333333, 1000 + 66666666]
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[1].gray[1, 0] == 255
    assert cam.read() is None
    assert cam.settings["width"] == 64 and cam.settings["height"] == 48


def test_fake_camera_fail_after():
    cam = FakeCamera(frame_k, fail_after=2)
    assert cam.read() is not None and cam.read() is not None
    assert cam.read() is None and cam.read() is None


def test_fake_camera_jpeg_mode():
    f = FakeCamera(frame_k, jpeg=True).read()
    back = cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert back.shape == (48, 64)
