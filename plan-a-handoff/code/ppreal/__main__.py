"""python -m ppreal <指令>：實機工具的入口。"""
import argparse
import sys


def _utf8_stdout():
    # 輸出被導到檔案或別的程式時，Windows 預設用 cp950，中文會變亂碼
    for s in (sys.stdout, sys.stderr):
        if (s.encoding or "").lower().replace("-", "") != "utf8":
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass


def _cam_args(p):
    p.add_argument("--device", default="", help="相機：名字的一部分或編號（預設：Windows 用名字含 PC CAMERA 的那顆）")
    p.add_argument("--fourcc", default="YUYV", choices=["YUYV", "MJPG"])
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--exposure", type=float, default=None,
                   help="手動曝光。Windows：2 的次方秒（-7 約 8 ms）；樹莓派：100 µs 為單位")
    p.add_argument("--any-camera", action="store_true",
                   help="用編號指定、而那顆名字不含 PC CAMERA 時也照開（小心會開到筆電內建鏡頭）")


def main(argv=None) -> int:
    _utf8_stdout()
    from . import camtool

    p = argparse.ArgumentParser(prog="python -m ppreal", description="乒乓球平台的實機工具")
    sub = p.add_subparsers(dest="group", required=True)

    cam = sub.add_parser("cam", help="相機").add_subparsers(dest="cmd", required=True)
    c = cam.add_parser("list", help="列出看得到的相機")
    c.set_defaults(func=camtool.cmd_list)

    c = cam.add_parser("check", help="錄 10 秒，看實際每秒幾張、有沒有掉張")
    _cam_args(c)
    c.add_argument("--seconds", type=float, default=10.0)
    c.add_argument("--settings", action="store_true", help="先打開相機的屬性視窗（Windows，可以手動調曝光）")
    c.add_argument("--note", default="cam check")
    c.add_argument("--no-view", action="store_true")
    c.set_defaults(func=camtool.cmd_check)

    c = cam.add_parser("rec", help="錄影（每張都存、順便找球）")
    _cam_args(c)
    c.add_argument("seconds", type=float)
    c.add_argument("--note", default="")
    c.add_argument("--quality", type=int, default=90)
    c.add_argument("--no-detect", action="store_true")
    c.add_argument("--no-view", action="store_true")
    c.set_defaults(func=camtool.cmd_rec)

    c = cam.add_parser("view", help="只看畫面、不存檔（量視角、架相機、看找球準不準）",
                       description="只看畫面、不存檔：量視角、架相機、看找球準不準時用。按 q 或關掉視窗結束。")
    _cam_args(c)
    c.set_defaults(func=camtool.cmd_view)

    c = cam.add_parser("fov", help="量視角：離牆 D cm 時畫面在牆上寬 W cm（高 H cm）→ 視角和要架多高")
    c.add_argument("--d", type=float, required=True)
    c.add_argument("--w", type=float, required=True)
    c.add_argument("--h", type=float, default=None)
    c.add_argument("--mount", type=float, default=None, help="打算架多高（鏡頭到平台面，cm）")
    c.set_defaults(func=camtool.cmd_fov)

    c = cam.add_parser("calib", help="相機對平台：找平台圓邊、球放中心／正面／右邊定方向")
    _cam_args(c)
    c.add_argument("--check", action="store_true", help="每天開工用：只看相機有沒有被碰歪")
    c.set_defaults(func=camtool.cmd_calib)

    c = sub.add_parser("report", help="讀一次測試的資料夾，出成績、圖、疊圖影片")
    c.add_argument("run_dir")
    c.add_argument("--no-video", action="store_true")
    c.set_defaults(func=lambda a: __import__("ppreal.report", fromlist=["cmd_report"]).cmd_report(a))

    from .hwtool import register
    register(sub)
    from .balance import register as register_balance
    register_balance(sub)

    args = p.parse_args(argv)
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("已停止。")
        return 130
    except (RuntimeError, ValueError, TimeoutError, OSError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
