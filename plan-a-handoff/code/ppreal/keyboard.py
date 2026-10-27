"""Windows console polling through ctypes, which releases the Python GIL."""
import ctypes
import sys

_crt = None


def poll_key():
    global _crt
    if sys.platform == "win32":
        if _crt is None:
            _crt = ctypes.CDLL("ucrtbase.dll")
            _crt._kbhit.argtypes = ()
            _crt._kbhit.restype = ctypes.c_int
            _crt._getwch.argtypes = ()
            _crt._getwch.restype = ctypes.c_uint16
        if not _crt._kbhit():
            return ""
        c = _crt._getwch()
        if c in (0, 224):
            _crt._getwch()
            return ""
        return chr(c)
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().rstrip("\n") or "\n"
    return ""
