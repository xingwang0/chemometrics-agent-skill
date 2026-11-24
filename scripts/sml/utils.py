# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""通用工具：路径/编码/日志/计时/内存守卫/JSON 读写。

Windows + 中文路径是本机常态，因此这里统一约定：
* 所有文件读写显式 encoding="utf-8"
* 启动时 reconfigure stdout/stderr，避免 cmd.exe 下 UnicodeEncodeError
* 不用 pandas 的默认路径拼接，一律 pathlib.Path
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "setup_console",
    "as_path",
    "ensure_dir",
    "read_json",
    "write_json",
    "write_text",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "info",
    "warn",
    "err",
    "banner",
    "Timer",
    "timed",
    "available_memory_gb",
    "mem_guard",
    "non_ascii",
    "configure_matplotlib",
    "suppress_warnings",
    "quiet_library_warnings",
]


# --------------------------------------------------------------------------- #
# 控制台
# --------------------------------------------------------------------------- #
def setup_console() -> None:
    """让 stdout/stderr 在 cmd/PowerShell/Git-Bash 下都能输出中文。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconf = getattr(stream, "reconfigure", None)
        if reconf is not None:
            try:
                reconf(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_VERBOSE = True
_WARN_SEEN: dict[str, int] = {}
_WARN_LIMIT = 3          # 同一条告警最多打印 3 次，其余折叠成"另有 N 次"


def set_verbose(flag: bool) -> None:
    global _VERBOSE
    _VERBOSE = bool(flag)


def info(msg: str) -> None:
    if _VERBOSE:
        print(f"[sml] {msg}", flush=True)


def warn(msg: str) -> None:
    """去重告警。

    嵌套 CV 里 ``SafePLS`` 的 n_components 裁剪、``PCR`` 的越界提示会在每折每候选
    各触发一次；2000 列光谱上「每折几十个候选 × 25 折」就是上千行 stderr——既拖慢
    输出（Windows 控制台同步写），也把真正需要注意的告警淹掉。
    同一条消息前 3 次正常打印，之后只累计计数，结束时不补打。
    """
    seen = _WARN_SEEN.get(msg, 0) + 1
    _WARN_SEEN[msg] = seen
    if seen <= _WARN_LIMIT:
        print(f"[sml][warn] {msg}", file=sys.stderr, flush=True)
    elif seen == _WARN_LIMIT + 1:
        print(f"[sml][warn] {msg}（同类告警已重复出现，后续仅在总结里给出次数）",
              file=sys.stderr, flush=True)


def warn_summary() -> None:
    """打印被折叠的重复告警次数（各 stage 结束时调用一次）。"""
    extra = {m: n for m, n in _WARN_SEEN.items() if n > _WARN_LIMIT}
    if not extra:
        return
    total = sum(n - _WARN_LIMIT for n in extra.values())
    print(f"[sml][warn] 另有 {total} 条重复告警已折叠（{len(extra)} 类）", file=sys.stderr, flush=True)


def err(msg: str) -> None:
    print(f"[sml][error] {msg}", file=sys.stderr, flush=True)


def banner(title: str) -> None:
    if not _VERBOSE:
        return
    line = "=" * max(24, min(78, len(title) + 8))
    print(f"\n{line}\n  {title}\n{line}", flush=True)


# --------------------------------------------------------------------------- #
# 路径与 IO
# --------------------------------------------------------------------------- #
def as_path(p: str | os.PathLike) -> Path:
    """接受任意分隔符的字符串路径，返回绝对 Path（不解析符号链接）。"""
    path = Path(str(p).replace("\\", "/")).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def ensure_dir(p: str | os.PathLike) -> Path:
    path = as_path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(p: str | os.PathLike) -> Any:
    with open(as_path(p), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return repr(obj)


def write_json(p: str | os.PathLike, obj: Any, indent: int = 2) -> Path:
    path = as_path(p)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, default=_json_default)
        fh.write("\n")
    return path


def write_text(p: str | os.PathLike, text: str) -> Path:
    path = as_path(p)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------------- #
# 哈希（复现锁定用）
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(p: str | os.PathLike, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(as_path(p), "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.laps: list[tuple[str, float]] = []

    def lap(self, label: str) -> float:
        dt = time.perf_counter() - self.t0
        self.laps.append((label, dt))
        return dt

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        info(f"{func.__name__} 完成，耗时 {time.perf_counter() - t0:.1f}s")
        return out

    return wrapper


def human_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    if seconds < 3600:
        return f"{seconds / 60:.1f} 分钟"
    return f"{seconds / 3600:.2f} 小时"


# --------------------------------------------------------------------------- #
# 内存
# --------------------------------------------------------------------------- #
def available_memory_gb() -> float | None:
    """可用物理内存（GB）。优先 psutil，其次 Windows ctypes，失败返回 None。"""
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullAvailPhys / (1024**3)
        except Exception:
            return None
    return None


@contextmanager
def mem_guard(need_gb: float, label: str = "任务"):
    """在启动重活前检查可用内存，不足则拒绝运行（比 OOM 崩在半夜好）。"""
    avail = available_memory_gb()
    if avail is not None and avail < need_gb:
        raise MemoryError(
            f"{label} 预计需要约 {need_gb:.1f} GB 可用内存，当前仅 {avail:.1f} GB。"
            f" 请降低 repeats / n_iter / bootstrap 次数，或改用 --budget fast。"
        )
    yield avail


def non_ascii(text: str) -> bool:
    try:
        str(text).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


# --------------------------------------------------------------------------- #
# matplotlib 统一配置（中文 + Agg）
# --------------------------------------------------------------------------- #
CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "DengXian",
    "SimSun",
]


def configure_matplotlib(dpi: int = 300, style_path: str | os.PathLike | None = None):
    """返回 pyplot。统一 Agg 后端、中文字体、负号显示。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [name for name in CJK_FONT_CANDIDATES if name in available]
    base = ["DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = picked + base
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["savefig.bbox"] = "tight"
    if style_path is not None and as_path(style_path).exists():
        try:
            plt.style.use(str(as_path(style_path)))
        except Exception:
            warn(f"样式文件加载失败，已忽略：{style_path}")
    return plt


@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def quiet_library_warnings() -> None:
    """``--quiet`` 时压制第三方库的刷屏警告（GP 收敛、sklearn 弃用等）。

    只压库级 ``warnings``，本技能自己的 ``[sml][warn]`` 仍会输出——那些是结论
    相关提示，不能被 --quiet 一起吞掉。
    """
    try:
        from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
    except Exception:  # noqa: BLE001
        ConvergenceWarning = UserWarning  # type: ignore[assignment,misc]
        UndefinedMetricWarning = UserWarning  # type: ignore[assignment,misc]
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")
    warnings.filterwarnings("ignore", message=r".*does not have valid feature names.*")


def flatten(items: Iterable[Iterable]) -> list:
    out = []
    for it in items:
        out.extend(it)
    return out
