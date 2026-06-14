from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re
import shutil
import signal
import socket
import subprocess
import sys

GIB = 1024 ** 3


def run(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return 999, f"{type(exc).__name__}: {exc}"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(pid: int, timeout_sec: int = 10) -> bool:
    if not pid_alive(pid):
        return True
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    import time
    for _ in range(timeout_sec * 10):
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    return not pid_alive(pid)


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def disk_free_gib(path: str) -> float:
    return shutil.disk_usage(path).free / GIB


def swap_used_gib() -> float | None:
    if sys.platform == "darwin":
        rc, out = run(["sysctl", "vm.swapusage"], timeout=5)
        if rc != 0:
            return None
        match = re.search(r"used\s*=\s*([0-9.]+)([MGT])", out)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2)
        return value / 1024 if unit == "M" else value if unit == "G" else value * 1024
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        data: dict[str, int] = {}
        for line in meminfo.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].rstrip(":") in {"SwapTotal", "SwapFree"}:
                data[parts[0].rstrip(":")] = int(parts[1])
        if "SwapTotal" in data and "SwapFree" in data:
            return (data["SwapTotal"] - data["SwapFree"]) * 1024 / GIB
    return None


def path_size_bytes(path: str) -> int:
    p = Path(path)
    if not p.exists() and not p.is_symlink():
        return 0
    if p.is_file() or p.is_symlink():
        try:
            return p.lstat().st_size
        except OSError:
            return 0
    total = 0
    for root, dirs, files in os.walk(p, onerror=lambda _e: None):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:
                pass
        for name in dirs:
            fp = Path(root) / name
            if fp.is_symlink():
                try:
                    total += fp.lstat().st_size
                except OSError:
                    pass
    return total


def human_bytes(num: int | float) -> str:
    value = float(num)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"
