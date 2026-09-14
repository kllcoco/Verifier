from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong),
                ]

            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return None


def _gpus() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    result = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            memory_mb = int(parts[1])
        except ValueError:
            memory_mb = None
        result.append({"name": parts[0], "memory_mb": memory_mb, "driver": parts[2]})
    return result


def _docker_version() -> str | None:
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        return proc.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def load_model_requirements(baseline: Path) -> dict[str, Any]:
    roots = [baseline] if baseline.is_dir() else [baseline.parent]
    for root in roots:
        for name in ("verifier_model.json", "model_requirements.json"):
            path = root / name
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError(f"{path} must contain an object")
                return data
    return {}


def probe_host(workspace: Path, baseline: Path | None = None) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(workspace.resolve())
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "disk_free_bytes": disk.free,
        "gpus": _gpus(),
        "docker_version": _docker_version(),
    }
    if baseline is not None:
        info["model_requirements"] = load_model_requirements(baseline)
    return info


def readiness(info: dict[str, Any], require_gpu: bool) -> dict[str, Any]:
    issues: list[str] = []
    if not info.get("docker_version"):
        issues.append("Docker is not available")
    gpus = info.get("gpus") or []
    if require_gpu and not gpus:
        issues.append("NVIDIA GPU is not visible")
    req = info.get("model_requirements") or {}
    if req:
        min_ram_mb = req.get("minimum_ram_mb")
        if min_ram_mb and info.get("memory_bytes") and info["memory_bytes"] < int(min_ram_mb) * 1024 * 1024:
            issues.append("system RAM is below model minimum")
        min_disk_mb = req.get("minimum_free_disk_mb")
        if min_disk_mb and info.get("disk_free_bytes") and info["disk_free_bytes"] < int(min_disk_mb) * 1024 * 1024:
            issues.append("free disk is below model minimum")
        min_vram_mb = req.get("minimum_training_vram_mb")
        if min_vram_mb and require_gpu:
            max_vram = max((gpu.get("memory_mb") or 0 for gpu in gpus), default=0)
            if max_vram < int(min_vram_mb):
                issues.append("GPU VRAM is below model training minimum")
    return {"ready": not issues, "issues": issues}
