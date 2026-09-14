from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import LimitConfig


@dataclass(frozen=True)
class Mount:
    source: Path
    target: str
    read_only: bool = True


@dataclass(frozen=True)
class ExecutionResult:
    elapsed_seconds: float
    returncode: int
    stdout: str
    stderr: str


class DockerRunner:
    def __init__(self, limits: LimitConfig):
        self.limits = limits

    def check(self) -> None:
        subprocess.run(["docker", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self, image: str, command: tuple[str, ...], mounts: list[Mount], env: dict[str, str], timeout_seconds: float) -> ExecutionResult:
        name = f"ncpa-verifier-{uuid.uuid4().hex[:12]}"
        argv = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--pids-limit", str(self.limits.pids_limit),
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=1g",
        ]
        if self.limits.cpus is not None:
            argv += ["--cpus", str(self.limits.cpus)]
        if self.limits.memory:
            argv += ["--memory", self.limits.memory]
        if self.limits.gpus:
            argv += ["--gpus", self.limits.gpus]

        for mount in mounts:
            source = mount.source.resolve()
            if not source.exists():
                raise FileNotFoundError(source)
            spec = f"type=bind,src={source},dst={mount.target}"
            if mount.read_only:
                spec += ",readonly"
            argv += ["--mount", spec]

        clean_env = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0"}
        clean_env.update(env)
        for key, value in sorted(clean_env.items()):
            argv += ["--env", f"{key}={value}"]

        argv += [image, *command]
        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                text=True,
                capture_output=True,
                timeout=max(1.0, timeout_seconds),
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raise TimeoutError(f"container exceeded {timeout_seconds:.1f}s")
        return ExecutionResult(time.monotonic() - started, proc.returncode, proc.stdout, proc.stderr)
