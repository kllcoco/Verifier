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
    stdout_path: Path
    stderr_path: Path
    image_id: str


class DockerRunner:
    def __init__(self, limits: LimitConfig):
        self.limits = limits

    def check(self) -> str:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=True,
            text=True,
            capture_output=True,
            env={"PATH": os.environ.get("PATH", "")},
        )
        return proc.stdout.strip()

    def image_identity(self, image: str) -> str:
        proc = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            check=True,
            text=True,
            capture_output=True,
            env={"PATH": os.environ.get("PATH", "")},
        )
        value = proc.stdout.strip()
        if not value.startswith("sha256:"):
            raise RuntimeError("docker image did not resolve to a content identity")
        return value

    def run(
        self,
        image_id: str,
        command: tuple[str, ...],
        mounts: list[Mount],
        env: dict[str, str],
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
    ) -> ExecutionResult:
        name = f"ncpa-verifier-{uuid.uuid4().hex[:12]}"
        argv = [
            "docker", "run", "--rm", "--init", "--name", name,
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--pids-limit", str(self.limits.pids_limit),
            "--shm-size", self.limits.shm_size,
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
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
        clean_env = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "HOME": "/tmp",
            "XDG_CACHE_HOME": "/tmp/cache",
            "HF_HOME": "/tmp/hf",
            "TORCH_HOME": "/tmp/torch",
        }
        clean_env.update(env)
        for key, value in sorted(clean_env.items()):
            argv += ["--env", f"{key}={value}"]
        argv += [image_id, *command]
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            try:
                proc = subprocess.run(
                    argv,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=max(1.0, timeout_seconds),
                    env={"PATH": os.environ.get("PATH", "")},
                )
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                raise TimeoutError(f"container exceeded {timeout_seconds:.1f}s")
        return ExecutionResult(time.monotonic() - started, proc.returncode, stdout_path, stderr_path, image_id)
