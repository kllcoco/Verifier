from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    image: str
    train_command: tuple[str, ...]
    eval_command: tuple[str, ...]


@dataclass(frozen=True)
class LimitConfig:
    cpus: float | None
    memory: str | None
    gpus: str | None
    pids_limit: int
    timeout_seconds: int
    hourly_cost_usd: float
    max_cost_usd: float


@dataclass(frozen=True)
class EvalConfig:
    score_key: str
    higher_is_better: bool


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    limits: LimitConfig
    evaluation: EvalConfig


def _command(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ValueError(f"{key} must be a non-empty string array")
    return tuple(value)


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        raw = tomllib.load(f)

    model = raw.get("model") or {}
    limits = raw.get("limits") or {}
    evaluation = raw.get("evaluation") or {}

    image = model.get("image")
    if not isinstance(image, str) or not image.strip():
        raise ValueError("model.image is required")

    hourly_cost = float(limits.get("hourly_cost_usd", 0.0))
    max_cost = float(limits.get("max_cost_usd", 0.0))
    timeout = int(limits.get("timeout_seconds", 3600))
    pids = int(limits.get("pids_limit", 512))
    cpus = limits.get("cpus")
    cpus = None if cpus is None else float(cpus)
    memory = limits.get("memory")
    memory = None if memory is None else str(memory)
    gpus = limits.get("gpus")
    gpus = None if gpus in (None, "") else str(gpus)

    if hourly_cost < 0 or max_cost < 0 or timeout <= 0 or pids <= 0:
        raise ValueError("invalid limits")

    score_key = evaluation.get("score_key", "score")
    if not isinstance(score_key, str) or not score_key:
        raise ValueError("evaluation.score_key must be a non-empty string")

    return Config(
        model=ModelConfig(
            image=image,
            train_command=_command(model.get("train_command"), "model.train_command"),
            eval_command=_command(model.get("eval_command"), "model.eval_command"),
        ),
        limits=LimitConfig(
            cpus=cpus,
            memory=memory,
            gpus=gpus,
            pids_limit=pids,
            timeout_seconds=timeout,
            hourly_cost_usd=hourly_cost,
            max_cost_usd=max_cost,
        ),
        evaluation=EvalConfig(
            score_key=score_key,
            higher_is_better=bool(evaluation.get("higher_is_better", True)),
        ),
    )
