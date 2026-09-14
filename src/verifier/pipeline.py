from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .docker import DockerRunner, Mount
from .hashutil import sha256_file, sha256_json, sha256_tree
from .ledger import Ledger, verify_ledger


_NONCE = re.compile(r"^[A-Za-z0-9._-]{8,128}$")


class Budget:
    def __init__(self, max_cost_usd: float, hourly_cost_usd: float, timeout_seconds: int):
        self.max_cost_usd = max_cost_usd
        self.hourly_cost_usd = hourly_cost_usd
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = 0.0

    @property
    def cost_usd(self) -> float:
        return self.elapsed_seconds * self.hourly_cost_usd / 3600.0

    def next_timeout(self) -> float:
        wall_left = self.timeout_seconds - self.elapsed_seconds
        if wall_left <= 0:
            raise TimeoutError("run wall-time budget exhausted")
        if self.max_cost_usd <= 0 or self.hourly_cost_usd <= 0:
            return wall_left
        cost_left = self.max_cost_usd - self.cost_usd
        if cost_left <= 0:
            raise RuntimeError("run cash budget exhausted")
        return min(wall_left, cost_left * 3600.0 / self.hourly_cost_usd)

    def charge(self, elapsed: float) -> None:
        self.elapsed_seconds += elapsed
        if self.max_cost_usd > 0 and self.cost_usd > self.max_cost_usd + 1e-9:
            raise RuntimeError("run cash budget exceeded")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("metrics.json must contain an object")
    return data


def _score(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {key!r} must be numeric")
    return float(value)


def _source_hash() -> str:
    return sha256_tree(Path(__file__).resolve().parent)


def run_verification(config: Config, config_path: Path, baseline: Path, train_data: Path, secret_test: Path, output_root: Path, nonce: str) -> Path:
    if not _NONCE.fullmatch(nonce):
        raise ValueError("nonce must be 8-128 characters: letters, digits, dot, underscore or hyphen")

    baseline = baseline.resolve()
    train_data = train_data.resolve()
    secret_test = secret_test.resolve()
    output_root = output_root.resolve()

    baseline_hash = sha256_tree(baseline)
    train_hash = sha256_tree(train_data)
    secret_hash = sha256_tree(secret_test)
    config_hash = sha256_file(config_path.resolve())
    verifier_hash = _source_hash()

    identity = {
        "nonce": nonce,
        "baseline_sha256": baseline_hash,
        "train_sha256": train_hash,
        "secret_test_sha256": secret_hash,
        "config_sha256": config_hash,
        "verifier_sha256": verifier_hash,
    }
    run_digest = sha256_json(identity)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{stamp}-{run_digest[:16]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    ledger = Ledger(run_dir / "ledger.jsonl")
    manifest = {
        "run_id": run_digest,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **identity,
        "image": config.model.image,
        "train_command": list(config.model.train_command),
        "eval_command": list(config.model.eval_command),
        "limits": asdict(config.limits),
        "evaluation": asdict(config.evaluation),
    }
    _write_json(run_dir / "manifest.json", manifest)
    ledger.append("run_frozen", manifest)

    runner = DockerRunner(config.limits)
    runner.check()
    budget = Budget(config.limits.max_cost_usd, config.limits.hourly_cost_usd, config.limits.timeout_seconds)

    baseline_out = run_dir / "baseline_eval"
    candidate_out = run_dir / "candidate"
    candidate_eval_out = run_dir / "candidate_eval"
    baseline_out.mkdir()
    candidate_out.mkdir()
    candidate_eval_out.mkdir()

    try:
        result = runner.run(
            config.model.image,
            config.model.eval_command,
            [Mount(baseline, "/model/input", True), Mount(secret_test, "/data/test", True), Mount(baseline_out, "/output", False)],
            {"VERIFIER_PHASE": "baseline_eval", "VERIFIER_OUTPUT": "/output/metrics.json"},
            budget.next_timeout(),
        )
        budget.charge(result.elapsed_seconds)
        ledger.append("baseline_eval_finished", {"returncode": result.returncode, "elapsed_seconds": result.elapsed_seconds, "cost_usd": budget.cost_usd})
        if result.returncode != 0:
            raise RuntimeError(f"baseline evaluation failed: {result.stderr[-4000:]}")
        baseline_metrics_path = baseline_out / "metrics.json"
        if not baseline_metrics_path.is_file():
            raise RuntimeError("baseline evaluator did not write /output/metrics.json")
        baseline_metrics = _read_metrics(baseline_metrics_path)

        result = runner.run(
            config.model.image,
            config.model.train_command,
            [Mount(baseline, "/model/input", True), Mount(train_data, "/data/train", True), Mount(candidate_out, "/output", False)],
            {"VERIFIER_PHASE": "train", "VERIFIER_MODEL_OUT": "/output/model", "VERIFIER_TRAIN_DATA": "/data/train"},
            budget.next_timeout(),
        )
        budget.charge(result.elapsed_seconds)
        ledger.append("training_finished", {"returncode": result.returncode, "elapsed_seconds": result.elapsed_seconds, "cost_usd": budget.cost_usd})
        if result.returncode != 0:
            raise RuntimeError(f"training failed: {result.stderr[-4000:]}")

        model_dir = candidate_out / "model"
        if not model_dir.is_dir():
            raise RuntimeError("trainer did not write /output/model")
        candidate_hash = sha256_tree(model_dir)
        ledger.append("candidate_frozen", {"candidate_sha256": candidate_hash})

        result = runner.run(
            config.model.image,
            config.model.eval_command,
            [Mount(model_dir, "/model/input", True), Mount(secret_test, "/data/test", True), Mount(candidate_eval_out, "/output", False)],
            {"VERIFIER_PHASE": "candidate_eval", "VERIFIER_OUTPUT": "/output/metrics.json"},
            budget.next_timeout(),
        )
        budget.charge(result.elapsed_seconds)
        ledger.append("candidate_eval_finished", {"returncode": result.returncode, "elapsed_seconds": result.elapsed_seconds, "cost_usd": budget.cost_usd})
        if result.returncode != 0:
            raise RuntimeError(f"candidate evaluation failed: {result.stderr[-4000:]}")
        candidate_metrics_path = candidate_eval_out / "metrics.json"
        if not candidate_metrics_path.is_file():
            raise RuntimeError("candidate evaluator did not write /output/metrics.json")
        candidate_metrics = _read_metrics(candidate_metrics_path)

        before = _score(baseline_metrics, config.evaluation.score_key)
        after = _score(candidate_metrics, config.evaluation.score_key)
        delta = after - before
        improved = delta > 0 if config.evaluation.higher_is_better else delta < 0

        report = {
            "run_id": run_digest,
            "status": "PASS",
            "nonce": nonce,
            "baseline_sha256": baseline_hash,
            "candidate_sha256": candidate_hash,
            "train_sha256": train_hash,
            "secret_test_sha256": secret_hash,
            "config_sha256": config_hash,
            "verifier_sha256": verifier_hash,
            "score_key": config.evaluation.score_key,
            "higher_is_better": config.evaluation.higher_is_better,
            "baseline_score": before,
            "candidate_score": after,
            "delta": delta,
            "improved": improved,
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "elapsed_seconds": budget.elapsed_seconds,
            "estimated_cost_usd": budget.cost_usd,
            "network_enabled": False,
            "secret_mounted_during_training": False,
        }
        _write_json(run_dir / "report.json", report)
        ledger.append("report_written", {"report_sha256": sha256_file(run_dir / "report.json")})
        final_hash = verify_ledger(run_dir / "ledger.jsonl")
        _write_json(run_dir / "receipt.json", {"run_id": run_digest, "report_sha256": sha256_file(run_dir / "report.json"), "ledger_terminal_hash": final_hash})
        return run_dir
    except Exception as exc:
        ledger.append("run_failed", {"error_type": type(exc).__name__, "error": str(exc)})
        _write_json(run_dir / "failure.json", {
            "run_id": run_digest,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": budget.elapsed_seconds,
            "estimated_cost_usd": budget.cost_usd,
        })
        raise
