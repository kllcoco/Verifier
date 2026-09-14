from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from .config import Config, load_config
from .docker import DockerRunner, Mount
from .hardware import probe_host, readiness
from .hashutil import sha256_file, sha256_json, sha256_tree
from .ledger import Ledger, verify_ledger


_NONCE = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
_STATE = "session.json"


class Budget:
    def __init__(self, max_cost_usd: float, hourly_cost_usd: float, elapsed_seconds: float = 0.0):
        self.max_cost_usd = max_cost_usd
        self.hourly_cost_usd = hourly_cost_usd
        self.elapsed_seconds = elapsed_seconds

    @property
    def cost_usd(self) -> float:
        return self.elapsed_seconds * self.hourly_cost_usd / 3600.0

    def timeout(self, requested_seconds: int) -> float:
        if requested_seconds <= 0:
            raise ValueError("timeout must be positive")
        if self.max_cost_usd <= 0 or self.hourly_cost_usd <= 0:
            return float(requested_seconds)
        cost_left = self.max_cost_usd - self.cost_usd
        if cost_left <= 0:
            raise RuntimeError("run cash budget exhausted")
        return min(float(requested_seconds), cost_left * 3600.0 / self.hourly_cost_usd)

    def charge(self, elapsed: float) -> None:
        self.elapsed_seconds += elapsed
        if self.max_cost_usd > 0 and self.cost_usd > self.max_cost_usd + 1e-9:
            raise RuntimeError("run cash budget exceeded")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_metrics(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _score(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {key!r} must be numeric")
    return float(value)


def _source_hash() -> str:
    return sha256_tree(Path(__file__).resolve().parent)


def _copy_snapshot(source: Path, destination: Path) -> str:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    source_hash = sha256_tree(source)
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        for child in sorted(source.rglob("*"), key=lambda p: p.relative_to(source).as_posix()):
            if child.is_symlink():
                raise ValueError(f"symlink not allowed: {child}")
            target = destination / child.relative_to(source)
            if child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif child.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, target)
    snapshot_hash = sha256_tree(destination)
    if source_hash != snapshot_hash:
        raise RuntimeError("input changed while being frozen")
    return snapshot_hash


def _tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _log_receipt(result: Any) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "elapsed_seconds": result.elapsed_seconds,
        "image_id": result.image_id,
        "stdout_sha256": sha256_file(result.stdout_path),
        "stderr_sha256": sha256_file(result.stderr_path),
    }


def _qualification(config: Config, before: float, after: float) -> dict[str, Any]:
    signed_delta = after - before if config.evaluation.higher_is_better else before - after
    checks = []
    if config.evaluation.minimum_delta is not None:
        checks.append({"name": "minimum_delta", "required": config.evaluation.minimum_delta, "actual": signed_delta, "pass": signed_delta >= config.evaluation.minimum_delta})
    if config.evaluation.minimum_score is not None:
        actual = after if config.evaluation.higher_is_better else -after
        required = config.evaluation.minimum_score if config.evaluation.higher_is_better else -config.evaluation.minimum_score
        checks.append({"name": "minimum_score", "required": config.evaluation.minimum_score, "actual": after, "pass": actual >= required})
    if not checks:
        status = "UNSPECIFIED"
    else:
        status = "PASS" if all(item["pass"] for item in checks) else "FAIL"
    return {"status": status, "checks": checks, "signed_delta": signed_delta}


def train_session(config: Config, config_path: Path, baseline: Path, train_data: Path, output_root: Path, nonce: str | None = None) -> Path:
    nonce = nonce or secrets.token_hex(16)
    if not _NONCE.fullmatch(nonce):
        raise ValueError("nonce must be 8-128 characters: letters, digits, dot, underscore or hyphen")
    baseline = baseline.resolve()
    train_data = train_data.resolve()
    output_root = output_root.resolve()
    config_path = config_path.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_hash = sha256_tree(baseline)
    train_source_hash = sha256_tree(train_data)
    config_hash = sha256_file(config_path)
    verifier_hash = _source_hash()
    runner = DockerRunner(config.limits)
    docker_version = runner.check()
    image_id = runner.image_identity(config.model.image)
    hardware = probe_host(output_root, baseline)
    ready = readiness(hardware, require_gpu=bool(config.limits.gpus))
    if not ready["ready"]:
        raise RuntimeError("; ".join(ready["issues"]))
    identity = {
        "nonce": nonce,
        "baseline_sha256": baseline_hash,
        "train_source_sha256": train_source_hash,
        "config_sha256": config_hash,
        "verifier_sha256": verifier_hash,
        "image_id": image_id,
    }
    run_digest = sha256_json(identity)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{stamp}-{run_digest[:16]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(run_dir / "ledger.jsonl")
    train_snapshot = run_dir / "sealed" / ("train.data" if train_data.is_file() else "train")
    snapshot_hash = _copy_snapshot(train_data, train_snapshot)
    if snapshot_hash != train_source_hash:
        raise RuntimeError("frozen training input hash mismatch")
    manifest = {
        "schema_version": 2,
        "run_id": run_digest,
        "created_at": _now(),
        **identity,
        "train_snapshot_sha256": snapshot_hash,
        "docker_version": docker_version,
        "hardware": hardware,
        "limits": asdict(config.limits),
        "evaluation": asdict(config.evaluation),
        "train_command": list(config.model.train_command),
        "eval_command": list(config.model.eval_command),
    }
    _write_json(run_dir / "manifest.json", manifest)
    ledger.append("run_frozen", manifest)
    ledger.append("training_input_frozen", {"sha256": snapshot_hash})
    candidate_out = run_dir / "candidate"
    candidate_out.mkdir()
    logs = run_dir / "logs"
    budget = Budget(config.limits.max_cost_usd, config.limits.hourly_cost_usd)
    try:
        result = runner.run(
            image_id,
            config.model.train_command,
            [Mount(baseline, "/model/input", True), Mount(train_snapshot, "/data/train", True), Mount(candidate_out, "/output", False)],
            {"VERIFIER_PHASE": "train", "VERIFIER_MODEL_OUT": "/output/model", "VERIFIER_TRAIN_DATA": "/data/train", "VERIFIER_RUN_ID": run_digest},
            budget.timeout(config.limits.train_timeout_seconds),
            logs / "train.stdout.log",
            logs / "train.stderr.log",
        )
        budget.charge(result.elapsed_seconds)
        receipt = _log_receipt(result)
        receipt["estimated_cost_usd"] = budget.cost_usd
        receipt["mount_targets"] = ["/model/input:ro", "/data/train:ro", "/output:rw"]
        receipt["network"] = "none"
        ledger.append("training_finished", receipt)
        if result.returncode != 0:
            raise RuntimeError(f"training failed: {_tail(result.stderr_path)}")
        model_dir = candidate_out / "model"
        if not model_dir.is_dir():
            raise RuntimeError("trainer did not write /output/model")
        candidate_hash = sha256_tree(model_dir)
        if sha256_tree(baseline) != baseline_hash:
            raise RuntimeError("baseline changed during training")
        ledger.append("candidate_frozen", {"candidate_sha256": candidate_hash, "secret_test_admitted": False})
        state = {
            "schema_version": 2,
            "run_id": run_digest,
            "phase": "TRAINED",
            "nonce": nonce,
            "baseline_path": str(baseline),
            "baseline_sha256": baseline_hash,
            "train_snapshot_sha256": snapshot_hash,
            "candidate_sha256": candidate_hash,
            "candidate_relpath": "candidate/model",
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "verifier_sha256": verifier_hash,
            "image_id": image_id,
            "elapsed_seconds": budget.elapsed_seconds,
            "estimated_cost_usd": budget.cost_usd,
            "test_rounds": 0,
        }
        _write_json(run_dir / _STATE, state)
        state_hash = sha256_file(run_dir / _STATE)
        terminal = ledger.append("training_state_committed", {"session_sha256": state_hash})
        commitment_payload = {"run_id": run_digest, "session_sha256": state_hash, "ledger_terminal_hash": terminal, "candidate_sha256": candidate_hash}
        commitment = sha256_json(commitment_payload)
        _write_json(run_dir / "train_receipt.json", {**commitment_payload, "commitment": commitment})
        return run_dir
    except Exception as exc:
        ledger.append("run_failed", {"phase": "train", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(run_dir / "failure.json", {"run_id": run_digest, "phase": "train", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": budget.elapsed_seconds, "estimated_cost_usd": budget.cost_usd})
        raise


def evaluate_session(run_dir: Path, secret_test: Path) -> Path:
    run_dir = run_dir.resolve()
    secret_test = secret_test.resolve()
    state_path = run_dir / _STATE
    receipt_path = run_dir / "train_receipt.json"
    state = _read_json(state_path)
    receipt = _read_json(receipt_path)
    if state.get("phase") != "TRAINED":
        raise RuntimeError("session is not in a trained state")
    if sha256_file(state_path) != receipt.get("session_sha256"):
        raise RuntimeError("training state does not match frozen receipt")
    if verify_ledger(run_dir / "ledger.jsonl") != receipt.get("ledger_terminal_hash"):
        raise RuntimeError("ledger changed after training commitment")
    config_path = Path(state["config_path"])
    config = load_config(config_path)
    baseline = Path(state["baseline_path"])
    candidate = run_dir / state["candidate_relpath"]
    if sha256_file(config_path) != state["config_sha256"]:
        raise RuntimeError("config changed after training")
    if _source_hash() != state["verifier_sha256"]:
        raise RuntimeError("verifier source changed after training")
    if sha256_tree(baseline) != state["baseline_sha256"]:
        raise RuntimeError("baseline changed after training")
    if sha256_tree(candidate) != state["candidate_sha256"]:
        raise RuntimeError("candidate changed after training")
    runner = DockerRunner(config.limits)
    runner.check()
    if runner.image_identity(state["image_id"]) != state["image_id"]:
        raise RuntimeError("training image identity is unavailable or changed")
    ledger = Ledger(run_dir / "ledger.jsonl")
    round_number = int(state.get("test_rounds", 0)) + 1
    round_dir = run_dir / "tests" / f"round-{round_number:04d}"
    round_dir.mkdir(parents=True, exist_ok=False)
    test_snapshot = round_dir / ("test.data" if secret_test.is_file() else "test")
    test_hash = _copy_snapshot(secret_test, test_snapshot)
    ledger.append("secret_test_admitted", {"round": round_number, "secret_test_sha256": test_hash, "candidate_sha256": state["candidate_sha256"]})
    budget = Budget(config.limits.max_cost_usd, config.limits.hourly_cost_usd, float(state.get("elapsed_seconds", 0.0)))
    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    baseline_metrics_all: list[dict[str, Any]] = []
    candidate_metrics_all: list[dict[str, Any]] = []
    try:
        for repeat in range(config.evaluation.repeats):
            seed = repeat
            base_out = round_dir / f"baseline-{repeat:02d}"
            cand_out = round_dir / f"candidate-{repeat:02d}"
            base_out.mkdir()
            cand_out.mkdir()
            base_result = runner.run(
                state["image_id"],
                config.model.eval_command,
                [Mount(baseline, "/model/input", True), Mount(test_snapshot, "/data/test", True), Mount(base_out, "/output", False)],
                {"VERIFIER_PHASE": "baseline_eval", "VERIFIER_OUTPUT": "/output/metrics.json", "VERIFIER_RUN_ID": state["run_id"], "VERIFIER_SEED": str(seed)},
                budget.timeout(config.limits.eval_timeout_seconds),
                base_out / "stdout.log",
                base_out / "stderr.log",
            )
            budget.charge(base_result.elapsed_seconds)
            ledger.append("baseline_eval_finished", {"round": round_number, "repeat": repeat, **_log_receipt(base_result), "estimated_cost_usd": budget.cost_usd})
            if base_result.returncode != 0:
                raise RuntimeError(f"baseline evaluation failed: {_tail(base_result.stderr_path)}")
            base_metrics = _read_metrics(base_out / "metrics.json")
            baseline_metrics_all.append(base_metrics)
            baseline_scores.append(_score(base_metrics, config.evaluation.score_key))
            cand_result = runner.run(
                state["image_id"],
                config.model.eval_command,
                [Mount(candidate, "/model/input", True), Mount(test_snapshot, "/data/test", True), Mount(cand_out, "/output", False)],
                {"VERIFIER_PHASE": "candidate_eval", "VERIFIER_OUTPUT": "/output/metrics.json", "VERIFIER_RUN_ID": state["run_id"], "VERIFIER_SEED": str(seed)},
                budget.timeout(config.limits.eval_timeout_seconds),
                cand_out / "stdout.log",
                cand_out / "stderr.log",
            )
            budget.charge(cand_result.elapsed_seconds)
            ledger.append("candidate_eval_finished", {"round": round_number, "repeat": repeat, **_log_receipt(cand_result), "estimated_cost_usd": budget.cost_usd})
            if cand_result.returncode != 0:
                raise RuntimeError(f"candidate evaluation failed: {_tail(cand_result.stderr_path)}")
            cand_metrics = _read_metrics(cand_out / "metrics.json")
            candidate_metrics_all.append(cand_metrics)
            candidate_scores.append(_score(cand_metrics, config.evaluation.score_key))
        if sha256_tree(baseline) != state["baseline_sha256"]:
            raise RuntimeError("baseline changed during evaluation")
        if sha256_tree(candidate) != state["candidate_sha256"]:
            raise RuntimeError("candidate changed during evaluation")
        if sha256_tree(test_snapshot) != test_hash:
            raise RuntimeError("secret test changed during evaluation")
        before = fmean(baseline_scores)
        after = fmean(candidate_scores)
        qualification = _qualification(config, before, after)
        report = {
            "schema_version": 2,
            "run_id": state["run_id"],
            "round": round_number,
            "status": "COMPLETE",
            "qualification": qualification,
            "nonce": state["nonce"],
            "baseline_sha256": state["baseline_sha256"],
            "candidate_sha256": state["candidate_sha256"],
            "train_snapshot_sha256": state["train_snapshot_sha256"],
            "secret_test_sha256": test_hash,
            "config_sha256": state["config_sha256"],
            "verifier_sha256": state["verifier_sha256"],
            "image_id": state["image_id"],
            "score_key": config.evaluation.score_key,
            "higher_is_better": config.evaluation.higher_is_better,
            "baseline_score": before,
            "candidate_score": after,
            "delta": after - before,
            "baseline_scores": baseline_scores,
            "candidate_scores": candidate_scores,
            "baseline_metrics": baseline_metrics_all,
            "candidate_metrics": candidate_metrics_all,
            "elapsed_seconds_total": budget.elapsed_seconds,
            "estimated_cost_usd_total": budget.cost_usd,
            "network_enabled": False,
            "secret_test_existed_in_training_session": False,
            "candidate_frozen_before_secret_test": True,
            "test_admitted_at": _now(),
        }
        report_path = round_dir / "report.json"
        _write_json(report_path, report)
        ledger.append("report_written", {"round": round_number, "report_sha256": sha256_file(report_path)})
        state["test_rounds"] = round_number
        state["elapsed_seconds"] = budget.elapsed_seconds
        state["estimated_cost_usd"] = budget.cost_usd
        _write_json(state_path, state)
        final_terminal = verify_ledger(run_dir / "ledger.jsonl")
        final_receipt = {
            "run_id": state["run_id"],
            "round": round_number,
            "train_commitment": receipt["commitment"],
            "report_sha256": sha256_file(report_path),
            "ledger_terminal_hash": final_terminal,
            "candidate_sha256": state["candidate_sha256"],
            "secret_test_sha256": test_hash,
        }
        final_receipt["receipt_hash"] = sha256_json(final_receipt)
        _write_json(round_dir / "receipt.json", final_receipt)
        _write_json(run_dir / "latest_report.json", report)
        return report_path
    except Exception as exc:
        ledger.append("run_failed", {"phase": "evaluate", "round": round_number, "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(round_dir / "failure.json", {"run_id": state["run_id"], "phase": "evaluate", "round": round_number, "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds_total": budget.elapsed_seconds, "estimated_cost_usd_total": budget.cost_usd})
        raise


def audit_session(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    train_receipt = _read_json(run_dir / "train_receipt.json")
    state = _read_json(run_dir / _STATE)
    state_hash = sha256_file(run_dir / _STATE)
    ledger_terminal = verify_ledger(run_dir / "ledger.jsonl")
    candidate = run_dir / state["candidate_relpath"]
    result = {
        "run_id": state["run_id"],
        "train_commitment": train_receipt["commitment"],
        "candidate_sha256": sha256_tree(candidate),
        "candidate_matches": sha256_tree(candidate) == state["candidate_sha256"],
        "state_matches_original_train_receipt": state_hash == train_receipt["session_sha256"],
        "ledger_terminal_hash": ledger_terminal,
        "test_rounds": state.get("test_rounds", 0),
    }
    return result


def run_verification(config: Config, config_path: Path, baseline: Path, train_data: Path, secret_test: Path, output_root: Path, nonce: str) -> Path:
    run_dir = train_session(config, config_path, baseline, train_data, output_root, nonce)
    evaluate_session(run_dir, secret_test)
    return run_dir
