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
from .ledger import Ledger, ledger_contains_hash, verify_ledger

_NONCE = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
_STATE = "session.json"
_TRAIN_STATE = "train_state.json"


class Budget:
    def __init__(self, max_cost_usd: float, hourly_cost_usd: float, elapsed_seconds: float = 0.0):
        self.max_cost_usd = max_cost_usd
        self.hourly_cost_usd = hourly_cost_usd
        self.elapsed_seconds = elapsed_seconds

    @property
    def cost_usd(self) -> float:
        return self.elapsed_seconds * self.hourly_cost_usd / 3600.0

    def timeout(self, seconds: int) -> float:
        if seconds <= 0:
            raise ValueError("timeout must be positive")
        if self.max_cost_usd <= 0 or self.hourly_cost_usd <= 0:
            return float(seconds)
        left = self.max_cost_usd - self.cost_usd
        if left <= 0:
            raise RuntimeError("run cash budget exhausted")
        return min(float(seconds), left * 3600.0 / self.hourly_cost_usd)

    def charge(self, seconds: float) -> None:
        self.elapsed_seconds += seconds
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
    before = sha256_tree(source)
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
    after = sha256_tree(destination)
    if before != after:
        raise RuntimeError("input changed while being frozen")
    return after


def _tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    return path.read_bytes()[-limit:].decode("utf-8", errors="replace")


def _execution_receipt(result: Any) -> dict[str, Any]:
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
        ok = after >= config.evaluation.minimum_score if config.evaluation.higher_is_better else after <= config.evaluation.minimum_score
        checks.append({"name": "minimum_score", "required": config.evaluation.minimum_score, "actual": after, "pass": ok})
    return {"status": ("UNSPECIFIED" if not checks else ("PASS" if all(x["pass"] for x in checks) else "FAIL")), "checks": checks, "signed_delta": signed_delta}


def train_session(config: Config, config_path: Path, baseline: Path, train_data: Path, output_root: Path, nonce: str | None = None) -> Path:
    nonce = nonce or secrets.token_hex(16)
    if not _NONCE.fullmatch(nonce):
        raise ValueError("nonce must be 8-128 characters: letters, digits, dot, underscore or hyphen")
    baseline, train_data, output_root, config_path = baseline.resolve(), train_data.resolve(), output_root.resolve(), config_path.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_hash, train_source_hash = sha256_tree(baseline), sha256_tree(train_data)
    config_hash, verifier_hash = sha256_file(config_path), _source_hash()
    runner = DockerRunner(config.limits)
    docker_version = runner.check()
    image_id = runner.image_identity(config.model.image)
    hardware = probe_host(output_root, baseline)
    ready = readiness(hardware, require_gpu=bool(config.limits.gpus))
    if not ready["ready"]:
        raise RuntimeError("; ".join(ready["issues"]))
    identity = {"nonce": nonce, "baseline_sha256": baseline_hash, "train_source_sha256": train_source_hash, "config_sha256": config_hash, "verifier_sha256": verifier_hash, "image_id": image_id}
    run_id = sha256_json(identity)
    run_dir = output_root / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{run_id[:16]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(run_dir / "ledger.jsonl")
    train_snapshot = run_dir / "sealed" / ("train.data" if train_data.is_file() else "train")
    train_hash = _copy_snapshot(train_data, train_snapshot)
    manifest = {"schema_version": 2, "run_id": run_id, "created_at": _now(), **identity, "train_snapshot_sha256": train_hash, "docker_version": docker_version, "hardware": hardware, "limits": asdict(config.limits), "evaluation": asdict(config.evaluation), "train_command": list(config.model.train_command), "eval_command": list(config.model.eval_command)}
    _write_json(run_dir / "manifest.json", manifest)
    ledger.append("run_frozen", manifest)
    candidate_out = run_dir / "candidate"
    candidate_out.mkdir()
    budget = Budget(config.limits.max_cost_usd, config.limits.hourly_cost_usd)
    try:
        result = runner.run(image_id, config.model.train_command, [Mount(baseline, "/model/input", True), Mount(train_snapshot, "/data/train", True), Mount(candidate_out, "/output", False)], {"VERIFIER_PHASE": "train", "VERIFIER_MODEL_OUT": "/output/model", "VERIFIER_TRAIN_DATA": "/data/train", "VERIFIER_RUN_ID": run_id}, budget.timeout(config.limits.train_timeout_seconds), run_dir / "logs/train.stdout.log", run_dir / "logs/train.stderr.log")
        budget.charge(result.elapsed_seconds)
        rec = _execution_receipt(result) | {"estimated_cost_usd": budget.cost_usd, "mount_targets": ["/model/input:ro", "/data/train:ro", "/output:rw"], "network": "none"}
        ledger.append("training_finished", rec)
        if result.returncode != 0:
            raise RuntimeError(f"training failed: {_tail(result.stderr_path)}")
        candidate = candidate_out / "model"
        if not candidate.is_dir():
            raise RuntimeError("trainer did not write /output/model")
        candidate_hash = sha256_tree(candidate)
        if sha256_tree(baseline) != baseline_hash:
            raise RuntimeError("baseline changed during training")
        ledger.append("candidate_frozen", {"candidate_sha256": candidate_hash, "secret_test_admitted": False})
        frozen = {"schema_version": 2, "run_id": run_id, "phase": "TRAINED", "nonce": nonce, "baseline_path": str(baseline), "baseline_sha256": baseline_hash, "train_snapshot_sha256": train_hash, "candidate_sha256": candidate_hash, "candidate_relpath": "candidate/model", "config_path": str(config_path), "config_sha256": config_hash, "verifier_sha256": verifier_hash, "image_id": image_id}
        session = frozen | {"elapsed_seconds": budget.elapsed_seconds, "estimated_cost_usd": budget.cost_usd, "test_rounds": 0}
        _write_json(run_dir / _TRAIN_STATE, frozen)
        _write_json(run_dir / _STATE, session)
        frozen_hash = sha256_file(run_dir / _TRAIN_STATE)
        terminal = ledger.append("training_state_committed", {"train_state_sha256": frozen_hash})
        commitment_payload = {"run_id": run_id, "session_sha256": frozen_hash, "ledger_terminal_hash": terminal, "candidate_sha256": candidate_hash}
        _write_json(run_dir / "train_receipt.json", commitment_payload | {"commitment": sha256_json(commitment_payload)})
        return run_dir
    except Exception as exc:
        ledger.append("run_failed", {"phase": "train", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(run_dir / "failure.json", {"run_id": run_id, "phase": "train", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
        raise


def evaluate_session(run_dir: Path, secret_test: Path) -> Path:
    run_dir, secret_test = run_dir.resolve(), secret_test.resolve()
    state, frozen, receipt = _read_json(run_dir / _STATE), _read_json(run_dir / _TRAIN_STATE), _read_json(run_dir / "train_receipt.json")
    if sha256_file(run_dir / _TRAIN_STATE) != receipt.get("session_sha256"):
        raise RuntimeError("frozen training state does not match receipt")
    verify_ledger(run_dir / "ledger.jsonl")
    if not ledger_contains_hash(run_dir / "ledger.jsonl", receipt.get("ledger_terminal_hash", "")):
        raise RuntimeError("training commitment is not present in ledger")
    for key in ("run_id", "nonce", "baseline_sha256", "train_snapshot_sha256", "candidate_sha256", "candidate_relpath", "config_sha256", "verifier_sha256", "image_id"):
        if state.get(key) != frozen.get(key):
            raise RuntimeError(f"mutable session changed frozen field: {key}")
    config_path, baseline, candidate = Path(frozen["config_path"]), Path(frozen["baseline_path"]), run_dir / frozen["candidate_relpath"]
    config = load_config(config_path)
    if sha256_file(config_path) != frozen["config_sha256"] or _source_hash() != frozen["verifier_sha256"]:
        raise RuntimeError("config or verifier changed after training")
    if sha256_tree(baseline) != frozen["baseline_sha256"] or sha256_tree(candidate) != frozen["candidate_sha256"]:
        raise RuntimeError("baseline or candidate changed after training")
    runner = DockerRunner(config.limits)
    runner.check()
    if runner.image_identity(frozen["image_id"]) != frozen["image_id"]:
        raise RuntimeError("training image identity is unavailable or changed")
    ledger = Ledger(run_dir / "ledger.jsonl")
    round_no = int(state.get("test_rounds", 0)) + 1
    round_dir = run_dir / "tests" / f"round-{round_no:04d}"
    round_dir.mkdir(parents=True, exist_ok=False)
    test_snapshot = round_dir / ("test.data" if secret_test.is_file() else "test")
    test_hash = _copy_snapshot(secret_test, test_snapshot)
    ledger.append("secret_test_admitted", {"round": round_no, "secret_test_sha256": test_hash, "candidate_sha256": frozen["candidate_sha256"]})
    budget = Budget(config.limits.max_cost_usd, config.limits.hourly_cost_usd, float(state.get("elapsed_seconds", 0.0)))
    base_scores, cand_scores, base_metrics, cand_metrics = [], [], [], []
    try:
        for repeat in range(config.evaluation.repeats):
            bo, co = round_dir / f"baseline-{repeat:02d}", round_dir / f"candidate-{repeat:02d}"
            bo.mkdir(); co.mkdir()
            br = runner.run(frozen["image_id"], config.model.eval_command, [Mount(baseline, "/model/input", True), Mount(test_snapshot, "/data/test", True), Mount(bo, "/output", False)], {"VERIFIER_PHASE": "baseline_eval", "VERIFIER_OUTPUT": "/output/metrics.json", "VERIFIER_RUN_ID": frozen["run_id"], "VERIFIER_SEED": str(repeat)}, budget.timeout(config.limits.eval_timeout_seconds), bo / "stdout.log", bo / "stderr.log")
            budget.charge(br.elapsed_seconds); ledger.append("baseline_eval_finished", {"round": round_no, "repeat": repeat, **_execution_receipt(br), "estimated_cost_usd": budget.cost_usd})
            if br.returncode != 0: raise RuntimeError(f"baseline evaluation failed: {_tail(br.stderr_path)}")
            bm = _read_json(bo / "metrics.json"); base_metrics.append(bm); base_scores.append(_score(bm, config.evaluation.score_key))
            cr = runner.run(frozen["image_id"], config.model.eval_command, [Mount(candidate, "/model/input", True), Mount(test_snapshot, "/data/test", True), Mount(co, "/output", False)], {"VERIFIER_PHASE": "candidate_eval", "VERIFIER_OUTPUT": "/output/metrics.json", "VERIFIER_RUN_ID": frozen["run_id"], "VERIFIER_SEED": str(repeat)}, budget.timeout(config.limits.eval_timeout_seconds), co / "stdout.log", co / "stderr.log")
            budget.charge(cr.elapsed_seconds); ledger.append("candidate_eval_finished", {"round": round_no, "repeat": repeat, **_execution_receipt(cr), "estimated_cost_usd": budget.cost_usd})
            if cr.returncode != 0: raise RuntimeError(f"candidate evaluation failed: {_tail(cr.stderr_path)}")
            cm = _read_json(co / "metrics.json"); cand_metrics.append(cm); cand_scores.append(_score(cm, config.evaluation.score_key))
        if sha256_tree(baseline) != frozen["baseline_sha256"] or sha256_tree(candidate) != frozen["candidate_sha256"] or sha256_tree(test_snapshot) != test_hash:
            raise RuntimeError("frozen artifact changed during evaluation")
        before, after = fmean(base_scores), fmean(cand_scores)
        report = {"schema_version": 2, "run_id": frozen["run_id"], "round": round_no, "status": "COMPLETE", "qualification": _qualification(config, before, after), "nonce": frozen["nonce"], "baseline_sha256": frozen["baseline_sha256"], "candidate_sha256": frozen["candidate_sha256"], "train_snapshot_sha256": frozen["train_snapshot_sha256"], "secret_test_sha256": test_hash, "config_sha256": frozen["config_sha256"], "verifier_sha256": frozen["verifier_sha256"], "image_id": frozen["image_id"], "score_key": config.evaluation.score_key, "higher_is_better": config.evaluation.higher_is_better, "baseline_score": before, "candidate_score": after, "delta": after - before, "baseline_scores": base_scores, "candidate_scores": cand_scores, "baseline_metrics": base_metrics, "candidate_metrics": cand_metrics, "elapsed_seconds_total": budget.elapsed_seconds, "estimated_cost_usd_total": budget.cost_usd, "network_enabled": False, "secret_test_existed_in_training_session": False, "candidate_frozen_before_secret_test": True, "test_admitted_at": _now()}
        report_path = round_dir / "report.json"; _write_json(report_path, report)
        ledger.append("report_written", {"round": round_no, "report_sha256": sha256_file(report_path)})
        state["test_rounds"], state["elapsed_seconds"], state["estimated_cost_usd"] = round_no, budget.elapsed_seconds, budget.cost_usd
        _write_json(run_dir / _STATE, state)
        final = {"run_id": frozen["run_id"], "round": round_no, "train_commitment": receipt["commitment"], "report_sha256": sha256_file(report_path), "ledger_terminal_hash": verify_ledger(run_dir / "ledger.jsonl"), "candidate_sha256": frozen["candidate_sha256"], "secret_test_sha256": test_hash}
        _write_json(round_dir / "receipt.json", final | {"receipt_hash": sha256_json(final)})
        _write_json(run_dir / "latest_report.json", report)
        return report_path
    except Exception as exc:
        ledger.append("run_failed", {"phase": "evaluate", "round": round_no, "error_type": type(exc).__name__, "error": str(exc)})
        raise


def audit_session(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    receipt, state = _read_json(run_dir / "train_receipt.json"), _read_json(run_dir / _STATE)
    candidate = run_dir / state["candidate_relpath"]
    return {"run_id": state["run_id"], "train_commitment": receipt["commitment"], "candidate_sha256": sha256_tree(candidate), "candidate_matches": sha256_tree(candidate) == state["candidate_sha256"], "frozen_train_state_matches_receipt": sha256_file(run_dir / _TRAIN_STATE) == receipt["session_sha256"], "training_commitment_present_in_ledger": ledger_contains_hash(run_dir / "ledger.jsonl", receipt["ledger_terminal_hash"]), "ledger_terminal_hash": verify_ledger(run_dir / "ledger.jsonl"), "test_rounds": state.get("test_rounds", 0)}


def run_verification(config: Config, config_path: Path, baseline: Path, train_data: Path, secret_test: Path, output_root: Path, nonce: str) -> Path:
    run_dir = train_session(config, config_path, baseline, train_data, output_root, nonce)
    evaluate_session(run_dir, secret_test)
    return run_dir
