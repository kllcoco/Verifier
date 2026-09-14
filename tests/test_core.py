import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from verifier.config import load_config
from verifier.docker import ExecutionResult
from verifier.hashutil import sha256_tree
from verifier.ledger import Ledger, verify_ledger
from verifier.pipeline import audit_session, evaluate_session, train_session


class FakeRunner:
    calls = []

    def __init__(self, limits):
        self.limits = limits

    def check(self):
        return "test"

    def image_identity(self, image):
        if image.startswith("sha256:"):
            return image
        return "sha256:" + "a" * 64

    def run(self, image_id, command, mounts, env, timeout_seconds, stdout_path, stderr_path):
        self.calls.append({"phase": env["VERIFIER_PHASE"], "targets": [m.target for m in mounts]})
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        output = next(m.source for m in mounts if m.target == "/output")
        if env["VERIFIER_PHASE"] == "train":
            model = output / "model"
            model.mkdir(parents=True, exist_ok=True)
            (model / "weights.bin").write_bytes(b"candidate")
        else:
            score = 0.25 if env["VERIFIER_PHASE"] == "baseline_eval" else 0.75
            (output / "metrics.json").write_text(json.dumps({"accuracy": score}), encoding="utf-8")
        return ExecutionResult(0.01, 0, stdout_path, stderr_path, image_id)


class CoreTests(unittest.TestCase):
    def test_tree_hash_is_content_stable(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            pa = Path(a)
            pb = Path(b)
            (pa / "x").mkdir()
            (pb / "x").mkdir()
            (pa / "x" / "a.txt").write_text("one", encoding="utf-8")
            (pb / "x" / "a.txt").write_text("one", encoding="utf-8")
            self.assertEqual(sha256_tree(pa), sha256_tree(pb))
            (pb / "x" / "a.txt").write_text("two", encoding="utf-8")
            self.assertNotEqual(sha256_tree(pa), sha256_tree(pb))

    def test_ledger_detects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            ledger = Ledger(path)
            ledger.append("a", {"x": 1})
            terminal = ledger.append("b", {"y": 2})
            self.assertEqual(terminal, verify_ledger(path))
            rows = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["payload"]["x"] = 2
            rows[0] = json.dumps(first)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_ledger(path)

    def _config(self, root: Path) -> Path:
        path = root / "run.toml"
        path.write_text('''
[model]
image = "model:test"
train_command = ["train"]
eval_command = ["eval"]

[limits]
train_timeout_seconds = 120
eval_timeout_seconds = 60
hourly_cost_usd = 1.5
max_cost_usd = 2.0

[evaluation]
score_key = "accuracy"
higher_is_better = true
minimum_delta = 0.25
repeats = 2
''', encoding="utf-8")
        return path

    def test_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(self._config(Path(d)))
            self.assertEqual(cfg.model.image, "model:test")
            self.assertEqual(cfg.limits.train_timeout_seconds, 120)
            self.assertEqual(cfg.evaluation.repeats, 2)

    @patch("verifier.pipeline.readiness", return_value={"ready": True, "issues": []})
    @patch("verifier.pipeline.probe_host", return_value={"docker_version": "test", "gpus": []})
    @patch("verifier.pipeline.DockerRunner", FakeRunner)
    def test_two_phase_secret_never_enters_training(self, *_):
        FakeRunner.calls = []
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "weights.bin").write_bytes(b"base")
            train = root / "train.bin"
            train.write_bytes(b"training")
            secret = root / "secret.bin"
            secret.write_bytes(b"secret")
            config_path = self._config(root)
            run_dir = train_session(load_config(config_path), config_path, baseline, train, root / "runs", "investor-1234")
            train_calls = [c for c in FakeRunner.calls if c["phase"] == "train"]
            self.assertEqual(len(train_calls), 1)
            self.assertNotIn("/data/test", train_calls[0]["targets"])
            receipt = json.loads((run_dir / "train_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(len(receipt["commitment"]), 64)
            report_path = evaluate_session(run_dir, secret)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["qualification"]["status"], "PASS")
            self.assertEqual(report["baseline_score"], 0.25)
            self.assertEqual(report["candidate_score"], 0.75)
            self.assertTrue(report["candidate_frozen_before_secret_test"])

    @patch("verifier.pipeline.readiness", return_value={"ready": True, "issues": []})
    @patch("verifier.pipeline.probe_host", return_value={"docker_version": "test", "gpus": []})
    @patch("verifier.pipeline.DockerRunner", FakeRunner)
    def test_candidate_tamper_is_rejected(self, *_):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "weights.bin").write_bytes(b"base")
            train = root / "train.bin"
            train.write_bytes(b"training")
            secret = root / "secret.bin"
            secret.write_bytes(b"secret")
            config_path = self._config(root)
            run_dir = train_session(load_config(config_path), config_path, baseline, train, root / "runs", "investor-1234")
            (run_dir / "candidate" / "model" / "weights.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "candidate changed"):
                evaluate_session(run_dir, secret)

    @patch("verifier.pipeline.readiness", return_value={"ready": True, "issues": []})
    @patch("verifier.pipeline.probe_host", return_value={"docker_version": "test", "gpus": []})
    @patch("verifier.pipeline.DockerRunner", FakeRunner)
    def test_audit_reports_candidate_match(self, *_):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "weights.bin").write_bytes(b"base")
            train = root / "train.bin"
            train.write_bytes(b"training")
            config_path = self._config(root)
            run_dir = train_session(load_config(config_path), config_path, baseline, train, root / "runs", "investor-1234")
            audit = audit_session(run_dir)
            self.assertTrue(audit["candidate_matches"])


if __name__ == "__main__":
    unittest.main()
