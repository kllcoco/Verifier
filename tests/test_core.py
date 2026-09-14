import json
import tempfile
import unittest
from pathlib import Path

from verifier.config import load_config
from verifier.hashutil import sha256_tree
from verifier.ledger import Ledger, verify_ledger


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

    def test_config(self):
        text = '''
[model]
image = "model:test"
train_command = ["train"]
eval_command = ["eval"]

[limits]
timeout_seconds = 120
hourly_cost_usd = 1.5
max_cost_usd = 2.0

[evaluation]
score_key = "accuracy"
higher_is_better = true
'''
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.toml"
            path.write_text(text, encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.model.image, "model:test")
            self.assertEqual(cfg.limits.timeout_seconds, 120)
            self.assertEqual(cfg.evaluation.score_key, "accuracy")


if __name__ == "__main__":
    unittest.main()
