from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .ledger import verify_ledger
from .pipeline import run_verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ncpa-verify")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--baseline", type=Path, required=True)
    run.add_argument("--train", type=Path, required=True)
    run.add_argument("--secret-test", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--nonce", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("run_dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        run_dir = run_verification(
            config=load_config(args.config),
            config_path=args.config,
            baseline=args.baseline,
            train_data=args.train,
            secret_test=args.secret_test,
            output_root=args.output,
            nonce=args.nonce,
        )
        print(run_dir)
        return 0

    terminal = verify_ledger(args.run_dir / "ledger.jsonl")
    receipt = json.loads((args.run_dir / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("ledger_terminal_hash") != terminal:
        raise SystemExit("ledger terminal hash mismatch")
    print(terminal)
    return 0
