from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .hardware import probe_host, readiness
from .pipeline import audit_session, evaluate_session, run_verification, train_session
from .webapp import serve


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ncpa-verify")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--baseline", type=Path, required=True)
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--nonce")

    test = sub.add_parser("test")
    test.add_argument("run_dir", type=Path)
    test.add_argument("--secret-test", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--baseline", type=Path, required=True)
    run.add_argument("--train", type=Path, required=True)
    run.add_argument("--secret-test", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--nonce", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("run_dir", type=Path)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--config", type=Path, required=True)
    doctor.add_argument("--baseline", type=Path, required=True)
    doctor.add_argument("--workspace", type=Path, default=Path("verifier-runs"))

    ui = sub.add_parser("serve")
    ui.add_argument("--config", type=Path, default=Path("verifier.toml"))
    ui.add_argument("--baseline", type=Path, default=Path("model"))
    ui.add_argument("--workspace", type=Path, default=Path("verifier-runs"))
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "train":
        run_dir = train_session(load_config(args.config), args.config, args.baseline, args.train, args.output, args.nonce)
        receipt = json.loads((run_dir / "train_receipt.json").read_text(encoding="utf-8"))
        print(json.dumps({"run_dir": str(run_dir), "commitment": receipt["commitment"]}, ensure_ascii=False))
        return 0
    if args.command == "test":
        report = evaluate_session(args.run_dir, args.secret_test)
        print(report)
        return 0
    if args.command == "run":
        print(run_verification(load_config(args.config), args.config, args.baseline, args.train, args.secret_test, args.output, args.nonce))
        return 0
    if args.command == "audit":
        print(json.dumps(audit_session(args.run_dir), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "doctor":
        config = load_config(args.config)
        info = probe_host(args.workspace, args.baseline)
        print(json.dumps({"hardware": info, "readiness": readiness(info, require_gpu=bool(config.limits.gpus))}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    serve(args.config, args.baseline, args.workspace, args.host, args.port)
    return 0
