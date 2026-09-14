from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashutil import canonical_json, sha256_bytes


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = "0" * 64
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)["entry_hash"]
        return last

    def append(self, event: str, payload: dict[str, Any]) -> str:
        entry = {
            "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event": event,
            "payload": payload,
            "prev_hash": self.last_hash,
        }
        entry_hash = sha256_bytes(canonical_json(entry))
        entry["entry_hash"] = entry_hash
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.last_hash = entry_hash
        return entry_hash


def verify_ledger(path: Path) -> str:
    prev = "0" * 64
    with path.open("r", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            claimed = entry.pop("entry_hash")
            if entry.get("prev_hash") != prev:
                raise ValueError(f"broken ledger chain at line {number}")
            actual = sha256_bytes(canonical_json(entry))
            if claimed != actual:
                raise ValueError(f"invalid ledger hash at line {number}")
            prev = claimed
    return prev
