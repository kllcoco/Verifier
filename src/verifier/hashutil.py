from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)

    h = hashlib.sha256()
    for child in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        if child.is_symlink():
            raise ValueError(f"symlink not allowed: {child}")
        if not child.is_file():
            continue
        rel = child.relative_to(path).as_posix().encode("utf-8")
        digest = bytes.fromhex(sha256_file(child))
        size = child.stat().st_size
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(size.to_bytes(8, "big"))
        h.update(digest)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))
