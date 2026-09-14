from __future__ import annotations

import json
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import load_config
from .hardware import probe_host, readiness
from .pipeline import audit_session, evaluate_session, train_session


_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NCPA Verifier</title><style>body{font-family:system-ui;margin:0;background:#0b0d10;color:#eee}.wrap{max-width:820px;margin:40px auto;padding:20px}.card{background:#15191e;border:1px solid #2a3038;border-radius:14px;padding:22px;margin:18px 0}input,button{font:inherit}input[type=file],input[type=text]{width:100%;box-sizing:border-box;margin:10px 0;padding:12px;background:#0f1216;color:#eee;border:1px solid #343b45;border-radius:9px}button{padding:12px 18px;border:0;border-radius:9px;cursor:pointer;font-weight:700}.primary{background:#eee;color:#111}.muted{color:#9aa4af}pre{white-space:pre-wrap;word-break:break-all;background:#0f1216;padding:14px;border-radius:9px}</style></head><body><div class="wrap"><h1>NCPA Verifier</h1><div class="card"><h2>1. 训练</h2><input id="train" type="file"><input id="nonce" type="text" placeholder="Investor nonce（可选）"><button class="primary" onclick="startTrain()">上传并开始训练</button></div><div class="card"><h2>2. 投资人测试</h2><input id="test" type="file" onchange="startTest()"><div class="muted">训练完成后，上传题目即自动开始黑盒测试。</div></div><div class="card"><h2>状态</h2><pre id="status">loading...</pre></div></div><script>const token='__TOKEN__';async function req(url,opt={}){opt.headers=Object.assign({'X-Verifier-Token':token},opt.headers||{});let r=await fetch(url,opt);let t=await r.text();if(!r.ok)throw new Error(t);return t?JSON.parse(t):{};}async function upload(url,file){return req(url,{method:'PUT',headers:{'Content-Type':'application/octet-stream','X-File-Name':file.name},body:file});}async function startTrain(){try{let f=document.getElementById('train').files[0];if(!f)throw new Error('请选择训练数据');await upload('/api/train-data',f);let nonce=document.getElementById('nonce').value.trim();await req('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({nonce})});}catch(e){alert(e.message)}}async function startTest(){try{let f=document.getElementById('test').files[0];if(!f)return;await upload('/api/test-data',f);await req('/api/test',{method:'POST'});}catch(e){alert(e.message)}}async function poll(){try{let s=await req('/api/status');document.getElementById('status').textContent=JSON.stringify(s,null,2);}catch(e){document.getElementById('status').textContent=e.message}setTimeout(poll,1000)}poll();</script></body></html>'''


class AppState:
    def __init__(self, config_path: Path, baseline: Path, workspace: Path):
        self.config_path = config_path.resolve()
        self.baseline = baseline.resolve()
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.inbox = self.workspace / "inbox"
        self.inbox.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.busy = False
        self.phase = "IDLE"
        self.message = "ready"
        self.run_dir: Path | None = None
        self.report_path: Path | None = None
        self.error: str | None = None
        self.token = secrets.token_urlsafe(24)
        self._load_active()

    def _active_path(self) -> Path:
        return self.workspace / "active_session.json"

    def _load_active(self) -> None:
        path = self._active_path()
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                run_dir = Path(data["run_dir"])
                if run_dir.is_dir():
                    self.run_dir = run_dir
                    self.phase = "TRAINED"
                    self.message = "trained session restored"
            except Exception:
                pass

    def _save_active(self) -> None:
        if self.run_dir:
            self._active_path().write_text(json.dumps({"run_dir": str(self.run_dir)}), encoding="utf-8")

    def _configuration_state(self) -> tuple[Any | None, list[str]]:
        issues: list[str] = []
        if not self.config_path.is_file():
            issues.append(f"configuration missing: {self.config_path.name}")
            config = None
        else:
            try:
                config = load_config(self.config_path)
            except Exception as exc:
                issues.append(f"configuration invalid: {exc}")
                config = None
        if not self.baseline.exists():
            issues.append(f"baseline model missing: {self.baseline.name}")
        return config, issues

    def status(self) -> dict[str, Any]:
        config, setup_issues = self._configuration_state()
        hardware = probe_host(self.workspace, self.baseline if self.baseline.exists() else None)
        if config is None:
            ready = {"ready": False, "issues": setup_issues}
        else:
            ready = readiness(hardware, require_gpu=bool(config.limits.gpus))
            if setup_issues:
                ready = {"ready": False, "issues": [*setup_issues, *ready["issues"]]}
        result: dict[str, Any] = {
            "phase": self.phase,
            "busy": self.busy,
            "message": self.message,
            "error": self.error,
            "hardware": hardware,
            "readiness": ready,
        }
        if self.run_dir:
            result["run_dir"] = str(self.run_dir)
            try:
                result["audit"] = audit_session(self.run_dir)
            except Exception as exc:
                result["audit_error"] = str(exc)
        if self.report_path and self.report_path.is_file():
            result["report"] = json.loads(self.report_path.read_text(encoding="utf-8"))
        return result

    def _require_ready(self):
        config, issues = self._configuration_state()
        if config is None or issues:
            raise RuntimeError("; ".join(issues) or "verifier is not configured")
        hardware = probe_host(self.workspace, self.baseline)
        state = readiness(hardware, require_gpu=bool(config.limits.gpus))
        if not state["ready"]:
            raise RuntimeError("; ".join(state["issues"]))
        return config

    def _start(self, target, phase: str) -> None:
        with self.lock:
            if self.busy:
                raise RuntimeError("another operation is already running")
            self.busy = True
            self.phase = phase
            self.error = None
        def worker():
            try:
                target()
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.phase = "ERROR"
                    self.message = str(exc)
            finally:
                with self.lock:
                    self.busy = False
        threading.Thread(target=worker, daemon=True).start()

    def start_training(self, train_path: Path, nonce: str | None) -> None:
        config = self._require_ready()
        def work():
            run_dir = train_session(config, self.config_path, self.baseline, train_path, self.workspace / "runs", nonce or None)
            receipt = json.loads((run_dir / "train_receipt.json").read_text(encoding="utf-8"))
            with self.lock:
                self.run_dir = run_dir
                self.phase = "TRAINED"
                self.message = f"training complete; commitment={receipt['commitment']}"
                self._save_active()
        self._start(work, "TRAINING")

    def start_test(self, test_path: Path) -> None:
        if not self.run_dir:
            raise RuntimeError("no trained session")
        run_dir = self.run_dir
        def work():
            report = evaluate_session(run_dir, test_path)
            with self.lock:
                self.report_path = report
                self.phase = "COMPLETE"
                self.message = "black-box evaluation complete"
        self._start(work, "EVALUATING")


class Handler(BaseHTTPRequestHandler):
    server_version = "Verifier"

    @property
    def app(self) -> AppState:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _auth(self) -> bool:
        return self.headers.get("X-Verifier-Token") == self.app.token

    def _json(self, status: int, value: Any) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/":
            data = _HTML.replace("__TOKEN__", self.app.token).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/api/status":
            if not self._auth():
                self._json(403, {"error": "forbidden"})
                return
            try:
                self._json(200, self.app.status())
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return
        self._json(404, {"error": "not found"})

    def _upload(self, target: Path) -> None:
        if not self._auth():
            self._json(403, {"error": "forbidden"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._json(400, {"error": "empty upload"})
            return
        tmp = target.with_suffix(target.suffix + ".tmp")
        remaining = length
        with tmp.open("wb") as f:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ConnectionError("upload ended early")
                f.write(chunk)
                remaining -= len(chunk)
        tmp.replace(target)
        self._json(200, {"bytes": length})

    def do_PUT(self) -> None:
        if self.path == "/api/train-data":
            self._upload(self.app.inbox / "train.data")
            return
        if self.path == "/api/test-data":
            self._upload(self.app.inbox / "test.data")
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._auth():
            self._json(403, {"error": "forbidden"})
            return
        try:
            if self.path == "/api/train":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b"{}"
                payload = json.loads(body or b"{}")
                train = self.app.inbox / "train.data"
                if not train.is_file():
                    raise RuntimeError("training data has not been uploaded")
                self.app.start_training(train, str(payload.get("nonce") or "") or None)
                self._json(202, {"started": True})
                return
            if self.path == "/api/test":
                test = self.app.inbox / "test.data"
                if not test.is_file():
                    raise RuntimeError("test data has not been uploaded")
                self.app.start_test(test)
                self._json(202, {"started": True})
                return
            self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(409, {"error": str(exc)})


def serve(config_path: Path, baseline: Path, workspace: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    app = AppState(config_path, baseline, workspace)
    server = ThreadingHTTPServer((host, port), Handler)
    server.app = app  # type: ignore[attr-defined]
    print(f"http://{host}:{port}")
    server.serve_forever()
