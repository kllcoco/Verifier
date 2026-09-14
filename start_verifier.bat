@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
python -m verifier doctor --config verifier.toml --baseline model --workspace verifier-runs
if errorlevel 1 (
  echo.
  echo Hardware or runtime check failed.
  pause
  exit /b 1
)
start "" http://127.0.0.1:8765
python -m verifier serve --config verifier.toml --baseline model --workspace verifier-runs --host 127.0.0.1 --port 8765
