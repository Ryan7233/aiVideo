#!/bin/bash
set -e

# Robustly (re)start uvicorn and wait for health using project venv

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

# Prefer project venv if present
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Ensure dependencies (install if loguru missing)
python3 - <<'PY'
try:
    import loguru  # type: ignore
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]) 
PY

# Kill existing uvicorn if any
pkill -f 'uvicorn api.main:app' >/dev/null 2>&1 || true

# Start via python -m uvicorn to honor current interpreter (venv)
nohup python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log \
  > logs/uvicorn.out 2>&1 &

# Wait for health (max ~10s)
for i in {1..20}; do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null; then
    echo "API is up on http://127.0.0.1:8000"
    exit 0
  fi
  sleep 0.5
done

echo "API failed to start. Recent logs:"
tail -n 120 logs/uvicorn.out || true
exit 1


