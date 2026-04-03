#!/usr/bin/env bash
set -euo pipefail

echo "[STARTUP] Starting FastAPI ML backend on port 8000..."
python -m uvicorn ml.server:app --host 0.0.0.0 --port 8000 --workers 1 &
FASTAPI_PID=$!

echo "[STARTUP] Waiting for FastAPI to be ready..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "[STARTUP] FastAPI ready after ${i}s"
    break
  fi
  sleep 1
done

echo "[STARTUP] Starting Express proxy on port ${PORT:-3001}..."
node server.js &
EXPRESS_PID=$!

# Wait for either process to exit — if one dies, exit so Railway restarts
wait -n $FASTAPI_PID $EXPRESS_PID
EXIT_CODE=$?
echo "[FATAL] Process exited with code $EXIT_CODE — shutting down"
kill $FASTAPI_PID $EXPRESS_PID 2>/dev/null || true
exit $EXIT_CODE
