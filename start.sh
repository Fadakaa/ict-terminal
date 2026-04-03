#!/bin/bash
# ICT Terminal — start everything
cd "$(dirname "$0")"
source ~/dealfinder/bin/activate

# Kill any stale processes
lsof -ti:5173 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1

# Start backend (FastAPI + scanner)
python -m uvicorn ml.server:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend (Vite dev server with proxy)
npx vite --port 5173 &
FRONTEND_PID=$!

echo ""
echo "✓ Backend:  http://localhost:8000"
echo "✓ Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
