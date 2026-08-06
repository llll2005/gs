#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8080
EDGE="/opt/microsoft/msedge-beta/microsoft-edge-beta"
URL="http://localhost:$PORT"

# kill any previous serve.py on this port
if lsof -ti tcp:$PORT &>/dev/null; then
  echo "Stopping existing server on port $PORT..."
  lsof -ti tcp:$PORT | xargs kill 2>/dev/null || true
  sleep 0.5
fi

# start serve.py in background, log to /tmp/citygs_serve.log
echo "Starting serve.py on $URL ..."
nohup python "$SCRIPT_DIR/serve.py" "$PORT" \
  > /tmp/citygs_serve.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# wait until server is up (max 5s)
for i in $(seq 1 10); do
  if curl -sf "$URL/files.json" &>/dev/null; then
    break
  fi
  sleep 0.5
done

echo "Opening $URL in Edge..."
"$EDGE" --new-window "$URL" &>/dev/null &

echo "Done. Server log: /tmp/citygs_serve.log"
