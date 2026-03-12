#!/usr/bin/env bash
# Guaardvark Discord Bot Launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_ROOT/pids"
PID_FILE="$PID_DIR/discord_bot.pid"
LOG_FILE="$PROJECT_ROOT/logs/discord_bot.log"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export GUAARDVARK_ROOT="$PROJECT_ROOT"

if [ -z "${DISCORD_BOT_TOKEN:-}" ]; then
    echo "ERROR: DISCORD_BOT_TOKEN not set."
    echo "Set it: export DISCORD_BOT_TOKEN=your_token_here"
    exit 1
fi

API_PORT="${FLASK_PORT:-5002}"
echo "Checking Guaardvark backend at localhost:$API_PORT..."
if curl -sf "http://localhost:$API_PORT/api/health" > /dev/null 2>&1; then
    echo "Backend is online."
else
    echo "WARNING: Backend not reachable. Bot will start but commands may fail."
fi

VENV_PATH="$PROJECT_ROOT/backend/venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Using backend venv: $VENV_PATH"
else
    echo "WARNING: No venv found at $VENV_PATH. Using system Python."
fi

pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || true

mkdir -p "$PID_DIR"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing bot (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 1
    fi
    rm -f "$PID_FILE"
fi

echo "Starting Discord bot..."
cd "$PROJECT_ROOT"
python -m discord_bot.bot >> "$LOG_FILE" 2>&1 &
BOT_PID=$!
echo "$BOT_PID" > "$PID_FILE"
echo "Discord bot started (PID $BOT_PID). Logs: $LOG_FILE"
