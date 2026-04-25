#!/bin/bash
# Wrapper for the Mira LaunchAgent. Kills any stale server.py before starting
# so that launchd restarts after a crash never hit "port already in use".
set -e

SERVER_DIR="/Users/miguel/Documents/Projects/mira-core"
PYTHON="$SERVER_DIR/.venv/bin/python"

pkill -f "python.*server\.py" 2>/dev/null || true
sleep 0.3

cd "$SERVER_DIR"
exec "$PYTHON" server.py
