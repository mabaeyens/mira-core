#!/usr/bin/env bash
#
# Mira Core one-line bootstrap.
#
#   curl -LsSf https://raw.githubusercontent.com/mabaeyens/mira-core/main/install.sh | bash
#
# Clones the repo to ~/mira-core (override with $MIRA_HOME) on a fresh machine,
# or reuses the current checkout if you're already inside one, then hands off to
# scripts/setup.sh — which holds all the real logic. Pass any setup.sh flags
# through, e.g.:  ... | bash -s -- --with-ollama --with-launchagent
set -euo pipefail

REPO_URL="https://github.com/mabaeyens/mira-core.git"
TARGET="${MIRA_HOME:-$HOME/mira-core}"

# Already inside a checkout? Use it directly.
if [ -f "./scripts/setup.sh" ] && [ -f "./server.py" ]; then
  exec bash ./scripts/setup.sh "$@"
fi

command -v git >/dev/null 2>&1 || { echo "git is required. Install Xcode CLT: xcode-select --install"; exit 1; }

if [ -d "$TARGET/.git" ]; then
  echo "==> Updating existing checkout at $TARGET"
  git -C "$TARGET" pull --ff-only || true
else
  echo "==> Cloning mira-core into $TARGET"
  git clone "$REPO_URL" "$TARGET"
fi

exec bash "$TARGET/scripts/setup.sh" "$@"
