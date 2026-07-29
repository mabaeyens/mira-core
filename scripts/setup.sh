#!/usr/bin/env bash
#
# Mira Core installer — idempotent. Safe to re-run.
#
#   bash scripts/setup.sh [flags]
#
# Flags:
#   --with-ollama            install ollama + pull gemma4:26b (optional backend)
#   --with-ocr               install tesseract (OCR for scanned PDFs)
#   --with-launchagent       install the macOS LaunchAgent (run server at login)
#   --with-tailscale <host>  configure HTTPS/Tailscale certs in the LaunchAgent
#   --skip-preflight         skip the disk + memory check
#   --force                  proceed even if disk space is low
#   -y, --yes                non-interactive (assume yes)
#
# Everything except `uv sync` + config is opt-in. The big oMLX app and its
# model are GUI-gated, so this script detects them and tells you what to do —
# it never claims to have installed them for you.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

WITH_OLLAMA=0
WITH_OCR=0
WITH_LAUNCHAGENT=0
TAILSCALE_HOST=""
ASSUME_YES=0
SKIP_PREFLIGHT=0
FORCE=0

G="\033[32m"; R="\033[31m"; Y="\033[33m"; D="\033[2m"; N="\033[0m"
info() { printf "${Y}==>${N} %s\n" "$1"; }
ok()   { printf "${G}  ✓${N} %s\n" "$1"; }
warn() { printf "${R}  !${N} %s\n" "$1"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --with-ollama) WITH_OLLAMA=1 ;;
    --with-ocr) WITH_OCR=1 ;;
    --with-launchagent) WITH_LAUNCHAGENT=1 ;;
    --with-tailscale) TAILSCALE_HOST="${2:-}"; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --force) FORCE=1 ;;
    -y|--yes) ASSUME_YES=1 ;;
    *) warn "unknown flag: $1" ;;
  esac
  shift
done

# ── preflight ────────────────────────────────────────────────────────────────
info "Preflight"
if [ "$(uname -s)" != "Darwin" ]; then
  warn "Mira's inference backends are macOS Apple-Silicon only. Aborting."
  exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
  warn "Not Apple Silicon (arm64) — MLX backends will not work."
fi
ok "macOS $(sw_vers -productVersion 2>/dev/null || echo) / $(uname -m)"

if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { warn "uv still not on PATH — open a new shell and re-run."; exit 1; }
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# ── preflight: disk + memory ──────────────────────────────────────────────────
# Runs on system python3 (stdlib only) before the venv exists. Aborts on low disk
# (set -e) unless --force / --skip-preflight.
if [ "$SKIP_PREFLIGHT" = "0" ]; then
  PRE_ARGS=()
  [ "$ASSUME_YES" = "1" ] && PRE_ARGS+=(-y)
  [ "$FORCE" = "1" ] && PRE_ARGS+=(--force)
  [ "$WITH_OLLAMA" = "1" ] && PRE_ARGS+=(--include ollama)
  /usr/bin/python3 "$REPO_ROOT/mira_cli.py" preflight "${PRE_ARGS[@]+"${PRE_ARGS[@]}"}"
fi

# ── python deps ──────────────────────────────────────────────────────────────
info "Syncing Python dependencies (uv sync)"
cd "$REPO_ROOT"
uv sync
ok "Virtualenv ready at .venv"

# ── config ───────────────────────────────────────────────────────────────────
if [ -f "$REPO_ROOT/mira.yaml" ]; then
  ok "mira.yaml already present — left untouched"
else
  cp "$REPO_ROOT/mira.yaml.example" "$REPO_ROOT/mira.yaml"
  ok "Created mira.yaml from example"
fi

# ── optional: ollama ─────────────────────────────────────────────────────────
if [ "$WITH_OLLAMA" = "1" ]; then
  info "Setting up ollama (optional Gemma4 backend)"
  if ! command -v ollama >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then brew install ollama; else
      warn "Homebrew not found — install ollama manually from https://ollama.com"; fi
  fi
  command -v ollama >/dev/null 2>&1 && { ollama pull gemma4:26b || warn "ollama pull failed (start the ollama app first)"; ok "ollama ready"; }
fi

# ── optional: OCR (tesseract) ────────────────────────────────────────────────
if [ "$WITH_OCR" = "1" ]; then
  info "Setting up tesseract (OCR for scanned PDFs)"
  if ! command -v tesseract >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then brew install tesseract; else
      warn "Homebrew not found — install tesseract manually (https://github.com/tesseract-ocr/tesseract)"; fi
  fi
  command -v tesseract >/dev/null 2>&1 && ok "tesseract ready ($(tesseract --version 2>&1 | head -1))"
fi

# ── optional: LaunchAgent ────────────────────────────────────────────────────
if [ "$WITH_LAUNCHAGENT" = "1" ]; then
  info "Installing macOS LaunchAgent"
  PLIST_SRC="$REPO_ROOT/com.mab.mira.plist.template"
  PLIST_OUT="$REPO_ROOT/com.mab.mira.plist"
  WRAP_SRC="$REPO_ROOT/start-mira-server.sh.template"
  WRAP_OUT="$REPO_ROOT/start-mira-server.sh"
  PLIST_DST="$HOME/Library/LaunchAgents/com.mab.mira.plist"

  sed -e "s|<MIRA_DIR>|$REPO_ROOT|g" -e "s|<YOUR_HOME>|$HOME|g" "$PLIST_SRC" > "$PLIST_OUT"
  if [ -n "$TAILSCALE_HOST" ]; then
    sed -i '' -e "s|<TAILSCALE_HOST>|$TAILSCALE_HOST|g" "$PLIST_OUT"
    ok "HTTPS/Tailscale configured for $TAILSCALE_HOST (ensure certs exist at /path/to/certs)"
  else
    # Strip the optional SSL block when not using Tailscale.
    grep -v -e "Remove the two keys" -e "SSL_CERTFILE" -e "SSL_KEYFILE" -e "/path/to/certs/" \
      "$PLIST_OUT" > "$PLIST_OUT.tmp" && mv "$PLIST_OUT.tmp" "$PLIST_OUT"
  fi

  sed -e "s|<MIRA_DIR>|$REPO_ROOT|g" "$WRAP_SRC" > "$WRAP_OUT"
  chmod +x "$WRAP_OUT"

  launchctl unload "$PLIST_DST" 2>/dev/null || true
  cp "$PLIST_OUT" "$PLIST_DST"
  launchctl load "$PLIST_DST"
  ok "LaunchAgent loaded (server starts at login; logs at /tmp/com.mab.mira.log)"
fi

# ── Tailscale: allowlist the MagicDNS name ───────────────────────────────────
# The Tailscale cert is issued for the MagicDNS name only, so remote clients must
# connect by name. The Host-header gate rejects any name not in allowed_hosts,
# answering 403 on a healthy connection — which every client reports as "cannot
# reach server". Seed it here so the first remote connection just works.
if [ -n "$TAILSCALE_HOST" ] && [ -f "$REPO_ROOT/mira.yaml" ]; then
  if grep -q "^[[:space:]]*-[[:space:]]*$TAILSCALE_HOST[[:space:]]*$" "$REPO_ROOT/mira.yaml"; then
    ok "allowed_hosts already lists $TAILSCALE_HOST"
  elif grep -q "^allowed_hosts:" "$REPO_ROOT/mira.yaml"; then
    warn "mira.yaml has allowed_hosts but not $TAILSCALE_HOST — add it by hand, or remote clients get 403"
  else
    printf '\n# Host header values accepted beyond loopback and the bare tailnet IP.\n# Added by setup.sh --with-tailscale.\nallowed_hosts:\n  - %s\n' \
      "$TAILSCALE_HOST" >> "$REPO_ROOT/mira.yaml"
    ok "Added $TAILSCALE_HOST to allowed_hosts in mira.yaml"
  fi
fi

# ── oMLX (detect + instruct) ─────────────────────────────────────────────────
info "Checking oMLX (default inference backend)"
if [ -d "/Applications/oMLX.app" ]; then
  ok "oMLX.app found in /Applications"
else
  warn "oMLX.app not found. This is the only manual step:"
  printf "${D}     1. Download from https://github.com/jundot/omlx/releases\n"
  printf "     2. Drag oMLX.app to /Applications and open it once (accept prompts)\n"
  printf "     3. In the oMLX model library, load: Qwen3.6-35B-A3B${N}\n"
fi

# ── doctor ───────────────────────────────────────────────────────────────────
echo
uv run python "$REPO_ROOT/mira_cli.py" doctor || true

echo
ok "Setup complete. Start the server with: make serve  (or: mira serve)"
