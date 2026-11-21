#!/usr/bin/env bash
#
# Mira Core installer — idempotent. Safe to re-run.
#
#   bash scripts/setup.sh [flags]
#
# Flags:
#   --with-ocr               install tesseract (OCR for scanned PDFs)
#   --with-launchagent       install the macOS LaunchAgent (run server at login)
#   --with-tailscale <host>  configure HTTPS/Tailscale certs in the LaunchAgent
#   --skip-preflight         skip the disk + memory check
#   --force                  proceed even if disk space is low
#   -y, --yes                non-interactive (assume yes)
#   --recommended            one-step install: tuned settings, model + MTP head, the
#                            LaunchAgent, then wait until the server is healthy.
#                            (install.sh passes this by default.)
#
# Everything except `uv sync` + config is opt-in. The default backend (mira-mlx)
# is a bundled Python module and needs no separate app; its model downloads into
# the HF cache on first run. If mira.yaml selects the oMLX backend instead, that
# app is GUI-gated — this script detects it and tells you what to do, never
# claiming to have installed it for you.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

WITH_OCR=0
WITH_LAUNCHAGENT=0
TAILSCALE_HOST=""
ASSUME_YES=0
SKIP_PREFLIGHT=0
FORCE=0
RECOMMENDED=0

G="\033[32m"; R="\033[31m"; Y="\033[33m"; D="\033[2m"; N="\033[0m"
info() { printf "${Y}==>${N} %s\n" "$1"; }
ok()   { printf "${G}  ✓${N} %s\n" "$1"; }
warn() { printf "${R}  !${N} %s\n" "$1"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --with-ocr) WITH_OCR=1 ;;
    --with-launchagent) WITH_LAUNCHAGENT=1 ;;
    --with-tailscale) TAILSCALE_HOST="${2:-}"; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --force) FORCE=1 ;;
    -y|--yes) ASSUME_YES=1 ;;
    --recommended) RECOMMENDED=1; WITH_LAUNCHAGENT=1; ASSUME_YES=1 ;;
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
  /usr/bin/python3 "$REPO_ROOT/mira_cli.py" preflight "${PRE_ARGS[@]+"${PRE_ARGS[@]}"}"
fi

# ── python deps ──────────────────────────────────────────────────────────────
info "Syncing Python dependencies (uv sync)"
cd "$REPO_ROOT"
uv sync
ok "Virtualenv ready at .venv"

# ── config ───────────────────────────────────────────────────────────────────
# MIRA_CONFIG (if set) redirects the config out of the repo — the Homebrew formula
# uses it to keep mira.yaml in $HOME instead of the read-only Cellar. Mirrors
# core/config.py's resolution so setup, doctor and the server agree on one file.
MIRA_YAML="${MIRA_CONFIG:-$REPO_ROOT/mira.yaml}"
mkdir -p "$(dirname "$MIRA_YAML")"
if [ -f "$MIRA_YAML" ]; then
  ok "mira.yaml already present ($MIRA_YAML) — left untouched"
else
  cp "$REPO_ROOT/mira.yaml.example" "$MIRA_YAML"
  ok "Created mira.yaml from example ($MIRA_YAML)"
fi
if [ "$RECOMMENDED" = "1" ]; then
  MIRA_CONFIG="$MIRA_YAML" uv run python "$REPO_ROOT/mira_cli.py" profile recommended
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

# launchctl load/unload are deprecated and fail with a bare "Input/output error" when the
# label is disabled. bootstrap/bootout plus an explicit enable give a real error and work.
LA_LABEL="com.mab.mira"
LA_DOMAIN="gui/$(id -u)"
la_unload() { launchctl bootout "$LA_DOMAIN/$LA_LABEL" 2>/dev/null || true; }
la_load() {
  launchctl enable "$LA_DOMAIN/$LA_LABEL" 2>/dev/null || true
  launchctl bootstrap "$LA_DOMAIN" "$1"
}

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

  la_unload
  cp "$PLIST_OUT" "$PLIST_DST"
  if [ "$RECOMMENDED" = "1" ]; then
    ok "LaunchAgent installed (loaded after the model is ready)"
  else
    la_load "$PLIST_DST" || { warn "could not load the LaunchAgent (see the launchctl error above)"; exit 1; }
    ok "LaunchAgent loaded (server starts at login; logs at /tmp/com.mab.mira.log)"
  fi
fi

# ── Tailscale: allowlist the MagicDNS name ───────────────────────────────────
# The Tailscale cert is issued for the MagicDNS name only, so remote clients must
# connect by name. The Host-header gate rejects any name not in allowed_hosts,
# answering 403 on a healthy connection — which every client reports as "cannot
# reach server". Seed it here so the first remote connection just works.
if [ -n "$TAILSCALE_HOST" ] && [ -f "$MIRA_YAML" ]; then
  if grep -q "^[[:space:]]*-[[:space:]]*$TAILSCALE_HOST[[:space:]]*$" "$MIRA_YAML"; then
    ok "allowed_hosts already lists $TAILSCALE_HOST"
  elif grep -q "^allowed_hosts:" "$MIRA_YAML"; then
    warn "mira.yaml has allowed_hosts but not $TAILSCALE_HOST — add it by hand, or remote clients get 403"
  else
    printf '\n# Host header values accepted beyond loopback and the bare tailnet IP.\n# Added by setup.sh --with-tailscale.\nallowed_hosts:\n  - %s\n' \
      "$TAILSCALE_HOST" >> "$MIRA_YAML"
    ok "Added $TAILSCALE_HOST to allowed_hosts in mira.yaml"
  fi
fi

# ── inference backend (detect + instruct) ────────────────────────────────────
# Only oMLX needs a separate GUI app. Read the configured backend so a default
# (mira-mlx) install doesn't end on a false "oMLX missing" alarm.
BACKEND=$(sed -n 's/^backend:[[:space:]]*\([^[:space:]#"'\'']*\).*/\1/p' "$MIRA_YAML" 2>/dev/null | head -1)
BACKEND=${BACKEND:-mira-mlx}
info "Inference backend: ${BACKEND}"
if [ "$BACKEND" = "omlx" ]; then
  if [ -d "/Applications/oMLX.app" ]; then
    ok "oMLX.app found in /Applications"
  else
    warn "backend is 'omlx' but oMLX.app is not installed — the manual steps:"
    printf "${D}     1. Download from https://github.com/jundot/omlx/releases\n"
    printf "     2. Drag oMLX.app to /Applications and open it once (accept prompts)\n"
    printf "     3. In the oMLX model library, load: Qwen3.6-35B-A3B${N}\n"
  fi
else
  ok "mira-mlx is a bundled Python module — no separate app to install"
  printf "${D}     The default model downloads into the HF cache on first run (~19 GB).${N}\n"
  # Pre-fetch so the first start isn't a silent 19 GB wait. --recommended always
  # downloads (the disk check already ran); otherwise ask, reading the answer from the
  # terminal so `curl | bash` still gets asked. fetch-model is idempotent.
  REPLY_DL=""
  if [ "$RECOMMENDED" = "1" ]; then
    REPLY_DL=y
  elif [ "$ASSUME_YES" = "1" ]; then
    printf "${D}     Pre-fetch it now with:  mira fetch-model${N}\n"
  elif [ -r /dev/tty ]; then
    printf "\n"
    read -r -p "  Download the default model now (~19 GB)? [y/N] " REPLY_DL < /dev/tty || REPLY_DL=""
  fi
  case "$REPLY_DL" in
    y|Y|yes|YES)
      FETCH_ARGS=()
      MODEL_NOW=$(sed -n 's/^model:[[:space:]]*\([^[:space:]#"'\'']*\).*/\1/p' "$MIRA_YAML" | head -1)
      [ -n "$MODEL_NOW" ] && FETCH_ARGS+=(--model "$MODEL_NOW")
      # MTP head only for the 35B stock model (the small fallback has none).
      if [ "$RECOMMENDED" = "1" ] && { [ -z "$MODEL_NOW" ] || [ "$MODEL_NOW" = "mlx-community/Qwen3.6-35B-A3B-4bit" ]; }; then
        FETCH_ARGS+=(--mtp --write-config)
      fi
      MIRA_CONFIG="$MIRA_YAML" uv run python "$REPO_ROOT/mira_cli.py" fetch-model "${FETCH_ARGS[@]+"${FETCH_ARGS[@]}"}" \
        || warn "download didn't finish. Re-run: cd $REPO_ROOT && .venv/bin/mira fetch-model" ;;
    *)
      printf "${D}     Skipped. Pre-fetch later with:  cd $REPO_ROOT && .venv/bin/mira fetch-model${N}\n" ;;
  esac
fi

# ── doctor ───────────────────────────────────────────────────────────────────
echo
uv run python "$REPO_ROOT/mira_cli.py" doctor || true

echo
if [ "$RECOMMENDED" = "1" ] && [ "$WITH_LAUNCHAGENT" = "1" ]; then
  info "Starting Mira (first start loads the model, this can take a minute)"
  # Reload so the server picks up the final config written above.
  la_unload
  sleep 2
  if ! la_load "$HOME/Library/LaunchAgents/com.mab.mira.plist"; then
    warn "could not start Mira (launchctl error above). Start it by hand: cd $REPO_ROOT && make serve"
    exit 1
  fi
  UP=0
  for _ in $(seq 1 120); do
    # Wait for the ENGINE (backend_ready), not just the web server answering.
    if curl -fs --max-time 3 http://localhost:8000/health 2>/dev/null | grep -q '"backend_ready":true'; then UP=1; break; fi
    # Fail fast if launchd gave up on the job instead of waiting out the full timeout.
    if ! launchctl print "$LA_DOMAIN/$LA_LABEL" >/dev/null 2>&1; then break; fi
    sleep 3
  done
  if [ "$UP" = "1" ]; then
    ok "Mira is running:  http://localhost:8000"
    printf "${D}     It starts at login, unloads the model when idle and wakes on request.\n"
    printf "     Settings:  $MIRA_YAML   Logs: /tmp/com.mab.mira.log${N}\n"
  else
    warn "Mira didn't answer within 6 minutes. Check: tail /tmp/com.mab.mira.log"
  fi
else
  ok "Setup complete. Start the server with:  cd $REPO_ROOT && make serve"
fi
