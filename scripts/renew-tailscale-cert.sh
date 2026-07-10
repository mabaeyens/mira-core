#!/bin/bash
# Renews the Tailscale HTTPS cert used by mira-core and reloads the server
# so uvicorn picks up the new cert files. Tailscale issues 90-day Let's
# Encrypt certs and does not auto-renew them on its own — this must run
# periodically (see com.mab.mira-cert-renew.plist, monthly).
set -euo pipefail

CERT_DIR="/Users/miguel/Documents/Projects/mira-apps/certs"
HOSTNAME="miguels-macbook-pro.tail51ad7d.ts.net"
LOG="/tmp/com.mab.mira-cert-renew.log"

{
  echo "=== $(date) ==="
  /usr/local/bin/tailscale cert \
    --cert-file "${CERT_DIR}/${HOSTNAME}.crt" \
    --key-file "${CERT_DIR}/${HOSTNAME}.key" \
    "${HOSTNAME}"
  /bin/launchctl kickstart -k "gui/$(id -u)/com.mab.mira"
  echo "Renewed and reloaded mira server."
} >> "$LOG" 2>&1
