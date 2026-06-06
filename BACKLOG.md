# Backlog

See [CHANGELOG.md](CHANGELOG.md) for recent changes.

## Pending

- [ ] Unsloth UD-MLX-4bit bench — `unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit` (15 GB) cached locally; bench vs uniform 4-bit
- [ ] Scanned PDF OCR — detect scanned PDFs (empty text layer), run OCR (e.g. `tesseract`) before indexing
- [ ] Server-side auth token check — add `verify_token` FastAPI dependency to `/chat`; reads `MIRA_TOKEN` env var; no-op if unset. Client already sends `Bearer` token. ~15 lines in `server.py`
- [ ] HTTPS on LAN — self-signed CA on startup, `.mobileconfig` endpoint, QR code sheet in mira-apps connection settings (Tailscale HTTPS already works; this covers direct LAN only)
