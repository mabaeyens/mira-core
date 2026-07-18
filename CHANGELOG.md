# Changelog

## Unreleased

- **KV-cache quantization wired into mira-mlx, bench-validated** — `kv_bits`/`kv_group_size`/
  `quantized_kv_start` now thread end-to-end (CLI args, `mira.yaml`, disk prompt cache key)
  on top of the mlx-lm fork's quantized rotating cache. Confirmed no regression on the full
  13-question bench suite (Ministral 3 14B, Qwen3.6-35B-A3B), ~1.88x KV-cache compression, and a
  clean rotation past `max_kv_size` on a real production model. `mira_mlx_kv_bits: 8` is now
  the local default.
- **OCR fallback for image attachments on mira-mlx** — mira-mlx has no real vision seam, so
  attached images are now OCR'd via the existing tesseract path and the recovered text is
  folded into the prompt, instead of just being rejected. Falls back to a clear error when
  OCR is unavailable or finds no text.
- **Fixed: mira-mlx fallback defaults were stale** — `config.py`'s defaults (used when
  `mira.yaml` omits a key) still pointed at the old omlx-era model naming even though
  mira-mlx has been the default backend since v0.9.2.
- **Fixed: Tailscale HTTPS remote access could stay dead after a reboot** — the server only
  checked once at startup for a bindable Tailscale address; now it polls every 15s until
  Tailscale comes up. Added a monthly cert-renewal LaunchAgent for the 90-day Let's Encrypt
  Tailscale HTTPS cert (previously had no auto-renewal, so an expired cert could break iOS
  Safari access).

## v0.9.2 — July 2026

- **mira-mlx is now the default backend** — Mira's own MLX inference server
  (`core/inference/mira_mlx_server.py`) replaces omlx as the default, with RAM-aware
  sizing, a disk-backed prompt cache, and a `/v1/stats` endpoint. No separate GUI app to
  install for new setups; omlx remains fully supported as an alternative backend.
- **Mistral-family models fully supported**, including tool-calling — Ministral 3 14B
  joins Qwen3.6 as a first-class model option, servable via mira-mlx, omlx, vllm-mlx,
  ollama, or mlx-lm.
- **Fixed: Qwen3.6 wouldn't call tools on mira-mlx** — agentic actions (running shell
  commands, editing files, etc.) silently failed to fire on mira-mlx while working fine
  on omlx. Fixed three stacked bugs; re-verified 7/7 on the full agentic bench suite.
- **mira-mlx Apple Silicon tuning** — live memory stats surfaced via `/v1/stats`,
  automatic Metal cache-limit tuning, and a startup check confirming M-series GPU
  acceleration is active.
- **vllm-mlx wired end-to-end** for the Mistral family, with an agent-loop fix for
  Mistral's strict user/assistant role-alternation requirement.
- Docs (README, architecture, dev reference, model comparison) updated throughout to
  match the above.

## v0.9.1 — June 2026

- **Inference tuning results documented** — `docs/inference-tuning-2026-06-27.md` records the
  latest decode-path bench sweep: `burst_decode` (aggressive) adopted for a ~10% throughput
  gain; DFlash, MTP, and speculative prefill were evaluated and rejected for the Qwen3.6 MoE
  config (3B-active decode is too cheap to benefit from speculation). No runtime behavior
  change — documentation only.

## v0.9.0 — June 2026

Ships alongside the mobile apps v0.2.0 release.

- **Remote access hardened** — plain HTTP (`:8000`) is now loopback-only, and off-host
  access is HTTPS-only over Tailscale: the `:8443` listener binds the Tailscale interface
  (so the socket exists only on your tailnet) and **fails closed to loopback** when
  Tailscale is down. Added a source-IP allowlist and a constant-time bearer-token check.
  New **`docs/remote-access.md`** documents the posture, travelling with Tailscale (and the
  iOS Proton-VPN conflict), and the opt-in plain-LAN escape hatch.
- **Thinking toggle fixed on omlx** — the per-turn `enable_thinking` flag is now honored on
  the default omlx backend, so "thinking off" actually takes effect on Qwen3.6 (previously
  it silently fell back to the model's template default).
- **Defaults reconciled with the docs** — the code default backend is now `omlx` /
  `Qwen3.6-35B-A3B` (was `mlx-lm`); `mlx-lm` removed from the default model picker.
- **Repo cleanup** — pruned stale benchmark logs and internal process docs, refreshed
  `SECURITY.md` and `architecture.md`, and removed the duplicate legacy issue template.

## v0.8.3 — June 2026

- **Installer preflight** — `scripts/setup.sh` now runs a disk + memory check before any
  work (`mira preflight`, stdlib-only, runs on system python before the venv exists). It
  lets you pick which models count toward the budget, estimates total disk (incl. the
  GUI-gated oMLX models — the default `Qwen3.6-35B-A3B` alone is ~19 GB), and aborts if you
  don't have that plus ~15 GB breathing room (override with `--force`). Warns when RAM is
  tight: 32 GB can't co-host two large models; below 24 GB the default may OOM at large
  context. New flags `--skip-preflight` / `--force`; `mira doctor` now shows free disk.

## v0.8.2 — June 2026

- **One-command installer** — `install.sh` (curl-able bootstrap that clones to `~/mira-core`
  or reuses the current checkout) and `scripts/setup.sh` (idempotent: installs `uv`, runs
  `uv sync`, creates `mira.yaml`, optional `--with-ollama` / `--with-launchagent` /
  `--with-tailscale`, oMLX detect-and-instruct). Plus a `Makefile` (`make install` / `serve` /
  `chat` / `doctor`).
- **`mira` command** — packaged via `uv` (`uv tool install --editable .`): `mira setup`,
  `mira serve`, `mira chat`, and a stdlib-only `mira doctor` health check.
- **Packaging** — `pyproject.toml` is now a real installable package (hatchling backend,
  `[project.scripts] mira`); project renamed from `ollama-search-tool` to `mira-core`.
- README setup rewritten around the three one-line install paths.

## v0.8.1 — June 2026

### Inference

- **oMLX becomes the default backend** — replaces dFlash; KV cache held in RAM gives ~0ms
  TTFT on every new conversation after a one-time 5.5s startup warm-up (vs ~48s with dFlash
  SSD prefix cache restore). Benchmark: omlx 0.4.1 median TTFT 0ms vs ollama 0.30.6 MLX
  at 90ms vs dFlash at ~48s; all measured with the full Mira system prompt (1 488 tokens).
- **oMLX startup warm-up** — `ensure_backend_running` now seeds the system-prompt KV cache
  at server start for omlx (same pattern as existing dFlash/mlx-lm warmup); `_warmup_model`
  gains an `api_key` parameter for backends that require Bearer auth.
- `mira.yaml` updated: `backend: omlx`, `model: Qwen3.6-35B-A3B`; `prefill_step_size`
  retained (used when switching to dFlash/mlx-lm) with a note that it is ignored by omlx.
- **Multimodal vision** — Qwen3.6-35B-A3B accepts image attachments (JPEG, PNG) via oMLX.
  The orchestrator's `_normalize_messages_for_oai` already emits the correct `image_url`
  content part for all OpenAI-compatible backends; no code change was required.

### Backends and configuration

- **Dynamic model picker presets** — `GET /backends` serves the `backends:` list from
  `mira.yaml` to the iOS/macOS model picker; add or change a backend preset without pushing
  an app update (reflected on next server restart).
- **Hardcoded CLI paths removed** — `MLX_LM_CLI`, `DFLASH_CLI`, and `OMLX_CLI` moved from
  `core/backend_manager.py` to `mira.yaml` under a `paths:` section (with cross-user defaults);
  `mira.yaml.example` documents the new block; `mira.db` added to `.gitignore`.

### Conversations

- **Weekly briefing** — Mira generates a Monday briefing summarising conversations from the
  past week, delivered as a new pinned conversation. Runs automatically on the first server
  startup of each week.

### Reliability

- **Structured output robustness** — `_llm_chat_sync` retries without `response_format` if the
  backend rejects it; `generate_title` uses `re.search` to extract JSON from anywhere in the
  response, handling models that wrap JSON in prose.

## v0.8.0 — June 2026

First tagged release. Captures the backend overhaul from Ollama to mlx-lm and the dFlash
speculative decoding work, plus a series of search, RAG, and reliability improvements.

### Inference

- **dFlash speculative decoding** — integrated as the default inference backend; per-model draft
  model mapping; `prefill_step_size` and `dflash_diagnostics` exposed as `mira.yaml` tunables;
  auto-restart after OOM crash; `--max-tokens` raised to 16 384
- **mlx-lm promoted to primary backend** — replaces Ollama as the default; model warmup on
  startup eliminates the 29–34 s cold-start penalty on first request
- **Adaptive thinking** — budget cap added; tool-simulation guard prevents spurious thinking
  on trivial queries; backend switch allowlists include dFlash and mlx-lm

### RAG

- **Qwen3-Reranker-0.6B-4bit** replaces CrossEncoder as the in-process reranker (mlx, no
  external model download required)

### Search and fetch

- **Brave Search** integration added (set `brave_api_key` in `mira.yaml`)
- **Enhanced URL fetcher** — Jina fallback for JS-rendered pages that return empty content

### Conversations and memory

- **Conversation search** via SQLite FTS5
- **Scheduled reminders** with macOS Notification Center delivery
- **Cross-device conversation fix** — DB row recreated when an unknown `conversation_id` is
  supplied (handles iOS ↔ Mac handoff edge case)
- `compress_threshold` and `compress_keep_recent` exposed as `mira.yaml` tunables

### Other

- **GitHub gate** — workspace tools hidden when no local path is set and no GitHub repo is
  configured, preventing model hallucination on unavailable tools
- **Vision resize** — large images downscaled before multimodal encoding
- **Local file prompt guard** — model asks to attach file instead of searching the web when
  a filename is mentioned with no workspace open
- Soft-pause tool limits and conversation lifecycle cleanup (delete unsent turns)
