# Changelog

## v0.9.0 — June 2026

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

### Backends and model picker

- **Dynamic model picker presets** — `GET /backends` serves the `backends:` list from
  `mira.yaml` to connected clients. Adding or changing a backend preset in `mira.yaml` is
  reflected in the app picker on next server restart — no app update required.

### Conversations

- **Weekly briefing** — Mira generates a Monday briefing summarising conversations from the
  past week, delivered as a new pinned conversation. Runs automatically on the first server
  startup of each week.

## v0.8.1 — June 2026

### Backends and configuration

- **Dynamic model picker presets** — `GET /backends` serves the `backends:` list from `mira.yaml`
  to the iOS/macOS model picker; add or change models without pushing an app update
- **Hardcoded CLI paths removed** — `MLX_LM_CLI`, `DFLASH_CLI`, and `OMLX_CLI` moved from
  `core/backend_manager.py` to `mira.yaml` under a `paths:` section (with cross-user defaults);
  `mira.yaml.example` documents the new block; `mira.db` added to `.gitignore`

### Reliability

- **Structured output robustness** — `_llm_chat_sync` retries without `response_format` if the
  backend rejects it; `generate_title` uses `re.search` to extract JSON from anywhere in the
  response, handling models that wrap JSON in prose

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
