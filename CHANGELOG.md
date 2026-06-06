# Changelog

## v0.1.0 — June 2026

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
