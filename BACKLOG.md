# Backlog

## Done

- [2026-05-24] mlx-lm benchmarked as Ollama replacement — installed `mlx-lm 0.31.3` via `uv tool install mlx-lm`; tested `Qwen3.6-35B-A3B-4bit` (44 tok/s) and `mlx-community/gemma-4-26b-a4b-it-4bit` (38 tok/s); both models cached locally. Raw throughput is competitive but thinking mode cannot be disabled in this version — TTFT of 28–62s on simple tasks makes it worse UX than Ollama live streaming. No Mira code changed. Revisit when mlx-lm exposes `max_thinking_tokens` as an API param.

- [2026-05-23] Web UI + CLI parity with mira-core backend — web UI now handles `agent_step` events (step counter in status bar), `/compact` slash command intercept, "Think" toggle (sends `thinking_enabled` flag), project panel in sidebar (list/create/delete/select projects, active project badge in header, new conversations scoped to active project), and backend loading banner with `/health` polling. CLI now renders `tool_start`/`tool_done`/`agent_step`/`compress` events and supports `/compact` command (`main.py`, `static/index.html`).
- [2026-05-23] Configurable shell timeout — `run_shell` now accepts an optional `timeout` parameter (1–300s, default 30s); model can request longer timeouts for builds/test suites; capped server-side to prevent abuse (`core/tools.py`, `core/shell_tools.py`, `core/orchestrator.py`).
- [2026-05-23] End-to-end `task_done` test — validated full agentic loop against mira-core project: 2 agent_step events, no tool_start for task_done, done event with summary, divergence guard not triggered; all assertions pass.
- [2026-05-23] Parallel tool execution — all tool calls emitted by the model in a single step now execute concurrently via `ThreadPoolExecutor(max_workers=4)`; results yielded in original order; web_search and fetch_url handled uniformly; task_done short-circuits before the thread pool fires (`core/orchestrator.py`).
- [2026-05-23] Native app UI for `agent_step` events — step counter row ("Step N — tool_name") in activity indicator area during multi-step tasks; `task_done` no longer emits `tool_start` (was causing stuck spinner); `streamingWaitMessage` now cleared on `done` event for task_done path where no tokens are streamed (`core/orchestrator.py`, `Shared/Models/ServerEvent.swift`, `ChatViewModel.swift`, `ChatView.swift`, `MessageListView.swift`).
- [2026-05-23] Agentic loop — `task_done` tool signals explicit task completion; divergence guard injects redirect after 2 identical tool+args repeats; all tool results wrapped in `{status, payload, error_details}`; `agent_step` SSE event emitted after every tool call; step cap raised to 15; RULE 7 added to system prompt (`core/orchestrator.py`, `core/tools.py`, `core/prompts.py`, `core/config.py`).
- [2026-05-23] Conciseness rule for Mira — system prompt now instructs Mira to ask one clarifying question when a request is ambiguous, produce one answer (not multiple variants), and avoid multi-paragraph explanations (`core/prompts.py`).
- [2026-05-23] Switched default model to `gemma4:26b-mlx` — MLX format cuts cold TTFT from ~31s to ~2.6s with no TPS regression; updated `core/config.py` and `docs/model-comparison-m5-macbook.md`; oMLX tombstone note added (crashed base M5 32GB, permanently abandoned).
- [2026-05-21] Released v0.1.31 (build 31) to TestFlight — iOS text selection fix + session isolation + string verification mandate.
- [2026-05-21] String verification mandate (RULE 6) — system prompt now requires Mira to use `run_shell` / `python3 -c` for any opaque string comparison >20 chars (API keys, tokens, hashes); visual inspection banned. Prevents anchor bias (stop after first mismatch found) from hiding subsequent errors (`core/prompts.py`).
- [2026-05-21] Fixed session bleed — empty `conversation_id` in `/chat` used to silently inherit the currently-loaded orchestrator session; now always creates a fresh conversation, so curl tests or clients without an explicit ID never inject into the user's active chat (`server.py`).
- [2026-05-21] Fixed iOS tap-and-hold text selection — `UIScrollView.delaysContentTouches` was true on the SwiftUI ScrollView wrapper, consuming UITextView's long-press gesture before it could fire; now disabled via post-layout superview walk (`MessageBubble.swift`).
- [2026-05-21] Fixed thinking toggle ignored in adaptive mode — `stream_chat()` in adaptive mode was fully overriding the client's `thinking_enabled` flag with the heuristic result; changed to OR so client "force on" always wins (`core/orchestrator.py`).
- [2026-05-21] SSE stability — Spec 4: backend readiness banner — `ChatViewModel` polls `/health` every 3s on startup; shows "Model loading…" spinner for first 120s, then persistent offline banner with Start button; banner clears when `backend_ready: true` (`ChatView.swift`, `ChatViewModel.swift`).
- [2026-05-21] SSE stability — Spec 3: loading state from send, not first token — `send()` sets `streamingWaitMessage = "Sending…"` immediately before any SSE event; transitions to "Thinking…" on `.thinking` event; clears on error or completion (`ChatViewModel.swift`).
- [2026-05-21] SSE stability — Spec 2: URLSession timeout — dedicated `sseSession` with `timeoutIntervalForRequest = 300` and `timeoutIntervalForResource = 3600`; health probes keep short 5s timeout (`SSEClient.swift`).
- [2026-05-21] SSE stability — Spec 1: heartbeat handling — stale-connection watchdog Task resets on every heartbeat event; 15s gap triggers reconnection logic without flashing UI (`ChatViewModel.swift`).
- [2026-05-20] Released v0.1.30 (build 30) to TestFlight — `/compact` slash command ships in this build.
- [2026-05-20] Implemented `/compact` slash command — `POST /compact` server endpoint + Swift intercept in `send()`; appends `.info` bubble with result; no-op handled gracefully for empty history.
- [2026-05-20] Improved `_should_think()` heuristic — replaced binary check with scoring function; trivial acknowledgements short-circuit; attachments, length, code signals, and think-verbs each scored; threshold ≥ 3. Reduces unnecessary thinking on casual messages.
- [2026-05-20] Added `num_keep 768` to `gemma4-optimized.modelfile` — pins system prompt tokens in KV cache across turns; saves ~200ms prefill per turn.
- [2026-05-20] Fixed invalid `llama.cpp.*` params in Modelfile — those were never valid Ollama syntax and caused `ollama create` to fail silently. kv_cache_type and flash_attn are global-only env vars.
- [2026-05-20] Added `specs/` to `.gitignore` — local pre-coding session plans, not for review.
- [2026-05-20] Refreshed `CLAUDE.md` — reflects current stack (gemma4:26b via Ollama, no oMLX).

## Pending

### Agentic loop
- [ ] End-to-end test `task_done` with a real multi-step task in a project context — verify divergence guard fires on repeated failures


### Inference speed
- [ ] mlx-lm thinking mode — `max_thinking_tokens` API param not yet exposed in mlx-lm 0.31.3; both Qwen3.6-35B-A3B-4bit and gemma-4-26b-a4b-it-4bit are cached locally at ~/.cache/huggingface/hub; re-benchmark when this lands (check mlx-lm release notes)
- [ ] Watch for quantized MLX gemma4:26b variant in Ollama registry — current `gemma4:26b` tag runs llama.cpp at ~38 tok/s; MLX path requires bf16 (52GB, won't fit 32GB RAM) and has a cold-prefill bug (#16051); revisit when a q4/q8 MLX tag appears
- [ ] ~~oMLX~~ — permanently abandoned; crashed base M5 32GB, do not revisit on this hardware

### Future / nice-to-have
- [ ] Scanned PDF OCR — detect scanned PDFs (empty text layer) and run OCR (e.g. `tesseract`) before indexing
- [ ] Persistent RAG index — ChromaDB `PersistentClient` option so documents survive server restarts

## Notes

- Projects have three modes: local-only (local_path only), GitHub-only (github_repo only), or both. Tool availability depends on local_path — no local path means fs/shell tools are hidden from the model entirely
- workspace_root flows per-call through all fs_tools and shell_tools via a `root` parameter; shell sandbox pattern rebuilt per-call from the active root (not cached at import time)
- `_LOCAL_TOOLS` set in tools.py drives filtering in orchestrator._active_tools — add new local-only tools there
- `DB_PATH` in `core/config.py` — database lives at `~/.local/share/mira/conversations.db`
- The `mira-server` skill copies plist from `~/Documents/Projects/mira-core/com.mab.mira.plist`; reload required any time the plist or server paths change
- Mac app bundle ID remains `com.mab.OllamaSearch` (no rebuild needed for Python-side refactors)
- `_ollama_ready` is set True after the warm-up loop regardless of success/failure — app never hangs indefinitely if Ollama is permanently down
- Model validation at startup uses `client.list()` (all installed models), not `client.ps()` (only in-memory loaded models) — do not revert
- `OLLAMA_KV_CACHE_TYPE` is global-only — cannot be overridden per model via Modelfile; currently set to `q8_0` in `~/.zprofile`; going to `q4_k` would save ~4GB KV at 64k ctx but affects all models
- gemma4:26b is a MoE model (4B active params), not dense — decode at ~38 tok/s on llama.cpp; Ollama 0.24 MLX path not yet viable for this model (size + cold prefill bug)
- mlx-lm 0.31.3: all tested models (Qwen3.6-35B-A3B, gemma-4-26b) have thinking mode forced on with no API disable path; streaming is buffered (silent wait then burst), not live; raw tok/s is 38–44 on M5 base 24GB but TTFT of 28–62s makes it worse than Ollama for production; installed at `~/.local/bin/mlx_lm.server`
- Agentic Critic LLM call considered and rejected — proposed plan called for a secondary model call after every tool observation to decide CONTINUE/FINISH; replaced with `task_done` tool + ReAct-style prompting (RULE 7); zero extra latency, same effect
- Router Agent considered and rejected — adds ~300ms overhead to every complex query (majority of Mira sessions); heuristic-based thinking toggle is the right lever at zero cost
