# Architecture Reference

Detailed reference for subsystems. Read this file when working on events, RAG, cancel, config, or model quirks.

## Module map

```
main.py (CLI)     server.py (FastAPI + SSE)
        └─────────────────┘
                 │
      core/orchestrator.py → ChatOrchestrator
            │  stream_chat(user_message, attachments=None, thinking_enabled=True) → yields events
            ├── _call_llm() → openai.chat.completions.create() (mira-mlx/omlx/mlx-lm/vllm-mlx — all OpenAI-compatible, one code path)
            ├── core/search_engine.py → SearchEngine → Brave API (primary, if keyed) / ddgs.text() (fallback)
            ├── core/url_fetcher.py → fetch_url() → BeautifulSoup, Jina Reader fallback
            └── core/rag_engine.py → RagEngine
                      ├── SentenceTransformer (nomic-ai/nomic-embed-text-v1.5, local, 768 dims)
                      ├── chromadb: PersistentClient per project (survives restarts) / EphemeralClient with no project open
                      └── reranker: Qwen3-Reranker-0.6B-4bit (mlx, default) or CrossEncoder ms-marco (sentence-transformers)

core/config.py       — all tunables; loads overrides from mira.yaml (git-ignored)
core/file_handler.py — load_file() / load_file_bytes(): PDF→RAG, HTML→text, image→base64
core/tools.py        — OpenAI-compatible tool schema for web_search and all tools
core/prompts.py      — build_system_prompt() injects today's date + search rules
core/formatter.py    — Rich console helpers (CLI only)
core/db.py           — SQLite conversation persistence
core/workspace.py    — sandbox path enforcement
core/fs_tools.py     — filesystem tool implementations
core/shell_tools.py  — shell execution tool
core/github_tools.py — GitHub API tool
static/index.html    — single-page web UI (vanilla HTML/CSS/JS + marked.js)
```

## Event protocol

`ChatOrchestrator.stream_chat()` yields typed dicts consumed by both CLI (`main.py`) and web server (`server.py`):

| Event | Payload | Meaning |
|-------|---------|---------|
| `thinking` | `content` (optional) | Model is processing; thinking text arrives in `content`. Two source shapes: a `delta.reasoning`/`reasoning_content` field (omlx), or inline `<think>` markers in `delta.content` that `ThinkingStripper` separates (mira-mlx) |
| `token` | `content` | Answer token (buffered by CLI, streamed by web) |
| `search_start` | `query` | Web search beginning |
| `search_done` | `query, count, results` | Search complete; `results` is `[{title, url}]` (snippet stripped before SSE) |
| `fetch_start` | `url` | Page fetch beginning |
| `fetch_done` | `url, chars` | Fetch complete; `chars` is length of text returned |
| `fetch_context` | `fetches` | All pages fetched this turn; `[{url, chars, preview}]` — emitted right before `done` |
| `rag_indexing` | `name` | Document being indexed |
| `rag_done` | `name, chunks` | Indexing complete |
| `rag_context` | `chunks` | RAG chunks injected this turn; `[{source, score, preview}]` — emitted right before `done` |
| `stats` | `input_tokens, output_tokens, context_pct` | Cumulative session token counts — emitted just before `done` |
| `warning` | `message` | Non-fatal issue (scanned PDF, chunk limit) |
| `done` | `content` | Turn complete, full answer |
| `error` | `message` | Fatal error |
| `title` | `conv_id, title` | New conversation title generated — emitted after `done`, only on first turn |
| `compress` | `message` | Context window compressed — emitted after `done` when `context_pct` exceeded threshold |
| `heartbeat` | — | Keepalive — emitted periodically during long tool calls to prevent connection timeout |

**Thinking toggle:** `stream_chat()` accepts `thinking_enabled: bool = True`. The orchestrator decides whether to think via `_should_think(message, has_attachments)` — a scoring heuristic that also checks `_NEVER_THINK` (trivial commands: time/date queries, "fix file.ext" patterns) and `_REASONING_INTENT` (strong signals that always trigger thinking). On **Qwen3**, thinking is controlled per-request via `extra_body={"chat_template_kwargs": {"enable_thinking": True/False}}`; the `reasoning` field in SSE delta chunks carries thinking tokens and is emitted as `thinking` events. Warm thinking overhead on mlx-lm is ≤14 ms (noise). The server form field `thinking_enabled` (default `true`) flows from the app's thinking toggle.

**Mockable boundary:** `_call_llm()` is the single point tests mock — returns an iterable of stream chunks with `.message.content`, `.message.tool_calls`, `.done`.

## RAG internals

**Flow:** PDFs always go through RAG. HTML/text > 80k chars also go through RAG. On every turn where the RAG index is non-empty, `rag_engine.query()` retrieves and reranks chunks; those scoring > `RAG_SCORE_THRESHOLD` (default 0.0) are prepended to the user message as `[Relevant document sections]`.

**Score threshold bypass (anti-hallucination):** When the user attaches a file in the current turn (`rag_indexed_this_turn = True` in `orchestrator.py`), `query()` is called with `score_threshold=float('-inf')`, retrieving top-K chunks unconditionally. On subsequent turns the normal threshold applies.

Why this matters: meta-instructions like "summarize this" or "translate this to English" embed nothing like document content, so all chunks score negative and get dropped without the bypass. The model then receives zero context and hallucinates that no file was attached ("No has adjuntado ningún artículo"). Do not remove this bypass.

## File loading

Two paths into `stream_chat(attachments=[...])`:
- CLI: `/attach <path>` → `file_handler.load_file(path)`
- Web upload: multipart → `load_file_bytes(name, data)`
- Web path field: `paths[]` form field → `load_file(path)` server-side (same machine, absolute path)

## Cancel / Stop mechanism

- `cancel_event = threading.Event()` is module-level in `server.py`
- Each `/chat` request clears it; `POST /cancel` sets it
- `produce()` thread checks `cancel_event.is_set()` before each `queue.put_nowait` — breaks out immediately when set
- `event_stream()` receives the `None` sentinel from `produce()`, checks cancel_event, and truncates `orchestrator.conversation_history` back to pre-turn length — as if the turn never happened
- Client JS awaits `POST /cancel` first, then calls `reader.cancel()` to close the SSE stream

**Key constraint:** History rollback relies on `event_stream()` receiving the `None` sentinel before the SSE connection closes. If the connection drops before `None` arrives (network issue, browser crash), history is left with the partial user message — not catastrophic, user can hit Reset.

## Context compression

When `context_pct` exceeds `COMPRESS_THRESHOLD` (default 70%), the orchestrator compresses conversation history at the end of the turn:

- Keeps the system prompt and the most recent `COMPRESS_KEEP_RECENT` messages (default 6) verbatim
- Summarises older messages into a single assistant message using the LLM
- Emits a `compress` event after `done` so clients can show a notice
- The compressed history is written back to `orchestrator.conversation_history` and saved to the database

This is transparent to clients — the next turn proceeds normally with a shorter history. Token counts in `stats` reflect the compressed window going forward.

## Endpoint reference

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/chat` | Stream a turn (multipart: `message`, `conversation_id`, `files[]`, `paths[]`) |
| `POST` | `/cancel` | Abort in-progress response; triggers history rollback |
| `POST` | `/reset` | New conversation (preserves active project) |
| `GET` | `/health` | `200` ready, `503` starting, other → unavailable |
| `GET` | `/status` | Model name, cumulative tokens, `context_pct`, `workspace_root` |
| `GET` | `/browse` | Directory listing (sandboxed to `$HOME`); query param `path` |
| `POST` | `/ask` | One-shot ephemeral query — no tools, no DB writes |
| `GET/POST` | `/projects` | List / create projects |
| `DELETE` | `/projects/{id}` | Delete project |
| `GET/POST` | `/conversations` | List / create conversations |
| `PATCH/DELETE` | `/conversations/{id}` | Rename / delete conversation |
| `GET` | `/conversations/{id}/messages` | Full message history |
| `GET` | `/info` | Model name, backend, host, context_window, hardware |
| `GET` | `/backend` | Current backend/model/host/context_window |
| `POST` | `/backend` | Switch inference backend, keeping that backend's default model (`{"backend": "mira-mlx"}`); blocks until ready |
| `POST` | `/models/switch` | Switch to a specific backend + model (`{"backend": "mira-mlx", "model_id": "mlx-community/Ministral-3-14B-Instruct-2512-4bit"}`); blocks until ready |
| `GET` | `/backends` | Named backend presets from `mira.yaml` `backends:` list, with `active` flag; populates app model picker |
| `GET` | `/rag/documents` | List indexed RAG documents |
| `DELETE` | `/rag/documents/{name}` | Remove a RAG document |

## iOS/macOS client integration

The native clients (mira-apps) connect to this server over HTTP/HTTPS. Key integration points:

- **Discovery:** macOS connects to `localhost:8000`; iOS uses a saved or user-configured URL (Tailscale over HTTPS `:8443`). Bonjour/mDNS discovery was removed from the apps — the clients are HTTP/HTTPS only.
- **SSE streaming:** `SSEClient.swift` opens `POST /chat` as an `AsyncThrowingStream<ServerEvent>`, parsing each `data:` line as JSON.
- **Event mapping:** All events in the table above have a corresponding `ServerEvent` Swift enum case consumed by `ChatViewModel`.
- **Cancel:** iOS/macOS send `POST /cancel` then discard the stream; the server rolls back history.
- **File uploads:** Sent as multipart form-data, same schema as the web UI.
- **`title` and `compress` events** arrive after `done`; clients must keep the SSE connection open until the server closes it (signalled by the absence of further events, not by a sentinel).
- **Model switcher:** toolbar label button opens `ModelPickerView` sheet. The picker fetches `GET /backends` on load — a list of named presets defined in `mira.yaml` — so adding a new model+backend combo requires only a server config edit, no app update. Tapping an inactive preset shows a confirmation step (warns about 30–60 s pause), then calls `POST /models/switch`. `ChatViewModel.switchModel(backend:modelId:)` drives `switchStatusMessage` through timed stages ("Stopping…", "Starting…", "Loading weights…", "Almost ready…") which `ModelPickerView` displays during the switch.
- **Thinking toggle:** brain icon in `InputBar` toggles `thinkingEnabled`; passed as `thinking_enabled` form field to `POST /chat`. Honoured by every backend.

See `mira-apps/OllamaSearch/Shared/Networking/` for client implementation.

## Search providers (Brave primary, DDGS fallback)

`SearchEngine.search()` tries providers in order:

1. **Brave Search** — used whenever `brave_api_key` (in `mira.yaml`) or the `BRAVE_API_KEY` env var is set. This is the primary provider in normal operation. The free tier is **2,000 queries/month at ~1 query/second**; the key never gets committed (`mira.yaml` is git-ignored — only `mira.yaml.example` ships, with a placeholder).
2. **DuckDuckGo (ddgs)** — the no-key, no-tracking fallback. Serves requests whenever Brave is unkeyed or errors (including rate-limit/429 responses, which currently fall through silently rather than retrying).

There used to be a third, first-in-order provider: Ollama's native web search, gated behind `USE_NATIVE_SEARCH`. It was never switched on (signup wants a phone number, queries tie to an authenticated account with no privacy disclosure, and the model in use was not supported), so it sat as dead code until the Ollama backend was retired on 2026-08-01. Both the flag and the branch are gone.

## Backend startup

On Mira startup, `backend_manager.ensure_backend_running()` launches whichever backend `mira.yaml` configures (default: **mira-mlx**, port 8080 — `core/inference/mira_mlx_server.py`, spawned as `python -m core.inference.mira_mlx_server`; the other options are omlx, mlx-lm and vllm-mlx). The app's `/health` endpoint returns `backend_ready: false` until the inference server is reachable with the configured model, and the iOS/macOS chat view shows a banner with a "Start" button during that time.

`KNOWN_BACKENDS` in `backend_manager.py` is the single list every other check derives from — `server.py` validates `POST /backend` against it rather than repeating the names in string literals, and `switch_to`/`switch_to_model` dispatch through a `_STARTERS` table after `_stop_all_backends()`. An unknown name raises. This matters because the old code path had an implicit `else` that meant "start ollama", so a typo used to start the wrong backend instead of failing.

**Retired 2026-08-01: dflash and ollama.** Both are gone from the code, not just hidden from the picker, and their Python dependencies (`ollama`, `dflash-mlx`) came out of `pyproject.toml` with them. No model coverage was lost: ollama only ever served `ministral-3:14b`, which runs on mira-mlx, mlx-lm and vllm-mlx, and Gemma 4 is still reachable through omlx. The `ollama` key survives in `GET /models`, always empty, so an older app build that still decodes that field does not fail on a missing key.

After the server is confirmed reachable, `_warmup_model(...)` sends a 1-token completion request (non-streaming) to force the model into GPU memory. This eliminates the first-request penalty that occurs when a backend must load a new model into VRAM (mira-mlx: RAM-aware sizing already derives context/cache budgets at spawn time via `core/hardware.py`, so this is mainly relevant for backends that lazy-load). The warmup runs in the background startup thread — both on fresh start and when the server was already running.

### mira-mlx specifics

`core/inference/mira_mlx_server.py` is Mira's own MLX inference server (built on `mlx-lm`'s `BatchGenerator` continuous batching), promoted to the default backend on 2026-07-09. It runs in a single dedicated "engine thread" per process — MLX streams are thread-local, so HTTP handlers only tokenize and hand work off through a `queue.Queue`; they never touch MLX directly. Notable pieces:

- `core/hardware.py` — derives per-machine RAM-aware budgets (prompt-cache pool size, context window ceiling, disk-cache size) so the same code behaves sensibly from an 8 GB Mac to a 128 GB Mac Studio; also computes a proactive Metal cache limit and confirms M-series GPU acceleration (Metal-4/NAX) is active at startup.
- `core/inference/disk_prompt_cache.py` — evicted prompt-cache entries overflow to disk (content-addressed, safetensors) instead of being discarded, surviving both memory-pressure trims and process restarts.
- `GET /v1/stats` on the mira-mlx port — cache hit/miss rate, disk-cache hits, memory-pressure trim events, latency percentiles, and live MLX memory (active/cache/peak/wired-limit bytes).
- Oversized single prompts (≥ the derived context ceiling) are rejected with a clear `ValueError` rather than left to `RotatingKVCache`'s undefined behavior.
- mlx-lm itself is pinned to a mira-owned fork (`github.com/mabaeyens/mlx-lm`) at an explicit commit, not a branch, so a force-push upstream cannot change the installed tree. The SHA lives in exactly one place — `pyproject.toml`, explained in **[docs/mlx-lm-pin.md](mlx-lm-pin.md)**, which also covers what the fork carries and when it is worth moving. Do not restate the SHA here; it goes stale the day it moves.
- Vision is optional and off by default (`mira_mlx_vision` in `mira.yaml`). See "Vision on mira-mlx" below.

## Configuration reference

**External config (preferred):** copy `mira.yaml.example` → `mira.yaml` (git-ignored) and edit.

All 45 settings, with defaults, are in **[configuration.md](configuration.md)**. Do not restate
them here — three partial copies of that table is how `reranker_model` came to be documented as
source-only when it had been a config field for weeks.

Two mechanics worth knowing that belong here rather than there:

- `_get(key, default)` in `core/config.py` is the single reader. A key that is not passed
  through `_get` is not a `mira.yaml` setting, whatever the example file suggests.
- `host` is named `BACKEND_HOST` in `config.py`. It was `OLLAMA_HOST` until 2026-08-01, which
  made a retired backend look load-bearing.

**Knobs that genuinely have no `mira.yaml` equivalent** (edit `core/config.py`):
`MAX_SEARCH_RESULTS`, `SEARCH_TIMEOUT`, `MAX_RETRIES`, `MAX_TOOL_STEPS`, `VERBOSE_DEFAULT`,
`RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_RETRIEVE_K`, `RAG_RERANK_TOP_K`,
`RAG_SCORE_THRESHOLD`, `RAG_MAX_CHUNKS`.

## Model quirks

Behaviours that are intentional and must not be removed:

- **Gemma4 (mlx-lm):** emits `tool_calls` in an intermediate chunk (`done=False`) — handled by `accumulated_tool_calls` in `orchestrator.py`.
- **Gemma4:** occasionally emits LaTeX (e.g. `$\rightarrow$`) — `preprocessLatex()` in `index.html` converts to Unicode.
- **Qwen3.6 thinking:** controlled per-request via `extra_body={"chat_template_kwargs": {"enable_thinking": True/False}}`. Where the reasoning text then *arrives* is backend-specific and this is the part that bites: omlx splits it into a `reasoning`/`reasoning_content` delta field, while **mira-mlx leaves it inline in `delta.content`** wrapped in `<think>` markers, so the orchestrator's `ThinkingStripper` is what separates the two. Either way the orchestrator emits `thinking` events and only the answer reaches `token` events. Overhead on mlx-lm is ≤14 ms warm, negligible.
- **`_NEVER_THINK` / `_ANALYTICAL_WITH_ATTACHMENT`:** `_should_think()` uses a scoring heuristic (threshold 4). `_NEVER_THINK` short-circuits to False for time/date queries and "fix file.ext" patterns. Bare attachments score +1; analytical verbs (explain, analyze, review, debug, refactor…) combined with an attachment add a further +2.
- **Qwen3.6 thinking toggle must be threaded through per-backend:** `_call_llm()` (`core/orchestrator.py`) gates the `enable_thinking`/`chat_template_kwargs` override behind an explicit backend allow-list — every OpenAI-compatible backend that honors it (`mira-mlx`, `omlx`, `mlx-lm`, `vllm-mlx`) must be listed there, or that backend silently falls back to the model's chat-template default (thinking always on) regardless of the per-turn toggle. Found and fixed 2026-07-10 when `mira-mlx` was missing from the tuple. The predicate is now `_uses_qwen_thinking_template(backend, model)`, shared with the response side below so the two cannot drift apart.

- **The same template that honors `enable_thinking` also pre-opens the tag, and the stripper has to know:** Qwen3's `chat_template.jinja` appends a bare `<think>\n` to the *prompt* when thinking is on, and a pre-closed `<think>\n\n</think>\n\n` when it is off. So on a thinking turn the model's output starts *inside* the block and only ever emits the closing tag. `ThinkingStripper` used to enter its thinking state only on a literal `<think>` in the output, which therefore never arrived: every reasoning turn on mira-mlx served the whole chain of thought to the user as the answer, with a stray `</think>` in the middle (`find("<think>")` does not match `</think>`), and `thinking_chars` stayed 0 so `ctxmgr.thinking_tokens()` undercounted every thinking turn. The polluted text was then appended to `conversation_history` and persisted. Fixed 2026-08-01: `ThinkingStripper(preopened=True)` starts inside the block, `saw_reasoning()` disarms that when a backend delivers reasoning out of band (otherwise `content` holds only the answer and would be swallowed whole), and `drain()` reclassifies an unclosed pre-opened block as the answer so a model that never closes the tag cannot produce an empty assistant message. Thinking *off* was never affected, because that template branch emits no tags at all.
- **Mistral/Ministral tool calls (mira-mlx):** Mistral's chat template uses a one-sided `[TOOL_CALLS]` marker (no closing token — the model relies on EOS to end the call), while Qwen's uses a two-sided `<tool_call>...</tool_call>` marker. `mira_mlx_server.py`'s tool-text buffering has a `prev_state == "tool"` fallback to still capture the last chunk before EOS for Mistral's one-sided case — but for two-sided markers this same fallback also captures the closing marker token itself, corrupting the parser's input. The closing marker is stripped before parsing (no-op for Mistral, which has none).
- **Mistral agent-loop role alternation (vllm-mlx):** Mistral-family chat templates require strict user/assistant role alternation; the agent tool-call loop's message construction was fixed to satisfy this when Mistral is the active model.
- **Images take one of two paths, decided by a preset flag:** `backend_manager.PRESETS[backend]["vision"]`. When it is `False`, `orchestrator.py` runs each attached image through `file_handler.ocr_image_from_base64()` (system `tesseract`, the same optional dependency scanned PDFs use) and folds the recovered text into the prompt as a regular text block; `_prepare_messages()` in `mira_mlx_server.py` still raises a `ValueError` (→ 400) on an `image_url` part as a backstop. OCR handles the common troubleshooting case — error dialogs, menus, terminal output — and is genuinely cheaper than vision for text-heavy screenshots. When it finds nothing (photos, diagrams, layout questions) the user gets an inline error suggesting `brew install tesseract` or a vision-capable backend. `omlx` has `vision: True` unconditionally; mira-mlx's flag follows `mira_mlx_vision` in `mira.yaml`.

## Vision on mira-mlx

Optional, off by default, added 2026-08-01. Set `mira_mlx_vision: true` in `mira.yaml`. It only works on a checkpoint that ships a vision tower, and the default one does: `mlx-community/Qwen3.6-35B-A3B-4bit` carries `vision_config`, `image_token_id: 248056`, a `preprocessor_config.json`, and 333 of its 2090 tensors under `vision_tower.*` (0.89GB, left unquantized while the language model is 4-bit). `mlx_lm/models/qwen3_5_moe.py`'s `sanitize()` drops those keys at load, so stock mlx-lm hands back a text-only model and the tower sits unused on disk.

Three pieces make it work:

- **The fork seam.** mira-mlx wraps mlx-lm's `BatchGenerator`, and it was specifically the *batched* path that could not express image input: `insert_segments()` took token ids only. The language model never needed changing — `mlx_lm/models/qwen3_5.py` has always accepted `input_embeddings` through `Model.__call__` → `language_model` → the decoder stack. So the fork threads an optional per-sequence embeddings array through `insert_segments()` and `PromptProcessingBatch`, sliced in lockstep with the tokens across chunked prefill, with token-only sequences in a mixed batch embedded from the model's own table (`_embed_tokens`, which walks nested `model`/`language_model`/`transformer` wrappers — Qwen3.6 keeps its table two levels down).
- **`core/inference/qwen3_vl_vision.py`** — mlx-vlm 0.6.8's Qwen3-VL tower, vendored verbatim (MIT, © 2025 Prince Canuma) rather than taking the dependency, which would also pull `opencv-python` and `mlx-audio`. Do not hand-edit it; re-vendor if the tower changes upstream.
- **`core/inference/vision_tower.py`** — resolves the repo id to a snapshot, loads only the `vision_tower.*` weights, preprocesses (a port of transformers' `Qwen2VLImageProcessor._smart_resize`, asserted equal to it in tests), and splices the resulting patch embeddings into the text embeddings at the placeholder positions.

Things that bite, and how they are handled:

- **Image turns skip the prompt cache, deliberately.** `DiskBackedPromptCache.fetch_nearest_cache()` keys on token ids, and an image is N repetitions of one placeholder id. Two different screenshots at the same resolution in one conversation produce a byte-identical prefix, so the cache would happily answer about the wrong image. `_start_job` forces a cold prefill when a turn carries an image.
- **Deepstack is refused, not ignored.** Qwen3-VL can inject intermediate tower features into early language layers, which text-only mlx-lm has no hook for. This checkpoint declares `deepstack_visual_indexes: []`, so nothing is lost; a checkpoint that wants them raises at load rather than answering with part of its visual signal silently missing. A hidden-size mismatch and a missing `vision_config` are refused the same way.
- **Failures are visible in `/v1/stats`, not in a log.** The mira-mlx subprocess writes to `DEVNULL`, so a warning about a tower that failed to load would reach nobody. `GET /v1/stats` carries a `vision` block instead: `{enabled, tower_bytes, images_embedded, image_tokens}`, or `{enabled: false, error: ...}`. The backend keeps serving text either way.
- **Data URLs only.** `_decode_image_part()` refuses remote URLs — fetching one would make the inference server an SSRF vector.

Measured cost on a 32GB Mac: 1.1GB active and 1.3GB peak, of which 0.89GB is the tower. A 640x480 image spends 300 context tokens, 1024x768 spends 768 (`file_handler._IMAGE_MAX_PX` caps the longest edge at 1024).

Worth recording because it blocked the work for six weeks: the 2026-07-18 rejection rested on two premises that were both wrong. It claimed no vision-capable checkpoint existed for this model (the one already running has one) and that the fork patch meant porting vision forward passes and an M-RoPE scheme from scratch (the language side needed no change at all).

## Test patterns

`tests/test_queries.py` — mocks `_call_llm` via `patch.object`:
```python
patch.object(orchestrator, '_call_llm', return_value=iter([chunk]))
# chunk is a MagicMock with .message.content, .message.tool_calls, .done
```
Covers: search trigger, search_done payload, fetch_url dispatch, Gemma4 intermediate tool calls (`accumulated_tool_calls`), RAG threshold bypass, verbose toggle, conversation reset.

`tests/test_cancel.py` — uses FastAPI `TestClient`; mocks `orchestrator.stream_chat` directly.
Covers: cancel endpoint, cancel cleared on new chat, events dropped after cancel, history rollback.
