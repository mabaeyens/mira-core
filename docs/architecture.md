# Architecture Reference

Detailed reference for subsystems. Read this file when working on events, RAG, cancel, config, or model quirks.

## Module map

```
main.py (CLI)     server.py (FastAPI + SSE)
        └─────────────────┘
                 │
      core/orchestrator.py → ChatOrchestrator
            │  stream_chat(user_message, attachments=None, thinking_enabled=True) → yields events
            ├── _call_llm() → openai.chat.completions.create() (mira-mlx/omlx/dflash/mlx-lm/vllm-mlx, OpenAI-compatible; ollama via ollama client)
            ├── core/search_engine.py → SearchEngine → Brave API (primary, if keyed) / ddgs.text() (fallback)
            ├── core/url_fetcher.py → fetch_url() → BeautifulSoup, Jina Reader fallback
            └── core/rag_engine.py → RagEngine
                      ├── SentenceTransformer (nomic-ai/nomic-embed-text-v1.5, local, 768 dims)
                      ├── chromadb.EphemeralClient (in-memory)
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
| `thinking` | `content` (optional) | Model is processing; thinking text arrives in `content` (Ollama: `chunk.message.thinking`; mlx-lm/Qwen3: `delta.reasoning` field) |
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

**Thinking toggle:** `stream_chat()` accepts `thinking_enabled: bool = True`. The orchestrator decides whether to think via `_should_think(message, has_attachments)` — a scoring heuristic that also checks `_NEVER_THINK` (trivial commands: time/date queries, "fix file.ext" patterns) and `_REASONING_INTENT` (strong signals that always trigger thinking). On **Ollama**, passes `think=thinking_enabled`; Gemma4 yields thinking text in `chunk.message.thinking`. On **mlx-lm with Qwen3**, thinking is controlled per-request via `extra_body={"chat_template_kwargs": {"enable_thinking": True/False}}`; the `reasoning` field in SSE delta chunks carries thinking tokens and is emitted as `thinking` events. Warm thinking overhead on mlx-lm is ≤14 ms (noise). The server form field `thinking_enabled` (default `true`) flows from the app's thinking toggle.

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
- **Thinking toggle:** brain icon in `InputBar` toggles `thinkingEnabled`; passed as `thinking_enabled` form field to `POST /chat`. Works on both backends.

See `mira-apps/OllamaSearch/Shared/Networking/` for client implementation.

## Search providers (Brave primary, DDGS fallback)

`SearchEngine.search()` tries providers in order:

1. **Ollama native** — only if `USE_NATIVE_SEARCH` is `True` (it is `False`). Disabled deliberately: signup requires a phone number, Ollama's docs carry no privacy disclosures (queries tie to an authenticated account and can be logged), and `gemma4:26b` isn't a supported model for it. Leave `USE_NATIVE_SEARCH` `False`.
2. **Brave Search** — used whenever `brave_api_key` (in `mira.yaml`) or the `BRAVE_API_KEY` env var is set. This is the primary provider in normal operation. The free tier is **2,000 queries/month at ~1 query/second**; the key never gets committed (`mira.yaml` is git-ignored — only `mira.yaml.example` ships, with a placeholder).
3. **DuckDuckGo (ddgs)** — the no-key, no-tracking fallback. Serves requests whenever native is off and Brave is unkeyed or errors (including rate-limit/429 responses, which currently fall through silently rather than retrying).

## Backend startup

On Mira startup, `backend_manager.ensure_backend_running()` launches whichever backend `mira.yaml` configures (default: **mira-mlx**, port 8080 — `core/inference/mira_mlx_server.py`, spawned as `python -m core.inference.mira_mlx_server`; other options are omlx, dflash, mlx-lm, vllm-mlx, or ollama). The app's `/health` endpoint returns `backend_ready: false` until the inference server is reachable with the configured model, and the iOS/macOS chat view shows a banner with a "Start" button during that time. Ollama (port 11434) is optional — no longer required for RAG embeddings (sentence-transformers runs locally); start manually only when using Ollama as an inference fallback.

After the server is confirmed reachable, `_warmup_model(...)` sends a 1-token completion request (non-streaming) to force the model into GPU memory. This eliminates the first-request penalty that occurs when a backend must load a new model into VRAM (mira-mlx: RAM-aware sizing already derives context/cache budgets at spawn time via `core/hardware.py`, so this is mainly relevant for backends that lazy-load). The warmup runs in the background startup thread — both on fresh start and when the server was already running.

### mira-mlx specifics

`core/inference/mira_mlx_server.py` is Mira's own MLX inference server (built on `mlx-lm`'s `BatchGenerator` continuous batching), promoted to the default backend on 2026-07-09. It runs in a single dedicated "engine thread" per process — MLX streams are thread-local, so HTTP handlers only tokenize and hand work off through a `queue.Queue`; they never touch MLX directly. Notable pieces:

- `core/hardware.py` — derives per-machine RAM-aware budgets (prompt-cache pool size, context window ceiling, disk-cache size) so the same code behaves sensibly from an 8 GB Mac to a 128 GB Mac Studio; also computes a proactive Metal cache limit and confirms M-series GPU acceleration (Metal-4/NAX) is active at startup.
- `core/inference/disk_prompt_cache.py` — evicted prompt-cache entries overflow to disk (content-addressed, safetensors) instead of being discarded, surviving both memory-pressure trims and process restarts.
- `GET /v1/stats` on the mira-mlx port — cache hit/miss rate, disk-cache hits, memory-pressure trim events, latency percentiles, and live MLX memory (active/cache/peak/wired-limit bytes).
- Oversized single prompts (≥ the derived context ceiling) are rejected with a clear `ValueError` rather than left to `RotatingKVCache`'s undefined behavior.
- mlx-lm itself is pinned to a mira-owned fork (`github.com/mabaeyens/mlx-lm`, branch `mira-mistral-tool-call-fix`) carrying a Mistral tool-call-flush fix (tracks upstream `ml-explore/mlx-lm#1373`).

## Configuration reference

**External config (preferred):** copy `mira.yaml.example` → `mira.yaml` (git-ignored) and edit:

| Field | Default | Notes |
|-------|---------|-------|
| `backend` | `mira-mlx` | `mira-mlx` (default), `omlx`, `dflash`, `mlx-lm`, `vllm-mlx`, or `ollama` |
| `model` | `mlx-community/Qwen3.6-35B-A3B-4bit` | Model identifier — mlx-community repo id for mira-mlx/dflash/mlx-lm/vllm-mlx (e.g. `mlx-community/Ministral-3-14B-Instruct-2512-4bit` for the Mistral family), omlx's own model name for `backend: omlx` |
| `host` | `http://localhost:8080` | LLM server URL |
| `embed_model` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace embedding model for RAG (sentence-transformers) |
| `context_window` | `65536` | Token context window |

**RAG / search knobs in `core/config.py`** (no `mira.yaml` equivalent — edit directly):
`USE_NATIVE_SEARCH`, `MAX_SEARCH_RESULTS`, `SEARCH_TIMEOUT`, `MAX_RETRIES`, `MAX_TOOL_STEPS`, `VERBOSE_DEFAULT`, `RERANK_MODEL`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_RETRIEVE_K`, `RAG_RERANK_TOP_K`, `RAG_SCORE_THRESHOLD`, `RAG_MAX_CHUNKS`

## Model quirks

Behaviours that are intentional and must not be removed:

- **Gemma4 (mlx-lm):** emits `tool_calls` in an intermediate chunk (`done=False`) — handled by `accumulated_tool_calls` in `orchestrator.py`.
- **Gemma4:** occasionally emits LaTeX (e.g. `$\rightarrow$`) — `preprocessLatex()` in `index.html` converts to Unicode.
- **Gemma4 (Ollama) thinking:** when `think=True`, Ollama yields thinking text in `chunk.message.thinking` (separate from `chunk.message.content`). The streaming loop checks this field and emits `thinking` events.
- **Qwen3.6 (mlx-lm) thinking:** controlled per-request via `extra_body={"chat_template_kwargs": {"enable_thinking": True/False}}`; thinking tokens stream in the `reasoning` field of SSE delta chunks; the orchestrator emits them as `thinking` events and strips thinking markers before emitting the final content. Overhead on mlx-lm is ≤14 ms warm — negligible.
- **`_NEVER_THINK` / `_ANALYTICAL_WITH_ATTACHMENT`:** `_should_think()` uses a scoring heuristic (threshold 4). `_NEVER_THINK` short-circuits to False for time/date queries and "fix file.ext" patterns. Bare attachments score +1; analytical verbs (explain, analyze, review, debug, refactor…) combined with an attachment add a further +2.
- **Qwen3.6 thinking toggle must be threaded through per-backend:** `_call_llm()` (`core/orchestrator.py`) gates the `enable_thinking`/`chat_template_kwargs` override behind an explicit backend allow-list — every OpenAI-compatible backend that honors it (`mlx-lm`, `dflash`, `omlx`, `mira-mlx`) must be listed there, or that backend silently falls back to the model's chat-template default (thinking always on) regardless of the per-turn toggle. Found and fixed 2026-07-10 when `mira-mlx` was missing from the tuple.
- **Mistral/Ministral tool calls (mira-mlx):** Mistral's chat template uses a one-sided `[TOOL_CALLS]` marker (no closing token — the model relies on EOS to end the call), while Qwen's uses a two-sided `<tool_call>...</tool_call>` marker. `mira_mlx_server.py`'s tool-text buffering has a `prev_state == "tool"` fallback to still capture the last chunk before EOS for Mistral's one-sided case — but for two-sided markers this same fallback also captures the closing marker token itself, corrupting the parser's input. The closing marker is stripped before parsing (no-op for Mistral, which has none).
- **Mistral agent-loop role alternation (vllm-mlx):** Mistral-family chat templates require strict user/assistant role alternation; the agent tool-call loop's message construction was fixed to satisfy this when Mistral is the active model.
- **No vision support on mira-mlx (OCR fallback instead):** mira-mlx wraps mlx-lm's `BatchGenerator` continuous-batching engine, and it is specifically the *batched* path that cannot express image input: `insert_segments()` takes `segments: List[List[List[int]]]` (token ids only) and `PromptProcessingBatch.process()` prefills with `self.model(tokens[:, :n_to_process], cache=...)`. The non-batched `generate`/`stream_generate` path does accept an `input_embeddings` argument, and so does the text model itself (`mlx_lm/models/qwen3_5.py`, which `qwen3_5_moe` subclasses). What mlx-lm lacks is vision *towers*: every "VLM" model file (`qwen2_vl.py`, `pixtral.py`, `qwen3_5_moe.py`) is a text-only stub whose `sanitize()` discards the `vision_tower.*` weights at load time. `_prepare_messages()` in `mira_mlx_server.py` raises a `ValueError` (→ 400) as a backstop if a message's `content` contains an `image_url`/`image` part, but `orchestrator.py` normally never lets it get that far: when `backend_manager.PRESETS[backend]["vision"]` is `False`, it runs each attached image through `file_handler.ocr_image_from_base64()` (system `tesseract` binary, same optional dependency used for scanned-PDF OCR) and folds any recovered text into the prompt as a regular text block instead of sending the raw image. This handles the common troubleshooting case — screenshots of error dialogs, menus, terminal output — without real vision. If OCR is unavailable or finds no text (photos, diagrams, pure UI layout questions), the user gets a clear inline error pointing at omlx or suggesting `brew install tesseract`. `omlx` remains the only backend with real vision (`PRESETS["omlx"]["vision"] = True`).

  **Re-investigated 2026-08-01, and the 2026-07-18 rejection rested on two false premises.** Both are recorded here because they blocked the work for the wrong reason, not because the work has been done (it has not; the OCR fallback above is still what ships).

  The first premise was that no vision-capable checkpoint exists on `mlx-community` for Qwen3.6-35B-A3B. It does, and it is the one already running: `mlx-community/Qwen3.6-35B-A3B-4bit` carries `vision_config`, `image_token_id: 248056`, `vision_start/end_token_id`, a `preprocessor_config.json`, and a real vision tower in the weights (333 of 2090 tensors under `vision_tower.*`, 0.89GB, left unquantized while the language model is 4-bit). `mlx_lm/models/qwen3_5_moe.py`'s `sanitize()` skips those keys at load, so `mlx_lm.utils.load()` hands back a text-only model and the tower sits unused on disk. The `Qwen3-VL`-is-a-different-family reasoning was answering a question that did not need to be asked.

  The second premise was that the fork patch means porting vision-tower forward passes and an M-RoPE-style position scheme from scratch. It does not. `mlx_lm/models/qwen3_5.py` already accepts and honors `input_embeddings` through `Model.__call__` → `language_model` → the decoder stack, so the language side needs no change at all; only `BatchGenerator.insert_segments()` and `PromptProcessingBatch.process()` need an optional per-sequence embeddings array threaded through the existing right-padding path. Separately, `mlx-vlm` 0.6.8 (2026-07-27) ships `mlx_vlm/models/qwen3_5_moe/`, matching the checkpoint's `model_type` exactly, and its `vision.py` is a 5-line subclass of `qwen3_vl`'s 445-line `VisionModel`, which is small enough to vendor rather than take the full dependency (`mlx-vlm` also pulls `opencv-python` and `mlx-audio`; its `mlx>=0.32.0` floor is no longer a blocker since mira-core moved to mlx 0.32.0 on 2026-08-01).

  What genuinely does constrain the work, and did not appear in the original write-up: `DiskBackedPromptCache.fetch_nearest_cache()` keys on token ids alone, and an image is represented in the token stream as N repetitions of `image_token_id`. Two different screenshots at the same resolution in one conversation therefore produce a byte-identical prefix and the cache would answer about the wrong image. Any vision path has to skip or salt the prompt cache on image turns. Sizing on a 32GB Mac: +0.89GB resident for the tower, and 768 context tokens for a 1024x768 screenshot (`file_handler._IMAGE_MAX_PX` caps the longest edge at 1024). `omlx` remains the only vision-capable backend today.

## Test patterns

`tests/test_queries.py` — mocks `_call_llm` via `patch.object`:
```python
patch.object(orchestrator, '_call_llm', return_value=iter([chunk]))
# chunk is a MagicMock with .message.content, .message.tool_calls, .done
```
Covers: search trigger, search_done payload, fetch_url dispatch, Gemma4 intermediate tool calls (`accumulated_tool_calls`), RAG threshold bypass, verbose toggle, conversation reset.

`tests/test_cancel.py` — uses FastAPI `TestClient`; mocks `orchestrator.stream_chat` directly.
Covers: cancel endpoint, cancel cleared on new chat, events dropped after cancel, history rollback.
