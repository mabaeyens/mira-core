# Architecture Reference

Detailed reference for subsystems. Read this file when working on events, RAG, cancel, or search config.

## Event protocol

`ChatOrchestrator.stream_chat()` yields typed dicts consumed by both CLI (`main.py`) and web server (`server.py`):

| Event | Payload | Meaning |
|-------|---------|---------|
| `thinking` | — | Model is processing |
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

**Mockable boundary:** `_call_ollama()` is the single point tests mock — returns an iterable of stream chunks with `.message.content`, `.message.tool_calls`, `.done`.

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

## Why DDGS over Ollama native search

Ollama offers a free-tier web search API, but signup requires a phone number and Ollama's docs contain no privacy disclosures — queries are tied to an authenticated account and can be logged. DuckDuckGo's core value proposition is no-tracking search. This project is local-first and privacy-conscious, so DDGS is the deliberate choice. Additionally, `gemma4:26b` is not listed as a supported model for Ollama native search.

`USE_NATIVE_SEARCH` is `False` in `config.py` — do not change it.

## Test patterns

`tests/test_queries.py` — mocks `_call_ollama` via `patch.object`:
```python
patch.object(orchestrator, '_call_ollama', return_value=iter([chunk]))
# chunk is a MagicMock with .message.content, .message.tool_calls, .done
```
Covers: search trigger, search_done payload, fetch_url dispatch, Gemma4 intermediate tool calls (`accumulated_tool_calls`), RAG threshold bypass, verbose toggle, conversation reset.

`tests/test_cancel.py` — uses FastAPI `TestClient`; mocks `orchestrator.stream_chat` directly.
Covers: cancel endpoint, cancel cleared on new chat, events dropped after cancel, history rollback.
