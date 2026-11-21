# Engine lifecycle: idle unload and wake on request

By default the mira-mlx engine starts with the server and stays loaded (~20GB wired for
Qwen3.6-35B-A3B) until the server exits. Setting `engine_idle_unload_minutes` above 0 makes
Mira stop the engine after that many idle minutes and start it again on the next request.
It is off by default and applies to the mira-mlx backend only.

```yaml
engine_idle_unload_minutes: 15
```

## What it costs

Measured 2026-11-21 with the production config, from process spawn to the first streamed
token: about 9 s with a warm file cache and 14 s on a first run (kernel warm-up included).
The first turn after a wake also re-prefills its whole context, because no prompt cache
survives the process. Embeddings, the reranker and RAG live in the server process and keep
working while the engine sleeps.

## States

| `engine_state` | Meaning |
|---|---|
| `running` | Engine loaded and serving. |
| `unloading` | Being stopped (idle window passed, or `POST /engine/unload`). |
| `sleeping` | Stopped on purpose. The next request wakes it. |
| `waking` | Starting. Requests wait for it, up to ~260 s, then fail. |

A request never races an unload: one that arrives mid-stop waits for the stop and then
wakes the engine. Concurrent requests share a single start. Nothing unloads while a request
or a stream is in flight, and a backend or model switch owns the engine until it finishes.

## API for clients

`GET /health` (unauthenticated) gains `engine_state` when the lifecycle is on:

```json
{"status": "ok", "backend_ready": true, "engine_state": "sleeping"}
```

- `sleeping` reports `backend_ready: true`. A sleeping engine is deliberate and wakes on the
  next request, so apps that predate `engine_state` keep working unchanged: they send a
  message and the server wakes the engine. They never show an "offline" banner for it.
- `waking` and `unloading` report `backend_ready: false`.
- `running` keeps the real probe, so a crashed engine is still `backend_ready: false`.
- When the lifecycle is off, `engine_state` is absent and nothing changes.

`GET /info` also carries `engine_state` when the lifecycle is on.

Authenticated routes:

| Route | Effect |
|---|---|
| `GET /engine` | `{enabled, state, idle_unload_minutes, idle_seconds, inflight}` |
| `POST /engine/wake` | Wake now and return when it is up, so the first message does not pay the load. 503 if the wake failed. |
| `POST /engine/unload` | Unload now to free RAM (for example before opening a heavy app). 409 if the lifecycle is off or a request is in flight. |

### What an app should do (optional polish, nothing breaks without it)

1. Poll `/health` as today. When `engine_state == "sleeping"`, show a quiet "Mira is asleep,
   it wakes when you send a message" state instead of nothing.
2. Call `POST /engine/wake` when the user focuses the composer or opens a conversation, so
   the load overlaps with typing. Show "waking up" while `engine_state == "waking"`.
3. Expect the first reply after a sleep to take 10 to 20 s longer. Chat timeouts must allow
   for it (the server holds the request while it wakes).
4. Offer an "Unload now" action calling `POST /engine/unload`.

## Notes

- Hooked at the OpenAI-client layer in the orchestrator, so title, summary, forced-JSON and
  streamed turns all count as activity. `/v1/stats` and `/health` polling never do.
- Unloading also stops an engine Mira adopted rather than spawned (found by port and checked
  to be the mira-mlx server before it is signalled).
- Design and review notes: `specs/engine-on-demand-lifecycle.md` (local).
