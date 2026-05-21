# Specs: Mira Streaming Stability

Four independent specs derived from the May 2026 stability analysis. Each targets a distinct failure mode that causes the "shoot and guess" UX on the iOS app. All are additive — implement in any order.

---

## Spec 1 — Heartbeat handling: keep the connection alive and show progress

```
1. Problem: server sends {"type":"heartbeat"} every 5s during long inference but iOS
   URLSession may treat it as inactivity and kill the SSE connection, causing silent timeouts.
   Even when the connection stays alive, the app shows nothing during the wait.

2. Files: mira-apps — wherever SSE events are parsed (likely ChatViewModel or APIClient.swift).
   server.py is already correct (heartbeat at line 355); no server changes needed.

3. Constraint: heartbeat must reset the app's internal "stale connection" timer without
   triggering any visible UI change beyond a subtle activity indicator already on screen.
   Do not show a new spinner or message per heartbeat — just keep the existing one alive.

4. Edge cases:
   (a) heartbeat arrives before any token — indicator should already be showing (see Spec 3)
   (b) heartbeat arrives after thinking phase starts — keep spinner, don't restart it
   (c) connection genuinely drops mid-stream — distinguish timeout from heartbeat gap > 15s

5. Done: sending a prompt that takes >10s (complex question, gemma4:26b) shows a continuous
   activity indicator with no visible interruption; app does not time out or show an error
   on responses that complete within 120s.
```

---

## Spec 2 — URLSession timeout: allow long inference to complete

```
1. Problem: iOS URLSession resource timeout defaults to 60s. Complex prompts on gemma4:26b
   (~38 t/s) can produce 1000+ tokens in 30–60s; with any added latency (RAG, tool call,
   model swap) the connection drops before the response arrives.

2. Files: mira-apps — APIClient.swift (URLSession configuration, or wherever the SSE
   session is initialised). Check if URLSessionConfiguration.timeoutIntervalForResource
   or timeoutIntervalForRequest is set; if not, the OS default (60s) applies.

3. Constraint: set timeoutIntervalForResource ≥ 300s (5 min) for the SSE chat request
   only — not for health probes or other short requests, which should keep a short timeout
   (5–10s) so failures are detected quickly.

4. Edge cases:
   (a) user navigates away mid-generation — cancel the URLSession task, don't wait for timeout
   (b) server sends heartbeats (Spec 1) — these count as activity for URLSession, resetting
       timeoutIntervalForRequest automatically; only timeoutIntervalForResource needs raising
   (c) VPN / cellular handoff mid-stream — handle stream error and surface a "connection lost,
       tap to retry" message rather than silently failing

5. Done: a prompt that generates 1500 tokens (~40s at 38 t/s) completes successfully on
   WiFi and LTE without a timeout error. A cancelled request (user backs out) stops within 2s.
```

---

## Spec 3 — Loading state from request send, not first token

```
1. Problem: the app shows no visual feedback between "send button tapped" and "first SSE event
   arrives" (which includes: network round-trip + thinking phase + prompt eval). On a warm model
   this gap is 0.5–1s; on a cold start or after a model swap it can be 10–20s. The user sees
   nothing and assumes the app is frozen.

2. Files: mira-apps — the chat send action (button handler or ChatViewModel.sendMessage).
   The fix is UI-only: set a "pending" or "generating" state on the view model the moment
   the request is dispatched, before any SSE event arrives.

3. Constraint: the loading indicator must clear on error as well as on success. If the SSE
   connection drops or returns an error event, the indicator must not stay on screen.
   Do not wait for a "thinking" or "token" SSE event to start showing the indicator.

4. Edge cases:
   (a) request fails immediately (503 backend not ready) — show error in <1s, clear indicator
   (b) server sends "thinking" event — transition from generic "sending…" to "thinking…" label
   (c) user hits cancel during the pending phase — cancel the request, clear indicator
   (d) double-tap send — guard against duplicate in-flight requests at the view model level

5. Done: tapping send shows an activity indicator within 100ms on the user's message bubble
   (or in the input bar). Indicator transitions to "thinking…" on a thinking SSE event, then
   to streaming tokens. Indicator always clears when response is done or on error.
```

---

## Spec 4 — Surface backend readiness before the first message

```
1. Problem: on app launch (or reconnect after sleep), the server may still be loading the
   model. The first message sent during this window gets a 503 or a long TTFT with no
   explanation. The user thinks the app is broken.

2. Files: mira-apps — APIClient.swift (already has probe/health logic per collaboration-notes.md).
   server.py /health already returns {status: "starting", backend_ready: false} at line 157.
   The fix is surfacing this state in the UI before the user types.

3. Constraint: do not block the UI — the user should be able to type while the model loads.
   Show a non-blocking banner or subtitle ("Model loading…") that disappears automatically
   when /health returns {backend_ready: true}. Poll /health every 3s during startup only;
   stop polling once ready.

4. Edge cases:
   (a) model loads faster than the poll interval — banner may never appear; that's fine
   (b) model never becomes ready (crash, OOM) — after 120s of backend_ready: false, show
       a persistent "Backend unavailable" state with a manual retry button
   (c) user sends a message while banner is showing — allow it; queue or send immediately
       and let the server's _ready() dependency return the 503, which surfaces as an error

5. Done: on a cold-start Mac (model not yet loaded), the app shows "Model loading…" within
   5s of launch and the banner clears within 3s of the model becoming ready. Existing
   connection-failure banner (already implemented) is not regressed.
```
