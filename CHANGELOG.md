# Changelog

## v1.5.0 — August 2026

The release where the engine learned to speculate on its own. v1.4.0 made Mira distributable; this one makes the bundled engine fast without an external app, and builds the eval discipline to prove a change is worth shipping before it ships.

- **Mira's own engine now does speculative decoding — no external app.** mira-mlx can run a Qwen3.5/3.6 checkpoint's own multi-token-prediction (MTP) head as a self-speculator: a tiny extra head drafts a few tokens ahead and the backbone verifies the whole run in one pass, spending compute that decode otherwise leaves idle. On the Qwen3.6-35B-A3B MoE that's roughly 1.2–1.3× faster generation, and it's lossless in practice — it stays correct even with the repetition-penalty runaway guard on. Until now this only worked through the separate oMLX app; the bundled default backend now serves it itself. It's off by default and needs a checkpoint carrying the `model-mtp.safetensors` sidecar (`mira_mlx_mtp_enabled`). The MTP path also ships an adaptive draft-depth controller, an accept-rate readout in `/v1/stats`, a reduced-vocab draft projection (a free ~6% on top), and a sort-threshold tweak in the MoE expert gather (+4.6%). I wrote up the whole thing two ways: a plain-language primer, [Small Mac, fast AI](https://askmira.es/writing/small-mac-fast-ai) (no ML degree required), and the engineering deep-dive, [Chasing 80 tok/s](https://askmira.es/writing/chasing-80-tokens) — a field guide to native MTP on a mixture-of-experts model and an honest ledger of where the speed is still hiding. The code-level notes are in `docs/multi-token-prediction.md`.
- **The context window now sizes itself to real peak memory, not just the KV cache.** On a 32 GB Mac, asking for a 64k window used to over-commit and could OOM mid-run, because the real ceiling is weights plus the growing KV plus the prefill attention transient — and that transient dominates. Sizing now accounts for all of it, so the same aspirational `context_window` request resolves to what actually fits (~39k on this 32 GB Mac, the full amount on a larger one) instead of failing late.
- **A failed model can no longer sit on ~4.5 GB of dead prompt cache.** On Qwen3.6 (and any hybrid model with recurrent-attention layers) most per-turn cache entries can never be reused, yet they filled the cache pool up to its budget and starved the very memory the engine needs to prefill a long prompt. Those entries are now skipped at the source; the reusable ones (the system-prompt openers, ~79% reuse) are untouched, and dense models like Ministral are byte-for-byte unchanged.
- **Switching model actually switches the model on every backend.** The mlx-lm and vllm-mlx paths were ignoring the configured model id in some cases; they now honor it, matching mira-mlx and omlx.
- **A fast evaluation harness for ranking changes.** Scoring a real GPQA run takes 12–14 hours, which is too slow to steer week-to-week tuning. There's now a documented two-instrument contract (`docs/eval-contract.md`): a ~1-minute proxy that *ranks* candidate changes on relative deltas, and the full apparatus that decides what actually ships. This is developer tooling, not a user-facing feature, but it's why the performance claims above come with numbers.

## v1.4.0 — August 2026

- **Mira is now distributable.** A new `mira fetch-model` command downloads the
  default backend's ~19 GB model ahead of time, with progress, so the first
  `mira serve` no longer looks like a hang (idempotent; takes an optional repo id
  to pull a different model). The installer, `mira doctor` and `mira preflight` are
  now built around the mira-mlx default rather than assuming the optional oMLX app,
  and a Homebrew formula ships under `packaging/homebrew/` (tap publish pending).
- **A failed generation can no longer poison your history.** An assistant reply
  that comes back empty, or degenerates into a single repeated character
  (`!!!!…`), is no longer saved and is stripped from loaded history before it
  re-enters the model's context. A blank reply also used to leave a bubble stuck
  in the "typing…" state after you reopened the conversation — that's fixed at the
  source, since the blank turn is never persisted.
- **Truncated replies are now distinguishable after the fact.** Each stored reply
  records the engine's `finish_reason`, so a response cut off by the length cap or
  one that came back empty can be told apart from a clean finish.
- **The "your Mac is low on memory" notification was blaming other apps for
  memory Mira itself was holding.** The advisory splits an eviction into the idle
  treadmill (macOS compressing an idle model, nothing to act on) and a real
  shortage (other apps taking the memory, close one), and only the second is
  supposed to interrupt. But it decided which was which from the OS pressure level
  alone, and compressing Mira's own ~20GB model is itself enough to push pressure
  to warn, so the treadmill kept tripping the "real shortage" branch: seven false
  notifications in one morning while everything other than Mira held under a
  gigabyte. The cause is now attributed per-process — it subtracts Mira's own
  compressed pages from the system compressor and only calls it external when the
  non-Mira remainder is real (`EXTERNAL_COMPRESSED_MIN_BYTES`, 2GB), matching the
  per-process principle the eviction verdict itself already used. The idle
  treadmill stays silent even when it briefly lifts OS pressure.

## v1.3.0 — August 2026

The release that went looking for broken replies and found them. v1.2.0 shipped
nine measured performance wins and zero measured quality signals; this one reads
what Mira actually sent people. Most of what follows was found in real
conversation logs rather than in tests.

- **Replies were being cut off, and a cut-off reply poisoned the rest of the
  conversation.** Two batches of real conversations put 13 of 51 turns into the
  token cap and every one of them arrived broken. Some showed nineteen thousand
  characters of "The user wants me to..." where the answer should have been. One
  conversation got five turns in a row of a single exclamation mark repeated 4,096
  times and never recovered, because a ruined reply goes into the history and
  produces itself again on the next turn. The cap was hardcoded to 4096 in three
  places, which was never enough for a model that thinks: reasoning on ordinary
  questions ran to nineteen thousand characters, so thinking spent the whole
  budget before the answer started. It is a setting now, `max_output_tokens`,
  defaulting to 16384. A model that is done still emits its stop token, so raising
  the ceiling does not make replies longer, it stops them being truncated. A reply
  that is one character repeated is now discarded rather than sent, judged over
  non-drawing characters so ASCII diagrams survive, and a turn genuinely cut at
  the cap says so instead of returning empty.

- **The thinking budget was a number that crossed the whole stack and did
  nothing.** `max_thinking_tokens` travelled as a chat-template kwarg, and
  Qwen3.6's template never mentions it. Jinja discards unknown kwargs silently, so
  the setting reached the model and evaporated at every value anyone set. It is
  now a logits processor that forces `</think>` once reasoning runs past the
  budget, rather than stopping generation: a hard stop at the budget yields a
  reply that is all reasoning and no answer, which is the failure above. Wiring it
  exposed a worse bug underneath. mlx-lm replaces a sequence's logits-processor
  list with `None` whenever any list in the batch is empty, then iterates it, and
  Mira runs two jobs per turn. The moment one carried a processor and the other
  did not, the engine thread died with a `TypeError` after a single token.

- **One bad request could take the engine down, and a dead engine kept answering
  the door.** Admission ran unguarded past the prepare block, so anything from the
  cache lookup to segment insertion killed the engine thread instead of just that
  request. Failures in the generation loop cannot recover in place, since the KV
  state and the MLX stream are of unknown validity afterwards, so the engine now
  fails every in-flight and queued job with the real exception and the HTTP layer
  answers 503 with the cause. Previously it hung every client that asked.

- **Retrying a failed reply no longer saves your question twice.** The client
  drops the failed exchange from its own list and re-sends; the server was never
  told, and saves append. So the conversation kept the question twice with the
  broken answer between them, and every later turn was built on both copies.
  Demonstrated live before fixing: two identical posts on one conversation left
  four rows. `/chat` now takes a `retry` flag that drops the last user message and
  everything saved after it, from the database and from the in-memory history
  both, under the conversation lock and before the turn starts.

- **`fetch_url` was handing the model a page's CSS and calling it the page.**
  Asked to read Apple's notification guidelines, Mira got 969 characters back. Not
  an error, not a warning, just a successful fetch of a `baseUrl` variable and a
  font-family declaration. It could not find what it had been asked about so it
  went searching instead, and nobody watching could tell the page had never
  arrived. markdownify's `strip` argument skips a tag and then walks its children
  anyway, so everything inside `script` and `style` came through as body text.

- **A tool that answers in plain English no longer kills the whole reply.** Three
  turns of a corpus conversation ended with "Internal error, see server logs" and
  the logs held nothing. `list_attachments` returns a string, and the done-label
  lambda called `.get` on it, killing the stream after the client had already been
  told the tool started. The empty log is fixed too: the handler bound the
  exception and then dropped it, so the message telling you to check the logs
  pointed at nothing.

- **Sampling, penalties and the seed are yours to set, and regenerate now returns
  a different answer.** Mira never sent a sampling parameter from anywhere, so the
  engine's own defaults of 0.0 applied to every reply and greedy decoding was the
  product of an absence rather than a decision. `temperature`, `top_p` and `top_k`
  are configurable now, defaulting to exactly the old behaviour. Repetition,
  presence and frequency penalties existed in mlx-lm with no route from
  `mira.yaml`, so the one lever against a degenerate reply was unreachable; they
  all ship off, because penalising repetition also penalises the legitimate kind
  and code and tables are full of it. The seed was process-global and never set,
  which made sampling reproducible by accident of ordering, so regenerating handed
  you back the answer you had just rejected. It is now drawn per request.

- **The token counter finally counts.** The stream carried no usage at all, so
  the context gauge sat at 0 however full the window got. mira-mlx now emits usage
  the way OpenAI does, including `reasoning_tokens` and `cached_tokens`. Counting
  reasoning needed care: Qwen3 templates open `<think>` inside the prompt, so the
  model never emits the marker and every thinking token would have been filed as
  answer text.

- **Starting a new conversation is faster.** Openers were re-prefilling the whole
  system prompt every time. A snapshot taken at the system boundary is shared by
  every conversation on the machine, which took openers from 10.0% to 78.7% prompt
  reuse over ten fresh conversations, 98.4% per hit, leaving 43 tokens to prefill
  instead of 2,599.

- **Memory pressure handling stopped over-correcting, and stopped waking you
  up.** The pressure trim halved the prompt-cache pool and then cleared MLX's
  allocator pool, which is backwards: the allocator pool costs a reallocation to
  rebuild, a prompt-cache entry costs a prefill somebody pays again. Across five
  trims in 70 hours the overshoot was 0.00 to 0.11GB and the response freed 0.53
  to 1.14GB, up to 53 times more than needed. Eviction notifications are off by
  default now: measured over the same window they fired 302 times, one every 14
  minutes, round the clock including 01:30, 03:00 and 06:53. There is nothing to
  do about another process taking memory and it is not something you caused.

- **Concurrent requests no longer crash quantized KV on Qwen3.6 or any other GQA
  model.** At batch 2 or more the attention scores go 5-D while the mask stays
  4-D, and broadcasting aligns from the right, so the mask's batch axis lands on
  the KV head count. At batch 1 the leading dimension broadcasts against anything,
  which is why a single user never saw it and the first concurrent bench did.
  Upstream had fixed this on 2026-07-09 and our pin was 24 commits behind it. A
  pin freezes the bugs along with the API.

- **Quality is measured now, not just speed.** The bench scores answers rather
  than clocks, in two tiers: deterministic checks that can fail a build, and
  judged ones that print with a measured noise floor beside them. The floor is
  real rather than assumed, from three runs of one build. Six new questions cover
  bugs that already shipped, because the deterministic half had scored full marks
  three runs running and stopped telling us anything. Finding all this took fixing
  the harness first: the judge was writing every verdict into the real
  conversation database, scoring correct answers zero for facts it could not
  check, and reporting a clean pass on an injection test whose payload was never
  served.

- **Documentation now matches the code.** All 45 `mira.yaml` settings are
  documented with their defaults in `docs/configuration.md`; seven of them,
  including three that control the shell sandbox and SSRF protection, had never
  appeared in `mira.yaml.example` at all. The bug report template asks for
  something useful instead of your iPhone 6's browser, `CONTRIBUTING.md` carries
  the spec format instead of pointing at a file you cannot see, and roughly 320
  lines describing retired backends came out.

## v1.2.0 — August 2026

- **Multi-turn chat can stop re-prefilling itself, if you turn it on.** Plain
  conversation on Qwen3.6 was re-reading the whole history on every single turn:
  measured 2026-08-08, a 27,614-token second turn reused nothing at all and took
  48.7 seconds. Two things close the normal reuse paths on this model, and
  neither is fixable where it lives. Its hybrid linear-attention cache keeps
  running summaries with no per-token slots, so it cannot be trimmed back at all,
  and the chat template's `<think>` scaffold means one turn's prompt is not a
  prefix of the next. So prefill now splits at the history boundary and the state
  there is captured on the way past, because it cannot be recovered afterwards.
  The same turn with `boundary_snapshot: true` takes 5.0 seconds and reuses
  27,500 of 27,614 tokens, for a 14ms snapshot cost. Agentic tool loops already
  reused well and are unchanged. **Off by default** — it changes the prefill path
  of every request and wants a week of ordinary use across more than one machine
  before that flips.
- **Mira notices when something else takes its memory, and says so.** When
  another app pushes the model out of RAM, macOS compresses it out and the next
  reply costs about 15 to 17 seconds against a warm half-second. That reply is
  itself what fixes it, so the symptom is one unexplained slow answer and then
  normality, which is the least debuggable shape a performance problem can have.
  There is now a macOS notification on the transition into that state (at most
  one every 15 minutes, `memory_advisory_notifications`, on by default), and
  `GET /hardware` carries a `system_memory` block whose `advisory` field the apps
  can show. It never blocks or delays anything: if a memory advisory could stop
  you talking to Mira, another app opening tabs could take Mira offline, which is
  worse than one slow reply.
- **The memory ceiling is derived from the machine, not from its spec sheet.**
  Sizing used to come off `hw.memsize`, which describes a Mac with nothing else
  running on it. It now re-derives from live system state every 30 seconds, so
  the context window and cache budgets reflect what is actually available.
  `proactive_decompress` (off by default) additionally faults the model back in
  on the engine's idle branch rather than leaving the bill for whoever asks next.
  It is roughly memory-neutral because emptying the compressor pays for most of
  the expansion, and it is skipped on battery, at critical pressure and without
  headroom. The catch is that a request arriving mid-reclaim waits behind it, up
  to about 19.5 seconds on a full eviction, which is never worse than doing
  nothing but is not free either.
- **The disk prompt cache is off, and deleting what it left is worth doing.** It
  had been accumulating since July and had served zero reads, not through a bug
  but by construction: a lookup is an exact match on a hash of the whole prompt
  while an entry is keyed on the prompt plus everything generated, so a hit would
  need a byte-identical repeat of an entire conversation. Left alone it reached
  39.75GB and was evicting old entries to make room for new ones that could not
  be read either. Turning it off stops the growth but deletes nothing, so an
  upgraded install keeps whatever it had. Mira now says so once at startup and in
  `mira doctor`, with the size and the exact command; it does not delete
  gigabytes out of your data directory on its own. The setting stays as
  `disk_prompt_cache` for a future prefix-capable version, but enabling this one
  only refills the disk.
- **A reply can no longer come back empty.** The model could call `task_done`
  as the entire content of a turn, having run no tools and written nothing, which
  ended the turn with nothing for you to read. That call is now refused once per
  turn when there is no visible output and no tool has run, and the refusal comes
  back as a tool result telling the model to write the answer itself. Across
  seven agentic bench questions the guard fired zero times while every question
  still exited through `task_done` normally, so it is inert on legitimate turns.
- **The engine keeps its logs.** mira-mlx's stdout went to `/dev/null`, so its
  prompt-cache decisions, disk-cache activity and decompression timings were
  unobservable — which is why "the prompt cache reports no hits" had no evidence
  attached for weeks. It now writes to `~/.local/share/mira/mira-mlx.log`, capped
  at 32MB, and every cache miss also logs why it missed.

## v1.1.0 — August 2026

- **Vision stopped being expensive.** The Qwen3.6 checkpoint ships a
  `preprocessor_config.json` asking for a 16,777,216-pixel ceiling, which caps
  nothing on real photographs. A 5712x4284 image off a phone survived at 16,170
  image tokens — twelve percent of a 128k context window for one picture — and
  cost 243 seconds in the vision tower and 126MB of embeddings. There is now a
  `mira_mlx_vision_max_pixels` ceiling, defaulting to 1 MP, which holds any image
  to roughly 1,000 tokens and about 1.6 seconds whatever came off the camera.
  End to end that turned a photo that took over four minutes into one that takes
  about eight seconds. Context cost per image no longer depends on the source
  resolution at all, which makes budgeting a conversation with pictures in it
  actually possible.
- **1 MP was chosen by measurement, not by taste.** The obvious fear is losing
  small text in screenshots, so four real images from 2.4MP to 24MP were run at
  both 1 MP and 2 MP. At 1 MP a game screenshot still named the game and still
  read every UI label and all five skill names, and OCR remains the better path
  for genuinely dense text. What does soften is fine visual detail: the same
  screenshot gave "glowing blue and silver armor" at 2 MP and "blue skin, dark
  armor" at 1 MP. If that matters more than four seconds a turn, set
  `mira_mlx_vision_max_pixels: 2097152`. The setting only ever lowers the
  checkpoint's own ceiling, never raises it.
- **The vision tower is no longer resident.** It used to load at startup
  whenever vision was on, so a session that never sent an image still paid about
  0.89GB for the privilege. It now loads on the first image turn and is released
  again after `mira_mlx_vision_tower_idle_timeout` seconds of no images (default
  300, set 0 to keep it). The reload costs well under two seconds because Metal
  kernels survive the round trip, and MLX materialises the weights lazily anyway,
  so even a loaded tower costs nothing until an image is actually processed. On a
  32GB machine this is the difference between vision being a standing tax and a
  per-use one. A tower that fails to load now turns vision off for the process
  instead of retrying on every image, so a checkpoint without one costs a single
  attempt.
- **`GET /v1/stats` says more about vision.** New `tower_resident`,
  `tower_loads`, `tower_unloads`, `tower_last_reclaimed_bytes`, `max_pixels` and
  `idle_timeout_s`. The reclaimed figure is measured at release rather than
  assumed from the tower's own weight count, because anything still holding a
  reference would otherwise free nothing quietly. `vision.enabled` now follows
  your configuration rather than whether the tower happens to be in memory, so an
  idle release does not read as a failure.

## v1.0.0 — August 2026

- **Mira can see.** Set `mira_mlx_vision: true` in `mira.yaml` and image
  attachments are read by the model's own vision tower instead of being run
  through OCR — screenshots, charts, diagrams, photos, things with no text in
  them at all. It works on the default checkpoint: `Qwen3.6-35B-A3B-4bit` ships a
  vision tower that stock `mlx-lm` throws away at load time, so nothing new needs
  downloading. Off by default, because it costs about 1.1 GB of memory and OCR is
  genuinely better for text-heavy screenshots. A 640x480 image spends 300 context
  tokens, 1024x768 spends 768. Two things to know if you turn it on: image turns
  skip the prompt cache on purpose (an image is N copies of one placeholder token,
  so two same-sized screenshots would otherwise collide into a false cache hit),
  and if the tower fails to load the backend keeps serving text and tells you why
  under `vision.error` in `GET /v1/stats`.
- **Retired the dflash and Ollama backends.** Mira's backend is mira-mlx; omlx is
  the backup, and mlx-lm and vllm-mlx stay because they are cheap to keep and
  useful for comparison. Both retired backends are gone from the code rather than
  hidden from the picker, and their Python dependencies went with them. No model
  coverage was lost: Ollama only ever served `ministral-3:14b`, which runs on three
  of the remaining backends, and Gemma 4 is still reachable through omlx. The
  `ollama` key stays in `GET /models`, always empty, so an older app build that
  still decodes that field does not fail on a missing key. `OLLAMA_HOST` in
  `config.py` is now `BACKEND_HOST` — it had stopped meaning Ollama long ago and
  made a retired backend look load-bearing. The Ollama-native web search went too;
  it had been dead code behind a flag that was never switched on, so Brave when
  keyed and DuckDuckGo otherwise is now all the module claims to do.
- **Fixed reasoning being served as the answer on thinking turns.** Qwen3's chat
  template puts the opening `<think>` in the prompt, so the model's output starts
  inside the block and only ever emits the closing tag; the streaming stripper was
  waiting for an opening tag that never came and passed the whole chain of thought
  through as answer text, stray `</think>` included. Thinking token counts were
  also being undercounted, and the polluted text was persisted to conversation
  history. Turning thinking off was never affected.
- Moved to `mlx` 0.32.0 and `mlx-metal` 0.32.0. No behaviour change on the
  paths mira actually uses: the four bugs 0.32 fixes were all reproduced on this
  hardware but none of them reach mira's shapes. Batched decode gets about 24%
  faster at eight concurrent sequences, which is above normal single-user load.
  Done ahead of the vision work, which needs `mlx>=0.32.0` for `mlx-vlm`.

## v0.9.5 — July 2026

- Retrieved content is now handled as data, not instructions. Tool output —
  file and GitHub reads, fetched pages, search results, RAG chunks, attachments,
  and OCR text — is wrapped in a per-session trust boundary, and a new
  system-prompt rule tells the model to report, never obey, any instructions
  embedded in that content. The out-of-band approval gate remains the
  load-bearing control for destructive actions.
- `run_shell` now runs inside an OS sandbox that confines its file writes to the
  active workspace.
- Inference backends verify a listener's model identity before adopting it, so a
  mismatched or unexpected backend process is not silently trusted.
- Corrected the destructive-action confirmation wording (approval is out of band,
  not a flag the model sets) and normalised search-result titles and URLs so a
  crafted result cannot forge additional result blocks.

## v0.9.4 — July 2026

- **⚠️ BREAKING CHANGE — destructive tool calls now require out-of-band approval.**
  The model can no longer approve its own destructive actions (`rm -rf`,
  `git reset --hard`, `sudo`, file/branch deletion, PR merge). The server refuses
  them and emits an `approval_required` event; the client must show it to the user
  and echo back a content-derived approval token before the command runs. **Clients
  older than the app's build 38 / v0.2.1 do not understand this handshake**, so on
  those clients destructive commands are refused with no way to approve them.
  Everyday non-destructive commands are unaffected. Update the app before relying on
  destructive tools. Wire format: `approval_required` SSE event carrying
  `{tool, action, approval_token, target, matched, message}`.
- Hardened request handling: `Host`/`Origin` are validated ahead of the auth token,
  request models are bounded, project paths are confined, and tailnet interface
  discovery is narrowed to interfaces that can carry it.
- Upload filenames are normalised to a bare name before joining the workspace root;
  `url_fetcher` declines private and loopback targets unless explicitly allowed.
- Remote code execution (`trust_remote_code`) is now opt-in via config rather than
  on by default.
- Inference: shard names from a model index are constrained to bare filenames and
  safetensors header reads are capped; fixes a HuggingFace-cache symlink case that
  broke expert offload for cached models.
- `mlx-lm` is pinned to an explicit commit instead of a branch, so the installed
  tree is reproducible and cannot change under a force-push.

## v0.9.3 — July 2026

- **`context_window` in `mira.yaml` now actually reaches mira-mlx** — the top-level
  `context_window:` key was only used for orchestrator bookkeeping; the mira-mlx subprocess's
  `--max-kv-size` was silently pinned to a separate hardcoded constant. Bumped and stability-
  tested up to 128K tokens on Ministral 3 14B (single-shot prompts through ~29.5K tokens and a
  realistic two-turn ~42K-char injected-file scenario all completed cleanly).
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

- **Inference tuning results documented** — `docs/bench-archive/inference-tuning-2026-06-27.md` records the
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
