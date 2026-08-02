# Changelog

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
