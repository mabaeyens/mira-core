# Configuration reference

Every setting Mira reads from `mira.yaml`. Copy `mira.yaml.example` to `mira.yaml` (git-ignored)
and edit; all fields are optional, and omitting one keeps the built-in default.

`mira.yaml.example` is the annotated version — it carries the reasoning behind the awkward
defaults, which is worth reading before changing sampling, penalties or the opt-in performance
flags. This file is the complete list, so you can tell at a glance whether a knob exists.

Defaults below are the ones in `core/config.py`. Where a default here and a comment in
`mira.yaml.example` ever disagree, `core/config.py` is what actually runs.

**A note on scope.** Settings prefixed `mira_mlx_` apply to the mira-mlx backend only and are
ignored by omlx, mlx-lm and vllm-mlx. `prefill_step_size` applies to mira-mlx and mlx-lm.

---

## Access control

Read the README's *Access control* section and [remote-access.md](remote-access.md) before
changing any of these — the server can run shell commands, so an open bind is a remote code
execution hole.

| Setting | Default | What it does |
|---|---|---|
| `auth_token` | `""` (or `$MIRA_TOKEN`) | Shared bearer token required by every route except `/health` and the static UI. Unset means the server binds `127.0.0.1` only and refuses a non-loopback interface. `chmod 600 mira.yaml` once you set it. |
| `allowed_source_cidrs` | `["127.0.0.0/8", "::1/128", "100.64.0.0/10"]` | Source-IP allowlist: loopback plus the Tailscale range. Defense in depth, not the primary gate. |
| `allowed_hosts` | `[]` | Host-header allowlist (anti-DNS-rebinding), checked *before* auth. IPs in `allowed_source_cidrs` pass automatically; a **name** does not, so remote clients connecting by MagicDNS name need it listed or every request 403s. |
| `url_fetch_allow_private` | `false` | Whether `fetch_url` may reach loopback, private or link-local addresses. Off by default because the model picks the URL after reading attacker-influenceable text — a crafted page should not be able to steer it at a LAN device or a metadata endpoint. |
| `shell_sandbox` | `true` | Wrap every `run_shell` command in a macOS `sandbox-exec` profile confining writes to the workspace and temp dirs. Fails closed if `sandbox-exec` is unavailable. Regex prefiltering cannot contain a shell; this can. |
| `shell_sandbox_allow_network` | `true` | Whether sandboxed commands may open outbound connections. On by default because `git pull` and `npm install` are legitimate here; set `false` for untrusted sessions. |

## Backend and model

| Setting | Default | What it does |
|---|---|---|
| `backend` | `mira-mlx` | One of `mira-mlx`, `omlx`, `mlx-lm`, `vllm-mlx`. An unknown name raises rather than silently starting something else. |
| `model` | `mlx-community/Qwen3.6-35B-A3B-4bit` | An mlx-community repo id for mira-mlx/mlx-lm/vllm-mlx; omlx wants its own model name instead (`Qwen3.6-35B-A3B`). |
| `host` | `http://localhost:8080` | Where the backend listens. Only one backend runs at a time. |
| `backends` | built-in preset list | Named backend+model combinations served to the iOS/macOS model picker via `GET /backends`. Adding a combination needs only a config edit, no app update. |
| `context_window` | `65536` | Token context window. mira-mlx lowers this on its own if the machine's RAM cannot hold it. |
| `max_output_tokens` | `16384` | Ceiling on a single reply. Raising it does not make replies longer — a finished model still emits its stop token — it stops them being cut off. |
| `paths` | built-in defaults | Absolute paths to backend binaries: `mlx_lm_cli` and `omlx_cli`. Omit either to use the default. |

## Thinking

| Setting | Default | What it does |
|---|---|---|
| `thinking_mode` | `adaptive` | `adaptive` (a per-request heuristic), `always`, or `never`. |
| `max_thinking_tokens` | `2048` | Tokens the model may spend inside its reasoning block before the closing tag is forced. **Not a stop** — generation continues and the model still answers with what remains of `max_output_tokens`. `0` = uncapped. Enforced since 2026-08-12; before that it was passed to a chat template that never read it and did nothing at any value. |

## Sampling

The defaults reproduce Mira's behaviour before sampling parameters existed: greedy decoding.
They are **not** what the model asks for — Qwen3.6-35B-A3B's own `generation_config.json`
specifies `temperature: 1.0, top_k: 20, top_p: 0.95`. Change all three together or none;
temperature alone removes the guard without the truncation that keeps sampling out of the tail.

| Setting | Default | What it does |
|---|---|---|
| `temperature` | `0.0` | Sampling temperature. `0.0` is argmax. |
| `top_p` | `0.0` | Nucleus truncation. |
| `top_k` | `0` | Top-k truncation. Not an OpenAI-API parameter — passed via `extra_body`, honoured by mira-mlx, possibly ignored elsewhere. `0` disables. |
| `seed` | unset | Unset means a fresh seed per request, which is what makes "regenerate" return a different reply. Pinning it does nothing at temperature 0, and cannot make output reproducible while other requests are in flight — continuous batching changes the arithmetic itself. |

## Repetition penalties

All off, which is how mira-mlx has always run. A penalty makes a repetition loop less likely to
be entered but cannot remove it, and it taxes legitimate repetition too — this project's own
traffic has produced 16 identical lines legitimately against 355 for a degenerate case. Start low
and read a few long code replies before keeping one.

| Setting | Default | What it does |
|---|---|---|
| `repetition_penalty` | unset | Multiplicative; `1.0` = no effect, `1.05`–`1.1` is a gentle start. |
| `repetition_context_size` | `20` | Recent tokens the repetition penalty looks at. Only applies when the penalty is set. |
| `presence_penalty` | unset | Additive; `0.0` = no effect. |
| `presence_context_size` | `20` | Recent tokens the presence penalty looks at. |
| `frequency_penalty` | unset | Additive; `0.0` = no effect. |
| `frequency_context_size` | `20` | Recent tokens the frequency penalty looks at. |

## RAG, search and conversation

| Setting | Default | What it does |
|---|---|---|
| `embed_model` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace embedding model, downloaded once and cached. |
| `reranker_backend` | `qwen3` | `qwen3` uses Qwen3-Reranker-0.6B-4bit via mlx (~0.35 GB, scores 0–1); `crossencoder` uses ms-marco-MiniLM-L-6-v2 via sentence-transformers on CPU. |
| `reranker_model` | `mlx-community/Qwen3-Reranker-0.6B-4bit` | The reranker checkpoint for the chosen backend. |
| `brave_api_key` | `""` (or `$BRAVE_API_KEY`) | When set, Brave is the primary search provider; DuckDuckGo is the no-key fallback. Free tier is 2,000 queries/month. **Keep the real key out of the committed example file.** |
| `compress_threshold` | `70` | Context-fill percentage at which the summarize-and-compress pass fires at the end of a turn. |
| `compress_keep_recent` | `6` | Recent messages kept verbatim when compressing. Minimum 2, enforced in code. |

The remaining RAG knobs (`RAG_CHUNK_SIZE`, `RAG_RETRIEVE_K`, `RAG_RERANK_TOP_K`,
`RAG_SCORE_THRESHOLD`, `RAG_MAX_CHUNKS`, `MAX_SEARCH_RESULTS`, `MAX_TOOL_STEPS`, `MAX_RETRIES`,
`SEARCH_TIMEOUT`) have no `mira.yaml` equivalent — edit `core/config.py` directly.

## Inference tuning (mira-mlx)

| Setting | Default | What it does |
|---|---|---|
| `prefill_step_size` | `1024` | Tokens per prefill chunk. Powers of two only: 256, 512, 1024, 2048 (2048 is experimental on 32 GB). Applies to mira-mlx and mlx-lm. |
| `mira_mlx_kv_bits` | unset | Quantize the KV cache to this many bits. Only `8` is numerically validated (the fork's own suite, rtol 4e-2), buying roughly 1.6–2× usable context. 4-bit is unproven here. |
| `mira_mlx_kv_group_size` | `64` | Quantization group size for `mira_mlx_kv_bits`. |
| `mira_mlx_trust_remote_code` | `false` | Execute a model repo's own Python at load time. With this on, a model id is equivalent to code execution. Enable only for a specific model you trust that genuinely ships a custom tokenizer. |
| `memory_advisory_notifications` | `true` | Post a macOS notification when another app evicts Mira's model from memory. On since 2026-08-13; the `cause` field filters out the harmless idle-reclaim treadmill so it fires only for a genuine shortage. Fires only on the transition into that state, at most once every 15 minutes. |

## Vision (mira-mlx)

| Setting | Default | What it does |
|---|---|---|
| `mira_mlx_vision` | `false` | Read images with the checkpoint's own vision tower instead of OCR-ing them. Only works on a checkpoint that ships one; Qwen3.6-35B-A3B-4bit does. The tower loads on the first image turn, not at startup, so turning this on costs nothing until an image arrives. Image turns skip the prompt cache deliberately — an image is N copies of one placeholder token id, so two same-size screenshots would collide into a false hit. |
| `mira_mlx_vision_max_pixels` | `1048576` | Ceiling on an image's pixel count, and by far the biggest lever on what vision costs: 1 MP holds any image to roughly 1,000 tokens and ~1.6 s. Raise to `2097152` if fine visual detail matters. Only ever lowers the checkpoint's own ceiling. |
| `mira_mlx_vision_tower_idle_timeout` | `300` | Seconds without an image before the tower's 0.89 GB is released. It reloads in under two seconds. `0` keeps it resident once loaded. |

## MoE expert offload (mira-mlx)

| Setting | Default | What it does |
|---|---|---|
| `mira_mlx_expert_offload` | `auto` | `auto` offloads only when the fully-resident model would not fit, decided per model at launch. `on`/`true` always offloads; `off`/`false` never does. Makes a model whose expert table exceeds unified memory runnable at all, at a real throughput cost (~5× slower decode, ~8–12× slower prefill on Qwen3.6-35B-A3B). No-op on dense models. |
| `mira_mlx_resident_expert_fraction` | `0.3` | Fraction of each MoE layer's experts kept resident when offload is on. Lower = less RAM, more cold-prefill misses. `>= 1.0` is the same as offload off. |
| `mira_mlx_expert_ram_aware` | `true` | In `auto`, size the resident fraction to available RAM rather than the flat fraction above, which otherwise leaves memory idle. Measured +12% decode and +9% prefill on 8-bit Qwen3.6 with no quality change. The flat fraction becomes a floor — sizing only ever raises it. |
| `mira_mlx_profile_experts` | `false` | Log MoE expert-routing decisions to JSONL for offload analysis. Meant for a deliberate profiling window, not permanent use. |
| `mira_mlx_expert_profile_path` | `<data dir>/expert_profile/<unix-ts>.jsonl` | Where that log goes. |

## Opt-in performance flags

Each is measured and working on the maintainer's machine; what they have not had is a week of
ordinary use across different hardware, and each changes a path every request goes through.

| Setting | Default | What it does |
|---|---|---|
| `boundary_snapshot` | `false` | Reuse the prompt cache across turns of one conversation. Without it, plain multi-turn chat on Qwen3.6 re-prefills the whole conversation every turn: a 27,614-token second turn reused nothing and took 48.7 s; with it, 5.0 s reusing 27,500 tokens for a 14 ms snapshot cost. Agentic tool loops already reused well and are unaffected. |
| `proactive_decompress` | `false` | Fault the model back into RAM on the engine's idle branch when another app has had it compressed out, instead of leaving the bill for whoever asks next. Measured: 17.6 s next reply against a warm 0.45 s, with nothing recovering on its own for 12 minutes. Skipped on battery, at critical pressure, and without headroom. |
| `disk_prompt_cache` | `false` | Overflow evicted prompt-cache entries to disk. **Not merely unproven — it cannot work as built:** lookup is an exact hash of the whole prompt while entries are keyed on prompt plus everything generated, so a hit needs a byte-identical repeat of an entire conversation. Left on for three weeks it accumulated 39.75 GB and served zero reads. It stays because a prefix-capable version is a real idea, not because this one is worth enabling. |
| `mira_mlx_enable_tf32` | `true` | Let MLX use TF32 accumulation on M5+ NAX kernels. Listed because it changes results, not just speed: off costs ~2.6× on fp32 matmul and ~2.9× on the 4-bit MoE expert path, and buys ~10 mantissa bits that sit well under 4-bit quantization noise. Decode is unaffected either way. Set `false` only to reproduce an upstream bit-equivalence test. Write it as a YAML bool — the underlying env var is parsed as an integer, so the string `"true"` reads as OFF. |

---

## Environment variables

Not `mira.yaml` keys, but they override or complement it:

| Variable | Effect |
|---|---|
| `MIRA_TOKEN` | Supplies `auth_token` without putting it in a file. |
| `MIRA_DATA_DIR` | Moves everything Mira persists (`conversations.db`, the Chroma store, the prompt cache, engine logs). Defaults to `~/.local/share/mira`. The test suite sets it to a temp dir; before it existed, `pytest` wrote to the real database. |
| `MIRA_BACKEND_HOST` | Default for `host`. |
| `MIRA_HOST` | Bind address. Needed (with `allowed_source_cidrs`) to opt into plain-LAN mode, which is plaintext and sniffable. |
| `BRAVE_API_KEY` | Default for `brave_api_key`. |
| `MIRA_HOME` | Repo root used by `mira doctor` and the `mira` CLI when resolving a checkout. |
