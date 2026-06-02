# Backlog

## Done (recent — see `git log` for full history)

- [2026-05-31] mlx-lm model warmup — `_warmup_model()` added to `backend_manager.py`; called at end of `ensure_backend_running("mlx-lm")` whether server was already running or freshly started. Eliminates 29–34 s cold-start penalty for the first user request (model now pre-loaded in VRAM during background startup thread).
- [2026-05-31] Backend benchmark — `scripts/benchmark.py` added: 9-cell matrix (3 prompts × 3 positions × reps), Ollama + mlx-lm, Haiku analysis via `claude` CLI. Key findings: mlx-lm warm TTFT 300–450 ms, Ollama cache saves 4–5× on warm prefix, Qwen3 thinking overhead negligible on mlx-lm (7 ms), expensive on Ollama (104% TTFT). Recommendation: default to mlx-lm + Qwen3.6-35B with thinking on.
- [2026-05-31] Thinking heuristics — `_NEVER_THINK` pattern blocks thinking on trivial commands (time, date, "fix X.ext"); `_ANALYTICAL_WITH_ATTACHMENT` gates the attachment bonus to analytical verbs only; bare-attachment score bonus reduced (3 → 1+2); threshold raised (3 → 4). Eliminates thinking on "summarize this file in 3 words" and similar.
- [2026-05-31] Local file prompt guard — system prompt now opens with a hard block: if user mentions a filename with no workspace open, ask to attach the file instead of searching GitHub/web.
- [2026-05-30] Q13 prompt refinement + divergence guard validation — "call run_shell once per check, no shell loops or sleep" constraint added to Q13 prompt. Validated on both models (gemma4 2/2, qwen3.6 2/2). Guard is now reliably triggered.
- [2026-05-30] mira-apps mlx-lm follow-up — model pill shows `modelDisplayName`; thinking chip/toggle disabled on mlx-lm backend; `backendLabel()` helper; CLAUDE.md and collaboration-notes.md updated to reflect mlx-lm as primary engine.
- [2026-05-30] mlx-lm promoted to first-class backend — `backend_manager.py` gains `mlx-lm` preset and proper `is_backend_ready`/`ensure_backend_running`/`switch_to` branches. `mira.yaml` switched to `backend: mlx-lm`. Smoke-tested end-to-end.

## Pending

### Deferred
- [ ] Unsloth UD-MLX-4bit bench — `unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit` (15 GB) cached locally; bench vs uniform 4-bit. Only worth running in a dedicated bench session.
- [ ] Scanned PDF OCR — detect scanned PDFs (empty text layer), run OCR (e.g. `tesseract`) before indexing.
- [ ] Server-side auth token check — add `verify_token` FastAPI dependency to `/chat`; reads `MIRA_TOKEN` env var; no-op if unset (backwards compatible). Client already sends `Bearer` token. ~15 lines in `server.py`.
- [ ] HTTPS on LAN via profile flow — self-signed CA on startup, `.mobileconfig` endpoint, QR code sheet in mira-apps connection settings. Tailscale HTTPS already works; this covers direct LAN only.

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
- mlx-lm 0.31.3 (**CURRENT DEFAULT** as of 2026-05-30): thinking suppression via `--chat-template-args '{"enable_thinking": false}'` (template-level). Gemma4 warm TTFT 250–505ms, ~35–36 t/s, 4–6× wall time improvement vs Ollama on agentic tasks. Binary at `~/.local/bin/mlx_lm.server`. mira.yaml: `backend: mlx-lm, model: mlx-community/gemma-4-26b-a4b-it-4bit, host: http://localhost:8080`. mlx-lm is now a first-class backend in `backend_manager.py` (not routed through omlx). Caffeinate is server-PID-bound — works for any backend. Ollama login item disabled; start manually when RAG embeddings needed.
- Quantization audit complete (2026-05-30): uniform 4-bit is optimal for 32GB M5. OptiQ mixed-precision costs 20–25% decode throughput (Apple Silicon memory bandwidth bottleneck). 8-bit barely fits, kills KV headroom. MTP for Gemma 4 is vision-path only (mlx-vlm), not available in text-only mlx-lm. Model cache paths: `docs/model-cache.md` (gitignored, machine-specific). Also cached but not active: OptiQ-4bit (16 GB) and unsloth UD-MLX-4bit (15 GB).
- omlx 0.3.12: reinstalled May 2026. Did not crash (0.3.8/0.3.9 regression fixed). No performance advantage over mlx-lm — gemma4 throughput identical but wall time 3–4× worse; qwen3.6 15–30× TTFT regression. Not recommended as Mira backend. See `docs/omlx-ctl.md`.
- MLX community benchmark leaderboard (2026-05-30): gemma-4-26b-a4b-it 75.2% (#5 overall, strong on medium 82.3%); qwen3.6-35b-a3b local 52.5% (#10, weak on medium 46.4% and very hard 36%). Note: this contradicts SWE-bench results — different task distribution. Use both leaderboards together for full picture.
- Agentic Critic LLM call considered and rejected — proposed plan called for a secondary model call after every tool observation to decide CONTINUE/FINISH; replaced with `task_done` tool + ReAct-style prompting (RULE 7); zero extra latency, same effect
- Benchmark design (2026-05-31): Ollama tok/s values are char-count estimates (usage stats not returned) — unreliable, don't compare to mlx-lm. mlx-lm "cold" first-model TTFT (~30 s) is a one-time model-switch cost, not a KV miss — becomes irrelevant once `_warmup_model()` pre-loads on startup. Ollama/Qwen3 thinking mode with max_tokens=512 hits the generation window before emitting content for medium/long prompts (TTFT=null); use higher max_tokens or no_think for those lengths.
- Router Agent considered and rejected — adds ~300ms overhead to every complex query (majority of Mira sessions); heuristic-based thinking toggle is the right lever at zero cost
