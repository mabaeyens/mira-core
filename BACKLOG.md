# Backlog

## Done

- [2026-05-20] Implemented `/compact` slash command — `POST /compact` server endpoint + Swift intercept in `send()`; appends `.info` bubble with result; no-op handled gracefully for empty history.
- [2026-05-20] Improved `_should_think()` heuristic — replaced binary check with scoring function; trivial acknowledgements short-circuit; attachments, length, code signals, and think-verbs each scored; threshold ≥ 3. Reduces unnecessary thinking on casual messages.
- [2026-05-20] Added `num_keep 768` to `gemma4-optimized.modelfile` — pins system prompt tokens in KV cache across turns; saves ~200ms prefill per turn.
- [2026-05-20] Fixed invalid `llama.cpp.*` params in Modelfile — those were never valid Ollama syntax and caused `ollama create` to fail silently. kv_cache_type and flash_attn are global-only env vars.
- [2026-05-20] Added `specs/` to `.gitignore` — local pre-coding session plans, not for review.
- [2026-05-20] Refreshed `CLAUDE.md` — reflects current stack (gemma4:26b via Ollama, no oMLX).

## Pending

### Harness quality
- [ ] Parallel tool execution — orchestrator runs only the first tool call per step; if the model emits multiple tool calls in one response, the rest are dropped; execute all in parallel and merge results before the next turn
- [ ] Shell timeout 30s → configurable per-call — long builds and test suites time out; add optional `timeout` arg to `run_shell` (cap at e.g. 300s)

### Inference speed
- [ ] Watch for quantized MLX gemma4:26b variant in Ollama registry — current `gemma4:26b` tag runs llama.cpp at ~38 tok/s; MLX path requires bf16 (52GB, won't fit 32GB RAM) and has a cold-prefill bug (#16051); revisit when a q4/q8 MLX tag appears
- [ ] Monitor oMLX tool-calling stability for gemma4 — issues #617/#666 block it as a Mira backend; check again after oMLX 0.4.x

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
- Router Agent considered and rejected — adds ~300ms overhead to every complex query (majority of Mira sessions); heuristic-based thinking toggle is the right lever at zero cost
