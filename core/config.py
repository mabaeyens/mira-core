"""Configuration settings for Mira. Defaults can be overridden via mira.yaml in the project root."""

import os
from pathlib import Path
from typing import Optional

# ── mira.yaml override loader ─────────────────────────────────────────────────
def _load_yaml_config() -> dict:
    # MIRA_CONFIG points at an alternative mira.yaml. The bench uses it to run an
    # isolated server on a copy of the live config with one setting changed,
    # instead of editing the real file and hoping to put it back. Unset in normal
    # use, which is every case except a bench.
    override = os.getenv("MIRA_CONFIG")
    yaml_path = Path(override) if override else Path(__file__).parent.parent / "mira.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

_cfg = _load_yaml_config()

def _get(key: str, default):
    return _cfg.get(key, default)

# ── Backend ───────────────────────────────────────────────────────────────────
# Every supported backend speaks the OpenAI-compatible API.
# Shared-secret auth. When set, every sensitive route requires
# `Authorization: Bearer <token>`. When unset, the server refuses to bind a
# non-loopback host (see server.py __main__) so an open server stays local-only.
# Read from mira.yaml `auth_token:` or the MIRA_TOKEN env var.
AUTH_TOKEN: str = _get("auth_token", os.getenv("MIRA_TOKEN", ""))

# Minimum token length we consider safe when binding off-host. A short hand-set
# token triggers a loud startup warning (see server.py __main__).
MIN_TOKEN_LENGTH: int = 32

# Source-IP allowlist (defense-in-depth behind the off-host bind). Off-host listeners
# bind the Tailscale interface, so the socket only exists on the tailnet; this list is
# a secondary guard (e.g. against a future MIRA_HOST=0.0.0.0 misconfig). Defaults:
# loopback + the Tailscale CGNAT range. Add a LAN subnet here only to opt back into
# plain-WiFi access (documented as plaintext/sniffable — see docs/remote-access.md).
ALLOWED_SOURCE_CIDRS: list = _get(
    "allowed_source_cidrs", ["127.0.0.0/8", "::1/128", "100.64.0.0/10"]
)

# Extra Host header values to accept, beyond loopback and the discovered tailnet
# address. Needed if you reach Mira by a name rather than an IP — a Tailscale
# MagicDNS name like "mira.tail1234.ts.net", or a custom /etc/hosts alias.
#
# Why the Host header is checked at all: a browser can be pointed at a hostname
# the attacker controls which resolves to 127.0.0.1 (DNS rebinding). The request
# then arrives on loopback, from a real browser, carrying the attacker's page as
# the origin — the source-IP allowlist cannot tell it apart. Pinning the accepted
# Host values is what distinguishes "the user typed localhost" from "a page the
# user visited pointed its own domain at localhost".
ALLOWED_HOSTS: list = _get("allowed_hosts", [])

# Whether fetch_url may reach loopback / private / link-local addresses. The URL
# is chosen by the model, which reads attacker-influenceable text, so the default
# is off: a crafted page should not be able to steer it at a LAN device, a
# loopback service, or a metadata endpoint. Turn on if you genuinely want Mira
# summarizing localhost or LAN pages.
URL_FETCH_ALLOW_PRIVATE: bool = _get("url_fetch_allow_private", False)

# mira-mlx is the default backend (see mira.yaml / README). These code defaults must
# match mira.yaml.example so a fresh install with no mira.yaml behaves as documented.
BACKEND: str = _get("backend", "mira-mlx")
MODEL_NAME: str = _get("model", "mlx-community/Qwen3.6-35B-A3B-4bit")
# The inference backend's base URL. Named OLLAMA_HOST until 2026-08-01,
# when the ollama backend was retired; it had long since stopped meaning
# ollama and just meant whatever `host:` in mira.yaml points at.
BACKEND_HOST: str = _get("host", os.getenv("MIRA_BACKEND_HOST", "http://localhost:8080"))

# Named backend presets exposed via GET /backends and shown in the app model picker.
# If empty (no `backends:` in mira.yaml), backend_manager falls back to its PRESETS dict.
BACKENDS: list = _get("backends", [])

# ── Embedding backend (for RAG) ───────────────────────────────────────────────
EMBED_MODEL: str = _get("embed_model", "nomic-ai/nomic-embed-text-v1.5")

# ── Context window ────────────────────────────────────────────────────────────
CONTEXT_WINDOW: int = _get("context_window", 65536)

# Cap on a single generation. Was hardcoded to 4096 in three places until
# 2026-08-11, when two batches of real conversations showed what that costs:
# 13 of 51 turns hit the cap, and every one of them reached the user broken --
# either as raw chain of thought presented as the answer, or as 4096 copies of
# one character, which then poisons the conversation history for good.
#
# 4096 was never enough for a thinking model. Measured reasoning on ordinary
# questions ran to 19k characters, so thinking alone consumed the whole budget
# before the answer began. Note that MAX_THINKING_TOKENS was 8192 at the time --
# double the old total -- which is how long this went unnoticed: a budget that
# large can never bind, and the backend ignored it entirely until 2026-08-12.
#
# Raising it does not make replies longer, it stops them being cut off; a model
# that is done still emits its stop token. The cost of a higher ceiling is the
# worst case, not the common case: 16384 tokens at ~59 tok/s is about four and
# a half minutes, which the longest healthy turns already reached.
MAX_OUTPUT_TOKENS: int = _get("max_output_tokens", 16384)

# ── Thinking mode ─────────────────────────────────────────────────────────────
THINKING_MODE: str = _get("thinking_mode", "adaptive")  # adaptive | always | never
# 0 = uncapped; minimum useful value ~512.
#
# Enforced since 2026-08-12 (a logits processor that forces `</think>` once the
# block runs past the budget -- it does NOT stop generation, so the model still
# gets to answer). Before that the number was sent as a chat-template kwarg that
# Qwen3.6's template never reads, so it did nothing at any value.
#
# 8192 could not bind on anything real: the worst turn measured spent ~5k
# reasoning tokens, so the budget only ever caught a runaway. Lowered to 2048 on
# 2026-08-12 to make it an actual budget. The reason it is worth binding: decode
# is 77-88% of a turn's wall clock and reasoning is 24-44% of everything
# generated, so tokens not spent thinking come off the clock nearly one for one.
#
# This DOES change replies, which is why it wants a corpus run behind it rather
# than an argument. The risk it runs is the failure this project already fixed
# once -- an answer cut short because thinking ate the budget -- except that the
# mechanism is different now: at the cap the closer is forced and the remaining
# MAX_OUTPUT_TOKENS are still available for the answer.
MAX_THINKING_TOKENS: int = _get("max_thinking_tokens", 2048)

# ── Sampling ──────────────────────────────────────────────────────────────────
# Until 2026-08-09 none of these existed and nothing sent them, so every Mira
# reply was greedy-decoded: mira_mlx_server defaulted temperature and top_p to
# 0.0 and the orchestrator never overrode them.
#
# The defaults below preserve that behaviour exactly, so this change adds a knob
# without altering a single reply. They are NOT what the model asks for:
# Qwen3.6-35B-A3B ships generation_config.json with do_sample: true,
# temperature: 1.0, top_k: 20, top_p: 0.95.
#
# Set all three together rather than temperature alone -- that is the model
# author's own specification, and top_k/top_p are the truncation that keeps
# sampling out of the distribution's tail.
#
# That advice rests on the shipped generation_config, NOT on local measurement.
# Two probe rounds on 2026-08-09 contradicted each other: round 1 saw temperature
# 1.0 alone produce 303 repeats of one sentence, round 2 saw the same setting
# produce the cleanest output of five runs. Output on this machine is not
# reproducible across server processes, so single runs rank nothing. Treat any
# sampling comparison here as unmeasured until someone runs repeats against a
# fixed cache state. See mira.yaml.example.
#
# top_k is passed through extra_body: it is not an OpenAI-API parameter, but
# mlx_lm's make_sampler supports it and mira-mlx honours it. 0 disables it.
TEMPERATURE: float = _get("temperature", 0.0)
TOP_P: float = _get("top_p", 0.0)
TOP_K: int = _get("top_k", 0)

# Seed. Output is byte-identical for the same (prompt, params) even at
# temperature 1.0 -- two identical requests come back the same, so regenerating
# a reply hands the user exactly what they just rejected. There is no response
# cache; mlx-lm's samplers thread mx.random.state through mx.compile, so every
# request effectively starts from the same RNG state.
#
# null (the default) means the engine draws a fresh seed per request, so a
# regenerate genuinely resamples. An integer pins it, which is what a
# reproducibility run wants. Either way this is inert at TEMPERATURE 0.0:
# make_sampler returns argmax and never touches the RNG. It also cannot make
# output reproducible across concurrent traffic -- continuous batching changes
# the arithmetic itself (see docs/batch-invariance.md).
SEED: Optional[int] = _get("seed", None)

# ── Runaway guard: repetition penalties ───────────────────────────────────────
# mlx-lm has had these all along and mira-mlx called make_logits_processors()
# with no arguments, so every one of them was off. This wires them; the defaults
# below keep them off, so output is unchanged until someone sets one.
#
# What they are for: a repetition loop is a high-probability region of the
# model's own distribution (Holtzman et al., ICLR 2020, measured 43% repeated
# n-grams under greedy decoding against 0.5% for humans), so decoding can make a
# loop less likely to be entered but cannot remove one. Treat these as a guard
# that lowers the rate, never as a fix. The measured local case was one line
# repeated 355 times, filling an 8,192-token budget.
#
# Setting them is not free. Clean output in this project's own traffic reached
# x16 identical lines legitimately -- a JSON config block -- so a penalty strong
# enough to stop x355 also taxes real code, tables and lists. Start at
# repetition_penalty ~1.05-1.1 over a large context window rather than the
# aggressive values found in forum posts, and read a few long code replies
# before keeping it.
#
# repetition_penalty is multiplicative and sign-aware (1.0 = no effect);
# presence and frequency are additive (0.0 = no effect). None/0 means the
# processor is never constructed, which is why the defaults are None rather
# than the neutral numbers.
REPETITION_PENALTY: Optional[float] = _get("repetition_penalty", None)
REPETITION_CONTEXT_SIZE: int = _get("repetition_context_size", 20)
PRESENCE_PENALTY: Optional[float] = _get("presence_penalty", None)
PRESENCE_CONTEXT_SIZE: int = _get("presence_context_size", 20)
FREQUENCY_PENALTY: Optional[float] = _get("frequency_penalty", None)
FREQUENCY_CONTEXT_SIZE: int = _get("frequency_context_size", 20)

# ── Search ────────────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS = 5
MAX_AGENT_STEPS = 15     # raised cap for agentic multi-step tasks
AGENT_DIVERGENCE_LIMIT = 1  # identical tool+args repeats before injecting a redirect
MAX_TOOL_CALLS_PER_TURN = 20  # hard total tool call cap across all steps in one turn
SAME_TOOL_REPEAT_LIMIT = 15  # same tool name N times in one turn → bail (catches near-identical loops)
TOOL_SOFT_LIMIT = 10     # per-tool calls before pausing to check in with the user
UNPRODUCTIVE_TOOL_REPEAT_LIMITS: dict = {}  # no per-tool hard caps; soft limit handles research use cases
MAX_RETRIES = 3          # API-level error retries per model call
# Brave is primary when a key is set, DuckDuckGo is the fallback
# (see docs/architecture.md).
SEARCH_TIMEOUT = 30
BRAVE_API_KEY: str = _get("brave_api_key", os.getenv("BRAVE_API_KEY", ""))

# ── Display ───────────────────────────────────────────────────────────────────
VERBOSE_DEFAULT = False
ANSWER_PREFIX = "🤖 "
SEARCH_PREFIX = "🔍 "
ERROR_PREFIX = "❌ "

# ── RAG ───────────────────────────────────────────────────────────────────────
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # downloaded on first use (~100 MB)
RERANKER_BACKEND: str = _get("reranker_backend", "qwen3")   # qwen3 | crossencoder
RERANKER_MODEL: str = _get("reranker_model", "mlx-community/Qwen3-Reranker-0.6B-4bit")
RAG_CHUNK_SIZE = 400        # words per chunk
RAG_CHUNK_OVERLAP = 40      # word overlap between adjacent chunks
RAG_RETRIEVE_K = 10         # candidates retrieved before reranking
RAG_RERANK_TOP_K = 4        # chunks injected into context after reranking
RAG_SCORE_THRESHOLD = 0.0   # CrossEncoder scores below this are dropped
RAG_MAX_CHUNKS = 10_000     # warn user to unload documents above this total

# ── CLI paths (local install locations; override in mira.yaml under paths:) ───
_paths = _get("paths", {})
MLX_LM_CLI: str = _paths.get("mlx_lm_cli", str(Path.home() / ".local" / "bin" / "mlx_lm.server"))
OMLX_CLI: str = _paths.get("omlx_cli", "/Applications/oMLX.app/Contents/MacOS/omlx-cli")
VLLM_MLX_CLI: str = _paths.get("vllm_mlx_cli", str(Path.home() / ".local" / "bin" / "vllm-mlx"))

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT", str(Path.home() / "workspace"))
SHELL_TIMEOUT = 30  # seconds per shell command
# run_shell wraps every command in an OS sandbox (macOS sandbox-exec) that
# confines writes to the workspace + temp dirs. Regex prefiltering cannot contain
# a shell; this can. Fails closed if sandbox-exec is unavailable.
SHELL_SANDBOX: bool = _get("shell_sandbox", True)
# Network defaults ON — git pull / npm install / curl are legitimate here. Set
# false in mira.yaml for untrusted sessions to also block outbound connections.
SHELL_SANDBOX_ALLOW_NETWORK: bool = _get("shell_sandbox_allow_network", True)
TEMP_WORKSPACE_MAX_MB = 100  # per-conversation attachment workspace size cap

# ── Conversation persistence ──────────────────────────────────────────────────
# Everything Mira persists lives under one directory, overridable with
# MIRA_DATA_DIR. The default is unchanged, so a normal install and the running
# LaunchAgent see exactly the same paths as before. The override exists so the
# test suite can point at a throwaway directory: without it `pytest` wrote to the
# real conversations.db and left rows in the user's own history (22 of them in a
# single session, found 2026-08-01). Also useful for running a second instance
# against separate data.
DATA_DIR = Path(os.getenv("MIRA_DATA_DIR", str(Path.home() / ".local" / "share" / "mira")))
DB_PATH = DATA_DIR / "conversations.db"
RAG_DIR = DATA_DIR / "chroma_db"
MAX_CONVERSATIONS = 1000
COMPRESS_THRESHOLD: int = _get("compress_threshold", 70)   # context_pct % at which summarize-and-compress fires
COMPRESS_KEEP_RECENT: int = max(2, _get("compress_keep_recent", 6))  # number of recent messages kept verbatim (min 2)
PREFILL_STEP_SIZE: int = _get("prefill_step_size", 1024)  # tokens per prefill chunk; must be power of 2 (256/512/1024/2048)
# mira-mlx only. None (default) = unquantized fp16 KV cache, today's behavior.
# 8 is the only numerically-validated bit width (mlx-lm fork's own test suite,
# rtol=4e-2); 4-bit is unproven anywhere in this codebase.
MIRA_MLX_KV_BITS: Optional[int] = _get("mira_mlx_kv_bits", None)
MIRA_MLX_KV_GROUP_SIZE: int = _get("mira_mlx_kv_group_size", 64)
# Load the checkpoint's own vision tower so screenshots are read as images
# instead of run through OCR. Off by default: it costs about 0.89GB resident on
# Qwen3.6-35B-A3B and only some checkpoints ship a tower at all. With this off,
# nothing about vision is imported and images keep taking the OCR path.
MIRA_MLX_VISION: bool = _get("mira_mlx_vision", False)
# Ceiling on an image's pixel count after Qwen's smart-resize. The checkpoint
# asks for 16,777,216, which caps nothing in practice: a 5712x4284 phone photo
# stays at 16,170 image tokens, 243s of tower time and 126MB of embeddings.
# 1 MP holds any image to ~1k tokens and ~1.6s in the tower.
#
# 1 MP over 2 MP is measured, not assumed (2026-08-02, four real images from
# 2.4MP to 24MP). A game screenshot at 1 MP still read every UI label and all
# five skill names, so the "small text needs 2 MP" worry did not survive
# contact: text held, and OCR covers the dense-text case anyway. What does
# degrade is fine visual attributes - the same screenshot gave "glowing blue
# and silver armor" at 2 MP and "blue skin, dark armor" at 1 MP. Raise this to
# 2097152 if that kind of detail matters more than ~4s per turn.
#
# Only ever lowers the checkpoint's own ceiling, never raises it.
MIRA_MLX_VISION_MAX_PIXELS: int = _get("mira_mlx_vision_max_pixels", 1024 * 1024)
# Seconds without an image before the vision tower's 0.89GB is released again.
# The tower is loaded lazily on the first image turn, never at startup, so a
# text-only session never pays for it. Reload is 0.14s page-cached (1.94s cold)
# and Metal kernels survive the round trip. 0 keeps it resident once loaded.
MIRA_MLX_VISION_TOWER_IDLE_TIMEOUT: float = _get(
    "mira_mlx_vision_tower_idle_timeout", 300.0
)
# Opt-in MoE expert-routing logging for the expert-offloading go/no-go decision
# (docs/moe-offload-case-study.md). False (default) = zero overhead,
# no-op on dense models. Not meant to stay on by default even for Qwen3.6 —
# only enable for a deliberate profiling window.
MIRA_MLX_PROFILE_EXPERTS: bool = _get("mira_mlx_profile_experts", False)
MIRA_MLX_EXPERT_PROFILE_PATH: Optional[str] = _get("mira_mlx_expert_profile_path", None)
# Executing a model repo's own Python at load time. Default off: with this on,
# loading any repo runs that repo's tokenizer code in-process, so a model id is
# equivalent to code execution. Enable only for a specific model you trust that
# genuinely ships a custom tokenizer class.
MIRA_MLX_TRUST_REMOTE_CODE: bool = _get("mira_mlx_trust_remote_code", False)
# Post a macOS notification when another app evicts the model from memory.
#
# OFF by default since 2026-08-11, reversing the original on-by-default. The
# reasoning that put it on still holds — the state is otherwise invisible and
# produces one unexplained slow reply (15.37s against a warm 0.47s) — but it
# assumed evictions were rare. Measured over 70.5 hours they are not: 302
# transitions, one every ~14 minutes, round the clock, including three that woke
# nobody up only because they arrived at 01:30, 03:00 and 06:53.
#
# The notification also fails both tests a user-facing alert has to pass: there
# is nothing the user can do about another process taking memory, and it is not
# something they caused. `proactive_decompress` makes it worse by clearing the
# advisory each time, which re-arms the transition and manufactures the next
# notification.
#
# The watcher still runs and still logs every transition — that record is what
# made the 302 finding possible. Only the interruption is gone.
MEMORY_ADVISORY_NOTIFICATIONS: bool = _get("memory_advisory_notifications", False)
# Fault the model back into RAM on the engine's idle branch when another app has
# had it compressed out, instead of letting the next reply pay for it. Measured
# on a real unforced eviction (2026-08-08): all 18.80GB compressed, next turn
# 17.60s against a warm 0.45s, and no self-recovery in the 12 minutes before that
# turn. A half-evicted model cost 3.38s, so the saving scales with how much went
# out. Roughly memory-neutral (~1GB net, measured against a warm-turn control),
# because emptying the compressor pays for most of the expansion.
#
# OFF by default for now. The advisory notification already tells the user what
# is happening, and this one spends unrequested memory traffic on a machine that
# is by definition short of memory, so it wants a week of real use behind it
# before it becomes the default. Preconditions live in mira_mlx_server.py:
# per-process eviction signal only, availability floor, not on battery, not at
# critical pressure, once per eviction event.
PROACTIVE_DECOMPRESS: bool = _get("proactive_decompress", False)
# Off by default while it proves itself in real use, like proactive_decompress
# before it: it changes the prefill path of every request.
BOUNDARY_SNAPSHOT: bool = _get("boundary_snapshot", False)
# Overflow prompt-cache entries evicted from memory to disk
# (core/inference/disk_prompt_cache.py).
#
# OFF, because measured over three weeks of real use it served **zero reads**
# while holding 39.75GB at its own 39.86GB cap, evicting entries to make room
# for new ones that could not be read either. That is not a bug in the store: a
# lookup is an exact-match sha256 over the *full* token list, while an entry is
# keyed on prompt + everything generated. For a hit, a new prompt would have to
# equal some earlier prompt-plus-completion byte for byte, which the chat
# template alone makes impossible. The layer above it matches prefixes; this one
# cannot, so it can only ever hit on a literal repeat.
#
# Verified three independent ways before turning it off (2026-08-08):
# `disk_cache_hits` 0, no read spread over the period (the only later atimes
# cluster into two analysis sweeps), and the structural argument above.
#
# The code stays because a *prefix-capable* disk layer is still a real idea
# (specs/prefix-aware-disk-prompt-cache.md). It must not be revived on this flag
# alone: entries are 38-262MB and a median load is ~0.02s, so a candidate search
# that loads entries has to be measured against the ~4.8s prefill it replaces
# before it is worth any disk at all.
DISK_PROMPT_CACHE: bool = _get("disk_prompt_cache", False)
# TF32 accumulation on the M5+ NAX kernels. MLX defaults this on and until now
# Mira inherited that default without ever choosing it, which matters because
# the flag changes numerics: mlx#3897 traced the M5 batch-vs-single attention
# divergence to TF32 accumulation inside the NAX kernel (about 2^-11), and
# MLX_ENABLE_TF32=0 is what makes mlx-lm's test_generate pass 28/28 here.
#
# Kept ON deliberately. Measured on this machine (M5, applegpu_g17g, mlx
# 0.32.0, clean run 2026-08-08), turning it off costs:
#   fp32 4096-square matmul   8804 -> 3412 GFLOP/s   (2.58x)
#   4-bit quantized matmul    8590 -> 3185 GFLOP/s   (2.70x)
#   gather_qmm, real MoE dims 8933 -> 3095 GFLOP/s   (2.89x)
# and buys about 10 mantissa bits back on fp32 accumulation. On a 4-bit model
# that accuracy is far below the quantization noise floor, so paying up to 2.9x
# of prefill for it is the wrong trade. Nothing is lost at decode either way:
# the NAX path needs more than 16 rows to engage (measured identical at 16,
# 1135 vs 1136, and 1.74x apart at 32), and decode runs at one row, where the
# setting measures identically on and off (359 vs 369 GFLOP/s).
#
# Set false only to reproduce an upstream bit-equivalence test.
#
# NOTE the value is emitted as "1"/"0" deliberately: MLX parses this variable as
# an integer, so MLX_ENABLE_TF32=true measures as OFF (3185 GFLOP/s, same as
# "0"), not on. Keep this a YAML bool here and let the conversion happen below.
MIRA_MLX_ENABLE_TF32: bool = _get("mira_mlx_enable_tf32", True)
# MoE expert disk offloading (docs/offload-resident-sizing.md).
# Only `mira_mlx_resident_expert_fraction` of each MoE layer's experts stay
# resident; the rest are fetched on demand from the model's own safetensors
# shards and LRU-evicted. This makes a model whose expert table exceeds unified
# memory runnable at all — but it has a REAL throughput cost, measured
# 2026-07-19 on Qwen3.6-35B-A3B: ~5x slower decode (57 -> 11 tok/s) and ~8-12x
# slower prefill vs fully resident, because a diverse prefill touches nearly
# every expert and the resident fraction can't hold that working set (warm ~=
# cold). So it is NOT a free RAM win — pay it only when a model would not
# otherwise fit.
#
# Mode (`mira_mlx_expert_offload`):
#   auto (default) — offload ONLY when the fully-resident model would not fit
#                    (see resolve_offload_fraction); models that fit run resident
#                    at full speed. This is per-model, decided at launch.
#   on / true      — always offload (the tuning knob's fraction)
#   off / false    — never offload (fully resident, the pre-offload path)
def _parse_offload_mode(v) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    s = str(v).strip().lower()
    if s in ("on", "true", "yes", "1"):
        return "on"
    if s in ("off", "false", "no", "0"):
        return "off"
    return "auto"


MIRA_MLX_EXPERT_OFFLOAD_MODE: str = _parse_offload_mode(_get("mira_mlx_expert_offload", "auto"))

# In "auto" mode, when a model is offloaded because it would not fit fully
# resident, size the resident fraction to the RAM actually available instead of
# the flat tuning knob: an over-DRAM model leaves memory idle at 0.3 (8bit
# Qwen3.6 peaks ~12.7GB on a 32GB Mac). Sizing to keep peak near 55% of RAM
# raises the 8bit to 0.45 and was measured at +12% decode AND +9% prefill (fewer
# misses, still ample prefill-transient headroom), no prediction, no quality
# change. Never lowers below the configured fraction. Off => flat everywhere.
MIRA_MLX_EXPERT_RAM_AWARE: bool = _get("mira_mlx_expert_ram_aware", True)


def _resolve_resident_fraction(offload_enabled: bool, raw_fraction) -> Optional[float]:
    """Collapse the on/off switch and the tuning knob into the single effective
    resident fraction the rest of the stack gates on (`... is not None`). Returns
    None when offloading is off — either the flag is false, the fraction is unset
    to null, or it's >= 1.0 ("keep everything resident"), so no consumer has to
    special-case any of those."""
    if not offload_enabled or raw_fraction is None or float(raw_fraction) >= 1.0:
        return None
    return float(raw_fraction)


# The configured resident fraction (the tuning knob), or None if force-off /
# unset / >= 1.0. In "auto" mode this is the fraction USED when offload turns on;
# whether it turns on is decided per-model by resolve_offload_fraction().
MIRA_MLX_RESIDENT_EXPERT_FRACTION: Optional[float] = _resolve_resident_fraction(
    MIRA_MLX_EXPERT_OFFLOAD_MODE != "off", _get("mira_mlx_resident_expert_fraction", 0.3)
)


def resolve_offload_fraction(model_id: str) -> Optional[float]:
    """Effective resident-expert fraction for a specific model at launch.

    off  -> None (never offload).
    on   -> the configured fraction (always offload).
    auto -> offload only if the FULLY-RESIDENT model would not fit in memory
            (hardware.fits_in_memory with resident_expert_fraction=None); models
            that fit run resident at full speed. Offload's ~5x decode cost is
            only worth paying when it's the difference between running and not.
            When it does offload, the fraction is RAM-aware (sized to available
            memory, >= the configured knob) unless mira_mlx_expert_ram_aware is
            off — an over-DRAM model shouldn't leave memory idle at a flat 0.3.
    """
    candidate = MIRA_MLX_RESIDENT_EXPERT_FRACTION
    if candidate is None or MIRA_MLX_EXPERT_OFFLOAD_MODE == "off":
        return None
    if MIRA_MLX_EXPERT_OFFLOAD_MODE == "on":
        return candidate  # explicit manual override — respect the knob as-is
    # auto: enable offload only when the model can't fit fully resident.
    from core import hardware
    fits, _reason = hardware.fits_in_memory(
        model_id, kv_bits=MIRA_MLX_KV_BITS, kv_group_size=MIRA_MLX_KV_GROUP_SIZE,
        resident_expert_fraction=None,
    )
    if fits:
        return None
    if MIRA_MLX_EXPERT_RAM_AWARE:
        return hardware.derive_resident_expert_fraction(model_id, floor_fraction=candidate)
    return candidate
