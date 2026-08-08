"""Configuration settings for Mira. Defaults can be overridden via mira.yaml in the project root."""

import os
from pathlib import Path
from typing import Optional

# ── mira.yaml override loader ─────────────────────────────────────────────────
def _load_yaml_config() -> dict:
    yaml_path = Path(__file__).parent.parent / "mira.yaml"
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

# ── Thinking mode ─────────────────────────────────────────────────────────────
THINKING_MODE: str = _get("thinking_mode", "adaptive")  # adaptive | always | never
MAX_THINKING_TOKENS: int = _get("max_thinking_tokens", 8192)  # 0 = uncapped; minimum useful value ~512

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
# Post a macOS notification when another app evicts the model from memory. On by
# default because the state it reports is otherwise invisible: the user gets one
# unexplained slow reply (measured 15.37s against a warm 0.47s) and then normal
# speed again, because the slow reply is itself what fixes it. Rate-limited and
# fired only on the transition into eviction (see core/memory_watch.py).
MEMORY_ADVISORY_NOTIFICATIONS: bool = _get("memory_advisory_notifications", True)
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
