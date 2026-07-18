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
# "mlx-lm" and "omlx" use the OpenAI-compatible API; "ollama" uses the ollama Python client.
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

# mira-mlx is the default backend (see mira.yaml / README). These code defaults must
# match mira.yaml.example so a fresh install with no mira.yaml behaves as documented.
BACKEND: str = _get("backend", "mira-mlx")
MODEL_NAME: str = _get("model", "mlx-community/Qwen3.6-35B-A3B-4bit")
OLLAMA_HOST: str = _get("host", os.getenv("OLLAMA_HOST", "http://localhost:11434"))

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
USE_NATIVE_SEARCH = False  # Ollama native search off; Brave is primary when keyed, DDGS is the fallback (see docs/architecture.md)
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
DFLASH_CLI: str = _paths.get("dflash_cli", str(Path(__file__).parent.parent / ".venv" / "bin" / "dflash"))
OMLX_CLI: str = _paths.get("omlx_cli", "/Applications/oMLX.app/Contents/MacOS/omlx-cli")
VLLM_MLX_CLI: str = _paths.get("vllm_mlx_cli", str(Path.home() / ".local" / "bin" / "vllm-mlx"))

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT", str(Path.home() / "workspace"))
SHELL_TIMEOUT = 30  # seconds per shell command
TEMP_WORKSPACE_MAX_MB = 100  # per-conversation attachment workspace size cap

# ── Conversation persistence ──────────────────────────────────────────────────
DB_PATH = Path.home() / ".local" / "share" / "mira" / "conversations.db"
RAG_DIR = DB_PATH.parent / "chroma_db"
MAX_CONVERSATIONS = 1000
COMPRESS_THRESHOLD: int = _get("compress_threshold", 70)   # context_pct % at which summarize-and-compress fires
COMPRESS_KEEP_RECENT: int = max(2, _get("compress_keep_recent", 6))  # number of recent messages kept verbatim (min 2)
PREFILL_STEP_SIZE: int = _get("prefill_step_size", 1024)  # tokens per prefill chunk; must be power of 2 (256/512/1024/2048)
DFLASH_DIAGNOSTICS: str = _get("dflash_diagnostics", "off")  # off | basic | full; basic=request/cache logs, full=+memory waterfall
# mira-mlx only. None (default) = unquantized fp16 KV cache, today's behavior.
# 8 is the only numerically-validated bit width (mlx-lm fork's own test suite,
# rtol=4e-2); 4-bit is unproven anywhere in this codebase.
MIRA_MLX_KV_BITS: Optional[int] = _get("mira_mlx_kv_bits", None)
MIRA_MLX_KV_GROUP_SIZE: int = _get("mira_mlx_kv_group_size", 64)
# Opt-in MoE expert-routing logging for the expert-offloading go/no-go decision
# (specs/moe-expert-offload-01-profiling.md). False (default) = zero overhead,
# no-op on dense models. Not meant to stay on by default even for Qwen3.6 —
# only enable for a deliberate profiling window.
MIRA_MLX_PROFILE_EXPERTS: bool = _get("mira_mlx_profile_experts", False)
MIRA_MLX_EXPERT_PROFILE_PATH: Optional[str] = _get("mira_mlx_expert_profile_path", None)
