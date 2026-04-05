"""Configuration settings for the ollama Search Tool."""

import os
from pathlib import Path

# Model settings
MODEL_NAME = "gemma4:26b"
MAX_SEARCH_RESULTS = 5
MAX_TOOL_STEPS = 5   # max tool calls per user turn before giving up
MAX_RETRIES = 3      # API-level error retries per model call

# Search settings
USE_NATIVE_SEARCH = False  # Ollama native search is free-tier but requires a phone-verified account with no privacy guarantees; DDGS chosen for privacy
SEARCH_TIMEOUT = 30

# Display settings
VERBOSE_DEFAULT = False  # Set to True to always show search details
ANSWER_PREFIX = "🤖 "
SEARCH_PREFIX = "🔍 "
ERROR_PREFIX = "❌ "

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# RAG settings (Phase 3)
EMBED_MODEL = "nomic-embed-text"                          # pulled via: ollama pull nomic-embed-text
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # downloaded on first use (~100 MB)
RAG_CHUNK_SIZE = 400        # words per chunk
RAG_CHUNK_OVERLAP = 40      # word overlap between adjacent chunks
RAG_RETRIEVE_K = 10         # candidates retrieved before reranking
RAG_RERANK_TOP_K = 4        # chunks injected into context after reranking
RAG_SCORE_THRESHOLD = 0.0   # CrossEncoder scores below this are dropped (negatives = irrelevant)
RAG_MAX_CHUNKS = 10_000     # warn user to unload documents above this total

# Context window
CONTEXT_WINDOW = 65536  # 64k tokens — configured context for gemma4:26b

# Conversation persistence
DB_PATH = Path(__file__).parent / "conversations.db"
MAX_CONVERSATIONS = 100    # oldest evicted when exceeded
COMPRESS_THRESHOLD = 70    # context_pct % at which summarize-and-compress fires
COMPRESS_KEEP_RECENT = 6   # number of recent messages kept verbatim (not summarized)