"""Pre-download the RAG embedding + reranker models, with retries.

Used by CI to warm the HuggingFace cache in one isolated, retryable step instead of
letting a transient HF outage fail mid-test-run (which previously produced 30+ minute
hangs and cascading errors). Once the cache is populated, subsequent runs are offline.
"""
import sys
import time

from sentence_transformers import CrossEncoder, SentenceTransformer

from core.config import EMBED_MODEL, RERANK_MODEL


def _fetch() -> None:
    SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    CrossEncoder(RERANK_MODEL)


def main() -> int:
    for attempt in range(1, 4):
        try:
            _fetch()
            print(f"models cached on attempt {attempt}: {EMBED_MODEL}, {RERANK_MODEL}")
            return 0
        except Exception as e:  # noqa: BLE001 — any HF/network error is retryable
            print(f"prefetch attempt {attempt}/3 failed: {e}", file=sys.stderr)
            if attempt < 3:
                time.sleep(10)
    print("model prefetch failed after 3 attempts", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
