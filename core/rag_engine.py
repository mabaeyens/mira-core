"""
RAG engine for Mira.

Pipeline: chunk → embed → store (ChromaDB) → retrieve → rerank (CrossEncoder)

Embedding backend: Ollama (ollama.embed) or oMLX/OpenAI-compatible (/v1/embeddings).
Configured via EMBED_BACKEND in config / mira.yaml.

Key design decisions:
- Per-project PersistentClient: indexed docs survive server restarts, scoped to project
- No-project sessions use EphemeralClient (fully in-memory, no persistence issues)
- Manual metadata filtering for remove(): where-clause API is unreliable in ChromaDB 1.x
- clear() recreates the client — collection.delete() alone does not release RAM reliably
- CrossEncoder is loaded lazily but pre-warmed in a background thread at startup
- Batch embedding: 64 texts per call for efficiency
"""

import logging
import threading
import uuid
from typing import List, Dict, Optional

import chromadb
import ollama
import openai as _openai

from .config import (
    OLLAMA_HOST, EMBED_BACKEND, EMBED_HOST,
    EMBED_MODEL, RERANK_MODEL,
    RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP,
    RAG_RETRIEVE_K, RAG_RERANK_TOP_K,
    RAG_SCORE_THRESHOLD, RAG_MAX_CHUNKS,
    RAG_DIR,
)

logger = logging.getLogger(__name__)

# nomic-embed-text context is 8192 tokens.  Normal 400-word chunks are ~2 400 chars
# and never come close.  This cap only fires for pathological content (e.g. binary
# files decoded as text) where a single "word" can be thousands of bytes long.
_EMBED_CHAR_LIMIT = 4096


class RagEngine:
    """RAG index for Mira. Per-project sessions use a PersistentClient; no-project sessions use EphemeralClient."""

    def __init__(self):
        if EMBED_BACKEND == "ollama":
            self._ollama = ollama.Client(host=EMBED_HOST)
            self._oai_embed = None
        else:
            self._ollama = None
            base = EMBED_HOST.rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"
            from .orchestrator import _read_omlx_api_key
            self._oai_embed = _openai.OpenAI(base_url=base, api_key=_read_omlx_api_key())
        self._reranker = None
        self._reranker_lock = threading.Lock()
        self._init_db()
        # Pre-warm the reranker in the background so the first query isn't slow
        threading.Thread(target=self._load_reranker, daemon=True).start()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _init_db(self, project_id: Optional[str] = None) -> None:
        self._project_id = project_id
        if project_id is not None:
            project_dir = RAG_DIR / project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(project_dir))
            self._col_name = "docs"
            self._collection = self._client.get_or_create_collection(
                name=self._col_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._persistent = True
        else:
            self._client = chromadb.EphemeralClient()
            # Unique name per instance: ChromaDB 1.x uses a shared Rust backend per
            # process, so two EphemeralClient instances would collide on the same name.
            self._col_name = f"docs_{uuid.uuid4().hex[:8]}"
            self._collection = self._client.create_collection(
                name=self._col_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._persistent = False

    def reinitialize_client(self, embed_backend: str, embed_host: str) -> None:
        """Switch embedding backend at runtime."""
        if embed_backend == "ollama":
            self._ollama = ollama.Client(host=embed_host)
            self._oai_embed = None
        else:
            self._ollama = None
            base = embed_host.rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"
            from .orchestrator import _read_omlx_api_key
            self._oai_embed = _openai.OpenAI(base_url=base, api_key=_read_omlx_api_key())

    def clear(self) -> None:
        """Wipe all indexed documents. For ephemeral: recreates in-memory store. For persistent: deletes and recreates the on-disk collection."""
        if self._persistent:
            self._client.delete_collection(self._col_name)
            self._collection = self._client.create_collection(
                name=self._col_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            del self._collection
            del self._client
            self._init_db()

    def load_project(self, project_id: Optional[str] = None) -> None:
        """Switch to a project's persistent ChromaDB store, or back to ephemeral if None."""
        del self._collection
        del self._client
        self._init_db(project_id)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> List[str]:
        """Return sorted list of indexed document names."""
        if self.chunk_count == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        return sorted({m["source"] for m in result["metadatas"]})

    def index(self, name: str, text: str) -> int:
        """
        Chunk, embed, and store a document.
        If a document with the same name already exists it is replaced.
        Returns the number of chunks stored.
        """
        self.remove(name)  # idempotent replace

        chunks = self._chunk(text)
        if not chunks:
            return 0

        BATCH = 64  # nomic-embed-text handles large batches efficiently
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            texts = [c["text"] for c in batch]
            embeddings = self._embed(texts)
            self._collection.add(
                ids=[f"{name}__c{i + j}" for j in range(len(batch))],
                documents=texts,
                embeddings=embeddings,
                metadatas=[{"source": name, "chunk_index": c["idx"]} for c in batch],
            )
            logger.debug(f"Indexed batch {i // BATCH + 1} for '{name}' ({len(batch)} chunks)")

        total = len(chunks)
        logger.info(f"Indexed '{name}': {total} chunks, collection total: {self.chunk_count}")
        return total

    def query(self, question: str, score_threshold: float = RAG_SCORE_THRESHOLD) -> List[Dict]:
        """
        Retrieve and rerank chunks relevant to the question.
        Returns list of {"text", "source", "score"} sorted by descending score.
        Chunks with CrossEncoder score <= score_threshold are dropped.
        Pass score_threshold=float('-inf') to return top-K regardless of score.
        """
        if self.chunk_count == 0:
            return []

        k = min(RAG_RETRIEVE_K, self.chunk_count)
        q_emb = self._embed([question])[0]

        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            return []

        reranker = self._get_reranker()
        scores = reranker.predict([[question, d] for d in docs])

        ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)

        return [
            {"text": doc, "source": meta["source"], "score": float(score)}
            for doc, meta, score in ranked[:RAG_RERANK_TOP_K]
            if score > score_threshold
        ]

    def remove(self, name: str) -> None:
        """Remove all chunks for a given document."""
        if self.chunk_count == 0:
            return
        try:
            result = self._collection.get(include=["metadatas"])
            ids = [id_ for id_, m in zip(result["ids"], result["metadatas"]) if m["source"] == name]
            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"Removed '{name}' ({len(ids)} chunks)")
        except Exception as e:
            logger.warning(f"remove('{name}'): {e}")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _chunk(self, text: str) -> List[Dict]:
        """Split text into overlapping word-count-based chunks."""
        words = text.split()
        chunks, start, idx = [], 0, 0
        while start < len(words):
            end = min(start + RAG_CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])
            # Hard cap: protects the embedding model from binary garbage where
            # "words" are raw bytes (no spaces) and a 400-word chunk can be
            # orders of magnitude larger than the model's token limit.
            if len(chunk_text) > _EMBED_CHAR_LIMIT:
                chunk_text = chunk_text[:_EMBED_CHAR_LIMIT]
            chunks.append({"text": chunk_text, "idx": idx})
            if end == len(words):
                break
            start += RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP
            idx += 1
        return chunks

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed texts using the configured embedding backend."""
        if EMBED_BACKEND == "ollama":
            response = self._ollama.embed(
                model=EMBED_MODEL,
                input=texts,
                options={"num_ctx": 2048},
            )
            return response.embeddings
        else:
            response = self._oai_embed.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]

    def _load_reranker(self):
        """Load the CrossEncoder model (thread-safe, called once)."""
        with self._reranker_lock:
            if self._reranker is None:
                import contextlib, io, os
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading reranker: {RERANK_MODEL}")
                # Try loading from local cache silently (model already downloaded).
                # If cache is missing (fresh install), fall back to online download.
                orig = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        self._reranker = CrossEncoder(RERANK_MODEL)
                except Exception:
                    # Not cached yet — download it (output visible so user sees progress)
                    if orig is None:
                        del os.environ["HF_HUB_OFFLINE"]
                    else:
                        os.environ["HF_HUB_OFFLINE"] = orig
                    logger.info(f"Downloading reranker model (first run): {RERANK_MODEL}")
                    self._reranker = CrossEncoder(RERANK_MODEL)
                logger.info("Reranker ready")

    def _get_reranker(self):
        if self._reranker is None:
            self._load_reranker()
        return self._reranker
