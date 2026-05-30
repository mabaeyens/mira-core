# Spec: Drop Ollama — Local sentence-transformers Embeddings

## Status: DONE (implemented 2026-05-30, commit b92d062)

---

## Problem

Ollama is used only for `nomic-embed-text` RAG embeddings (`EMBED_BACKEND = "ollama"` in
`core/config.py`). mlx-lm handles all inference. Every session that uses RAG requires Ollama
to be running separately — if it's down, embeddings silently fail (ChromaDB gets zeros or errors).

`sentence_transformers` is already installed (it powers the CrossEncoder reranker). The same
`nomic-embed-text-v1.5` model is available on HuggingFace and produces identical vectors to the
Ollama version. Switching eliminates the last Ollama runtime dependency.

---

## Approach

Replace `ollama.Client().embed()` in `core/rag_engine.py` with a `SentenceTransformer` model
loaded once at startup. No new dependencies — `sentence_transformers` is already in
`pyproject.toml`.

### Key changes

**`core/config.py`**
```python
# Add alongside RERANK_MODEL:
EMBED_MODEL: str = _get("embed_model", "nomic-ai/nomic-embed-text-v1.5")
```
Remove `EMBED_BACKEND` and `EMBED_HOST` (no longer needed when ST is the only backend).
Keep `EMBED_MODEL` so users can override to a different HF model via `mira.yaml`.

**`core/rag_engine.py`**
```python
from sentence_transformers import SentenceTransformer

class RagEngine:
    def __init__(self, ...):
        self._embedder = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
        # trust_remote_code=True required for nomic-embed-text-v1.5 pooling config
        ...

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.encode(texts, batch_size=64).tolist()
```

Remove: `import ollama`, `EMBED_BACKEND` branch logic, `reinitialize_client()` (or simplify
to just update the model path if needed), `_ollama` and `_oai_embed` attributes.

**`core/config.py`**
Remove: `EMBED_BACKEND`, `EMBED_HOST`, `OLLAMA_HOST` (if only used for embed routing).
Keep: `OLLAMA_HOST` if it appears elsewhere (check `grep -rn OLLAMA_HOST core/`).

**`server.py`**
Remove: any Ollama startup check that gates embedding readiness. The ST model loads from
`~/.cache/huggingface/` on first use and is always available.

**`README.md`**
Remove `ollama pull nomic-embed-text` from Prerequisites. Update mira.yaml example to drop
`embed_backend` / `embed_host`. Keep Ollama in Prerequisites only if still used elsewhere
(check `grep -rn ollama core/` after removal).

---

## Constraints

- `nomic-embed-text-v1.5` requires `trust_remote_code=True` (custom pooling layer) — safe for
  local use, same model already trusted via Ollama
- First run downloads ~270 MB from HuggingFace; cached to `~/.cache/huggingface/`
- Vector dimensions are identical (768) — existing ChromaDB collections remain valid; no
  re-indexing needed for current data
- `mira.yaml` `embed_model` key should remain supported so users can swap to a different ST model

---

## Verification

1. `uv run python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True); print(m.encode(['hello']).shape)"` → `(1, 768)`
2. Start server with Ollama stopped — `/health` returns `backend_ready: true`; attach a PDF → RAG indexes without error
3. Ask a question over the indexed doc → model retrieves correct chunks
4. Existing ChromaDB collections (in `~/.local/share/mira/chroma_db/`) continue to work
5. `grep -rn "import ollama" core/` → no results
