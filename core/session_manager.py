"""Per-conversation orchestrator pool (Phase 3).

Replaces the single global ``ChatOrchestrator`` so concurrent requests on different
conversations no longer share mutable state (history, token counters, attachments,
temp workspace). Each conversation gets its own orchestrator — cheap now that the RAG
models are process-wide shared (see ``rag_engine``). Turns on the *same* conversation
serialize through a per-conversation ``asyncio.Lock``; different conversations run
concurrently.

Locks are created lazily in the running event loop, which also sidesteps the
"asyncio primitive bound to a different event loop" test-isolation problem the old
module-level ``_orch_lock`` had.
"""

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .config import BACKEND, CONTEXT_WINDOW, MODEL_NAME, OLLAMA_HOST, VERBOSE_DEFAULT
from .orchestrator import ChatOrchestrator

logger = logging.getLogger(__name__)


class SessionManager:
    """Bounded LRU pool of per-conversation orchestrators with per-conversation locks."""

    def __init__(self, max_sessions: int = 8, verbose: bool = VERBOSE_DEFAULT):
        self._sessions: "OrderedDict[str, ChatOrchestrator]" = OrderedDict()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()          # protects _sessions / _locks mutation
        self._max = max_sessions
        self._verbose = verbose
        # Current runtime backend preset; updated on a backend/model switch so newly
        # created sessions inherit it (config provides the initial values).
        self._preset = {
            "backend": BACKEND,
            "model": MODEL_NAME,
            "host": OLLAMA_HOST,
            "context_window": CONTEXT_WINDOW,
        }
        # A single conversation-less orchestrator for stateless one-shot LLM calls
        # (e.g. the /ask endpoint). Built lazily, kept in sync with the preset.
        self._scratch: Optional[ChatOrchestrator] = None
        self._scratch_lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._preset["model"]

    @property
    def backend(self) -> str:
        return self._preset["backend"]

    @property
    def verbose(self) -> bool:
        return self._verbose

    def set_verbose(self, enabled: bool) -> None:
        """Apply a verbose toggle to every live session, the scratch orchestrator,
        and future ones."""
        self._verbose = enabled
        for orch in self._sessions.values():
            orch.verbose = enabled
        if self._scratch is not None:
            self._scratch.verbose = enabled

    # ── internals ─────────────────────────────────────────────────────────────

    def _lock_for(self, conv_id: str) -> asyncio.Lock:
        lock = self._locks.get(conv_id)
        if lock is None:
            lock = asyncio.Lock()             # created in the running loop
            self._locks[conv_id] = lock
        return lock

    def _build(self) -> ChatOrchestrator:
        orch = ChatOrchestrator(verbose=self._verbose)
        p = self._preset
        # Apply the current runtime preset if it differs from the config defaults the
        # orchestrator was constructed with (i.e. a backend switch happened).
        if (orch.backend, orch.model, orch.context_window) != (
            p["backend"], p["model"], p["context_window"]
        ):
            orch.reinitialize_client(p["backend"], p["model"], p["host"], p["context_window"])
        return orch

    def _evict_if_needed(self) -> None:
        while len(self._sessions) > self._max:
            old_id, old_orch = self._sessions.popitem(last=False)
            try:
                old_orch.reset_conversation()   # frees the temp attachment workspace
            except Exception as e:              # pragma: no cover - best effort
                logger.debug("evict cleanup failed for %s: %s", old_id, e)
            self._locks.pop(old_id, None)
            logger.info("Evicted LRU session %s", old_id)

    # ── public API ──────────────────────────────────────────────────────────────

    async def acquire(
        self, conv_id: str, project: Optional[Dict] = None, fresh: bool = False
    ) -> Tuple[ChatOrchestrator, asyncio.Lock]:
        """Return ``(orchestrator, lock)`` for ``conv_id``, creating the session on
        first use. ``fresh=True`` starts a new (empty) conversation; otherwise the
        history is loaded from the DB. A cached session is reused as-is regardless of
        ``fresh``. The caller must run the turn inside ``async with lock:``."""
        async with self._guard:
            orch = self._sessions.get(conv_id)
            if orch is None:
                orch = self._build()
                if fresh:
                    orch.new_conversation(conv_id, project=project)
                else:
                    orch.load_conversation(conv_id, project=project)
                self._sessions[conv_id] = orch
                self._evict_if_needed()
            self._sessions.move_to_end(conv_id)   # mark most-recently-used
            lock = self._lock_for(conv_id)
        return orch, lock

    def get(self, conv_id: str) -> Optional[ChatOrchestrator]:
        """Return the cached orchestrator for ``conv_id`` or ``None`` (no creation)."""
        return self._sessions.get(conv_id)

    async def reinitialize_all(
        self, backend: str, model: str, host: str, context_window: int
    ) -> None:
        """Switch backend/model for every live session and for future ones."""
        async with self._guard:
            self._preset = {
                "backend": backend, "model": model,
                "host": host, "context_window": context_window,
            }
            for orch in self._sessions.values():
                orch.reinitialize_client(backend, model, host, context_window)
            if self._scratch is not None:
                self._scratch.reinitialize_client(backend, model, host, context_window)

    def llm_chat_sync(self, messages: List[Dict], format: Optional[dict] = None) -> str:
        """One-shot, conversation-less LLM call (for /ask). Uses a shared scratch
        orchestrator built from the current preset. Synchronous — call from an
        executor thread, not the event loop."""
        with self._scratch_lock:
            if self._scratch is None:
                self._scratch = self._build()
                self._scratch.verbose = self._verbose
            scratch = self._scratch
        return scratch._llm_chat_sync(messages, format=format)

    async def remove(self, conv_id: str) -> None:
        """Drop a session (e.g. when its conversation is deleted)."""
        async with self._guard:
            orch = self._sessions.pop(conv_id, None)
            self._locks.pop(conv_id, None)
            if orch is not None:
                try:
                    orch.reset_conversation()
                except Exception as e:           # pragma: no cover - best effort
                    logger.debug("remove cleanup failed for %s: %s", conv_id, e)
