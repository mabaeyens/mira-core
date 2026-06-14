"""Unit tests for the per-conversation SessionManager pool (Phase 3).

The orchestrator is mocked, so these are fast and don't load models. Async methods
are driven via asyncio.run (no pytest-asyncio dependency)."""
import asyncio

from unittest.mock import patch

from core import session_manager


class _FakeOrch:
    def __init__(self, verbose=False):
        # Differ from config defaults so _build() exercises reinitialize_client.
        self.backend = "fake"
        self.model = "fake-model"
        self.context_window = 1
        self.loaded = None
        self.fresh = None
        self.reinit = None
        self.reset_calls = 0

    def new_conversation(self, conv_id, project=None):
        self.fresh = (conv_id, project)

    def load_conversation(self, conv_id, project=None):
        self.loaded = (conv_id, project)

    def reinitialize_client(self, backend, model, host, context_window):
        self.reinit = (backend, model, host, context_window)

    def reset_conversation(self):
        self.reset_calls += 1


def _mgr(max_sessions=8):
    return session_manager.SessionManager(max_sessions=max_sessions)


def test_acquire_creates_then_reuses_same_session_and_lock():
    with patch.object(session_manager, "ChatOrchestrator", _FakeOrch):
        m = _mgr()

        async def go():
            o1, l1 = await m.acquire("c1", fresh=True)
            o2, l2 = await m.acquire("c1")          # cached → reuse, ignore fresh
            return o1, l1, o2, l2

        o1, l1, o2, l2 = asyncio.run(go())
        assert o1 is o2, "same conversation must reuse its orchestrator"
        assert l1 is l2, "same conversation must reuse its lock"
        assert o1.fresh == ("c1", None)
        assert o1.loaded is None


def test_distinct_conversations_are_isolated():
    with patch.object(session_manager, "ChatOrchestrator", _FakeOrch):
        m = _mgr()

        async def go():
            oa, la = await m.acquire("a", fresh=True)
            ob, lb = await m.acquire("b", fresh=False)
            return oa, la, ob, lb

        oa, la, ob, lb = asyncio.run(go())
        assert oa is not ob, "different conversations get different orchestrators"
        assert la is not lb, "different conversations get different locks"
        assert ob.loaded == ("b", None), "fresh=False loads from db"


def test_lru_eviction_resets_oldest():
    with patch.object(session_manager, "ChatOrchestrator", _FakeOrch):
        m = _mgr(max_sessions=2)

        async def go():
            await m.acquire("a", fresh=True)
            await m.acquire("b", fresh=True)
            evicted = m.get("a")
            await m.acquire("c", fresh=True)       # over cap → evict LRU ("a")
            return evicted

        evicted = asyncio.run(go())
        assert m.get("a") is None, "oldest session evicted"
        assert m.get("b") is not None and m.get("c") is not None
        assert evicted.reset_calls == 1, "evicted session cleaned up"


def test_reinitialize_all_applies_to_all_and_future():
    with patch.object(session_manager, "ChatOrchestrator", _FakeOrch):
        m = _mgr()

        async def go():
            oa, _ = await m.acquire("a", fresh=True)
            ob, _ = await m.acquire("b", fresh=True)
            await m.reinitialize_all("omlx", "new-model", "http://h", 4096)
            oc, _ = await m.acquire("c", fresh=True)   # created after switch
            return oa, ob, oc

        oa, ob, oc = asyncio.run(go())
        assert oa.reinit == ("omlx", "new-model", "http://h", 4096)
        assert ob.reinit == ("omlx", "new-model", "http://h", 4096)
        assert m.model == "new-model"
        # New session built after the switch inherits the preset via _build().
        assert oc.reinit == ("omlx", "new-model", "http://h", 4096)


class _HistOrch(_FakeOrch):
    """Fake orchestrator that carries a per-instance history list."""
    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.history = []


def test_per_conversation_locks_serialize_same_conv_and_isolate_across():
    """Concurrent turns on the same conversation serialize through its lock (no
    interleaving / history loss); turns on different conversations run independently."""
    with patch.object(session_manager, "ChatOrchestrator", _HistOrch):
        m = _mgr()

        async def turn(conv_id, tag, hold):
            orch, lock = await m.acquire(conv_id, fresh=True)
            async with lock:
                start = len(orch.history)
                await asyncio.sleep(hold)        # simulate streaming the turn
                orch.history.append(tag)
                # If a same-conversation turn had interleaved, this would fail.
                assert len(orch.history) == start + 1

        async def go():
            await asyncio.gather(
                turn("a", "a1", 0.02),           # holds conv-a's lock while "sleeping"
                turn("a", "a2", 0.0),            # same conv → must wait for a1
                turn("b", "b1", 0.0),            # different conv → runs concurrently
            )
            return m.get("a").history, m.get("b").history

        hist_a, hist_b = asyncio.run(go())
        assert sorted(hist_a) == ["a1", "a2"], "both same-conv turns recorded, none lost"
        assert hist_b == ["b1"], "other conversation's history stays independent"


def test_remove_drops_session():
    with patch.object(session_manager, "ChatOrchestrator", _FakeOrch):
        m = _mgr()

        async def go():
            o, _ = await m.acquire("a", fresh=True)
            await m.remove("a")
            return o

        o = asyncio.run(go())
        assert m.get("a") is None
        assert o.reset_calls == 1
