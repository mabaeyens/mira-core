"""Caching the state at the history boundary, on the way past.

Why it exists: Qwen3's generation prompt ends with a thinking scaffold the chat
template never re-emits when it replays that assistant turn, so turn N's prompt
is not a prefix of turn N+1's — and this model's cache cannot be trimmed back to
the divergence, because its linear-attention layers hold recurrent state. The
consequence, measured on bench Q10 (2026-08-08): a 27,614-token second turn
reused 0 tokens and took 48.7s.

The history *without* the scaffold is a prefix of every later turn. Prefill
passes through that state; it just never stopped to keep it. These tests cover
the stopping and the keeping, plus the ways both must fail safe — a lost
snapshot costs speed, but a snapshot keyed on tokens that were never processed
would hand back a mismatched cache and silently change output.
"""
import pytest

pytest.importorskip("mlx.core")  # mlx is macOS-only (Apple Silicon), absent on Linux CI

from types import SimpleNamespace  # noqa: E402

from core.inference.mira_mlx_server import (  # noqa: E402
    SNAPSHOT_CACHE_TYPE,
    SNAPSHOT_MIN_BOUNDARY_TOKENS,
    GenerationEngine,
    plan_prefill_segments,
)


# ── where prefill splits ──────────────────────────────────────────────────────

def test_prefill_splits_at_the_boundary():
    rest = list(range(100))
    segments, to_cache = plan_prefill_segments(rest, prompt_cache_count=0, boundaries=[60])
    assert segments == [rest[:60], rest[60:]]
    assert to_cache == [60]


def test_the_split_is_offset_by_what_the_cache_already_covered():
    """A boundary indexes the whole prompt; rest starts after the reused prefix.
    Splitting at the boundary directly would cut 40 tokens too late and cache a
    sequence that was never the boundary."""
    rest = list(range(100))
    segments, to_cache = plan_prefill_segments(rest, prompt_cache_count=40, boundaries=[60])
    assert segments == [rest[:20], rest[20:]]
    assert to_cache == [60]


def test_a_boundary_already_covered_by_the_cache_needs_no_split():
    segments, to_cache = plan_prefill_segments(list(range(100)), 60, [60])
    assert to_cache == []
    assert len(segments) == 1


def test_a_boundary_at_the_very_end_needs_no_split():
    """A trailing segment would be empty, and there would be nothing to generate
    from."""
    segments, to_cache = plan_prefill_segments(list(range(100)), 0, [100])
    assert to_cache == []
    assert len(segments) == 1


def test_no_boundary_means_the_prompt_is_prefilled_as_before():
    rest = list(range(10))
    assert plan_prefill_segments(rest, 0, []) == ([rest], [])
    assert plan_prefill_segments(rest, 0, [None]) == ([rest], [])
    assert plan_prefill_segments(rest, 0, None) == ([rest], [])


def test_two_boundaries_make_three_segments():
    """The system boundary and the history boundary both land in the same
    prompt on a continuation, and each needs its own segment to be snapshotted
    at."""
    rest = list(range(100))
    segments, to_cache = plan_prefill_segments(rest, 0, [30, 70])
    assert segments == [rest[:30], rest[30:70], rest[70:]]
    assert to_cache == [30, 70]


def test_boundaries_are_sorted_before_splitting():
    """The finders run in their own order, not in prompt order. Splitting in the
    order given would produce negative-length slices."""
    rest = list(range(100))
    assert plan_prefill_segments(rest, 0, [70, 30])[1] == [30, 70]


def test_a_repeated_boundary_yields_one_segment_not_an_empty_one():
    """A conversation whose history is only the system prompt gives both finders
    the same index. Two cuts at the same point would put a zero-length segment
    in the middle, which prefills nothing and would pop an end_of_segment the
    queue is not expecting."""
    rest = list(range(100))
    segments, to_cache = plan_prefill_segments(rest, 0, [40, 40])
    assert segments == [rest[:40], rest[40:]]
    assert to_cache == [40]
    assert all(len(s) > 0 for s in segments)


def test_the_segments_always_reconstruct_the_prompt():
    """Any off-by-one here silently drops or duplicates a token, which would
    corrupt generation rather than merely slow it."""
    rest = list(range(1000))
    cases = (
        (0, [1]), (0, [999]), (250, [500]), (999, [1000]),
        (0, [1, 999]), (0, [250, 500, 750]), (100, [50, 300, 1500]),
        (0, [400, 400]),
    )
    for count, boundaries in cases:
        segments, _ = plan_prefill_segments(rest, count, boundaries)
        assert [t for s in segments for t in s] == rest, (count, boundaries)
        assert all(len(s) > 0 for s in segments), (count, boundaries)


class _Tok:
    """Renders the way a real Qwen3 template does, including its refusals.

    The important behaviour reproduced here is that a message list with no user
    turn raises. Qwen3's template does exactly that ("No user query found in
    messages"), which is what made the first version of _system_boundary return
    None on every single request.
    """

    SCAFFOLD = [901, 902, 903]
    # Must match the fillers _system_boundary probes with.
    PROBE_FILLERS = ("zzz alpha", "qqq beta")

    def __init__(self, history_ids=None, prefix_ok=True, boom=False,
                 system_ids=None, refuse_userless=True):
        self.history_ids = history_ids if history_ids is not None else list(range(1, 2001))
        # The system block the two probe renders agree on.
        self.system_ids = system_ids
        self.prefix_ok = prefix_ok
        self.boom = boom
        self.refuse_userless = refuse_userless

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=True, **kw):
        if self.boom:
            raise RuntimeError("template exploded")
        roles = [m.get("role") for m in messages]
        if roles and all(r == "system" for r in roles) and self.refuse_userless:
            raise ValueError("No user query found in messages")
        if (self.system_ids is not None and roles and roles[-1] == "user"
                and messages[-1]["content"] in self.PROBE_FILLERS):
            # A probe render: shared system block, then filler-specific tokens.
            return f"probe:{messages[-1]['content']}"
        return "history-text"

    def encode(self, text, add_special_tokens=False):
        if text.startswith("probe:"):
            # Same system block for both probes, then filler-dependent tail.
            tail = [7000 + (ord(c) % 97) for c in text[6:9]]
            ids = list(self.system_ids) + tail
            return ids if self.prefix_ok else [-1] + ids[1:]
        if self.prefix_ok:
            return list(self.history_ids)
        return [-1] + list(self.history_ids)[1:]

    def full(self):
        return list(self.history_ids) + self.SCAFFOLD


@pytest.fixture
def engine():
    e = GenerationEngine(model_path="fake/model", boundary_snapshot=True)
    e.tokenizer = _Tok()
    return e


# ── finding the boundary ──────────────────────────────────────────────────────

def test_it_finds_the_boundary_before_the_scaffold(engine):
    tok = engine.tokenizer
    n = engine._history_boundary([], None, {}, tok.full())
    assert n == len(tok.history_ids)


def test_a_boundary_that_is_not_a_prefix_is_refused(engine):
    """The dangerous case. A template that renders history differently with and
    without add_generation_prompt would produce an entry keyed on tokens that
    were never processed, which changes output rather than just slowing it."""
    engine.tokenizer = _Tok(prefix_ok=False)
    assert engine._history_boundary([], None, {}, engine.tokenizer.full()) is None


def test_a_template_failure_is_not_fatal(engine):
    engine.tokenizer = _Tok(boom=True)
    assert engine._history_boundary([], None, {}, engine.tokenizer.full()) is None


def test_a_boundary_covering_the_whole_prompt_is_refused(engine):
    """Nothing left to prefill after it, so there is no second segment and
    nothing to snapshot."""
    tok = engine.tokenizer
    assert engine._history_boundary([], None, {}, list(tok.history_ids)) is None


# ── the system boundary ───────────────────────────────────────────────────────

SYS = {"role": "system", "content": "s"}
USER = {"role": "user", "content": "u"}


def test_the_system_boundary_survives_a_template_that_refuses_userless_renders(engine):
    """THE regression this file exists for now. Qwen3's template raises
    "No user query found in messages" for a system-only list, so rendering the
    system messages directly returns None on every request and the entry is
    never created. Measured live 2026-08-11 before the two-probe fix landed."""
    engine.tokenizer = _Tok(system_ids=list(range(1, 1501)), refuse_userless=True)
    tok = engine.tokenizer
    n = engine._system_boundary([SYS, USER], None, {}, tok.full())
    assert n == 1500, "system boundary lost to the template's userless refusal"


def test_the_system_boundary_sits_ahead_of_the_history_one(engine):
    """The point of this entry: it is a prefix of the FIRST turn of every
    conversation, so it sits strictly before the history boundary."""
    engine.tokenizer = _Tok(system_ids=list(range(1, 1501)))
    tok = engine.tokenizer
    assert (engine._system_boundary([SYS, USER], None, {}, tok.full())
            < engine._history_boundary([SYS, USER], None, {}, tok.full()))


def test_the_boundary_does_not_depend_on_what_the_user_asked(engine):
    """Two probes with different fillers agree only on the shared block. If the
    boundary moved with the real user message, the entry would stop being
    shareable between conversations, which is the entire point of it."""
    engine.tokenizer = _Tok(system_ids=list(range(1, 1501)))
    tok = engine.tokenizer
    a = engine._system_boundary([SYS, {"role": "user", "content": "aardvark"}],
                                None, {}, tok.full())
    engine._system_probe_cache.clear()
    b = engine._system_boundary([SYS, {"role": "user", "content": "zeppelin"}],
                                None, {}, tok.full())
    assert a == b == 1500


def test_no_system_message_means_no_system_boundary(engine):
    engine.tokenizer = _Tok(system_ids=list(range(1, 1501)))
    assert engine._system_boundary([USER], None, {}, engine.tokenizer.full()) is None


def test_only_the_leading_system_messages_count(engine):
    """A system message appearing later in the conversation is not a prefix of
    anything, so the scan stops at the first non-system role."""
    engine.tokenizer = _Tok(system_ids=list(range(1, 1501)))
    captured = []
    real = engine.tokenizer.apply_chat_template

    def spy(messages, **kw):
        captured.append([m.get("role") for m in messages])
        return real(messages, **kw)

    engine.tokenizer.apply_chat_template = spy
    engine._system_boundary([SYS, SYS, USER, SYS], None, {}, engine.tokenizer.full())
    # Two probe renders, each of the two leading system messages plus a filler.
    assert captured == [["system", "system", "user"], ["system", "system", "user"]]


def test_a_system_boundary_that_is_not_a_prefix_is_refused(engine):
    """Same safety argument as the history boundary: a probe that does not match
    the prompt would key an entry on tokens that were never processed."""
    engine.tokenizer = _Tok(system_ids=list(range(1, 1501)), prefix_ok=False)
    assert engine._system_boundary([SYS, USER], None, {}, engine.tokenizer.full()) is None


def test_a_system_template_failure_is_not_fatal(engine):
    engine.tokenizer = _Tok(system_ids=list(range(1, 1501)), boom=True)
    assert engine._system_boundary([SYS, USER], None, {}, engine.tokenizer.full()) is None


def test_the_probe_is_computed_once_and_memoised(engine):
    """Two ~3,600-token renders and encodes on every request is real latency for
    a value that changes only when the system prompt does."""
    engine.tokenizer = _Tok(system_ids=list(range(1, 1501)))
    calls = []
    real = engine.tokenizer.apply_chat_template
    engine.tokenizer.apply_chat_template = (
        lambda messages, **kw: calls.append(1) or real(messages, **kw)
    )
    for _ in range(4):
        engine._system_boundary([SYS, USER], None, {}, engine.tokenizer.full())
    assert len(calls) == 2, "probe should render twice in total, not twice per request"


# ── taking the snapshot ───────────────────────────────────────────────────────

class _Batch:
    def __init__(self, uids, cache=None, boom=False):
        self.uids = list(uids)
        self._cache = cache if cache is not None else [SimpleNamespace(nbytes=64)]
        self.boom = boom
        self.extracted = []

    def extract_cache(self, idx):
        if self.boom:
            raise RuntimeError("extract failed")
        self.extracted.append(idx)
        return self._cache


class _PromptCache:
    def __init__(self):
        self.inserted = []

    def insert_cache(self, model, tokens, cache, *, cache_type="assistant"):
        # cache_type mirrors the real LRUPromptCache signature, default and all.
        # A mock that accepted **kwargs would swallow a misspelled keyword and
        # report priority working when nothing had been tagged.
        self.inserted.append((model, list(tokens), cache, cache_type))


def _armed(engine, uid=7, tokens=(1, 2, 3), queue=None):
    engine.prompt_cache = _PromptCache()
    engine.batch_generator = SimpleNamespace(_prompt_batch=_Batch([uid]))
    if queue is None:
        queue = [("history", list(tokens))]
    engine._pending = {uid: {"snapshot_queue": queue}}
    return uid


def _resp(uid, end_of_segment=True):
    return SimpleNamespace(uid=uid, end_of_segment=end_of_segment,
                           end_of_prompt=False, progress=(0, 0))


def test_it_caches_the_boundary_state(engine):
    uid = _armed(engine, tokens=(11, 22, 33))
    engine._maybe_snapshot_boundary(_resp(uid))

    assert len(engine.prompt_cache.inserted) == 1
    model, tokens, _, _ = engine.prompt_cache.inserted[0]
    assert model == "fake/model"
    assert tokens == [11, 22, 33], "entry keyed on something other than the boundary"
    assert engine._snapshots_taken == 1


def test_the_queue_drains_once_per_segment_and_then_stops(engine):
    """mlx-lm raises end_of_segment as each segment drains and once more when the
    sequence moves to generation, so events always outnumber boundaries by one.
    The extra event must find an empty queue rather than cache the prompt minus
    its last token."""
    uid = _armed(engine)
    for _ in range(3):
        engine._maybe_snapshot_boundary(_resp(uid))
    assert len(engine.prompt_cache.inserted) == 1
    assert engine._snapshots_taken == 1


def test_two_boundaries_are_snapshotted_in_prefill_order(engine):
    """The queue is paired with the segments by position, so an out-of-order pop
    would key the system entry on the history tokens and vice versa — a cache
    that returns the wrong state, not merely a slow one."""
    uid = _armed(engine, queue=[("system", [1, 2]), ("history", [1, 2, 3, 4])])
    engine._maybe_snapshot_boundary(_resp(uid))
    engine._maybe_snapshot_boundary(_resp(uid))
    engine._maybe_snapshot_boundary(_resp(uid))  # the generation-transition event

    assert [t for _, t, _, _ in engine.prompt_cache.inserted] == [[1, 2], [1, 2, 3, 4]]
    assert engine._snapshots_taken == 2
    assert engine._snapshots_by_kind == {"system": 1, "history": 1}


def test_each_kind_is_counted_separately(engine):
    """A change that keeps total snapshots flat while silently stopping one kind
    from firing is exactly the regression this counter exists to catch."""
    uid = _armed(engine, queue=[("system", [1, 2])])
    engine._maybe_snapshot_boundary(_resp(uid))
    assert engine._snapshots_by_kind == {"system": 1, "history": 0}


def test_a_response_that_is_not_a_segment_end_does_nothing(engine):
    uid = _armed(engine)
    engine._maybe_snapshot_boundary(_resp(uid, end_of_segment=False))
    assert engine.prompt_cache.inserted == []


def test_a_job_whose_prefill_was_not_split_is_left_alone(engine):
    """The queue is non-empty only when boundary segments actually exist.
    Without that guard the first end_of_segment of an unsplit job would cache
    the prompt minus its last token, which is not a prefix of anything."""
    uid = 7
    engine.prompt_cache = _PromptCache()
    engine.batch_generator = SimpleNamespace(_prompt_batch=_Batch([uid]))
    engine._pending = {uid: {"snapshot_queue": []}}
    engine._maybe_snapshot_boundary(_resp(uid))
    assert engine.prompt_cache.inserted == []


def test_an_unknown_uid_is_ignored(engine):
    _armed(engine, uid=7)
    engine._maybe_snapshot_boundary(_resp(999))
    assert engine.prompt_cache.inserted == []


def test_the_sequence_index_is_resolved_by_uid(engine):
    """Sequences migrate between batches as they finish prefill, so a cached
    index would extract another conversation's cache."""
    uid = 42
    engine.prompt_cache = _PromptCache()
    batch = _Batch([5, 13, uid])
    engine.batch_generator = SimpleNamespace(_prompt_batch=batch)
    engine._pending = {uid: {"snapshot_queue": [("history", [1])]}}
    engine._maybe_snapshot_boundary(_resp(uid))
    assert batch.extracted == [2]


def test_a_failed_extract_costs_the_snapshot_and_nothing_else(engine):
    uid = 7
    engine.prompt_cache = _PromptCache()
    engine.batch_generator = SimpleNamespace(_prompt_batch=_Batch([uid], boom=True))
    engine._pending = {uid: {"snapshot_queue": [("history", [1, 2])]}}

    engine._maybe_snapshot_boundary(_resp(uid))  # must not raise

    assert engine.prompt_cache.inserted == []
    assert engine._snapshot_failures == 1
    assert engine._snapshots_taken == 0


def test_it_records_what_it_cost(engine):
    uid = _armed(engine)
    engine._maybe_snapshot_boundary(_resp(uid))
    assert engine._snapshot_last_bytes == 64
    assert engine._snapshot_last_seconds >= 0


# ── the switch ────────────────────────────────────────────────────────────────

def test_disabled_by_default():
    """It changes the prefill path of every request, so it earns its default the
    way proactive decompression did: off until real use says otherwise."""
    assert GenerationEngine(model_path="fake/model").boundary_snapshot is False


def test_the_entry_ceiling_leaves_room_for_three_entries_per_turn():
    """A turn can now store the completed turn, its history snapshot AND the
    shared system snapshot. Left at mlx-lm's default of 10 this would have
    quietly cut how many conversations stay warm, which is a slowdown nothing
    would have reported."""
    from core.inference.mira_mlx_server import PROMPT_CACHE_MAX_ENTRIES
    assert PROMPT_CACHE_MAX_ENTRIES >= 30


def test_the_threshold_is_above_a_bare_system_prompt():
    """Below this the entry saves little and still costs an LRU slot; coverage
    measured at ~44% for a short exchange against ~95% for a real conversation."""
    assert SNAPSHOT_MIN_BOUNDARY_TOKENS >= 1024


# ── eviction priority ─────────────────────────────────────────────────────────
#
# The 2026-08-11 phase (a) run lost the shared system entry to a single
# memory-pressure trim and re-prefilled the system prompt for the next
# conversation. The entries are not equally valuable and plain LRU cannot tell:
# the system snapshot serves every conversation, a completed turn serves one
# request that has already finished. mlx-lm's CacheOrder already ranks classes;
# Mira just never passed the tag.


def test_the_shared_system_entry_outranks_a_per_conversation_one():
    assert SNAPSHOT_CACHE_TYPE["system"] == "system"
    assert SNAPSHOT_CACHE_TYPE["history"] == "user"


def test_each_boundary_is_tagged_with_its_eviction_class(engine):
    """Untagged, every entry lands in "assistant" and the ordering collapses to
    plain LRU -- which is the bug, and it is invisible: the cache still works,
    just loses the most valuable entry first."""
    uid = _armed(engine, queue=[("system", [1, 2]), ("history", [1, 2, 3, 4])])
    engine._maybe_snapshot_boundary(_resp(uid))
    engine._maybe_snapshot_boundary(_resp(uid))

    tags = [row[3] for row in engine.prompt_cache.inserted]
    assert tags == ["system", "user"]


def test_a_completed_turn_is_the_first_thing_evicted(engine):
    """The other insert site. It is the default, but an explicit tag is what
    keeps the three classes greppable in one place."""
    import inspect
    src = inspect.getsource(engine._handle_response)
    assert 'cache_type="assistant"' in src, (
        "the post-generation insert stopped declaring its eviction class"
    )


# The four below run against the REAL LRUPromptCache, no mock. The mechanism is
# upstream's, so what is worth testing is Mira's *use* of it plus the assumption
# that upstream still behaves this way -- these fail loudly if a future mlx-lm
# reorders the classes or changes pop().


class _StubLayer:
    """Enough surface for LRUPromptCache: it only reads .nbytes and calls
    .is_trimmable(). False is the Qwen3.6 case (hybrid cache, ArraysCache
    layers), which is what production actually runs."""

    nbytes = 1_000_000

    def __init__(self, trimmable=False):
        self._trimmable = trimmable

    def is_trimmable(self):
        return self._trimmable


def _lru(max_size=100):
    from mlx_lm.models.cache import LRUPromptCache
    return LRUPromptCache(max_size=max_size)


def _present(cache, tokens):
    return cache._trie.search("m", tokens).exact is not None


def test_the_system_entry_survives_a_trim_that_takes_everything_else():
    cache = _lru()
    cache.insert_cache("m", [1, 2], [_StubLayer()], cache_type="system")
    for i in range(10):
        cache.insert_cache("m", [10 + i, 20 + i], [_StubLayer()])

    cache.trim_to(n_bytes=3_000_000)  # room for ~3 entries

    assert _present(cache, [1, 2]), "the shared entry was evicted before per-turn ones"
    assert len(cache._lru) < 11, "nothing was actually evicted; the test proves nothing"


def test_untagged_the_system_entry_dies_first():
    """The control. Without it the test above passes for the wrong reason -- it
    would look like a win even if trim_to happened to spare the oldest entry."""
    cache = _lru()
    cache.insert_cache("m", [1, 2], [_StubLayer()])  # untagged: today's bug
    for i in range(10):
        cache.insert_cache("m", [10 + i, 20 + i], [_StubLayer()])

    cache.trim_to(n_bytes=3_000_000)

    assert not _present(cache, [1, 2])


def test_priority_is_not_a_leak():
    """Last-to-go is correct; never-goes is a slow memory leak with a good
    excuse. An entry that cannot be evicted is eventually the reason the machine
    runs out of memory, and this matters more than the happy path."""
    cache = _lru()
    cache.insert_cache("m", [1, 2], [_StubLayer()], cache_type="system")

    cache.trim_to(n_bytes=0)

    assert len(cache._lru) == 0
    assert not _present(cache, [1, 2])


def test_on_a_dense_model_priority_does_not_protect_a_prefix_entry():
    """Documents a real gap rather than asserting a fix.

    With a trimmable cache (Ministral 3: full attention, no recurrent state)
    insert_cache drops entries that are strict prefixes of a new one, via
    pop_prefixes, BEFORE any eviction ordering runs -- so cache_type does
    nothing there. Arguably right on a dense model, since the longer entry can
    be trimmed back to serve the same prefix, but it means this priority fix is
    Qwen3.6-shaped. If this test ever fails, upstream changed that behaviour and
    the Ministral question in specs/shared-checkpoint-priority.md is reopened."""
    cache = _lru()
    cache.insert_cache("m", [1, 2], [_StubLayer(trimmable=True)], cache_type="system")
    cache.insert_cache("m", [1, 2, 3, 4], [_StubLayer(trimmable=True)])

    assert not _present(cache, [1, 2]), "pop_prefixes no longer drops the prefix entry"
