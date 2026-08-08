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
    SNAPSHOT_MIN_BOUNDARY_TOKENS,
    GenerationEngine,
    plan_prefill_segments,
)


# ── where prefill splits ──────────────────────────────────────────────────────

def test_prefill_splits_at_the_boundary():
    rest = list(range(100))
    segments, to_cache = plan_prefill_segments(rest, prompt_cache_count=0, boundary_n=60)
    assert segments == [rest[:60], rest[60:]]
    assert to_cache == 60


def test_the_split_is_offset_by_what_the_cache_already_covered():
    """boundary_n indexes the whole prompt; rest starts after the reused prefix.
    Splitting at boundary_n directly would cut 40 tokens too late and cache a
    sequence that was never the boundary."""
    rest = list(range(100))
    segments, to_cache = plan_prefill_segments(rest, prompt_cache_count=40, boundary_n=60)
    assert segments == [rest[:20], rest[20:]]
    assert to_cache == 60


def test_a_boundary_already_covered_by_the_cache_needs_no_split():
    segments, to_cache = plan_prefill_segments(list(range(100)), 60, 60)
    assert to_cache is None
    assert len(segments) == 1


def test_a_boundary_at_the_very_end_needs_no_split():
    """A second segment would be empty, and there would be nothing to generate
    from."""
    segments, to_cache = plan_prefill_segments(list(range(100)), 0, 100)
    assert to_cache is None
    assert len(segments) == 1


def test_no_boundary_means_the_prompt_is_prefilled_as_before():
    rest = list(range(10))
    assert plan_prefill_segments(rest, 0, None) == ([rest], None)


def test_the_two_segments_always_reconstruct_the_prompt():
    """Any off-by-one here silently drops or duplicates a token, which would
    corrupt generation rather than merely slow it."""
    rest = list(range(1000))
    for count, boundary in ((0, 1), (0, 999), (250, 500), (999, 1000)):
        segments, _ = plan_prefill_segments(rest, count, boundary)
        assert [t for s in segments for t in s] == rest, (count, boundary)


class _Tok:
    """Renders history and history+scaffold, the way a real template does."""

    SCAFFOLD = [901, 902, 903]

    def __init__(self, history_ids=None, prefix_ok=True, boom=False):
        self.history_ids = history_ids if history_ids is not None else list(range(1, 2001))
        self.prefix_ok = prefix_ok
        self.boom = boom

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=True, **kw):
        if self.boom:
            raise RuntimeError("template exploded")
        return "history-text"

    def encode(self, text, add_special_tokens=False):
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

    def insert_cache(self, model, tokens, cache):
        self.inserted.append((model, list(tokens), cache))


def _armed(engine, uid=7, tokens=(1, 2, 3)):
    engine.prompt_cache = _PromptCache()
    engine.batch_generator = SimpleNamespace(_prompt_batch=_Batch([uid]))
    engine._pending = {uid: {"snapshot_tokens": list(tokens)}}
    return uid


def _resp(uid, end_of_segment=True):
    return SimpleNamespace(uid=uid, end_of_segment=end_of_segment,
                           end_of_prompt=False, progress=(0, 0))


def test_it_caches_the_boundary_state(engine):
    uid = _armed(engine, tokens=(11, 22, 33))
    engine._maybe_snapshot_boundary(_resp(uid))

    assert len(engine.prompt_cache.inserted) == 1
    model, tokens, _ = engine.prompt_cache.inserted[0]
    assert model == "fake/model"
    assert tokens == [11, 22, 33], "entry keyed on something other than the boundary"
    assert engine._snapshots_taken == 1


def test_only_the_first_segment_is_snapshotted(engine):
    """insert_segments always peels the final token into its own segment, so
    end_of_segment fires more than once per job. Only the first is the boundary."""
    uid = _armed(engine)
    engine._maybe_snapshot_boundary(_resp(uid))
    engine._maybe_snapshot_boundary(_resp(uid))
    engine._maybe_snapshot_boundary(_resp(uid))
    assert len(engine.prompt_cache.inserted) == 1
    assert engine._snapshots_taken == 1


def test_a_response_that_is_not_a_segment_end_does_nothing(engine):
    uid = _armed(engine)
    engine._maybe_snapshot_boundary(_resp(uid, end_of_segment=False))
    assert engine.prompt_cache.inserted == []


def test_a_job_whose_prefill_was_not_split_is_left_alone(engine):
    """snapshot_tokens is set only when a boundary segment actually exists.
    Without that guard the first end_of_segment of an unsplit job would cache
    the prompt minus its last token, which is not a prefix of anything."""
    uid = 7
    engine.prompt_cache = _PromptCache()
    engine.batch_generator = SimpleNamespace(_prompt_batch=_Batch([uid]))
    engine._pending = {uid: {"snapshot_tokens": None}}
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
    engine._pending = {uid: {"snapshot_tokens": [1]}}
    engine._maybe_snapshot_boundary(_resp(uid))
    assert batch.extracted == [2]


def test_a_failed_extract_costs_the_snapshot_and_nothing_else(engine):
    uid = 7
    engine.prompt_cache = _PromptCache()
    engine.batch_generator = SimpleNamespace(_prompt_batch=_Batch([uid], boom=True))
    engine._pending = {uid: {"snapshot_tokens": [1, 2]}}

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


def test_the_entry_ceiling_leaves_room_for_two_entries_per_turn():
    """A turn now stores the completed turn AND its boundary snapshot. Left at
    mlx-lm's default of 10 this would have quietly halved how many
    conversations stay warm, which is a slowdown nothing would have reported."""
    from core.inference.mira_mlx_server import PROMPT_CACHE_MAX_ENTRIES
    assert PROMPT_CACHE_MAX_ENTRIES >= 20


def test_the_threshold_is_above_a_bare_system_prompt():
    """Below this the entry saves little and still costs an LRU slot; coverage
    measured at ~44% for a short exchange against ~95% for a real conversation."""
    assert SNAPSHOT_MIN_BOUNDARY_TOKENS >= 1024
