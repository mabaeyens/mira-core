"""Phase A correctness gate for specs/moe-expert-offload-02-runtime-cache.md:
enable_offload() must produce bit-identical output vs the unmodified
eager/fully-resident path, both on a cold cache (every call is a miss) and a
warm one (repeated calls hit the LRU), for both SwitchLinear and
QuantizedSwitchLinear. No downloaded model needed — a small random-weight
switch layer is saved to a real safetensors shard on disk (the exact format
mlx_lm.utils.load() reads), so DiskExpertCacheStore's byte-range reads are
exercised for real against the same on-disk layout Qwen3.6 uses.
"""
import pytest

mx = pytest.importorskip("mlx.core")  # mlx is macOS-only (Apple Silicon), absent on Linux CI
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU, SwitchLinear

from core.inference.disk_expert_cache import DiskExpertCacheStore

INPUT_DIMS = 32
OUTPUT_DIMS = 48
NUM_EXPERTS = 16
RESIDENT_SLOTS = 3  # << num_experts, forces frequent eviction


def _random_indices(n_calls=6, tokens_per_call=5, seed=0):
    key = mx.random.key(seed)
    out = []
    for i in range(n_calls):
        key, sub = mx.random.split(key)
        out.append(mx.random.randint(0, NUM_EXPERTS, (tokens_per_call, 1), key=sub))
    return out


def _make_switch_linear():
    return SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)


def _make_quantized_switch_linear():
    return _make_switch_linear().to_quantized(group_size=32, bits=4)


def _run_baseline_vs_offload(make_module, attrs, tmp_path):
    baseline = make_module()
    mx.eval(baseline.parameters())

    weights_to_save = {f"test.switch.{attr}": baseline[attr] for attr in attrs if baseline.get(attr) is not None}
    shard = tmp_path / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), weights_to_save)

    offloaded = make_module()
    # Overwrite offloaded's random init with baseline's exact weights so both
    # modules start from identical values before any offload/eviction.
    for attr in attrs:
        if baseline.get(attr) is not None:
            offloaded[attr] = baseline[attr]
    mx.eval(offloaded.parameters())

    store = DiskExpertCacheStore(tmp_path)
    fetch_fn = store.reader_for("test.switch", attrs)
    offloaded.enable_offload(RESIDENT_SLOTS, fetch_fn)

    x = mx.random.normal((5, 1, 1, INPUT_DIMS))
    all_indices = _random_indices()

    # Cold pass: every distinct expert id seen for the first time is a miss.
    for indices in all_indices:
        expected = baseline(x, indices)
        actual = offloaded(x, indices)
        assert mx.array_equal(expected, actual), "offload output diverged from baseline on cold/mixed cache"

    # Warm pass: replay the exact same indices sequence — mix of hits (still
    # LRU-resident) and misses (evicted since by later calls in the cold pass).
    for indices in all_indices:
        expected = baseline(x, indices)
        actual = offloaded(x, indices)
        assert mx.array_equal(expected, actual), "offload output diverged from baseline on warm replay"

    assert store.misses > 0, "test setup bug: no misses were recorded, cache was never exercised"
    return store


def test_switch_linear_offload_matches_baseline(tmp_path):
    _run_baseline_vs_offload(_make_switch_linear, ["weight"], tmp_path)


def test_quantized_switch_linear_offload_matches_baseline(tmp_path):
    _run_baseline_vs_offload(_make_quantized_switch_linear, ["weight", "scales", "biases"], tmp_path)


def test_offload_matches_baseline_when_call_exceeds_capacity(tmp_path):
    """A single call's unique expert count can exceed resident_slots (e.g. a
    large batch with diverse routing) — every needed expert must still be
    correctly gathered, even though it can't all stay resident afterward."""
    global RESIDENT_SLOTS
    original = RESIDENT_SLOTS
    RESIDENT_SLOTS = 2
    try:
        store = _run_baseline_vs_offload(
            _make_switch_linear, ["weight"], tmp_path,
        )
    finally:
        RESIDENT_SLOTS = original
    assert store.misses > 0


def test_offload_matches_baseline_with_realistic_top_k_routing(tmp_path):
    """Regression test for a real bug found during implementation: the first
    chunked-resolve version assumed gather_mm/gather_qmm's output has exactly
    one more trailing dim than `indices` (true only for top_k=1, which is all
    the other tests above use) — with real multi-slot-per-token routing
    (top_k > 1, matching SwitchGLU's actual (N, top_k) indices shape),
    output.ndim is indices.ndim + 2, not + 1, so the mask broadcast silently
    duplicated data across the top_k axis instead of selecting per-position.
    Uses 32 experts / resident=4 / max_stack=4 (many chunks per call) and
    top_k=8 over 20 tokens — up to 160 selections against 32 experts, forcing
    heavy chunking on every call, not just an occasional one."""
    d, o, n_experts, resident, max_stack = 32, 48, 32, 4, 4
    n_tokens, top_k = 20, 8

    baseline = SwitchLinear(d, o, n_experts, bias=False)
    mx.eval(baseline.parameters())
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(str(shard), {"test.switch.weight": baseline.weight})

    offloaded = SwitchLinear(d, o, n_experts, bias=False)
    offloaded.weight = baseline.weight
    mx.eval(offloaded.parameters())
    store = DiskExpertCacheStore(tmp_path)
    offloaded.enable_offload(resident, store.reader_for("test.switch", ["weight"]), max_stack_size=max_stack)

    x = mx.expand_dims(mx.random.normal((n_tokens, d)), (-2, -3))
    key = mx.random.key(1)
    for _ in range(5):
        key, sub = mx.random.split(key)
        indices = mx.random.randint(0, n_experts, (n_tokens, top_k), key=sub)
        expected = baseline(x, indices)
        actual = offloaded(x, indices)
        assert expected.shape == actual.shape
        assert mx.array_equal(expected, actual)
    assert store.misses > 0


def test_offload_bounds_cache_size_during_large_diverse_call(tmp_path):
    """The whole point of chunking: a call whose unique-expert count is much
    larger than resident_slots must never let the cache dict grow anywhere
    near that unique count — it should stay bounded by roughly
    capacity + max_stack_size throughout the call, not balloon to near the
    full expert table the way the pre-chunking design did (that ballooning,
    on top of a freshly-stacked near-full-size temp tensor, is what caused
    the real Metal OOM crash under a large prefill)."""
    d, o, n_experts, resident, max_stack = 16, 24, 64, 4, 4

    baseline = SwitchLinear(d, o, n_experts, bias=False)
    mx.eval(baseline.parameters())
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(str(shard), {"test.switch.weight": baseline.weight})

    offloaded = SwitchLinear(d, o, n_experts, bias=False)
    offloaded.weight = baseline.weight
    mx.eval(offloaded.parameters())
    store = DiskExpertCacheStore(tmp_path)

    max_observed_cache_size = 0
    real_fetch = store.reader_for("test.switch", ["weight"])

    def spying_fetch(expert_id):
        nonlocal max_observed_cache_size
        max_observed_cache_size = max(max_observed_cache_size, len(offloaded._offload_cache))
        return real_fetch(expert_id)

    offloaded.enable_offload(resident, spying_fetch, max_stack_size=max_stack)

    # All 64 experts touched by one call — worst-case diversity.
    x = mx.expand_dims(mx.random.normal((64, d)), (-2, -3))
    indices = mx.arange(64).reshape(64, 1)
    baseline(x, indices)
    offloaded(x, indices)

    assert max_observed_cache_size <= resident + max_stack, (
        f"cache dict grew to {max_observed_cache_size} entries during a single call "
        f"(bound should be capacity {resident} + max_stack_size {max_stack})"
    )


def test_offload_frees_full_weight_and_preserves_num_experts(tmp_path):
    """The fix for the Phase C 'peak memory > full-resident baseline' gap.

    The old seed sliced `module.weight[:resident_slots]` and kept it as
    module.weight — but an mx prefix-slice is a VIEW that pins the entire
    parent buffer, so the full expert table was never actually freed and
    offload only ever added overhead on top of it. The seed now comes from
    disk (independent buffers) and module.weight is replaced by a 1-row
    stand-in, so the full table has no remaining reference and is freed. The
    stand-in can't carry the true expert count in shape[0], so num_experts
    must come from the stashed value (the model routes on the true count)."""
    for make, attrs in (
        (_make_switch_linear, ["weight"]),
        (_make_quantized_switch_linear, ["weight", "scales", "biases"]),
    ):
        baseline = make()
        mx.eval(baseline.parameters())
        weights = {f"test.switch.{a}": baseline[a] for a in attrs if baseline.get(a) is not None}
        shard = tmp_path / "model-00001-of-00001.safetensors"
        mx.save_safetensors(str(shard), weights)

        module = make()
        for a in attrs:
            if baseline.get(a) is not None:
                module[a] = baseline[a]
        mx.eval(module.parameters())
        store = DiskExpertCacheStore(tmp_path)
        module.enable_offload(RESIDENT_SLOTS, store.reader_for("test.switch", attrs))

        assert module.num_experts == NUM_EXPERTS, "num_experts must report the true table size after enable"
        assert module.weight.shape[0] == 1, "full weight must be replaced by a 1-row stand-in (table freed)"
        assert module.input_dims == INPUT_DIMS, "input_dims must still resolve off the stand-in"
        assert module.output_dims == OUTPUT_DIMS, "output_dims must still resolve off the stand-in"
        assert len(module._offload_cache) == RESIDENT_SLOTS, "exactly resident_slots experts seeded from disk"
        shard.unlink()


# --------------------------------------------------------------------------
# Compact chunked-gather path (sorted_indices=True): when SwitchGLU/SwitchMLP
# pre-sort the routing (they do whenever indices.size >= 64, i.e. the
# multi-group prefill case), _offload_chunked_gather slices x per contiguous
# group and concatenates, instead of running every group's gather over ALL
# positions and mx.where-combining (the "G-fold" prefill tax).
#
# Ground truth note: the shipping offload gather_fn always passes
# sorted_indices=False to mx.gather_mm/gather_qmm, and mx's sorted_indices=True
# kernel hint is NOT bit-identical to False (~2e-3, verified). So the exact
# reference for ANY offload path (mask or compact) is the full-resident call
# with sorted_indices=False, and the strongest regression gate is
# compact-output == mask-output, which these tests assert directly.
# --------------------------------------------------------------------------

_COMPACT_D, _COMPACT_O, _COMPACT_NE = 32, 48, 32
_COMPACT_RESIDENT, _COMPACT_MAX_STACK = 4, 4  # 4-wide groups << 32 experts => many groups


def _make_compact_switch_linear():
    return SwitchLinear(_COMPACT_D, _COMPACT_O, _COMPACT_NE, bias=False)


def _make_compact_quantized():
    return _make_compact_switch_linear().to_quantized(group_size=32, bits=4)


def _compact_baseline_and_offload(make, attrs, tmp_path):
    baseline = make()
    mx.eval(baseline.parameters())
    shard = tmp_path / "model-00001-of-00001.safetensors"
    mx.save_safetensors(
        str(shard), {f"test.switch.{a}": baseline[a] for a in attrs if baseline.get(a) is not None}
    )
    offloaded = make()
    for a in attrs:
        if baseline.get(a) is not None:
            offloaded[a] = baseline[a]
    mx.eval(offloaded.parameters())
    store = DiskExpertCacheStore(tmp_path)
    offloaded.enable_offload(
        _COMPACT_RESIDENT, store.reader_for("test.switch", attrs), max_stack_size=_COMPACT_MAX_STACK
    )
    return baseline, offloaded, store


def _sorted_x_and_idx(n=80, seed=3):
    """1-D ascending expert ids over many groups, plus an (n, 1, d) activation
    — exactly the shape/order SwitchGLU produces after _gather_sort."""
    idx = mx.sort(mx.random.randint(0, _COMPACT_NE, (n,), key=mx.random.key(seed)))
    x = mx.expand_dims(mx.random.normal((n, _COMPACT_D), key=mx.random.key(seed + 100)), (-2, -3)).flatten(0, -3)
    return x, idx


@pytest.mark.parametrize(
    "make,attrs",
    [
        (_make_compact_switch_linear, ["weight"]),
        (_make_compact_quantized, ["weight", "scales", "biases"]),
    ],
)
def test_compact_path_matches_plain_gather(make, attrs, tmp_path):
    """Compact-path output is bit-identical to the full-resident plain gather
    (sorted_indices=False, the exact reference for the offload feature), for
    both SwitchLinear and QuantizedSwitchLinear."""
    baseline, offloaded, store = _compact_baseline_and_offload(make, attrs, tmp_path)
    x, idx = _sorted_x_and_idx()
    expected = baseline(x, idx, sorted_indices=False)  # plain gather = exact ground truth
    actual = offloaded(x, idx, sorted_indices=True)     # compact (slice+concat) path
    assert expected.shape == actual.shape
    assert mx.array_equal(expected, actual), "compact path diverged from full-resident plain gather"
    assert store.misses > 0, "test setup bug: compact path never faulted an expert to disk"


def test_compact_path_bit_identical_to_mask_path(tmp_path, monkeypatch):
    """The core regression gate: enabling compaction must not change output vs
    the shipping mask path by a single bit. Run the same offloaded module +
    inputs once through the compact path, once with _offload_chunked_gather
    monkey-forced onto the mask path (sorted_indices=False), and compare."""
    import mlx_lm.models.switch_layers as sl

    _baseline, offloaded, _store = _compact_baseline_and_offload(_make_compact_switch_linear, ["weight"], tmp_path)
    x, idx = _sorted_x_and_idx(seed=21)

    compact = offloaded(x, idx, sorted_indices=True)  # compact branch (guaranteed sorted)

    orig = sl._offload_chunked_gather
    monkeypatch.setattr(
        sl, "_offload_chunked_gather",
        lambda m, xx, ii, g, sorted_indices=False: orig(m, xx, ii, g, sorted_indices=False),
    )
    mask = offloaded(x, idx, sorted_indices=True)  # same call, forced down mask path
    assert mx.array_equal(compact, mask), "compact path is not bit-identical to the mask path"


def test_compact_path_is_actually_taken_when_sorted(tmp_path, monkeypatch):
    """Prove the sorted call goes through the compact (slice+concat) branch and
    NOT the mask branch: mx.where is used ONLY by the mask path, so if it is
    never called on a sorted multi-group call, the compact branch ran. The same
    call with sorted_indices=False must, by contrast, hit mx.where."""
    _baseline, offloaded, _store = _compact_baseline_and_offload(_make_compact_switch_linear, ["weight"], tmp_path)
    x, idx = _sorted_x_and_idx(seed=7)

    where_calls = {"n": 0}
    real_where = mx.where
    monkeypatch.setattr(
        mx, "where",
        lambda *a, **k: (where_calls.__setitem__("n", where_calls["n"] + 1), real_where(*a, **k))[1],
    )

    offloaded(x, idx, sorted_indices=True)
    assert where_calls["n"] == 0, "sorted multi-group call should skip mx.where (compact path not taken)"

    offloaded(x, idx, sorted_indices=False)
    assert where_calls["n"] > 0, "unsorted call should use the mx.where mask path"


def test_compact_path_falls_back_when_sorted_flag_lies(tmp_path):
    """A caller could pass sorted_indices=True with indices that are NOT
    actually monotonic. The compact path must detect this (its own monotonicity
    check, not just the flag) and fall back to the always-correct mask path, so
    output is never silently scattered to the wrong positions."""
    _baseline, offloaded, store = _compact_baseline_and_offload(_make_compact_switch_linear, ["weight"], tmp_path)
    baseline = _baseline

    # Deliberately UNSORTED 1-D indices, but we lie and pass sorted_indices=True.
    idx = mx.random.randint(0, _COMPACT_NE, (80,), key=mx.random.key(11))
    assert not bool(mx.all(idx[1:] >= idx[:-1])), "test needs genuinely unsorted indices"
    x = mx.expand_dims(mx.random.normal((80, _COMPACT_D), key=mx.random.key(12)), (-2, -3)).flatten(0, -3)

    expected = baseline(x, idx, sorted_indices=False)  # unsorted => plain gather is ground truth
    actual = offloaded(x, idx, sorted_indices=True)     # flag lies; guard must catch it
    assert mx.array_equal(expected, actual), "mislabelled sorted flag corrupted output (guard failed)"
    assert store.misses > 0


def test_switchglu_end_to_end_compact_matches_mask(tmp_path, monkeypatch):
    """Full SwitchGLU dispatch: its own _gather_sort feeds sorted indices to the
    three offloaded sublayers, exercising the compact path through the real call
    path. Compare an offloaded SwitchGLU's output with compaction ON vs the same
    module forced onto the mask path — they must be bit-identical. (Can't compare
    to the full-resident SwitchGLU: it uses gather sorted_indices=True internally,
    whose kernel hint isn't bit-identical to the offload path's False.)"""
    import mlx_lm.models.switch_layers as sl

    d, hidden, n_experts, top_k, n_tokens = 32, 48, 32, 8, 16
    subs = ["gate_proj", "up_proj", "down_proj"]

    ref = SwitchGLU(d, hidden, n_experts, bias=False)
    mx.eval(ref.parameters())
    shard = tmp_path / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), {f"glu.{s}.weight": ref[s].weight for s in subs})

    offloaded = SwitchGLU(d, hidden, n_experts, bias=False)
    for s in subs:
        offloaded[s].weight = ref[s].weight
    mx.eval(offloaded.parameters())
    store = DiskExpertCacheStore(tmp_path)
    for s in subs:
        offloaded[s].enable_offload(4, store.reader_for(f"glu.{s}", ["weight"]), max_stack_size=4)

    x = mx.random.normal((n_tokens, d), key=mx.random.key(5))
    indices = mx.random.randint(0, n_experts, (n_tokens, top_k), key=mx.random.key(6))
    assert indices.size >= 64, "test must trigger SwitchGLU's do_sort branch"

    compact = offloaded(x, indices)  # compact path in all three sublayers

    orig = sl._offload_chunked_gather
    monkeypatch.setattr(
        sl, "_offload_chunked_gather",
        lambda m, xx, ii, g, sorted_indices=False: orig(m, xx, ii, g, sorted_indices=False),
    )
    mask = offloaded(x, indices)  # forced mask path

    assert compact.shape == mask.shape
    assert mx.array_equal(compact, mask), "SwitchGLU compact path diverged from mask path through the sort dispatch"
    assert store.misses > 0


def test_enable_offload_is_noop_when_resident_slots_covers_all_experts(tmp_path):
    """resident_slots >= num_experts must leave the module fully unchanged —
    the default/unset behavior guarantee the spec requires end to end."""
    module = _make_switch_linear()
    original_weight = module.weight
    store = DiskExpertCacheStore(tmp_path)  # never touched: enable_offload should early-return

    def _unused_fetch(expert_id):
        raise AssertionError("fetch_fn must never be called when offload is a no-op")

    module.enable_offload(NUM_EXPERTS, _unused_fetch)
    assert module.weight.shape == original_weight.shape
    assert not hasattr(module, "_offload_fetch")
