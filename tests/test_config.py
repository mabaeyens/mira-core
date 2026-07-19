"""Config resolution rules that don't need MLX or a loaded model."""
from core.config import _resolve_resident_fraction


def test_offload_on_uses_the_fraction():
    assert _resolve_resident_fraction(True, 0.3) == 0.3
    assert _resolve_resident_fraction(True, 0.15) == 0.15


def test_offload_flag_false_disables_regardless_of_fraction():
    # The "go back to the simpler lazy mechanism" flag wins over any fraction.
    assert _resolve_resident_fraction(False, 0.3) is None
    assert _resolve_resident_fraction(False, 0.15) is None


def test_fraction_ge_one_means_keep_everything_resident():
    assert _resolve_resident_fraction(True, 1.0) is None
    assert _resolve_resident_fraction(True, 2.0) is None


def test_null_fraction_disables():
    assert _resolve_resident_fraction(True, None) is None


def test_returns_float_even_for_int_like_input():
    got = _resolve_resident_fraction(True, "0.3")  # yaml can hand back a string
    assert got == 0.3 and isinstance(got, float)
