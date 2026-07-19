"""Config resolution rules that don't need MLX or a loaded model."""
import core.config as config
from core.config import _parse_offload_mode, _resolve_resident_fraction, resolve_offload_fraction


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


def test_parse_offload_mode():
    assert _parse_offload_mode("auto") == "auto"
    assert _parse_offload_mode(None) == "auto"  # unknown/absent -> auto
    assert _parse_offload_mode(True) == "on"
    assert _parse_offload_mode("true") == "on"
    assert _parse_offload_mode("ON") == "on"
    assert _parse_offload_mode(False) == "off"
    assert _parse_offload_mode("false") == "off"
    assert _parse_offload_mode("off") == "off"


def _patch(monkeypatch, *, mode, fraction, fits):
    monkeypatch.setattr(config, "MIRA_MLX_EXPERT_OFFLOAD_MODE", mode)
    monkeypatch.setattr(config, "MIRA_MLX_RESIDENT_EXPERT_FRACTION", fraction)
    import core.hardware as hardware
    monkeypatch.setattr(hardware, "fits_in_memory", lambda *a, **k: (fits, "test"))


def test_resolve_offload_off_never_offloads(monkeypatch):
    _patch(monkeypatch, mode="off", fraction=0.3, fits=False)
    assert resolve_offload_fraction("any/model") is None


def test_resolve_offload_on_always_offloads(monkeypatch):
    _patch(monkeypatch, mode="on", fraction=0.3, fits=True)  # fits, but forced on
    assert resolve_offload_fraction("any/model") == 0.3


def test_resolve_offload_auto_fits_stays_resident(monkeypatch):
    _patch(monkeypatch, mode="auto", fraction=0.3, fits=True)
    assert resolve_offload_fraction("fits/model") is None


def test_resolve_offload_auto_overflow_offloads(monkeypatch):
    _patch(monkeypatch, mode="auto", fraction=0.3, fits=False)
    assert resolve_offload_fraction("huge/model") == 0.3


def test_resolve_offload_auto_no_fraction_stays_off(monkeypatch):
    # If the fraction knob itself disabled offload, auto can't turn it on.
    _patch(monkeypatch, mode="auto", fraction=None, fits=False)
    assert resolve_offload_fraction("huge/model") is None
