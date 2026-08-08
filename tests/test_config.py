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


def test_mira_config_env_redirects_the_yaml_and_is_off_by_default(tmp_path, monkeypatch):
    """MIRA_CONFIG lets the bench run on a copy of the live config with one
    setting changed, instead of editing the real file and hoping to restore it.

    The default path must be unaffected: this exists for benches, and the setting
    it was added for (private-URL fetching) is one whose secure default must not
    move for everyone else.
    """
    import yaml

    alt = tmp_path / "alt.yaml"
    alt.write_text(yaml.safe_dump({"url_fetch_allow_private": True,
                                   "model": "someone/else"}))

    monkeypatch.setenv("MIRA_CONFIG", str(alt))
    loaded = config._load_yaml_config()
    assert loaded["url_fetch_allow_private"] is True
    assert loaded["model"] == "someone/else"

    monkeypatch.delenv("MIRA_CONFIG")
    assert config._load_yaml_config().get("url_fetch_allow_private", False) is False

    # A path that does not exist falls back to empty rather than crashing the
    # server on a typo'd env var.
    monkeypatch.setenv("MIRA_CONFIG", str(tmp_path / "nope.yaml"))
    assert config._load_yaml_config() == {}
