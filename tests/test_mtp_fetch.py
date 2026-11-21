"""fetch_mtp_sidecar: range-fetches only mtp.* tensors and writes a valid sidecar.

Offline: the upstream repo is faked by patching the two HTTP helpers.
"""
import json
import struct

import pytest

pytest.importorskip("mlx.core")

from core.inference.mtp import sidecar as sc  # noqa: E402


def _shard(tensors: dict[str, bytes]) -> bytes:
    header, off = {}, 0
    for name, data in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [off, off + len(data)]}
        off += len(data)
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"".join(tensors.values())


@pytest.fixture
def upstream(monkeypatch):
    shards = {
        "model-00001.safetensors": _shard({"model.layers.0.w": b"x" * 50, "mtp.fc.weight": b"A" * 40}),
        "model-00002.safetensors": _shard({"mtp.norm.weight": b"B" * 24, "lm_head.weight": b"y" * 30}),
    }
    index = {"weight_map": {
        "model.layers.0.w": "model-00001.safetensors", "mtp.fc.weight": "model-00001.safetensors",
        "mtp.norm.weight": "model-00002.safetensors", "lm_head.weight": "model-00002.safetensors"}}
    fetched = []

    def fake_json(url):
        return index

    def fake_range(url, start, end, **kw):
        fetched.append(end - start + 1)
        return shards[url.rsplit("/", 1)[1]][start:end + 1]

    monkeypatch.setattr(sc, "_http_json", fake_json)
    monkeypatch.setattr(sc, "_http_range", fake_range)
    return fetched


def test_fetches_only_mtp_tensors_and_validates(tmp_path, upstream):
    dest = tmp_path / "m" / "model-mtp.safetensors"
    seen = []
    sc.fetch_mtp_sidecar(dest, progress=lambda d, t: seen.append((d, t)))
    assert sc.sidecar_has_mtp_tensors(dest)
    with dest.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        data = f.read()
    assert set(header) == {"mtp.fc.weight", "mtp.norm.weight"}  # non-mtp tensors excluded
    assert (8 + n) % 8 == 0
    fc = header["mtp.fc.weight"]["data_offsets"]
    nm = header["mtp.norm.weight"]["data_offsets"]
    assert data[fc[0]:fc[1]] == b"A" * 40 and data[nm[0]:nm[1]] == b"B" * 24
    assert seen[-1] == (64, 64)
    assert not dest.with_name(dest.name + ".part").exists()


def test_is_idempotent(tmp_path, upstream):
    dest = tmp_path / "model-mtp.safetensors"
    sc.fetch_mtp_sidecar(dest)
    before = len(upstream)
    sc.fetch_mtp_sidecar(dest)
    assert len(upstream) == before  # no network on the second call


def test_repo_without_mtp_tensors_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_http_json", lambda url: {"weight_map": {"a": "s.safetensors"}})
    with pytest.raises(ValueError):
        sc.fetch_mtp_sidecar(tmp_path / "x.safetensors")


def test_set_yaml_scalar_replaces_uncomments_or_appends(tmp_path, monkeypatch):
    import mira_cli

    cfg = tmp_path / "mira.yaml"
    cfg.write_text("backend: mira-mlx\nmodel: old/repo   # c\n# mira_mlx_mtp_enabled: false\nother: 1\n")
    monkeypatch.setenv("MIRA_CONFIG", str(cfg))
    mira_cli._set_yaml_scalar("model", "/new/dir")
    mira_cli._set_yaml_scalar("mira_mlx_mtp_enabled", "true")
    mira_cli._set_yaml_scalar("mira_mlx_mtp_draft_vocab", "131072")
    lines = cfg.read_text().splitlines()
    assert lines == ["backend: mira-mlx", "model: /new/dir", "mira_mlx_mtp_enabled: true",
                     "other: 1", "mira_mlx_mtp_draft_vocab: 131072"]
