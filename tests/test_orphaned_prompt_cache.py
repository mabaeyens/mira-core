"""The leftover-disk-cache warning.

Turning the disk prompt cache off (DISK_PROMPT_CACHE, 2026-08-08) stops it being
written but deletes nothing: the only code that ever removed one of those files
was the store's own LRU eviction, which no longer runs. So every install that
upgrades keeps whatever it had — 39.75GB on the machine where this was found —
with nothing referencing it. These tests cover the reporting, which is the whole
mitigation: Mira does not delete gigabytes out of a user's data directory on its
own, it says what is there and what to run.

`core.hardware` imports no mlx, so this runs on Linux CI.
"""
import pytest

from core import hardware


def test_reports_nothing_for_a_missing_directory(tmp_path):
    assert hardware.orphaned_prompt_cache(tmp_path / "not_there") == (0, 0)


def test_reports_nothing_for_an_empty_directory(tmp_path):
    assert hardware.orphaned_prompt_cache(tmp_path) == (0, 0)


def test_counts_and_sums_cache_files(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "b.safetensors").write_bytes(b"x" * 250)
    assert hardware.orphaned_prompt_cache(tmp_path) == (2, 350)


def test_ignores_files_that_are_not_cache_entries(tmp_path):
    """The real directory sits beside conversations.db; a glob that took
    everything would report the user's own data as garbage to delete."""
    (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "conversations.db").write_bytes(b"x" * 9999)
    (tmp_path / "notes.txt").write_bytes(b"x" * 9999)
    assert hardware.orphaned_prompt_cache(tmp_path) == (1, 10)


def test_survives_a_file_vanishing_mid_scan(tmp_path, monkeypatch):
    """A running engine can unlink an entry between glob and stat. The health
    check must degrade, not raise."""
    (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "b.safetensors").write_bytes(b"x" * 10)

    real_stat = hardware.Path.stat
    calls = {"n": 0}

    def flaky_stat(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("vanished")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(hardware.Path, "stat", flaky_stat)
    count, nbytes = hardware.orphaned_prompt_cache(tmp_path)
    assert (count, nbytes) == (1, 10)


@pytest.mark.parametrize("n,expected", [
    (42_678_000_000, "39.75 GB"),
    (4_500_000, "4.3 MB"),
    (5_000, "5 KB"),
    (12, "12 bytes"),
    (0, "0 bytes"),
])
def test_sizes_stay_informative_at_both_ends(n, expected):
    """A fixed GB unit printed "0.00 GB of dead files" for a few megabytes, which
    reads as nothing at all inside a warning asking someone to act."""
    assert hardware.format_bytes(n) == expected


def test_server_stays_silent_when_the_cache_is_enabled(monkeypatch, caplog):
    """No warning while the store is in use — those files are not orphans."""
    import server

    monkeypatch.setattr(server, "DISK_PROMPT_CACHE", True)
    monkeypatch.setattr(
        hardware, "orphaned_prompt_cache",
        lambda d: pytest.fail("scanned the directory while the cache was enabled"),
    )
    with caplog.at_level("WARNING"):
        server._warn_orphaned_prompt_cache()
    assert not caplog.records


def test_server_warns_with_the_size_and_the_command(monkeypatch, caplog):
    import server

    monkeypatch.setattr(server, "DISK_PROMPT_CACHE", False)
    monkeypatch.setattr(hardware, "orphaned_prompt_cache", lambda d: (296, 42_678_000_000))
    with caplog.at_level("WARNING"):
        server._warn_orphaned_prompt_cache()
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "296" in msg
    assert "39.7" in msg          # bytes rendered as GB, not as a raw integer
    assert "rm -rf" in msg        # actionable, not just descriptive
    assert "mira_mlx_cache" in msg


def test_server_says_nothing_when_there_is_nothing_left(monkeypatch, caplog):
    import server

    monkeypatch.setattr(server, "DISK_PROMPT_CACHE", False)
    monkeypatch.setattr(hardware, "orphaned_prompt_cache", lambda d: (0, 0))
    with caplog.at_level("WARNING"):
        server._warn_orphaned_prompt_cache()
    assert not caplog.records


def test_a_broken_scan_never_blocks_startup(monkeypatch, caplog):
    import server

    monkeypatch.setattr(server, "DISK_PROMPT_CACHE", False)

    def boom(_):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(hardware, "orphaned_prompt_cache", boom)
    with caplog.at_level("WARNING"):
        server._warn_orphaned_prompt_cache()  # must not raise
    assert not caplog.records
