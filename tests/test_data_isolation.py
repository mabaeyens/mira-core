"""The test suite must not write to the user's real data directory.

Until 2026-08-01 it did. `core/config.py` hardcoded
`DB_PATH = ~/.local/share/mira/conversations.db` with no override, so every
TestClient run created rows in Miguel's own conversation history — 22 of them in
one ordinary development session (20 empty "New conversation", a
`test-rollback-conv` from test_cancel.py, and a 24-message `__claude-test__`),
which is what made a manual DB sweep necessary.

These tests are the tripwire. If someone reintroduces a hardcoded home path, or
moves the `MIRA_DATA_DIR` setup in conftest.py below the first project import,
this fails instead of quietly polluting real data again.
"""
from pathlib import Path

from core import config
from core import db


REAL_DATA_DIR = Path.home() / ".local" / "share" / "mira"


def _under(path, parent) -> bool:
    return parent.resolve() in Path(path).resolve().parents


def test_data_dir_is_not_the_real_one():
    assert config.DATA_DIR.resolve() != REAL_DATA_DIR.resolve()


def test_every_persistent_path_is_redirected():
    """DB, RAG store, prompt cache and profile logs all hang off DATA_DIR, so one
    override covers them. Assert that rather than trusting it stays true."""
    from core import backend_manager as bm

    for name, path in (
        ("DB_PATH", config.DB_PATH),
        ("RAG_DIR", config.RAG_DIR),
        ("MIRA_MLX_CACHE_DIR", bm.MIRA_MLX_CACHE_DIR),
    ):
        assert _under(path, config.DATA_DIR), f"{name} escapes DATA_DIR: {path}"
        assert not _under(path, REAL_DATA_DIR), f"{name} points at real data: {path}"


def test_db_module_bound_the_redirected_path():
    """core.db does `from .config import DB_PATH`, a second binding made at import.
    If conftest set the env var too late, config would be right and db wrong —
    which is the failure mode that actually matters, since db is what writes."""
    assert db.DB_PATH == config.DB_PATH
    assert not _under(db.DB_PATH, REAL_DATA_DIR)


def test_writes_land_in_the_temp_dir():
    """End to end: create a conversation, then prove the bytes are in the tmp file
    and that the real database was never opened."""
    conv_id = db.create_conversation("isolation-probe")
    try:
        assert db.DB_PATH.exists(), "no database file where we redirected it"
        with db._conn() as conn:
            row = conn.execute(
                "SELECT model_name FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
        assert row["model_name"] == "isolation-probe"
    finally:
        db.delete_conversation(conv_id)


def test_no_test_only_marker_reached_the_real_database():
    """Direct check on the real file, keyed to values ONLY the suite produces.

    Deliberately not keyed to conversation *ids*: `__claude-test__` is also used
    for live curl smoke tests against the running server, which write to the real
    database legitimately, so asserting on it made this fail for the wrong reason.
    `model_name` is the reliable marker — the fixtures here pass literals no real
    turn ever uses, since a real one records an mlx-community repo id.
    """
    real_db = REAL_DATA_DIR / "conversations.db"
    if not real_db.exists():
        return  # CI, or a machine that has never run Mira
    import sqlite3

    conn = sqlite3.connect(f"file:{real_db}?mode=ro", uri=True)
    try:
        models = {r[0] for r in conn.execute("SELECT model_name FROM conversations")}
    finally:
        conn.close()
    for marker in ("isolation-probe", "test-model"):
        assert marker not in models, f"a test wrote model_name={marker!r} to the real DB"
