import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Mark the process as a test run BEFORE any project module is imported. Modules
# check this to skip background warm-up threads (reranker prefetch, backend
# auto-start) that otherwise linger past interpreter shutdown — on CI those
# threads cause a SIGABRT at exit and noisy unhandled-thread warnings.
os.environ.setdefault("MIRA_TESTING", "1")

# Point every persistent store at a throwaway directory, for the same reason and
# with the same timing constraint: `core.config` computes DB_PATH at import and
# `core.db` re-binds it with `from .config import DB_PATH`, so a fixture would be
# far too late. Without this the suite writes to the real
# ~/.local/share/mira/conversations.db — one ordinary session left 22 rows in
# Miguel's own conversation history (2026-08-01). DATA_DIR also carries RAG_DIR,
# the mira-mlx prompt cache and the expert-profile logs, so all four are covered.
# setdefault, not assignment: an explicit MIRA_DATA_DIR from the caller wins.
if "MIRA_DATA_DIR" not in os.environ:
    _TEST_DATA_DIR = tempfile.mkdtemp(prefix="mira_test_data_")
    os.environ["MIRA_DATA_DIR"] = _TEST_DATA_DIR
    atexit.register(shutil.rmtree, _TEST_DATA_DIR, True)

# Neutralize the developer's local mira.yaml, same principle as the auth-token
# fixture below and with the same import-time constraint: core.config computes
# _cfg from mira.yaml at import, so a test that asserts the *unconfigured* posture
# (e.g. test_generation_guard's "nothing is sent when nothing is configured")
# fails locally the moment mira.yaml sets repetition_penalty/max_thinking_tokens,
# while passing on CI where no mira.yaml exists. MIRA_CONFIG points the loader at a
# file that yields an empty config, so the whole suite sees code-defaults — which
# by design match mira.yaml.example. setdefault, not assignment: the bench sets an
# explicit MIRA_CONFIG and must still win.
os.environ.setdefault("MIRA_CONFIG", "/dev/null")

# Allow imports from the project root regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create the schema, once, here — and note this import sits BELOW the
# MIRA_DATA_DIR block above for exactly the reason that block explains. Move it
# up and `core.config` computes DB_PATH against the real home directory before
# the override lands, which puts the suite back to writing Miguel's own
# conversation history.
#
# Why it is needed at all: `db.init_db()` is called from exactly one place in
# production, server.py's `lifespan` handler. Nothing in the harness calls it.
# So the tables existed during a test run only when some earlier test happened
# to stand up a TestClient first — tests/test_auth.py sorts early and does. The
# full suite was green by alphabetical accident, while
# `pytest tests/test_tools_enabled.py` on its own failed with
# "no such table: memories". init_db() is CREATE TABLE IF NOT EXISTS throughout,
# so calling it here is idempotent and leaves server.py's own call untouched.
from core import db as _db  # noqa: E402

_db.init_db()


@pytest.fixture(autouse=True)
def _no_local_auth_token(monkeypatch):
    """Tests must not depend on the developer's local `mira.yaml` secret. Default every
    test to the open/loopback posture (no token, no source-IP gate); the auth tests set
    a token explicitly where they need one."""
    import server
    monkeypatch.setattr(server, "AUTH_TOKEN", "", raising=False)
