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

# Allow imports from the project root regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _no_local_auth_token(monkeypatch):
    """Tests must not depend on the developer's local `mira.yaml` secret. Default every
    test to the open/loopback posture (no token, no source-IP gate); the auth tests set
    a token explicitly where they need one."""
    import server
    monkeypatch.setattr(server, "AUTH_TOKEN", "", raising=False)
