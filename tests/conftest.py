import os
import sys
from pathlib import Path

import pytest

# Mark the process as a test run BEFORE any project module is imported. Modules
# check this to skip background warm-up threads (reranker prefetch, backend
# auto-start) that otherwise linger past interpreter shutdown — on CI those
# threads cause a SIGABRT at exit and noisy unhandled-thread warnings.
os.environ.setdefault("MIRA_TESTING", "1")

# Allow imports from the project root regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _no_local_auth_token(monkeypatch):
    """Tests must not depend on the developer's local `mira.yaml` secret. Default every
    test to the open/loopback posture (no token, no source-IP gate); the auth tests set
    a token explicitly where they need one."""
    import server
    monkeypatch.setattr(server, "AUTH_TOKEN", "", raising=False)
