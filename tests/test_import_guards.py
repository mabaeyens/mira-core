"""Every test importing mlx must skip cleanly on Linux CI.

mlx is macOS/Apple-Silicon only. A test module that imports `mlx`, `mlx_lm` or
`core.inference.*` at module level without a `pytest.importorskip` above that
import raises ModuleNotFoundError during *collection*, and a collection error
aborts the entire run — so one unguarded file turns a green suite into
"6 skipped, 1 error" and nothing else executes.

This has now happened twice (2026-08-08, tests/test_cache_miss_detail.py). It is
invisible locally, because on this machine mlx imports fine. Checking it here
means the rule fails in a readable way on the developer's own machine instead of
silently on a Linux runner.

Pure file parsing: no mlx required, so this module itself always runs.
"""
import re
from pathlib import Path

TESTS = Path(__file__).parent

# Module-level import of something that only exists with mlx installed.
NEEDS_MLX = re.compile(r"^\s*(?:from|import)\s+(?:mlx\b|mlx_lm\b|core\.inference\b)", re.M)
GUARD = re.compile(r"importorskip\(")


def test_every_mlx_importing_test_is_guarded():
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        src = path.read_text()
        first_import = NEEDS_MLX.search(src)
        if first_import is None:
            continue
        guard = GUARD.search(src)
        if guard is None or guard.start() > first_import.start():
            offenders.append(path.name)

    assert not offenders, (
        "these test modules import mlx (directly or via core.inference) with no "
        "pytest.importorskip above the import, so they will abort the whole Linux "
        f"CI run at collection: {offenders}"
    )
