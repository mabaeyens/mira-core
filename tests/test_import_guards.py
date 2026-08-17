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

Escape hatch: NEEDS_MLX matches ``core.inference`` as a proxy for "needs mlx", but
a module under that package may keep every mlx import *lazy* (function-local, so
importing the module on Linux never pulls mlx). A test that imports such a module
runs its pure-logic assertions on CI and needs no importorskip. To attest that,
put an ``# import-guard: lazy-mlx`` marker on the flagged import line itself; the
guard then exempts that file. Use it ONLY for genuinely lazy-mlx modules — every
unmarked file keeps the strict rule (and its two real historical catches).

Pure file parsing: no mlx required, so this module itself always runs.
"""
import re
from pathlib import Path

TESTS = Path(__file__).parent

# Module-level import of something that only exists with mlx installed.
NEEDS_MLX = re.compile(r"^\s*(?:from|import)\s+(?:mlx\b|mlx_lm\b|core\.inference\b)", re.M)
GUARD = re.compile(r"importorskip\(")
# On the flagged import line: attests the imported module's mlx imports are lazy.
LAZY_MARKER = re.compile(r"#\s*import-guard:\s*lazy-mlx")


def test_every_mlx_importing_test_is_guarded():
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        src = path.read_text()
        first_import = NEEDS_MLX.search(src)
        if first_import is None:
            continue
        # A same-line marker attests the module is lazy-mlx (safe on Linux) → exempt.
        # NEEDS_MLX's leading ``\s*`` can swallow the preceding newline under re.M, so
        # anchor the import line to the match END (past the import keyword), not its start.
        line_start = src.rfind("\n", 0, first_import.end()) + 1
        line_end = src.find("\n", first_import.end())
        import_line = src[line_start: line_end if line_end != -1 else len(src)]
        if LAZY_MARKER.search(import_line):
            continue
        guard = GUARD.search(src)
        if guard is None or guard.start() > first_import.start():
            offenders.append(path.name)

    assert not offenders, (
        "these test modules import mlx (directly or via core.inference) with no "
        "pytest.importorskip above the import, so they will abort the whole Linux "
        f"CI run at collection: {offenders}"
    )
