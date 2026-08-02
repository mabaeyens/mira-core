"""Brave result formatting.

`_format_brave_results` was a plain function with a single `data` parameter,
sitting in the class body directly below `_clean_snippet`'s `@staticmethod`
decorator — which binds to `_clean_snippet`, not to it. Called as
`self._format_brave_results(resp.json())` that is two positional arguments into
a one-argument function, so every Brave search raised TypeError from June 2026
until 2026-08-02.

Nothing surfaced it because the caller catches broad `Exception` and logs
"Brave search failed... Falling back to DuckDuckGo", which is exactly what a
transient API error looks like. Searches kept working, just never via Brave.

There was no test file for this module at all. That is the actual bug.
"""
from core.search_engine import SearchEngine


BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {
                "title": "First result",
                "url": "https://example.com/one",
                "description": "A snippet   with\n\nragged   whitespace",
            },
            {
                "title": "Second result",
                "url": "https://example.com/two",
                "description": "Plain snippet",
            },
        ]
    }
}


def test_format_brave_results_is_callable_from_an_instance():
    """The regression itself: reachable the way the caller actually calls it."""
    engine = SearchEngine.__new__(SearchEngine)  # no network, no API key
    results = engine._format_brave_results(BRAVE_PAYLOAD)

    assert len(results) == 2
    assert results[0]["title"] == "First result"
    assert results[0]["url"] == "https://example.com/one"


def test_format_brave_results_normalizes_snippet_whitespace():
    results = SearchEngine._format_brave_results(BRAVE_PAYLOAD)
    assert results[0]["snippet"] == "A snippet with ragged whitespace"


def test_format_brave_results_tolerates_missing_fields():
    results = SearchEngine._format_brave_results({"web": {"results": [{}]}})
    assert results == [{"title": "No title", "url": "", "snippet": ""}]


def test_format_brave_results_handles_an_empty_payload():
    assert SearchEngine._format_brave_results({}) == []
