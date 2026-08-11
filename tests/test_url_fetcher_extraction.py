"""What counts as "the page came back" for fetch_url.

Both cases here came out of the 2026-08-11 conversation corpus, where Mira was
asked to fetch a page, got 969 characters of CSS and JavaScript, could not find
what it needed, and answered by looping on web_search — 36 searches across the
run. Nothing failed loudly: the fetch reported success and the model was handed
the shell of a page as if it were the page.
"""
from unittest.mock import patch

from core import url_fetcher as uf
from core.url_fetcher import _extract, fetch_url

# The shape of developer.apple.com: a small HTML shell whose only text is inline
# script and style, with the real content loaded client-side. Trimmed, but the
# `var baseUrl` and `.noscript{...}` are the actual strings that reached Mira.
JS_SHELL = """<!DOCTYPE html><html><head>
<title>Notifications | Apple Developer Documentation</title>
<script>var baseUrl = "/tutorials/"; var s_account="awdappledeveloper";
function init(){ %s }</script>
<style>.noscript{font-family:"SF Pro Display","SF Pro Icons",Helvetica,Arial;
color:#333;background:#fff;margin:0;padding:0} %s</style>
</head><body><div id="app"></div></body></html>""" % ("x=1;" * 400, "a{b:c}" * 400)

REAL_PAGE = """<!DOCTYPE html><html><head><title>Synthetic keys</title></head>
<body><article><h1>Synthetic keys</h1>
<p>%s</p></article></body></html>""" % ("Synthetic keys are created when two or "
                                        "more tables share more than one field. " * 40)


def test_script_and_style_are_not_page_content():
    """The defect. markdownify's `strip=` skips the tag but still walks its
    children, so everything inside <script> and <style> arrived as body text."""
    out = _extract(JS_SHELL)

    assert "var baseUrl" not in out, "JavaScript came back as page content"
    assert "font-family" not in out, "CSS came back as page content"
    assert "s_account" not in out


def test_a_js_shell_reads_as_empty_rather_than_short():
    """Why the above matters: 969 chars of script cleared fetch_url's 200-char
    "is there anything here?" floor, so the fallback that would have fetched the
    real page never ran. Stripped of script and style there is nothing left, and
    nothing is what the caller needs to see."""
    assert len(_extract(JS_SHELL)) < 200


def test_real_content_survives():
    out = _extract(REAL_PAGE)

    assert "Synthetic keys are created" in out
    assert len(out) > 1_000


def test_the_fallback_now_fires_for_a_small_js_shell():
    """End to end. The old rule required more than 50 KB of HTML before it would
    suspect a JS-rendered page; Apple's is 17 KB, so it never qualified and the
    model got the shell. Both halves of the fix are needed for this to pass."""
    with patch.object(uf, "_fetch_html", return_value=(JS_SHELL, None)), \
         patch.object(uf, "_fetch_jina", return_value="The real page. " * 200) as jina:
        out = fetch_url("https://developer.apple.com/design/human-interface-guidelines/notifications")

    assert jina.called, "never tried the fallback; the model gets the shell"
    assert "The real page." in out


def test_a_short_but_genuine_page_does_not_trigger_the_fallback():
    """The size floor earns its place here: on a small page a low text-to-HTML
    ratio means nothing, and paying for a network round trip to re-fetch a page
    that arrived intact is a cost with no benefit."""
    short = ("<html><body><article><p>" + ("Yes, that is correct. " * 30)
             + "</p></article></body></html>")
    with patch.object(uf, "_fetch_html", return_value=(short, None)), \
         patch.object(uf, "_fetch_jina", return_value="should not be used") as jina:
        out = fetch_url("https://example.com/short")

    assert not jina.called
    assert "Yes, that is correct." in out


def test_the_page_that_already_worked_still_works():
    """Regression guard on the ratio change: a large HTML page that extracts
    properly must not start going through the fallback."""
    with patch.object(uf, "_fetch_html", return_value=(REAL_PAGE * 30, None)), \
         patch.object(uf, "_fetch_jina", return_value="should not be used") as jina:
        out = fetch_url("https://help.qlik.com/synthetic-keys.htm")

    assert not jina.called
    assert "Synthetic keys are created" in out


def test_content_is_capped_and_says_so():
    """Explains the 24,038 that showed up on three unrelated URLs in the corpus
    and briefly looked like a bug: it is this cap plus the notice, nothing more."""
    huge = "<html><body><article><p>" + ("word " * 40_000) + "</p></article></body></html>"
    with patch.object(uf, "_fetch_html", return_value=(huge, None)):
        out = fetch_url("https://example.com/huge")

    assert len(out) == uf.MAX_CONTENT_CHARS + len(
        f"\n\n[… content truncated at {uf.MAX_CONTENT_CHARS} chars]")
    assert "content truncated" in out
