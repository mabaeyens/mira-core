"""Tests for the ollama Search Tool."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from orchestrator import ChatOrchestrator


# --- Mock factories ---

def _make_chunk(content="", tool_calls=None, done=True):
    """Create a single mock Ollama stream chunk."""
    chunk = MagicMock()
    chunk.message.content = content
    chunk.message.tool_calls = tool_calls
    chunk.message.thinking = ""
    chunk.done = done
    return chunk


def _tool_call_stream(query="test"):
    """Mock _call_ollama return for a web_search tool call."""
    tool_call = MagicMock()
    tool_call.function.name = "web_search"
    tool_call.function.arguments = {"query": query, "num_results": 5}
    return iter([_make_chunk(content="", tool_calls=[tool_call], done=True)])


def _fetch_url_stream(url="https://example.com"):
    """Mock _call_ollama return for a fetch_url tool call."""
    tool_call = MagicMock()
    tool_call.function.name = "fetch_url"
    tool_call.function.arguments = {"url": url}
    return iter([_make_chunk(content="", tool_calls=[tool_call], done=True)])


def _final_stream(content="Answer."):
    """Mock _call_ollama return for a final answer."""
    return iter([_make_chunk(content=content, tool_calls=None, done=True)])


FAKE_SEARCH_RESULTS = [
    {"title": "Test Result", "url": "http://example.com", "snippet": "Some content."}
]


# --- Helpers ---

def _consume(gen):
    """Consume a stream_chat generator. Returns (events, final_content)."""
    events = list(gen)
    done_events = [e for e in events if e["type"] == "done"]
    content = done_events[0]["content"] if done_events else ""
    return events, content


# --- Fixtures ---

@pytest.fixture
def orchestrator():
    return ChatOrchestrator(verbose=False)


# --- Tests ---

def test_historical_fact_no_search(orchestrator):
    """Historical facts should be answered without triggering a search."""
    with patch.object(orchestrator, '_call_ollama', return_value=_final_stream("William Shakespeare wrote Hamlet.")), \
         patch.object(orchestrator.search_engine, 'search') as mock_search:
        events, content = _consume(orchestrator.stream_chat("Who wrote Hamlet?"))
        mock_search.assert_not_called()
        assert content == "William Shakespeare wrote Hamlet."


def test_current_event_triggers_search(orchestrator):
    """Events after the knowledge cutoff should trigger a web search."""
    with patch.object(orchestrator, '_call_ollama', side_effect=[
        _tool_call_stream("2026 Super Bowl"),
        _final_stream("The Eagles won the 2026 Super Bowl."),
    ]), patch.object(orchestrator.search_engine, 'search', return_value=FAKE_SEARCH_RESULTS) as mock_search:
        events, content = _consume(orchestrator.stream_chat("What happened in the 2026 Super Bowl?"))
        mock_search.assert_called_once()
        assert content == "The Eagles won the 2026 Super Bowl."


def test_weather_query_triggers_search(orchestrator):
    """Time-sensitive queries should trigger a web search."""
    with patch.object(orchestrator, '_call_ollama', side_effect=[
        _tool_call_stream("weather today"),
        _final_stream("It is sunny today."),
    ]), patch.object(orchestrator.search_engine, 'search', return_value=FAKE_SEARCH_RESULTS) as mock_search:
        events, content = _consume(orchestrator.stream_chat("What's the weather like today?"))
        mock_search.assert_called_once()
        assert len(content) > 0


def test_general_knowledge_no_search(orchestrator):
    """General knowledge questions should not trigger a search."""
    with patch.object(orchestrator, '_call_ollama', return_value=_final_stream("Photosynthesis is the process...")), \
         patch.object(orchestrator.search_engine, 'search') as mock_search:
        events, content = _consume(orchestrator.stream_chat("Explain what photosynthesis is"))
        mock_search.assert_not_called()
        assert len(content) > 0


def test_search_done_event_contains_results(orchestrator):
    """search_done event should carry result count and results list."""
    with patch.object(orchestrator, '_call_ollama', side_effect=[
        _tool_call_stream("test query"),
        _final_stream("Done."),
    ]), patch.object(orchestrator.search_engine, 'search', return_value=FAKE_SEARCH_RESULTS):
        events, _ = _consume(orchestrator.stream_chat("test"))
        search_done = next(e for e in events if e["type"] == "search_done")
        assert search_done["count"] == 1
        assert search_done["results"] == FAKE_SEARCH_RESULTS


def test_toggle_verbose(orchestrator):
    """Verbose toggle should flip the flag each call."""
    assert orchestrator.verbose is False
    orchestrator.toggle_verbose()
    assert orchestrator.verbose is True
    orchestrator.toggle_verbose()
    assert orchestrator.verbose is False


def test_reset_conversation(orchestrator):
    """Reset should restore conversation history to system prompt only."""
    orchestrator.conversation_history.append({"role": "user", "content": "test"})
    orchestrator.conversation_history.append({"role": "assistant", "content": "test"})
    orchestrator.reset_conversation()
    assert len(orchestrator.conversation_history) == 1
    assert orchestrator.conversation_history[0]["role"] == "system"


def test_fetch_url_tool_dispatch(orchestrator):
    """fetch_url tool call should yield fetch_start/fetch_done events and pass page content to model."""
    fake_page = "Full page content from example.com"
    with patch.object(orchestrator, '_call_ollama', side_effect=[
        _fetch_url_stream("https://example.com"),
        _final_stream("The page says: Full page content."),
    ]), patch('url_fetcher.fetch_url', return_value=fake_page) as mock_fetch:
        events, content = _consume(orchestrator.stream_chat("What does example.com say?"))

        mock_fetch.assert_called_once_with("https://example.com")

        types = [e["type"] for e in events]
        assert "fetch_start" in types
        assert "fetch_done" in types

        fetch_start = next(e for e in events if e["type"] == "fetch_start")
        assert fetch_start["url"] == "https://example.com"

        fetch_done = next(e for e in events if e["type"] == "fetch_done")
        assert fetch_done["chars"] == len(fake_page)

        # Page content must be in conversation history so the model can use it
        tool_messages = [m for m in orchestrator.conversation_history if m.get("role") == "tool"]
        assert any(fake_page in m["content"] for m in tool_messages)


def test_accumulated_tool_calls_intermediate_chunk(orchestrator):
    """Tool calls emitted in a non-final chunk (Gemma4 behaviour) must still be executed.

    Gemma4:26b sends tool_calls in an intermediate chunk (done=False) and an
    empty done=True chunk. The old code only checked the final chunk — silently
    dropping the tool call and returning an empty response.
    """
    tool_call = MagicMock()
    tool_call.function.name = "web_search"
    tool_call.function.arguments = {"query": "Gemma4 test", "num_results": 5}

    # Intermediate chunk carries tool_calls; final chunk has none
    intermediate = _make_chunk(content="", tool_calls=[tool_call], done=False)
    final = _make_chunk(content="", tool_calls=None, done=True)
    # model_copy must return a message that carries the patched tool_calls
    patched = MagicMock()
    patched.tool_calls = [tool_call]
    final.message.model_copy = MagicMock(return_value=patched)

    with patch.object(orchestrator, '_call_ollama', side_effect=[
        iter([intermediate, final]),
        _final_stream("Found it."),
    ]), patch.object(orchestrator.search_engine, 'search', return_value=FAKE_SEARCH_RESULTS):
        events, content = _consume(orchestrator.stream_chat("Gemma4 tool call test"))
        assert any(e["type"] == "search_start" for e in events), \
            "Tool call from intermediate chunk was dropped — accumulated_tool_calls fix not working"
        assert content == "Found it."


def test_fetch_context_event_emitted_after_fetch_url(orchestrator):
    """fetch_context event must be emitted before done when fetch_url was called.

    Payload must include url, chars, and a 300-char preview for each fetch so
    the UI can render a collapsible 'pages read' panel below the answer.
    """
    fake_page = "A" * 500  # longer than 300 to verify truncation

    with patch.object(orchestrator, '_call_ollama', side_effect=[
        _fetch_url_stream("https://example.com/article"),
        _final_stream("The article says something."),
    ]), patch('url_fetcher.fetch_url', return_value=fake_page):
        events, _ = _consume(orchestrator.stream_chat("What does example.com say?"))

    ctx = next((e for e in events if e["type"] == "fetch_context"), None)
    assert ctx is not None, "fetch_context event not emitted"

    assert len(ctx["fetches"]) == 1
    fetch = ctx["fetches"][0]
    assert fetch["url"] == "https://example.com/article"
    assert fetch["chars"] == 500
    assert fetch["preview"].endswith("…")
    assert len(fetch["preview"]) <= 304  # 300 chars + "…"

    # fetch_context must come before done
    types = [e["type"] for e in events]
    assert types.index("fetch_context") < types.index("done")


def test_fetch_context_not_emitted_without_fetch(orchestrator):
    """fetch_context must not be emitted when no fetch_url tool call was made."""
    with patch.object(orchestrator, '_call_ollama', return_value=_final_stream("Plain answer.")):
        events, _ = _consume(orchestrator.stream_chat("What is 2+2?"))

    assert not any(e["type"] == "fetch_context" for e in events)


def test_fetch_context_multiple_fetches(orchestrator):
    """fetch_context must accumulate all fetch_url calls made in a single turn."""
    fake_page = "Page content here."

    # Two sequential fetch_url calls before the final answer
    fetch1 = MagicMock()
    fetch1.function.name = "fetch_url"
    fetch1.function.arguments = {"url": "https://alpha.com"}
    fetch2 = MagicMock()
    fetch2.function.name = "fetch_url"
    fetch2.function.arguments = {"url": "https://beta.com"}

    with patch.object(orchestrator, '_call_ollama', side_effect=[
        iter([_make_chunk(content="", tool_calls=[fetch1], done=True)]),
        iter([_make_chunk(content="", tool_calls=[fetch2], done=True)]),
        _final_stream("Done."),
    ]), patch('url_fetcher.fetch_url', return_value=fake_page):
        events, _ = _consume(orchestrator.stream_chat("Compare alpha and beta"))

    ctx = next((e for e in events if e["type"] == "fetch_context"), None)
    assert ctx is not None
    assert len(ctx["fetches"]) == 2
    urls = [f["url"] for f in ctx["fetches"]]
    assert "https://alpha.com" in urls
    assert "https://beta.com" in urls


def test_rag_context_event_emitted_when_chunks_retrieved(orchestrator):
    """rag_context event must be yielded before done when RAG chunks are used.

    Payload must include source, score, and preview for each chunk so the UI
    can render a collapsible 'document sections used' panel below the answer.
    """
    fake_chunks = [
        {"source": "report.pdf", "score": 1.42, "text": "Quarterly revenue grew by 12%."},
        {"source": "report.pdf", "score": 0.87, "text": "Operating costs remained stable."},
    ]

    with patch.object(orchestrator, '_call_ollama', return_value=_final_stream("Revenue grew.")), \
         patch.object(orchestrator.rag_engine, 'query', return_value=fake_chunks), \
         patch.object(type(orchestrator.rag_engine), 'chunk_count',
                      new_callable=PropertyMock, return_value=5):
        events, content = _consume(orchestrator.stream_chat("What was the revenue growth?"))

    rag_ctx = next((e for e in events if e["type"] == "rag_context"), None)
    assert rag_ctx is not None, "rag_context event not emitted"

    assert len(rag_ctx["chunks"]) == 2
    first = rag_ctx["chunks"][0]
    assert first["source"] == "report.pdf"
    assert first["score"] == 1.42
    assert "Quarterly revenue" in first["preview"]

    # rag_context must come before done
    types = [e["type"] for e in events]
    assert types.index("rag_context") < types.index("done")


def test_rag_context_not_emitted_when_no_chunks(orchestrator):
    """rag_context must not be emitted when RAG returns no chunks (empty index or filtered out)."""
    with patch.object(orchestrator, '_call_ollama', return_value=_final_stream("Plain answer.")), \
         patch.object(orchestrator.rag_engine, 'query', return_value=[]), \
         patch.object(type(orchestrator.rag_engine), 'chunk_count',
                      new_callable=PropertyMock, return_value=0):
        events, _ = _consume(orchestrator.stream_chat("What is 2+2?"))

    assert not any(e["type"] == "rag_context" for e in events)


def test_rag_score_threshold_bypassed_for_same_turn_attachment(orchestrator):
    """When a RAG file is indexed in the same turn, query() must use score_threshold=-inf.

    Meta-instructions like 'summarize this' embed nothing like document content,
    so normal threshold filtering drops all chunks and the model hallucinates
    that no file was attached.
    """
    rag_attachment = [{
        "type": "rag",
        "name": "book.pdf",
        "content": "Chapter one. " * 200,  # enough for at least one chunk
        "warning": None,
    }]

    with patch.object(orchestrator, '_call_ollama', return_value=_final_stream("Here is the summary.")), \
         patch.object(orchestrator.rag_engine, 'query', return_value=[]) as mock_query, \
         patch.object(orchestrator.rag_engine, 'index', return_value=3), \
         patch.object(type(orchestrator.rag_engine), 'chunk_count', new_callable=PropertyMock, return_value=3):
        _consume(orchestrator.stream_chat("Summarize this document", attachments=rag_attachment))

        mock_query.assert_called_once()
        _, kwargs = mock_query.call_args
        assert kwargs.get("score_threshold") == float('-inf'), \
            "score_threshold must be -inf when RAG files indexed this turn"


# ── Stats event tests ────────────────────────────────────────────────────────

def test_stats_event_emitted_before_done(orchestrator):
    """stats event must be emitted before done on every successful turn."""
    with patch.object(orchestrator, '_call_ollama', return_value=_final_stream("Answer.")):
        events, _ = _consume(orchestrator.stream_chat("Hello"))

    stats_events = [e for e in events if e["type"] == "stats"]
    assert len(stats_events) == 1, "exactly one stats event per turn"

    stats = stats_events[0]
    assert "input_tokens" in stats
    assert "output_tokens" in stats
    assert "context_pct" in stats
    assert isinstance(stats["input_tokens"], int)
    assert isinstance(stats["output_tokens"], int)
    assert isinstance(stats["context_pct"], int)

    types = [e["type"] for e in events]
    assert types.index("stats") < types.index("done"), "stats must precede done"


def test_stats_context_pct_bounded(orchestrator):
    """context_pct must stay in [0, 100] regardless of token counts."""
    # Simulate a very large prompt (bypasses isinstance check on mock)
    orchestrator.last_prompt_tokens = 200_000  # way over 64k
    assert orchestrator.context_pct == 100


def test_stats_reset_clears_token_counts(orchestrator):
    """reset_conversation must zero out all token counters."""
    orchestrator.total_input_tokens = 5000
    orchestrator.total_output_tokens = 1000
    orchestrator.last_prompt_tokens = 4500
    orchestrator.reset_conversation()
    assert orchestrator.total_input_tokens == 0
    assert orchestrator.total_output_tokens == 0
    assert orchestrator.last_prompt_tokens == 0
    assert orchestrator.context_pct == 0


def test_stats_token_capture_with_real_counts(orchestrator):
    """When a done chunk provides int token counts they must be tracked."""
    chunk = _make_chunk(content="Hi", tool_calls=None, done=True)
    chunk.prompt_eval_count = 1024
    chunk.eval_count = 32

    with patch.object(orchestrator, '_call_ollama', return_value=iter([chunk])):
        _consume(orchestrator.stream_chat("Hello"))

    assert orchestrator.last_prompt_tokens == 1024
    assert orchestrator.total_input_tokens == 1024
    assert orchestrator.total_output_tokens == 32
    assert orchestrator.context_pct == round(1024 / 65536 * 100)
