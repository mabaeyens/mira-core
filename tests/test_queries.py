"""Tests for the ollama Search Tool."""

import pytest
from unittest.mock import MagicMock, patch
from orchestrator import ChatOrchestrator


# --- Mock factories ---

def _make_chunk(content="", tool_calls=None, done=True):
    """Create a single mock Ollama stream chunk."""
    chunk = MagicMock()
    chunk.message.content = content
    chunk.message.tool_calls = tool_calls
    chunk.done = done
    return chunk


def _tool_call_stream(query="test"):
    """Mock _call_ollama return for a web_search tool call."""
    tool_call = MagicMock()
    tool_call.function.name = "web_search"
    tool_call.function.arguments = {"query": query, "num_results": 5}
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
