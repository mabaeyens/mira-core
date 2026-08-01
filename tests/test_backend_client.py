"""Unit tests for the backend wire-format adapters.

These cover the pure normalizers and the OpenAI stream adapter directly,
without the orchestrator or a live backend. The stream adapter in particular was
previously only reachable by driving the whole agent loop.
"""
import json
import types

import pytest

from core import backend_client as bc


# -- inject_no_think -------------------------------------------------------

def test_inject_no_think_prepends_to_system():
    out = bc.inject_no_think([{"role": "system", "content": "Be helpful."}])
    assert out[0]["content"] == "/no_think\nBe helpful."


def test_inject_no_think_is_idempotent():
    once = bc.inject_no_think([{"role": "system", "content": "X"}])
    twice = bc.inject_no_think(once)
    assert twice[0]["content"] == "/no_think\nX"


def test_inject_no_think_noop_without_system():
    msgs = [{"role": "user", "content": "hi"}]
    assert bc.inject_no_think(msgs) == msgs


def test_inject_no_think_does_not_mutate_input():
    original = [{"role": "system", "content": "S"}]
    bc.inject_no_think(original)
    assert original[0]["content"] == "S"


# -- strip_think -----------------------------------------------------------

def test_strip_think_removes_block_and_trims():
    assert bc.strip_think("<think>reasoning</think>  Answer ") == "Answer"


def test_strip_think_handles_multiline():
    assert bc.strip_think("<think>a\nb\nc</think>Result") == "Result"


def test_strip_think_leaves_plain_text():
    assert bc.strip_think("just text") == "just text"


# -- normalize_messages_for_oai --------------------------------------------

def test_oai_rewrites_user_images_to_multimodal_parts():
    msgs = [{"role": "user", "content": "look", "images": ["BASE64"]}]
    out = bc.normalize_messages_for_oai(msgs)
    content = out[0]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,BASE64"
    assert "images" not in out[0]


def test_oai_leaves_text_only_messages():
    msgs = [{"role": "user", "content": "hi"}]
    assert bc.normalize_messages_for_oai(msgs) == msgs


def test_oai_does_not_mutate_input():
    msgs = [{"role": "user", "content": "look", "images": ["B"]}]
    bc.normalize_messages_for_oai(msgs)
    assert msgs[0]["images"] == ["B"]


# -- message_to_history_dict -----------------------------------------------

def test_history_dict_passthrough_for_dict():
    d = {"role": "assistant", "content": "x"}
    assert bc.message_to_history_dict(d) is d


def test_history_dict_serializes_tool_call_args():
    tc = types.SimpleNamespace(
        id="call_1",
        function=types.SimpleNamespace(name="search", arguments={"q": "hi"}),
    )
    msg = types.SimpleNamespace(content="", tool_calls=[tc])
    out = bc.message_to_history_dict(msg)
    assert out["tool_calls"][0]["function"]["arguments"] == '{"q": "hi"}'
    assert out["tool_calls"][0]["id"] == "call_1"
    assert out["tool_calls"][0]["type"] == "function"


def test_history_dict_none_content_becomes_empty():
    msg = types.SimpleNamespace(content=None, tool_calls=None)
    assert bc.message_to_history_dict(msg)["content"] == ""


# -- normalize_oai_stream --------------------------------------------------

def _delta(content=None, tool_calls=None, reasoning=None):
    d = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    # The adapter reads reasoning off both the attribute and model_extra.
    d.reasoning = reasoning
    d.model_extra = {}
    return d


def _chunk(delta=None, finish_reason=None, usage=None):
    choices = [] if delta is None else [
        types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
    ]
    return types.SimpleNamespace(choices=choices, usage=usage)


def _usage(prompt, completion):
    return types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def test_stream_content_then_done_with_usage():
    stream = [
        _chunk(_delta(content="Hello ")),
        _chunk(_delta(content="world"), finish_reason="stop"),
        _chunk(usage=_usage(12, 5)),  # trailing usage-only chunk
    ]
    out = list(bc.normalize_oai_stream(stream))
    texts = [c.message.content for c in out]
    assert "".join(texts) == "Hello world"
    final = out[-1]
    assert final.done is True
    assert final.prompt_eval_count == 12
    assert final.eval_count == 5


def test_stream_surfaces_reasoning_as_thinking():
    stream = [_chunk(_delta(reasoning="thinking aloud"))]
    out = list(bc.normalize_oai_stream(stream))
    assert out[0].message.thinking == "thinking aloud"


def test_stream_accumulates_split_tool_call():
    tc_a = types.SimpleNamespace(
        index=0, id="call_0",
        function=types.SimpleNamespace(name="sea", arguments='{"q":'),
    )
    tc_b = types.SimpleNamespace(
        index=0, id=None,
        function=types.SimpleNamespace(name="rch", arguments=' "hi"}'),
    )
    stream = [
        _chunk(_delta(tool_calls=[tc_a])),
        _chunk(_delta(tool_calls=[tc_b])),
        _chunk(_delta(content=""), finish_reason="tool_calls", usage=_usage(3, 4)),
    ]
    out = list(bc.normalize_oai_stream(stream))
    final = out[-1]
    assert final.done is True
    calls = final.message.tool_calls
    assert len(calls) == 1
    assert calls[0].function.name == "search"
    assert calls[0].function.arguments == {"q": "hi"}
    assert calls[0].id == "call_0"


# -- parse_xml_tool_calls (qwen3_coder XML, e.g. Nemotron-3) ----------------

def test_parse_xml_tool_calls_single():
    calls = bc.parse_xml_tool_calls(
        "</think>\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nMadrid\n</parameter>\n</function>\n</tool_call>"
    )
    assert len(calls) == 1
    assert calls[0].function.name == "get_weather"
    assert calls[0].function.arguments == {"city": "Madrid"}


def test_parse_xml_tool_calls_coerces_scalars_and_multi_param():
    calls = bc.parse_xml_tool_calls(
        "<tool_call><function=run_shell>"
        "<parameter=command>\nls -la\n</parameter>"
        "<parameter=timeout>\n30\n</parameter>"
        "</function></tool_call>"
    )
    assert calls[0].function.arguments == {"command": "ls -la", "timeout": 30}


def test_parse_xml_tool_calls_none_on_plain_text():
    assert bc.parse_xml_tool_calls("Just a normal answer, no tools.") is None


def test_stream_parses_xml_tool_call_from_content():
    # Nemotron-3 streams the tool call as XML text across content deltas.
    stream = [
        _chunk(_delta(content="<tool_call>\n<function=get_weather>\n")),
        _chunk(_delta(content="<parameter=city>\nMadrid\n</parameter>\n")),
        _chunk(_delta(content="</function>\n</tool_call>"), finish_reason="stop",
               usage=_usage(20, 9)),
    ]
    out = list(bc.normalize_oai_stream(stream))
    final = out[-1]
    assert final.done is True
    calls = final.message.tool_calls
    assert len(calls) == 1
    assert calls[0].function.name == "get_weather"
    assert calls[0].function.arguments == {"city": "Madrid"}


def test_stream_native_tool_calls_take_precedence_over_xml():
    # If a backend sends structured tool_calls, XML-looking content is ignored.
    tc = types.SimpleNamespace(
        index=0, id="call_0",
        function=types.SimpleNamespace(name="search", arguments='{"q": "hi"}'),
    )
    stream = [
        _chunk(_delta(content="<tool_call><function=ignored></function></tool_call>",
                      tool_calls=[tc]), finish_reason="tool_calls", usage=_usage(3, 4)),
    ]
    out = list(bc.normalize_oai_stream(stream))
    calls = out[-1].message.tool_calls
    assert len(calls) == 1
    assert calls[0].function.name == "search"
