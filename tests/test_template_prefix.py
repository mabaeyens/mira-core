"""Whether step N's prompt is a prefix of step N+1's, per conversation shape.

This is the single precondition for prompt-cache reuse on Qwen3.6. Trimming a
cache back to a divergence point is impossible for this model — its linear
attention layers keep a Gated DeltaNet recurrent state and a sliding conv window
(`ArraysCache`, see test_cache_trimmability.py), neither of which has per-token
slots to drop — so an entry is reusable only if it is a *whole* prefix.

Measured 2026-08-08: agentic tool steps satisfy that and plain chat turns do not,
which is why the same bench run shows `HIT 4,119/4,206` on tool loops and
`MISS 0/27,621` on a two-turn chat. The cause is that the generation prompt ends
with a thinking scaffold that the template never re-emits when it replays a plain
assistant message.

Skipped unless the model's tokenizer is already in the local HF cache; this must
never trigger a download.
"""
import pytest

hub = pytest.importorskip("huggingface_hub")
pytest.importorskip("transformers")

from pathlib import Path  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"


@pytest.fixture(scope="module")
def tok():
    try:
        path = Path(hub.snapshot_download(
            MODEL, allow_patterns=["*.json", "*.txt", "*.model"], local_files_only=True))
    except Exception as exc:  # not cached, offline, or hub unavailable
        pytest.skip(f"{MODEL} tokenizer not in the local cache ({type(exc).__name__})")
    return AutoTokenizer.from_pretrained(path)


SYS = {"role": "system", "content": "You are Mira."}
U1 = {"role": "user", "content": "Summarise this file."}
U2 = {"role": "user", "content": "Now where is the guard?"}
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "read a file",
    "parameters": {"type": "object", "properties": {}}}}]


def _render(tok, msgs, tools=None, **kw):
    return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                   tokenize=False, **kw)


def test_an_agentic_tool_step_keeps_the_prefix(tok):
    """Why tool loops hit. The template replays an assistant message carrying
    tool_calls in a form that still contains the generation prompt's scaffold,
    so step 0's prompt survives intact inside step 1's."""
    first = _render(tok, [SYS, U1], TOOLS)
    nxt = _render(tok, [SYS, U1,
                        {"role": "assistant", "content": "Let me look.",
                         "tool_calls": [{"type": "function",
                                         "function": {"name": "read_file", "arguments": {}}}]},
                        {"role": "tool", "name": "read_file", "content": "body"}], TOOLS)
    assert nxt.startswith(first)


def test_a_plain_chat_turn_loses_the_prefix(tok):
    """Why Q10 re-prefilled 27,614 tokens. Turn 1's prompt ends with the thinking
    scaffold; replaying the assistant message emits the content in its place, so
    turn 1 is not contained in turn 2 and no reuse is possible."""
    first = _render(tok, [SYS, U1])
    nxt = _render(tok, [SYS, U1, {"role": "assistant", "content": "It defines x."}, U2])
    assert not nxt.startswith(first)
    assert first.endswith("<think>\n")


def test_no_thinking_flag_rescues_the_chat_case(tok):
    """The obvious fix does not work, and this pins that so it is not retried.
    enable_thinking=False only swaps one scaffold for another
    (`<think>\\n\\n</think>\\n\\n`); the template still never re-emits it on replay."""
    for kw in ({}, {"enable_thinking": False}, {"enable_thinking": True}):
        first = _render(tok, [SYS, U1], **kw)
        nxt = _render(tok, [SYS, U1,
                            {"role": "assistant", "content": "It defines x."}, U2], **kw)
        assert not nxt.startswith(first), f"prefix unexpectedly held for {kw}"


def test_the_assistant_header_boundary_is_a_compounding_prefix(tok):
    """The precondition for specs/assistant-boundary-snapshot.md.

    Rendering the history WITHOUT the generation prompt gives a sequence that is
    a prefix of this turn's prompt (so it can be produced by splitting prefill)
    and of every later turn's prompt (so the snapshot is reusable). Without both,
    the whole design is void.
    """
    def ids(msgs, gen):
        text = tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False)
        return tok.encode(text, add_special_tokens=False)

    # A realistic conversation: history dominates the new user turn. That ratio
    # is the whole point - coverage is 95% when the history is long and only
    # ~44% on a toy two-liner, so the fix pays off exactly on the expensive
    # turns and barely on the cheap ones.
    u1 = {"role": "user", "content": "Summarise this file.\n" + "filler " * 500}
    a1 = {"role": "assistant", "content": "It defines x."}
    a2 = {"role": "assistant", "content": "In orchestrator.py."}
    u3 = {"role": "user", "content": "Quote it."}

    boundary = ids([SYS, u1], False)
    turn1 = ids([SYS, u1], True)
    turn2 = ids([SYS, u1, a1, U2], True)
    turn3 = ids([SYS, u1, a1, U2, a2, u3], True)

    def is_prefix(a, b):
        return len(a) <= len(b) and list(a) == list(b[:len(a)])

    assert is_prefix(boundary, turn1), "cannot be produced by splitting this turn's prefill"
    assert is_prefix(boundary, turn2), "snapshot would not be reusable next turn"
    assert is_prefix(boundary, turn3), "reuse does not survive to turn 3"
    assert len(turn1) - len(boundary) < 20, "scaffold unexpectedly large; re-measure the design"

    # The win has to be worth the LRU slot it costs. Holds because history
    # dominates here; see the comment above for why that is the realistic case.
    assert len(boundary) / len(turn2) > 0.9


def test_supplying_the_think_block_in_content_does_not_help(tok):
    """The other obvious fix. Storing the assistant turn with its reasoning
    wrapped back in does not reproduce the scaffold either."""
    first = _render(tok, [SYS, U1])
    nxt = _render(tok, [SYS, U1,
                        {"role": "assistant",
                         "content": "<think>\nreasoning</think>\n\nIt defines x."}, U2])
    assert not nxt.startswith(first)
