"""Backend wire-format adapters.

Pure helpers that translate between Mira's stored conversation history and the
two LLM wire formats it speaks (Ollama's Pydantic chat API and the
OpenAI-compatible streaming API used by mlx-lm / dflash / omlx):

  * message normalizers — reshape history for each backend's quirks;
  * :func:`normalize_oai_stream` — adapt an OpenAI-compatible stream into the
    Ollama-style chunk objects the orchestrator's streaming loop expects;
  * :func:`message_to_history_dict` — fold a completed message back into history.

None of these touch the orchestrator, network clients, or model selection — the
backend call itself stays in ``ChatOrchestrator._call_llm`` (the test mock seam).
That keeps every byte-shaping rule here, where it can be tested directly.
"""

import json
import re
import types
from typing import List, Dict


def inject_no_think(messages: List[Dict]) -> List[Dict]:
    """Prepend /no_think to the system message so Qwen3 skips chain-of-thought
    regardless of Ollama version. Safe to call on any message list."""
    result = list(messages)
    if result and result[0].get("role") == "system":
        content = result[0].get("content", "")
        if not content.startswith("/no_think"):
            result[0] = {**result[0], "content": "/no_think\n" + content}
    return result


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from a string."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def normalize_messages_for_ollama(messages: List[Dict]) -> List[Dict]:
    """Ollama's Pydantic Message model requires tool_calls[].function.arguments as dict.
    History stores them as JSON strings (OpenAI wire format). Parse them back."""
    result = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            msg = dict(msg)
            tcs = []
            for tc in msg["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                tc["function"] = fn
                tcs.append(tc)
            msg["tool_calls"] = tcs
        result.append(msg)
    return result


def normalize_messages_for_oai(messages: List[Dict]) -> List[Dict]:
    """Convert history messages to OpenAI-compat multimodal format.
    User messages with an 'images' key are rewritten so that content becomes
    a list of content parts: [{type:text,...}, {type:image_url,...}, ...]."""
    result = []
    for msg in messages:
        if msg.get("role") == "user" and msg.get("images"):
            msg = dict(msg)
            images = msg.pop("images")
            parts: List[Dict] = [{"type": "text", "text": msg["content"]}]
            for b64 in images:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            msg["content"] = parts
        result.append(msg)
    return result


def normalize_oai_stream(stream):
    """Yield Ollama-compatible chunk objects from an OpenAI-compatible stream."""
    acc_args: dict[int, str] = {}
    acc_calls: dict[int, dict] = {}
    last_usage = None
    pending_done: object = None

    for chunk in stream:
        if hasattr(chunk, 'usage') and chunk.usage:
            last_usage = chunk.usage

        if not chunk.choices:
            if pending_done is not None:
                if last_usage:
                    pending_done.prompt_eval_count = getattr(last_usage, 'prompt_tokens', 0) or 0
                    pending_done.eval_count = getattr(last_usage, 'completion_tokens', 0) or 0
                yield pending_done
                pending_done = None
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        finish_reason = choice.finish_reason

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in acc_calls:
                    acc_calls[idx] = {"id": tc.id or f"call_{idx}", "name": ""}
                    acc_args[idx] = ""
                if tc.function:
                    if tc.function.name:
                        acc_calls[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        acc_args[idx] += tc.function.arguments

        content = delta.content or ""
        # mlx_lm.server and dflash serve stream chain-of-thought in a separate field on
        # the delta. The field name may be `reasoning` or `reasoning_content` depending
        # on the server version. The OpenAI SDK keeps non-standard fields on model_extra
        # and also exposes them as attributes — read both to be robust.
        _extra = getattr(delta, "model_extra", None) or {}
        reasoning = getattr(delta, "reasoning", None) \
            or _extra.get("reasoning") \
            or _extra.get("reasoning_content") \
            or ""
        is_done = finish_reason is not None

        msg = types.SimpleNamespace(content=content, tool_calls=None, thinking=reasoning)
        fake = types.SimpleNamespace(
            message=msg, done=is_done, prompt_eval_count=0, eval_count=0
        )

        if is_done and acc_calls:
            tool_calls = []
            for idx in sorted(acc_calls.keys()):
                try:
                    args_dict = json.loads(acc_args[idx] or "{}")
                except json.JSONDecodeError:
                    args_dict = {}
                fn = types.SimpleNamespace(
                    name=acc_calls[idx]["name"],
                    arguments=args_dict,
                )
                tool_calls.append(types.SimpleNamespace(
                    id=acc_calls[idx]["id"],
                    function=fn,
                ))
            msg.tool_calls = tool_calls

        if is_done:
            if last_usage:
                fake.prompt_eval_count = getattr(last_usage, 'prompt_tokens', 0) or 0
                fake.eval_count = getattr(last_usage, 'completion_tokens', 0) or 0
                yield fake
            else:
                pending_done = fake
        else:
            yield fake

    if pending_done is not None:
        if last_usage:
            pending_done.prompt_eval_count = getattr(last_usage, 'prompt_tokens', 0) or 0
            pending_done.eval_count = getattr(last_usage, 'completion_tokens', 0) or 0
        yield pending_done


def message_to_history_dict(msg) -> dict:
    """Convert an Ollama Message object or SimpleNamespace to a history-compatible dict."""
    if isinstance(msg, dict):
        return msg

    d: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = []
        for i, tc in enumerate(msg.tool_calls):
            args = tc.function.arguments
            if isinstance(args, dict):
                args_str = json.dumps(args)
            else:
                args_str = str(args)
            d["tool_calls"].append({
                "id": getattr(tc, 'id', None) or f"call_{i}",
                "type": "function",
                "function": {"name": tc.function.name, "arguments": args_str},
            })
    return d
