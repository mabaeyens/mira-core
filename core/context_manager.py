"""Context-window bookkeeping: history compression planning and token math.

Pure helpers extracted from the orchestrator so the decisions — what to
compress, how to rebuild history around a summary, how a prompt fills the
context window, how thinking characters map to tokens — can be tested without
the LLM or the orchestrator's mutable state. The summarization LLM call and the
running token counters stay on ``ChatOrchestrator``; only the stateless logic
lives here.
"""

from typing import List, Dict, Optional, Tuple

# Fallback for backends that don't report reasoning_tokens. Their eval_count covers
# only visible content tokens; thinking tokens arrive separately and are not counted,
# so we approximate them from character length and the displayed output-token total
# still reflects real compute. ~3.5 chars/token matches typical Qwen3/Gemma
# tokenization closely enough for a usage readout. mira-mlx counts reasoning tokens
# exactly (from its sequence state machine) and the orchestrator skips this estimate
# whenever a backend reports them — adding both would double-count.
_THINKING_CHARS_PER_TOKEN = 3.5

# Cap each compressed message's excerpt so a few huge messages can't blow up the
# summarization prompt itself.
_EXCERPT_CHARS_PER_MESSAGE = 2000


def thinking_tokens(chars: int) -> int:
    """Approximate token count for ``chars`` of thinking text."""
    return round(chars / _THINKING_CHARS_PER_TOKEN)


def context_pct(last_prompt_tokens: int, context_window: int) -> int:
    """Percentage of the context window filled by the last prompt (0–100)."""
    if not context_window or last_prompt_tokens == 0:
        return 0
    return min(100, round(last_prompt_tokens / context_window * 100))


def plan_compression(
    history: List[Dict], keep_recent: int
) -> Optional[Tuple[List[Dict], List[Dict]]]:
    """Split non-system history into (to_compress, to_keep), keeping the last
    ``keep_recent`` messages verbatim. Returns None when there is nothing old
    enough to compress."""
    non_system = [m for m in history if m["role"] != "system"]
    if len(non_system) <= keep_recent:
        return None
    return non_system[:-keep_recent], non_system[-keep_recent:]


def build_summary_prompt(to_compress: List[Dict]) -> List[Dict]:
    """Build the single-user-message prompt that asks the LLM to summarize the
    older messages."""
    excerpt = "\n".join(
        f"{m['role'].upper()}: {str(m.get('content', ''))[:_EXCERPT_CHARS_PER_MESSAGE]}"
        for m in to_compress
    )
    return [{
        "role": "user",
        "content": (
            "Summarize this conversation excerpt in a concise paragraph. "
            "Preserve key facts, decisions, URLs found, and files discussed. "
            "Be specific:\n\n" + excerpt
        ),
    }]


def rebuild_history(history: List[Dict], summary: str, to_keep: List[Dict]) -> List[Dict]:
    """Reassemble history as: system messages, the summary framed as a turn,
    then the recent messages kept verbatim."""
    system_msgs = [m for m in history if m["role"] == "system"]
    return system_msgs + [
        {"role": "user",      "content": f"[Earlier conversation summary]\n{summary}"},
        {"role": "assistant", "content": "Understood, I have the context."},
    ] + to_keep
