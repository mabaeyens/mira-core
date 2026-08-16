"""Core orchestration logic for tool calling and search."""

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Iterator

import openai as _openai

from .config import (
    MODEL_NAME, BACKEND, BACKEND_HOST,
    MAX_RETRIES, MAX_AGENT_STEPS, AGENT_DIVERGENCE_LIMIT,
    MAX_TOOL_CALLS_PER_TURN, SAME_TOOL_REPEAT_LIMIT, UNPRODUCTIVE_TOOL_REPEAT_LIMITS, TOOL_SOFT_LIMIT, VERBOSE_DEFAULT,
    RAG_MAX_CHUNKS, CONTEXT_WINDOW,
    COMPRESS_THRESHOLD, COMPRESS_KEEP_RECENT,
    THINKING_MODE, MAX_THINKING_TOKENS, TEMP_WORKSPACE_MAX_MB,
    TEMPERATURE, TOP_P, TOP_K, SEED,
    REPETITION_PENALTY, REPETITION_CONTEXT_SIZE,
    PRESENCE_PENALTY, PRESENCE_CONTEXT_SIZE,
    FREQUENCY_PENALTY, FREQUENCY_CONTEXT_SIZE,
)
from .tools import TOOLS, GITHUB_TOOLS, _LOCAL_TOOLS, _TEMP_WORKSPACE_TOOLS
from .prompts import build_system_prompt, current_time_note, SEARCH_RESULT_TEMPLATE, wrap_untrusted
from .search_engine import SearchEngine
from .rag_engine import RagEngine
from . import url_fetcher
from . import tool_registry
from .backend_manager import PRESETS
from . import file_handler
from .thinking_stripper import ThinkingStripper
from .workspace import safe_filename
from . import backend_client as bc
from . import context_manager as ctxmgr

logger = logging.getLogger(__name__)


# Box-drawing, rules and padding. A good ASCII diagram is legitimately dominated
# by these characters, and a guard that ignored them would throw away some of the
# better answers Mira gives — several of the diagrams in the 2026-08-11 corpus are
# over 40% one glyph.
_DRAWING = set("─│┌┐└┘├┤┬┴┼═║╔╗╚╝▼▲◄►·•*#=-_+|<>. \t\n")

# Below this, a repetitive-looking reply is more likely to be a terse real answer
# ("yes", "42", an emoji) than a generation loop.
_DEGENERATE_MIN_CHARS = 200
_DEGENERATE_SHARE = 0.4


def _degenerate_run(text: str):
    """Detect a reply that is one character repeated to fill the budget.

    Returns ``(char, share)`` when the text is degenerate, else ``None``.

    Judged over the non-drawing characters only, so a diagram is not mistaken for
    a loop. The threshold is deliberately loose: real prose is never 40% one
    letter, and the failure this exists for was 100%.
    """
    body = [c for c in text if c not in _DRAWING]
    if len(body) < _DEGENERATE_MIN_CHARS:
        return None
    char, count = Counter(body).most_common(1)[0]
    share = count / len(body)
    return (char, share) if share > _DEGENERATE_SHARE else None


def _sanitize_loaded_history(messages: List[Dict]) -> List[Dict]:
    """Drop empty or degenerate assistant turns before history re-enters context.

    Generation-time guards stop *new* poison, but history saved before those
    guards (or by any future edge case) can still hold a blank or `!!!!`-style
    assistant turn. Feeding one back makes the model condition on garbage, so
    strip them here. Only the model-facing history is filtered — the DB rows and
    the app's own view (load_messages) are untouched.
    """
    clean, dropped = [], 0
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            if not content.strip() or _degenerate_run(content):
                dropped += 1
                continue
        clean.append(m)
    if dropped:
        logger.info("Sanitized loaded history: dropped %d empty/degenerate assistant turn(s)", dropped)
    return clean


def _tool_ui_labels(name: str, args: dict):
    """Return (start_label, done_label_fn) for a tool call."""
    # Not every tool returns a dict. list_attachments and read_attachment hand
    # back a plain string, and calling .get on one raised mid-stream, killing
    # three turns of a corpus conversation with "Internal error". The registry
    # has always allowed a string result; this layer just never believed it.
    def _err(r): return r.get("error", "") if isinstance(r, dict) else ""
    def _ok(r, msg): return msg if not _err(r) else f"Error: {_err(r)}"

    if name == "read_file":
        p = args.get("path", "")
        return f"Reading {p}", lambda r: _ok(r, f"Read {r.get('size', 0):,} chars — {p}")
    if name == "write_file":
        p = args.get("path", "")
        return f"Writing {p}", lambda r: _ok(r, f"{r.get('action','Wrote')} {r.get('bytes_written', 0):,} bytes — {p}")
    if name == "edit_file":
        p = args.get("path", "")
        return f"Editing {p}", lambda r: _ok(r, f"Edited line {r.get('line', '?')} — {p}")
    if name == "list_files":
        p = args.get("path", ".")
        return f"Listing {p}", lambda r: _ok(r, f"{r.get('count', 0)} entries — {p}")
    if name == "search_files":
        pat = args.get("pattern", "")
        return f"Searching for '{pat}'", lambda r: _ok(r, f"{r.get('count', 0)} match(es) for '{pat}'")
    if name == "move_file":
        return f"Moving {args.get('src', '')} → {args.get('dst', '')}", lambda r: _ok(r, "Moved")
    if name == "delete_file":
        p = args.get("path", "")
        return f"Deleting {p}", lambda r: _ok(r, f"Deleted {p}") if not r.get("requires_confirmation") else "Needs confirmation"
    if name == "run_shell":
        cmd = args.get("command", "")[:60]
        return f"$ {cmd}", lambda r: _ok(r, f"exit {r.get('exit_code', '?')} — {cmd}")
    if name == "github_clone_repo":
        repo = args.get("repo", "")
        return f"Cloning {repo}", lambda r: _ok(r, f"Cloned to {r.get('cloned_to', '?')} — registered as project '{r.get('project_name', '')}'")
    if name.startswith("github_"):
        label = name.replace("github_", "GitHub: ").replace("_", " ")
        repo = args.get("repo", "")
        suffix = f" {repo}" if repo else ""
        return f"{label}{suffix}", lambda r: _ok(r, f"Done — {label.strip()}")
    return name, lambda r: "Done" if not _err(r) else f"Error: {_err(r)}"


def _read_omlx_api_key() -> str:
    import json
    settings = Path.home() / ".omlx" / "settings.json"
    try:
        return json.loads(settings.read_text())["auth"]["api_key"]
    except Exception:
        return "none"


def _make_oai_client(host: str, api_key: str = "none") -> _openai.OpenAI:
    """Create an OpenAI-compatible client pointed at the given host."""
    base = host.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return _openai.OpenAI(base_url=base, api_key=api_key)


_THINK_VERBS = re.compile(
    r"\b(why|how|fix|debug|implement|refactor|explain|design|"
    r"analyze|review|optimize|architect|plan|compare|difference|tradeoff)\b",
    re.IGNORECASE,
)
_CODE_SIGNAL = re.compile(
    r"```|def |class |import |error:|traceback"
    r"|\b\w+\.(swift|py|js|ts|kt|java|go|rs|rb|cpp|c|h|vue|tsx|jsx)\b",
    re.IGNORECASE,
)

# Short acknowledgements that are never complex regardless of other signals.
_TRIVIAL = re.compile(
    r"^\s*(ok|okay|thanks|thank you|got it|sounds good|perfect|great|sure|yes|no|yep|nope|cool)\s*[.!]?\s*$",
    re.IGNORECASE,
)

# Backends whose OpenAI-compatible server applies the model's own chat template,
# so an `enable_thinking` kwarg reaches that template (and its side effects).
_CHAT_TEMPLATE_BACKENDS = ("mlx-lm", "omlx", "mira-mlx", "vllm-mlx")


def _uses_qwen_thinking_template(backend: str, model: str) -> bool:
    """True when a turn goes through a Qwen3 chat template that honors
    `enable_thinking`. Shared by the request side (which sends the kwarg) and
    the response side (which must know the template pre-opened `<think>`), so
    the two can never drift apart."""
    return backend in _CHAT_TEMPLATE_BACKENDS and (
        "Qwen3" in model or "qwen3" in model.lower()
    )


# Queries/commands that should never trigger thinking regardless of other signals.
_NEVER_THINK = re.compile(
    r"^\s*(what\s+(time|day|date)|what'?s\s+the\s+(time|date|day))"
    r"|^\s*(summarize|summary|list|show|open|read|run|fix|check|test|display)\s+\S+\s+in\s+\d+\s+\w+"
    r"|^\s*(fix|run|read|open|show|check|test)\s+\S+\.\w+\s*$",
    re.IGNORECASE,
)

# Strong reasoning intent — these imply the user wants the model to reason, regardless
# of message length or code formatting (which mobile users rarely type). Matched on its
# own, not scored, so short questions like "why is it O(n²)" trigger thinking.
_REASONING_INTENT = re.compile(
    r"\b(why|explain|analyz|compare|contrast|debug|optimi[sz]e|prove|derive|"
    r"design|architect|refactor|trade[- ]?off|difference between|pros and cons|"
    r"step[- ]by[- ]step|walk me through)\b"
    r"|\bhow (do|does|did|can|could|would|should|to|is|are|come)\b",
    re.IGNORECASE,
)

# Analytical verbs that, combined with an attachment, justify the +2 bonus.
_ANALYTICAL_WITH_ATTACHMENT = re.compile(
    r"\b(explain|analyze|analyse|compare|debug|refactor|review|architect|understand|"
    r"improve|evaluate|assess|summarize|describe|interpret)\b",
    re.IGNORECASE,
)


_MULTI_STEP_RE = re.compile(
    r'\b(then\b|after that|and then|next,|finally,)\b',
    re.IGNORECASE,
)

_TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
}


def _should_think(message: str, has_attachments: bool = False) -> bool:
    """Return True if the message warrants extended reasoning."""
    if _TRIVIAL.match(message):
        return False
    if _NEVER_THINK.match(message):
        return False
    # Reasoning questions warrant thinking on their own — independent of length or
    # code formatting that mobile users won't type.
    if _REASONING_INTENT.search(message):
        return True
    score = 0
    if has_attachments:
        score += 1
        if _ANALYTICAL_WITH_ATTACHMENT.search(message):
            score += 2
    if len(message) > 500:
        score += 2
    elif len(message) > 150:
        score += 1
    if _CODE_SIGNAL.search(message):
        score += 2
    if _THINK_VERBS.search(message):
        score += 1
    return score >= 4


def _append_step_nudge(history: List[Dict], text: str) -> None:
    """Append an agent-loop nudge, respecting strict tool/assistant alternation.

    Some backends' chat templates (e.g. Mistral family) reject a bare `user`
    turn immediately following a `tool` result with no intervening `assistant`
    turn. Fold the nudge into the trailing tool message's content instead of
    appending a new turn when that's the case; only append as a standalone
    `user` message when the history already ends on an `assistant` turn.
    """
    if history and history[-1].get("role") == "tool":
        # Reassign (not mutate) the trailing dict — callers may pass a
        # shallow copy of conversation_history and share dict references
        # with the original list.
        history[-1] = {**history[-1], "content": f"{history[-1]['content']}\n\n{text}"}
    else:
        history.append({"role": "user", "content": text})


class ChatOrchestrator:
    """Manages the conversation loop with tool calling."""

    def __init__(self, model: str = MODEL_NAME, verbose: bool = VERBOSE_DEFAULT):
        self.model = model
        self.verbose = verbose
        self.backend = BACKEND
        self.context_window = CONTEXT_WINDOW
        self.thinking_mode = THINKING_MODE

        api_key = _read_omlx_api_key() if self.backend == "omlx" else "none"
        self._oai = _make_oai_client(BACKEND_HOST, api_key=api_key)

        self.search_engine = SearchEngine()
        self.rag_engine = RagEngine()
        self.conversation_history: List[Dict] = []
        self.last_finish_reason: Optional[str] = None  # engine's finish_reason for the last answer
        self.system_prompt_added = False
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.last_prompt_tokens: int = 0
        # True once this turn's backend has reported reasoning_tokens, meaning
        # eval_count already covers the thinking stream and the character-based
        # estimate must not be added on top. Re-armed at the start of every turn.
        self._backend_counts_reasoning: bool = False
        self.conv_id: Optional[str] = None
        self._is_new_conv: bool = False
        self.project: Optional[Dict] = None
        self._temp_workspace: Optional[str] = None
        self._attachment_registry: Dict[str, Dict] = {}
        self._github_tools_enabled: bool = False
        # Per-turn kill switch for the whole toolset. Default on: every existing
        # caller keeps today's behaviour without passing anything.
        self._tools_enabled: bool = True
        # Fail closed: any dispatch path that does not go through stream_chat
        # (e.g. /ask) carries no approvals, so destructive actions are refused.
        self._approved_tokens: frozenset = frozenset()
        self._add_system_prompt()

    @property
    def workspace_root(self) -> Optional[str]:
        return self.project.get("local_path") if self.project else None

    @property
    def _active_tools(self) -> List[Dict]:
        # Asked for a pure-generation turn: hand the model nothing, not even
        # task_done. Anything short of an empty list leaves it an escape hatch —
        # a model that can declare itself finished sometimes will, and the user
        # gets a one-line summary in place of the answer they asked for.
        if not self._tools_enabled:
            return []
        if self.workspace_root:
            base = TOOLS
        elif self._temp_workspace:
            excluded = _LOCAL_TOOLS - _TEMP_WORKSPACE_TOOLS
            base = [t for t in TOOLS if t["function"]["name"] not in excluded]
        else:
            base = [t for t in TOOLS if t["function"]["name"] not in _LOCAL_TOOLS]
        if self._github_tools_enabled:
            base = base + GITHUB_TOOLS
        return base

    def _get_or_create_temp_workspace(self) -> str:
        if self._temp_workspace is None:
            self._temp_workspace = tempfile.mkdtemp(prefix="mira_workspace_")
            logger.info("Created temp workspace: %s", self._temp_workspace)
        return self._temp_workspace

    def _cleanup_temp_workspace(self) -> None:
        if self._temp_workspace:
            shutil.rmtree(self._temp_workspace, ignore_errors=True)
            logger.info("Removed temp workspace: %s", self._temp_workspace)
            self._temp_workspace = None

    def set_project(self, project: Optional[Dict]) -> None:
        self.project = project
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            from . import db
            memories = [m["text"] for m in db.get_memories()]
            self.conversation_history[0]["content"] = build_system_prompt(project=project, memories=memories)
        else:
            self.system_prompt_added = False
            self._add_system_prompt()

    def reinitialize_client(self, backend: str, model: str, host: str,
                            context_window: int) -> None:
        """Switch to a different inference backend at runtime without restarting."""
        self.backend = backend
        self.model = model
        self.context_window = context_window
        api_key = _read_omlx_api_key() if backend == "omlx" else "none"
        self._oai = _make_oai_client(host, api_key=api_key)

    def _add_system_prompt(self):
        if not self.system_prompt_added:
            from . import db
            memories = [m["text"] for m in db.get_memories()]
            self.conversation_history.append({
                "role": "system",
                "content": build_system_prompt(project=self.project, memories=memories)
            })
            self.system_prompt_added = True

    def _thinking_off_extra(self) -> dict:
        """`extra_body` that disables reasoning, for the utility calls that go
        straight to the model instead of through `_call_llm`.

        Title, history-summary and forced-JSON turns bypass `_call_llm`, so
        without this they inherit Qwen3.6's thinking-on template default and,
        having no budget guard either, can reason to the full output cap on a
        request as trivial as "name this conversation" — measured 2026-08-13, a
        title turn ran 1651 reasoning tokens (~29s) for a two-sentence chat.
        `enable_thinking=False` is the primary guard; the budget is a backstop
        for the turns where the model opens a block anyway. Mirrors the thinking
        half of `_call_llm` so both paths stay in step.
        """
        if _uses_qwen_thinking_template(self.backend, self.model):
            ckwargs: dict = {"enable_thinking": False}
            if MAX_THINKING_TOKENS > 0:
                ckwargs["thinking_budget"] = MAX_THINKING_TOKENS
            return {"chat_template_kwargs": ckwargs}
        return {"enable_thinking": False}

    def _prefill_system_prompt(self) -> None:
        """Send the current system prompt to the LLM in a background thread to warm the prefix cache."""
        if not self.conversation_history or self.conversation_history[0]["role"] != "system":
            return
        system_msg = self.conversation_history[0]
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[system_msg, {"role": "user", "content": "Hi"}],
                max_tokens=1,
                stream=False,
                extra_body=self._thinking_off_extra(),
            )
            logger.debug("prefix cache warm for new conversation")
        except Exception as exc:
            logger.debug("background prefill failed (non-fatal): %s", exc)

    # ── Conversation lifecycle ────────────────────────────────────────────────

    def new_conversation(self, conv_id: str, project: Optional[Dict] = None) -> None:
        self.conv_id = conv_id
        self._is_new_conv = True
        self.project = project
        self.conversation_history = []
        self.system_prompt_added = False
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_prompt_tokens = 0
        self._add_system_prompt()
        self.rag_engine.load_project(project["id"] if project else None)
        threading.Thread(target=self._prefill_system_prompt, daemon=True).start()

    def load_conversation(self, conv_id: str, project: Optional[Dict] = None) -> None:
        from . import db
        messages = _sanitize_loaded_history(db.load_messages(conv_id))
        self.conv_id = conv_id
        self._is_new_conv = False
        self.project = project
        self.conversation_history = []
        self.system_prompt_added = False
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_prompt_tokens = 0
        self._add_system_prompt()
        self.conversation_history.extend(messages)
        self.rag_engine.load_project(project["id"] if project else None)
        logger.info("Loaded conversation %s: %d messages", conv_id, len(messages))

    # ── Post-turn helpers ────────────────────────────────────────────────────

    def _llm_chat_sync(self, messages: List[Dict], format: Optional[dict] = None) -> str:
        """Non-streaming single-turn LLM call. Returns the text content.

        These are utility turns (titles, summaries, forced JSON), never the
        user-facing answer, so reasoning is suppressed via extra_body — see
        `_thinking_off_extra`. Without it a title turn can reason to the full
        output cap, adding tens of seconds to an otherwise trivial request.
        """
        extra = self._thinking_off_extra()
        kwargs = {"response_format": {"type": "json_object"}} if format else {}
        if True:
            try:
                resp = self._oai.chat.completions.create(
                    model=self.model, messages=messages, extra_body=extra, **kwargs
                )
            except Exception:
                # Backend doesn't support response_format — retry without it
                resp = self._oai.chat.completions.create(
                    model=self.model, messages=messages, extra_body=extra
                )
            return bc.strip_think((resp.choices[0].message.content or "").strip())

    def generate_title(self, first_user_message: str) -> str:
        try:
            raw = self._llm_chat_sync(
                [{"role": "user", "content": (
                    "Reply with a JSON object {\"title\": \"...\"} containing a short title "
                    "for a conversation that starts with this message. 4-6 words:\n\n"
                    + first_user_message[:300]
                )}],
                format=_TITLE_SCHEMA,
            )
            stripped = re.sub(r"^```[a-z]*\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            match = re.search(r'\{.*\}', stripped, re.DOTALL)
            data = json.loads(match.group(0) if match else stripped)
            return data.get("title", raw)[:80]
        except Exception:
            return first_user_message[:60].strip()

    def compress_history(self) -> Optional[str]:
        plan = ctxmgr.plan_compression(self.conversation_history, COMPRESS_KEEP_RECENT)
        if plan is None:
            return None
        to_compress, to_keep = plan

        try:
            summary = self._llm_chat_sync(ctxmgr.build_summary_prompt(to_compress))
        except Exception as e:
            logger.warning("Compression LLM call failed: %s", e)
            return None

        if not summary:
            return None

        self.conversation_history = ctxmgr.rebuild_history(
            self.conversation_history, summary, to_keep
        )
        logger.info("Compressed %d messages into summary", len(to_compress))
        return summary

    # ── Main stream ───────────────────────────────────────────────────────────

    def stream_chat(self, *args, **kwargs) -> Iterator[Dict]:
        """Time one whole user turn, then delegate to the real implementation.

        A thin wrapper rather than a try/finally inside the body because the
        body has many exit paths (refusals, forced summaries, errors) and a turn
        that ends down one of them is exactly the kind we most want measured.
        ``finally`` on a generator runs on close as well as exhaustion, so an
        abandoned stream is still accounted.
        """
        self._turn_timing = {
            "llm_ms": 0.0, "tool_ms": 0.0, "llm_calls": 0, "tool_batches": 0,
            "prompt_tokens": 0, "cached_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0,
            "engine_ttft_ms": 0.0, "engine_decode_ms": 0.0,
        }
        started = time.monotonic()
        try:
            yield from self._stream_chat_impl(*args, **kwargs)
        finally:
            self._log_turn_timing(started)

    def _log_turn_timing(self, started: float) -> None:
        """One machine-readable line per turn, for notes/turn_timing.py.

        ``other_ms`` is the residual — RAG, history load, context compression,
        SSE overhead, and anything else not attributed. It is reported rather
        than distributed, because a residual that grows is itself the finding.
        """
        t = getattr(self, "_turn_timing", None)
        if not t:
            return
        wall_ms = (time.monotonic() - started) * 1000
        t["wall_ms"] = round(wall_ms, 1)
        t["other_ms"] = round(wall_ms - t["llm_ms"] - t["tool_ms"], 1)
        for k in ("llm_ms", "tool_ms", "engine_ttft_ms", "engine_decode_ms"):
            t[k] = round(t[k], 1)
        if t["engine_decode_ms"] > 0 and t["completion_tokens"] > 0:
            t["decode_tps"] = round(t["completion_tokens"] / (t["engine_decode_ms"] / 1000), 2)
        # Effective rate over the whole turn: the number a user actually feels,
        # and the one that diverges from decode_tps by the factor this whole
        # exercise exists to explain.
        if wall_ms > 0 and t["completion_tokens"] > 0:
            t["effective_tps"] = round(t["completion_tokens"] / (wall_ms / 1000), 2)
        logger.info("TURN_TIMING %s", json.dumps(t, sort_keys=True))

    def _stream_chat_impl(
        self,
        user_message: str,
        attachments=None,
        thinking_enabled: Optional[bool] = None,
        github_tools_enabled: bool = False,
        approved_tokens: Optional[frozenset] = None,
        tools_enabled: bool = True,
    ) -> Iterator[Dict]:
        """
        Process a user message and yield events for consumers (CLI, web).

        Event types: thinking, token, search_start/done, fetch_start/done/context,
        rag_indexing/done/context, stats, warning, done, error.
        """
        self._github_tools_enabled = github_tools_enabled
        self._tools_enabled = tools_enabled
        # Scoped to this turn only: an approval the user gave for one command must
        # not silently authorise a later turn's command.
        self._approved_tokens = frozenset(approved_tokens or ())
        if attachments:
            for att in attachments:
                if att.get("warning"):
                    yield {"type": "warning", "message": att["warning"]}

        # Adaptive thinking: heuristic decides when client sends no preference;
        # explicit True/False from the client is always respected as-is.
        if self.thinking_mode == "adaptive":
            if thinking_enabled is None:
                thinking_enabled = _should_think(user_message, has_attachments=bool(attachments))
        elif self.thinking_mode == "always":
            thinking_enabled = True
        elif self.thinking_mode == "never":
            thinking_enabled = False
        if thinking_enabled is None:
            thinking_enabled = False

        rag_indexed_this_turn = False
        if attachments:
            for att in attachments:
                if att["type"] != "image":
                    self._attachment_registry[att["name"]] = {
                        "name": att["name"],
                        "type": att["type"],
                        "size": len((att.get("content") or "").encode()),
                        "content": att.get("content") or "",
                    }
        if attachments:
            for att in attachments:
                if att["type"] == "rag" and att["content"]:
                    yield {"type": "rag_indexing", "name": att["name"]}
                    try:
                        n_chunks = self.rag_engine.index(att["name"], att["content"])
                        yield {"type": "rag_done", "name": att["name"], "chunks": n_chunks}
                        rag_indexed_this_turn = True
                        if not self.workspace_root:
                            ws = self._get_or_create_temp_workspace()
                            # att["name"] is untrusted (upload header, or a path
                            # attachment). Sanitized at ingestion too; repeated
                            # here so the sink is safe on its own.
                            ws_path = Path(ws) / safe_filename(att["name"])
                            used_mb = sum(
                                f.stat().st_size for f in Path(ws).rglob("*") if f.is_file()
                            ) / 1_048_576
                            file_mb = len(att["content"].encode()) / 1_048_576
                            if used_mb + file_mb < TEMP_WORKSPACE_MAX_MB:
                                ws_path.write_text(att["content"], encoding="utf-8")
                                logger.info("Wrote attachment to temp workspace: %s", ws_path)
                        if self.rag_engine.chunk_count > RAG_MAX_CHUNKS:
                            yield {
                                "type": "warning",
                                "message": (
                                    f"RAG index has {self.rag_engine.chunk_count:,} chunks. "
                                    "Consider unloading documents you no longer need."
                                ),
                            }
                    except Exception as e:
                        yield {"type": "error", "message": f"Failed to index '{att['name']}': {e}"}
                        return

        rag_chunks = []
        if self.rag_engine.chunk_count > 0:
            try:
                if rag_indexed_this_turn:
                    rag_chunks = self.rag_engine.query(user_message, score_threshold=float('-inf'))
                else:
                    rag_chunks = self.rag_engine.query(user_message)
            except Exception as e:
                logger.warning("RAG query failed: %s", e)

        full_message = user_message
        images = []
        if attachments:
            text_parts = [
                wrap_untrusted(f"[File: {att['name']}]\n{att['content']}", source="attachment")
                for att in attachments
                if att["type"] == "text" and att["content"]
            ]
            images = [att for att in attachments if att["type"] == "image"]
            if images and not PRESETS.get(self.backend, {}).get("vision", False):
                ocr_parts = []
                for att in images:
                    ocr_text = file_handler.ocr_image_from_base64(att["content"])
                    if ocr_text:
                        ocr_parts.append(
                            wrap_untrusted(
                                f"[OCR text extracted from screenshot '{att['name']}']\n{ocr_text}",
                                source="ocr",
                            )
                        )
                if ocr_parts:
                    text_parts.extend(ocr_parts)
                    images = []
                else:
                    yield {
                        "type": "error",
                        "message": (
                            f"The active backend ({self.backend}) doesn't support image "
                            "inputs, and OCR found no readable text in the attached "
                            "image(s) — switch to omlx in Settings for photos/diagrams, "
                            "or install tesseract (`brew install tesseract`) for "
                            "text-heavy screenshots."
                        ),
                    }
                    return
            images = [att["content"] for att in images]
            if text_parts:
                full_message = '\n\n'.join(text_parts) + '\n\n' + full_message

        if rag_chunks:
            if rag_indexed_this_turn:
                attached_names = ", ".join(
                    f"`{att['name']}`" for att in (attachments or []) if att["type"] == "rag"
                )
                context = wrap_untrusted(
                    "\n\n".join(f"[File: {c['source']}]\n{c['text']}" for c in rag_chunks),
                    source="attachment",
                )
                framing = (
                    f"The user attached these files: {attached_names}. "
                    "Their content is provided below. "
                    "Use `read_attachment(name)` to read them directly — "
                    "do NOT use web or GitHub tools to open local files."
                )
                full_message = f"{framing}\n\n[Attached file content]\n{context}\n\n---\n\n{full_message}"
            else:
                context = wrap_untrusted(
                    "\n\n".join(
                        f"[Source: {c['source']} | Score: {c['score']:.2f}]\n{c['text']}"
                        for c in rag_chunks
                    ),
                    source="rag",
                )
                full_message = f"[Relevant document sections]\n{context}\n\n---\n\n{full_message}"

        # Inject CURRENT TASK block for multi-step requests to anchor the goal
        if _MULTI_STEP_RE.search(user_message):
            full_message = f"CURRENT TASK: {user_message}\n\n{full_message}"

        # Timestamp lives per-turn (not in the system prompt) so the system
        # prompt's prefix stays cache-stable across turns.
        full_message = f"{full_message}\n\n{current_time_note()}"

        user_msg: Dict = {"role": "user", "content": full_message}
        if images:
            user_msg["images"] = images

        self.conversation_history.append(user_msg)

        fetch_results = []
        _thinking_chars = 0  # accumulated across all tool steps for this turn
        # Reset per turn, not per process: a backend switch mid-conversation can
        # change whether reasoning_tokens is reported at all, and a sticky flag
        # would either double-count or silently stop counting after the switch.
        self._backend_counts_reasoning = False
        self._task_done = False
        self._task_done_summary = ""
        self._task_done_refused = False   # one refusal per turn; see the guard below
        self._tool_call_hashes: dict = {}  # call_hash -> repeat count
        self._total_tool_calls = 0           # hard cap: MAX_TOOL_CALLS_PER_TURN
        self._tool_name_counts: dict = {}    # tool name -> call count for SAME_TOOL_REPEAT_LIMIT

        for step in range(MAX_AGENT_STEPS):
            if thinking_enabled:
                yield {"type": "thinking"}

            # Soft step limit: warn the model at the halfway point and again near the hard cap.
            if step == MAX_AGENT_STEPS // 2:
                _append_step_nudge(
                    self.conversation_history,
                    f"[System: You have made {step} tool calls so far. "
                    f"If your task is complete, call task_done now with a summary. "
                    f"Otherwise consolidate your remaining work into as few commands as possible — "
                    f"the hard limit is {MAX_AGENT_STEPS} total steps.]",
                )
            elif step == MAX_AGENT_STEPS - 3:
                _append_step_nudge(
                    self.conversation_history,
                    f"[System: URGENT — only {MAX_AGENT_STEPS - step} steps remain before hard cutoff. "
                    f"You MUST call task_done NOW with a summary of what you have found so far. "
                    f"Do not make any more tool calls.]",
                )
            elif step == MAX_AGENT_STEPS - 2 and self._total_tool_calls > 0 and not self._task_done:
                # Hard forced summary: model has been running too long — synthesize a
                # response from what it has gathered so far.
                try:
                    forced_history = list(self.conversation_history)
                    _append_step_nudge(
                        forced_history,
                        "Summarize what you have accomplished or found so far, in 2-3 sentences.",
                    )
                    forced = self._llm_chat_sync(forced_history)
                    if forced:
                        self._mark_task_done(forced)
                        if _thinking_chars and not self._backend_counts_reasoning:
                            self.total_output_tokens += ctxmgr.thinking_tokens(_thinking_chars)
                        yield {
                            "type": "stats",
                            "input_tokens": self.total_input_tokens,
                            "output_tokens": self.total_output_tokens,
                            "context_pct": self.context_pct,
                        }
                        yield {"type": "done", "content": forced, "task_done": True}
                        return
                except Exception as _e:
                    logger.warning("Hard forced summary failed: %s", _e)

            full_content = ""
            final_message = None
            _llm_started = time.monotonic()
            for event in self._stream_llm_with_thinking(thinking_enabled):
                if event["type"] == "llm_done":
                    full_content = event["full_content"]
                    final_message = event["final_message"]
                    _thinking_chars += event["thinking_chars"]
                elif event["type"] == "error":
                    yield event
                    return
                else:
                    yield event
            self._turn_timing["llm_ms"] += (time.monotonic() - _llm_started) * 1000
            self._turn_timing["llm_calls"] += 1

            tool_calls = final_message.tool_calls

            if not tool_calls:
                # Silent completion: model used tools but produced no text response.
                # Inject a forced summary turn so the user never sees an empty bubble.
                if not full_content.strip() and step > 0:
                    try:
                        forced = self._llm_chat_sync(
                            self.conversation_history + [{
                                "role": "user",
                                "content": "Briefly summarize what you just accomplished in 1–2 sentences.",
                            }]
                        )
                        if forced:
                            full_content = forced
                            # Mark as implicit task_done so done event carries the flag
                            self._mark_task_done(forced)
                    except Exception as _e:
                        logger.warning("Forced summary call failed: %s", _e)
                elif full_content.strip() and self._total_tool_calls > 0:
                    # Model completed tool work and responded with text — first-class exit.
                    self._mark_task_done(full_content)
                self.conversation_history.append({"role": "assistant", "content": full_content})
                if fetch_results:
                    yield {"type": "fetch_context", "fetches": fetch_results}
                if rag_chunks:
                    yield {
                        "type": "rag_context",
                        "chunks": [
                            {
                                "source": c["source"],
                                "score": round(c["score"], 2),
                                "preview": c["text"][:150].rstrip() + ("…" if len(c["text"]) > 150 else ""),
                            }
                            for c in rag_chunks
                        ],
                    }
                # Fallback only. Backends that report reasoning_tokens already count
                # the thinking stream inside eval_count (mira-mlx does, from its
                # sequence state machine). For the ones that don't, convert
                # accumulated thinking chars to approximate tokens (~3.5 chars/tok)
                # so the display still reflects actual compute rather than ignoring
                # what is often most of the generation.
                if _thinking_chars and not self._backend_counts_reasoning:
                    self.total_output_tokens += ctxmgr.thinking_tokens(_thinking_chars)
                yield {
                    "type": "stats",
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "context_pct": self.context_pct,
                }
                yield {"type": "done", "content": full_content, "task_done": self._task_done}
                return

            # Model requested tool call(s) — normalize history append
            self.conversation_history.append(bc.message_to_history_dict(final_message))

            # Prepare: resolve tc_id, divergence for each call in the batch
            prepared = []
            for i, tc in enumerate(tool_calls):
                tc_id = getattr(tc, 'id', None) or f"call_{step}_{i}"
                name = tc.function.name
                args = tc.function.arguments
                if name != "task_done":
                    call_hash = str(hash(name + json.dumps(args, sort_keys=True)))
                    repeat_count = self._tool_call_hashes.get(call_hash, 0) + 1
                    self._tool_call_hashes[call_hash] = repeat_count
                    diverged = repeat_count > AGENT_DIVERGENCE_LIMIT
                    self._total_tool_calls += 1
                    self._tool_name_counts[name] = self._tool_name_counts.get(name, 0) + 1
                else:
                    diverged = False
                prepared.append((tc_id, name, args, diverged))

            # Hard cap: bail before executing if the absolute turn limit is exceeded.
            if self._total_tool_calls > MAX_TOOL_CALLS_PER_TURN:
                yield from self._soft_pause_checkin(f"{self._total_tool_calls} total tool calls")
                return

            # Per-tool hard cap (SAME_TOOL_REPEAT_LIMIT / UNPRODUCTIVE limits).
            over_limit = [
                (n, c) for n, c in self._tool_name_counts.items()
                if c > UNPRODUCTIVE_TOOL_REPEAT_LIMITS.get(n, SAME_TOOL_REPEAT_LIMIT)
            ]
            if over_limit:
                names_str = ", ".join(f"{n}×{c}" for n, c in over_limit)
                yield from self._soft_pause_checkin(names_str)
                return

            # Soft check-in: when any single tool reaches TOOL_SOFT_LIMIT, ask the
            # user before continuing instead of running straight to the hard cap.
            soft_hit = [
                (n, c) for n, c in self._tool_name_counts.items()
                if c == TOOL_SOFT_LIMIT
            ]
            if soft_hit:
                names_str = ", ".join(f"{n}×{c}" for n, c in soft_hit)
                yield from self._soft_pause_checkin(names_str)
                return

            # task_done short-circuits the whole batch immediately
            if any(name == "task_done" for _, name, _, _ in prepared):
                # Refuse a task_done that would be the entire user-visible output.
                # With no tool run and no text streamed, the summary is a claim
                # about work the user cannot see: the model declares an answer
                # instead of giving one. Bench Q4 on 2026-08-01 is the
                # reproduction — "Provided a complete Python context manager for
                # sqlite3 ..." and no context manager.
                #
                # The condition keys on tools having run, not on empty content
                # alone, so a genuinely agentic turn whose answer *is* a summary
                # ("deleted 3 files") still exits here. And because any
                # non-task_done call in this batch already incremented
                # _total_tool_calls above, reaching this branch means every entry
                # in `prepared` is a task_done — nothing else goes unanswered.
                if (
                    not full_content.strip()
                    and self._total_tool_calls == 0
                    and not self._task_done_refused
                ):
                    self._task_done_refused = True
                    for tc_id, name, _, _ in prepared:
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": name,
                            "content": (
                                "Refused: task_done is not an answer. You have run no tools "
                                "and produced no output for this request, so there is nothing "
                                "for the user to read. Write the full answer itself in your "
                                "next reply. Do not call task_done again."
                            ),
                        })
                    logger.info("Refused task_done: no visible content and no tools ran")
                    continue
                task_done_args = next(args for _, name, args, _ in prepared if name == "task_done")
                self._mark_task_done(task_done_args.get("summary", "Task complete."))
                if _thinking_chars and not self._backend_counts_reasoning:
                    self.total_output_tokens += ctxmgr.thinking_tokens(_thinking_chars)
                yield {
                    "type": "stats",
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "context_pct": self.context_pct,
                }
                yield {"type": "done", "content": self._task_done_summary, "task_done": True}
                return

            _tools_started = time.monotonic()
            yield from self._execute_tools(prepared, step, fetch_results)
            self._turn_timing["tool_ms"] += (time.monotonic() - _tools_started) * 1000
            self._turn_timing["tool_batches"] += 1

        yield from self._soft_pause_checkin(f"{MAX_AGENT_STEPS} agent steps")

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _wrap_observation(self, tool_name: str, result: dict) -> dict:
        """Normalise a tool result into a structured observation for the model."""
        if isinstance(result, dict) and "error" in result:
            return {"status": "error", "payload": None, "error_details": result["error"]}
        return {"status": "success", "payload": result}

    def _soft_pause_checkin(self, reason: str):
        """
        Instead of hard-stopping when a tool limit is reached, inject a forced
        prompt so the model summarises its findings and asks the user whether to
        continue.  Yields token events followed by a done event — identical to a
        normal turn completion, so the server saves the response and the client
        renders it as a regular assistant message.
        """
        forced_note = (
            f"You have reached a tool-use limit ({reason}). "
            "Summarise what you have found so far, then ask the user whether they "
            "would like you to continue and, if so, how many more steps to take. "
            "Be concise. Do not call any tools."
        )
        tmp_messages = self.conversation_history + [{"role": "user", "content": forced_note}]
        full_text = ""
        try:
            for chunk in self._call_llm(tmp_messages, tools=None, thinking_enabled=False):
                token = chunk.message.content or ""
                if token:
                    full_text += token
                    yield {"type": "token", "content": token}
        except Exception as e:
            logger.warning("soft_pause_checkin LLM call failed: %s", e)
            full_text = (
                f"I've reached the tool-use limit ({reason}). "
                "Would you like me to continue? If so, let me know how many more steps to take."
            )
            yield {"type": "token", "content": full_text}
        yield {"type": "done", "content": full_text}

    def _mark_task_done(self, summary: str) -> dict:
        self._task_done = True
        self._task_done_summary = summary
        return {"done": True}

    def _dispatch_tool(self, name: str, args: dict) -> dict:
        """Run a tool via the central registry (schema↔handler parity guaranteed by
        tests/test_tool_registry.py). web_search/fetch_url are handled in the loop."""
        ctx = tool_registry.ToolContext(
            workspace_root=self.workspace_root,
            temp_workspace=self._temp_workspace,
            attachments=self._attachment_registry,
            mark_task_done=self._mark_task_done,
            approved=self._approved_tokens,
        )
        return tool_registry.dispatch(name, args, ctx)

    # ── LLM backend ──────────────────────────────────────────────────────────

    def _stream_llm_with_thinking(self, thinking_enabled: bool):
        """Call the LLM, strip thinking tags, yield typed events.

        Final event: {"type": "llm_done", "full_content": str, "final_message": obj,
                      "thinking_chars": int, "finish_reason": str | None}
        ``finish_reason`` is what the engine reported — "stop" when the model chose
        to end, "length" when it was cut off at max_tokens with more to say, None
        when the backend didn't say. See specs/generation-runaway-guard.md.
        On failure:  {"type": "error", "message": str} — caller should forward and return.
        """
        full_content = ""
        final_message = None
        final_finish_reason = None
        thinking_chars = 0

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                accumulated_tool_calls = None
                # Qwen3 templates append a bare "<think>\n" to the prompt when
                # thinking is on, so the model's output starts inside the block
                # and never sends an opening tag. Tell the stripper, or the whole
                # reasoning stream is served to the user as the answer.
                stripper = ThinkingStripper(
                    preopened=thinking_enabled
                    and _uses_qwen_thinking_template(self.backend, self.model)
                )

                for chunk in self._call_llm(
                    self.conversation_history,
                    tools=self._active_tools,
                    thinking_enabled=thinking_enabled,
                ):
                    if chunk.message.tool_calls:
                        accumulated_tool_calls = chunk.message.tool_calls
                        logger.info("chunk HAS tool_calls: done=%s tool_calls=%s", chunk.done, chunk.message.tool_calls)

                    thinking_token = getattr(chunk.message, "thinking", None) or ""
                    if thinking_token:
                        # This backend splits reasoning into its own channel, so
                        # `content` carries only the answer — never treat it as a
                        # pre-opened think block.
                        stripper.saw_reasoning()
                        thinking_chars += len(thinking_token)
                        yield {"type": "thinking", "content": thinking_token}

                    raw_token = chunk.message.content or ""
                    if raw_token:
                        # ThinkingStripper runs the dual-pass <think>/Gemma-channel
                        # state machine and yields {"thinking"|"token"} events.
                        yield from stripper.feed(raw_token)
                        full_content = stripper.full_content

                    if chunk.done:
                        final_message = chunk.message
                        final_finish_reason = getattr(chunk, "finish_reason", None)
                        if not final_message.tool_calls and accumulated_tool_calls:
                            # Gemma4 quirk: tool_calls arrive in intermediate chunks
                            if hasattr(final_message, 'model_copy'):
                                final_message = final_message.model_copy(
                                    update={"tool_calls": accumulated_tool_calls}
                                )
                            else:
                                final_message.tool_calls = accumulated_tool_calls
                        p = getattr(chunk, 'prompt_eval_count', None)
                        e = getattr(chunk, 'eval_count', None)
                        if isinstance(p, int):
                            self.last_prompt_tokens = p
                            self.total_input_tokens += p
                        if isinstance(e, int):
                            self.total_output_tokens += e
                        # A backend that reports reasoning_tokens has already
                        # counted the thinking stream inside eval_count, so the
                        # character estimate elsewhere would double it. isinstance,
                        # not `is not None`, for the same reason as p and e above.
                        if isinstance(getattr(chunk, 'reasoning_tokens', None), int):
                            self._backend_counts_reasoning = True

                        # Per-turn attribution. Summed across every LLM call in
                        # the turn, so an agentic turn's four prefills all land
                        # in engine_ttft_ms and the tool rounds between them in
                        # tool_ms — which is the whole point of the split.
                        tt = getattr(self, "_turn_timing", None)
                        if tt is not None:
                            if isinstance(p, int):
                                tt["prompt_tokens"] += p
                            if isinstance(e, int):
                                tt["completion_tokens"] += e
                            c = getattr(chunk, 'cached_tokens', None)
                            if isinstance(c, int):
                                tt["cached_tokens"] += c
                            rt = getattr(chunk, 'reasoning_tokens', None)
                            if isinstance(rt, int):
                                tt["reasoning_tokens"] += rt
                            tm = getattr(chunk, 'timing', None) or {}
                            if isinstance(tm.get("ttft_ms"), (int, float)):
                                tt["engine_ttft_ms"] += tm["ttft_ms"]
                            if isinstance(tm.get("decode_ms"), (int, float)):
                                tt["engine_decode_ms"] += tm["decode_ms"]

                # Tell the stripper *why* the stream ended before it decides what
                # to do with an unclosed think block. It cannot tell "the model
                # chose not to close it" from "the model was cut off" on its own,
                # and the two want opposite handling.
                if final_finish_reason == "length":
                    stripper.truncated()
                # Drain remaining buffered content.
                yield from stripper.drain()
                full_content = stripper.full_content
                thinking_chars += stripper.thinking_chars
                break
            except Exception as e:
                if full_content:
                    yield {"type": "error", "message": str(e)}
                    return
                if attempt == MAX_RETRIES:
                    yield {"type": "error", "message": str(e)}
                    return
                logger.warning("LLM API error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)

        if final_message is None:
            yield {"type": "error", "message": "LLM stream closed without a completion signal."}
            return

        logger.info(
            "LLM response — content_len=%d tool_calls=%s thinking_len=%d finish_reason=%s",
            len(final_message.content or ""),
            bool(final_message.tool_calls),
            len(getattr(final_message, 'thinking', None) or ""),
            final_finish_reason,
        )
        if final_finish_reason == "length":
            # Logged loudly on purpose: this is the only record that a reply was cut
            # off mid-sentence rather than finished, and counting these in the log is
            # how we size how often it happens. See specs/generation-runaway-guard.md.
            logger.warning(
                "LLM hit the token cap (finish_reason=length) — reply is truncated. "
                "content_len=%d thinking_len=%d",
                len(final_message.content or ""),
                len(getattr(final_message, 'thinking', None) or ""),
            )

        degenerate = _degenerate_run(full_content)
        if degenerate:
            # A reply that is one character repeated thousands of times is not an
            # answer, and it must not survive: it gets saved to the conversation
            # and fed back in, after which every following turn returns the same
            # thing. Observed 2026-08-11 — five consecutive turns of exactly 4096
            # '!' , the last four with no tool calls at all, in a conversation
            # that could not be recovered from the app.
            logger.warning(
                "discarding a degenerate reply: %d chars, %.0f%% the character %r",
                len(full_content), degenerate[1] * 100, degenerate[0],
            )
            full_content = ""

        if not full_content and not final_message.tool_calls and (
                final_finish_reason == "length" or degenerate):
            # Say what happened rather than return an empty turn. The reasoning
            # has already been streamed on its own channel, so the user can still
            # see the work; what they must not get is that work presented as the
            # answer.
            full_content = (
                "I ran out of room before I finished answering. Ask me again, "
                "or narrow the question, and I'll get further."
            )

        # Record for persistence — the last LLM call in the turn is the final
        # answer, so this ends up holding that answer's finish_reason at save time.
        self.last_finish_reason = final_finish_reason
        yield {
            "type": "llm_done",
            "full_content": full_content,
            "final_message": final_message,
            "thinking_chars": thinking_chars,
            "finish_reason": final_finish_reason,
        }

    def _execute_tools(self, prepared: list, step: int, fetch_results: list):
        """Emit start events, run tools in parallel, emit result events, update history."""
        for tc_id, name, args, diverged in prepared:
            if name == "web_search":
                yield {"type": "search_start", "query": args.get("query", "")}
            elif name == "fetch_url":
                if args.get("url", "").startswith(("http://", "https://")):
                    yield {"type": "fetch_start", "url": args.get("url", "")}
            else:
                label_start, _ = _tool_ui_labels(name, args)
                yield {"type": "tool_start", "tool": name, "label": label_start}

        def _run_tool(tc_id, name, args, diverged):
            if diverged:
                return {"error": "Same tool+args repeated too many times. Try a different approach."}
            if name == "web_search":
                query = args.get("query", "")
                try:
                    results = self.search_engine.search(query)
                except Exception as e:
                    logger.error("Search failed: %s", e)
                    results = []
                return {"_web_search": True, "query": query, "results": results}
            if name == "fetch_url":
                url = args.get("url", "")
                content = url_fetcher.fetch_url(url)
                return {"_fetch_url": True, "url": url, "content": content}
            return self._dispatch_tool(name, args)

        with ThreadPoolExecutor(max_workers=min(len(prepared), 4)) as pool:
            futures = {
                pool.submit(_run_tool, tc_id, name, args, diverged): i
                for i, (tc_id, name, args, diverged) in enumerate(prepared)
            }
            results_by_idx: dict = {}
            for fut in as_completed(futures):
                results_by_idx[futures[fut]] = fut.result()

        for i, (tc_id, name, args, diverged) in enumerate(prepared):
            result = results_by_idx[i]
            if name == "web_search":
                query = result.get("query", "")
                web_results = result.get("results", [])
                yield {"type": "search_done", "query": query, "count": len(web_results), "results": web_results}
                yield {"type": "agent_step", "step": step + 1, "tool": "web_search", "status": "success" if web_results else "error"}
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": SEARCH_RESULT_TEMPLATE.format(
                        query=query,
                        results_text=wrap_untrusted(
                            self.search_engine.format_tool_result(web_results),
                            source="web_search",
                        ),
                    ),
                })
            elif name == "fetch_url":
                url = result.get("url", "")
                content = result.get("content", "")
                if url.startswith(("http://", "https://")):
                    yield {"type": "fetch_done", "url": url, "chars": len(content)}
                    fetch_results.append({
                        "url": url,
                        "chars": len(content),
                        "preview": content[:300].rstrip() + ("…" if len(content) > 300 else ""),
                    })
                yield {"type": "agent_step", "step": step + 1, "tool": "fetch_url", "status": "success"}
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": wrap_untrusted(content, source="fetch_url"),
                })
            else:
                _, label_done_fn = _tool_ui_labels(name, args)
                observation = self._wrap_observation(name, result)
                yield {"type": "tool_done", "tool": name, "label": label_done_fn(result)}
                # A destructive action the server refused. Surface the approval
                # token so the CLIENT can offer the user an approve control; the
                # model cannot approve it (see core/approvals.py).
                if isinstance(result, dict) and result.get("requires_confirmation"):
                    yield {
                        "type": "approval_required",
                        "tool": name,
                        "action": result.get("action", name),
                        "approval_token": result.get("approval_token"),
                        "target": result.get("command") or result.get("path") or "",
                        "matched": result.get("matched", ""),
                        "message": result.get("message", ""),
                    }
                if diverged:
                    yield {"type": "divergence_guard", "tool": name, "step": step + 1}
                yield {"type": "agent_step", "step": step + 1, "tool": name, "status": observation["status"]}
                # Wrap only genuine retrieved DATA as untrusted (RULE 10). Errors and
                # approval-gate confirmations are Mira-generated control metadata, not
                # attacker content — labelling them "data, not instruction" would blur
                # RULE 4's destructive-action surfacing.
                obs_json = json.dumps(observation)
                is_control = observation["status"] == "error" or (
                    isinstance(result, dict) and result.get("requires_confirmation")
                )
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": obs_json if is_control else wrap_untrusted(obs_json, source=name),
                })

    def _call_llm(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        thinking_enabled: bool = True,
    ):
        """Call the configured LLM backend with streaming. Mockable in tests."""
        if True:
            extra: dict = {}
            if _uses_qwen_thinking_template(self.backend, self.model):
                # Qwen3's chat template controls thinking via the enable_thinking
                # kwarg, which the OpenAI-compatible servers (mlx-lm, omlx,
                # mira-mlx, vllm-mlx) only honor when nested under chat_template_kwargs. omlx is
                # the default backend, so it MUST be included here — otherwise the
                # per-turn thinking toggle silently falls through to the model's
                # template default (thinking ON) and "off" never takes effect.
                # Other models (gemma4, etc.) don't have this variable — skip it.
                ckwargs: dict = {"enable_thinking": thinking_enabled}
                # Send the budget even when thinking is "off". Qwen3.6 can open a
                # reasoning block anyway on some turns, and only the budget bounds
                # it. The engine arms the guard only if a block actually opens, so
                # this is inert when thinking is skipped. (The utility calls that
                # bypass this method get the same treatment via
                # _thinking_off_extra; that path was the real 2026-08-13 runaway.)
                if MAX_THINKING_TOKENS > 0:
                    ckwargs["thinking_budget"] = MAX_THINKING_TOKENS
                extra["extra_body"] = {"chat_template_kwargs": ckwargs}
            elif not thinking_enabled:
                extra["extra_body"] = {"enable_thinking": False}
            # Sampling. Before 2026-08-09 none of this was sent, so the server's
            # own 0.0 defaults applied and every reply was greedy-decoded. The
            # config defaults preserve that; mira.yaml can now change it.
            # top_k is not an OpenAI parameter, so it rides in extra_body, which
            # must be merged rather than assigned — thinking config is already
            # in there and clobbering it would silently disable the toggle.
            if TOP_K > 0:
                extra.setdefault("extra_body", {})["top_k"] = TOP_K
            # Same treatment for the repetition penalties and the seed, with one
            # extra reason to send nothing when they are unset: these keys are
            # not in the OpenAI schema, and mira-mlx is not the only backend
            # behind this client. An unconfigured install therefore puts exactly
            # the same body on the wire as before this existed.
            for name, value, ctx_size in (
                ("repetition", REPETITION_PENALTY, REPETITION_CONTEXT_SIZE),
                ("presence", PRESENCE_PENALTY, PRESENCE_CONTEXT_SIZE),
                ("frequency", FREQUENCY_PENALTY, FREQUENCY_CONTEXT_SIZE),
            ):
                if value:
                    body = extra.setdefault("extra_body", {})
                    body[f"{name}_penalty"] = value
                    body[f"{name}_context_size"] = ctx_size
            if SEED is not None:
                extra.setdefault("extra_body", {})["seed"] = SEED
            return bc.normalize_oai_stream(
                self._oai.chat.completions.create(
                    model=self.model,
                    messages=bc.normalize_messages_for_oai(messages),
                    tools=tools or None,
                    stream=True,
                    stream_options={"include_usage": True},
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    **extra,
                )
            )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def toggle_verbose(self):
        self.verbose = not self.verbose
        logger.info("Verbose mode %s.", "enabled" if self.verbose else "disabled")
        return self.verbose

    @property
    def context_pct(self) -> int:
        return ctxmgr.context_pct(self.last_prompt_tokens, self.context_window)

    def reset_conversation(self):
        self._cleanup_temp_workspace()
        self._attachment_registry = {}
        self.conversation_history = []
        self.system_prompt_added = False
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_prompt_tokens = 0
        self.conv_id = None
        self._is_new_conv = False
        self._add_system_prompt()
        self.rag_engine.load_project(self.project["id"] if self.project else None)
        logger.info("Conversation reset.")
