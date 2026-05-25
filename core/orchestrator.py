"""Core orchestration logic for tool calling and search."""

import json
import logging
import os
import re
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Iterator

import ollama
import openai as _openai

from .config import (
    MODEL_NAME, BACKEND, OLLAMA_HOST,
    MAX_RETRIES, MAX_TOOL_STEPS, MAX_AGENT_STEPS, AGENT_DIVERGENCE_LIMIT,
    MAX_TOOL_CALLS_PER_TURN, SAME_TOOL_REPEAT_LIMIT, VERBOSE_DEFAULT,
    RAG_MAX_CHUNKS, CONTEXT_WINDOW,
    COMPRESS_THRESHOLD, COMPRESS_KEEP_RECENT,
    THINKING_MODE,
)
from .tools import TOOLS, _LOCAL_TOOLS
from .prompts import build_system_prompt, SEARCH_RESULT_TEMPLATE
from .search_engine import SearchEngine
from .rag_engine import RagEngine
from . import url_fetcher
from . import fs_tools
from . import shell_tools
from . import github_tools

logger = logging.getLogger(__name__)


def _tool_ui_labels(name: str, args: dict):
    """Return (start_label, done_label_fn) for a tool call."""
    def _err(r): return r.get("error", "")
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


def _make_oai_client(host: str) -> _openai.OpenAI:
    """Create an OpenAI-compatible client pointed at the given host."""
    base = host.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return _openai.OpenAI(base_url=base, api_key=_read_omlx_api_key())


_THINK_VERBS = re.compile(
    r"\b(why|how|fix|debug|implement|refactor|explain|design|"
    r"analyze|review|optimize|architect|plan|compare|difference|tradeoff)\b",
    re.IGNORECASE,
)
_CODE_SIGNAL = re.compile(r"```|def |class |import |error:|traceback", re.IGNORECASE)
# Short acknowledgements that are never complex regardless of other signals.
_TRIVIAL = re.compile(
    r"^\s*(ok|okay|thanks|thank you|got it|sounds good|perfect|great|sure|yes|no|yep|nope|cool)\s*[.!]?\s*$",
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
    score = 0
    if has_attachments:
        score += 3
    if len(message) > 500:
        score += 2
    elif len(message) > 150:
        score += 1
    if _CODE_SIGNAL.search(message):
        score += 2
    if _THINK_VERBS.search(message):
        score += 1
    return score >= 3


class ChatOrchestrator:
    """Manages the conversation loop with tool calling."""

    def __init__(self, model: str = MODEL_NAME, verbose: bool = VERBOSE_DEFAULT):
        self.model = model
        self.verbose = verbose
        self.backend = BACKEND
        self.context_window = CONTEXT_WINDOW
        self.thinking_mode = THINKING_MODE

        if self.backend == "ollama":
            self._ollama = ollama.Client(host=OLLAMA_HOST)
            self._oai = None
        else:
            self._ollama = None
            self._oai = _make_oai_client(OLLAMA_HOST)

        self.search_engine = SearchEngine()
        self.rag_engine = RagEngine()
        self.conversation_history: List[Dict] = []
        self.system_prompt_added = False
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.last_prompt_tokens: int = 0
        self.conv_id: Optional[str] = None
        self._is_new_conv: bool = False
        self.project: Optional[Dict] = None
        self._add_system_prompt()

    @property
    def workspace_root(self) -> Optional[str]:
        return self.project.get("local_path") if self.project else None

    @property
    def _active_tools(self) -> List[Dict]:
        if self.workspace_root:
            return TOOLS
        return [t for t in TOOLS if t["function"]["name"] not in _LOCAL_TOOLS]

    def set_project(self, project: Optional[Dict]) -> None:
        self.project = project
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = build_system_prompt(project=project)
        else:
            self.system_prompt_added = False
            self._add_system_prompt()

    def reinitialize_client(self, backend: str, model: str, host: str,
                            embed_backend: str, embed_host: str,
                            context_window: int) -> None:
        """Switch to a different inference backend at runtime without restarting."""
        self.backend = backend
        self.model = model
        self.context_window = context_window
        if backend == "ollama":
            self._ollama = ollama.Client(host=host)
            self._oai = None
        else:
            self._ollama = None
            self._oai = _make_oai_client(host)
        self.rag_engine.reinitialize_client(embed_backend, embed_host)

    def _add_system_prompt(self):
        if not self.system_prompt_added:
            self.conversation_history.append({
                "role": "system",
                "content": build_system_prompt(project=self.project)
            })
            self.system_prompt_added = True

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

    def load_conversation(self, conv_id: str, project: Optional[Dict] = None) -> None:
        from . import db
        messages = db.load_messages(conv_id)
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
        """Non-streaming single-turn LLM call. Returns the text content."""
        if self.backend == "ollama":
            resp = self._ollama.chat(
                model=self.model,
                messages=_normalize_messages_for_ollama(messages),
                stream=False,
                **({"format": format} if format else {}),
            )
            return (resp.message.content or "").strip()
        else:
            resp = self._oai.chat.completions.create(
                model=self.model,
                messages=messages,
                **({"response_format": {"type": "json_object"}} if format else {}),
            )
            return _strip_think((resp.choices[0].message.content or "").strip())

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
            # Strip markdown fences some models wrap around JSON output
            stripped = re.sub(r"^```[a-z]*\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            return json.loads(stripped).get("title", raw)[:80]
        except Exception:
            return first_user_message[:60].strip()

    def compress_history(self) -> Optional[str]:
        non_system = [m for m in self.conversation_history if m["role"] != "system"]
        if len(non_system) <= COMPRESS_KEEP_RECENT:
            return None

        to_compress = non_system[:-COMPRESS_KEEP_RECENT]
        to_keep = non_system[-COMPRESS_KEEP_RECENT:]

        excerpt = "\n".join(
            f"{m['role'].upper()}: {str(m.get('content', ''))[:2000]}"
            for m in to_compress
        )
        try:
            summary = self._llm_chat_sync([{
                "role": "user",
                "content": (
                    "Summarize this conversation excerpt in a concise paragraph. "
                    "Preserve key facts, decisions, URLs found, and files discussed. "
                    "Be specific:\n\n" + excerpt
                ),
            }])
        except Exception as e:
            logger.warning("Compression LLM call failed: %s", e)
            return None

        if not summary:
            return None

        system_msgs = [m for m in self.conversation_history if m["role"] == "system"]
        self.conversation_history = system_msgs + [
            {"role": "user",      "content": f"[Earlier conversation summary]\n{summary}"},
            {"role": "assistant", "content": "Understood, I have the context."},
        ] + to_keep

        logger.info("Compressed %d messages into summary", len(to_compress))
        return summary

    # ── Main stream ───────────────────────────────────────────────────────────

    def stream_chat(
        self,
        user_message: str,
        attachments=None,
        thinking_enabled: bool = True,
    ) -> Iterator[Dict]:
        """
        Process a user message and yield events for consumers (CLI, web).

        Event types: thinking, token, search_start/done, fetch_start/done/context,
        rag_indexing/done/context, stats, warning, done, error.
        """
        if attachments:
            for att in attachments:
                if att.get("warning"):
                    yield {"type": "warning", "message": att["warning"]}

        # Adaptive thinking: heuristic decides, but client "force on" always wins
        if self.thinking_mode == "adaptive":
            thinking_enabled = thinking_enabled or _should_think(user_message, has_attachments=bool(attachments))
        elif self.thinking_mode == "always":
            thinking_enabled = True
        elif self.thinking_mode == "never":
            thinking_enabled = False

        rag_indexed_this_turn = False
        if attachments:
            for att in attachments:
                if att["type"] == "rag" and att["content"]:
                    yield {"type": "rag_indexing", "name": att["name"]}
                    try:
                        n_chunks = self.rag_engine.index(att["name"], att["content"])
                        yield {"type": "rag_done", "name": att["name"], "chunks": n_chunks}
                        rag_indexed_this_turn = True
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
                f"[File: {att['name']}]\n{att['content']}\n---"
                for att in attachments
                if att["type"] == "text" and att["content"]
            ]
            images = [att["content"] for att in attachments if att["type"] == "image"]
            if text_parts:
                full_message = '\n\n'.join(text_parts) + '\n\n' + full_message

        if rag_chunks:
            context = "\n\n".join(
                f"[Source: {c['source']} | Score: {c['score']:.2f}]\n{c['text']}"
                for c in rag_chunks
            )
            full_message = f"[Relevant document sections]\n{context}\n\n---\n\n{full_message}"

        # Inject CURRENT TASK block for multi-step requests to anchor the goal
        if _MULTI_STEP_RE.search(user_message):
            full_message = f"CURRENT TASK: {user_message}\n\n{full_message}"

        user_msg: Dict = {"role": "user", "content": full_message}
        if images:
            user_msg["images"] = images

        self.conversation_history.append(user_msg)

        fetch_results = []
        _thinking_chars = 0  # accumulated across all tool steps for this turn
        self._task_done = False
        self._task_done_summary = ""
        self._tool_call_hashes: dict = {}  # call_hash -> repeat count
        self._total_tool_calls = 0           # hard cap: MAX_TOOL_CALLS_PER_TURN
        self._tool_name_counts: dict = {}    # tool name -> call count for SAME_TOOL_REPEAT_LIMIT

        for step in range(MAX_AGENT_STEPS):
            if thinking_enabled:
                yield {"type": "thinking"}

            # Soft step limit: warn the model at the halfway point and again near the hard cap.
            if step == MAX_AGENT_STEPS // 2:
                self.conversation_history.append({
                    "role": "user",
                    "content": (
                        f"[System: You have made {step} tool calls so far. "
                        f"If your task is complete, call task_done now with a summary. "
                        f"Otherwise consolidate your remaining work into as few commands as possible — "
                        f"the hard limit is {MAX_AGENT_STEPS} total steps.]"
                    ),
                })
            elif step == MAX_AGENT_STEPS - 3:
                self.conversation_history.append({
                    "role": "user",
                    "content": (
                        f"[System: URGENT — only {MAX_AGENT_STEPS - step} steps remain before hard cutoff. "
                        f"You MUST call task_done NOW with a summary of what you have found so far. "
                        f"Do not make any more tool calls.]"
                    ),
                })
            elif step == MAX_AGENT_STEPS - 2 and self._total_tool_calls > 0 and not self._task_done:
                # Hard forced summary: model has been running too long — synthesize a
                # response from what it has gathered so far.
                try:
                    forced = self._llm_chat_sync(
                        self.conversation_history + [{
                            "role": "user",
                            "content": "Summarize what you have accomplished or found so far, in 2-3 sentences.",
                        }]
                    )
                    if forced:
                        self._mark_task_done(forced)
                        if _thinking_chars:
                            self.total_output_tokens += round(_thinking_chars / 3.5)
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

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    accumulated_tool_calls = None
                    # Buffer for <think> tag stripping
                    think_buf = ""
                    in_thinking = False

                    for chunk in self._call_llm(
                        self.conversation_history,
                        tools=self._active_tools,
                        thinking_enabled=thinking_enabled,
                    ):
                        if chunk.message.tool_calls:
                            accumulated_tool_calls = chunk.message.tool_calls
                            logger.info("chunk HAS tool_calls: done=%s tool_calls=%s", chunk.done, chunk.message.tool_calls)

                        # Ollama yields thinking content in chunk.message.thinking when
                        # think=True; emit it as a thinking event so the UI can show it
                        # collapsed, then continue to the normal content path.
                        thinking_token = getattr(chunk.message, "thinking", None) or ""
                        if thinking_token:
                            _thinking_chars += len(thinking_token)
                            yield {"type": "thinking", "content": thinking_token}

                        raw_token = chunk.message.content or ""
                        if raw_token:
                            think_buf += raw_token
                            # Process buffered content, stripping <think>...</think>
                            while think_buf:
                                if in_thinking:
                                    close = think_buf.find("</think>")
                                    if close == -1:
                                        think_buf = ""  # all thinking, consume
                                        break
                                    in_thinking = False
                                    think_buf = think_buf[close + len("</think>"):]
                                else:
                                    open_tag = think_buf.find("<think>")
                                    if open_tag == -1:
                                        yield {"type": "token", "content": think_buf}
                                        full_content += think_buf
                                        think_buf = ""
                                        break
                                    if open_tag > 0:
                                        regular = think_buf[:open_tag]
                                        yield {"type": "token", "content": regular}
                                        full_content += regular
                                        think_buf = think_buf[open_tag:]
                                    else:
                                        in_thinking = True
                                        think_buf = think_buf[len("<think>"):]

                        if chunk.done:
                            final_message = chunk.message
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
                "LLM response — content_len=%d tool_calls=%s thinking_len=%d",
                len(final_message.content or ""),
                bool(final_message.tool_calls),
                len(getattr(final_message, 'thinking', None) or ""),
            )

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
                # Ollama's eval_count covers only the visible content tokens; thinking
                # tokens arrive separately via chunk.message.thinking and are not included.
                # Convert accumulated thinking chars to approximate tokens (~3.5 chars/tok)
                # and fold into total_output_tokens so the display reflects actual compute.
                if _thinking_chars:
                    self.total_output_tokens += round(_thinking_chars / 3.5)
                yield {
                    "type": "stats",
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "context_pct": self.context_pct,
                }
                yield {"type": "done", "content": full_content, "task_done": self._task_done}
                return

            # Model requested tool call(s) — normalize history append
            self.conversation_history.append(_message_to_history_dict(final_message))

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

            # Hard caps: bail before executing if limits are exceeded
            if self._total_tool_calls > MAX_TOOL_CALLS_PER_TURN:
                yield {"type": "error", "message": f"Stopped: exceeded {MAX_TOOL_CALLS_PER_TURN} tool calls in one turn."}
                return
            over_limit = [(n, c) for n, c in self._tool_name_counts.items() if c > SAME_TOOL_REPEAT_LIMIT]
            if over_limit:
                names_str = ", ".join(f"{n}×{c}" for n, c in over_limit)
                yield {"type": "error", "message": f"Stopped: {names_str} — same tool called too many times in one turn."}
                return

            # task_done short-circuits the whole batch immediately
            if any(name == "task_done" for _, name, _, _ in prepared):
                task_done_args = next(args for _, name, args, _ in prepared if name == "task_done")
                self._mark_task_done(task_done_args.get("summary", "Task complete."))
                if _thinking_chars:
                    self.total_output_tokens += round(_thinking_chars / 3.5)
                yield {
                    "type": "stats",
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "context_pct": self.context_pct,
                }
                yield {"type": "done", "content": self._task_done_summary, "task_done": True}
                return

            # Emit start events for all tools upfront before parallel execution
            for tc_id, name, args, diverged in prepared:
                if name == "web_search":
                    yield {"type": "search_start", "query": args.get("query", "")}
                elif name == "fetch_url":
                    yield {"type": "fetch_start", "url": args.get("url", "")}
                else:
                    label_start, _ = _tool_ui_labels(name, args)
                    yield {"type": "tool_start", "tool": name, "label": label_start}

            # Execute all tool calls in parallel
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

            # Emit results and append to history in original order
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
                            results_text=self.search_engine.format_tool_result(web_results)
                        ),
                    })
                elif name == "fetch_url":
                    url = result.get("url", "")
                    content = result.get("content", "")
                    yield {"type": "fetch_done", "url": url, "chars": len(content)}
                    yield {"type": "agent_step", "step": step + 1, "tool": "fetch_url", "status": "success"}
                    fetch_results.append({
                        "url": url,
                        "chars": len(content),
                        "preview": content[:300].rstrip() + ("…" if len(content) > 300 else ""),
                    })
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": name,
                        "content": content,
                    })
                else:
                    _, label_done_fn = _tool_ui_labels(name, args)
                    observation = self._wrap_observation(name, result)
                    yield {"type": "tool_done", "tool": name, "label": label_done_fn(result)}
                    yield {"type": "agent_step", "step": step + 1, "tool": name, "status": observation["status"]}
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": name,
                        "content": json.dumps(observation),
                    })

        yield {"type": "error", "message": f"Reached {MAX_AGENT_STEPS} tool calls without a final answer."}

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _wrap_observation(self, tool_name: str, result: dict) -> dict:
        """Normalise a tool result into a structured observation for the model."""
        if isinstance(result, dict) and "error" in result:
            return {"status": "error", "payload": None, "error_details": result["error"]}
        return {"status": "success", "payload": result}

    def _mark_task_done(self, summary: str) -> dict:
        self._task_done = True
        self._task_done_summary = summary
        return {"done": True}

    def _dispatch_tool(self, name: str, args: dict) -> dict:
        dispatch = {
            "read_file":    lambda a: fs_tools.read_file(a.get("path", ""), root=self.workspace_root),
            "write_file":   lambda a: fs_tools.write_file(a.get("path", ""), a.get("content", ""), root=self.workspace_root),
            "edit_file":    lambda a: fs_tools.edit_file(a.get("path", ""), a.get("old_str", ""), a.get("new_str", ""), root=self.workspace_root),
            "list_files":   lambda a: fs_tools.list_files(a.get("path", "."), a.get("recursive", False), root=self.workspace_root),
            "search_files": lambda a: fs_tools.search_files(a.get("pattern", ""), a.get("path", "."), a.get("case_sensitive", False), root=self.workspace_root),
            "move_file":    lambda a: fs_tools.move_file(a.get("src", ""), a.get("dst", ""), root=self.workspace_root),
            "delete_file":  lambda a: fs_tools.delete_file(a.get("path", ""), a.get("confirm", False), root=self.workspace_root),
            "run_shell":    lambda a: shell_tools.run_shell(a.get("command", ""), a.get("cwd", "."), a.get("force", False), root=self.workspace_root, timeout=a.get("timeout", 30)),
            "github_clone_repo":    lambda a: self._clone_and_register(a),
            "github_list_repos":    lambda a: github_tools.github_list_repos(a.get("repo_type", "owner")),
            "github_read_file":     lambda a: github_tools.github_read_file(a["repo"], a["path"], a.get("ref", "")),
            "github_list_files":    lambda a: github_tools.github_list_files(a["repo"], a.get("path", ""), a.get("ref", "")),
            "github_list_issues":   lambda a: github_tools.github_list_issues(a["repo"], a.get("state", "open")),
            "github_list_prs":      lambda a: github_tools.github_list_prs(a["repo"], a.get("state", "open")),
            "github_search_code":   lambda a: github_tools.github_search_code(a["query"], a.get("repo", "")),
            "github_write_file":    lambda a: github_tools.github_write_file(a["repo"], a["path"], a["content"], a["message"], a.get("branch", ""), a.get("sha", "")),
            "github_create_repo":   lambda a: github_tools.github_create_repo(a["name"], a.get("private", True), a.get("description", ""), a.get("auto_init", True)),
            "github_create_issue":  lambda a: github_tools.github_create_issue(a["repo"], a["title"], a.get("body", "")),
            "github_create_branch": lambda a: github_tools.github_create_branch(a["repo"], a["branch"], a.get("from_ref", "")),
            "github_create_pr":     lambda a: github_tools.github_create_pr(a["repo"], a["title"], a.get("body", ""), a.get("head", ""), a.get("base", "")),
            "github_merge_pr":      lambda a: github_tools.github_merge_pr(a["repo"], a["pr_number"], a.get("merge_method", "merge"), a.get("confirm", False)),
            "github_delete_file":   lambda a: github_tools.github_delete_file(a["repo"], a["path"], a["message"], a.get("branch", ""), a.get("confirm", False)),
            "github_delete_branch": lambda a: github_tools.github_delete_branch(a["repo"], a["branch"], a.get("confirm", False)),
            "task_done":            lambda a: self._mark_task_done(a.get("summary", "Task complete.")),
        }
        fn = dispatch.get(name)
        if fn is None:
            logger.warning("Unknown tool: %s", name)
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(args)
        except Exception as e:
            logger.error("Tool %s raised: %s", name, e)
            return {"error": str(e)}

    def _clone_and_register(self, args: dict) -> dict:
        from . import db
        result = github_tools.github_clone_repo(args["repo"], args.get("dest", ""))
        if "error" in result:
            return result
        repo_name = args["repo"].split("/")[-1]
        project_name = args.get("project_name", "").strip() or repo_name
        project_id = db.create_project(project_name, local_path=result["cloned_to"], github_repo=args["repo"])
        result["project_id"] = project_id
        result["project_name"] = project_name
        return result

    # ── LLM backend ──────────────────────────────────────────────────────────

    def _call_llm(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        thinking_enabled: bool = True,
    ):
        """Call the configured LLM backend with streaming. Mockable in tests."""
        if self.backend == "ollama":
            msgs = _normalize_messages_for_ollama(messages)
            if not thinking_enabled:
                # Belt-and-suspenders: Qwen3 respects /no_think in the system prompt
                # independently of the `think` API parameter (Ollama version-agnostic).
                msgs = _inject_no_think(msgs)
            return self._ollama.chat(
                model=self.model, messages=msgs,
                tools=tools, stream=True, think=thinking_enabled,
            )
        else:
            extra: dict = {}
            if not thinking_enabled:
                extra["extra_body"] = {"enable_thinking": False}
            return self._normalize_oai_stream(
                self._oai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools or None,
                    stream=True,
                    stream_options={"include_usage": True},
                    **extra,
                )
            )

    def _normalize_oai_stream(self, stream):
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
            is_done = finish_reason is not None

            msg = types.SimpleNamespace(content=content, tool_calls=None, thinking="")
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

    # ── Utilities ─────────────────────────────────────────────────────────────

    def toggle_verbose(self):
        self.verbose = not self.verbose
        logger.info("Verbose mode %s.", "enabled" if self.verbose else "disabled")
        return self.verbose

    @property
    def context_pct(self) -> int:
        if not self.context_window or self.last_prompt_tokens == 0:
            return 0
        return min(100, round(self.last_prompt_tokens / self.context_window * 100))

    def reset_conversation(self):
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


# ── Module-level helpers ─────────────────────────────────────────────────────

def _inject_no_think(messages: List[Dict]) -> List[Dict]:
    """Prepend /no_think to the system message so Qwen3 skips chain-of-thought
    regardless of Ollama version. Safe to call on any message list."""
    result = list(messages)
    if result and result[0].get("role") == "system":
        content = result[0].get("content", "")
        if not content.startswith("/no_think"):
            result[0] = {**result[0], "content": "/no_think\n" + content}
    return result


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from a string."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _normalize_messages_for_ollama(messages: List[Dict]) -> List[Dict]:
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


def _message_to_history_dict(msg) -> dict:
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
