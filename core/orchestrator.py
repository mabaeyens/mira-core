"""Core orchestration logic for tool calling and search."""

import logging
from typing import List, Dict, Optional, Iterator

import json

import ollama
from .config import (
    MODEL_NAME, MAX_RETRIES, MAX_TOOL_STEPS, VERBOSE_DEFAULT,
    RAG_MAX_CHUNKS, CONTEXT_WINDOW, OLLAMA_HOST,
    COMPRESS_THRESHOLD, COMPRESS_KEEP_RECENT,
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


class ChatOrchestrator:
    """Manages the conversation loop with tool calling."""

    def __init__(self, model: str = MODEL_NAME, verbose: bool = VERBOSE_DEFAULT):
        self.model = model
        self.verbose = verbose
        self._ollama = ollama.Client(host=OLLAMA_HOST)
        self.search_engine = SearchEngine()
        self.rag_engine = RagEngine()
        self.conversation_history: List[Dict] = []
        self.system_prompt_added = False
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.last_prompt_tokens: int = 0
        # Persistence
        self.conv_id: Optional[str] = None
        self._is_new_conv: bool = False
        # Project / workspace
        self.project: Optional[Dict] = None
        self._add_system_prompt()

    @property
    def workspace_root(self) -> Optional[str]:
        """Active local workspace path, or None if no local project is set."""
        return self.project.get("local_path") if self.project else None

    @property
    def _active_tools(self) -> List[Dict]:
        """Tool list filtered to what's available in the current project context."""
        if self.workspace_root:
            return TOOLS
        return [t for t in TOOLS if t["function"]["name"] not in _LOCAL_TOOLS]

    def set_project(self, project: Optional[Dict]) -> None:
        """Attach a project to the active conversation and rebuild the system prompt."""
        self.project = project
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = build_system_prompt(project=project)
        else:
            self.system_prompt_added = False
            self._add_system_prompt()

    def _add_system_prompt(self):
        if not self.system_prompt_added:
            self.conversation_history.append({
                "role": "system",
                "content": build_system_prompt(project=self.project)
            })
            self.system_prompt_added = True

    # ── Conversation lifecycle ────────────────────────────────────────────────

    def new_conversation(self, conv_id: str, project: Optional[Dict] = None) -> None:
        """Switch to a brand-new (empty) conversation."""
        self.conv_id = conv_id
        self._is_new_conv = True
        self.project = project
        self.conversation_history = []
        self.system_prompt_added = False
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_prompt_tokens = 0
        self._add_system_prompt()
        self.rag_engine.clear()

    def load_conversation(self, conv_id: str, project: Optional[Dict] = None) -> None:
        """Load an existing conversation from DB into memory."""
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
        self.rag_engine.clear()
        logger.info(f"Loaded conversation {conv_id}: {len(messages)} messages")

    # ── Post-turn helpers (called from server.py produce() thread) ────────────

    def generate_title(self, first_user_message: str) -> str:
        """One-shot LLM call to produce a short title for a new conversation."""
        try:
            resp = self._ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Reply with a short title for a conversation that starts "
                        "with this message. 4-6 words, no quotes, no trailing period:\n\n"
                        + first_user_message[:300]
                    ),
                }],
                stream=False,
            )
            return (resp.message.content or "").strip().strip("\"'").strip()[:80]
        except Exception:
            return first_user_message[:60].strip()

    def compress_history(self) -> Optional[str]:
        """
        Summarize old messages and replace them with a compact summary block.

        Returns the summary string on success, None if compression was skipped
        (too few messages) or failed.  Updates self.conversation_history in
        place.  The caller is responsible for updating the DB.
        """
        non_system = [m for m in self.conversation_history if m["role"] != "system"]
        if len(non_system) <= COMPRESS_KEEP_RECENT:
            return None

        to_compress = non_system[:-COMPRESS_KEEP_RECENT]
        to_keep = non_system[-COMPRESS_KEEP_RECENT:]

        excerpt = "\n".join(
            f"{m['role'].upper()}: {str(m.get('content', ''))[:400]}"
            for m in to_compress
        )
        try:
            resp = self._ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarize this conversation excerpt in a concise paragraph. "
                        "Preserve key facts, decisions, URLs found, and files discussed. "
                        "Be specific:\n\n" + excerpt
                    ),
                }],
                stream=False,
            )
        except Exception as e:
            logger.warning(f"Compression LLM call failed: {e}")
            return None

        summary = (resp.message.content or "").strip()
        if not summary:
            return None

        system_msgs = [m for m in self.conversation_history if m["role"] == "system"]
        self.conversation_history = system_msgs + [
            {"role": "user",      "content": f"[Earlier conversation summary]\n{summary}"},
            {"role": "assistant", "content": "Understood, I have the context."},
        ] + to_keep

        logger.info(f"Compressed {len(to_compress)} messages into summary")
        return summary

    # ── Main stream ───────────────────────────────────────────────────────────

    def stream_chat(self, user_message: str, attachments=None) -> Iterator[Dict]:
        """
        Process a user message and yield events for consumers (CLI, web).

        Event types:
          {"type": "thinking"}
          {"type": "token", "content": "..."}
          {"type": "search_start", "query": "..."}
          {"type": "search_done", "query": "...", "count": N, "results": [...]}
          {"type": "fetch_start", "url": "..."}
          {"type": "fetch_done", "url": "...", "chars": N}
          {"type": "fetch_context", "fetches": [...]}
          {"type": "rag_context", "chunks": [...]}
          {"type": "stats", "input_tokens": N, "output_tokens": N, "context_pct": N}
          {"type": "done", "content": "..."}
          {"type": "warning", "message": "..."}
          {"type": "error", "message": "..."}

        attachments: list of dicts from file_handler.load_file() / load_file_bytes()
          {"type": "text"|"image"|"rag", "name": str, "content": str, "warning": str|None}
        """
        # Emit per-file warnings (scanned PDFs, binary files, etc.)
        if attachments:
            for att in attachments:
                if att.get("warning"):
                    yield {"type": "warning", "message": att["warning"]}

        # Index RAG attachments (PDFs and large text/HTML)
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

        # Auto-retrieve RAG context if index is non-empty.
        # Bypass score threshold when files were just indexed (meta-query guard).
        rag_chunks = []
        if self.rag_engine.chunk_count > 0:
            try:
                if rag_indexed_this_turn:
                    rag_chunks = self.rag_engine.query(user_message, score_threshold=float('-inf'))
                else:
                    rag_chunks = self.rag_engine.query(user_message)
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        # Build the user message: RAG context + text attachments + user text + images
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

        user_msg: Dict = {"role": "user", "content": full_message}
        if images:
            user_msg["images"] = images

        self.conversation_history.append(user_msg)

        fetch_results = []

        for step in range(MAX_TOOL_STEPS):
            yield {"type": "thinking"}

            full_content = ""
            final_message = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    accumulated_tool_calls = None
                    for chunk in self._call_ollama(self.conversation_history, tools=self._active_tools):
                        if chunk.message.tool_calls:
                            accumulated_tool_calls = chunk.message.tool_calls
                            logger.info("chunk HAS tool_calls: done=%s tool_calls=%s", chunk.done, chunk.message.tool_calls)
                        token = chunk.message.content or ""
                        if token:
                            yield {"type": "token", "content": token}
                            full_content += token
                        if chunk.done:
                            final_message = chunk.message
                            if not final_message.tool_calls and accumulated_tool_calls:
                                final_message = final_message.model_copy(
                                    update={"tool_calls": accumulated_tool_calls}
                                )
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
                    logger.warning(f"Ollama API error (attempt {attempt}/{MAX_RETRIES}): {e}")

            if final_message:
                logger.info(
                    "Ollama response — content_len=%d tool_calls=%s thinking_len=%d",
                    len(final_message.content or ""),
                    bool(final_message.tool_calls),
                    len(final_message.thinking or ""),
                )

            tool_calls = final_message.tool_calls if final_message else None

            if not tool_calls:
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
                yield {
                    "type": "stats",
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "context_pct": self.context_pct,
                }
                yield {"type": "done", "content": full_content}
                return

            # Model requested a tool call
            self.conversation_history.append(final_message)
            tool_call = tool_calls[0]

            if tool_call.function.name == "web_search":
                query = tool_call.function.arguments.get("query", "")
                num_results = tool_call.function.arguments.get("num_results", 5)

                yield {"type": "search_start", "query": query}

                try:
                    results = self.search_engine.search(query, max_results=num_results)
                except Exception as e:
                    logger.error(f"Search failed: {e}")
                    results = []

                yield {"type": "search_done", "query": query, "count": len(results), "results": results}

                self.conversation_history.append({
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": SEARCH_RESULT_TEMPLATE.format(
                        query=query,
                        results_text=self.search_engine.get_search_summary(results)
                    )
                })
            elif tool_call.function.name == "fetch_url":
                url = tool_call.function.arguments.get("url", "")

                yield {"type": "fetch_start", "url": url}

                content = url_fetcher.fetch_url(url)

                yield {"type": "fetch_done", "url": url, "chars": len(content)}

                fetch_results.append({
                    "url": url,
                    "chars": len(content),
                    "preview": content[:300].rstrip() + ("…" if len(content) > 300 else ""),
                })

                self.conversation_history.append({
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": content
                })
            else:
                name = tool_call.function.name
                args = tool_call.function.arguments
                label_start, label_done_fn = _tool_ui_labels(name, args)
                yield {"type": "tool_start", "tool": name, "label": label_start}
                result = self._dispatch_tool(name, args)
                yield {"type": "tool_done", "tool": name, "label": label_done_fn(result)}
                self.conversation_history.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result),
                })

        yield {"type": "error", "message": f"Reached {MAX_TOOL_STEPS} tool calls without a final answer."}

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, args: dict) -> dict:
        """Dispatch a tool call and return its result dict."""
        dispatch = {
            # Filesystem (root passed so each call uses the active project workspace)
            "read_file":    lambda a: fs_tools.read_file(a.get("path", ""), root=self.workspace_root),
            "write_file":   lambda a: fs_tools.write_file(a.get("path", ""), a.get("content", ""), root=self.workspace_root),
            "edit_file":    lambda a: fs_tools.edit_file(a.get("path", ""), a.get("old_str", ""), a.get("new_str", ""), root=self.workspace_root),
            "list_files":   lambda a: fs_tools.list_files(a.get("path", "."), a.get("recursive", False), root=self.workspace_root),
            "search_files": lambda a: fs_tools.search_files(a.get("pattern", ""), a.get("path", "."), a.get("case_sensitive", False), root=self.workspace_root),
            "move_file":    lambda a: fs_tools.move_file(a.get("src", ""), a.get("dst", ""), root=self.workspace_root),
            "delete_file":  lambda a: fs_tools.delete_file(a.get("path", ""), a.get("confirm", False), root=self.workspace_root),
            # Shell
            "run_shell":    lambda a: shell_tools.run_shell(a.get("command", ""), a.get("cwd", "."), a.get("force", False), root=self.workspace_root),
            # GitHub — clone + register
            "github_clone_repo":   lambda a: self._clone_and_register(a),
            # GitHub — read
            "github_list_repos":   lambda a: github_tools.github_list_repos(a.get("repo_type", "owner")),
            "github_read_file":    lambda a: github_tools.github_read_file(a["repo"], a["path"], a.get("ref", "")),
            "github_list_files":   lambda a: github_tools.github_list_files(a["repo"], a.get("path", ""), a.get("ref", "")),
            "github_list_issues":  lambda a: github_tools.github_list_issues(a["repo"], a.get("state", "open")),
            "github_list_prs":     lambda a: github_tools.github_list_prs(a["repo"], a.get("state", "open")),
            "github_search_code":  lambda a: github_tools.github_search_code(a["query"], a.get("repo", "")),
            # GitHub — write
            "github_write_file":   lambda a: github_tools.github_write_file(a["repo"], a["path"], a["content"], a["message"], a.get("branch", ""), a.get("sha", "")),
            "github_create_repo":  lambda a: github_tools.github_create_repo(a["name"], a.get("private", True), a.get("description", ""), a.get("auto_init", True)),
            "github_create_issue": lambda a: github_tools.github_create_issue(a["repo"], a["title"], a.get("body", "")),
            "github_create_branch":lambda a: github_tools.github_create_branch(a["repo"], a["branch"], a.get("from_ref", "")),
            # GitHub — destructive
            "github_create_pr":    lambda a: github_tools.github_create_pr(a["repo"], a["title"], a.get("body", ""), a.get("head", ""), a.get("base", "")),
            "github_merge_pr":     lambda a: github_tools.github_merge_pr(a["repo"], a["pr_number"], a.get("merge_method", "merge"), a.get("confirm", False)),
            "github_delete_file":  lambda a: github_tools.github_delete_file(a["repo"], a["path"], a["message"], a.get("branch", ""), a.get("confirm", False)),
            "github_delete_branch":lambda a: github_tools.github_delete_branch(a["repo"], a["branch"], a.get("confirm", False)),
        }
        fn = dispatch.get(name)
        if fn is None:
            logger.warning(f"Unknown tool: {name}")
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(args)
        except Exception as e:
            logger.error(f"Tool {name} raised: {e}")
            return {"error": str(e)}

    def _clone_and_register(self, args: dict) -> dict:
        """Clone a GitHub repo and register it as a Mira project in the DB."""
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

    def _call_ollama(self, messages: List[Dict], tools: Optional[List] = None):
        """Call Ollama with streaming. Isolated here to make it mockable in tests."""
        return self._ollama.chat(model=self.model, messages=messages, tools=tools, stream=True)

    def toggle_verbose(self):
        self.verbose = not self.verbose
        status = "enabled" if self.verbose else "disabled"
        print(f"Verbose mode {status}.")
        return self.verbose

    @property
    def context_pct(self) -> int:
        """Current context window usage as a percentage (0–100)."""
        if not CONTEXT_WINDOW or self.last_prompt_tokens == 0:
            return 0
        return min(100, round(self.last_prompt_tokens / CONTEXT_WINDOW * 100))

    def reset_conversation(self):
        """Clear in-memory state. Caller is responsible for creating a new DB record."""
        self.conversation_history = []
        self.system_prompt_added = False
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_prompt_tokens = 0
        self.conv_id = None
        self._is_new_conv = False
        self._add_system_prompt()
        self.rag_engine.clear()
        print("Conversation reset.")
