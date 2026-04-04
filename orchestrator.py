"""Core orchestration logic for tool calling and search."""

import logging
from typing import List, Dict, Optional, Iterator

import ollama
from config import MODEL_NAME, MAX_RETRIES, MAX_TOOL_STEPS, VERBOSE_DEFAULT, RAG_MAX_CHUNKS
from tools import TOOLS
from prompts import build_system_prompt, SEARCH_RESULT_TEMPLATE
from search_engine import SearchEngine
from rag_engine import RagEngine
import url_fetcher

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Manages the conversation loop with tool calling."""

    def __init__(self, model: str = MODEL_NAME, verbose: bool = VERBOSE_DEFAULT):
        self.model = model
        self.verbose = verbose
        self.search_engine = SearchEngine()
        self.rag_engine = RagEngine()
        self.conversation_history: List[Dict] = []
        self.system_prompt_added = False
        self._add_system_prompt()

    def _add_system_prompt(self):
        if not self.system_prompt_added:
            self.conversation_history.append({
                "role": "system",
                "content": build_system_prompt()
            })
            self.system_prompt_added = True

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
          {"type": "done", "content": "..."}
          {"type": "warning", "message": "..."}
          {"type": "error", "message": "..."}

        attachments: list of dicts from file_handler.load_file() / load_file_bytes()
          {"type": "text"|"image", "name": str, "content": str, "warning": str|None}
        """
        # Emit per-file warnings (scanned PDFs, etc.)
        if attachments:
            for att in attachments:
                if att.get("warning"):
                    yield {"type": "warning", "message": att["warning"]}

        # Index RAG attachments (PDFs and large text/HTML)
        if attachments:
            for att in attachments:
                if att["type"] == "rag" and att["content"]:
                    yield {"type": "rag_indexing", "name": att["name"]}
                    try:
                        n_chunks = self.rag_engine.index(att["name"], att["content"])
                        yield {"type": "rag_done", "name": att["name"], "chunks": n_chunks}
                        # Warn if approaching memory limits
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

        # Auto-retrieve RAG context if index is non-empty
        rag_chunks = []
        if self.rag_engine.chunk_count > 0:
            try:
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

        for step in range(MAX_TOOL_STEPS):
            yield {"type": "thinking"}

            full_content = ""
            final_message = None

            # Retry loop — only retries if no tokens have been yielded yet
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    accumulated_tool_calls = None
                    for chunk in self._call_ollama(self.conversation_history, tools=TOOLS):
                        # Accumulate tool calls from any chunk — Gemma4 may emit them
                        # before the final done=True chunk, not in it.
                        if chunk.message.tool_calls:
                            accumulated_tool_calls = chunk.message.tool_calls
                            logger.info("chunk HAS tool_calls: done=%s tool_calls=%s", chunk.done, chunk.message.tool_calls)
                        token = chunk.message.content or ""
                        if token:
                            yield {"type": "token", "content": token}
                            full_content += token
                        if chunk.done:
                            final_message = chunk.message
                            # Preserve tool calls if the done chunk doesn't have them
                            if not final_message.tool_calls and accumulated_tool_calls:
                                final_message = final_message.model_copy(
                                    update={"tool_calls": accumulated_tool_calls}
                                )
                    break
                except Exception as e:
                    if full_content:
                        # Mid-stream failure — can't retry cleanly
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

                self.conversation_history.append({
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": content
                })
            else:
                logger.warning(f"Unknown tool: {tool_call.function.name}")
                self.conversation_history.append({
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": f"Error: Unknown tool {tool_call.function.name}"
                })

        yield {"type": "error", "message": f"Reached {MAX_TOOL_STEPS} tool calls without a final answer."}

    def _call_ollama(self, messages: List[Dict], tools: Optional[List] = None):
        """Call Ollama with streaming. Isolated here to make it mockable in tests."""
        return ollama.chat(model=self.model, messages=messages, tools=tools, stream=True)

    def toggle_verbose(self):
        self.verbose = not self.verbose
        status = "enabled" if self.verbose else "disabled"
        print(f"Verbose mode {status}.")
        return self.verbose

    def reset_conversation(self):
        self.conversation_history = []
        self.system_prompt_added = False
        self._add_system_prompt()
        self.rag_engine.clear()
        print("Conversation reset.")
