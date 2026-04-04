"""Core orchestration logic for tool calling and search."""

import logging
from typing import List, Dict, Optional, Iterator

import ollama
from config import MODEL_NAME, MAX_RETRIES, MAX_TOOL_STEPS, VERBOSE_DEFAULT
from tools import TOOLS
from prompts import build_system_prompt, SEARCH_RESULT_TEMPLATE
from search_engine import SearchEngine

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Manages the conversation loop with tool calling."""

    def __init__(self, model: str = MODEL_NAME, verbose: bool = VERBOSE_DEFAULT):
        self.model = model
        self.verbose = verbose
        self.search_engine = SearchEngine()
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

    def stream_chat(self, user_message: str) -> Iterator[Dict]:
        """
        Process a user message and yield events for consumers (CLI, web).

        Event types:
          {"type": "thinking"}
          {"type": "token", "content": "..."}
          {"type": "search_start", "query": "..."}
          {"type": "search_done", "query": "...", "count": N, "results": [...]}
          {"type": "done", "content": "..."}
          {"type": "error", "message": "..."}
        """
        self.conversation_history.append({"role": "user", "content": user_message})

        for step in range(MAX_TOOL_STEPS):
            yield {"type": "thinking"}

            full_content = ""
            final_message = None

            # Retry loop — only retries if no tokens have been yielded yet
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    for chunk in self._call_ollama(self.conversation_history, tools=TOOLS):
                        token = chunk.message.content or ""
                        if token:
                            yield {"type": "token", "content": token}
                            full_content += token
                        if chunk.done:
                            final_message = chunk.message
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
        print("Conversation reset.")
