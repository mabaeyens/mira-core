#!/usr/bin/env python3
"""Main CLI entry point for ollama Search Tool."""

import sys
import logging
import readline  # noqa: F401 — enables arrow keys and history in input()

from config import VERBOSE_DEFAULT, ANSWER_PREFIX
from orchestrator import ChatOrchestrator
from formatter import (
    console, print_header, print_search_status, print_search_results,
    print_error, print_rule
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def _render_stream(orchestrator: ChatOrchestrator, user_input: str) -> None:
    """Consume stream_chat events and render them in the terminal."""
    spinner = None
    started_answer = False

    try:
        for event in orchestrator.stream_chat(user_input):
            etype = event["type"]

            if etype == "thinking":
                if spinner:
                    spinner.stop()
                spinner = console.status("[dim]Thinking…[/dim]", spinner="dots")
                spinner.start()

            elif etype == "token":
                if spinner:
                    spinner.stop()
                    spinner = None
                if not started_answer:
                    sys.stdout.write(f"\n{ANSWER_PREFIX}")
                    sys.stdout.flush()
                    started_answer = True
                sys.stdout.write(event["content"])
                sys.stdout.flush()

            elif etype == "search_start":
                if spinner:
                    spinner.stop()
                spinner = console.status(f"[dim]Searching: {event['query']}[/dim]", spinner="dots")
                spinner.start()

            elif etype == "search_done":
                if spinner:
                    spinner.stop()
                    spinner = None
                count = event["count"]
                query = event["query"]
                if count > 0:
                    print_search_status(query, f"Found {count} results")
                    if orchestrator.verbose:
                        print_search_results(event["results"])
                else:
                    print_search_status(query, "No results found")

            elif etype == "done":
                if started_answer:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                print_rule()

            elif etype == "error":
                print_error(event["message"])

    finally:
        if spinner:
            spinner.stop()


def main():
    """Main entry point."""
    print_header("🤖 ollama Search Tool (Gemma 4 + Ollama)")
    print("Type your message. Use /help for commands, /quit to exit.\n")

    orchestrator = ChatOrchestrator(verbose=VERBOSE_DEFAULT)

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input.lower()

                if cmd in ("/quit", "/exit"):
                    print("Goodbye!")
                    break

                elif cmd == "/help":
                    print("""
Available commands:
  /help      - Show this help message
  /quit      - Exit the program
  /toggle    - Toggle verbose mode (show/hide search details)
  /verbose   - Enable verbose mode
  /quiet     - Disable verbose mode
  /reset     - Reset conversation history
                    """)

                elif cmd == "/toggle":
                    orchestrator.toggle_verbose()

                elif cmd == "/verbose":
                    orchestrator.verbose = True
                    print("Verbose mode enabled.")

                elif cmd == "/quiet":
                    orchestrator.verbose = False
                    print("Verbose mode disabled.")

                elif cmd == "/reset":
                    orchestrator.reset_conversation()

                else:
                    print(f"Unknown command: {cmd}. Type /help for options.")

                continue

            _render_stream(orchestrator, user_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
