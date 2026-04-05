#!/usr/bin/env python3
"""Main CLI entry point for ollama Search Tool."""

import logging
import readline  # noqa: F401 — enables arrow keys and history in input()

from config import VERBOSE_DEFAULT
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


def _render_stream(orchestrator: ChatOrchestrator, user_input: str, attachments=None) -> None:
    """Consume stream_chat events and render them in the terminal."""
    spinner = None
    answer_buffer = []

    try:
        for event in orchestrator.stream_chat(user_input, attachments=attachments):
            etype = event["type"]

            if etype == "thinking":
                if spinner:
                    spinner.stop()
                spinner = console.status("[dim]Thinking…[/dim]", spinner="dots")
                spinner.start()

            elif etype == "token":
                answer_buffer.append(event["content"])

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
                if spinner:
                    spinner.stop()
                    spinner = None
                print_answer("".join(answer_buffer))
                answer_buffer.clear()
                print_rule()

            elif etype == "rag_indexing":
                if spinner:
                    spinner.stop()
                spinner = console.status(f"[dim]Indexing: {event['name']}[/dim]", spinner="dots")
                spinner.start()

            elif etype == "rag_done":
                if spinner:
                    spinner.stop()
                    spinner = None
                console.print(f"  [green]Indexed:[/green] {event['name']} ({event['chunks']} chunks)")

            elif etype == "fetch_start":
                if spinner:
                    spinner.stop()
                from urllib.parse import urlparse
                host = urlparse(event['url']).hostname or event['url']
                spinner = console.status(f"[dim]Reading: {host}[/dim]", spinner="dots")
                spinner.start()

            elif etype == "fetch_done":
                if spinner:
                    spinner.stop()
                    spinner = None
                from urllib.parse import urlparse
                host = urlparse(event['url']).hostname or event['url']
                console.print(f"  [blue]Fetched:[/blue] {host} ({event['chars']:,} chars)")

            elif etype == "fetch_context":
                if orchestrator.verbose:
                    for f in event["fetches"]:
                        console.print(f"  [dim]  Page read: {f['url']} — {f['preview'][:80]}…[/dim]")

            elif etype == "rag_context":
                if orchestrator.verbose:
                    for c in event["chunks"]:
                        console.print(f"  [dim]  RAG chunk: [{c['source']} | score {c['score']:.2f}] {c['preview'][:80]}…[/dim]")

            elif etype == "warning":
                console.print(f"  [yellow]⚠️  {event['message']}[/yellow]")

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
    staged_files = []

    while True:
        try:
            prompt = "You: " if not staged_files else f"You [📎 {', '.join(f['name'] for f in staged_files)}]: "
            user_input = input(prompt).strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                # Use the raw input for path extraction; lowercase only the command word
                parts = user_input.split(None, 1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit"):
                    print("Goodbye!")
                    break

                elif cmd == "/help":
                    print("""
Available commands:
  /help               - Show this help message
  /quit               - Exit the program
  /toggle             - Toggle verbose mode (show/hide search details)
  /verbose            - Enable verbose mode
  /quiet              - Disable verbose mode
  /reset              - Reset conversation history and RAG index
  /attach <path>      - Attach a file to the next message (PDF, HTML, image, text)
  /files              - List currently staged attachments
  /detach             - Clear all staged attachments
  /rag-list           - List documents currently in the RAG index
  /rag-remove <name>  - Remove a document from the RAG index
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

                elif cmd == "/attach":
                    if not arg:
                        print_error("Usage: /attach <path>")
                    else:
                        try:
                            from file_handler import load_file
                            att = load_file(arg)
                            staged_files.append(att)
                            if att["warning"]:
                                console.print(f"  [yellow]⚠️  {att['warning']}[/yellow]")
                            else:
                                console.print(f"  [green]Attached:[/green] {att['name']} ({att['type']})")
                        except Exception as e:
                            print_error(str(e))

                elif cmd == "/files":
                    if staged_files:
                        for f in staged_files:
                            console.print(f"  📎 {f['name']} ({f['type']})")
                    else:
                        console.print("  No files staged.")

                elif cmd == "/detach":
                    staged_files.clear()
                    console.print("  Attachments cleared.")

                elif cmd == "/rag-list":
                    docs = orchestrator.rag_engine.list_documents()
                    if docs:
                        for d in docs:
                            console.print(f"  📚 {d}")
                        console.print(f"  [dim]Total chunks: {orchestrator.rag_engine.chunk_count}[/dim]")
                    else:
                        console.print("  No documents in RAG index.")

                elif cmd == "/rag-remove":
                    if not arg:
                        print_error("Usage: /rag-remove <name>")
                    else:
                        orchestrator.rag_engine.remove(arg)
                        console.print(f"  Removed '{arg}' from RAG index.")

                else:
                    print(f"Unknown command: {cmd}. Type /help for options.")

                continue

            _render_stream(orchestrator, user_input, attachments=staged_files or None)
            staged_files = []

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
