#!/usr/bin/env python3
"""Serve scripts/bench_fixtures/ over loopback HTTP for the fetch_url bench vector.

The prompt-injection bench (Q16) needs a real HTTP URL to point fetch_url at.
This binds 127.0.0.1:<port> and serves the fixtures directory, so the fetched
page carries the embedded injection with no external network dependency.

Usage:
    python scripts/serve_bench_fixtures.py [port]   # default 8009

Note: fetch_url refuses loopback/private targets unless url_fetch_allow_private
is true in mira.yaml (SSRF guard, core/config.py). Enable it for the bench run
only, and restore it afterwards.
"""
import functools
import http.server
import socketserver
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "bench_fixtures"


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8009
    if not FIXTURES_DIR.is_dir():
        print(f"ERROR: fixtures dir not found: {FIXTURES_DIR}", file=sys.stderr)
        sys.exit(1)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(FIXTURES_DIR))
    # Bind loopback only — never expose the fixtures off-host.
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving {FIXTURES_DIR} at http://127.0.0.1:{port}/  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
