#!/usr/bin/env python3
"""Find TODO/FIXME comments in source files, excluding .venv, .git, __pycache__."""
import re, os

# Comment prefix patterns for common languages
COMMENT_RE = re.compile(
    r'(?P<comment_char>(?://|#|/\*|;|--|<!--))\s*'
    r'(?P<kw>TODO|FIXME)\b\s*:\s*'
    r'(?P<text>.*)',
    re.IGNORECASE
)

# Also match bare TODO/FIXME at start of a comment line (no colon)
COMMENT_BARE = re.compile(
    r'(?P<comment_char>(?://|#|/\*|;|--|<!--))\s*'
    r'(?P<kw>TODO|FIXME)\b\s*(?::\s*)?'
    r'(?P<text>.*)',
    re.IGNORECASE
)

results = {}

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ('.venv', '.git', '__pycache__')]
    for fname in files:
        path = os.path.join(root, fname)
        ext = os.path.splitext(fname)[1].lstrip('.')
        # Only process source code files
        code_exts = {
            'py', 'js', 'ts', 'tsx', 'jsx', 'go', 'rs', 'rb', 'java',
            'c', 'cpp', 'h', 'hpp', 'cs', 'php', 'swift', 'sh', 'bash',
            'r', 'scala', 'kt', 'm', 'mm', 'pl', 'pm', 'yaml', 'yml',
            'lua', 'vim', 'el', 'hs', 'ex', 'exs', 'erl', 'clj',
            'nim', 'zig', 'v', 'sol', 'vue', 'svelte',
        }
        if ext not in code_exts:
            continue
        try:
            with open(path, errors='ignore') as f:
                for lineno, line in enumerate(f, 1):
                    for pattern in (COMMENT_RE, COMMENT_BARE):
                        m = pattern.search(line)
                        if m:
                            label = m.group('kw').upper()
                            text = m.group('text').strip()
                            if not text:
                                continue
                            key = os.path.normpath(path)
                            if key not in results:
                                results[key] = []
                            results[key].append((lineno, label, text))
                            break
                        # check prefix-less TODO/FIXME (e.g., in raw prompt data)
                        # Skip lines that are part of JSON/JSONL or markdown tables
                        stripped = line.strip()
                        if stripped.startswith('{') or stripped.startswith('['):
                            continue
                        if '|' in stripped and stripped.startswith('|'):
                            continue
        except Exception:
            pass

if not results:
    print('No TODO or FIXME comments found (excluding .venv, .git, __pycache__).')
else:
    for path in sorted(results.keys()):
        print()
        print(f'### `{path}`')
        for (lineno, label, text) in results[path]:
            print(f'  - **Line {lineno}** ({label}): {text}')
