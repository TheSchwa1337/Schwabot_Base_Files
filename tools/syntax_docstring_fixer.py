#!/usr/bin/env python3
'''
Syntax Doc-String Fixer
=======================

Scans every ``.py`` file under the repository root, detects two common
patterns that trigger ``flake8 E999 SyntaxError`` after the bulk
doc-string injection, and repairs them in-place.

Patterns handled
+----------------
1. Trailing ``."""`` after a closed doc-string::

       """Some text."""."""

   becomes::

       """Some text."""

2. Unterminated triple-quoted strings (odd count of ``"""``). If a file
   ends with an open doc-string, the fixer appends a closing ``"""`` plus
   a newline.

3. Mixed Windows ``CRLF`` line endings are converted to ``LF`` for
   predictable cross-platform behaviour.

At the end it prints how many files were scanned and how many were
fixed.
'''
from __future__ import annotations

import ast
import pathlib
import sys
from typing import List
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent  # repo root


def iter_py_files() -> List[pathlib.Path]:
    """Yield all .py paths beneath the repo root (excluding venv etc.)."""
    EXCLUDE_DIRS = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        ".env",
    }
    for path in ROOT.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def fix_file(path: pathlib.Path) -> bool:
    """Return True if file was modified."""
    try:
        original_text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        # Skip files that cannot be decoded (e.g. binary or non-UTF8)
        return False
    # Normalize line endings to LF for cross-platform consistency
    text = original_text.replace("\r\n", "\n").replace("\r", "\n")

    # Pattern 1: remove trailing dot+triple-quote fragments like
    # """.\"\"\"  (with optional whitespace after the dot)
    text = re.sub(r'"""\.\s*"""', '"""', text)

    # fallback simple pattern for exact sequence
    text = text.replace('"""."""', '"""')

    # Pattern 2: ensure even count of triple quotes
    triple_count = text.count('"""')
    if triple_count % 2 == 1:
        # Append closing triple quote
        if not text.endswith("\n"):
            text += "\n"
        text += '"""\n'

    if text == original_text:
        return False

    # Before writing, double-check new content parses
    try:
        ast.parse(text)
    except SyntaxError:
        # Parsing still fails; do not overwrite
        return False

    try:
        path.write_text(text, encoding="utf-8")
    except OSError:
        # Disk write error – skip modification
        return False
    return True


def main() -> None:
    modified = 0
    total = 0
    skipped = 0
    try:
        for py_path in iter_py_files():
            total += 1
            # Always attempt to apply the pattern-based fixes first.
            if fix_file(py_path):
                modified += 1
    finally:
        print(f"Scanned {total} files, fixed {modified} syntax issues.")


if __name__ == "__main__":
    sys.exit(main()) 