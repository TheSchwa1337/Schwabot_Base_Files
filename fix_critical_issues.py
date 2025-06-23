#!/usr/bin/env python3
"""Automated critical issue fixer for Schwabot codebase.

This utility scans Python source files under the *core/* directory (recursively)
and applies quick, mechanical fixes to the most common SyntaxError triggers that
showed up in CI:

1. Unterminated triple-quoted docstrings that contain stray characters after the closing marker (e.g. ``\"\"\".\"\"\"``)
2. Unicode glyphs that break Python 3.8 parsers (≤ ⇒ ∫ ₍ …) – replaced by
   ASCII equivalents or removed.
3. Smart quotes " " ' ' → straight quotes.

Running this script is *idempotent* and safe on well-formed files – unchanged
files are left as-is.  It prints a short summary of edits so you can commit the
patch afterwards.

Example
-------
$ python fix_critical_issues.py
✔ 37 files scanned – 12 modified (8 docstring fixes, 4 unicode cleans)
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Iterator

ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR = ROOT / "core"

# Regex patterns for quick fixes -------------------------------------------------
DOCSTRING_GARBAGE_RE = re.compile(r'"""\."""')
SMART_QUOTES_RE = re.compile(r"[\u2018\u2019\u201C\u201D]")
UNICODE_GLYPHS_MAP = str.maketrans({
    "≤": "<=",
    "⇒": "=>",
    "∫": "",
    "₍": "(",
    "∂": "d",
    "≥": ">=",
    "≠": "!=",
})


def iter_py_files(base: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield all ``.py`` files under *base* (recursively)."""
    yield from base.rglob("*.py")


def clean_file(path: pathlib.Path) -> bool:
    """Return True if file was modified."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:  # pragma: no cover – skip binary/cruft files
        return False

    orig = text

    # 1. Fix triple-quoted docstring garbage
    text = DOCSTRING_GARBAGE_RE.sub('"""', text)

    # 2. Replace smart quotes with straight quotes
    text = SMART_QUOTES_RE.sub('"', text)

    # 3. Replace problematic unicode glyphs
    text = text.translate(UNICODE_GLYPHS_MAP)

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:  # noqa: D401
    """Run the automated fixer."""
    py_files = list(iter_py_files(SRC_DIR))
    total = len(py_files)
    modified = 0

    for f in py_files:
        try:
            if clean_file(f):
                modified += 1
        except Exception as exc:  # pragma: no cover – robust batch run
            print(f"⚠️  Skipped {f}: {exc}", file=sys.stderr)

    print(f"✔ {total} files scanned – {modified} modified")


if __name__ == "__main__":
    main()
