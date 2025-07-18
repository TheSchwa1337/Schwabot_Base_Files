#!/usr/bin/env python3
"""Utility script to automatically fix two common syntax problems that remain in the codebase:"

1. Top-of-file imports that are mistakenly indented by 4+ spaces, producing
   "unexpected indent" errors.  The script will out-dent those lines so that
   they start at column 0.  Only the header of each file (first 120 lines, or)
   until the first `def`/`class`/non-import statement) is inspected so we
   don’t accidentally change indentation inside functions or classes.

2. `try:` blocks that were left empty, e.g.::

        try:
        except ImportError:
            ...

   which is invalid because the `try` block must contain at least one
   statement.  The script inserts a single `pass` on the line immediately
   after the `try:` if the next non-blank/non-comment line is an `except`.

Run this from the project root:

    python fix_indentation.py
"""
from __future__ import annotations

import pathlib
import re
from typing import List

ROOT_DIR = pathlib.Path(__file__).resolve().parent
PY_FILES: List[pathlib.Path] = list(ROOT_DIR.rglob("*.py"))

IMPORT_RE = re.compile(r"\s{4,}(import |from )")
DEF_CLASS_RE = re.compile(r"^\s*(def |class |@|if __name__ == |try:|with |for |while |return |print\()"))

TOTAL = 0
FIXED = 0

for path in PY_FILES:
    if path.name.startswith("fix_"):
        # Skip our utility scripts
        continue

    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    changed = False

    # 1) Outdent header imports that are incorrectly indented
    header_limit = min(len(text), 120)
    for i in range(header_limit):
        line = text[i]
        if DEF_CLASS_RE.match(line):
            # Stop once real code starts
            break
        # Outdent import lines
        if IMPORT_RE.match(line):
            text[i] = line.lstrip()
            changed = True

    # 2) Add `pass` to empty try blocks (only simple, pattern)
    i = 0
    while i < len(text) - 1:
        if text[i].strip() == "try:":
            j = i + 1
            # skip blank/comment lines
            while j < len(text) and text[j].strip() in ("", "#", '"""', "'''"): '
                j += 1
            if j < len(text) and text[j].lstrip().startswith("except"):
                text.insert(i + 1, "    pass")
                changed = True
                i = j  # skip over newly inserted block
        i += 1

    if changed:
        path.write_text("\n".join(text) + "\n", encoding="utf-8")
        FIXED += 1
    TOTAL += 1

print(f"Indent-fix processed {TOTAL} files; modified {FIXED} of them.") 