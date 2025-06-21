#!/usr/bin/env python3
"""Auto-fix D400 and D205 docstring style violations.

D400 – First line should end with a period.
D205 – 1 blank line required between summary line and description.

The script walks all *.py files (excluding common virtual-env / cache dirs),
parses each with ``ast`` to locate docstrings, mutates the in-memory source
if a quick, mechanical fix is possible, and writes back only changed files.

It is intentionally conservative – it will only:
1.  Append a period to the summary line when the line currently ends with an
    alphanumeric character.
2.  Insert a single blank line after the summary line when the docstring has
    multiple lines and the second line is non-blank.

No re-flow or complex formatting is attempted.
"""
from __future__ import annotations

import ast
import pathlib
from typing import List

EXCLUDE_DIRS = {".venv", "__pycache__", "node_modules", "build", "dist", ".git"}


def iter_py_files(root: pathlib.Path) -> List[pathlib.Path]:
    """Yield all *.py files below *root* that are not in *EXCLUDE_DIRS*."""
    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def fix_docstring_in_source(source: str) -> str:
    """Return *source* potentially modified to satisfy D400/D205."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source  # skip files with syntax errors

    lines = source.splitlines(keepends=True)
    modified = False

    # Helper to apply edits to a docstring node
    def _fix_node_docstring(node: ast.AST) -> None:
        """Normalize *node*'s doc-string and fix surrounding blank-line rules.

        Handles:
        • D400 – first line ends with a period.
        • D205 – exactly one blank line between summary line and body.
        • leading blank lines inside the string (variant of D205).
        • D202 – no blank line after function doc-string.
        • D204 – exactly one blank line after class doc-string.
        """
        nonlocal modified

        # Only modules / classes / (async-)functions carry doc-strings we care about
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        if not node.body or not isinstance(node.body[0], ast.Expr):
            return
        expr = node.body[0]
        if not (isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str)):
            return

        # ----- gather original lines -------------------------------------------------
        start_line = expr.lineno - 1  # zero-based
        end_line = expr.end_lineno - 1  # inclusive
        indent = " " * expr.col_offset
        raw_lines = [
            l[len(indent) :] if l.startswith(indent) else l for l in lines[start_line : end_line + 1]
        ]

        # Detect the quote mark used (""" or ''') and strip delimiters ----------------
        quote_mark = '"""' if raw_lines[0].lstrip().startswith('"""') else "'''"

        # Guard against a malformed one-liner that has no closing delimiter on the same line.
        if len(raw_lines) == 1 and quote_mark not in raw_lines[0][len(quote_mark):]:
            return  # skip invalid string without closing quotes

        try:
            first_line_body = raw_lines[0].split(quote_mark, 1)[1]
        except IndexError:
            return  # malformed docstring – skip
          
        # Build *content_lines* safely for both one-line and multi-line doc-strings
        if len(raw_lines) == 1:
            # One-liner => body is between the opening and closing quotes on same line
            inner = raw_lines[0].split(quote_mark, 1)[1].rsplit(quote_mark, 1)[0]
            content_lines: list[str] = [inner]
        else:
            # Multi-line => first / last lines already stripped of the delimiters above
            try:
                last_line_body = raw_lines[-1].rsplit(quote_mark, 1)[0]
            except IndexError:
                return
            content_lines = [first_line_body] + raw_lines[1:-1] + [last_line_body]

        changed = False
        # ----- inside-string normalisation -------------------------------------------
        # drop leading blank lines
        while content_lines and not content_lines[0].strip():
            content_lines.pop(0)
            changed = True

        if not content_lines:
            return  # (empty doc-string – leave untouched)

        # D400 – summary ends with a period
        if not content_lines[0].rstrip().endswith("."):
            content_lines[0] = content_lines[0].rstrip() + "."
            changed = True

        # D205 – exactly one blank line after summary when description exists
        if len(content_lines) > 1:
            if content_lines[1].strip():
                content_lines.insert(1, "")
                changed = True
            while len(content_lines) > 2 and not content_lines[1].strip() and not content_lines[2].strip():
                content_lines.pop(2)
                changed = True

        # ----- write doc-string back --------------------------------------------------
        if changed:
            new_body = "\n".join(content_lines)
            new_doc = f"{quote_mark}{new_body}{quote_mark}"
            new_doc_lines = (
                [indent + new_doc + "\n"]
                if "\n" not in new_doc
                else [indent + l + "\n" for l in new_doc.splitlines()]
            )
            lines[start_line : end_line + 1] = new_doc_lines
        else:
            new_doc_lines = raw_lines  # unchanged content

        # ----- adjust blank lines *after* the doc-string ------------------------------
        after_idx = start_line + len(new_doc_lines)

        def _is_blank(i: int) -> bool:
            return i < len(lines) and lines[i].strip() == ""

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # D202 – remove all blank lines immediately following function doc-string
            while _is_blank(after_idx):
                del lines[after_idx]
                modified = True
        elif isinstance(node, ast.ClassDef):
            # D204 – ensure *exactly one* blank line after class doc-string
            blank_count = 0
            while _is_blank(after_idx):
                blank_count += 1
                if blank_count > 1:
                    del lines[after_idx]
                    modified = True
                else:
                    after_idx += 1
            if blank_count == 0:
                lines.insert(after_idx, indent + "\n")
                modified = True

        # mark file modified when anything changed in this node
        if changed:
            modified = True

    # Walk module + nested defs/classes
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            _fix_node_docstring(node)

    return "".join(lines) if modified else source


def main() -> None:
    """TODO: document main."""
    root = pathlib.Path(".")
    changed_files = 0
    for py_file in iter_py_files(root):
        try:
            original = py_file.read_text(encoding="utf-8")
            fixed = fix_docstring_in_source(original)
            if fixed != original:
                py_file.write_text(fixed, encoding="utf-8")
                changed_files += 1
                print(f"✔️  Fixed docstrings in {py_file}")
        except Exception as exc:
            print(f"⚠️  Error processing {py_file}: {exc}")
    print(f"🏁 Docstring pass complete – files updated: {changed_files}")


if __name__ == "__main__":
    main() 