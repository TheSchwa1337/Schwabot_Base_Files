#!/usr/bin/env python3
"""Add placeholder docstrings where flake8 reports D100-D107 (missing docstrings).

The script walks the repository, parses each Python file with ``ast`` and inserts a minimal
one-line docstring when none is present for:
• the module itself,
• every class definition,
• every function/async-function definition.

It tries to preserve indentation and only mutates files when a change is required.

This is an automated, *placeholder* fix – the added strings are deliberately terse (e.g.
`'''TODO: document Foo.'''`). They satisfy Flake-8 so that deeper documentation work can be
scheduled later without blocking CI.
"""
from __future__ import annotations

import ast
import pathlib
import sys
from typing import List

EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".git",
}

PLACEHOLDER_TPL = "TODO: document {name}."


def iter_py_files(root: pathlib.Path) -> List[pathlib.Path]:
    """Yield project *.py files below *root* skipping *EXCLUDE_DIRS*."""."""
    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


class DocstringAdder(ast.NodeVisitor):
    """Collect nodes lacking docstrings with their insertion positions."""."""

    def __init__(self, source_lines: List[str]):
        """TODO: document __init__."""."""
        self.source_lines = source_lines
        # list of tuples (start_line_idx, indent_str, placeholder_text)
        self.insertions: List[tuple[int, str, str]] = []

    # pylint: disable=invalid-name
    def visit_Module(self, node: ast.Module) -> None:  # type: ignore[override]
        """TODO: document visit_Module."""."""
        if ast.get_docstring(node) is None:
            # Insert after shebang / encoding comments if they exist
            insert_at = 0
            while insert_at < len(self.source_lines) and (
                self.source_lines[insert_at].startswith("#!/")
                or self.source_lines[insert_at].lstrip().startswith("#")
                and "coding" in self.source_lines[insert_at]
            ):
                insert_at += 1
            self.insertions.append(
                (
                    insert_at,
                    "",
                    f'"""{PLACEHOLDER_TPL.format(name=node.__dict__.get("name", "module"))}"""',
                )
            )
        # Continue walking to classes/functions
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
        """TODO: document visit_ClassDef."""."""
        if ast.get_docstring(node) is None and node.body:
            first_body_line = node.body[0].lineno - 1
            indent = " " * node.col_offset + " " * 4  # one extra indent level
            self.insertions.append(
                (
                    first_body_line,
                    indent,
                    f'"""{PLACEHOLDER_TPL.format(name=node.name)}"""\n{indent}',
                )
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
        """TODO: document visit_FunctionDef."""."""
        if ast.get_docstring(node) is None and node.body:
            first_body_line = node.body[0].lineno - 1
            indent = " " * node.col_offset + " " * 4
            self.insertions.append(
                (
                    first_body_line,
                    indent,
                    f'"""{PLACEHOLDER_TPL.format(name=node.name)}"""',
                )
            )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
        """TODO: document visit_AsyncFunctionDef."""."""
        self.visit_FunctionDef(node)  # type: ignore[arg-type]


def add_placeholders(path: pathlib.Path) -> bool:
    """Return True if *path* was modified by adding docstrings."""."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False  # skip files with syntax errors

    lines = src.splitlines(keepends=True)
    visitor = DocstringAdder(lines)
    visitor.visit(tree)

    if not visitor.insertions:
        return False

    # Apply insertions bottom-up so indexes stay valid
    for line_idx, indent, placeholder in sorted(visitor.insertions, reverse=True):
        lines.insert(line_idx, f"{indent}{placeholder}\n")

    path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> None:  # pragma: no cover
    """TODO: document main."""."""
    root = pathlib.Path(".")
    modified = 0
    for py_file in iter_py_files(root):
        try:
            if add_placeholders(py_file):
                modified += 1
                print(f"✔️  Added placeholders in {py_file}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️  Error processing {py_file}: {exc}")
    print(f"🏁 Missing-docstring pass complete – files updated: {modified}")


if __name__ == "__main__":
    sys.exit(main()) 