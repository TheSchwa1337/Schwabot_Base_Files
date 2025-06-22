#!/usr/bin/env python3
'''Fix Stub Files - Automated Stub Correction Utility.

Searches for files containing the malformed pattern
    `"""Stub main function."""."""`
inside the workspace and replaces it with a valid stub implementation:

    def main() -> None:
        """Stub main function."""
        pass

This eliminates E999 syntax errors that originate from these auto-generated
stub files.
'''

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Regex to locate the malformed stub pattern (multiline, dot-quote sequence)
STUB_PATTERN = re.compile(
    r'(def\s+main\s*\([^)]*\)\s*->[^:]*:\s*\n\s*)'  # function signature
    r'("""Stub main function\."""\."""?)',  # bad docstring line
    re.MULTILINE,
)

# Replacement template with correct docstring & pass statement
REPLACEMENT = (
    r"\1\"\"\"Stub main function.\"\"\"\n    pass"
)


def fix_stub_file(path: Path) -> bool:
    """Return True if the file was modified."""
    try:
        original = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Skip %s (%s)", path, exc)
        return False

    if not STUB_PATTERN.search(original):
        return False

    patched = STUB_PATTERN.sub(REPLACEMENT, original)

    # Ensure 'if __name__ == "__main__"' block is present
    if "if __name__ == \"__main__\"" not in patched:
        patched += "\n\nif __name__ == \"__main__\":\n    main()\n"

    path.write_text(patched, encoding="utf-8")
    logger.info("Fixed stub in %s", path)
    return True


def scan_and_fix(root: Path, dry_run: bool = False) -> List[Path]:
    modified: List[Path] = []
    for py_file in root.rglob("*.py"):
        if py_file.is_file():
            if dry_run:
                if STUB_PATTERN.search(py_file.read_text(encoding="utf-8")):
                    modified.append(py_file)
            else:
                if fix_stub_file(py_file):
                    modified.append(py_file)
    return modified


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix malformed stub files.")
    parser.add_argument("--path", default=".", help="Root directory to scan (default: current directory)")
    parser.add_argument("--dry-run", action="store_true", help="Only list files that would be modified")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    modified = scan_and_fix(root, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("%d stub files would be modified.", len(modified))
        for p in modified:
            print(p)
    else:
        logger.info("%d stub files patched.", len(modified))


if __name__ == "__main__":
    main() 