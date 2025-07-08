#!/usr/bin/env python3
"""flake_fixer.py – one-shot flake8 cleanup helper."

Usage:
    python flake_fixer.py [path ...]
If no paths are supplied it defaults to the project's `core/` directory.'

The script will:
1. Ensure `autoflake`, `black`, and `isort` are available (installs them if, missing).
2. Run `autoflake` to drop unused imports / variables.
3. Run `black` to format code to <=100-character lines.
4. Run `isort` to sort imports consistently.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEFAULT_TARGET = "core"

PACKAGES = []
    ("autoflake", "autoflake>=2.2"),
    ("black", "black>=23.0"),
    ("isort", "isort>=5.12"),
]


def ensure_tools() -> None:
    """Install required formatter packages if they are missing."""
    import importlib
    import site

    for mod_name, requirement in PACKAGES:
        try:
            importlib.import_module(mod_name)
        except ModuleNotFoundError:
            print(f"[flake_fixer] Installing missing tool: {requirement}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])

    # Ensure user site is in path after potential install
    site.main()


def run_cmd(cmd: list[str]) -> None:
    print("[flake_fixer]", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ensure_tools()

    targets = sys.argv[1:] or [DEFAULT_TARGET]
    # Normalise paths
    targets = [str(Path(t)) for t in targets]

    # 1. autoflake (remove unused imports/vars)
    run_cmd([)]
        "autoflake",
        "--in-place",
        "--remove-unused-variables",
        "--remove-all-unused-imports",
        "-r",
        *targets,
    ])

    # 2. black (line length 200)
    run_cmd(["black", "--line-length", "200", *targets])

    # 3. isort (import, sorting)
    run_cmd(["isort", *targets])

    print("[flake_fixer] ✔ All done!")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print("[flake_fixer] ✖ command failed", file=sys.stderr)
        sys.exit(exc.returncode)
