from __future__ import annotations

"""Schwabot universal launcher.

This module is the *only* file that every packager (PyInstaller, py2app, Linux
`.desktop` files) needs to point at.  It detects which runtime mode the user
wants and forwards control to the appropriate subsystem so that the rest of
Schwabot's codebase can stay platform-agnostic.

Usage
-----
$ schwabot                 # default GUI
$ schwabot gui             # explicit GUI
$ schwabot cli             # command-line interface
$ schwabot tray            # background tray icon
"""

from importlib import import_module
import sys
from types import ModuleType
from typing import Callable

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_optional(
    module_path: str, default: Callable[[], None]
) -> Callable[[], None]:
    """Try to import *module_path* and return its callable.  If that fails, return
    *default* so that the program still runs gracefully on systems that do not
    ship every optional dependency yet (e.g. GUI requirements on headless
    servers).
    """
    try:
        module_name, func_name = module_path.split(":")
        module: ModuleType = import_module(module_name)
        return getattr(module, func_name)  # type: ignore[attr-defined]
    except Exception as e:  # pylint: disable=broad-except
        # Delay import-errors until the user actually requests that mode.
        def _fallback(exc=e) -> None:  # noqa: D401  (simple-function-name)
            print(
                f"⚠️  Optional dependency for '{module_path}' is missing →\n"
                f"   {exc}\n"
                "   Falling back to CLI interface instead.",
                file=sys.stderr,
            )
            _import_optional("schwabot.cli:main", _noop)()

        return _fallback


def _noop() -> None:
    """Empty fallback when absolutely nothing else is available."""


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def run() -> None:  # pylint: disable=too-many-branches
    """Entry-point function referenced from *setup.py* entry_points.

    Accepts an optional positional argument specifying the launch mode.  If no
    argument is supplied, GUI mode is attempted first, falling back to CLI if
    necessary.
    """
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "gui"

    if mode in {"gui", "window", "visual"}:
        _import_optional("schwabot.gui:launch", _noop)()

    elif mode in {"cli", "terminal", "cmd"}:
        _import_optional("schwabot.cli:main", _noop)()

    elif mode == "tray":
        _import_optional("schwabot.tray:run_tray", _noop)()

    else:
        print(f"Unknown mode '{mode}'. Available: gui, cli, tray.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()
