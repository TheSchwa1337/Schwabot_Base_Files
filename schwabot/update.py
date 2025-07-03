from __future__ import annotations

"""Self-update routine used by the `schwabot --update` flag.

This is intentionally *very* small so that it keeps working even when the rest
of the package evolves.  It simply performs a `git pull` in the project root
and reinstalls dependencies if *requirements.txt* changed.
"""

import hashlib
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REQ_FILE = PROJECT_ROOT / "requirements.txt"


def _file_hash(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _pip_install(requirements: pathlib.Path) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def do_update() -> None:
    """Pull latest commits from the Git remote and update dependencies."""
    old_hash: str = _file_hash(REQ_FILE) if REQ_FILE.exists() else ""

    print("🔄  Fetching latest Schwabot code…")
    subprocess.check_call(["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"])

    if not REQ_FILE.exists():
        print("📦  No requirements.txt found; skipping dependency update.")
        return

    new_hash = _file_hash(REQ_FILE)
    if new_hash != old_hash:
        print("📦  requirements.txt changed – reinstalling dependencies…")
        _pip_install(REQ_FILE)
    else:
        print("✅  Dependencies already up-to-date.")

    print("✨  Schwabot is now on the latest version.")
