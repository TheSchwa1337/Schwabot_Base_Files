#!/usr/bin/env python3
"""Ghost memory – lightweight hash store for profitable trade snapshots.

Purpose
-------
Keep a bounded history of *profitable* hash signatures so that the
:pyclass:`ghost_router.GhostRouter` can detect repeatable market states via
hash-drift comparison.

Design goals
~~~~~~~~~~~~
1. **O(1) look-ups** by keeping the *most recent* profitable hash at `self.last`.
2. **Memory-bounded** – default ring buffer length of 1 000 entries.
3. **Pure-Python + std-lib only** – no heavy DB, avoids extra deps.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Final, List

__all__: list[str] = ["GhostMemory", "store_ghost_hash", "last_profitable_hash"]

_DEFAULT_CAPACITY: Final = 1000


class GhostMemory:
    """Ring-buffer store of profitable trade hashes."""

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:  # noqa: D401
        self._buf: Deque[str] = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, hash_hex: str) -> None:
        """Append *hash_hex* to the buffer.

        Parameters
        ----------
        hash_hex
            64-character SHA-256 hex digest.
        """
        if len(hash_hex) != 64:
            raise ValueError("hash_hex must be 64-char SHA-256 digest")
        self._buf.append(hash_hex)

    def last(self) -> str | None:  # noqa: D401
        """Return the most recent stored hash or ``None`` if empty."""
        return self._buf[-1] if self._buf else None

    def all(self) -> List[str]:
        """Return list copy of all stored hashes (newest last)."""
        return list(self._buf)


# -----------------------------------------------------------------------------
# Module-level singleton & functional helpers – mirrors legacy stubs
# -----------------------------------------------------------------------------

_memory = GhostMemory()


def store_ghost_hash(hash_hex: str) -> None:  # noqa: D401
    """Add *hash_hex* to global ghost memory ring-buffer."""
    _memory.add(hash_hex)


def last_profitable_hash() -> str | None:  # noqa: D401
    """Return last profitable hash stored globally, or ``None``."""
    return _memory.last() 