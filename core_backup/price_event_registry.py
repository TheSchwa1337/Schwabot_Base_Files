"""core/price_event_registry.py"""

Thread-safe rolling registry for :mod:`core.price_event.PriceEvent` objects.
Keeps the last *N* events in memory for quick access and offers a JSON dump
helper for audit/replay purposes.
""""""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Deque, List

from core.price_event import PriceEvent

__all__ = ["record", "last", "dump_to_file"]

_MAX_EVENTS: int = 100_000  # configurable if needed
_events: Deque[PriceEvent] = deque(maxlen=_MAX_EVENTS)
_lock = threading.Lock()


def record(event: PriceEvent) -> None:
    """Add a *PriceEvent* to the rolling registry (thread-safe)."""
    with _lock:
        _events.append(event)


def last(n: int = 100) -> List[PriceEvent]:
    """Return *n* most recent events (<= total stored)."""
    with _lock:
        if n <= 0:
            return []
        return list(_events)[-n:]


def dump_to_file(path: str | Path) -> None:
    """Write the full in-memory registry to *path* as pretty-printed JSON."""
    with _lock:
        serialised = [e.to_dict() for e in _events]
    path = Path(path)
    path.write_text(json.dumps(serialised, ensure_ascii=False, indent=2), encoding="utf-8")
