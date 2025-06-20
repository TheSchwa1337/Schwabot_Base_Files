#!/usr/bin/env python3
"""Phantom memory – decay-corrected ghost state recall.

Implements the memory formula:
    M_r = Σ ζ_i ∘ t_i where Ξ ∈ Σ(ghost_log)

This module maintains a rolling window of ghost events with exponential decay
weighting to preserve the most relevant historical triggers while allowing
older signals to fade naturally.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

__all__: list[str] = ["PhantomMemory", "GhostEvent", "compute_memory_recall"]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GhostEvent:
    """Single ghost trigger event with timestamp and intensity."""

    timestamp: float
    zeta: float
    xi_ghost: float
    event_type: str = "trigger"


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------


class PhantomMemory:
    """Rolling memory buffer for ghost events with decay weighting."""

    def __init__(self, *, max_events: int = 1000, decay_lambda: float = 0.01):
        """Initialize memory with capacity and decay rate."""
        self._events: list[GhostEvent] = []
        self._max_events = max_events
        self._decay_lambda = decay_lambda

    def add_event(self, event: GhostEvent) -> None:
        """Add new ghost event to memory buffer."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            # Remove oldest event
            self._events.pop(0)

    def compute_recall(self, current_time: float | None = None) -> float:
        """Return M_r memory recall value with decay weighting."""
        if current_time is None:
            current_time = time.time()

        if not self._events:
            return 0.0

        recall_sum = 0.0
        for event in self._events:
            dt = current_time - event.timestamp
            decay_weight = math.exp(-self._decay_lambda * dt)
            recall_sum += event.zeta * event.xi_ghost * decay_weight

        return recall_sum

    def get_recent_events(self, window_seconds: float) -> list[GhostEvent]:
        """Return events from the last window_seconds."""
        if not self._events:
            return []

        current_time = time.time()
        cutoff_time = current_time - window_seconds
        return [e for e in self._events if e.timestamp >= cutoff_time]

    def clear_old_events(self, max_age_seconds: float) -> int:
        """Remove events older than max_age_seconds. Return count removed."""
        if not self._events:
            return 0

        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        initial_count = len(self._events)
        self._events = [e for e in self._events if e.timestamp >= cutoff_time]
        return initial_count - len(self._events)

    @property
    def event_count(self) -> int:
        """Return current number of stored events."""
        return len(self._events)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def compute_memory_recall(
    events: Sequence[GhostEvent],
    current_time: float | None = None,
    decay_lambda: float = 0.01,
) -> float:  # noqa: D401
    """Compute M_r recall from event sequence (functional interface).

    Parameters
    ----------
    events
        Sequence of ghost events to process.
    current_time
        Reference time for decay calculation. Uses time.time() if None.
    decay_lambda
        Exponential decay rate (larger = faster decay).
    """
    if current_time is None:
        current_time = time.time()

    if not events:
        return 0.0

    recall_sum = 0.0
    for event in events:
        dt = current_time - event.timestamp
        decay_weight = math.exp(-decay_lambda * dt)
        recall_sum += event.zeta * event.xi_ghost * decay_weight

    return recall_sum 