#!/usr/bin/env python3
"""Lantern vector memory – exponential decay buffer M_L."""

from __future__ import annotations

from collections import deque
from typing import Deque, List

import numpy as np

__all__: list[str] = ["LanternMemory"]


class LanternMemory:
    """Store γᵢ values with exponential decay."""

    def __init__(self, decay: float = 0.9, maxlen: int = 64) -> None:  # noqa: D401
        self.decay = decay
        self._buf: Deque[np.ndarray] = deque(maxlen=maxlen)

    def add(self, vec: np.ndarray) -> None:  # noqa: D401
        self._buf.append(vec.astype(float))

    def value(self) -> np.ndarray:
        """Return decayed sum Σ γᵢ e^(−ζ·τ). Here ζ=1."""
        result = None
        factor = 1.0
        for vec in reversed(self._buf):
            if result is None:
                result = factor * vec
            else:
                result += factor * vec
            factor *= self.decay
        return result if result is not None else np.array([]) 