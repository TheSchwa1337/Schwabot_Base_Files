#!/usr/bin/env python3
"""Glyph hysteresis field – prevent flip-flopping on glyph activation.

Implements a simple Schmitt-trigger style hysteresis on glyph *strength*:

    active ↦ deactivate threshold = decay_threshold
    inactive ↦ activate threshold = activation_threshold

with activation_threshold > decay_threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__: list[str] = ["HysteresisField"]


@dataclass(slots=True)
class HysteresisField:
    activation_threshold: float = 0.7
    decay_threshold: float = 0.3
    _active: bool = False

    def update(self, strength: float) -> bool:  # noqa: D401
        """Update with *strength* ∈ [0,1] and return new active state."""
        if self._active:
            if strength < self.decay_threshold:
                self._active = False
        else:
            if strength > self.activation_threshold:
                self._active = True
        return self._active 