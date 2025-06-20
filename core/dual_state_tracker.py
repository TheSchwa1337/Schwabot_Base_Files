#!/usr/bin/env python3
"""Dual-state tracker utility.

This module introduces a *very* lightweight dual-number structure that lets the
Schwabot maths stack propagate first-order derivatives (``dx``) alongside scalar
values.  It will be used by recursive filters that need cheap automatic-
differentiation without depending on a heavy AD framework.

Mathematical background
-----------------------
A dual number is a two-component object

    x̂ = x + ε·x′    with ε² = 0

Under first-order Taylor expansion rules we have

    f(x̂) = f(x) + ε·f′(x)

so the *ε* coefficient carries the derivative automatically through algebraic
operations.  For now we only expose addition, subtraction, multiplication and
division which are enough for most Schwabot analytic filters.

This file is intentionally small so that it passes Flake8 *and* gives a clean
API surface.  Advanced Jacobian / nested-dual logic can be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = ["DualNumber", "dual_state_tracker"]


@dataclass(slots=True)
class DualNumber:
    """A first-order dual number *x + ε·dx*."""

    x: float  # primal value
    dx: float  # derivative w.r.t. some scalar variable

    # ---------------------------------------------------------------------
    # Basic arithmetic
    # ---------------------------------------------------------------------
    def __add__(self, other: "DualNumber | float") -> "DualNumber":  # noqa: D401
        if isinstance(other, DualNumber):
            return DualNumber(self.x + other.x, self.dx + other.dx)
        return DualNumber(self.x + float(other), self.dx)

    __radd__ = __add__

    def __sub__(self, other: "DualNumber | float") -> "DualNumber":
        if isinstance(other, DualNumber):
            return DualNumber(self.x - other.x, self.dx - other.dx)
        return DualNumber(self.x - float(other), self.dx)

    def __rsub__(self, other: float) -> "DualNumber":
        return DualNumber(float(other) - self.x, -self.dx)

    def __mul__(self, other: "DualNumber | float") -> "DualNumber":
        if isinstance(other, DualNumber):
            return DualNumber(self.x * other.x, self.x * other.dx + self.dx * other.x)
        other_f = float(other)
        return DualNumber(self.x * other_f, self.dx * other_f)

    __rmul__ = __mul__

    def __truediv__(self, other: "DualNumber | float") -> "DualNumber":
        if isinstance(other, DualNumber):
            denom = other.x ** 2
            return DualNumber(
                self.x / other.x,
                (self.dx * other.x - self.x * other.dx) / denom,
            )
        other_f = float(other)
        return DualNumber(self.x / other_f, self.dx / other_f)

    def __rtruediv__(self, other: float) -> "DualNumber":
        denom = self.x ** 2
        return DualNumber(float(other) / self.x, (-float(other) * self.dx) / denom)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def as_tuple(self) -> Tuple[float, float]:
        """Return ``(x, dx)`` tuple for downstream consumers."""
        return self.x, self.dx

    # Human-friendly representation --------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        return f"DualNumber(x={self.x:.6g}, dx={self.dx:.6g})"


# -----------------------------------------------------------------------------
# Public helper – minimal wrapper requested by Flake8 stub reports
# -----------------------------------------------------------------------------

def dual_state_tracker(value: float, derivative: float) -> DualNumber:
    """Wrap *value* and *derivative* into a :class:`DualNumber` instance.

    Parameters
    ----------
    value
        Primal scalar value *x*.
    derivative
        Associated first-order derivative *dx*.

    Returns
    -------
    DualNumber
        ``x + ε·dx`` dual-number form.
    """
    return DualNumber(x=float(value), dx=float(derivative)) 