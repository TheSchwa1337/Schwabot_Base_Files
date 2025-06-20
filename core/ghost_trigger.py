#!/usr/bin/env python3
"""Ghost trigger – stealth-mode activation predicate.

This micro-module exposes a *single* public helper – :func:`ghost_trigger` –
that evaluates whether Schwabot should enter *ghost mode* based on three
continuous signals:

* ``entropy``   – instantaneous entropy estimate (from GAN filter).
* ``momentum``  – projection of price momentum onto the latent vector.
* ``delta_p``   – delta between expected and realised profit.

The reference equation in the design doc is:

    Γ₍ghost₎(t, Pₘ, Δₛ) = Λₛ(t) · exp(−η · |Δₛ − Δ₀|)

The current implementation simplifies this to a logistic gate so we stay
CPU-light inside tight loops.  All parameters have sane defaults but can be
overridden by callers.
"""

from __future__ import annotations

import math
from typing import Final

__all__: list[str] = ["ghost_trigger"]

# -----------------------------------------------------------------------------
# Tunable constants
# -----------------------------------------------------------------------------

_BASE_DELTA: Final = 0.0  # Δ₀ in the docstring
_DAMPING: Final = 0.75  # η in the docstring – larger ⇒ stricter gate
_MOMENTUM_SCALE: Final = 1.0  # scales Λₛ(t) before logistic
_THRESHOLD: Final = 0.5  # logistic output above which trigger fires


def _logistic(x: float) -> float:
    """Cheap logistic activation without `math.exp` overflow."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def ghost_trigger(
    entropy: float,
    momentum: float,
    delta_p: float,
    *,
    eta: float = _DAMPING,
    delta0: float = _BASE_DELTA,
    momentum_scale: float = _MOMENTUM_SCALE,
    threshold: float = _THRESHOLD,
) -> bool:
    """Return ``True`` if ghost mode should activate.

    Parameters
    ----------
    entropy
        Instantaneous entropy metric (higher ⇒ noisier market).
    momentum
        Projected momentum value ``Pₘ``.
    delta_p
        Profit delta ``Δₛ`` (expected ‑ realised).
    eta
        Dampening coefficient **η**.
    delta0
        Baseline delta **Δ₀**.
    momentum_scale
        Scaling applied to momentum before gating.
    threshold
        Logistic output threshold above which mode triggers.
    """
    # Core formula (simplified logistic gate)
    delta_term = math.exp(-eta * abs(delta_p - delta0))
    raw_score = entropy * (momentum * momentum_scale) * delta_term

    # Normalise through logistic to keep range (0,1)
    score = _logistic(raw_score)
    return score > threshold 