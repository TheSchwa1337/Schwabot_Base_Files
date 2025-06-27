from __future__ import annotations

from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Ghost trigger \\u2013 stealth-mode activation predicate.

This micro-module exposes a *single* public helper \\u2013 :func:`ghost_trigger` \\u2013
that evaluates whether Schwabot should enter *ghost mode* based on three
continuous signals:

* ``entropy``   \\u2013 instantaneous entropy estimate (from GAN filter).
* ``momentum``  \\u2013 projection of price momentum onto the latent vector.
* ``delta_p``   \\u2013 delta between expected and realised profit.

The reference equation in the design doc is:

    \\u0393\\u208dghost\\u208e(t, P\\u2098, \\u0394\\u209b) = \\u039b\\u209b(t) \\u00b7 exp(\\u2212\\u03b7 \\u00b7 |\\u0394\\u209b \\u2212 \\u0394\\u2080|)

The current implementation simplifies this to a logistic gate so we stay
CPU-light inside tight loops.  All parameters have sane defaults but can be
overridden by callers.
"""


from core.unified_math_system import unified_math
from typing import Final

__all__: list[str] = ["ghost_trigger"]

# -----------------------------------------------------------------------------
# Tunable constants
# -----------------------------------------------------------------------------

_BASE_DELTA: Final = 0.0  # \\u0394\\u2080 in the docstring
_DAMPING: Final = 0.75  # \\u03b7 in the docstring \\u2013 larger \\u21d2 stricter gate
_MOMENTUM_SCALE: Final = 1.0  # scales \\u039b\\u209b(t) before logistic
_THRESHOLD: Final = 0.5  # logistic output above which trigger fires


def _logistic(x: float) -> float:
    """Cheap logistic activation without `unified_math.exp` overflow."""
    if x >= 0:
        z = unified_math.exp(-x)
        return 1.0 / (1.0 + z)
    z = unified_math.unified_math.exp(x)
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
        Instantaneous entropy metric (higher \\u21d2 noisier market).
    momentum
        Projected momentum value ``P\\u2098``.
    delta_p
        Profit delta ``\\u0394\\u209b`` (expected \\u2011 realised).
    eta
        Dampening coefficient **\\u03b7**.
    delta0
        Baseline delta **\\u0394\\u2080**.
    momentum_scale
        Scaling applied to momentum before gating.
    threshold
        Logistic output threshold above which mode triggers.
    """
    # Core formula (simplified logistic gate)
    delta_term = unified_math.exp(-eta * unified_math.abs(delta_p - delta0))
    raw_score = entropy * (momentum * momentum_scale) * delta_term

    # Normalise through logistic to keep range (0,1)
    score = _logistic(raw_score)
    return score > threshold

"""