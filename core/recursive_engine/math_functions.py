import numpy as np
from typing import Union

__all__ = [
    "coherence_trigger",
    "fractal_signal",
    "recursive_output",
    "recursive_sum",
]

Number = Union[int, float]

def coherence_trigger(omega: Number) -> float:
    """Compute coherence score from accumulated Ω.

    C(t) = 1 - exp(-Ω(t))
    The output is bounded in (0, 1) given Ω ≥ 0.
    """
    # ensure numeric stability
    omega_val = max(0.0, float(omega))
    return 1.0 - np.exp(-omega_val)

def fractal_signal(dF: Number, Lambda: Number, phi: Number) -> float:
    """Compute ξ (tick evolution) from derivative of fractal core and phase.

    ξ(t) = dF/dt + sin(Λ + φ)
    """
    return float(dF) + np.sin(float(Lambda) + float(phi))

def recursive_output(F: Number, P: Number, C: Number, dt: Number) -> float:
    """Compute incremental Ω over a small time delta.

    Ω_inc = F * P * C * dt (Riemann approximation of integral)
    """
    return float(F) * float(P) * float(C) * float(dt)

def recursive_sum(omega: Number, xi: Number) -> float:
    """Compute Σ(t) = Ω(t) + ξ(t)."""
    return float(omega) + float(xi) 