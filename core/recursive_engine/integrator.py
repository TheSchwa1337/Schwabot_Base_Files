from __future__ import annotations

import logging
from typing import Dict, Any
from .math_functions import recursive_output

logger = logging.getLogger(__name__)

class SignalIntegrator:
    """Incrementally integrates Ω(t) and optionally Ψ(t).

    The class maintains running totals so that callers can query
    accumulated values after each step.
    """

    def __init__(self) -> None:
        self._omega: float = 0.0  # ∫ F·P·C dt
        self._psi: float = 0.0    # ∫ Σ·R dt (optional, computed if R provided)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def psi(self) -> float:
        return self._psi

    def step(self, *, F: float, P: float, C: float, dt: float, sigma: float | None = None, R: float | None = None) -> Dict[str, Any]:
        """Advance the integrator by one time step.

        Parameters
        ----------
        F : float
            Fractal output.
        P : float
            Profit ratio or profit metric.
        C : float
            Coherence score derived from omega.
        dt : float
            Time delta since last step.
        sigma : float, optional
            Σ(t) value for this step. If provided and `R` is also provided, Ψ will be
            integrated as ∫ Σ·R dt.
        R : float, optional
            Recursive trigger scalar. Required to update Ψ.
        Returns
        -------
        dict with keys: omega, psi (psi may remain previous value if R or sigma not provided)
        """
        # integrate Ω
        delta_omega = recursive_output(F, P, C, dt)
        self._omega += delta_omega

        # integrate Ψ if we have sigma and R
        if sigma is not None and R is not None:
            self._psi += sigma * R * dt

        logger.debug("Integrator step: ΔΩ=%.6f, Ω=%.6f, Ψ=%.6f", delta_omega, self._omega, self._psi)

        return {"omega": self._omega, "psi": self._psi} 