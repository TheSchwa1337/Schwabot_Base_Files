from __future__ import annotations
import logging
from typing import Dict, Any
from .math_functions import coherence_trigger, fractal_signal, recursive_sum
from .integrator import SignalIntegrator

logger = logging.getLogger(__name__)

class RecursiveEngine:
    """High-level orchestrator that ties together core recursive math.

    The engine receives raw numerical inputs each tick and produces a
    dictionary of derived recursive metrics. It is intentionally light-weight
    so it can run in high-frequency loops without becoming a bottleneck.
    """

    def __init__(self) -> None:
        self._integrator = SignalIntegrator()
        self._prev_F: float | None = None  # to compute derivative approx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_tick(
        self,
        *,
        F: float,
        P: float,
        Lambda: float,
        phi: float,
        R: float = 1.0,
        dt: float,
    ) -> Dict[str, Any]:
        """Process a single tick worth of inputs.

        Parameters
        ----------
        F : float
            Current fractal output.
        P : float
            Profit ratio / metric.
        Lambda : float
            Loop layer / phase signal.
        phi : float
            Static or dynamic phase shift.
        R : float, default 1.0
            Recursive trigger multiplier (use 1 if none provided).
        dt : float
            Time delta from previous tick.
        Returns
        -------
        dict containing omega, xi, sigma, psi, coherence (C)
        """
        # ------------------------------------------------------------------
        # 1) Coherence is based on last Ω (using integrator's current state)
        #    but we need the *previous* Ω for this tick's integration step.
        #    We'll approximate it with current omega before updating.
        # ------------------------------------------------------------------
        prev_omega = self._integrator.omega
        C = coherence_trigger(prev_omega)

        # ------------------------------------------------------------------
        # 2) Approximate derivative dF/dt
        # ------------------------------------------------------------------
        if self._prev_F is None:
            dF = 0.0
        else:
            # simple finite difference derivative
            dF = (F - self._prev_F) / dt if dt > 0 else 0.0
        self._prev_F = F

        # ------------------------------------------------------------------
        # 3) Compute ξ(t)
        # ------------------------------------------------------------------
        xi = fractal_signal(dF, Lambda, phi)

        # ------------------------------------------------------------------
        # 4) Integrate Ω(t) and optionally Ψ(t)
        # ------------------------------------------------------------------
        ints = self._integrator.step(F=F, P=P, C=C, dt=dt, sigma=None, R=None)
        omega = ints["omega"]

        # We now have updated Ω, so compute Σ and update Ψ
        sigma = recursive_sum(omega, xi)
        psi_dict = self._integrator.step(F=0.0, P=0.0, C=0.0, dt=dt, sigma=sigma, R=R)
        psi = psi_dict["psi"]

        metrics = {
            "omega": omega,
            "xi": xi,
            "sigma": sigma,
            "psi": psi,
            "coherence": C,
        }
        logger.debug("RecursiveEngine metrics: %s", metrics)
        return metrics 