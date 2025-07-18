from typing import Callable

import numpy as np

from fractals.fractal_base import FractalBase


class EcoFractal(FractalBase):
    def __init__(self, sensitivity_kernel: Callable[[float], float]):
        super().__init__()
        self.sensitivity_kernel = sensitivity_kernel  # ψ_n(θ) as a callable function
        self.eco_factor = 1.0  # Initial eco factor

    def eco_response(
        self,
        phi_fn: Callable[[float], float],
        theta_range: np.ndarray = np.linspace(-5, 5, 100),
    ) -> float:
        """
        Compute eco integral response.
        E_n(x) = Integral(Phi(x,theta) * psi_n(theta) d(theta))
        """
        # Simulating the integral as a sum over a range for discrete application
        response = sum(phi_fn(theta) * self.sensitivity_kernel(theta) for theta in theta_range)
        return response

    def update_memory(
        self,
        omega_n: float,
        delta_psi_n: float,
        phi_fn: Callable[[float], float],
        theta_range: np.ndarray = np.linspace(-5, 5, 100),
    ) -> float:
        """
        Update fractal memory shell with environmental input.
        M_{n+1} = gamma * M_n + beta * Omega_n * DeltaPsi_n * (1 + xi * E_n)
        This EcoFractal directly computes and applies E_n.
        """
        self.eco_factor = self.eco_response(phi_fn, theta_range)

        # Eco factor directly modulates the memory shell.
        # Here, it's applied as a direct multiplier, acting as xi * E_n from the general formula.
        # The core memory update from ForeverFractal will then incorporate this eco_factor.
        # This module primarily computes the eco_factor, which then feeds into the ForeverFractal's update.
        # For direct self-update here, we could update an internal `memory_shell` attribute.

        # For simplicity within EcoFractal, we will update its own internal state, and its output (eco_factor)
        # will be used by the FractalController to update the ForeverFractal.

        # Ensure eco_factor remains reasonable, e.g., for modulation it might be between -1 and 1 or 0 and 2.
        self.eco_factor = np.clip(self.eco_factor, -1.0, 1.0)  # Assuming it's a modulator

        # Update own entropy and coherence based on environmental response (simplified)
        self.entropy_anchor = self.compute_entropy([self.eco_factor])
        self.coherence = self.compute_coherence([self.eco_factor])

        return self.eco_factor
