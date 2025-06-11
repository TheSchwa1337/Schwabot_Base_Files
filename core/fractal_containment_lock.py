"""
fractal_containment_lock.py
===========================
Implements the Fractal Containment Lock for recursive drift/entropy stabilization.
"""

import numpy as np

class FractalContainmentLock:
    def __init__(self, lock_threshold=0.001, decay_lambda=0.01):
        self.lock_threshold = lock_threshold
        self.decay_lambda = decay_lambda

    def integrate_entropy_flux(self, entropy_vector):
        """
        Integrate entropy flux over time (∫(∂S/∂t)dt).
        """
        if len(entropy_vector) < 2:
            return 0.0
        diffs = np.diff(entropy_vector)
        integral = np.sum(np.abs(diffs)) * self.decay_lambda
        return integral

    def check_lock_condition(self, entropy_vector):
        """
        Returns True if the drift/entropy integral is below the lock threshold.
        """
        drift_integral = self.integrate_entropy_flux(entropy_vector)
        return drift_integral <= self.lock_threshold
