#!/usr/bin/env python3
"""
Time-Series Filters - Schwabot Mathematical Framework
===================================================

Implements Kalman filters, Particle filters, and adaptive signal processing
for cleaning price feeds and reducing noise before trading oracle processing.

Mathematical foundations:
- Kalman Filter: Optimal linear state estimation
- Particle Filter: Non-linear Bayesian state estimation
- EMA: Exponential Moving Average with time-awareness
- Adaptive filtering with dynamic parameter adjustment

Based on SxN-Math specifications for robust trading signal processing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional, Callable, Any
from decimal import Decimal, getcontext
from dataclasses import dataclass
import logging
from scipy.stats import multivariate_normal
import warnings

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]
StateVector = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """State representation for Kalman filter"""

    x: StateVector  # State estimate
    P: Matrix  # Covariance matrix
    timestamp: float
    likelihood: float = 0.0


class KalmanFilter:
    """
    Linear Kalman Filter for optimal state estimation

    Implements the standard predict-update cycle:
    Predict: x_k|k-1 = F * x_k-1|k-1 + B * u_k
             P_k|k-1 = F * P_k-1|k-1 * F^T + Q
    Update:  K_k = P_k|k-1 * H^T * (H * P_k|k-1 * H^T + R)^-1
             x_k|k = x_k|k-1 + K_k * (z_k - H * x_k|k-1)
             P_k|k = (I - K_k * H) * P_k|k-1
    """

    def __init__(
        self,
        F: Matrix,
        H: Matrix,
        Q: Matrix,
        R: Matrix,
        initial_state: StateVector,
        initial_covariance: Matrix,
    ):
        """
        Initialize Kalman Filter

        Args:
            F: State transition matrix
            H: Observation matrix
            Q: Process noise covariance
            R: Measurement noise covariance
            initial_state: Initial state estimate
            initial_covariance: Initial covariance estimate
        """
        self.F = F.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()

        self.state = KalmanState(
            x=initial_state.copy(), P=initial_covariance.copy(), timestamp=0.0
        )

        self.state_dim = len(initial_state)
        self.obs_dim = H.shape[0]

        # Identity matrix for updates
        self.I = np.eye(self.state_dim)

        # Numerical stability
        self.epsilon = 1e-12

        logger.info(
            f"Kalman Filter initialized: {
                self.state_dim}D state, {
                self.obs_dim}D observations")

    def predict(
        self,
        control_input: Optional[Vector] = None,
        B: Optional[Matrix] = None,
    ) -> KalmanState:
        """
        Prediction step of Kalman filter

        Args:
            control_input: Control vector u_k
            B: Control matrix

        Returns:
            Predicted state
        """
        try:
            # State prediction
            x_pred = self.F @ self.state.x
            if control_input is not None and B is not None:
                x_pred += B @ control_input

            # Covariance prediction
            P_pred = self.F @ self.state.P @ self.F.T + self.Q

            # Ensure positive definiteness
            P_pred = self._ensure_positive_definite(P_pred)

            self.state.x = x_pred
            self.state.P = P_pred

            return self.state

        except Exception as e:
            logger.error(f"Kalman prediction failed: {e}")
            raise

    def update(
        self, measurement: Vector, timestamp: float = 0.0
    ) -> KalmanState:
        """
        Update step of Kalman filter

        Args:
            measurement: Observation vector z_k
            timestamp: Measurement timestamp

        Returns:
            Updated state
        """
        try:
            # Innovation (residual)
            y = measurement - self.H @ self.state.x

            # Innovation covariance
            S = self.H @ self.state.P @ self.H.T + self.R
            S = self._ensure_positive_definite(S)

            # Kalman gain
            K = self.state.P @ self.H.T @ np.linalg.inv(S)

            # State update
            self.state.x = self.state.x + K @ y

            # Covariance update (Joseph form for numerical stability)
            I_KH = self.I - K @ self.H
            self.state.P = I_KH @ self.state.P @ I_KH.T + K @ self.R @ K.T

            # Calculate likelihood
            self.state.likelihood = self._calculate_likelihood(y, S)
            self.state.timestamp = timestamp

            return self.state

        except Exception as e:
            logger.error(f"Kalman update failed: {e}")
            raise

    def _ensure_positive_definite(self, matrix: Matrix) -> Matrix:
        """Ensure matrix is positive definite for numerical stability"""
        try:
            # Add small diagonal term if needed
            eigenvals = np.linalg.eigvals(matrix)
            if np.min(eigenvals) < self.epsilon:
                matrix += self.epsilon * np.eye(matrix.shape[0])
            return matrix
        except Exception:
            # Fallback: add regularization
            return matrix + self.epsilon * np.eye(matrix.shape[0])

    def _calculate_likelihood(
        self, innovation: Vector, innovation_cov: Matrix
    ) -> float:
        """Calculate log-likelihood of current measurement"""
        try:
            return multivariate_normal.logpdf(
                innovation, mean=np.zeros(len(innovation)), cov=innovation_cov
            )
        except Exception:
            return 0.0


@dataclass
class Particle:
    """Single particle for particle filter"""

    state: StateVector
    weight: float
    timestamp: float = 0.0


class ParticleFilter:
    """
    Particle Filter for non-linear state estimation

    Implements Sequential Monte Carlo estimation:
    1. Prediction: Sample from motion model
    2. Update: Weight particles by likelihood
    3. Resampling: Redistribute particles based on weights
    """

    def __init__(
        self,
        motion_model: Callable,
        observation_model: Callable,
        n_particles: int = 1000,
        state_dim: int = 2,
    ):
        """
        Initialize Particle Filter

        Args:
            motion_model: Function f(state, noise) -> new_state
            observation_model: Function h(state) -> observation
            n_particles: Number of particles
            state_dim: Dimension of state space
        """
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.n_particles = n_particles
        self.state_dim = state_dim

        # Initialize particles
        self.particles: List[Particle] = []
        self._initialize_particles()

        # Resampling threshold
        self.resample_threshold = n_particles / 3

        logger.info(
            f"Particle Filter initialized with {n_particles} particles"
        )

    def _initialize_particles(self) -> None:
        """Initialize particles with uniform distribution"""
        for i in range(self.n_particles):
            # Random initial state
            initial_state = np.random.randn(self.state_dim)
            particle = Particle(
                state=initial_state, weight=1.0 / self.n_particles
            )
            self.particles.append(particle)

    def predict(self, process_noise_std: float = 0.1) -> None:
        """
        Prediction step: propagate particles through motion model

        Args:
            process_noise_std: Standard deviation of process noise
        """
        try:
            for particle in self.particles:
                # Generate process noise
                noise = np.random.normal(0, process_noise_std, self.state_dim)

                # Propagate through motion model
                particle.state = self.motion_model(particle.state, noise)

        except Exception as e:
            logger.error(f"Particle prediction failed: {e}")
            raise

    def update(
        self,
        measurement: Vector,
        measurement_noise_std: float = 0.1,
        timestamp: float = 0.0,
    ) -> None:
        """
        Update step: weight particles by measurement likelihood

        Args:
            measurement: Observed measurement
            measurement_noise_std: Standard deviation of measurement noise
            timestamp: Measurement timestamp
        """
        try:
            total_weight = 0.0

            for particle in self.particles:
                # Predicted observation
                predicted_obs = self.observation_model(particle.state)

                # Calculate likelihood (assuming Gaussian measurement noise)
                residual = measurement - predicted_obs
                likelihood = np.exp(
                    -0.5 * np.sum(residual**2) / measurement_noise_std**2
                )

                # Update weight
                particle.weight *= likelihood
                total_weight += particle.weight
                particle.timestamp = timestamp

            # Normalize weights
            if total_weight > 0:
                for particle in self.particles:
                    particle.weight /= total_weight
            else:
                # Reinitialize if all weights are zero
                self._initialize_particles()

            # Check if resampling is needed
            effective_particles = 1.0 / np.sum(
                [p.weight**2 for p in self.particles]
            )
            if effective_particles < self.resample_threshold:
                self._resample()

        except Exception as e:
            logger.error(f"Particle update failed: {e}")
            raise

    def _resample(self) -> None:
        """Systematic resampling of particles"""
        try:
            # Extract weights
            weights = np.array([p.weight for p in self.particles])

            # Systematic resampling
            indices = self._systematic_resample(weights)

            # Create new particle set
            new_particles = []
            for idx in indices:
                new_particle = Particle(
                    state=self.particles[idx].state.copy(),
                    weight=1.0 / self.n_particles,
                    timestamp=self.particles[idx].timestamp,
                )
                new_particles.append(new_particle)

            self.particles = new_particles

        except Exception as e:
            logger.error(f"Particle resampling failed: {e}")
            raise

    def _systematic_resample(self, weights: Vector) -> List[int]:
        """Systematic resampling algorithm"""
        n = len(weights)
        indices = []

        # Cumulative sum
        cumsum = np.cumsum(weights)

        # Random start
        u = np.random.uniform(0, 1 / n)

        for i in range(n):
            while u > cumsum[len(indices)]:
                if len(indices) >= n - 1:
                    break
                indices.append(len(indices))
            indices.append(np.searchsorted(cumsum, u))
            u += 1 / n

        return indices[:n]

    def get_state_estimate(self) -> Tuple[StateVector, Matrix]:
        """
        Get weighted mean and covariance of particle distribution

        Returns:
            (mean_state, covariance_matrix)
        """
        try:
            # Extract states and weights
            states = np.array([p.state for p in self.particles])
            weights = np.array([p.weight for p in self.particles])

            # Weighted mean
            mean_state = np.average(states, weights=weights, axis=0)

            # Weighted covariance
            diff = states - mean_state
            cov_matrix = np.average(
                diff[:, :, np.newaxis] * diff[:, np.newaxis, :],
                weights=weights,
                axis=0,
            )

            return mean_state, cov_matrix

        except Exception as e:
            logger.error(f"State estimation failed: {e}")
            # Return default values
            return np.zeros(self.state_dim), np.eye(self.state_dim)


class TimeAwareEMA:
    """
    Time-aware Exponential Moving Average

    Adjusts smoothing factor based on actual time intervals
    rather than assuming regular sampling.
    """

    def __init__(self, alpha: float, initial_value: Optional[float] = None):
        """
        Initialize EMA filter

        Args:
            alpha: Base smoothing factor (0 < α < 1)
            initial_value: Initial EMA value
        """
        self.alpha = alpha
        self.value = initial_value
        self.last_time = None
        self.initialized = initial_value is not None

        logger.debug(f"TimeAwareEMA initialized with alpha={alpha}")

    def update(self, new_value: float, timestamp: float) -> float:
        """
        Update EMA with time-aware smoothing

        Formula: s_t = α_eff * x_t + (1 - α_eff) * s_{t-1}
        where α_eff = 1 - exp(-α * Δt)

        Args:
            new_value: New observation
            timestamp: Time of observation

        Returns:
            Updated EMA value
        """
        try:
            if not self.initialized:
                self.value = new_value
                self.last_time = timestamp
                self.initialized = True
                return self.value

            # Calculate time delta
            dt = (
                timestamp - self.last_time
                if self.last_time is not None
                else 1.0
            )
            dt = max(dt, 1e-6)  # Prevent division by zero

            # Time-adjusted smoothing factor
            alpha_eff = 1.0 - np.exp(-self.alpha * dt)
            alpha_eff = np.clip(alpha_eff, 0.0, 1.0)

            # Update EMA
            self.value = alpha_eff * new_value + (1.0 - alpha_eff) * self.value
            self.last_time = timestamp

            return self.value

        except Exception as e:
            logger.error(f"EMA update failed: {e}")
            return new_value


class AdaptiveFilter:
    """
    Adaptive filter that switches between different filtering strategies
    based on signal characteristics and market conditions.
    """

    def __init__(self):
        self.filters = {
            "ema_fast": TimeAwareEMA(alpha=0.3),
            "ema_slow": TimeAwareEMA(alpha=0.1),
            "ema_medium": TimeAwareEMA(alpha=0.2),
        }
        self.current_filter = "ema_medium"
        self.volatility_window = []
        self.volatility_threshold = 0.02

        logger.info("Adaptive filter initialized")

    def update(self, value: float, timestamp: float) -> float:
        """
        Update with adaptive filtering strategy

        Args:
            value: Input value
            timestamp: Time of observation

        Returns:
            Filtered value
        """
        try:
            # Update volatility estimate
            self._update_volatility(value)

            # Select appropriate filter based on volatility
            self._select_filter()

            # Apply selected filter
            filtered_value = self.filters[self.current_filter].update(
                value, timestamp
            )

            return filtered_value

        except Exception as e:
            logger.error(f"Adaptive filtering failed: {e}")
            return value

    def _update_volatility(self, value: float) -> None:
        """Update rolling volatility estimate"""
        self.volatility_window.append(value)
        if len(self.volatility_window) > 20:
            self.volatility_window.pop(0)

    def _select_filter(self) -> None:
        """Select filter based on current volatility"""
        if len(self.volatility_window) >= 10:
            volatility = np.std(self.volatility_window)

            if volatility > self.volatility_threshold:
                self.current_filter = (
                    "ema_slow"  # Smooth more in high volatility
                )
            else:
                self.current_filter = (
                    "ema_fast"  # React faster in low volatility
                )


# Convenience functions for external API
def warm_ema(alpha: float) -> TimeAwareEMA:
    """Create a warm (initialized) EMA filter"""
    return TimeAwareEMA(alpha)


def main() -> None:
    """Test and demonstration function"""
    # Test Kalman Filter
    print("Testing Kalman Filter...")
    F = np.array([[1, 1], [0, 1]])  # Position-velocity model
    H = np.array([[1, 0]])  # Observe position only
    Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise
    R = np.array([[1.0]])  # Measurement noise

    initial_state = np.array([0.0, 0.0])
    initial_cov = np.eye(2)

    kf = KalmanFilter(F, H, Q, R, initial_state, initial_cov)

    # Simulate measurements
    for i in range(10):
        kf.predict()
        measurement = np.array([i + np.random.normal(0, 0.5)])
        kf.update(measurement, float(i))

    print(f"Final Kalman state: {kf.state.x}")

    # Test EMA
    print("\nTesting Time-Aware EMA...")
    ema = TimeAwareEMA(alpha=0.3)

    for i in range(10):
        value = np.sin(i * 0.5) + np.random.normal(0, 0.1)
        filtered = ema.update(value, float(i))
        print(f"Time {i}: Raw={value:.3f}, Filtered={filtered:.3f}")

    print("Filters module test completed successfully")


if __name__ == "__main__":
    main()
