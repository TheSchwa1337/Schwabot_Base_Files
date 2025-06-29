# core/stochastic_core.py

import logging
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class StochasticProcess:
drift: float             # mu
volatility: float        # sigma
brownian_path: List[float]

    def evolve_price(self, S_t: float, dt: float, t: int) -> float:
    """"""
    Simulates next price using the geometric Brownian motion model:
    dS = muS dt + sigmaS dW.

    Args:
        S_t (float): Current price at time t.
            dt (float): Time step (e.g., 1 for tick).
        t (int): Current index in the Brownian path.

        Returns:
        float: The evolved price S_{t+dt}.
        """"""
                try:
            # dW is the increment of the Brownian motion
            dW = self.brownian_path[t + 1] - self.brownian_path[t]
                except IndexError:
                logger.warning("Brownian path index out of bounds. Generating random dW.")
                    # Fallback if path is shorter than expected; assumes dt corresponds to single step
                dW = np.random.normal(0, np.sqrt(dt))

                # Geometric Brownian Motion formula
                next_S_t = S_t + self.drift * S_t * dt + self.volatility * S_t * dW
                logger.debug(f"Price evolved: {S_t:.4f} -> {next_S_t:.4f} (drift={self.drift:.4f}, vol={self.volatility:.4f})")
            return float(next_S_t)

    def ito_integral(self, f: Callable[[int], float], t_end: int) -> float:
    """"""
    Approximates the Itô integral ∫₀ᵗ_end f(s) dW_s.
    Important property: non-anticipative (f(s) does not depend on future dW).

    Args:
            f (Callable[[int], float]): A function f(s) to be integrated with respect to dW_s.
                                    It should return a scalar value at each time step s.
        t_end (int): The upper limit of integration (number of steps).

    Returns:
        float: The approximated value of the Itô integral.
    """"""
    integral_val = 0.0
    # Iterate up to t_end-1 because dW_s involves (W[s+1] - W[s])
            for s in range(t_end):
                try:
            # Ensure that brownian_path has at least s+1 elements
                    if s + 1 >= len(self.brownian_path):
                logger.warning(f"Itô integral index out of bounds at step {s}. Breaking loop.")
                break

                delta_W = self.brownian_path[s + 1] - self.brownian_path[s]
                # f(s) is evaluated at the beginning of the interval [s, s+1]
                integral_val += f(s) * delta_W
                    except Exception as e:
                    logger.error(f"Error in Ito integration at step {s}: {e}")
                    break
                    logger.debug(f"Itô integral approximated up to {t_end} steps: {integral_val:.4f}")
                return float(integral_val)

                        if __name__ == "__main__":
                    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

                        # Generate a sample Brownian path for testing
                    # A simple way to generate: cumulative sum of random normals scaled by sqrt(dt)
                    np.random.seed(42) # For reproducibility
                    num_steps = 100
                    dt_val = 1.0 / num_steps # Small time step
                    random_walk_increments = np.random.normal(0, np.sqrt(dt_val), num_steps)
                    sample_brownian_path = np.cumsum(random_walk_increments)
                    sample_brownian_path = np.insert(sample_brownian_path, 0, 0.0) # Start from W_0 = 0

                    print("\n--- Testing StochasticProcess (Price Evolution) ---")
                    stochastic_proc = StochasticProcess(drift=0.1, volatility=0.2, brownian_path = list(sample_brownian_path))

                    initial_price = 100.0
                    current_price = initial_price
                    print(f"Initial Price: {current_price:.4f}")
                            for i in range(10):
                        current_price = stochastic_proc.evolve_price(current_price, dt = dt_val, t = i)
                        print(f"Price after step {i+1}: {current_price:.4f}")

                        print("\n--- Testing StochasticProcess (Itô Integral) ---")
                        # Define a simple function f(s) = s (linear growth)
    def linear_function(s: int) -> float:
    return float(s)

# Approximate Itô integral up to 50 steps
ito_result = stochastic_proc.ito_integral(linear_function, t_end=50)
print(f"Itô Integral of f(s)=s up to t=50: {ito_result:.4f}")

# Define a constant function f(s) = 10
    def constant_function(s: int) -> float:
    return 10.0

ito_result_const = stochastic_proc.ito_integral(constant_function, t_end=50)
print(f"Itô Integral of f(s)=10 up to t=50: {ito_result_const:.4f} (should approximate 10 * W_50)")
print(f"Actual W_50: {sample_brownian_path[50]:.4f}")
print(f"Expected 10 * W_50: {10 * sample_brownian_path[50]:.4f}")