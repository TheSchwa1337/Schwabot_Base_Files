"""
Warp Sync Core Module
---------------------
Implements the Warp Gradient Drift Envelope and Warp Decay Function,
essential for temporal acceleration and dynamic lattice management within Schwabot.
This module helps throttle entry timing or delay trades until ideal vector return.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional

class WarpSyncCore:
    """
    Manages the warp momentum of the hash system and its decay, influencing
    temporal acceleration and trade timing.
    """

    def __init__(self, initial_lambda: float = 0.05, initial_sigma_sq: float = 0.01):
        """
        Initializes the WarpSyncCore.

        Args:
            initial_lambda: Initial decay rate (λ) for the warp decay function.
            initial_sigma_sq: Initial variance (σ²) for the warp decay function.
        """
        self.lambda_decay = initial_lambda
        self.sigma_sq = initial_sigma_sq
        self.lattice_history: List[Dict[str, Any]] = [] # Stores {t, L(t), Omega(t)}
        self.metrics: Dict[str, Any] = {
            "total_warp_calculations": 0,
            "last_warp_calculation_time": None,
            "current_warp_momentum": 0.0
        }

    def _calculate_omega(self, delta_psi: float, current_time: float = None) -> float:
        """
        Calculates the warp drift entropy function Ω(t).

        Ω(t) = e^(-λt) · (σ² / ΔΨ)

        Args:
            delta_psi: Phase delta between time-step strategies (ΔΨ).
            current_time: The current time, used for the decay factor. If None, time.time() is used.

        Returns:
            The calculated warp drift entropy (Ω(t)).
        """
        if delta_psi == 0:
            # Handle division by zero for ΔΨ, potentially indicating a stable phase
            # We can return a default or very high value, or raise an error based on system needs.
            # For now, let's return a very high value to signify extreme decay if ΔΨ is zero.
            return np.inf 

        t = current_time if current_time is not None else time.time()
        decay_factor = np.exp(-self.lambda_decay * t)
        
        # Ensure delta_psi is not too close to zero to prevent overflow
        effective_delta_psi = max(delta_psi, 1e-9) 
        
        return decay_factor * (self.sigma_sq / effective_delta_psi)

    def calculate_warp_momentum(self, 
                                lattice_points: List[Dict[str, Any]], 
                                delta_psi_values: List[float], 
                                span_tau: Optional[float] = None) -> float:
        """
        Calculates the total warp momentum W(τ) over a given time span τ.

        W(τ) = ∫₀^τ L(t)·Ω(t) dt
        Approximated as a sum for discrete time steps: Σ [L(t) * Ω(t) * Δt]

        Args:
            lattice_points: A list of dictionaries, each containing 'L(t)' (lattice position)
                            and 't' (timestamp).
            delta_psi_values: A list of ΔΨ values corresponding to each lattice point.
            span_tau: The total time span over which to calculate the momentum. If None,
                      it calculates over the provided lattice_points.

        Returns:
            The total warp momentum W(τ).
        """
        start_time_calc = time.time()
        self.metrics["total_warp_calculations"] += 1

        if not lattice_points or not delta_psi_values or len(lattice_points) != len(delta_psi_values):
            # No data or mismatch in data lengths
            self.metrics["current_warp_momentum"] = 0.0
            return 0.0

        total_warp_momentum = 0.0
        
        # Sort lattice points by time if not already sorted
        sorted_lattice_points = sorted(lattice_points, key=lambda x: x['t'])

        for i in range(len(sorted_lattice_points)):
            current_l_t = sorted_lattice_points[i]['L(t)']
            current_t = sorted_lattice_points[i]['t']
            current_delta_psi = delta_psi_values[i]
            
            omega_t = self._calculate_omega(current_delta_psi, current_t)
            
            # Approximate Δt. For the first point, use a small default or 
            # assume it's the start of the interval. For subsequent points, 
            # use the difference from the previous tick.
            dt = 0.0 # Default for the first point
            if i > 0:
                prev_t = sorted_lattice_points[i-1]['t']
                dt = current_t - prev_t
            elif len(sorted_lattice_points) == 1:
                # If only one point, assume a unit time step or 0
                dt = 1.0 # Or based on typical tick resolution

            # W(τ) = Σ [L(t) * Ω(t) * Δt]
            total_warp_momentum += current_l_t * omega_t * dt
        
        self.metrics["current_warp_momentum"] = total_warp_momentum
        end_time_calc = time.time()
        self.metrics["last_warp_calculation_time"] = end_time_calc
        
        return total_warp_momentum

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns the operational metrics of the Warp Sync Core.
        """
        return self.metrics

    def update_parameters(self, new_lambda: Optional[float] = None, new_sigma_sq: Optional[float] = None):
        """
        Updates the parameters of the warp decay function.
        """
        if new_lambda is not None:
            self.lambda_decay = new_lambda
        if new_sigma_sq is not None:
            self.sigma_sq = new_sigma_sq
        print("Warp Sync Core parameters updated.")

    def reset(self):
        """
        Resets the core's history and metrics.
        """
        self.lattice_history = []
        self.metrics = {
            "total_warp_calculations": 0,
            "last_warp_calculation_time": None,
            "current_warp_momentum": 0.0
        }

if __name__ == "__main__":
    print("--- Warp Sync Core Demo ---")

    # Initialize the WarpSyncCore
    warp_core = WarpSyncCore(initial_lambda=0.01, initial_sigma_sq=0.005)

    # Simulate lattice points and delta_psi values over time
    # L(t) = SHA256(P_t, V_t, Δt) - For simplicity, L(t) will be represented as a float
    # ΔΨ(t) = phase delta between time-step strategies - represented as a float
    
    # Simulate a time series
    start_sim_time = time.time()
    
    # Lattice data: L(t) and its corresponding time (t)
    # delta_psi_values: ΔΨ(t)
    simulated_data = [
        {"L(t)": 0.5, "t": start_sim_time + 1, "delta_psi": 0.01},
        {"L(t)": 0.6, "t": start_sim_time + 2, "delta_psi": 0.02},
        {"L(t)": 0.7, "t": start_sim_time + 3, "delta_psi": 0.015},
        {"L(t)": 0.55, "t": start_sim_time + 4, "delta_psi": 0.03},
        {"L(t)": 0.62, "t": start_sim_time + 5, "delta_psi": 0.01},
    ]

    lattice_points = [{"L(t)": d["L(t)"], "t": d["t"]} for d in simulated_data]
    delta_psi_values = [d["delta_psi"] for d in simulated_data]

    print("\n--- Calculating Warp Momentum ---")
    warp_momentum = warp_core.calculate_warp_momentum(lattice_points, delta_psi_values)
    print(f"Calculated Warp Momentum: {warp_momentum:.6f}")

    print("\n--- Current Metrics ---")
    metrics = warp_core.get_metrics()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Simulate another calculation with updated parameters
    print("\n--- Updating Parameters and Recalculating ---")
    warp_core.update_parameters(new_lambda=0.02, new_sigma_sq=0.008)
    
    simulated_data_2 = [
        {"L(t)": 0.58, "t": start_sim_time + 6, "delta_psi": 0.025},
        {"L(t)": 0.65, "t": start_sim_time + 7, "delta_psi": 0.018},
    ]
    lattice_points_2 = [{"L(t)": d["L(t)"], "t": d["t"]} for d in simulated_data_2]
    delta_psi_values_2 = [d["delta_psi"] for d in simulated_data_2]
    
    # Combine old and new data for the new calculation
    combined_lattice_points = lattice_points + lattice_points_2
    combined_delta_psi_values = delta_psi_values + delta_psi_values_2

    warp_momentum_2 = warp_core.calculate_warp_momentum(combined_lattice_points, combined_delta_psi_values)
    print(f"Calculated Warp Momentum (with updated params): {warp_momentum_2:.6f}")

    print("\n--- Metrics After Update and Recalculation ---")
    metrics_2 = warp_core.get_metrics()
    for k, v in metrics_2.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Resetting the Core ---")
    warp_core.reset()
    print(f"Current Warp Momentum after reset: {warp_core.get_metrics()['current_warp_momentum']:.6f}")
    print(f"Total calculations after reset: {warp_core.get_metrics()['total_warp_calculations']}") 