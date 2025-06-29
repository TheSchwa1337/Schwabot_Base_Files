"""
Profit Memory Echo Module
-------------------------
Implements the Recursive Memory Projection (Fₑ(t)), allowing Schwabot to "echo"
previous profitable logic by replaying or biasing decisions based on past successful lattice states.
"""

import numpy as np
import time
from typing import Dict, Any, Optional

class ProfitMemoryEcho:
    """
    Manages the recursive memory projection to leverage past profitable lattice states.
    """

    def __init__(self, memory_offset: int = 72, volatility_scalar: float = 1.0):
        """
        Initializes the ProfitMemoryEcho.

        Args:
            memory_offset: The fractal memory offset (τ), e.g., 72 ticks ago.
            volatility_scalar: Volatility scalar (σ) for gain/loss risk shift.
        """
        self.memory_offset = memory_offset
        self.volatility_scalar = volatility_scalar
        self.lattice_history: Dict[int, Dict[str, Any]] = {}
        self.metrics: Dict[str, Any] = {
            "total_projections": 0,
            "successful_echoes": 0,
            "last_projection_time": None
        }

    def store_lattice_state(self, tick_id: int, lattice_state: float, profit_vector: float):
        """
        Stores a lattice state and its associated profit vector.

        Args:
            tick_id: The unique identifier for the tick (e.g., timestamp or tick count).
            lattice_state: The L(t) value for the current tick.
            profit_vector: The ΔL (change in lattice state = profit vector) for the current tick.
        """
        self.lattice_history[tick_id] = {
            "L(t)": lattice_state,
            "ΔL": profit_vector,
            "timestamp": time.time()
        }

    def retrieve_memory_projection(self, current_tick_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the recursive memory projection (Fₑ(t)) based on the memory offset.

        Fₑ(t) = L(t - τ) + ΔL * σ⁻¹

        Args:
            current_tick_id: The unique identifier for the current tick.

        Returns:
            A dictionary containing the projected memory state and its components,
            or None if the historical state is not found.
        """
        self.metrics["total_projections"] += 1
        self.metrics["last_projection_time"] = time.time()

        # Calculate the historical tick ID based on offset
        historical_tick_id = current_tick_id - self.memory_offset

        if historical_tick_id in self.lattice_history:
            historical_data = self.lattice_history[historical_tick_id]
            
            l_t_minus_tau = historical_data["L(t)"]
            delta_l = historical_data["ΔL"]
            
            # Ensure volatility_scalar is not zero to avoid division by zero
            effective_volatility_scalar = max(self.volatility_scalar, 1e-9)

            f_e_t = l_t_minus_tau + (delta_l / effective_volatility_scalar)
            
            self.metrics["successful_echoes"] += 1
            return {
                "projected_value": f_e_t,
                "L(t-τ)": l_t_minus_tau,
                "ΔL": delta_l,
                "σ_inv": 1 / effective_volatility_scalar,
                "historical_tick_id": historical_tick_id
            }
        else:
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns the operational metrics of the Profit Memory Echo.
        """
        return self.metrics

    def update_parameters(self, new_memory_offset: Optional[int] = None,
                            new_volatility_scalar: Optional[float] = None):
        """
        Updates the parameters for memory projection.
        """
        if new_memory_offset is not None:
            self.memory_offset = new_memory_offset
        if new_volatility_scalar is not None:
            self.volatility_scalar = new_volatility_scalar
        print("Profit Memory Echo parameters updated.")

    def reset(self):
        """
        Resets the memory echo's history and metrics.
        """
        self.lattice_history = {}
        self.metrics = {
            "total_projections": 0,
            "successful_echoes": 0,
            "last_projection_time": None
        }

if __name__ == "__main__":
    print("--- Profit Memory Echo Demo ---")

    memory_echo = ProfitMemoryEcho(memory_offset=5, volatility_scalar=0.5)

    # Simulate storing lattice states over time
    print("\n--- Storing Lattice States ---")
    for i in range(1, 11):
        # Simulate L(t) and ΔL (profit vector)
        lattice_val = 0.5 + (i * 0.01) # Simple increasing lattice value
        profit_change = 0.02 * (i % 3 - 1) # Oscillating profit change
        memory_echo.store_lattice_state(i, lattice_val, profit_change)
        print(f"Stored Tick {i}: L(t)={lattice_val:.2f}, ΔL={profit_change:.2f}")

    print("\n--- Retrieving Memory Projections ---")

    # Test Case 1: Retrieve memory for a tick within the offset
    current_tick_1 = 10
    projection_1 = memory_echo.retrieve_memory_projection(current_tick_1)
    if projection_1:
        print(f"Current Tick {current_tick_1}: Projected Memory: {projection_1['projected_value']:.4f}")
        print(f"  L(t-τ): {projection_1['L(t-τ)']:.2f}, ΔL: {projection_1['ΔL']:.2f}")
    else:
        print(f"Current Tick {current_tick_1}: No memory projection found.")
    print(f"Metrics: {memory_echo.get_metrics()}")

    # Test Case 2: Retrieve memory for a tick outside the offset (should be None)
    current_tick_2 = 3
    projection_2 = memory_echo.retrieve_memory_projection(current_tick_2)
    if projection_2:
        print(f"Current Tick {current_tick_2}: Projected Memory: {projection_2['projected_value']:.4f}")
    else:
        print(f"Current Tick {current_tick_2}: No memory projection found (expected).")
    print(f"Metrics: {memory_echo.get_metrics()}")

    # Test Case 3: Update parameters and test again
    print("\n--- Updating Parameters and Retesting ---")
    memory_echo.update_parameters(new_memory_offset=2, new_volatility_scalar=0.1)
    current_tick_3 = 10
    projection_3 = memory_echo.retrieve_memory_projection(current_tick_3)
    if projection_3:
        print(f"Current Tick {current_tick_3} (new offset): Projected Memory: {projection_3['projected_value']:.4f}")
    else:
        print(f"Current Tick {current_tick_3} (new offset): No memory projection found.")
    print(f"Metrics: {memory_echo.get_metrics()}")

    print("\n--- Resetting the Memory Echo ---")
    memory_echo.reset()
    print(f"Metrics after reset: {memory_echo.get_metrics()}")
    print(f"Lattice history after reset: {memory_echo.lattice_history}") 