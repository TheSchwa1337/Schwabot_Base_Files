# -*- coding: utf-8 -*-
""""""
Zygote Re-entry System - Recursive Re-entry into Earlier Profitable States
===========================================================================

This module implements the Zygote State Transfer Function (Ζ(t)), a critical mathematical
component that defines recursive re-entry into earlier profitable states via Zygote reactivation logic.
This allows Schwabot to reclaim lost vector alignment by re-entering a past successful strategy.

Mathematical Definition:
Z(t) = Σᵢ=₁ᴺ ζᵢ ⋅ e^(-λ ⋅ (t − tᵢ)) ⋅ δ(Pᵢ > P*)
Where:
- ζᵢ is the weight of each stored trade signal (e.g., historical profit, trade volume, confidence).
- λ is the decay constant (as usual).
- t is the current time.
- tᵢ is the timestamp of the i-th stored profitable state.
- δ(Pᵢ > P*) is the Dirac delta function, which is 1 if the prior profit Pᵢ exceeded a threshold P*, and 0 otherwise.
- N is the number of historical profitable states to consider.

If Ζ(t) crosses a temporal critical point, Schwabot re-enters a past strategy to reclaim lost vector alignment.

This system helps in:
- Enabling adaptive recovery from periods of underperformance.
- Leveraging historical successful trading patterns.
- Optimizing long-term profit generation by intelligent re-entry.
""""""

import math
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProfitableState:
    profit: float
    weight: float
    timestamp: float


class ZygoteReentrySystem:
    """"""
    Manages and calculates the Zygote State Transfer Function (Ζ(t)).
    """"""

    def __init__(self, decay_constant: float = 0.5, profit_threshold: float = 0.1):
        self.decay_constant = decay_constant
        self.profit_threshold = profit_threshold
        self.past_profitable_states: List[ProfitableState] = []  # Stores ProfitableState objects
        self.max_history_states: int = 100 # Maximum number of states to keep in history
        logger.info("🔂 Zygote Re-entry System initialized.")

    def add_profitable_state(self, profit: float, weight: float = 1.0, timestamp: Optional[float] = None):
        """"""
        Adds a profitable state to the history.

        Args:
            profit: The profit achieved in this state.
            weight: The weight of this state (e.g., confidence, volume).
            timestamp: The timestamp of this state. Defaults to current time.
        """"""
        if timestamp is None:
            timestamp = time.time()

        state = ProfitableState(profit=profit, weight=weight, timestamp=timestamp)
        self.past_profitable_states.append(state)

        if len(self.past_profitable_states) > self.max_history_states:
            self.past_profitable_states.pop(0) # Keep history size bounded

        logger.debug(f"Added profitable state: profit={profit:.4f}, weight={weight:.2f}")

    def calculate_zygote_state(self, current_time: Optional[float] = None) -> float:
        """"""
        Calculates the Zygote State Transfer Function (Ζ(t)).

        Args:
            current_time: The current time. Defaults to time.time().

        Returns:
            The calculated Zygote State (Ζ(t)).
        """"""
        if current_time is None:
            current_time = time.time()

        zygote_state = 0.0

        for state in self.past_profitable_states:
            zeta_i = state.weight
            t_i = state.timestamp
            P_i = state.profit

            # Calculate the exponential decay term
            decay_term = np.exp(-self.decay_constant * (current_time - t_i))

            # Dirac delta-like function (1 if profit > threshold, 0 otherwise)
            delta_function = 1.0 if P_i > self.profit_threshold else 0.0

            zygote_state += zeta_i * decay_term * delta_function

        logger.debug(f"Calculated Zygote State (Ζ): {zygote_state:.4f}")
        return float(zygote_state)

    def get_past_profitable_states(self, count: int = 10) -> List[ProfitableState]:
        """"""
        Returns a list of recent profitable states.
        """"""
        return self.past_profitable_states[-count:]

    def reset_states(self):
        """"""
        Resets the history of profitable states.
        """"""
        self.past_profitable_states = []
        logger.info("Zygote profitable states history reset.")


# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    zygote_system = ZygoteReentrySystem(decay_constant=0.1, profit_threshold=0.5)

    # Add some profitable states over time
    print("Adding profitable states...")
    time.sleep(0.5)
    zygote_system.add_profitable_state(profit=0.2, weight=1.5)
    time.sleep(0.5)
    zygote_system.add_profitable_state(profit=0.3, weight=0.8) # Below threshold
    time.sleep(0.5)
    zygote_system.add_profitable_state(profit=0.15, weight=1.2)
    time.sleep(0.5)
    zygote_system.add_profitable_state(profit=0.5, weight=2.0)

    print(f"Current Zygote State: {zygote_system.calculate_zygote_state():.4f}")

    # Simulate time passing
    print("Simulating time passing...")
    time.sleep(2.0)
    print(f"Zygote State after 2 seconds: {zygote_system.calculate_zygote_state():.4f}")

    # Add more states, some below threshold
    time.sleep(0.5)
    zygote_system.add_profitable_state(profit=0.1, weight=0.3)
    time.sleep(0.5)
    zygote_system.add_profitable_state(profit=0.3, weight=1.8)

    print(f"Zygote State after more states and time: {zygote_system.calculate_zygote_state():.4f}")

    print("Recent profitable states:")
    for state in zygote_system.get_past_profitable_states(5):
        print(f"  Timestamp: {state.timestamp:.2f}, Profit: {state.profit:.4f}, Weight: {state.weight:.2f}")

    zygote_system.reset_states()
    print(f"Zygote State after reset: {zygote_system.calculate_zygote_state():.4f}")