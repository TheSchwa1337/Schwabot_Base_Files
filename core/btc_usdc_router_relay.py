#!/usr/bin/env python3
"""btc_usdc_router_relay – BTC/USDC routing with ghost conditional triggers.

Implements the ghost conditional trigger logic:
    Θᴳ(t) = Σ θₖ·ζₖ(t) · δ(t − τₖ)

This module handles routing between BTC and USDC flows with conditional
trigger detection for the ghost protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__: list[str] = [
    "BTCUSDCRouterRelay",
    "compute_ghost_triggers",
    "route_btc_usdc_flow",
]


@dataclass(slots=True)
class BTCUSDCRouterRelay:
    """BTC/USDC router with ghost conditional triggers."""
    
    trigger_threshold: float = 0.5
    delta_tolerance: float = 0.1

    def compute_theta_g(
        self,
        theta_values: Sequence[float],
        zeta_series: Sequence[float],
        timestamps: Sequence[float],
        trigger_times: Sequence[float],
    ) -> float:
        """Compute Θᴳ(t) = Σ θₖ·ζₖ(t) · δ(t − τₖ).

        Parameters
        ----------
        theta_values
            Theta coefficients θₖ.
        zeta_series
            Zeta time series ζₖ(t).
        timestamps
            Time points t.
        trigger_times
            Trigger times τₖ.
        """
        if len(theta_values) != len(trigger_times):
            raise ValueError("theta_values and trigger_times must have same length")
        
        theta_array = np.asarray(theta_values, dtype=float)
        zeta_array = np.asarray(zeta_series, dtype=float)
        times = np.asarray(timestamps, dtype=float)
        triggers = np.asarray(trigger_times, dtype=float)
        
        theta_g_total = 0.0
        
        # Sum over all k: θₖ·ζₖ(t) · δ(t − τₖ)
        for k, (theta_k, tau_k) in enumerate(zip(theta_array, triggers)):
            # Find zeta value at trigger time (interpolate if needed)
            if len(zeta_array) == len(times):
                zeta_k_t = float(np.interp(tau_k, times, zeta_array))
            else:
                # Use index-based lookup if lengths don't match
                idx = min(k, len(zeta_array) - 1)
                zeta_k_t = zeta_array[idx]
            
            # Dirac delta approximation: δ(t − τₖ) ≈ 1 if |t - τₖ| < tolerance
            for t in times:
                if abs(t - tau_k) < self.delta_tolerance:
                    delta_term = 1.0 / self.delta_tolerance  # normalized
                    theta_g_total += theta_k * zeta_k_t * delta_term
        
        return theta_g_total

    def route_flow_decision(
        self,
        btc_flow: float,
        usdc_flow: float,
        ghost_trigger_strength: float,
    ) -> tuple[float, float]:
        """Route BTC/USDC flows based on ghost trigger strength.

        Parameters
        ----------
        btc_flow
            Current BTC flow rate.
        usdc_flow
            Current USDC flow rate.
        ghost_trigger_strength
            Computed Θᴳ(t) trigger strength.

        Returns
        -------
        tuple[float, float]
            (routed_btc_flow, routed_usdc_flow)
        """
        # Apply routing based on trigger strength
        if ghost_trigger_strength > self.trigger_threshold:
            # Strong trigger: favor BTC
            routing_factor = min(ghost_trigger_strength, 2.0)
            routed_btc = btc_flow * routing_factor
            routed_usdc = usdc_flow / routing_factor
        elif ghost_trigger_strength < -self.trigger_threshold:
            # Negative trigger: favor USDC
            routing_factor = min(abs(ghost_trigger_strength), 2.0)
            routed_btc = btc_flow / routing_factor
            routed_usdc = usdc_flow * routing_factor
        else:
            # Neutral: maintain current flows
            routed_btc = btc_flow
            routed_usdc = usdc_flow
        
        return routed_btc, routed_usdc

    def process_relay_cycle(
        self,
        btc_flows: Sequence[float],
        usdc_flows: Sequence[float],
        theta_values: Sequence[float],
        zeta_series: Sequence[float],
        timestamps: Sequence[float],
        trigger_times: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process complete relay cycle with ghost triggers.

        Parameters
        ----------
        btc_flows, usdc_flows
            Input flow sequences.
        theta_values, zeta_series, timestamps, trigger_times
            Ghost trigger calculation parameters.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (routed_btc_flows, routed_usdc_flows)
        """
        if len(btc_flows) != len(usdc_flows):
            raise ValueError("BTC and USDC flows must have same length")
        
        btc_array = np.asarray(btc_flows, dtype=float)
        usdc_array = np.asarray(usdc_flows, dtype=float)
        
        routed_btc = np.zeros_like(btc_array)
        routed_usdc = np.zeros_like(usdc_array)
        
        # Compute ghost trigger for this cycle
        ghost_strength = self.compute_theta_g(
            theta_values, zeta_series, timestamps, trigger_times
        )
        
        # Route each flow pair
        for i, (btc_flow, usdc_flow) in enumerate(zip(btc_array, usdc_array)):
            routed_btc[i], routed_usdc[i] = self.route_flow_decision(
                btc_flow, usdc_flow, ghost_strength
            )
        
        return routed_btc, routed_usdc


# Functional helpers

def compute_ghost_triggers(
    theta_values: Sequence[float],
    zeta_series: Sequence[float],
    timestamps: Sequence[float],
    trigger_times: Sequence[float],
    delta_tolerance: float = 0.1,
) -> float:  # noqa: D401
    """Compute ghost conditional triggers Θᴳ(t)."""
    relay = BTCUSDCRouterRelay(delta_tolerance=delta_tolerance)
    return relay.compute_theta_g(theta_values, zeta_series, timestamps, trigger_times)


def route_btc_usdc_flow(
    btc_flow: float,
    usdc_flow: float,
    ghost_trigger_strength: float,
    threshold: float = 0.5,
) -> tuple[float, float]:  # noqa: D401
    """Route BTC/USDC flows using ghost trigger strength."""
    relay = BTCUSDCRouterRelay(trigger_threshold=threshold)
    return relay.route_flow_decision(btc_flow, usdc_flow, ghost_trigger_strength) 