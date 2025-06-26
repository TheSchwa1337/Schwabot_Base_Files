# -*- coding: utf-8 -*-
from __future__ import annotations
from core.unified_math_system import unified_math
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import logging
import numpy as np
import math

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message):
            print(message)

        def info(message):
            print(f"[INFO] {message}")

        def warn(message):
            print(f"[WARN] {message}")

        def error(message):
            print(f"[ERROR] {message}")

        def success(message):
            print(f"[SUCCESS] {message}")

        def debug(message):
            print(f"[DEBUG] {message}")

"""Altitude Adjustment Math - Market Altitude & STAM Zone Management.

This module implements mathematical models for market altitude, density, and
stratified zones with velocity-altitude paradox calculations and correction vectors.

Mathematical Foundation:
- Market altitude = 1 - unified_math.min(volume_density, 1.0)
- STAM zones: Stratified Atmospheric Market zones
- Velocity-altitude paradox: v_correction = altitude * volatility_factor
- Autonomic reflex scoring: R_auto = Σ(drift_i * pressure_i)

Windows CLI compatible with comprehensive error handling.
"""

logger = logging.getLogger(__name__)


class STAMZone(Enum):
    """Stratified Atmospheric Market zones."""
    TROPOSPHERE = "troposphere"  # 0.0 - 0.3 altitude (dense, stable)
    STRATOSPHERE = "stratosphere"  # 0.3 - 0.6 altitude (moderate)
    MESOSPHERE = "mesosphere"  # 0.6 - 0.8 altitude (thin, volatile)
    THERMOSPHERE = "thermosphere"  # 0.8 - 1.0 altitude (extreme, chaotic)


@dataclass
class AltitudeMetrics:
    """Market altitude and related metrics."""
    altitude: float  # Market altitude [0, 1]
    stam_zone: STAMZone  # Current STAM zone
    velocity_correction: float  # Velocity correction factor
    pressure_gradient: float  # Pressure gradient
    autonomic_reflex: float  # Autonomic reflex score
    stability_index: float  # Zone stability index
    recommended_strategy: str  # Recommended trading strategy


@dataclass
class VelocityAltitudeState:
    """Velocity-altitude paradox state."""
    velocity: float
    altitude: float
    paradox_factor: float
    correction_vector: float
    energy_dissipation: float


class AltitudeAdjustmentEngine:
    """Engine for altitude-based market analysis and adjustments."""

    def __init__(self):
        """Initialize altitude adjustment engine."""
    self.altitude_history: List[float] = []
    self.velocity_history: List[float] = []
    self.pressure_history: List[float] = []
    self.max_history = 100

        # STAM zone thresholds
    self.stam_thresholds = {
            STAMZone.TROPOSPHERE: (0.0, 0.3),
            STAMZone.STRATOSPHERE: (0.3, 0.6),
            STAMZone.MESOSPHERE: (0.6, 0.8),
            STAMZone.THERMOSPHERE: (0.8, 1.0),
        }

        # Strategy recommendations per zone
    self.zone_strategies = {
            STAMZone.TROPOSPHERE: "aggressive_accumulation",
            STAMZone.STRATOSPHERE: "balanced_trading",
            STAMZone.MESOSPHERE: "conservative_scaling",
            STAMZone.THERMOSPHERE: "emergency_vault_mode",
        }

    def calculate_market_altitude(
        self,
        volume_density: float,
        volatility: float,
        liquidity_depth: float,
    ) -> float:
        """Calculate market altitude from density and volatility.

        Mathematical Formula:
        altitude = 1 - unified_math.min(volume_density, 1.0) + volatility_adjustment

        Parameters
        ----------
        volume_density : float
            Volume density score [0, 1]
        volatility : float
            Market volatility measure
        liquidity_depth : float
            Liquidity depth score [0, 1]

        Returns
        -------
        float
            Market altitude [0, 1]
        """
        try:
            # Base altitude from volume density (inverse relationship)
            base_altitude = 1.0 - unified_math.min(volume_density, 1.0)

            # Volatility adjustment (higher volatility = higher altitude)
            volatility_factor = unified_math.min(volatility * 0.3, 0.3)

            # Liquidity adjustment (lower liquidity = higher altitude)
            liquidity_factor = (1.0 - liquidity_depth) * 0.2

            # Combined altitude
            altitude = base_altitude + volatility_factor + liquidity_factor

            # Ensure bounds [0, 1]
            altitude = unified_math.max(0.0, unified_math.min(1.0, altitude))

            # Update history
                self.altitude_history.append(altitude)
            if len(self.altitude_history) > self.max_history:
                    self.altitude_history = self.altitude_history[-self.max_history:]

            return altitude
        except Exception as e:
            logger.error(f"Error calculating market altitude: {e}")
            return 0.5

    def determine_stam_zone(self, altitude: float) -> STAMZone:
        """Determine STAM zone from altitude."""
        for zone, (min_alt, max_alt) in self.stam_thresholds.items():
            if min_alt <= altitude < max_alt:
                return zone
        return STAMZone.THERMOSPHERE  # Default to highest zone

    def calculate_velocity_altitude_paradox(
        self,
        velocity: float,
        altitude: float,
        market_pressure: float,
    ) -> VelocityAltitudeState:
        """Calculate velocity-altitude paradox state.

        Mathematical Formula:
        paradox_factor = velocity² / (altitude + ε)
        correction_vector = altitude * volatility_factor * pressure_modifier

        Parameters
        ----------
        velocity : float
            Current market velocity
        altitude : float
            Market altitude [0, 1]
        market_pressure : float
            Market pressure indicator

        Returns
        -------
        VelocityAltitudeState
            The calculated state of the velocity-altitude paradox.
        """
        try:
            epsilon = 1e-6
            paradox_factor = (velocity**2) / (altitude + epsilon)

            # Pressure modifier: higher pressure dampens correction
            pressure_modifier = unified_math.exp(-market_pressure)

            # Volatility factor from recent altitude changes
            if len(self.altitude_history) > 1:
                volatility_factor = unified_math.std(
                    self.altitude_history[-10:])
            else:
                volatility_factor = 0

            correction_vector = altitude * volatility_factor * pressure_modifier

            # Energy dissipation
            energy_dissipation = 0.5 * (velocity**2) * (1 - pressure_modifier)

            state = VelocityAltitudeState(
                velocity=velocity,
                altitude=altitude,
                paradox_factor=paradox_factor,
                correction_vector=correction_vector,
                energy_dissipation=energy_dissipation,
            )
            return state

        except Exception as e:
            logger.error(
                f"Error in velocity-altitude paradox calculation: {e}")
            return VelocityAltitudeState(
                velocity=velocity,
                altitude=altitude,
                paradox_factor=0,
                correction_vector=0,
                energy_dissipation=0,
            )

    def get_altitude_metrics(
        self,
        altitude: float,
        drift_vector: List[float],
        pressure_vector: List[float],
    ) -> AltitudeMetrics:
        """Get comprehensive altitude metrics.

        Parameters
        ----------
        altitude : float
            Current market altitude.
        drift_vector : List[float]
            Vector of market drift components.
        pressure_vector : List[float]
            Vector of market pressure components.

        Returns
        -------
        AltitudeMetrics
            A dataclass containing all calculated altitude metrics.
        """
        try:
            stam_zone = self.determine_stam_zone(altitude)

            # Velocity correction based on zone
            if stam_zone == STAMZone.TROPOSPHERE:
                velocity_correction = 0.1
            elif stam_zone == STAMZone.STRATOSPHERE:
                velocity_correction = 0.3
            elif stam_zone == STAMZone.MESOSPHERE:
                velocity_correction = 0.6
            else:
                velocity_correction = 1.0

            # Pressure gradient from history
            if len(self.pressure_history) > 1:
                pressure_gradient = unified_math.mean(
                    np.diff(self.pressure_history[-10:])
                )
            else:
                pressure_gradient = 0

            # Autonomic reflex score
            autonomic_reflex = unified_math.sum(
                [d * p for d, p in zip(drift_vector, pressure_vector)]
            )

            # Stability index
            stability_index = 1.0 - altitude * (1 + abs(pressure_gradient))

            # Recommended strategy
            recommended_strategy = self.zone_strategies.get(
                stam_zone, "unknown"
            )

            metrics = AltitudeMetrics(
                altitude=altitude,
                stam_zone=stam_zone,
                velocity_correction=velocity_correction,
                pressure_gradient=pressure_gradient,
                autonomic_reflex=autonomic_reflex,
                stability_index=stability_index,
                recommended_strategy=recommended_strategy,
            )
            return metrics

        except Exception as e:
            logger.error(f"Error getting altitude metrics: {e}")
            return AltitudeMetrics(
                altitude=0.5,
                stam_zone=STAMZone.STRATOSPHERE,
                velocity_correction=0,
                pressure_gradient=0,
                autonomic_reflex=0,
                stability_index=0.5,
                recommended_strategy="unknown",
            )

    def update_history(
        self, altitude: float, velocity: float, pressure: float
    ):
        """Update historical data for altitude, velocity, and pressure."""
        self.altitude_history.append(altitude)
        self.velocity_history.append(velocity)
        self.pressure_history.append(pressure)

        if len(self.altitude_history) > self.max_history:
            self.altitude_history.pop(0)
        if len(self.velocity_history) > self.max_history:
                self.velocity_history.pop(0)
        if len(self.pressure_history) > self.max_history:
                    self.pressure_history.pop(0)


if __name__ == '__main__':
    # Example Usage
    engine = AltitudeAdjustmentEngine()

    # Simulate market data
    volume_density = 0.2
    volatility = 0.5
    liquidity_depth = 0.8
    velocity = 1.2
    pressure = 0.3
    drift = [0.1, -0.05, 0.2]
    pressures = [0.5, 0.6, 0.4]

    # Calculate altitude
    altitude = engine.calculate_market_altitude(
        volume_density, volatility, liquidity_depth
    )
    safe_print(f"Market Altitude: {altitude:.3f}")

    # Determine STAM Zone
    zone = engine.determine_stam_zone(altitude)
    safe_print(f"STAM Zone: {zone.value}")

    # Update history
    engine.update_history(altitude, velocity, pressure)

    # Get metrics
    metrics = engine.get_altitude_metrics(altitude, drift, pressures)
    safe_print("--- Altitude Metrics ---")
    safe_print(f"  Stability Index: {metrics.stability_index:.3f}")
    safe_print(f"  Autonomic Reflex: {metrics.autonomic_reflex:.3f}")
    safe_print(f"  Recommended Strategy: {metrics.recommended_strategy}")

    # Velocity-altitude paradox
    paradox_state = engine.calculate_velocity_altitude_paradox(
        velocity, altitude, pressure
    )
    safe_print("\n--- Velocity-Altitude Paradox ---")
    safe_print(f"  Paradox Factor: {paradox_state.paradox_factor:.3f}")
    safe_print(f"  Correction Vector: {paradox_state.correction_vector:.3f}")
    safe_print(f"  Energy Dissipation: {paradox_state.energy_dissipation:.3f}")

    # Example of how altitude affects strategy
    safe_print(
        f"\nAltitude: {metrics.altitude:.2f} -> Strategy: {metrics.recommended_strategy}")
