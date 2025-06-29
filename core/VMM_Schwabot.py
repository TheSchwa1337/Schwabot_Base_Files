# -*- coding: utf-8 -*-
"""
Vitruvian Man Management (VMM) System for Schwabot
=================================================

Advanced Vitruvian Man Management system that integrates with the existing
Schwabot mathematical foundation, including NCCO, SFS, UFS, ZPLS, RBMS,
and all mathematical states and ring valuations.

This system maps human geometric proportions to trading logic, creating
a recursive harmonic shell that operates in real-time with the existing
Schwabot infrastructure.

Mathematical Foundation:
- Vitruvian Ratios: Φ = 1.618033988749895 (Golden Ratio)
- Body Zone Mapping: Feet->Entry, Pelvis->Hold, Heart->Balance, Arms->Exit, Halo->Peak
- Recursive Shell Logic: Ψ_vit(t) = sum limb_i RBMS(limb_i_state) * f(ZPLS, theta_i)
- NCCO Integration: ΔΨᵢ = ∇ᵗ[Hₙ ⊕ S(tauᵢ)] · Λᵢ(t) -> Π(chiₙ)
- SFS/UFS Coordination: Unified fault system with sequential fractal stack
- ZPLS Core: Zero-point logic stack centered at navel (Φ-point)
- RBMS Integration: Recursive binary matrix strategy for limb vector flips
"""

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Import existing Schwabot mathematical systems
try:
    from core.balance_loader import get_balance_loader, update_load_metrics
    from core.ghost_trigger_manager import create_ghost_trigger, get_ghost_trigger_manager
    from core.multi_bit_btc_processor import MultiBitBTCProcessor
    from core.tick_management_system import get_tick_manager, run_tick_cycle

    SCHWABOT_CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Schwabot core components not fully available: {e}")
    SCHWABOT_CORE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Mathematical Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio ~= 1.618033988749895
PI = math.pi
E = math.e


class VitruvianZone(Enum):
    """Vitruvian body zones mapped to trading logic."""

    FEET_ENTRY = "feet_entry"  # 0.618 - Entry bucket zone
    PELVIS_HOLD = "pelvis_hold"  # 0.786 - Hold threshold
    HEART_BALANCE = "heart_balance"  # 1.000 - RSI balance zone
    ARMS_EXIT = "arms_exit"  # 1.414 - Exit trigger zone
    HALO_PEAK = "halo_peak"  # 1.618 - Sentiment peak


class LimbVector(Enum):
    """Limb vectors for RBMS integration."""

    LEFT_ARM = "left_arm"  # [0,1] XOR-flip echo symmetry
    RIGHT_ARM = "right_arm"  # [1,0] XOR-flip echo symmetry
    LEFT_LEG = "left_leg"  # [1,1] Static-mirror vector
    RIGHT_LEG = "right_leg"  # [0,0] Static-mirror vector
    HEAD_VECTOR = "head_vector"  # [1,0,0] Inversion over vertical
    SPINE_CORE = "spine_core"  # ZPLS core anchor


class CompressionMode(Enum):
    """Compression modes for ALIF/ALEPH coordination."""

    LO_SYNC = "LO_SYNC"  # Normal operation
    DELTA_DRIFT = "DELTA_DRIFT"  # ALIF fast, ALEPH lagging
    ECHO_GLIDE = "ECHO_GLIDE"  # ALEPH holding, ALIF free
    COMPRESS_HOLD = "COMPRESS_HOLD"  # Both systems restrict entropy
    OVERLOAD_FALLBACK = "OVERLOAD_FALLBACK"  # ALIF stalls, ALEPH fallback


@dataclass
class VitruvianState:
    """Complete state of the Vitruvian system."""

    timestamp: float = field(default_factory=time.time)
    phi_center: float = 0.0  # Navel center point (ZPLS)
    limb_positions: Dict[LimbVector, float] = field(default_factory=dict)
    zone_activations: Dict[VitruvianZone, bool] = field(default_factory=dict)
    compression_mode: CompressionMode = CompressionMode.LO_SYNC
    entropy_score: float = 0.0
    echo_strength: float = 0.0
    drift_score: float = 0.0
    ncco_state: float = 0.0
    sfs_state: float = 0.0
    ufs_state: float = 0.0
    zpls_state: float = 0.0
    rbms_state: float = 0.0
    thermal_state: str = "warm"
    bit_phase: int = 8


@dataclass
class VitruvianTrigger:
    """Trigger for Vitruvian-based trading decisions."""

    trigger_id: str
    zone: VitruvianZone
    limb_vector: LimbVector
    confidence: float
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    volume: float = 1.0
    timestamp: float = field(default_factory=time.time)
    profit_potential: float = 0.0
    risk_score: float = 0.0


class VitruvianManManager:
    """
    Advanced Vitruvian Man Management system for Schwabot.

    Integrates with existing mathematical foundation:
    - NCCO (Network Control and Coordination Orchestrator)
    - SFS (Sequential Fractal Stack)
    - UFS (Unified Fault System)
    - ZPLS (Zero-Point Logic Stack)
    - RBMS (Recursive Binary Matrix Strategy)
    """

    def __init__(self):
        """Initialize the Vitruvian Man Manager."""
        self.current_state = VitruvianState()
        self.trigger_history: List[VitruvianTrigger] = []
        self.limb_reservoir: Dict[str, VitruvianTrigger] = {}

        # Performance tracking
        self.total_triggers = 0
        self.successful_triggers = 0
        self.zone_activations = {zone: 0 for zone in VitruvianZone}

        # Mathematical integration
        self.ncco_integration = True
        self.sfs_integration = True
        self.ufs_integration = True
        self.zpls_integration = True
        self.rbms_integration = True

        # Callbacks for real-time monitoring
        self.state_callbacks: List[Callable[[VitruvianState], None]] = []
        self.trigger_callbacks: List[Callable[[VitruvianTrigger], None]] = []

        # Initialize limb positions
        self._initialize_limb_positions()

        logger.info("🧬 Vitruvian Man Manager initialized")

    def _initialize_limb_positions(self):
        """Initialize limb positions based on Vitruvian proportions."""
        self.current_state.limb_positions = {
            LimbVector.LEFT_ARM: -PHI,  # Left arm at -Φ
            LimbVector.RIGHT_ARM: PHI,  # Right arm at +Φ
            LimbVector.LEFT_LEG: -1.0,  # Left leg at -1
            LimbVector.RIGHT_LEG: 1.0,  # Right leg at +1
            LimbVector.HEAD_VECTOR: 2.0,  # Head at +2
            LimbVector.SPINE_CORE: 0.0,  # Spine at center (ZPLS)
        }

        # Initialize zone activations
        for zone in VitruvianZone:
            self.current_state.zone_activations[zone] = False

    def register_state_callback(self, callback: Callable[[VitruvianState], None]):
        """Register callback for state updates."""
        self.state_callbacks.append(callback)
        logger.debug(f"Registered state callback: {callback.__name__}")

    def register_trigger_callback(self, callback: Callable[[VitruvianTrigger], None]):
        """Register callback for trigger events."""
        self.trigger_callbacks.append(callback)
        logger.debug(f"Registered trigger callback: {callback.__name__}")

    def update_vitruvian_state(
        self, price: float, rsi: float, volume: float, entropy: float, echo_strength: float, drift_score: float
    ) -> VitruvianState:
        """Update the complete Vitruvian state based on market data."""
        try:
            # Update basic state
            self.current_state.timestamp = time.time()
            self.current_state.entropy_score = entropy
            self.current_state.echo_strength = echo_strength
            self.current_state.drift_score = drift_score

            # Calculate phi center (ZPLS integration)
            self.current_state.phi_center = self._calculate_phi_center(price, rsi)

            # Update limb positions based on market movement
            self._update_limb_positions(price, rsi, volume)

            # Determine zone activations
            self._determine_zone_activations(price, rsi)

            # Update compression mode
            self._update_compression_mode(entropy, echo_strength)

            # Integrate with existing mathematical systems
            self._integrate_mathematical_systems(price, rsi, entropy)

            # Execute callbacks
            self._execute_state_callbacks()

            return self.current_state

        except Exception as e:
            logger.error(f"Error updating Vitruvian state: {e}")
            return self.current_state

    def _calculate_phi_center(self, price: float, rsi: float) -> float:
        """Calculate phi center (ZPLS integration point)."""
        # ZPLS = Zero-Point Logic Stack centered at navel
        # Navel position = 5/8 of total height (Vitruvian proportion)
        base_center = 5.0 / 8.0  # 0.625

        # Adjust based on RSI (market sentiment)
        rsi_factor = (rsi - 50.0) / 50.0  # Normalize RSI to [-1, 1]

        # Apply golden ratio scaling
        phi_center = base_center + (rsi_factor * PHI * 0.1)

        return phi_center

    def _update_limb_positions(self, price: float, rsi: float, volume: float):
        """Update limb positions based on market movement."""
        # Calculate price movement factor
        price_factor = (price % 100000) / 100000  # Normalize price

        # Update arm positions (liquidity spread)
        arm_extension = PHI * price_factor
        self.current_state.limb_positions[LimbVector.LEFT_ARM] = -arm_extension
        self.current_state.limb_positions[LimbVector.RIGHT_ARM] = arm_extension

        # Update leg positions (support/resistance)
        leg_stance = 1.0 + (rsi - 50.0) / 100.0
        self.current_state.limb_positions[LimbVector.LEFT_LEG] = -leg_stance
        self.current_state.limb_positions[LimbVector.RIGHT_LEG] = leg_stance

        # Update head position (sentiment peak)
        head_height = 2.0 + (rsi - 50.0) / 25.0
        self.current_state.limb_positions[LimbVector.HEAD_VECTOR] = head_height

        # Spine remains at center (ZPLS anchor)
        self.current_state.limb_positions[LimbVector.SPINE_CORE] = 0.0

    def _determine_zone_activations(self, price: float, rsi: float):
        """Determine which Vitruvian zones are active."""
        # Feet Entry Zone (0.618)
        if rsi < 35 and price < self._calculate_fibonacci_level(0.618):
            self.current_state.zone_activations[VitruvianZone.FEET_ENTRY] = True
            self.zone_activations[VitruvianZone.FEET_ENTRY] += 1
        else:
            self.current_state.zone_activations[VitruvianZone.FEET_ENTRY] = False

        # Pelvis Hold Zone (0.786)
        if 35 <= rsi <= 45:
            self.current_state.zone_activations[VitruvianZone.PELVIS_HOLD] = True
            self.zone_activations[VitruvianZone.PELVIS_HOLD] += 1
        else:
            self.current_state.zone_activations[VitruvianZone.PELVIS_HOLD] = False

        # Heart Balance Zone (1.000)
        if 45 <= rsi <= 55:
            self.current_state.zone_activations[VitruvianZone.HEART_BALANCE] = True
            self.zone_activations[VitruvianZone.HEART_BALANCE] += 1
        else:
            self.current_state.zone_activations[VitruvianZone.HEART_BALANCE] = False

        # Arms Exit Zone (1.414)
        if rsi > 65 and price > self._calculate_fibonacci_level(1.414):
            self.current_state.zone_activations[VitruvianZone.ARMS_EXIT] = True
            self.zone_activations[VitruvianZone.ARMS_EXIT] += 1
        else:
            self.current_state.zone_activations[VitruvianZone.ARMS_EXIT] = False

        # Halo Peak Zone (1.618)
        if rsi > 75 and price > self._calculate_fibonacci_level(1.618):
            self.current_state.zone_activations[VitruvianZone.HALO_PEAK] = True
            self.zone_activations[VitruvianZone.HALO_PEAK] += 1
        else:
            self.current_state.zone_activations[VitruvianZone.HALO_PEAK] = False

    def _calculate_fibonacci_level(self, fib_ratio: float) -> float:
        """Calculate Fibonacci level based on golden ratio."""
        # This would integrate with your existing Fibonacci calculations
        # For now, using a simplified approach
        base_price = 65000.0  # Base BTC price
        return base_price * fib_ratio

    def _update_compression_mode(self, entropy: float, echo_strength: float):
        """Update compression mode based on entropy and echo strength."""
        if entropy > 0.8 and echo_strength < 0.3:
            self.current_state.compression_mode = CompressionMode.OVERLOAD_FALLBACK
        elif entropy > 0.6 and echo_strength < 0.5:
            self.current_state.compression_mode = CompressionMode.COMPRESS_HOLD
        elif entropy < 0.3 and echo_strength > 0.7:
            self.current_state.compression_mode = CompressionMode.ECHO_GLIDE
        elif entropy > 0.5 and echo_strength < 0.4:
            self.current_state.compression_mode = CompressionMode.DELTA_DRIFT
        else:
            self.current_state.compression_mode = CompressionMode.LO_SYNC

    def _integrate_mathematical_systems(self, price: float, rsi: float, entropy: float):
        """Integrate with existing mathematical systems (NCCO, SFS, UFS, ZPLS, RBMS)."""
        try:
            # NCCO Integration
            if self.ncco_integration:
                self.current_state.ncco_state = self._calculate_ncco_state(price, rsi, entropy)

            # SFS Integration
            if self.sfs_integration:
                self.current_state.sfs_state = self._calculate_sfs_state(entropy, self.current_state.echo_strength)

            # UFS Integration
            if self.ufs_integration:
                self.current_state.ufs_state = self._calculate_ufs_state(self.current_state.drift_score)

            # ZPLS Integration
            if self.zpls_integration:
                self.current_state.zpls_state = self.current_state.phi_center

            # RBMS Integration
            if self.rbms_integration:
                self.current_state.rbms_state = self._calculate_rbms_state()

            # Update thermal state and bit phase
            self._update_thermal_state()

        except Exception as e:
            logger.error(f"Error integrating mathematical systems: {e}")

    def _calculate_ncco_state(self, price: float, rsi: float, entropy: float) -> float:
        """Calculate NCCO state based on market data."""
        # NCCO = Network Control and Coordination Orchestrator
        # Simplified NCCO calculation
        price_factor = (price % 100000) / 100000
        rsi_factor = rsi / 100.0
        entropy_factor = entropy

        ncco_state = price_factor * 0.4 + rsi_factor * 0.3 + entropy_factor * 0.3
        return ncco_state

    def _calculate_sfs_state(self, entropy: float, echo_strength: float) -> float:
        """Calculate SFS (Sequential Fractal Stack) state."""
        # SFS = Sequential Fractal Stack
        # Simplified SFS calculation
        sfs_state = entropy * echo_strength * PHI
        return sfs_state

    def _calculate_ufs_state(self, drift_score: float) -> float:
        """Calculate UFS (Unified Fault System) state."""
        # UFS = Unified Fault System
        # Simplified UFS calculation
        ufs_state = 1.0 - abs(drift_score)  # Invert drift for stability
        return max(0.0, min(1.0, ufs_state))

    def _calculate_rbms_state(self) -> float:
        """Calculate RBMS (Recursive Binary Matrix Strategy) state."""
        # RBMS = Recursive Binary Matrix Strategy
        # Calculate based on limb positions
        limb_sum = sum(abs(pos) for pos in self.current_state.limb_positions.values())
        rbms_state = limb_sum / len(self.current_state.limb_positions)
        return rbms_state

    def _update_thermal_state(self):
        """Update thermal state and bit phase based on system load."""
        total_load = (
            self.current_state.entropy_score + self.current_state.echo_strength + self.current_state.drift_score
        ) / 3.0

        if total_load < 0.3:
            self.current_state.thermal_state = "cool"
            self.current_state.bit_phase = 4
        elif total_load < 0.6:
            self.current_state.thermal_state = "warm"
            self.current_state.bit_phase = 8
        elif total_load < 0.8:
            self.current_state.thermal_state = "hot"
            self.current_state.bit_phase = 32
        else:
            self.current_state.thermal_state = "critical"
            self.current_state.bit_phase = 42

    def create_vitruvian_trigger(
        self, zone: VitruvianZone, price: float, rsi: float, volume: float
    ) -> Optional[VitruvianTrigger]:
        """Create a Vitruvian-based trading trigger."""
        try:
            self.total_triggers += 1

            # Generate trigger ID
            trigger_id = f"vit_{self.total_triggers}_{int(time.time())}"

            # Determine limb vector based on zone
            limb_vector = self._zone_to_limb_vector(zone)

            # Calculate confidence based on zone activation
            confidence = self._calculate_trigger_confidence(zone, rsi, price)

            # Calculate profit potential and risk
            profit_potential = self._calculate_profit_potential(zone, price, rsi)
            risk_score = self._calculate_risk_score(zone, rsi, volume)

            trigger = VitruvianTrigger(
                trigger_id=trigger_id,
                zone=zone,
                limb_vector=limb_vector,
                confidence=confidence,
                entry_price=price if zone in [VitruvianZone.FEET_ENTRY, VitruvianZone.PELVIS_HOLD] else None,
                exit_price=price if zone in [VitruvianZone.ARMS_EXIT, VitruvianZone.HALO_PEAK] else None,
                volume=volume,
                profit_potential=profit_potential,
                risk_score=risk_score,
            )

            # Store in history
            self.trigger_history.append(trigger)

            # Execute callbacks
            self._execute_trigger_callbacks(trigger)

            logger.info(f"Created Vitruvian trigger: {trigger_id} in {zone.value} zone")
            return trigger

        except Exception as e:
            logger.error(f"Error creating Vitruvian trigger: {e}")
            return None

    def _zone_to_limb_vector(self, zone: VitruvianZone) -> LimbVector:
        """Map zone to appropriate limb vector."""
        zone_to_limb = {
            VitruvianZone.FEET_ENTRY: LimbVector.LEFT_LEG,
            VitruvianZone.PELVIS_HOLD: LimbVector.RIGHT_LEG,
            VitruvianZone.HEART_BALANCE: LimbVector.SPINE_CORE,
            VitruvianZone.ARMS_EXIT: LimbVector.LEFT_ARM,
            VitruvianZone.HALO_PEAK: LimbVector.HEAD_VECTOR,
        }
        return zone_to_limb.get(zone, LimbVector.SPINE_CORE)

    def _calculate_trigger_confidence(self, zone: VitruvianZone, rsi: float, price: float) -> float:
        """Calculate trigger confidence based on zone and market conditions."""
        base_confidence = 0.5

        # Zone-specific confidence adjustments
        if zone == VitruvianZone.FEET_ENTRY and rsi < 35:
            base_confidence += 0.3
        elif zone == VitruvianZone.PELVIS_HOLD and 35 <= rsi <= 45:
            base_confidence += 0.2
        elif zone == VitruvianZone.HEART_BALANCE and 45 <= rsi <= 55:
            base_confidence += 0.1
        elif zone == VitruvianZone.ARMS_EXIT and rsi > 65:
            base_confidence += 0.3
        elif zone == VitruvianZone.HALO_PEAK and rsi > 75:
            base_confidence += 0.4

        # Price movement confidence
        price_confidence = min(0.2, abs(price % 1000) / 10000)
        base_confidence += price_confidence

        return min(1.0, base_confidence)

    def _calculate_profit_potential(self, zone: VitruvianZone, price: float, rsi: float) -> float:
        """Calculate profit potential based on zone."""
        if zone == VitruvianZone.FEET_ENTRY:
            return 0.02 + (35 - rsi) / 1000  # 2-5% potential
        elif zone == VitruvianZone.ARMS_EXIT:
            return 0.015 + (rsi - 65) / 1000  # 1.5-3% potential
        elif zone == VitruvianZone.HALO_PEAK:
            return 0.025 + (rsi - 75) / 1000  # 2.5-5% potential
        else:
            return 0.01  # 1% for other zones

    def _calculate_risk_score(self, zone: VitruvianZone, rsi: float, volume: float) -> float:
        """Calculate risk score based on zone and market conditions."""
        base_risk = 0.5

        # Zone-specific risk adjustments
        if zone == VitruvianZone.FEET_ENTRY:
            base_risk += 0.2  # Higher risk for entry
        elif zone == VitruvianZone.HALO_PEAK:
            base_risk += 0.3  # Highest risk for peak
        elif zone == VitruvianZone.HEART_BALANCE:
            base_risk -= 0.1  # Lower risk for balance

        # RSI-based risk
        if rsi < 20 or rsi > 80:
            base_risk += 0.2  # Extreme RSI = higher risk

        # Volume-based risk
        volume_risk = min(0.1, volume / 1000000)  # Normalize volume
        base_risk += volume_risk

        return min(1.0, max(0.0, base_risk))

    def get_optimal_trading_route(self, price: float, rsi: float, volume: float) -> Dict[str, Any]:
        """Get optimal trading route based on Vitruvian analysis."""
        try:
            # Update state
            self.update_vitruvian_state(price, rsi, volume, 0.5, 0.5, 0.01)

            # Find active zones
            active_zones = [zone for zone, active in self.current_state.zone_activations.items() if active]

            if not active_zones:
                return {"action": "hold", "reason": "No active Vitruvian zones", "confidence": 0.5, "zone": None}

            # Select best zone based on confidence
            best_trigger = None
            best_confidence = 0.0

            for zone in active_zones:
                trigger = self.create_vitruvian_trigger(zone, price, rsi, volume)
                if trigger and trigger.confidence > best_confidence:
                    best_trigger = trigger
                    best_confidence = trigger.confidence

            if best_trigger:
                action = self._zone_to_action(best_trigger.zone)
                return {
                    "action": action,
                    "reason": f"Vitruvian {best_trigger.zone.value} zone active",
                    "confidence": best_trigger.confidence,
                    "zone": best_trigger.zone.value,
                    "profit_potential": best_trigger.profit_potential,
                    "risk_score": best_trigger.risk_score,
                    "trigger_id": best_trigger.trigger_id,
                }
            else:
                return {"action": "hold", "reason": "No high-confidence triggers", "confidence": 0.3, "zone": None}

        except Exception as e:
            logger.error(f"Error getting optimal trading route: {e}")
            return {"action": "hold", "reason": f"Error: {str(e)}", "confidence": 0.0, "zone": None}

    def _zone_to_action(self, zone: VitruvianZone) -> str:
        """Convert zone to trading action."""
        zone_actions = {
            VitruvianZone.FEET_ENTRY: "buy",
            VitruvianZone.PELVIS_HOLD: "hold",
            VitruvianZone.HEART_BALANCE: "balance",
            VitruvianZone.ARMS_EXIT: "sell",
            VitruvianZone.HALO_PEAK: "exit",
        }
        return zone_actions.get(zone, "hold")

    def _execute_state_callbacks(self):
        """Execute all registered state callbacks."""
        for callback in self.state_callbacks:
            try:
                callback(self.current_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def _execute_trigger_callbacks(self, trigger: VitruvianTrigger):
        """Execute all registered trigger callbacks."""
        for callback in self.trigger_callbacks:
            try:
                callback(trigger)
            except Exception as e:
                logger.error(f"Trigger callback error: {e}")

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "total_triggers": self.total_triggers,
            "successful_triggers": self.successful_triggers,
            "success_rate": self.successful_triggers / max(1, self.total_triggers),
            "zone_activations": self.zone_activations,
            "current_compression_mode": self.current_state.compression_mode.value,
            "current_thermal_state": self.current_state.thermal_state,
            "current_bit_phase": self.current_state.bit_phase,
            "phi_center": self.current_state.phi_center,
            "entropy_score": self.current_state.entropy_score,
            "echo_strength": self.current_state.echo_strength,
            "drift_score": self.current_state.drift_score,
            "mathematical_states": {
                "ncco_state": self.current_state.ncco_state,
                "sfs_state": self.current_state.sfs_state,
                "ufs_state": self.current_state.ufs_state,
                "zpls_state": self.current_state.zpls_state,
                "rbms_state": self.current_state.rbms_state,
            },
        }


# Global Vitruvian Man Manager instance
vitruvian_manager = VitruvianManManager()


# Integration functions for external use
def get_vitruvian_manager() -> VitruvianManManager:
    """Get the global Vitruvian Man Manager instance."""
    return vitruvian_manager


def update_vitruvian_state(
    price: float, rsi: float, volume: float, entropy: float, echo_strength: float, drift_score: float
) -> VitruvianState:
    """Update Vitruvian state with market data."""
    return vitruvian_manager.update_vitruvian_state(price, rsi, volume, entropy, echo_strength, drift_score)


def get_optimal_trading_route(price: float, rsi: float, volume: float) -> Dict[str, Any]:
    """Get optimal trading route based on Vitruvian analysis."""
    return vitruvian_manager.get_optimal_trading_route(price, rsi, volume)


def create_vitruvian_trigger(
    zone: VitruvianZone, price: float, rsi: float, volume: float
) -> Optional[VitruvianTrigger]:
    """Create a Vitruvian-based trading trigger."""
    return vitruvian_manager.create_vitruvian_trigger(zone, price, rsi, volume)


def get_vitruvian_statistics() -> Dict[str, Any]:
    """Get Vitruvian system statistics."""
    return vitruvian_manager.get_system_statistics()


def register_vitruvian_state_callback(callback: Callable[[VitruvianState], None]):
    """Register a callback for Vitruvian state updates."""
    vitruvian_manager.register_state_callback(callback)


def register_vitruvian_trigger_callback(callback: Callable[[VitruvianTrigger], None]):
    """Register a callback for Vitruvian trigger events."""
    vitruvian_manager.register_trigger_callback(callback)
