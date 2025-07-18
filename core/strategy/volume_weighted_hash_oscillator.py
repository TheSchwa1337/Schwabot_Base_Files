#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume-Weighted Hash Oscillator - Volume Entropy Collisions & VWAP+SHA Fusion
============================================================================

Implements the Nexus mathematics for volume-weighted hash oscillators:
- VWAP Drift Collapse: VW_shift = Σᵢ(Vᵢ⋅Pᵢ)/ΣᵢVᵢ - P_hash(t)
- Entropic Oscillator Pulse: H_osc(t) = sin(2πt/τ + ψ_SHA256) ⋅ log(Vₜ)
- Smart Money logic for recursive SHA-field volume behavior
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OscillatorMode(Enum):
    """Volume-weighted hash oscillator modes."""

    VWAP = "vwap"  # Volume-weighted average price
    HASH = "hash"  # Hash-based oscillation
    ENTROPIC = "entropic"  # Entropic oscillator pulse
    SMART_MONEY = "smart_money"  # Smart money detection
    LIQUIDITY = "liquidity"  # Liquidity wall detection


class SignalType(Enum):
    """Signal types for oscillator output."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    PUMP = "pump"
    DUMP = "dump"
    NEUTRAL = "neutral"


@dataclass
class VolumeData:
    """Volume data structure."""

    timestamp: float
    price: float
    volume: float
    bid: float
    ask: float
    high: float
    low: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HashOscillatorResult:
    """Hash oscillator calculation result."""

    timestamp: float
    vwap_value: float
    hash_value: float
    oscillator_value: float
    entropic_pulse: float
    signal_type: SignalType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VolumeWeightedHashOscillator:
    """
    Volume-Weighted Hash Oscillator - Volume Entropy Collisions & VWAP+SHA Fusion

    Implements the Nexus mathematics for volume-weighted hash oscillators:
    - VWAP Drift Collapse: VW_shift = Σᵢ(Vᵢ⋅Pᵢ)/ΣᵢVᵢ - P_hash(t)
    - Entropic Oscillator Pulse: H_osc(t) = sin(2πt/τ + ψ_SHA256) ⋅ log(Vₜ)
    - Smart Money logic for recursive SHA-field volume behavior
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Volume-Weighted Hash Oscillator."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.mode = OscillatorMode.VWAP
        self.initialized = False

        # Oscillator parameters
        self.period = self.config.get("period", 20)
        self.smoothing_period = self.config.get("smoothing_period", 10)
        self.hash_strength = self.config.get("hash_strength", 8)
        self.normalize = self.config.get("normalize", True)
        self.oscillator_range = self.config.get("oscillator_range", (-1.0, 1.0))

        # Entropic parameters
        self.tau_period = self.config.get("tau_period", 100)
        self.entropy_threshold = self.config.get("entropy_threshold", 0.1)
        self.liquidity_threshold = self.config.get("liquidity_threshold", 0.05)

        # Data storage
        self.volume_history: List[VolumeData] = []
        self.vwap_history: List[float] = []
        self.hash_history: List[float] = []
        self.oscillator_history: List[float] = []
        self.entropic_history: List[float] = []

        self._initialize_oscillator()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Volume-Weighted Hash Oscillator."""
        return {
            "period": 20,  # Look-back period
            "smoothing_period": 10,  # Smoothing period
            "hash_strength": 8,  # Hash strength (1-64)
            "normalize": True,  # Normalize output
            "oscillator_range": (-1.0, 1.0),  # Output range
            "tau_period": 100,  # Entropic period
            "entropy_threshold": 0.1,  # Entropy threshold
            "liquidity_threshold": 0.05,  # Liquidity threshold
            "pump_threshold": 0.7,  # Pump detection threshold
            "dump_threshold": -0.7,  # Dump detection threshold
        }

    def _initialize_oscillator(self) -> None:
        """Initialize the oscillator."""
        try:
            self.logger.info("Initializing Volume-Weighted Hash Oscillator...")

            # Validate hash strength
            if not (0 < self.hash_strength <= 64):
                raise ValueError("hash_strength must be between 1 and 64")

            # Initialize SHA256 context
            self.sha256_context = hashlib.sha256()

            # Initialize smoothing filters
            self.vwap_filter = np.ones(self.smoothing_period) / self.smoothing_period
            self.hash_filter = np.ones(self.smoothing_period) / self.smoothing_period

            self.initialized = True
            self.logger.info(
                "[SUCCESS] Volume-Weighted Hash Oscillator initialized successfully"
            )
        except Exception as e:
            self.logger.error(
                f"[FAIL] Error initializing Volume-Weighted Hash Oscillator: {e}"
            )
            self.initialized = False

    def compute_vwap_drift_collapse(self, volume_data: List[VolumeData]) -> float:
        """
        Compute VWAP drift collapse: VW_shift = Σᵢ(Vᵢ⋅Pᵢ)/ΣᵢVᵢ - P_hash(t)

        Args:
        volume_data: List of volume data points

        Returns:
        VWAP drift collapse value
        """
        try:
            if len(volume_data) < 2:
                return 0.0

            # Extract volumes and prices
            volumes = np.array([d.volume for d in volume_data])
            prices = np.array([d.price for d in volume_data])

            # Compute VWAP: Σᵢ(Vᵢ⋅Pᵢ)/ΣᵢVᵢ
            vwap = np.sum(volumes * prices) / np.sum(volumes)

            # Compute hash-based price: P_hash(t)
            # Create hash from current market state
            market_state = f"{prices[-1]:.6f}{volumes[-1]:.6f}{time.time():.6f}"
            self.sha256_context.update(market_state.encode())
            hash_hex = self.sha256_context.hexdigest()[: self.hash_strength]

            # Convert hash to price-like value
            hash_value = int(hash_hex, 16) / (16**self.hash_strength)
            p_hash = prices[0] + (prices[-1] - prices[0]) * hash_value

            # Compute VWAP drift collapse
            vw_shift = vwap - p_hash

            return vw_shift
        except Exception as e:
            self.logger.error(f"Error computing VWAP drift collapse: {e}")
            return 0.0

    def compute_entropic_oscillator_pulse(
        self, volume_data: List[VolumeData], t: float
    ) -> float:
        """
        Compute entropic oscillator pulse: H_osc(t) = sin(2πt/τ + ψ_SHA256) ⋅ log(Vₜ)

        Args:
        volume_data: List of volume data points
        t: Current time or index

        Returns:
        Entropic oscillator pulse value
        """
        try:
            if not volume_data:
                return 0.0
            Vt = volume_data[-1].volume
            tau = self.tau_period
            # Use SHA256 hash as phase offset
            hash_input = f"{Vt:.6f}{t:.6f}".encode()
            hash_digest = hashlib.sha256(hash_input).hexdigest()
            phase_offset = int(hash_digest[:8], 16) / 0xFFFFFFFF * 2 * np.pi
            pulse = np.sin(2 * np.pi * t / tau + phase_offset) * np.log1p(Vt)
            return pulse
        except Exception as e:
            self.logger.error(f"Error computing entropic oscillator pulse: {e}")
            return 0.0

    def detect_smart_money_patterns(
        self, volume_data: List[VolumeData]
    ) -> Dict[str, Any]:
        """
        Detect smart money patterns in volume data.

        Args:
        volume_data: List of volume data points

        Returns:
        Dictionary containing smart money detection results
        """
        try:
            if len(volume_data) < self.period:
                return {"detected": False, "confidence": 0.0}

            # Extract recent data
            recent_data = volume_data[-self.period:]
            volumes = np.array([d.volume for d in recent_data])
            prices = np.array([d.price for d in recent_data])

            # Calculate volume-price correlation
            volume_price_corr = np.corrcoef(volumes, prices)[0, 1]

            # Detect unusual volume spikes
            volume_mean = np.mean(volumes)
            volume_std = np.std(volumes)
            volume_z_scores = np.abs((volumes - volume_mean) / volume_std)
            unusual_volumes = np.sum(volume_z_scores > 2.0)

            # Smart money detection logic
            smart_money_detected = (
                volume_price_corr > 0.7 and unusual_volumes > len(volumes) * 0.3
            )

            confidence = min(1.0, (volume_price_corr + unusual_volumes / len(volumes)) / 2)

            return {
                "detected": smart_money_detected,
                "confidence": confidence,
                "volume_price_correlation": volume_price_corr,
                "unusual_volumes": unusual_volumes,
            }

        except Exception as e:
            self.logger.error(f"Error detecting smart money patterns: {e}")
            return {"detected": False, "confidence": 0.0}

    def compute_hash_oscillator(self, volume_data: List[VolumeData]) -> HashOscillatorResult:
        """
        Compute the complete hash oscillator value.

        Args:
        volume_data: List of volume data points

        Returns:
        HashOscillatorResult with all calculations
        """
        try:
            if not self.initialized:
                raise RuntimeError("Oscillator not initialized")

            if len(volume_data) < 2:
                return HashOscillatorResult(
                    timestamp=time.time(),
                    vwap_value=0.0,
                    hash_value=0.0,
                    oscillator_value=0.0,
                    entropic_pulse=0.0,
                    signal_type=SignalType.NEUTRAL,
                    confidence=0.0,
                )

            # Compute VWAP drift collapse
            vwap_shift = self.compute_vwap_drift_collapse(volume_data)

            # Compute entropic oscillator pulse
            current_time = time.time()
            entropic_pulse = self.compute_entropic_oscillator_pulse(volume_data, current_time)

            # Combine signals
            oscillator_value = vwap_shift + entropic_pulse

            # Normalize if requested
            if self.normalize:
                oscillator_value = np.clip(oscillator_value, *self.oscillator_range)

            # Determine signal type
            signal_type = self._determine_signal_type(oscillator_value)

            # Calculate confidence
            confidence = self._calculate_confidence(volume_data, oscillator_value)

            # Store in history
            self.vwap_history.append(vwap_shift)
            self.entropic_history.append(entropic_pulse)
            self.oscillator_history.append(oscillator_value)

            # Keep history within limits
            if len(self.vwap_history) > self.period * 2:
                self.vwap_history.pop(0)
                self.entropic_history.pop(0)
                self.oscillator_history.pop(0)

            return HashOscillatorResult(
                timestamp=current_time,
                vwap_value=vwap_shift,
                hash_value=oscillator_value,
                oscillator_value=oscillator_value,
                entropic_pulse=entropic_pulse,
                signal_type=signal_type,
                confidence=confidence,
                metadata={
                    "period": self.period,
                    "hash_strength": self.hash_strength,
                    "normalized": self.normalize,
                },
            )

        except Exception as e:
            self.logger.error(f"Error computing hash oscillator: {e}")
            return HashOscillatorResult(
                timestamp=time.time(),
                vwap_value=0.0,
                hash_value=0.0,
                oscillator_value=0.0,
                entropic_pulse=0.0,
                signal_type=SignalType.NEUTRAL,
                confidence=0.0,
                error=str(e),
            )

    def _determine_signal_type(self, oscillator_value: float) -> SignalType:
        """Determine signal type based on oscillator value."""
        if oscillator_value > self.config.get("pump_threshold", 0.7):
            return SignalType.PUMP
        elif oscillator_value < self.config.get("dump_threshold", -0.7):
            return SignalType.DUMP
        elif oscillator_value > 0.3:
            return SignalType.BUY
        elif oscillator_value < -0.3:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    def _calculate_confidence(self, volume_data: List[VolumeData], oscillator_value: float) -> float:
        """Calculate confidence level for the oscillator signal."""
        try:
            if len(volume_data) < 2:
                return 0.0

            # Base confidence on oscillator magnitude
            base_confidence = min(1.0, abs(oscillator_value))

            # Adjust based on volume consistency
            volumes = np.array([d.volume for d in volume_data])
            volume_consistency = 1.0 - (np.std(volumes) / np.mean(volumes))
            volume_confidence = max(0.0, volume_consistency)

            # Combine confidences
            final_confidence = (base_confidence + volume_confidence) / 2

            return min(1.0, max(0.0, final_confidence))

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get oscillator status."""
        return {
            "initialized": self.initialized,
            "mode": self.mode.value,
            "period": self.period,
            "hash_strength": self.hash_strength,
            "history_size": len(self.oscillator_history),
            "config": self.config,
        }


# Factory function
def create_volume_weighted_hash_oscillator(config: Optional[Dict[str, Any]] = None) -> VolumeWeightedHashOscillator:
    """Create a Volume-Weighted Hash Oscillator instance."""
    return VolumeWeightedHashOscillator(config) 