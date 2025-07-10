#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume-Weighted Hash Oscillator - Volume Entropy Collisions & VWAP+SHA Fusion

Implements Nexus mathematics for volume-weighted hash oscillators:
- VWAP Drift Collapse: VW_shift = Σᵢ(Vᵢ⋅Pᵢ)/ΣᵢVᵢ - P_hash(t)
- Entropic Oscillator Pulse: H_osc(t) = sin(2πt/τ + ψ_SHA256) ⋅ log(Vₜ)
- Smart Money logic for recursive SHA-field volume behavior
- Predicts sudden pumps/dumps triggered by liquidity wall movements
- Echoes the SmartMoney Zygot Scanner from April logs
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, ifft

logger = logging.getLogger(__name__)


class OscillatorMode(Enum):
    """Volume-weighted hash oscillator modes."""
    VWAP = "vwap"           # Volume-weighted average price
    HASH = "hash"           # Hash-based oscillation
    ENTROPIC = "entropic"   # Entropic oscillator pulse
    SMART_MONEY = "smart_money"  # Smart money detection
    LIQUIDITY = "liquidity" # Liquidity wall detection


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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Volume-Weighted Hash Oscillator."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.mode = OscillatorMode.VWAP
        self.initialized = False
        
        # Oscillator parameters
        self.period = self.config.get('period', 20)
        self.smoothing_period = self.config.get('smoothing_period', 10)
        self.hash_strength = self.config.get('hash_strength', 8)
        self.normalize = self.config.get('normalize', True)
        self.oscillator_range = self.config.get('oscillator_range', (-1.0, 1.0))
        
        # Entropic parameters
        self.tau_period = self.config.get('tau_period', 100)
        self.entropy_threshold = self.config.get('entropy_threshold', 0.1)
        self.liquidity_threshold = self.config.get('liquidity_threshold', 0.05)
        
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
            'period': 20,              # Look-back period
            'smoothing_period': 10,    # Smoothing period
            'hash_strength': 8,        # Hash strength (1-64)
            'normalize': True,         # Normalize output
            'oscillator_range': (-1.0, 1.0),  # Output range
            'tau_period': 100,         # Entropic period
            'entropy_threshold': 0.1,  # Entropy threshold
            'liquidity_threshold': 0.05,  # Liquidity threshold
            'pump_threshold': 0.7,     # Pump detection threshold
            'dump_threshold': -0.7,    # Dump detection threshold
        }
    
    def _initialize_oscillator(self):
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
            self.logger.info("[SUCCESS] Volume-Weighted Hash Oscillator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"[FAIL] Error initializing Volume-Weighted Hash Oscillator: {e}")
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
            hash_hex = self.sha256_context.hexdigest()[:self.hash_strength]
            
            # Convert hash to price-like value
            hash_value = int(hash_hex, 16) / (16 ** self.hash_strength)
            p_hash = prices[0] + (prices[-1] - prices[0]) * hash_value
            
            # Compute VWAP drift collapse
            vw_shift = vwap - p_hash
            
            return vw_shift
            
        except Exception as e:
            self.logger.error(f"Error computing VWAP drift collapse: {e}")
            return 0.0
    
    def compute_entropic_oscillator_pulse(self, volume_data: List[VolumeData], 
                                        t: float) -> float:
        """
        Compute entropic oscillator pulse: H_osc(t) = sin(2πt/τ + ψ_SHA256) ⋅ log(Vₜ)
        
        Args:
            volume_data: List of volume data points
            t: Current time
            
        Returns:
            Entropic oscillator pulse value
        """
        try:
            if not volume_data:
                return 0.0
            
            # Get current volume
            current_volume = volume_data[-1].volume
            
            # Compute SHA256 phase: ψ_SHA256
            volume_str = f"{current_volume:.6f}{t:.6f}"
            self.sha256_context.update(volume_str.encode())
            hash_hex = self.sha256_context.hexdigest()[:self.hash_strength]
            hash_value = int(hash_hex, 16) / (16 ** self.hash_strength)
            psi_sha256 = 2 * np.pi * hash_value
            
            # Compute entropic oscillator pulse
            # sin(2πt/τ + ψ_SHA256) ⋅ log(Vₜ)
            sine_component = np.sin(2 * np.pi * t / self.tau_period + psi_sha256)
            log_volume = np.log(max(current_volume, 1e-6))  # Avoid log(0)
            
            h_osc = sine_component * log_volume
            
            return h_osc
            
        except Exception as e:
            self.logger.error(f"Error computing entropic oscillator pulse: {e}")
            return 0.0
    
    def detect_smart_money_patterns(self, volume_data: List[VolumeData]) -> Dict[str, Any]:
        """
        Detect Smart Money patterns using volume entropy analysis.
        
        Args:
            volume_data: List of volume data points
            
        Returns:
            Smart Money detection results
        """
        try:
            if len(volume_data) < self.period:
                return {'smart_money_detected': False, 'confidence': 0.0}
            
            # Extract recent data
            recent_data = volume_data[-self.period:]
            volumes = np.array([d.volume for d in recent_data])
            prices = np.array([d.price for d in recent_data])
            
            # Compute volume entropy
            volume_entropy = stats.entropy(volumes + 1e-6)  # Add small constant
            
            # Compute price volatility
            price_volatility = np.std(prices)
            
            # Compute volume-price correlation
            volume_price_corr = np.corrcoef(volumes, prices)[0, 1]
            
            # Detect Smart Money patterns
            smart_money_indicators = []
            
            # High volume with low price movement (accumulation)
            if volume_entropy > self.entropy_threshold and price_volatility < 0.01:
                smart_money_indicators.append('accumulation')
            
            # Low volume with high price movement (distribution)
            if volume_entropy < self.entropy_threshold and price_volatility > 0.02:
                smart_money_indicators.append('distribution')
            
            # Volume spike with price reversal
            volume_spike = np.max(volumes) > 2 * np.mean(volumes)
            price_reversal = abs(prices[-1] - prices[0]) > 0.01
            if volume_spike and price_reversal:
                smart_money_indicators.append('manipulation')
            
            # Compute confidence based on indicators
            confidence = len(smart_money_indicators) / 3.0
            
            return {
                'smart_money_detected': len(smart_money_indicators) > 0,
                'confidence': confidence,
                'indicators': smart_money_indicators,
                'volume_entropy': volume_entropy,
                'price_volatility': price_volatility,
                'volume_price_correlation': volume_price_corr
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting Smart Money patterns: {e}")
            return {'smart_money_detected': False, 'confidence': 0.0}
    
    def detect_liquidity_walls(self, volume_data: List[VolumeData]) -> Dict[str, Any]:
        """
        Detect liquidity walls that trigger sudden pumps/dumps.
        
        Args:
            volume_data: List of volume data points
            
        Returns:
            Liquidity wall detection results
        """
        try:
            if len(volume_data) < self.period:
                return {'liquidity_wall_detected': False, 'direction': 'neutral'}
            
            # Extract recent data
            recent_data = volume_data[-self.period:]
            volumes = np.array([d.volume for d in recent_data])
            prices = np.array([d.price for d in recent_data])
            
            # Compute volume clusters
            volume_mean = np.mean(volumes)
            volume_std = np.std(volumes)
            
            # Detect volume clusters (potential liquidity walls)
            high_volume_clusters = volumes > (volume_mean + 2 * volume_std)
            low_volume_clusters = volumes < (volume_mean - 2 * volume_std)
            
            # Compute price momentum
            price_momentum = (prices[-1] - prices[0]) / prices[0]
            
            # Detect liquidity wall direction
            if np.any(high_volume_clusters) and price_momentum > self.liquidity_threshold:
                direction = 'pump'
                wall_strength = np.sum(high_volume_clusters) / len(high_volume_clusters)
            elif np.any(high_volume_clusters) and price_momentum < -self.liquidity_threshold:
                direction = 'dump'
                wall_strength = np.sum(high_volume_clusters) / len(high_volume_clusters)
            else:
                direction = 'neutral'
                wall_strength = 0.0
            
            return {
                'liquidity_wall_detected': wall_strength > 0.1,
                'direction': direction,
                'wall_strength': wall_strength,
                'price_momentum': price_momentum,
                'high_volume_clusters': np.sum(high_volume_clusters),
                'low_volume_clusters': np.sum(low_volume_clusters)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity walls: {e}")
            return {'liquidity_wall_detected': False, 'direction': 'neutral'}
    
    def compute_hash_oscillator(self, volume_data: List[VolumeData]) -> HashOscillatorResult:
        """
        Compute the complete hash oscillator signal.
        
        Args:
            volume_data: List of volume data points
            
        Returns:
            Hash oscillator result
        """
        try:
            if len(volume_data) < 2:
                return HashOscillatorResult(
                    timestamp=time.time(),
                    vwap_value=0.0,
                    hash_value=0.0,
                    oscillator_value=0.0,
                    entropic_pulse=0.0,
                    signal_type=SignalType.NEUTRAL,
                    confidence=0.0
                )
            
            # Compute VWAP drift collapse
            vwap_shift = self.compute_vwap_drift_collapse(volume_data)
            
            # Compute hash value
            current_data = volume_data[-1]
            market_state = f"{current_data.price:.6f}{current_data.volume:.6f}{current_data.timestamp:.6f}"
            self.sha256_context.update(market_state.encode())
            hash_hex = self.sha256_context.hexdigest()[:self.hash_strength]
            hash_value = int(hash_hex, 16) / (16 ** self.hash_strength)
            
            # Compute entropic oscillator pulse
            current_time = time.time()
            entropic_pulse = self.compute_entropic_oscillator_pulse(volume_data, current_time)
            
            # Combine signals for oscillator value
            oscillator_value = vwap_shift + hash_value + entropic_pulse
            
            # Apply smoothing
            if len(self.oscillator_history) >= self.smoothing_period:
                oscillator_value = np.mean(self.oscillator_history[-self.smoothing_period:] + [oscillator_value])
            
            # Normalize if requested
            if self.normalize:
                min_val, max_val = self.oscillator_range
                oscillator_value = np.clip(oscillator_value, min_val, max_val)
            
            # Determine signal type
            pump_threshold = self.config.get('pump_threshold', 0.7)
            dump_threshold = self.config.get('dump_threshold', -0.7)
            
            if oscillator_value > pump_threshold:
                signal_type = SignalType.PUMP
            elif oscillator_value < dump_threshold:
                signal_type = SignalType.DUMP
            elif oscillator_value > 0.1:
                signal_type = SignalType.BUY
            elif oscillator_value < -0.1:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Compute confidence based on signal strength
            confidence = min(1.0, abs(oscillator_value))
            
            # Store history
            self.vwap_history.append(vwap_shift)
            self.hash_history.append(hash_value)
            self.oscillator_history.append(oscillator_value)
            self.entropic_history.append(entropic_pulse)
            
            # Keep history manageable
            max_history = 1000
            if len(self.vwap_history) > max_history:
                self.vwap_history = self.vwap_history[-max_history:]
                self.hash_history = self.hash_history[-max_history:]
                self.oscillator_history = self.oscillator_history[-max_history:]
                self.entropic_history = self.entropic_history[-max_history:]
            
            return HashOscillatorResult(
                timestamp=current_time,
                vwap_value=vwap_shift,
                hash_value=hash_value,
                oscillator_value=oscillator_value,
                entropic_pulse=entropic_pulse,
                signal_type=signal_type,
                confidence=confidence
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
                confidence=0.0
            )
    
    def add_volume_data(self, volume_data: VolumeData):
        """Add new volume data to the oscillator."""
        self.volume_history.append(volume_data)
        
        # Keep only recent data
        max_data = self.period * 10
        if len(self.volume_history) > max_data:
            self.volume_history = self.volume_history[-max_data:]
    
    def get_oscillator_summary(self) -> Dict[str, Any]:
        """Get comprehensive oscillator summary."""
        if not self.oscillator_history:
            return {'status': 'no_data'}
        
        # Compute oscillator statistics
        oscillator_values = np.array(self.oscillator_history)
        vwap_values = np.array(self.vwap_history)
        hash_values = np.array(self.hash_history)
        entropic_values = np.array(self.entropic_history)
        
        return {
            'data_points': len(self.oscillator_history),
            'mean_oscillator': np.mean(oscillator_values),
            'std_oscillator': np.std(oscillator_values),
            'mean_vwap': np.mean(vwap_values),
            'mean_hash': np.mean(hash_values),
            'mean_entropic': np.mean(entropic_values),
            'oscillator_mode': self.mode.value,
            'initialized': self.initialized,
            'current_signal': self.oscillator_history[-1] if self.oscillator_history else 0.0
        }
    
    def set_mode(self, mode: OscillatorMode):
        """Set the oscillator mode."""
        self.mode = mode
        self.logger.info(f"Volume-Weighted Hash Oscillator mode set to: {mode.value}")


# Factory function
def create_volume_weighted_hash_oscillator(config: Optional[Dict[str, Any]] = None) -> VolumeWeightedHashOscillator:
    """Create a Volume-Weighted Hash Oscillator instance."""
    return VolumeWeightedHashOscillator(config)
