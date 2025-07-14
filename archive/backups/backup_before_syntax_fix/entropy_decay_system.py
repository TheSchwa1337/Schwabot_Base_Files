#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy Decay System for Schwabot
=================================
Implements time-based entropy degradation to prevent ghost overhang:
• Exponential decay of entropy signal weight over time
• Half-life management for signal memory
• Phantom memory cleanup
• Aging out of stale signals
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class DecayMode(Enum):
    """Entropy decay modes."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEPPED = "stepped"
    ADAPTIVE = "adaptive"

@dataclass
class EntropySignal:
    """Entropy signal with decay tracking."""
    signal_id: str
    initial_entropy: float
    activation_time: float
    current_weight: float
    decay_rate: float
    half_life: float
    asset: str
    strategy_code: int
    phantom_layer: bool = False
    last_accessed: float = field(default_factory=time.time)

@dataclass
class DecayResult:
    """Result of entropy decay calculation."""
    current_weight: float
    decay_factor: float
    age_seconds: float
    should_remove: bool
    confidence: float

class EntropyDecaySystem:
    """Time-based entropy decay system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize entropy decay system."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Signal storage
        self.active_signals: Dict[str, EntropySignal] = {}
        self.decay_history: List[Tuple[float, int]] = []  # (timestamp, signal_count)
        
        # Decay parameters
        self.lambda_decay = self.config['lambda_decay']
        self.phantom_boost = self.config['phantom_boost']
        self.cleanup_threshold = self.config['cleanup_threshold']
        
        self.logger.info("✅ Entropy Decay System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'lambda_decay': 0.1,  # Exponential decay rate
            'phantom_boost': 1.5,  # Phantom layer decay boost
            'cleanup_threshold': 0.01,  # Remove signals below this weight
            'max_signals': 1000,  # Maximum active signals
            'decay_mode': DecayMode.EXPONENTIAL,
            'half_life_seconds': 3600,  # 1 hour default half-life
            'adaptive_decay': True,
            'memory_efficient': True
        }
    
    def add_signal(self, signal_id: str, initial_entropy: float, asset: str = "BTC", 
                  strategy_code: int = 0, phantom_layer: bool = False) -> bool:
        """Add new entropy signal to decay system."""
        try:
            if len(self.active_signals) >= self.config['max_signals']:
                self._cleanup_old_signals()
            
            # Calculate decay parameters
            decay_rate = self._calculate_decay_rate(asset, phantom_layer)
            half_life = self._calculate_half_life(asset, phantom_layer)
            
            signal = EntropySignal(
                signal_id=signal_id,
                initial_entropy=initial_entropy,
                activation_time=time.time(),
                current_weight=initial_entropy,
                decay_rate=decay_rate,
                half_life=half_life,
                asset=asset,
                strategy_code=strategy_code,
                phantom_layer=phantom_layer
            )
            
            self.active_signals[signal_id] = signal
            self.logger.debug(f"Added signal {signal_id} with decay rate {decay_rate:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding signal {signal_id}: {e}")
            return False
    
    def calculate_decay(self, signal_id: str, current_time: Optional[float] = None) -> Optional[DecayResult]:
        """Calculate current decay for a signal."""
        try:
            if signal_id not in self.active_signals:
                return None
            
            signal = self.active_signals[signal_id]
            current_time = current_time or time.time()
            
            # Calculate age
            age_seconds = current_time - signal.activation_time
            
            # Calculate decay factor based on mode
            decay_factor = self._compute_decay_factor(signal, age_seconds)
            
            # Calculate current weight
            current_weight = signal.initial_entropy * decay_factor
            
            # Update signal
            signal.current_weight = current_weight
            signal.last_accessed = current_time
            
            # Determine if should be removed
            should_remove = current_weight < self.cleanup_threshold
            
            # Calculate confidence (inverse of age)
            confidence = max(0.0, 1.0 - (age_seconds / signal.half_life))
            
            return DecayResult(
                current_weight=current_weight,
                decay_factor=decay_factor,
                age_seconds=age_seconds,
                should_remove=should_remove,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating decay for {signal_id}: {e}")
            return None
    
    def get_active_signals(self, asset: Optional[str] = None, 
                          min_weight: float = 0.0) -> List[EntropySignal]:
        """Get active signals, optionally filtered by asset and minimum weight."""
        try:
            current_time = time.time()
            active_signals = []
            
            for signal in self.active_signals.values():
                # Calculate current decay
                decay_result = self.calculate_decay(signal.signal_id, current_time)
                if not decay_result:
                    continue
                
                # Apply filters
                if asset and signal.asset != asset:
                    continue
                if decay_result.current_weight < min_weight:
                    continue
                
                active_signals.append(signal)
            
            # Sort by current weight (highest first)
            active_signals.sort(key=lambda s: s.current_weight, reverse=True)
            return active_signals
            
        except Exception as e:
            self.logger.error(f"Error getting active signals: {e}")
            return []
    
    def cleanup_expired_signals(self) -> int:
        """Remove signals that have decayed below threshold."""
        try:
            initial_count = len(self.active_signals)
            current_time = time.time()
            
            expired_signals = []
            for signal_id, signal in self.active_signals.items():
                decay_result = self.calculate_decay(signal_id, current_time)
                if decay_result and decay_result.should_remove:
                    expired_signals.append(signal_id)
            
            # Remove expired signals
            for signal_id in expired_signals:
                del self.active_signals[signal_id]
            
            removed_count = initial_count - len(self.active_signals)
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} expired signals")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired signals: {e}")
            return 0
    
    def get_decay_statistics(self) -> Dict[str, Any]:
        """Get decay system statistics."""
        try:
            current_time = time.time()
            total_signals = len(self.active_signals)
            
            if total_signals == 0:
                return {
                    'total_signals': 0,
                    'average_weight': 0.0,
                    'average_age': 0.0,
                    'phantom_signals': 0,
                    'asset_distribution': {}
                }
            
            weights = []
            ages = []
            phantom_count = 0
            asset_distribution = {}
            
            for signal in self.active_signals.values():
                decay_result = self.calculate_decay(signal.signal_id, current_time)
                if decay_result:
                    weights.append(decay_result.current_weight)
                    ages.append(decay_result.age_seconds)
                    
                    if signal.phantom_layer:
                        phantom_count += 1
                    
                    asset_distribution[signal.asset] = asset_distribution.get(signal.asset, 0) + 1
            
            return {
                'total_signals': total_signals,
                'average_weight': np.mean(weights) if weights else 0.0,
                'average_age': np.mean(ages) if ages else 0.0,
                'phantom_signals': phantom_count,
                'asset_distribution': asset_distribution,
                'weight_std': np.std(weights) if weights else 0.0,
                'age_std': np.std(ages) if ages else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting decay statistics: {e}")
            return {}
    
    def _calculate_decay_rate(self, asset: str, phantom_layer: bool) -> float:
        """Calculate decay rate for asset and phantom status."""
        base_rate = self.lambda_decay
        
        # Asset-specific adjustments
        asset_multipliers = {
            "BTC": 1.0,
            "ETH": 1.1,
            "XRP": 0.9,
            "USDC": 0.8,
            "SOL": 1.2,
            "RANDOM": 1.0
        }
        
        asset_mult = asset_multipliers.get(asset, 1.0)
        
        # Phantom layer boost
        if phantom_layer:
            base_rate *= self.phantom_boost
        
        return base_rate * asset_mult
    
    def _calculate_half_life(self, asset: str, phantom_layer: bool) -> float:
        """Calculate half-life for asset and phantom status."""
        base_half_life = self.config['half_life_seconds']
        
        # Asset-specific half-life adjustments
        asset_multipliers = {
            "BTC": 1.0,
            "ETH": 0.9,
            "XRP": 1.2,
            "USDC": 1.5,
            "SOL": 0.8,
            "RANDOM": 1.0
        }
        
        asset_mult = asset_multipliers.get(asset, 1.0)
        
        # Phantom layer reduces half-life
        if phantom_layer:
            base_half_life *= 0.7
        
        return base_half_life * asset_mult
    
    def _compute_decay_factor(self, signal: EntropySignal, age_seconds: float) -> float:
        """Compute decay factor based on decay mode."""
        mode = self.config['decay_mode']
        
        if mode == DecayMode.EXPONENTIAL:
            # Exponential decay: exp(-λ * t)
            return np.exp(-signal.decay_rate * age_seconds)
        
        elif mode == DecayMode.LINEAR:
            # Linear decay: 1 - (t / half_life)
            return max(0.0, 1.0 - (age_seconds / signal.half_life))
        
        elif mode == DecayMode.STEPPED:
            # Stepped decay: discrete steps
            steps = int(age_seconds / (signal.half_life / 4))
            return max(0.0, 1.0 - (steps * 0.25))
        
        elif mode == DecayMode.ADAPTIVE:
            # Adaptive decay based on signal characteristics
            base_decay = np.exp(-signal.decay_rate * age_seconds)
            
            # Adjust based on asset volatility
            if signal.asset in ["SOL", "XRP"]:
                base_decay *= 0.9  # Faster decay for volatile assets
            
            return base_decay
        
        else:
            # Default to exponential
            return np.exp(-signal.decay_rate * age_seconds)
    
    def _cleanup_old_signals(self) -> None:
        """Clean up oldest signals when at capacity."""
        try:
            if len(self.active_signals) < self.config['max_signals']:
                return
            
            # Sort by last accessed time (oldest first)
            sorted_signals = sorted(
                self.active_signals.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest 10% of signals
            remove_count = max(1, len(sorted_signals) // 10)
            for i in range(remove_count):
                signal_id = sorted_signals[i][0]
                del self.active_signals[signal_id]
            
            self.logger.info(f"Cleaned up {remove_count} old signals due to capacity")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old signals: {e}")

# Factory function
def create_entropy_decay_system(config: Optional[Dict[str, Any]] = None) -> EntropyDecaySystem:
    """Create an entropy decay system instance."""
    return EntropyDecaySystem(config) 