from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Unified Signal Metrics - Schwabot Core Signal Processing

Provides unified signal metrics for BTC investment analysis, including
trading signal metrics and BTC-specific investment signals.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


@dataclass
class TradingSignalMetrics:
    """Unified trading signal metrics for core analysis."""
    
    # Core entropy and drift metrics
    triplet_entropy: float  # Triplet entropy measure
    theta_drift: float      # Theta drift coefficient
    coherence: float        # Signal coherence measure
    loop_volatility: float  # Loop volatility measure
    profit_decay: float     # Profit decay rate
    
    # Entry and execution metrics
    harmony: float          # Harmonic balance measure
    drift_penalty: float    # Drift penalty factor
    liquidity_score: float  # Liquidity assessment
    projected_profit: float # Projected profit potential
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    signal_quality: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)


@dataclass
class BTCInvestmentSignals:
    """BTC-specific investment signals."""
    
    # Core BTC metrics
    v_btc: float            # Volume BTC metric
    eta_btc: float          # Eta BTC efficiency
    xi_btc: float           # Xi BTC confidence
    price_pressure: float   # Price pressure indicator
    volume_profile: float   # Volume profile strength
    hash_correlation: float # Hash rate correlation
    network_strength: float # Network strength indicator
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    signal_quality: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)


class UnifiedSignalProcessor:
    """
    Processes and generates unified signal metrics for trading analysis.
    
    Responsibilities:
    - Calculate core trading signal metrics
    - Generate BTC-specific investment signals
    - Provide unified signal collection interface
    - Validate signal quality and consistency
    """
    
    def __init__(self):
        """Initialize the unified signal processor."""
        self.logger = logging.getLogger(__name__)
        self.signal_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
    def calculate_trading_signals(
        self,
        cursor_state: Optional[Dict] = None,
        fractal_state: Optional[Dict] = None,
        collapse_state: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
    ) -> TradingSignalMetrics:
        """Calculate unified trading signal metrics."""
        try:
            # Extract or calculate core metrics
            triplet_entropy = self._calculate_triplet_entropy(cursor_state, fractal_state)
            theta_drift = self._calculate_theta_drift(cursor_state, collapse_state)
            coherence = self._calculate_coherence(fractal_state, market_data)
            loop_volatility = self._calculate_loop_volatility(collapse_state, market_data)
            profit_decay = self._calculate_profit_decay(market_data)
            
            # Calculate entry and execution metrics
            harmony = self._calculate_harmony(fractal_state, cursor_state)
            drift_penalty = self._calculate_drift_penalty(theta_drift, loop_volatility)
            liquidity_score = self._calculate_liquidity_score(market_data)
            projected_profit = self._calculate_projected_profit(
                triplet_entropy, coherence, harmony
            )
            
            # Create unified metrics
            metrics = TradingSignalMetrics(
                triplet_entropy=triplet_entropy,
                theta_drift=theta_drift,
                coherence=coherence,
                loop_volatility=loop_volatility,
                profit_decay=profit_decay,
                harmony=harmony,
                drift_penalty=drift_penalty,
                liquidity_score=liquidity_score,
                projected_profit=projected_profit
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating trading signals: {e}")
            # Return safe fallback metrics
            return TradingSignalMetrics(
                triplet_entropy=0.5,
                theta_drift=0.5,
                coherence=0.5,
                loop_volatility=0.5,
                profit_decay=0.5,
                harmony=0.5,
                drift_penalty=0.5,
                liquidity_score=0.5,
                projected_profit=0.5
            )
    
    def calculate_btc_signals(
        self,
        btc_data: Optional[Dict] = None,
        volume_data: Optional[Dict] = None,
        network_data: Optional[Dict] = None,
    ) -> BTCInvestmentSignals:
        """Calculate BTC-specific investment signals."""
        try:
            # Extract or calculate BTC metrics
            v_btc = self._calculate_v_btc(volume_data, btc_data)
            eta_btc = self._calculate_eta_btc(btc_data, network_data)
            xi_btc = self._calculate_xi_btc(btc_data, volume_data)
            price_pressure = self._calculate_price_pressure(btc_data)
            volume_profile = self._calculate_volume_profile(volume_data)
            hash_correlation = self._calculate_hash_correlation(network_data, btc_data)
            network_strength = self._calculate_network_strength(network_data)
            
            # Create BTC signals
            signals = BTCInvestmentSignals(
                v_btc=v_btc,
                eta_btc=eta_btc,
                xi_btc=xi_btc,
                price_pressure=price_pressure,
                volume_profile=volume_profile,
                hash_correlation=hash_correlation,
                network_strength=network_strength
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error calculating BTC signals: {e}")
            # Return safe fallback signals
            return BTCInvestmentSignals(
                v_btc=0.5,
                eta_btc=0.5,
                xi_btc=0.5,
                price_pressure=0.5,
                volume_profile=0.5,
                hash_correlation=0.5,
                network_strength=0.5
            )
    
    def _calculate_triplet_entropy(self, cursor_state: Optional[Dict], fractal_state: Optional[Dict]) -> float:
        """Calculate triplet entropy measure."""
        try:
            # Extract relevant data from states
            cursor_entropy = cursor_state.get('entropy', 0.5) if cursor_state else 0.5
            fractal_entropy = fractal_state.get('fractal_entropy', 0.5) if fractal_state else 0.5
            
            # Calculate triplet entropy (weighted combination)
            triplet_entropy = (cursor_entropy * 0.6 + fractal_entropy * 0.4)
            return unified_math.max(0.0, unified_math.min(1.0, triplet_entropy))
        except Exception:
            return 0.5
    
    def _calculate_theta_drift(self, cursor_state: Optional[Dict], collapse_state: Optional[Dict]) -> float:
        """Calculate theta drift coefficient."""
        try:
            cursor_drift = cursor_state.get('drift', 0.5) if cursor_state else 0.5
            collapse_drift = collapse_state.get('collapse_drift', 0.5) if collapse_state else 0.5
            
            # Calculate theta drift (harmonic mean)
            theta_drift = 2.0 / (1.0 / cursor_drift + 1.0 / collapse_drift) if cursor_drift > 0 and collapse_drift > 0 else 0.5
            return unified_math.max(0.0, unified_math.min(1.0, theta_drift))
        except Exception:
            return 0.5
    
    def _calculate_coherence(self, fractal_state: Optional[Dict], market_data: Optional[Dict]) -> float:
        """Calculate signal coherence measure."""
        try:
            fractal_coherence = fractal_state.get('coherence', 0.5) if fractal_state else 0.5
            market_coherence = market_data.get('signal_coherence', 0.5) if market_data else 0.5
            
            # Calculate overall coherence
            coherence = (fractal_coherence * 0.7 + market_coherence * 0.3)
            return unified_math.max(0.0, unified_math.min(1.0, coherence))
        except Exception:
            return 0.5
    
    def _calculate_loop_volatility(self, collapse_state: Optional[Dict], market_data: Optional[Dict]) -> float:
        """Calculate loop volatility measure."""
        try:
            collapse_vol = collapse_state.get('volatility', 0.5) if collapse_state else 0.5
            market_vol = market_data.get('volatility', 0.5) if market_data else 0.5
            
            # Calculate loop volatility (geometric mean)
            loop_volatility = unified_math.unified_math.sqrt(collapse_vol * market_vol)
            return unified_math.max(0.0, unified_math.min(1.0, loop_volatility))
        except Exception:
            return 0.5
    
    def _calculate_profit_decay(self, market_data: Optional[Dict]) -> float:
        """Calculate profit decay rate."""
        try:
            if market_data:
                decay_rate = market_data.get('profit_decay', 0.5)
                return unified_math.max(0.0, unified_math.min(1.0, decay_rate))
            return 0.5
        except Exception:
            return 0.5
    
    def _calculate_harmony(self, fractal_state: Optional[Dict], cursor_state: Optional[Dict]) -> float:
        """Calculate harmonic balance measure."""
        try:
            fractal_harmony = fractal_state.get('harmony', 0.5) if fractal_state else 0.5
            cursor_harmony = cursor_state.get('harmony', 0.5) if cursor_state else 0.5
            
            # Calculate overall harmony
            harmony = (fractal_harmony * 0.6 + cursor_harmony * 0.4)
            return unified_math.max(0.0, unified_math.min(1.0, harmony))
        except Exception:
            return 0.5
    
    def _calculate_drift_penalty(self, theta_drift: float, loop_volatility: float) -> float:
        """Calculate drift penalty factor."""
        try:
            # Drift penalty increases with volatility and decreases with drift stability
            drift_penalty = loop_volatility * (1.0 - theta_drift)
            return unified_math.max(0.0, unified_math.min(1.0, drift_penalty))
        except Exception:
            return 0.5
    
    def _calculate_liquidity_score(self, market_data: Optional[Dict]) -> float:
        """Calculate liquidity assessment."""
        try:
            if market_data:
                liquidity = market_data.get('liquidity_score', 0.5)
                return unified_math.max(0.0, unified_math.min(1.0, liquidity))
            return 0.5
        except Exception:
            return 0.5
    
    def _calculate_projected_profit(self, triplet_entropy: float, coherence: float, harmony: float) -> float:
        """Calculate projected profit potential."""
        try:
            # Projected profit based on signal quality
            projected_profit = (triplet_entropy * 0.4 + coherence * 0.3 + harmony * 0.3)
            return unified_math.max(0.0, unified_math.min(1.0, projected_profit))
        except Exception:
            return 0.5
    
    def _calculate_v_btc(self, volume_data: Optional[Dict], btc_data: Optional[Dict]) -> float:
        """Calculate volume BTC metric."""
        try:
            volume_metric = volume_data.get('btc_volume_metric', 0.5) if volume_data else 0.5
            btc_volume = btc_data.get('volume_strength', 0.5) if btc_data else 0.5
            
            v_btc = (volume_metric * 0.6 + btc_volume * 0.4)
            return unified_math.max(0.0, unified_math.min(1.0, v_btc))
        except Exception:
            return 0.5
    
    def _calculate_eta_btc(self, btc_data: Optional[Dict], network_data: Optional[Dict]) -> float:
        """Calculate eta BTC efficiency."""
        try:
            btc_efficiency = btc_data.get('efficiency', 0.5) if btc_data else 0.5
            network_efficiency = network_data.get('network_efficiency', 0.5) if network_data else 0.5
            
            eta_btc = (btc_efficiency * 0.7 + network_efficiency * 0.3)
            return unified_math.max(0.0, unified_math.min(1.0, eta_btc))
        except Exception:
            return 0.5
    
    def _calculate_xi_btc(self, btc_data: Optional[Dict], volume_data: Optional[Dict]) -> float:
        """Calculate xi BTC confidence."""
        try:
            btc_confidence = btc_data.get('confidence', 0.5) if btc_data else 0.5
            volume_confidence = volume_data.get('volume_confidence', 0.5) if volume_data else 0.5
            
            xi_btc = (btc_confidence * 0.8 + volume_confidence * 0.2)
            return unified_math.max(0.0, unified_math.min(1.0, xi_btc))
        except Exception:
            return 0.5
    
    def _calculate_price_pressure(self, btc_data: Optional[Dict]) -> float:
        """Calculate price pressure indicator."""
        try:
            if btc_data:
                pressure = btc_data.get('price_pressure', 0.5)
                return unified_math.max(0.0, unified_math.min(1.0, pressure))
            return 0.5
        except Exception:
            return 0.5
    
    def _calculate_volume_profile(self, volume_data: Optional[Dict]) -> float:
        """Calculate volume profile strength."""
        try:
            if volume_data:
                profile = volume_data.get('volume_profile_strength', 0.5)
                return unified_math.max(0.0, unified_math.min(1.0, profile))
            return 0.5
        except Exception:
            return 0.5
    
    def _calculate_hash_correlation(self, network_data: Optional[Dict], btc_data: Optional[Dict]) -> float:
        """Calculate hash rate correlation."""
        try:
            network_hash = network_data.get('hash_correlation', 0.5) if network_data else 0.5
            btc_hash = btc_data.get('hash_rate_correlation', 0.5) if btc_data else 0.5
            
            hash_correlation = (network_hash * 0.6 + btc_hash * 0.4)
            return unified_math.max(0.0, unified_math.min(1.0, hash_correlation))
        except Exception:
            return 0.5
    
    def _calculate_network_strength(self, network_data: Optional[Dict]) -> float:
        """Calculate network strength indicator."""
        try:
            if network_data:
                strength = network_data.get('network_strength', 0.5)
                return unified_math.max(0.0, unified_math.min(1.0, strength))
            return 0.5
        except Exception:
            return 0.5


# Global processor instance
_signal_processor = UnifiedSignalProcessor()


def collect_unified_signals(
    cursor_state: Optional[Dict] = None,
    fractal_state: Optional[Dict] = None,
    collapse_state: Optional[Dict] = None,
    market_data: Optional[Dict] = None,
    btc_data: Optional[Dict] = None,
    volume_data: Optional[Dict] = None,
    network_data: Optional[Dict] = None,
) -> Tuple[TradingSignalMetrics, BTCInvestmentSignals]:
    """
    Collect unified signals from all available data sources.
    
    Returns:
        Tuple of (TradingSignalMetrics, BTCInvestmentSignals)
    """
    # Calculate trading signals
    trading_signals = _signal_processor.calculate_trading_signals(
        cursor_state, fractal_state, collapse_state, market_data
    )
    
    # Calculate BTC signals
    btc_signals = _signal_processor.calculate_btc_signals(
        btc_data, volume_data, network_data
    )
    
    return trading_signals, btc_signals


def get_signal_processor() -> UnifiedSignalProcessor:
    """Get the global signal processor instance."""
    return _signal_processor
