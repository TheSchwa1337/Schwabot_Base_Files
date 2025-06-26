from __future__ import annotations

from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Unified Signal Metrics - BTC Investment Ratio Signal Consolidation.

This module consolidates all mathematical signals used for BTC investment ratio
analysis into a single coherent interface. Eliminates F841 flake8 errors by
using named tuples instead of temporary variables.

Mathematical Foundation:
- T: Triplet entropy from cursor patterns
- Δθ: Braid angle drift from geometric analysis
- ε: Fractal coherence from pattern matching
- σ_f: Loop sum volatility from collapse engine
- τ_p: Profit-time decay modifier
- 𝓗: Tick harmony alignment score
- 𝓓ₚ: Phase drift penalty
- 𝓛: Liquidity depth score
- P̂: Projected profit ratio

Windows CLI compatible with proper error handling.
"""


import logging
import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


# Named tuple for signal consolidation (eliminates F841 temp variables)
class TradingSignalMetrics(NamedTuple):
    """Consolidated trading signal metrics."""

    triplet_entropy: float  # T - Information rate from patterns
    theta_drift: float  # Δθ - Braid angle drift
    coherence: float  # ε - Fractal coherence score
    loop_volatility: float  # σ_f - Loop sum volatility
    profit_decay: float  # τ_p - Time-weighted profit modifier
    harmony: float  # 𝓗 - Tick harmony alignment
    drift_penalty: float  # 𝓓ₚ - Phase drift penalty
    liquidity_score: float  # 𝓛 - Normalized liquidity depth
    projected_profit: float  # P̂ - Expected profit ratio
    timestamp: float  # Collection timestamp


class BTCInvestmentSignals(NamedTuple):
    """BTC-specific investment ratio signals."""

    v_btc: float  # Volume-weighted price vector
    eta_btc: float  # Price momentum with volume
    xi_btc: float  # Combined BTC confidence
    price_pressure: float  # Market pressure indicator
    volume_profile: float  # Volume distribution score
    hash_correlation: float  # BTC hash rate correlation
    network_strength: float  # Network health indicator


@dataclass
class SignalCollector:
    """Collects and consolidates trading signals from various engines."""

    def __init__(self):
        """Initialize signal collector."""
        self.signal_history: List[TradingSignalMetrics] = []
        self.btc_signal_history: List[BTCInvestmentSignals] = []
        self.max_history_size = 1000

    def collect_core_signals(
        self,
        cursor_state: Optional[Dict] = None,
        fractal_state: Optional[Dict] = None,
        collapse_state: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
    ) -> TradingSignalMetrics:
        """Collect core trading signals from all engines.

        Parameters
        ----------
        cursor_state : Dict, optional
            State from cursor engine (contains Δθ, T)
        fractal_state : Dict, optional
            State from fractal engine (contains ε)
        collapse_state : Dict, optional
            State from collapse engine (contains σ_f)
        market_data : Dict, optional
            Current market data for liquidity and profit calculations

        Returns
        -------
        TradingSignalMetrics
            Consolidated signal metrics
        """
        try:
            # Extract cursor signals (T, Δθ)
            if cursor_state:
                triplet_entropy = cursor_state.get("triplet_entropy", 0.0)
                theta_drift = cursor_state.get("braid_angle_drift", 0.0)
            else:
                triplet_entropy = 0.0
                theta_drift = 0.0

            # Extract fractal signals (ε)
            if fractal_state:
                coherence = fractal_state.get("coherence_score", 0.0)
            else:
                coherence = 0.0

            # Extract collapse signals (σ_f, τ_p)
            if collapse_state:
                loop_volatility = collapse_state.get("loop_sum_volatility", 0.0)
                profit_decay = collapse_state.get("profit_time_decay", 0.0)
            else:
                loop_volatility = 0.0
                profit_decay = 0.0

            # Calculate harmony and drift from tick data
            harmony = self._calculate_tick_harmony(market_data)
            drift_penalty = self._calculate_phase_drift()

            # Extract market signals
            if market_data:
                liquidity_score = self._calculate_liquidity_score(market_data)
                projected_profit = self._calculate_projected_profit(market_data)
            else:
                liquidity_score = 0.0
                projected_profit = 0.0

            # Create consolidated signal metrics
            signals = TradingSignalMetrics(
                triplet_entropy=triplet_entropy,
                theta_drift=theta_drift,
                coherence=coherence,
                loop_volatility=loop_volatility,
                profit_decay=profit_decay,
                harmony=harmony,
                drift_penalty=drift_penalty,
                liquidity_score=liquidity_score,
                projected_profit=projected_profit,
                timestamp=time.time(),
            )

            # Store in history
            self.signal_history.append(signals)
            if len(self.signal_history) > self.max_history_size:
                self.signal_history = self.signal_history[-500:]

            return signals

        except Exception as e:
            logger.error(f"Error collecting core signals: {e}")
            return TradingSignalMetrics(
                triplet_entropy=0.0,
                theta_drift=0.0,
                coherence=0.0,
                loop_volatility=0.0,
                profit_decay=0.0,
                harmony=0.0,
                drift_penalty=1.0,
                liquidity_score=0.0,
                projected_profit=0.0,
                timestamp=time.time(),
            )

    def collect_btc_signals(
        self,
        btc_data: Optional[Dict] = None,
        volume_data: Optional[Dict] = None,
        network_data: Optional[Dict] = None,
    ) -> BTCInvestmentSignals:
        """Collect BTC-specific investment signals.

        Parameters
        ----------
        btc_data : Dict, optional
            BTC price and trading data
        volume_data : Dict, optional
            Volume profile data
        network_data : Dict, optional
            BTC network metrics (hash rate, difficulty, etc.)

        Returns
        -------
        BTCInvestmentSignals
            BTC investment ratio signals
        """
        try:
            # Calculate BTC vector metrics
            if btc_data:
                v_btc = self._calculate_btc_vector(btc_data)
                eta_btc = self._calculate_btc_eta(btc_data)
                xi_btc = self._calculate_btc_xi(v_btc, eta_btc)
                price_pressure = self._calculate_price_pressure(btc_data)
            else:
                v_btc = eta_btc = xi_btc = price_pressure = 0.0

            # Calculate volume profile
            if volume_data:
                volume_profile = self._calculate_volume_profile(volume_data)
            else:
                volume_profile = 0.0

            # Calculate network metrics
            if network_data:
                hash_correlation = self._calculate_hash_correlation(network_data)
                network_strength = self._calculate_network_strength(network_data)
            else:
                hash_correlation = network_strength = 0.0

            signals = BTCInvestmentSignals(
                v_btc=v_btc,
                eta_btc=eta_btc,
                xi_btc=xi_btc,
                price_pressure=price_pressure,
                volume_profile=volume_profile,
                hash_correlation=hash_correlation,
                network_strength=network_strength,
            )

            # Store in history
            self.btc_signal_history.append(signals)
            if len(self.btc_signal_history) > self.max_history_size:
                self.btc_signal_history = self.btc_signal_history[-500:]

            return signals

        except Exception as e:
            logger.error(f"Error collecting BTC signals: {e}")
            return BTCInvestmentSignals(
                v_btc=0.0,
                eta_btc=0.0,
                xi_btc=0.0,
                price_pressure=0.0,
                volume_profile=0.0,
                hash_correlation=0.0,
                network_strength=0.0,
            )

    def _calculate_tick_harmony(self, market_data: Optional[Dict]) -> float:
        """Calculate tick harmony alignment score."""
        if not market_data or "tick_deltas" not in market_data:
            return 0.0

        try:
            from core.tick_resonance_engine import compute_harmony_vector

            tick_deltas = np.array(market_data["tick_deltas"])
            target_phase = market_data.get("target_phase", 0.125)  # 8-bit default
            return compute_harmony_vector(tick_deltas, target_phase)
        except Exception as e:
            logger.warning(f"Error calculating tick harmony: {e}")
            return 0.0

    def _calculate_phase_drift(self) -> float:
        """Calculate phase drift penalty."""
        try:
            from core.drift_phase_monitor import compute_phase_drift

            # Use current time and assumed phase start
            current_time = time.time()
            phase_start = current_time - 60  # Assume 1-minute phase
            expected_cycle = 30.0  # 30-second cycle
            return compute_phase_drift(phase_start, current_time, expected_cycle)
        except Exception as e:
            logger.warning(f"Error calculating phase drift: {e}")
            return 0.0

    def _calculate_liquidity_score(self, market_data: Dict) -> float:
        """Calculate normalized liquidity depth score."""
        try:
            order_book = market_data.get("order_book", {})
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])

            if not bids or not asks:
                return 0.0

            # Calculate depth within 1% of mid price
            mid_price = (bids[0][0] + asks[0][0]) / 2
            depth_range = mid_price * 0.01

            bid_depth = sum(
                qty for price, qty in bids if price >= mid_price - depth_range
            )
            ask_depth = sum(
                qty for price, qty in asks if price <= mid_price + depth_range
            )

            total_depth = bid_depth + ask_depth
            # Normalize to [0, 1] range (assuming 100 BTC is excellent liquidity)
            return unified_math.min(total_depth / 100.0, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating liquidity score: {e}")
            return 0.0

    def _calculate_projected_profit(self, market_data: Dict) -> float:
        """Calculate projected profit ratio."""
        try:
            # Simple projected profit based on spread and volatility
            order_book = market_data.get("order_book", {})
            if not order_book.get("bids") or not order_book.get("asks"):
                return 0.0

            bid_price = order_book["bids"][0][0]
            ask_price = order_book["asks"][0][0]
            spread = (ask_price - bid_price) / bid_price

            # Historical volatility proxy
            recent_prices = market_data.get("recent_prices", [bid_price])
            if len(recent_prices) > 1:
                volatility = unified_math.unified_math.std(
                    recent_prices) / unified_math.unified_math.mean(recent_prices)
            else:
                volatility = 0.01

            # Project profit as function of spread and volatility
            projected_profit = (spread * 0.5) + (volatility * 0.1)
            return unified_math.min(projected_profit, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating projected profit: {e}")
            return 0.0

    def _calculate_btc_vector(self, btc_data: Dict) -> float:
        """Calculate BTC volume-weighted price vector."""
        try:
            from core.btc_vector_aggregator import btc_vector

            exit_prices = btc_data.get("exit_prices", [])
            entry_prices = btc_data.get("entry_prices", [])
            volume_weights = btc_data.get("volume_weights", [])

            if not exit_prices or len(exit_prices) != len(entry_prices):
                return 0.0

            return btc_vector(exit_prices, entry_prices, volume_weights)

        except Exception as e:
            logger.warning(f"Error calculating BTC vector: {e}")
            return 0.0

    def _calculate_btc_eta(self, btc_data: Dict) -> float:
        """Calculate BTC momentum with volume."""
        try:
            from core.btc_vector_aggregator import btc_eta

            price_delta = btc_data.get("price_delta", 0.0)
            time_delta = btc_data.get("time_delta", 1.0)
            volume_weights = btc_data.get("volume_weights", [1.0])

            return btc_eta(price_delta, time_delta, volume_weights)

        except Exception as e:
            logger.warning(f"Error calculating BTC eta: {e}")
            return 0.0

    def _calculate_btc_xi(self, v_btc: float, eta_btc: float) -> float:
        """Calculate combined BTC confidence."""
        try:
            from core.btc_vector_aggregator import btc_xi

            return btc_xi(v_btc, eta_btc)
        except Exception as e:
            logger.warning(f"Error calculating BTC xi: {e}")
            return 0.0

    def _calculate_price_pressure(self, btc_data: Dict) -> float:
        """Calculate market price pressure indicator."""
        try:
            # Use entry logic from phantom module
            from core.phantom.entry_logic import entry_score

            dp_norm = btc_data.get("normalized_price_change", 0.0)
            sigma_vol = btc_data.get("volatility_measure", 0.0)

            pressure = entry_score(dp_norm, sigma_vol)
            # Normalize to [0, 1] range
            return unified_math.max(0.0, unified_math.min(1.0, (pressure + 1.0) / 2.0))

        except Exception as e:
            logger.warning(f"Error calculating price pressure: {e}")
            return 0.0

    def _calculate_volume_profile(self, volume_data: Dict) -> float:
        """Calculate volume distribution score."""
        try:
            volume_levels = volume_data.get("volume_levels", [])
            if not volume_levels:
                return 0.0

            # Calculate volume concentration
            total_volume = sum(volume_levels)
            if total_volume == 0:
                return 0.0

            # Measure how concentrated volume is (higher = more concentrated)
            volume_array = np.array(volume_levels)
            volume_normalized = volume_array / total_volume
            concentration = np.sum(volume_normalized**2)  # Herfindahl index

            return concentration

        except Exception as e:
            logger.warning(f"Error calculating volume profile: {e}")
            return 0.0

    def _calculate_hash_correlation(self, network_data: Dict) -> float:
        """Calculate BTC hash rate correlation."""
        try:
            hash_rate = network_data.get("hash_rate", 0.0)
            price = network_data.get("price", 0.0)

            if hash_rate == 0 or price == 0:
                return 0.0

            # Simple correlation proxy (in real implementation, use historical data)
            # Higher hash rate generally correlates with higher price
            normalized_hash = hash_rate / 1e18  # Normalize to reasonable range
            normalized_price = price / 100000  # Normalize to reasonable range

            correlation = unified_math.min(normalized_hash * normalized_price, 1.0)
            return correlation

        except Exception as e:
            logger.warning(f"Error calculating hash correlation: {e}")
            return 0.0

    def _calculate_network_strength(self, network_data: Dict) -> float:
        """Calculate BTC network health indicator."""
        try:
            difficulty = network_data.get("difficulty", 0.0)
            hash_rate = network_data.get("hash_rate", 0.0)
            mempool_size = network_data.get("mempool_size", 0.0)

            # Combine metrics for network strength
            if difficulty == 0 or hash_rate == 0:
                return 0.0

            # Higher difficulty and hash rate = stronger network
            # Lower mempool congestion = better network
            strength_score = (
                unified_math.min(difficulty / 1e12, 1.0) * 0.4  # Difficulty component
                + unified_math.min(hash_rate / 1e18, 1.0) * 0.4  # Hash rate component
                + unified_math.max(0, 1.0 - mempool_size / 100000) * 0.2  # Mempool component
            )

            return strength_score

        except Exception as e:
            logger.warning(f"Error calculating network strength: {e}")
            return 0.0

    def get_latest_signals(self) -> Optional[TradingSignalMetrics]:
        """Get the most recent trading signal metrics."""
        return self.signal_history[-1] if self.signal_history else None

    def get_latest_btc_signals(self) -> Optional[BTCInvestmentSignals]:
        """Get the most recent BTC investment signals."""
        return self.btc_signal_history[-1] if self.btc_signal_history else None

    def get_signal_summary(self) -> Dict:
        """Get summary of recent signal performance."""
        if not self.signal_history:
            return {"error": "No signal history available"}

        recent_signals = self.signal_history[-10:]  # Last 10 signals

        return {
            "signal_count": len(recent_signals),
            "avg_coherence": unified_math.mean([s.coherence for s in recent_signals]),
            "avg_harmony": unified_math.mean([s.harmony for s in recent_signals]),
            "avg_liquidity": unified_math.mean([s.liquidity_score for s in recent_signals]),
            "avg_projected_profit": unified_math.mean(
                [s.projected_profit for s in recent_signals]
            ),
            "latest_timestamp": recent_signals[-1].timestamp,
        }


# Global signal collector instance
signal_collector = SignalCollector()


def collect_unified_signals(
    cursor_state: Optional[Dict] = None,
    fractal_state: Optional[Dict] = None,
    collapse_state: Optional[Dict] = None,
    market_data: Optional[Dict] = None,
    btc_data: Optional[Dict] = None,
    volume_data: Optional[Dict] = None,
    network_data: Optional[Dict] = None,
) -> tuple[TradingSignalMetrics, BTCInvestmentSignals]:
    """Collect all unified signals in one call.

    Returns
    -------
    tuple[TradingSignalMetrics, BTCInvestmentSignals]
        Core trading signals and BTC investment signals
    """
    core_signals = signal_collector.collect_core_signals(
        cursor_state, fractal_state, collapse_state, market_data
    )

    btc_signals = signal_collector.collect_btc_signals(
        btc_data, volume_data, network_data
    )

    return core_signals, btc_signals


def main() -> None:
    """Demo function for testing unified signal metrics."""
    safe_print("Unified Signal Metrics Demo")
    safe_print("=" * 40)

    # Mock data for testing
    mock_cursor_state = {
        "triplet_entropy": 0.75,
        "braid_angle_drift": 0.12,
    }

    mock_fractal_state = {
        "coherence_score": 0.88,
    }

    mock_collapse_state = {
        "loop_sum_volatility": 0.15,
        "profit_time_decay": 0.03,
    }

    mock_market_data = {
        "tick_deltas": [0.12, 0.13, 0.11, 0.125, 0.14],
        "target_phase": 0.125,
        "order_book": {
            "bids": [[50000, 1.5], [49950, 2.0]],
            "asks": [[50050, 1.2], [50100, 1.8]],
        },
        "recent_prices": [50000, 50025, 49975, 50050],
    }

    mock_btc_data = {
        "exit_prices": [50100, 50200, 50150],
        "entry_prices": [50000, 50050, 50075],
        "volume_weights": [1.0, 1.5, 0.8],
        "price_delta": 100.0,
        "time_delta": 60.0,
        "normalized_price_change": 0.002,
        "volatility_measure": 0.015,
    }

    # Collect signals
    core_signals, btc_signals = collect_unified_signals(
        mock_cursor_state,
        mock_fractal_state,
        mock_collapse_state,
        mock_market_data,
        mock_btc_data,
    )

    safe_print("Core Trading Signals:")
    safe_print(f"  Triplet Entropy: {core_signals.triplet_entropy:.3f}")
    safe_print(f"  Theta Drift: {core_signals.theta_drift:.3f}")
    safe_print(f"  Coherence: {core_signals.coherence:.3f}")
    safe_print(f"  Harmony: {core_signals.harmony:.3f}")
    safe_print(f"  Liquidity Score: {core_signals.liquidity_score:.3f}")
    safe_print(f"  Projected Profit: {core_signals.projected_profit:.3f}")

    safe_print(f"\nBTC Investment Signals:")
    safe_print(f"  V_BTC: {btc_signals.v_btc:.3f}")
    safe_print(f"  Eta_BTC: {btc_signals.eta_btc:.3f}")
    safe_print(f"  Xi_BTC: {btc_signals.xi_btc:.3f}")
    safe_print(f"  Price Pressure: {btc_signals.price_pressure:.3f}")

    # Get summary
    summary = signal_collector.get_signal_summary()
    safe_print(f"\nSignal Summary: {summary}")


if __name__ == "__main__":
    main()
