from __future__ import annotations

from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""BTC Data Processor - Live Market Data Integration.

This module processes live BTC data, integrates volume, tick, and hash logic
for execution velocity calculations from live market volume and signal entropy.

Mathematical Foundation:
- Volume density triggers: ρ_market = 1 - unified_math.min(vol_density, 1.0)
- Tick entropy analysis: H_tick = -Σ(p_i * unified_math.log(p_i))
- Execution pressure derivation: P_exec = √(profit_residual / ρ_market)

Windows CLI compatible with comprehensive error handling.
"""


import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


@dataclass
class BTCDataMetrics:
    """BTC market data metrics for processing."""

    price: float
    volume: float
    timestamp: float
    tick_delta: float
    volume_density: float
    entropy_score: float
    execution_pressure: float
    hash_correlation: float


@dataclass
class VolumeAnalysis:
    """Volume analysis results."""

    density_score: float
    clustering_chi: float
    wall_detection: bool
    spoof_probability: float


class BTCDataProcessor:
    """Processes live BTC data with volume and entropy analysis."""

    def __init__(self) -> None:
        """Initialize BTC data processor."""
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.tick_history: List[float] = []
        self.entropy_window = 50
        self.max_history = 1000

        # Volume clustering detection parameters
        self.chi_threshold = 3.0
        self.spoof_price_threshold = 0.005  # 0.5% price movement threshold

    def process_btc_data(
        self,
        price: float,
        volume: float,
        order_book: Optional[Dict] = None,
        network_data: Optional[Dict] = None,
    ) -> BTCDataMetrics:
        """Process live BTC data and calculate all metrics.

        Parameters
        ----------
        price : float
            Current BTC price
        volume : float
            Current trading volume
        order_book : Dict, optional
            Order book data for wall detection
        network_data : Dict, optional
            BTC network data for hash correlation

        Returns
        -------
        BTCDataMetrics
            Processed BTC data metrics
        """
        try:
            current_time = time.time()

            # Update histories
            self._update_histories(price, volume, current_time)

            # Calculate tick delta
            tick_delta = self._calculate_tick_delta()

            # Calculate volume density
            volume_density = self._calculate_volume_density(volume)

            # Calculate tick entropy
            entropy_score = self._calculate_tick_entropy()

            # Calculate execution pressure
            execution_pressure = self._calculate_execution_pressure(
                volume_density, entropy_score
            )

            # Calculate hash correlation
            hash_correlation = self._calculate_hash_correlation(network_data)

            return BTCDataMetrics(
                price=price,
                volume=volume,
                timestamp=current_time,
                tick_delta=tick_delta,
                volume_density=volume_density,
                entropy_score=entropy_score,
                execution_pressure=execution_pressure,
                hash_correlation=hash_correlation,
            )

        except Exception as e:
            logger.error(f"Error processing BTC data: {e}")
            return self._create_safe_metrics(price, volume)

    def analyze_volume_clustering(
        self,
        current_volume: float,
        price_change: float,
    ) -> VolumeAnalysis:
        """Analyze volume for clustering and spoof detection.

        Mathematical Formula:
        χ_v = (volume_tick - median_n) / (stdev_n + ε)

        Parameters
        ----------
        current_volume : float
            Current tick volume
        price_change : float
            Price change percentage

        Returns
        -------
        VolumeAnalysis
            Volume analysis results
        """
        try:
            if len(self.volume_history) < 10:
                return VolumeAnalysis(0.5, 0.0, False, 0.0)

            # Calculate volume clustering chi score
            recent_volumes = np.array(self.volume_history[-20:])
            median_vol = np.median(recent_volumes)
            stdev_vol = unified_math.unified_math.std(recent_volumes)

            epsilon = 1e-6
            chi_v = (current_volume - median_vol) / (stdev_vol + epsilon)

            # Detect walls and spoofing
            wall_detected = chi_v > self.chi_threshold
            spoof_probability = 0.0

            if wall_detected and unified_math.abs(price_change) < self.spoof_price_threshold:
                spoof_probability = unified_math.min(chi_v / 10.0, 0.95)

            # Calculate density score
            density_score = unified_math.min(current_volume / (median_vol + epsilon), 2.0) / 2.0

            return VolumeAnalysis(
                density_score=density_score,
                clustering_chi=chi_v,
                wall_detection=wall_detected,
                spoof_probability=spoof_probability,
            )

        except Exception as e:
            logger.warning(f"Error in volume clustering analysis: {e}")
            return VolumeAnalysis(0.5, 0.0, False, 0.0)

    def calculate_execution_velocity(
        self,
        target_profit: float,
        current_pressure: float,
        market_density: float,
    ) -> float:
        """Calculate execution velocity from market conditions.

        Mathematical Formula:
        V_exec = √(target_profit / (market_density + ε)) * pressure_modifier

        Parameters
        ----------
        target_profit : float
            Target profit amount
        current_pressure : float
            Current execution pressure
        market_density : float
            Market density score [0, 1]

        Returns
        -------
        float
            Execution velocity score
        """
        try:
            epsilon = 1e-6
            density_factor = unified_math.max(market_density, 0.1)  # Prevent division by near-zero

            base_velocity = unified_math.unified_math.sqrt(target_profit / (density_factor + epsilon))
            pressure_modifier = 1.0 + (current_pressure - 0.5) * 0.5

            execution_velocity = base_velocity * pressure_modifier

            # Normalize to reasonable range [0.1, 3.0]
            return unified_math.max(0.1, unified_math.min(3.0, execution_velocity))

        except Exception as e:
            logger.warning(f"Error calculating execution velocity: {e}")
            return 1.0

    def detect_velocity_differential(
        self,
        actual_velocity: float,
        expected_velocity: float,
    ) -> Tuple[float, bool]:
        """Detect velocity differential and determine if delay needed.

        Mathematical Formula:
        V_diff = (v_actual - v_expected) / (v_expected + ε)

        Parameters
        ----------
        actual_velocity : float
            Observed velocity
        expected_velocity : float
            Expected velocity from STAM/volatility zone

        Returns
        -------
        Tuple[float, bool]
            (velocity_differential, should_delay_execution)
        """
        try:
            epsilon = 1e-6
            v_diff = (actual_velocity - expected_velocity) / (
                expected_velocity + epsilon
            )

            # Trigger delay if differential exceeds threshold
            should_delay = unified_math.abs(v_diff) > 0.3

            return v_diff, should_delay

        except Exception as e:
            logger.warning(f"Error calculating velocity differential: {e}")
            return 0.0, False

    def _update_histories(self, price: float, volume: float, timestamp: float) -> None:
        """Update price, volume, and tick histories."""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.tick_history.append(timestamp)

        # Trim histories to prevent memory growth
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-500:]
            self.volume_history = self.volume_history[-500:]
            self.tick_history = self.tick_history[-500:]

    def _calculate_tick_delta(self) -> float:
        """Calculate tick time delta."""
        if len(self.tick_history) < 2:
            return 0.0
        return self.tick_history[-1] - self.tick_history[-2]

    def _calculate_volume_density(self, current_volume: float) -> float:
        """Calculate volume density score.

        Formula: ρ_market = 1 - unified_math.min(vol_density, 1.0)
        """
        if len(self.volume_history) < 5:
            return 0.5

        try:
            recent_volumes = self.volume_history[-10:]
            avg_volume = unified_math.unified_math.mean(recent_volumes)

            if avg_volume == 0:
                return 0.5

            density_ratio = current_volume / avg_volume
            normalized_density = unified_math.min(density_ratio / 2.0, 1.0)

            # Market density is inverse of volume density
            market_density = 1.0 - normalized_density

            return unified_math.max(0.0, unified_math.min(1.0, market_density))

        except Exception as e:
            logger.warning(f"Error calculating volume density: {e}")
            return 0.5

    def _calculate_tick_entropy(self) -> float:
        """Calculate tick entropy score.

        Formula: H_tick = -Σ(p_i * unified_math.log(p_i))
        """
        if len(self.price_history) < self.entropy_window:
            return 0.5

        try:
            recent_prices = np.array(self.price_history[-self.entropy_window:])
            price_changes = np.diff(recent_prices)

            if len(price_changes) == 0:
                return 0.5

            # Discretize price changes into bins
            bins = 10
            hist, _ = np.histogram(price_changes, bins=bins, density=True)

            # Calculate entropy
            epsilon = 1e-10
            probabilities = hist + epsilon
            probabilities = probabilities / np.sum(probabilities)

            entropy = -np.sum(probabilities * unified_math.unified_math.log(probabilities))

            # Normalize to [0, 1] range
            max_entropy = unified_math.unified_math.log(bins)
            normalized_entropy = entropy / max_entropy

            return unified_math.max(0.0, unified_math.min(1.0, normalized_entropy))

        except Exception as e:
            logger.warning(f"Error calculating tick entropy: {e}")
            return 0.5

    def _calculate_execution_pressure(
        self,
        volume_density: float,
        entropy_score: float,
    ) -> float:
        """Calculate execution pressure.

        Formula: P_exec = √(profit_residual / ρ_market)
        """
        try:
            # Use entropy as proxy for profit residual
            profit_residual = entropy_score
            rho_market = unified_math.max(volume_density, 0.1)  # Prevent division by zero

            execution_pressure = unified_math.unified_math.sqrt(profit_residual / rho_market)

            # Normalize to [0, 1] range
            return unified_math.max(0.0, unified_math.min(1.0, execution_pressure))

        except Exception as e:
            logger.warning(f"Error calculating execution pressure: {e}")
            return 0.5

    def _calculate_hash_correlation(self, network_data: Optional[Dict]) -> float:
        """Calculate BTC hash correlation score."""
        if not network_data:
            return 0.5

        try:
            hash_rate = network_data.get('hash_rate', 0)
            difficulty = network_data.get('difficulty', 0)

            if hash_rate == 0 or difficulty == 0:
                return 0.5

            # Simple correlation based on hash rate and difficulty alignment
            normalized_hash = unified_math.min(hash_rate / 5e17, 1.0)  # Normalize to ~500 EH/s
            normalized_diff = unified_math.min(difficulty / 7e13, 1.0)  # Normalize to ~70T

            correlation = (normalized_hash + normalized_diff) / 2.0

            return unified_math.max(0.0, unified_math.min(1.0, correlation))

        except Exception as e:
            logger.warning(f"Error calculating hash correlation: {e}")
            return 0.5

    def _create_safe_metrics(self, price: float, volume: float) -> BTCDataMetrics:
        """Create safe fallback metrics."""
        return BTCDataMetrics(
            price=price,
            volume=volume,
            timestamp=time.time(),
            tick_delta=0.0,
            volume_density=0.5,
            entropy_score=0.5,
            execution_pressure=0.5,
            hash_correlation=0.5,
        )

    def get_processing_summary(self) -> Dict:
        """Get summary of processing state."""
        return {
            "price_history_size": len(self.price_history),
            "volume_history_size": len(self.volume_history),
            "tick_history_size": len(self.tick_history),
            "entropy_window": self.entropy_window,
            "chi_threshold": self.chi_threshold,
            "latest_price": self.price_history[-1] if self.price_history else 0.0,
            "latest_volume": self.volume_history[-1] if self.volume_history else 0.0,
        }


def main() -> None:
    """Demo function for testing BTC data processor."""
    safe_print("BTC Data Processor Demo")
    safe_print("=" * 30)

    processor = BTCDataProcessor()

    # Simulate BTC data processing
    test_data = [
        (50000, 1.5),
        (50050, 2.1),
        (49980, 1.8),
        (50100, 2.5),
        (50075, 1.9),
        (50200, 3.2),
        (50150, 2.0),
        (50080, 1.7),
    ]

    for price, volume in test_data:
        metrics = processor.process_btc_data(
            price=price,
            volume=volume,
            network_data={'hash_rate': 4.5e17, 'difficulty': 6.2e13},
        )

        safe_print(f"Price: ${price:,.0f}")
        safe_print(f"  Volume Density: {metrics.volume_density:.3f}")
        safe_print(f"  Entropy Score: {metrics.entropy_score:.3f}")
        safe_print(f"  Execution Pressure: {metrics.execution_pressure:.3f}")
        safe_print(f"  Hash Correlation: {metrics.hash_correlation:.3f}")

        # Test volume analysis
        price_change = 0.002 if price > 50000 else -0.002
        vol_analysis = processor.analyze_volume_clustering(volume, price_change)

        safe_print(f"  Volume Chi: {vol_analysis.clustering_chi:.2f}")
        safe_print(f"  Wall Detected: {vol_analysis.wall_detection}")
        safe_print(f"  Spoof Probability: {vol_analysis.spoof_probability:.3f}")
        print()

    # Test execution velocity
    velocity = processor.calculate_execution_velocity(
        target_profit=100.0, current_pressure=0.7, market_density=0.6
    )
    safe_print(f"Execution Velocity: {velocity:.3f}")

    # Test velocity differential
    v_diff, should_delay = processor.detect_velocity_differential(1.5, 1.0)
    safe_print(f"Velocity Differential: {v_diff:.3f}")
    safe_print(f"Should Delay: {should_delay}")

    # Processing summary
    summary = processor.get_processing_summary()
    safe_print(f"\nProcessing Summary: {summary}")


if __name__ == "__main__":
    main()
