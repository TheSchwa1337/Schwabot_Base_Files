# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import numpy as np
import math
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
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
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""Bitcoin Mining Analyzer - Mathematical Mining Analysis for Schwabot.

This module provides comprehensive Bitcoin mining analysis including hash rate
correlation, difficulty analysis, mining profitability calculations, and network
health metrics used in Schwabot's trading logic.

Mathematical Foundation:
- Hash rate correlation: ρ = Σ(h_i * p_i) / √(Σh_i² * Σp_i²)
- Difficulty adjustment: D_new = D_old * (T_target / T_actual)
- Mining profitability: P = (block_reward * hash_rate) / (difficulty * energy_cost)
- Network health: H = f(hash_rate, difficulty, mempool_size, block_time)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
# from core.unified_math_system import unified_math  # F811: duplicate import

logger = logging.getLogger(__name__)

@dataclass
class MiningMetrics:
    """Bitcoin mining metrics."""
    hash_rate: float  # EH/s
    difficulty: float  # Current difficulty
    block_time: float  # Average block time in seconds
    mempool_size: int  # Number of transactions in mempool
    block_reward: float  # Current block reward in BTC
    energy_cost: float  # Energy cost per TH/s in USD
    timestamp: float = field(default_factory=time.time)

@dataclass
class MiningAnalysis:
    """Mining analysis results."""
    profitability_score: float  # [0, 1]
    network_health: float  # [0, 1]
    difficulty_trend: float  # [-1, 1]
    hash_rate_correlation: float  # [-1, 1]
    mining_efficiency: float  # [0, 1]
    risk_assessment: float  # [0, 1]
    recommendations: List[str] = field(default_factory=list)

class BitcoinMiningAnalyzer:
    """Mathematical Bitcoin mining analysis for trading decisions."""

    def __init__(self):
        self.target_block_time = 600  # 10 minutes
        self.max_difficulty_adjustment = 4.0  # Maximum difficulty change factor
        self.mining_history: List[MiningMetrics] = []
        self.max_history_size = 1000
        logger.info("BitcoinMiningAnalyzer initialized")

    def analyze_mining_metrics(self, metrics: MiningMetrics,
                             price_data: Optional[Dict] = None) -> MiningAnalysis:
        """
        Analyze Bitcoin mining metrics for trading insights.

        Parameters:
        -----------
        metrics : MiningMetrics
            Current mining metrics
        price_data : Dict, optional
            Price data for correlation analysis

        Returns:
        --------
        MiningAnalysis
            Comprehensive mining analysis
        """
        try:
            # Store metrics in history
            self._update_history(metrics)

            # Calculate various analysis components
            profitability = self._calculate_profitability(metrics, price_data)
            network_health = self._assess_network_health(metrics)
            difficulty_trend = self._analyze_difficulty_trend()
            hash_correlation = self._calculate_hash_price_correlation(price_data)
            efficiency = self._calculate_mining_efficiency(metrics)
            risk = self._assess_mining_risk(metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                profitability, network_health, difficulty_trend, risk
            )

            return MiningAnalysis(
                profitability_score=profitability,
                network_health=network_health,
                difficulty_trend=difficulty_trend,
                hash_rate_correlation=hash_correlation,
                mining_efficiency=efficiency,
                risk_assessment=risk,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error analyzing mining metrics: {e}")
            return self._create_default_analysis()

    def _calculate_profitability(self, metrics: MiningMetrics,
                                price_data: Optional[Dict]) -> float:
        """
        Calculate mining profitability score.

        Mathematical Formula:
        P = (block_reward * hash_rate * price) / (difficulty * energy_cost)
        """
        try:
            if not price_data or 'price' not in price_data:
                return 0.5  # Neutral if no price data

            btc_price = price_data['price']

            # Basic profitability calculation
            daily_blocks = 86400 / metrics.block_time  # Blocks per day
            daily_revenue = daily_blocks * metrics.block_reward * btc_price

            # Energy cost (simplified)
            daily_energy_cost = metrics.hash_rate * 1e12 * metrics.energy_cost * 24

            if daily_energy_cost <= 0:
                return 0.5

            profitability_ratio = daily_revenue / daily_energy_cost

            # Normalize to [0, 1] range
            normalized_profitability = unified_math.min(1.0, profitability_ratio / 10.0)

            return unified_math.max(0.0, normalized_profitability)

        except Exception as e:
            logger.error(f"Error calculating profitability: {e}")
            return 0.5

    def _assess_network_health(self, metrics: MiningMetrics) -> float:
        """
        Assess overall network health.

        Factors:
        - Block time deviation from target
        - Mempool size
        - Hash rate stability
        """
        try:
            # Block time health (closer to 600s is better)
            block_time_health = 1.0 - unified_math.min(1.0, unified_math.abs(metrics.block_time - self.target_block_time) / self.target_block_time)

            # Mempool health (smaller is better, normalized)
            mempool_health = unified_math.max(0.0, 1.0 - (metrics.mempool_size / 100000))  # Normalize to 100k tx

            # Hash rate health (assume stable if recent data available)
            hash_rate_health = 0.8  # Default assumption

            if len(self.mining_history) >= 2:
                recent_hash_rates = [m.hash_rate for m in self.mining_history[-5:]]
                hash_rate_std = unified_math.unified_math.std(recent_hash_rates)
                hash_rate_mean = unified_math.unified_math.mean(recent_hash_rates)

                if hash_rate_mean > 0:
                    hash_rate_health = unified_math.max(0.0, 1.0 - (hash_rate_std / hash_rate_mean))

            # Weighted combination
            network_health = (
                block_time_health * 0.4 +
                mempool_health * 0.3 +
                hash_rate_health * 0.3
            )

            return unified_math.max(0.0, unified_math.min(1.0, network_health))

        except Exception as e:
            logger.error(f"Error assessing network health: {e}")
            return 0.5

    def _analyze_difficulty_trend(self) -> float:
        """Analyze difficulty trend over time."""
        try:
            if len(self.mining_history) < 3:
                return 0.0  # Neutral if insufficient data

            # Get recent difficulties
            recent_difficulties = [m.difficulty for m in self.mining_history[-10:]]

            if len(recent_difficulties) < 2:
                return 0.0

            # Calculate trend
            x = np.arange(len(recent_difficulties))
            y = np.array(recent_difficulties)

            # Linear regression
            slope = np.polyfit(x, y, 1)[0]

            # Normalize trend to [-1, 1]
            avg_difficulty = unified_math.unified_math.mean(recent_difficulties)
            if avg_difficulty > 0:
                normalized_trend = slope / avg_difficulty
                return max(-1.0, unified_math.min(1.0, normalized_trend * 100))  # Scale factor
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error analyzing difficulty trend: {e}")
            return 0.0

    def _calculate_hash_price_correlation(self, price_data: Optional[Dict]) -> float:
        """Calculate correlation between hash rate and price."""
        try:
            if not price_data or 'price_history' not in price_data:
                return 0.0

            price_history = price_data['price_history']

            if len(self.mining_history) < 5 or len(price_history) < 5:
                return 0.0

            # Align data lengths
            min_length = unified_math.min(len(self.mining_history), len(price_history))
            hash_rates = [m.hash_rate for m in self.mining_history[-min_length:]]
            prices = price_history[-min_length:]

            if len(hash_rates) < 2:
                return 0.0

            # Calculate correlation
            correlation = unified_math.unified_math.correlation(hash_rates, prices)[0, 1]

            return float(correlation) if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.error(f"Error calculating hash-price correlation: {e}")
            return 0.0

    def _calculate_mining_efficiency(self, metrics: MiningMetrics) -> float:
        """Calculate mining efficiency score."""
        try:
            # Efficiency factors
            block_time_efficiency = 1.0 - unified_math.min(1.0, unified_math.abs(metrics.block_time - self.target_block_time) / self.target_block_time)

            # Hash rate utilization (assume optimal if within reasonable range)
            hash_rate_efficiency = 0.9  # Default assumption

            # Difficulty efficiency (lower difficulty relative to hash rate is better)
            difficulty_efficiency = unified_math.max(0.0, 1.0 - (metrics.difficulty / (metrics.hash_rate * 1e12)))

            # Combined efficiency
            efficiency = (
                block_time_efficiency * 0.5 +
                hash_rate_efficiency * 0.3 +
                difficulty_efficiency * 0.2
            )

            return unified_math.max(0.0, unified_math.min(1.0, efficiency))

        except Exception as e:
            logger.error(f"Error calculating mining efficiency: {e}")
            return 0.5

    def _assess_mining_risk(self, metrics: MiningMetrics) -> float:
        """Assess mining-related risks."""
        try:
            risk_factors = []

            # Block time risk
            if metrics.block_time > self.target_block_time * 1.5:
                risk_factors.append(0.3)
            elif metrics.block_time < self.target_block_time * 0.5:
                risk_factors.append(0.2)
            else:
                risk_factors.append(0.0)

            # Mempool risk
            if metrics.mempool_size > 50000:
                risk_factors.append(0.2)
            else:
                risk_factors.append(0.0)

            # Hash rate volatility risk
            if len(self.mining_history) >= 3:
                recent_hash_rates = [m.hash_rate for m in self.mining_history[-3:]]
                hash_rate_volatility = unified_math.unified_math.std(recent_hash_rates) / unified_math.unified_math.mean(recent_hash_rates)
                if hash_rate_volatility > 0.1:  # 10% volatility threshold
                    risk_factors.append(0.3)
                else:
                    risk_factors.append(0.0)
            else:
                risk_factors.append(0.1)  # Default risk

            # Difficulty adjustment risk
            if len(self.mining_history) >= 2:
                difficulty_change = unified_math.abs(metrics.difficulty - self.mining_history[-2].difficulty) / self.mining_history[-2].difficulty
                if difficulty_change > 0.2:  # 20% change threshold
                    risk_factors.append(0.2)
                else:
                    risk_factors.append(0.0)
            else:
                risk_factors.append(0.1)

            # Calculate total risk
            total_risk = sum(risk_factors)
            return unified_math.min(1.0, total_risk)

        except Exception as e:
            logger.error(f"Error assessing mining risk: {e}")
            return 0.5

    def _generate_recommendations(self, profitability: float, network_health: float,
                                 difficulty_trend: float, risk: float) -> List[str]:
        """Generate trading recommendations based on mining analysis."""
        recommendations = []

        # Profitability-based recommendations
        if profitability > 0.7:
            recommendations.append("High mining profitability suggests bullish pressure")
        elif profitability < 0.3:
            recommendations.append("Low mining profitability may indicate bearish pressure")

        # Network health recommendations
        if network_health > 0.8:
            recommendations.append("Strong network health supports price stability")
        elif network_health < 0.4:
            recommendations.append("Poor network health may lead to price volatility")

        # Difficulty trend recommendations
        if difficulty_trend > 0.3:
            recommendations.append("Increasing difficulty suggests growing mining competition")
        elif difficulty_trend < -0.3:
            recommendations.append("Decreasing difficulty may indicate mining capitulation")

        # Risk-based recommendations
        if risk > 0.6:
            recommendations.append("High mining risk - consider defensive positioning")
        elif risk < 0.2:
            recommendations.append("Low mining risk - favorable for aggressive strategies")

        return recommendations

    def _update_history(self, metrics: MiningMetrics) -> None:
        """Update mining history."""
        self.mining_history.append(metrics)

        # Trim history to prevent memory growth
        if len(self.mining_history) > self.max_history_size:
            self.mining_history = self.mining_history[-self.max_history_size//2:]

    def _create_default_analysis(self) -> MiningAnalysis:
        """Create default analysis for error cases."""
        return MiningAnalysis(
            profitability_score=0.5,
            network_health=0.5,
            difficulty_trend=0.0,
            hash_rate_correlation=0.0,
            mining_efficiency=0.5,
            risk_assessment=0.5,
            recommendations=["Insufficient data for analysis"]
        )

    def get_mining_statistics(self) -> Dict[str, Any]:
        """Get mining statistics from history."""
        try:
            if not self.mining_history:
                return {"error": "No mining history available"}

            recent_metrics = self.mining_history[-50:]  # Last 50 entries

            return {
                "total_entries": len(self.mining_history),
                "avg_hash_rate": unified_math.mean([m.hash_rate for m in recent_metrics]),
                "avg_difficulty": unified_math.mean([m.difficulty for m in recent_metrics]),
                "avg_block_time": unified_math.mean([m.block_time for m in recent_metrics]),
                "hash_rate_volatility": unified_math.std([m.hash_rate for m in recent_metrics]),
                "difficulty_volatility": unified_math.std([m.difficulty for m in recent_metrics]),
                "latest_metrics": {
                    "hash_rate": recent_metrics[-1].hash_rate,
                    "difficulty": recent_metrics[-1].difficulty,
                    "block_time": recent_metrics[-1].block_time,
                    "mempool_size": recent_metrics[-1].mempool_size
                }
            }

        except Exception as e:
            logger.error(f"Error getting mining statistics: {e}")
            return {"error": str(e)}

def main() -> None:
    """Test function for BitcoinMiningAnalyzer."""
    safe_print("🧮 Testing Bitcoin Mining Analyzer...")

    analyzer = BitcoinMiningAnalyzer()

    # Test with sample mining metrics
    test_metrics = MiningMetrics(
        hash_rate=450.0,  # 450 EH/s
        difficulty=6.2e13,  # 62T
        block_time=580.0,  # 9.67 minutes
        mempool_size=25000,
        block_reward=6.25,
        energy_cost=0.05  # $0.05 per TH/s
    )

    # Mock price data
    price_data = {
        "price": 52000.0,
        "price_history": [50000, 51000, 52000, 51500, 53000]
    }

    # Analyze mining metrics
    analysis = analyzer.analyze_mining_metrics(test_metrics, price_data)

    safe_print(f"Profitability Score: {analysis.profitability_score:.3f}")
    safe_print(f"Network Health: {analysis.network_health:.3f}")
    safe_print(f"Difficulty Trend: {analysis.difficulty_trend:.3f}")
    safe_print(f"Hash Rate Correlation: {analysis.hash_rate_correlation:.3f}")
    safe_print(f"Mining Efficiency: {analysis.mining_efficiency:.3f}")
    safe_print(f"Risk Assessment: {analysis.risk_assessment:.3f}")

    safe_print("\nRecommendations:")
    for rec in analysis.recommendations:
        safe_print(f"  - {rec}")

    # Get statistics
    stats = analyzer.get_mining_statistics()
    safe_print(f"\nMining Statistics: {stats}")

    return 0

if __name__ == "__main__":
    exit(main())
