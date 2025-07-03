"""Truth Scorer: Semantic Interpretation Validator.

Validates semantic interpretations against historical price behavior,
providing confidence scores for Lantern Eye predictions.
This is the reality-check layer that grounds semantic meaning in actual market data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class TruthScore:
    """Truth score for semantic interpretation validation."""

    interpretation_id: str
    truth_score: float
    confidence_level: float
    validation_method: str
    historical_accuracy: float
    market_correlation: float
    temporal_stability: float
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "interpretation_id": self.interpretation_id,
            "truth_score": self.truth_score,
            "confidence_level": self.confidence_level,
            "validation_method": self.validation_method,
            "historical_accuracy": self.historical_accuracy,
            "market_correlation": self.market_correlation,
            "temporal_stability": self.temporal_stability,
            "created_at": self.created_at,
        }


class TruthScorer:
    """Validate semantic interpretations against market reality.

    Provides truth scoring for semantic patterns by comparing
    interpretations against historical price movements and outcomes.
    """

    def __init__(self) -> None:
        """Initialize the truth scorer with validation systems."""
        # Historical validation data
        self.price_history = []
        self.interpretation_history = []
        self.validation_cache = {}

        # Performance tracking
        self.total_validations = 0
        self.average_accuracy = 0.0
        self.last_validation_time = 0.0

        # Validation parameters
        self.lookback_window = 50  # Number of historical points to consider
        self.correlation_threshold = 0.6  # Minimum correlation for validation
        self.confidence_decay = 0.95  # Confidence decay factor over time

    def add_price_data(self, price: float, volume: float = None, timestamp: float = None) -> None:
        """Add new price data for validation."""
        if timestamp is None:
            timestamp = time.time()

        price_point = {
            "price": price,
            "volume": volume or 0.0,
            "timestamp": timestamp,
        }

        self.price_history.append(price_point)

        # Keep history manageable
        if len(self.price_history) > self.lookback_window * 2:
            self.price_history = self.price_history[-self.lookback_window :]

    def validate_semantic_interpretation(
        self,
        interpretation: Dict[str, Any],
        current_price: float,
        market_context: Dict[str, Any] = None,
    ) -> TruthScore:
        """Validate semantic interpretation against market reality."""
        # Extract interpretation details
        interpretation_id = interpretation.get("semantic_hash", f"interp_{time.time()}")
        primary_meaning = interpretation.get("primary_meaning", "")
        category = interpretation.get("category", "unknown")
        confidence_score = interpretation.get("confidence_score", 0.5)

        # Perform validation
        historical_accuracy = self._calculate_historical_accuracy(category, primary_meaning)
        market_correlation = self._calculate_market_correlation(interpretation, current_price)
        temporal_stability = self._calculate_temporal_stability(interpretation)

        # Calculate overall truth score
        truth_score = self._calculate_truth_score(
            historical_accuracy,
            market_correlation,
            temporal_stability,
            confidence_score,
        )

        # Determine confidence level
        confidence_level = self._determine_confidence_level(
            truth_score, market_correlation, historical_accuracy
        )

        # Create truth score object
        truth_score_obj = TruthScore(
            interpretation_id=interpretation_id,
            truth_score=truth_score,
            confidence_level=confidence_level,
            validation_method="multi_factor_analysis",
            historical_accuracy=historical_accuracy,
            market_correlation=market_correlation,
            temporal_stability=temporal_stability,
        )

        # Store for future reference
        self._store_validation_result(interpretation, truth_score_obj)

        # Update performance metrics
        self._update_performance_metrics(truth_score_obj)

        return truth_score_obj

    def _calculate_historical_accuracy(self, category: str, primary_meaning: str) -> float:
        """Calculate accuracy based on historical similar interpretations."""
        if not self.interpretation_history:
            return 0.5  # Neutral score for no history

        # Find similar interpretations in history
        similar_interpretations = [
            interp for interp in self.interpretation_history if interp.get("category") == category
        ]

        if not similar_interpretations:
            return 0.5  # Neutral score for no similar interpretations

        # Calculate accuracy of similar interpretations
        correct_predictions = 0
        total_predictions = len(similar_interpretations)

        for interp in similar_interpretations:
            if self._was_prediction_correct(interp):
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.5
        return max(0.0, min(1.0, accuracy))

    def _was_prediction_correct(self, interpretation: Dict[str, Any]) -> bool:
        """Check if a historical interpretation was correct."""
        # This is a simplified implementation
        # In practice, you'd compare the prediction with actual market movement

        predicted_direction = self._extract_direction_from_meaning(
            interpretation.get("primary_meaning", "")
        )

        # Get price movement after interpretation
        interp_time = interpretation.get("timestamp", 0)
        price_movement = self._get_price_movement_after_time(interp_time)

        if predicted_direction == "bullish" and price_movement > 0.01:  # 1% increase
            return True
        elif predicted_direction == "bearish" and price_movement < -0.01:  # 1% decrease
            return True
        # 0.5% range
        elif predicted_direction == "neutral" and abs(price_movement) < 0.005:
            return True

        return False

    def _extract_direction_from_meaning(self, meaning: str) -> str:
        """Extract market direction from semantic meaning."""
        meaning_lower = meaning.lower()

        bullish_words = [
            "surge",
            "momentum",
            "bullish",
            "rising",
            "upward",
            "acceleration",
        ]
        bearish_words = [
            "decline",
            "bearish",
            "falling",
            "downward",
            "pressure",
            "breakdown",
        ]
        neutral_words = ["stable", "consolidation", "sideways", "balanced", "neutral"]

        bullish_count = sum(1 for word in bullish_words if word in meaning_lower)
        bearish_count = sum(1 for word in bearish_words if word in meaning_lower)
        neutral_count = sum(1 for word in neutral_words if word in meaning_lower)

        if bullish_count > bearish_count and bullish_count > neutral_count:
            return "bullish"
        elif bearish_count > neutral_count:
            return "bearish"
        else:
            return "neutral"

    def _get_price_movement_after_time(self, timestamp: float) -> float:
        """Get price movement after a specific timestamp."""
        if not self.price_history:
            return 0.0

        # Find price at timestamp
        before_prices = [p for p in self.price_history if p["timestamp"] <= timestamp]
        after_prices = [p for p in self.price_history if p["timestamp"] > timestamp]

        if not before_prices or not after_prices:
            return 0.0

        before_price = before_prices[-1]["price"]

        # Use price after reasonable delay (e.g., 1 hour later in simulation)
        target_time = timestamp + 3600  # 1 hour
        relevant_after_prices = [p for p in after_prices if p["timestamp"] <= target_time]

        if not relevant_after_prices:
            return 0.0

        after_price = relevant_after_prices[-1]["price"]

        # Calculate percentage movement
        if before_price == 0:
            return 0.0

        movement = (after_price - before_price) / before_price
        return movement

    def _calculate_market_correlation(
        self, interpretation: Dict[str, Any], current_price: float
    ) -> float:
        """Calculate correlation between interpretation and current market state."""
        if not self.price_history:
            return 0.5  # Neutral correlation for no data

        # Get recent price trend
        recent_prices = (
            self.price_history[-10:] if len(self.price_history) >= 10 else self.price_history
        )

        if len(recent_prices) < 2:
            return 0.5

        # Calculate price trend
        prices = [p["price"] for p in recent_prices]
        price_trend = self._calculate_trend(prices)

        # Extract predicted direction from interpretation
        predicted_direction = self._extract_direction_from_meaning(
            interpretation.get("primary_meaning", "")
        )

        # Calculate correlation
        if predicted_direction == "bullish" and price_trend > 0:
            correlation = min(1.0, abs(price_trend) * 10)  # Scale trend to 0-1
        elif predicted_direction == "bearish" and price_trend < 0:
            correlation = min(1.0, abs(price_trend) * 10)
        elif predicted_direction == "neutral" and abs(price_trend) < 0.01:
            # Inverse correlation for stability
            correlation = 1.0 - abs(price_trend) * 50
        else:
            correlation = max(0.0, 0.5 - abs(price_trend) * 5)  # Penalty for mismatch

        return max(0.0, min(1.0, correlation))

    def _calculate_trend(self, prices: List[float]) -> float:
        """Calculate trend from price list."""
        if len(prices) < 2:
            return 0.0

        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        # Normalize by average price
        avg_price = np.mean(prices)
        if avg_price == 0:
            return 0.0

        normalized_slope = slope / avg_price
        return float(normalized_slope)

    def _calculate_temporal_stability(self, interpretation: Dict[str, Any]) -> float:
        """Calculate temporal stability of the interpretation."""
        # Check if interpretation remains valid over time
        interpretation_time = interpretation.get("timestamp", time.time())
        current_time = time.time()

        time_elapsed = current_time - interpretation_time

        # Decay factor based on time elapsed
        decay_rate = 0.693 / 3600  # Half-life of 1 hour
        stability = np.exp(-decay_rate * time_elapsed)

        # Adjust based on interpretation confidence
        base_confidence = interpretation.get("confidence_score", 0.5)
        adjusted_stability = stability * (0.5 + base_confidence * 0.5)

        return max(0.0, min(1.0, adjusted_stability))

    def _calculate_truth_score(
        self,
        historical_accuracy: float,
        market_correlation: float,
        temporal_stability: float,
        interpretation_confidence: float,
    ) -> float:
        """Calculate overall truth score from component scores."""
        # Weighted combination of factors
        weights = {
            "historical": 0.3,
            "correlation": 0.4,
            "temporal": 0.2,
            "confidence": 0.1,
        }

        truth_score = (
            historical_accuracy * weights["historical"]
            + market_correlation * weights["correlation"]
            + temporal_stability * weights["temporal"]
            + interpretation_confidence * weights["confidence"]
        )

        return max(0.0, min(1.0, truth_score))

    def _determine_confidence_level(
        self,
        truth_score: float,
        market_correlation: float,
        historical_accuracy: float,
    ) -> float:
        """Determine confidence level for the truth score."""
        # Base confidence on truth score
        base_confidence = truth_score

        # Boost confidence if both correlation and accuracy are high
        if market_correlation > 0.7 and historical_accuracy > 0.7:
            base_confidence = min(1.0, base_confidence * 1.2)

        # Reduce confidence if either is very low
        if market_correlation < 0.3 or historical_accuracy < 0.3:
            base_confidence = base_confidence * 0.8

        return max(0.0, min(1.0, base_confidence))

    def _store_validation_result(
        self, interpretation: Dict[str, Any], truth_score: TruthScore
    ) -> None:
        """Store validation result for future reference."""
        # Add to interpretation history
        interpretation_with_validation = interpretation.copy()
        interpretation_with_validation["truth_score"] = truth_score.truth_score
        interpretation_with_validation["validation_timestamp"] = truth_score.created_at

        self.interpretation_history.append(interpretation_with_validation)

        # Keep history manageable
        if len(self.interpretation_history) > 100:
            self.interpretation_history = self.interpretation_history[-50:]

        # Cache validation result
        self.validation_cache[truth_score.interpretation_id] = truth_score

    def _update_performance_metrics(self, truth_score: TruthScore) -> None:
        """Update performance tracking metrics."""
        self.total_validations += 1
        self.last_validation_time = truth_score.created_at

        # Update average accuracy
        self.average_accuracy = (
            self.average_accuracy * (self.total_validations - 1) + truth_score.truth_score
        ) / self.total_validations

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        if not self.interpretation_history:
            return {"status": "No validation history"}

        # Calculate accuracy by category
        category_stats = {}
        for interp in self.interpretation_history:
            category = interp.get("category", "unknown")
            if category not in category_stats:
                category_stats[category] = {"count": 0, "total_score": 0.0}

            category_stats[category]["count"] += 1
            category_stats[category]["total_score"] += interp.get("truth_score", 0.0)

        # Calculate average scores by category
        for category, stats in category_stats.items():
            stats["average_score"] = stats["total_score"] / stats["count"]

        return {
            "total_validations": self.total_validations,
            "average_accuracy": self.average_accuracy,
            "price_history_length": len(self.price_history),
            "interpretation_history_length": len(self.interpretation_history),
            "last_validation_time": self.last_validation_time,
            "category_statistics": category_stats,
            "validation_cache_size": len(self.validation_cache),
        }

    def get_cached_validation(self, interpretation_id: str) -> TruthScore:
        """Get cached validation result."""
        return self.validation_cache.get(interpretation_id)

    def clear_old_validations(self, max_age_hours: float = 24.0) -> int:
        """Clear old validation results and return number cleared."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        # Clear old cache entries
        old_keys = [
            key for key, score in self.validation_cache.items() if score.created_at < cutoff_time
        ]

        for key in old_keys:
            del self.validation_cache[key]

        # Clear old interpretation history
        self.interpretation_history = [
            interp
            for interp in self.interpretation_history
            if interp.get("validation_timestamp", current_time) >= cutoff_time
        ]

        return len(old_keys)


def demo_truth_scorer() -> Dict[str, Any]:
    """Demonstrate the Truth Scorer system."""
    print("🎯 TRUTH SCORER DEMONSTRATION")
    print("=" * 50)

    # Initialize truth scorer
    scorer = TruthScorer()

    # Add some mock price data
    base_price = 100.0
    for i in range(20):
        price = base_price + (i * 0.5) + np.random.normal(0, 0.2)
        scorer.add_price_data(price, timestamp=time.time() - (20 - i) * 300)  # 5-minute intervals

    # Create mock interpretation
    mock_interpretation = {
        "primary_meaning": "Strong bullish momentum with surge patterns indicating upward acceleration",
        "category": "bullish_momentum",
        "confidence_score": 0.75,
        "semantic_hash": "test_interp_001",
        "timestamp": time.time() - 600,  # 10 minutes ago
    }

    # Validate interpretation
    truth_score = scorer.validate_semantic_interpretation(
        mock_interpretation,
        base_price + 5.0,  # Price increased
    )

    print("📊 VALIDATION RESULTS:")
    print(f"   Truth Score: {truth_score.truth_score:.3f}")
    print(f"   Confidence Level: {truth_score.confidence_level:.3f}")
    print(f"   Historical Accuracy: {truth_score.historical_accuracy:.3f}")
    print(f"   Market Correlation: {truth_score.market_correlation:.3f}")
    print(f"   Temporal Stability: {truth_score.temporal_stability:.3f}")

    # Get statistics
    stats = scorer.get_validation_statistics()
    print("\n📈 SCORER STATISTICS:")
    print(f"   Total Validations: {stats['total_validations']}")
    print(f"   Average Accuracy: {stats['average_accuracy']:.3f}")

    return stats


if __name__ == "__main__":
    demo_truth_scorer()
