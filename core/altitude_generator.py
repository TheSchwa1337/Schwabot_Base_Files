# -*- coding: utf-8 -*-
"""
Altitude Generator - Core Altitude Generation Pipeline Component
==============================================================

This module provides altitude generation functionality for the Schwabot system.
It calculates altitude values based on market data, applies adjustment factors,
and determines altitude zones for trading decisions.

Core Functionality:
- Altitude generation from market metrics
- Adjustment factor calculations
- Altitude zone determination
- Confidence scoring
- Trend analysis
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


@dataclass
class AltitudeGenerationResult:
    """Result of altitude generation operation."""
    success: bool
    altitude_value: float
    generation_time: datetime
    confidence_score: float
    altitude_zone: str
    adjustment_factors: Dict[str, float]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class AltitudeMetrics:
    """Comprehensive altitude metrics."""
    base_altitude: float
    adjusted_altitude: float
    altitude_zone: str
    trend_direction: str
    volatility_factor: float
    liquidity_factor: float
    pressure_factor: float
    confidence_score: float
    generation_timestamp: datetime


class AltitudeGenerator:
    """Core altitude generation system for Schwabot."""

    def __init__(self):
        """Initialize the altitude generator."""
        self.altitude_history: List[float] = []
        self.generation_history: List[AltitudeGenerationResult] = []
        self.zone_thresholds = {
            "low": (0.0, 0.3),
            "medium": (0.3, 0.7),
            "high": (0.7, 1.0)
        }

        self.generation_count = 0
        logger.info("Altitude Generator initialized")

    def generate_altitude(self,
                          market_data: Dict[str,
                                            Any]) -> AltitudeGenerationResult:
        """Generate altitude based on market data."""
        try:
            # Extract key market metrics
            volume = market_data.get('volume', 0.0)
            price_change = market_data.get('price_change', 0.0)
            volatility = market_data.get('volatility', 0.0)
            liquidity = market_data.get('liquidity', 0.0)
            pressure = market_data.get('pressure', 0.5)

            # Calculate base altitude
            base_altitude = self._calculate_base_altitude(
                volume, price_change, volatility)

            # Apply adjustment factors
            adjusted_altitude = self._apply_altitude_adjustments(
                base_altitude, liquidity, pressure, volatility)

            # Determine altitude zone
            altitude_zone = self._determine_altitude_zone(adjusted_altitude)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(market_data)

            # Create adjustment factors
            adjustment_factors = {
                'liquidity_factor': self._calculate_liquidity_factor(liquidity),
                'pressure_factor': self._calculate_pressure_factor(pressure),
                'volatility_factor': self._calculate_volatility_factor(volatility),
                'volume_factor': self._calculate_volume_factor(volume)}

            result = AltitudeGenerationResult(
                success=True,
                altitude_value=adjusted_altitude,
                generation_time=datetime.now(),
                confidence_score=confidence_score,
                altitude_zone=altitude_zone,
                adjustment_factors=adjustment_factors,
                metadata={
                    'base_altitude': base_altitude,
                    'market_metrics': market_data,
                    'generation_count': self.generation_count
                }
            )

            # Update history
            self.altitude_history.append(adjusted_altitude)
            self.generation_history.append(result)
            self.generation_count += 1

            logger.info(
                f"Altitude generated: {
                    adjusted_altitude:.3f} ({altitude_zone})")
            return result

        except Exception as e:
            logger.error(f"Altitude generation error: {e}")
            return AltitudeGenerationResult(
                success=False,
                altitude_value=0.5,
                generation_time=datetime.now(),
                confidence_score=0.0,
                altitude_zone="unknown",
                adjustment_factors={},
                error_message=str(e)
            )

    def _calculate_base_altitude(
            self,
            volume: float,
            price_change: float,
            volatility: float) -> float:
        """Calculate base altitude from fundamental metrics."""
        try:
            # Volume component (higher volume = lower altitude)
            volume_component = 1.0 - unified_math.min(volume / 1000.0, 1.0)

            # Price change component (higher change = higher altitude)
            price_component = unified_math.min(
                unified_math.abs(price_change / 0.1), 1.0)

            # Volatility component (higher volatility = higher altitude)
            volatility_component = unified_math.min(volatility / 0.5, 1.0)

            # Combine components with weights
            base_altitude = (
                volume_component * 0.4 +
                price_component * 0.3 +
                volatility_component * 0.3
            )

            return unified_math.max(0.0, unified_math.min(1.0, base_altitude))

        except Exception as e:
            logger.error(f"Base altitude calculation error: {e}")
            return 0.5

    def _apply_altitude_adjustments(
            self,
            base_altitude: float,
            liquidity: float,
            pressure: float,
            volatility: float) -> float:
        """Apply adjustment factors to base altitude."""
        try:
            liquidity_factor = self._calculate_liquidity_factor(liquidity)
            pressure_factor = self._calculate_pressure_factor(pressure)
            volatility_factor = self._calculate_volatility_factor(volatility)

            adjusted_altitude = base_altitude * \
                (1 + liquidity_factor) * (1 + pressure_factor) * (1 + volatility_factor)

            return unified_math.max(
                0.0, unified_math.min(
                    1.0, adjusted_altitude))

        except Exception as e:
            logger.error(f"Altitude adjustment error: {e}")
            return base_altitude

    def _calculate_liquidity_factor(self, liquidity: float) -> float:
        """Calculate liquidity adjustment factor."""
        # Lower liquidity increases altitude adjustment
        return (1.0 - liquidity) * 0.1

    def _calculate_pressure_factor(self, pressure: float) -> float:
        """Calculate pressure adjustment factor."""
        # Higher pressure increases altitude adjustment
        return (pressure - 0.5) * 0.2

    def _calculate_volatility_factor(self, volatility: float) -> float:
        """Calculate volatility adjustment factor."""
        # Higher volatility increases altitude adjustment
        return volatility * 0.15

    def _calculate_volume_factor(self, volume: float) -> float:
        """Calculate volume adjustment factor."""
        # Higher volume decreases altitude adjustment
        return (1.0 - unified_math.min(volume / 1000.0, 1.0)) * 0.1

    def _determine_altitude_zone(self, altitude: float) -> str:
        """Determine altitude zone based on value."""
        for zone, (min_val, max_val) in self.zone_thresholds.items():
            if min_val <= altitude <= max_val:
                return zone
        return "unknown"

    def _calculate_confidence_score(
            self, market_data: Dict[str, Any]) -> float:
        """Calculate confidence score for altitude generation."""
        try:
            # Data quality factors
            data_completeness = sum(
                1 for key in [
                    'volume',
                    'price_change',
                    'volatility',
                    'liquidity'] if key in market_data and market_data[key] is not None) / 4.0

            # Market stability factors
            volatility = market_data.get('volatility', 0.0)
            stability_factor = 1.0 - unified_math.min(volatility, 1.0)

            # Volume reliability
            volume = market_data.get('volume', 0.0)
            volume_factor = unified_math.min(volume / 1000.0, 1.0)

            # Combine factors
            confidence = (
                data_completeness *
                0.4 +
                stability_factor *
                0.3 +
                volume_factor *
                0.3)
            return unified_math.max(0.0, unified_math.min(1.0, confidence))

        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.5

    def get_altitude_trend(self, window: int = 10) -> Tuple[str, float]:
        """Get altitude trend over a window of recent values."""
        if len(self.altitude_history) < window:
            return "insufficient_data", 0.0

        recent_altitudes = self.altitude_history[-window:]
        if len(recent_altitudes) < 2:
            return "insufficient_data", 0.0

        # Calculate trend
        trend_slope = (
            recent_altitudes[-1] - recent_altitudes[0]) / len(recent_altitudes)

        if trend_slope > 0.01:
            trend_direction = "increasing"
        elif trend_slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        return trend_direction, trend_slope

    def get_altitude_metrics(self) -> Optional[AltitudeMetrics]:
        """Get comprehensive altitude metrics."""
        if not self.altitude_history:
            return None

        current_altitude = self.altitude_history[-1]
        trend_direction, trend_slope = self.get_altitude_trend()

        # Get latest generation result
        latest_result = self.generation_history[-1] if self.generation_history else None

        return AltitudeMetrics(
            base_altitude=latest_result.metadata.get(
                'base_altitude',
                0.0) if latest_result else 0.0,
            adjusted_altitude=current_altitude,
            altitude_zone=self._determine_altitude_zone(current_altitude),
            trend_direction=trend_direction,
            volatility_factor=latest_result.adjustment_factors.get(
                'volatility_factor',
                0.0) if latest_result else 0.0,
            liquidity_factor=latest_result.adjustment_factors.get(
                'liquidity_factor',
                0.0) if latest_result else 0.0,
            pressure_factor=latest_result.adjustment_factors.get(
                'pressure_factor',
                0.0) if latest_result else 0.0,
            confidence_score=latest_result.confidence_score if latest_result else 0.0,
            generation_timestamp=latest_result.generation_time if latest_result else datetime.now())

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get altitude generation statistics."""
        total_generations = len(self.generation_history)
        successful_generations = sum(
            1 for result in self.generation_history if result.success)
        success_rate = successful_generations / \
            total_generations if total_generations > 0 else 0.0

        avg_altitude = sum(self.altitude_history) / \
            len(self.altitude_history) if self.altitude_history else 0.0

        return {
            "total_generations": total_generations,
            "successful_generations": successful_generations,
            "success_rate": success_rate,
            "average_altitude": avg_altitude,
            "current_altitude": self.altitude_history[-1] if self.altitude_history else 0.0,
            "altitude_history_length": len(self.altitude_history)
        }


def main():
    """Main function for testing altitude generation."""
    generator = AltitudeGenerator()

    # Test altitude generation
    test_market_data = {
        'volume': 500.0,
        'price_change': 0.05,
        'volatility': 0.3,
        'liquidity': 0.8,
        'pressure': 0.6
    }

    result = generator.generate_altitude(test_market_data)
    print(f"Altitude generation result: {result.success}")
    print(f"Altitude value: {result.altitude_value:.3f}")
    print(f"Altitude zone: {result.altitude_zone}")

    # Get statistics
    stats = generator.get_generation_statistics()
    print(f"Generation statistics: {stats}")


if __name__ == "__main__":
    main()
