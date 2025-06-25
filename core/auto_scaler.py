# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Auto Scaler - Dynamic Position Size Calculator with DLT Integration
==================================================================

Provides automatic position scaling based on execution confidence and projected profit metrics.
Integrates with DLT waveform for mathematical position sizing optimization.

Mathematical Foundation:
scale_factor = base_scale * (1 + confidence_multiplier + profit_multiplier)

Where:
- confidence_multiplier = unified_math.max(0, (Ξ - threshold) * confidence_weight)
- profit_multiplier = projected_profit * profit_weight
- Result is clamped to [min_scale, max_scale] range

Based on Schwabot's mathematical framework and DLT waveform integration.
"""

import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from core.unified_math_system import unified_math

from .type_defs import Price, Amount, Confidence, ProfitRatio
from .mathlib_v4 import MathLibV4

logger = logging.getLogger(__name__)

# Default scaling parameters
DEFAULT_BASE_SCALE = 1.0
DEFAULT_MIN_SCALE = 0.1
DEFAULT_MAX_SCALE = 5.0
DEFAULT_CONFIDENCE_THRESHOLD = 1.15
DEFAULT_CONFIDENCE_WEIGHT = 2.0
DEFAULT_PROFIT_WEIGHT = 10.0

# Risk management parameters
MAX_POSITION_RISK = 0.02  # 2% of portfolio per position
MIN_POSITION_SIZE = 0.001  # Minimum position size


@dataclass
class ScalingResult:
    """Result of position scaling calculation."""
    scale_factor: float
    final_position: Amount
    confidence_multiplier: float
    profit_multiplier: float
    risk_percentage: float
    scaling_applied: bool
    risk_limited: bool
    timestamp: datetime = field(default_factory=datetime.now)


class AutoScaler:
    """
    Dynamic position size calculator with DLT waveform integration.

    Mathematical Foundation:
    - Uses DLT waveform confidence scores for scaling
    - Applies profit vector projections for position sizing
    - Integrates with MathLib v4 for mathematical operations
    - Provides risk-adjusted position scaling
    """

    def __init__(
        self,
        confidence_weight: float = DEFAULT_CONFIDENCE_WEIGHT,
        profit_weight: float = DEFAULT_PROFIT_WEIGHT,
        max_scale: float = DEFAULT_MAX_SCALE,
        adaptive_optimization: bool = True,
    ) -> None:
        """Initialize the auto scaler."""
        self.confidence_weight = confidence_weight
        self.profit_weight = profit_weight
        self.max_scale = max_scale
        self.adaptive_optimization = adaptive_optimization

        # Mathematical integration
        self.mathlib = MathLibV4()

        # Performance tracking
        self.scaling_history: List[ScalingResult] = []
        self.total_scalings = 0
        self.average_scale_factor = 1.0

        logger.info("Auto Scaler initialized with DLT integration")

    def scale_position(
        self,
        confidence: Confidence,
        projected_profit: ProfitRatio,
        base_scale: float = DEFAULT_BASE_SCALE,
        min_scale: float = DEFAULT_MIN_SCALE,
        max_scale: float = DEFAULT_MAX_SCALE,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> float:
        """
        Calculate position scale factor based on confidence and profit.

        Mathematical Process:
        1. Apply DLT confidence adjustments
        2. Calculate profit multiplier using DLT projections
        3. Combine multipliers with mathematical weighting
        4. Apply bounds and constraints
        """
        try:
            # Validate inputs
            if confidence < 0 or projected_profit < 0:
                logger.warning(
                    f"Invalid inputs: confidence={confidence}, profit={projected_profit}"
                )
                return base_scale

            # Apply DLT confidence adjustments
            dlt_adjusted_confidence = self.mathlib.apply_dlt_confidence_adjustment(confidence)

            # Calculate confidence multiplier
            confidence_excess = unified_math.max(0.0, dlt_adjusted_confidence - confidence_threshold)
            confidence_multiplier = confidence_excess * self.confidence_weight

            # Calculate profit multiplier using DLT projections
            dlt_profit_projection = self.mathlib.apply_dlt_profit_projection(projected_profit)
            profit_multiplier = dlt_profit_projection * self.profit_weight

            # Combine multipliers with mathematical weighting
            scale_factor = base_scale * (1.0 + confidence_multiplier + profit_multiplier)

            # Apply bounds
            scale_factor = unified_math.max(min_scale, unified_math.min(max_scale, scale_factor))

            return scale_factor

        except Exception as e:
            logger.error(f"Error calculating position scale: {e}")
            return base_scale

    def calculate_position_size(
        self,
        base_position: Amount,
        confidence: Confidence,
        projected_profit: ProfitRatio,
        account_balance: float,
        max_risk_per_trade: float = MAX_POSITION_RISK,
        **scaling_params: float,
    ) -> Tuple[Amount, ScalingResult]:
        """
        Calculate actual position size with risk management and DLT integration.

        Mathematical Process:
        1. Calculate DLT-adjusted scale factor
        2. Apply risk management constraints
        3. Integrate with mathematical bounds
        4. Return position size and detailed results
        """
        try:
            # Calculate scale factor with DLT integration
            scale_factor = self.scale_position(confidence, projected_profit, **scaling_params)

            # Calculate scaled position
            scaled_position = float(base_position) * scale_factor

            # Apply risk management constraints
            max_position = account_balance * max_risk_per_trade
            risk_limited_position = unified_math.min(scaled_position, max_position)

            # Apply minimum position constraint
            final_position = unified_math.max(MIN_POSITION_SIZE, risk_limited_position)

            # Calculate multipliers for result
            confidence_excess = unified_math.max(0.0, confidence - scaling_params.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD))
            confidence_multiplier = confidence_excess * self.confidence_weight
            profit_multiplier = projected_profit * self.profit_weight

            # Create scaling result
            result = ScalingResult(
                scale_factor=scale_factor,
                final_position=Amount(final_position),
                confidence_multiplier=confidence_multiplier,
                profit_multiplier=profit_multiplier,
                risk_percentage=(final_position / account_balance) * 100,
                scaling_applied=scale_factor != DEFAULT_BASE_SCALE,
                risk_limited=scaled_position > max_position,
            )

            # Update performance metrics
            self._update_performance_metrics(result)

            return Amount(final_position), result

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return base_position, ScalingResult(
                scale_factor=1.0,
                final_position=base_position,
                confidence_multiplier=0.0,
                profit_multiplier=0.0,
                risk_percentage=0.0,
                scaling_applied=False,
                risk_limited=False,
            )

    def _update_performance_metrics(self, result: ScalingResult) -> None:
        """Update performance metrics with scaling result."""
        self.scaling_history.append(result)
        self.total_scalings += 1

        # Update average scale factor
        self.average_scale_factor = (
            (self.average_scale_factor * (self.total_scalings - 1) + result.scale_factor)
            / self.total_scalings
        )

    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary of scaling operations."""
        if not self.scaling_history:
            return {"error": "No scaling history available"}

        recent_scalings = self.scaling_history[-10:]  # Last 10 scalings

        return {
            "total_scalings": self.total_scalings,
            "average_scale_factor": self.average_scale_factor,
            "recent_scalings": len(recent_scalings),
            "average_confidence_multiplier": unified_math.mean([s.confidence_multiplier for s in recent_scalings]),
            "average_profit_multiplier": unified_math.mean([s.profit_multiplier for s in recent_scalings]),
            "average_risk_percentage": unified_math.mean([s.risk_percentage for s in recent_scalings]),
            "scaling_applied_rate": unified_math.mean([s.scaling_applied for s in recent_scalings]),
            "risk_limited_rate": unified_math.mean([s.risk_limited for s in recent_scalings])
        }

    def reset_history(self) -> None:
        """Reset scaling history."""
        self.scaling_history.clear()
        self.total_scalings = 0
        self.average_scale_factor = 1.0
        logger.info("Auto Scaler history reset")


def validate_scaling_inputs(
    confidence: Confidence,
    projected_profit: ProfitRatio,
    base_position: Amount,
    account_balance: float,
) -> bool:
    """Validate scaling inputs."""
    try:
        # Check confidence bounds
        if not (0.0 <= confidence <= 10.0):
            logger.warning(f"Confidence out of bounds: {confidence}")
            return False

        # Check profit bounds
        if not (0.0 <= projected_profit <= 1.0):
            logger.warning(f"Projected profit out of bounds: {projected_profit}")
            return False

        # Check position bounds
        if base_position <= 0:
            logger.warning(f"Invalid base position: {base_position}")
            return False

        # Check account balance
        if account_balance <= 0:
            logger.warning(f"Invalid account balance: {account_balance}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating scaling inputs: {e}")
        return False


def main() -> None:
    """Main function for testing the auto scaler."""
    logging.basicConfig(level=logging.INFO)

    # Create auto scaler
    scaler = AutoScaler()

    # Test scaling calculations
    test_cases = [
        (0.8, 0.05, "Low confidence, low profit"),
        (1.5, 0.15, "High confidence, moderate profit"),
        (2.0, 0.25, "Very high confidence, high profit"),
    ]

    base_position = Amount(1000.0)
    account_balance = 50000.0

    safe_print("🧮 Testing Auto Scaler with DLT Integration")
    safe_print("=" * 50)

    for confidence, profit, description in test_cases:
        # Validate inputs
        if not validate_scaling_inputs(confidence, profit, base_position, account_balance):
            safe_print(f"❌ Invalid inputs for: {description}")
            continue

        # Calculate position size
        final_position, result = scaler.calculate_position_size(
            base_position, confidence, profit, account_balance
        )

        safe_print(f"📊 {description}:")
        safe_print(f"   Confidence: {confidence:.2f}, Profit: {profit:.3f}")
        safe_print(f"   Scale Factor: {result.scale_factor:.3f}")
        safe_print(f"   Final Position: ${final_position:.2f}")
        safe_print(f"   Risk Percentage: {result.risk_percentage:.2f}%")
        safe_print(f"   Scaling Applied: {result.scaling_applied}")
        print()

    # Get performance summary
    summary = scaler.get_performance_summary()
    safe_print("📈 Performance Summary:")
    safe_print(f"   Total Scalings: {summary['total_scalings']}")
    safe_print(f"   Average Scale Factor: {summary['average_scale_factor']:.3f}")
    safe_print(f"   Scaling Applied Rate: {summary['scaling_applied_rate']:.2%}")


if __name__ == "__main__":
    main()
