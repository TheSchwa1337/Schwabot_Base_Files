from __future__ import annotations

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
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
#!/usr/bin/env python3
"""ZPE Hybrid Mode Selector - Dynamic Mode Selection System.

This module provides intelligent mode selection between ZPE (recursive velocity)
and reactive tasking based on market conditions, volatility, timeframes, and
other factors. Implements the "both/and" approach for maximum flexibility.
"""


import logging
# from core.unified_math_system import unified_math  # F811: duplicate import
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

# Import ZPE Mathematical Framework
try:
    from core.zpe_core import ZPECore
    ZPE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ZPE modules not available: {e}")
    ZPE_MODULES_AVAILABLE = False

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, safe_format_error, log_safe
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Available trading modes."""
    ZPE_RECURSIVE = "zpe_recursive"      # Rotational velocity for bull runs
    REACTIVE_TASKING = "reactive_tasking"  # Proven methods for instability
    HYBRID_BLEND = "hybrid_blend"        # Mixed approach for mixed conditions
    EMERGENCY_FALLBACK = "emergency_fallback"  # Last resort


class MarketCondition(Enum):
    """Market condition classifications."""
    BULL_RUN = "bull_run"              # Strong uptrend, use ZPE
    BEAR_MARKET = "bear_market"        # Downtrend, use reactive
    SIDEWAYS = "sideways"              # Range-bound, use hybrid
    HIGH_VOLATILITY = "high_volatility"  # Unstable, use reactive
    LOW_VOLATILITY = "low_volatility"    # Stable, use ZPE
    CRISIS = "crisis"                  # Emergency, use fallback


@dataclass
class ModeSelectionCriteria:
    """Criteria for mode selection."""
    market_condition: MarketCondition
    volatility_score: float  # 0.0 to 1.0
    trend_strength: float    # -1.0 to 1.0
    timeframe: str          # "hourly", "daily", "weekly"
    profit_performance: float  # Recent profit performance
    zpe_availability: bool = True
    reactive_availability: bool = True
    emergency_triggered: bool = False


@dataclass
class ModeSelectionResult:
    """Result of mode selection."""
    selected_mode: TradingMode
    confidence_score: float
    reasoning: List[str]
    market_condition: MarketCondition
    mode_weights: Dict[TradingMode, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioAsset:
    """Portfolio asset for retroactive tasking."""
    symbol: str
    current_value: float
    allocation_percentage: float
    last_trade_time: datetime
    profit_performance: float
    volatility: float
    can_retroactive_task: bool = True


class ZPEHybridModeSelector:
    """
    Hybrid Mode Selector for dynamic ZPE/Reactive mode selection.

    Implements intelligent mode selection based on:
    - Market conditions (bull/bear/sideways)
    - Volatility levels
    - Timeframe considerations
    - Profit performance history
    - Portfolio asset characteristics
    - 488 and 42-bit phase logic
    """

    def __init__(self):
        """Initialize the hybrid mode selector."""
        self.zpe_core = ZPECore() if ZPE_MODULES_AVAILABLE else None

        # Mode selection thresholds
        self.zpe_threshold = 0.7      # High confidence needed for ZPE
        self.reactive_threshold = 0.6  # Lower threshold for reactive
        self.hybrid_threshold = 0.4    # Medium threshold for hybrid

        # Market condition thresholds
        self.bull_run_threshold = 0.6
        self.bear_market_threshold = -0.4
        self.high_volatility_threshold = 0.7
        self.low_volatility_threshold = 0.3

        # Timeframe weights
        self.timeframe_weights = {
            "hourly": {"zpe": 0.3, "reactive": 0.7},    # Reactive preferred for speed
            "daily": {"zpe": 0.6, "reactive": 0.4},     # Balanced approach
            "weekly": {"zpe": 0.8, "reactive": 0.2}     # ZPE preferred for analysis
        }

        # 488 and 42-bit phase logic
        self.phase_488_active = False
        self.phase_42_active = False
        self.phase_switch_counter = 0

        # Portfolio tracking
        self.portfolio_assets: Dict[str, PortfolioAsset] = {}
        self.retroactive_task_history: List[Dict] = []

        # Performance tracking
        self.mode_performance_history: Dict[TradingMode, List[float]] = {
            mode: [] for mode in TradingMode
        }

        safe_safe_print("🔄 ZPE Hybrid Mode Selector initialized")

    def select_mode(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Optional[Dict[str, Any]] = None,
        timeframe: str = "daily"
    ) -> ModeSelectionResult:
        """
        Select the optimal trading mode based on current conditions.

        Args:
            market_data: Current market data
            portfolio_data: Current portfolio data
            timeframe: Trading timeframe

        Returns:
            ModeSelectionResult with selected mode and reasoning
        """
        try:
            # Analyze market conditions
            market_condition = self._analyze_market_condition(market_data)

            # Calculate mode selection criteria
            criteria = self._calculate_selection_criteria(market_data, market_condition, timeframe)

            # Apply 488 and 42-bit phase logic
            self._update_phase_logic(market_data)

            # Select mode based on criteria
            selected_mode, confidence, reasoning = self._select_mode_by_criteria(criteria)

            # Calculate mode weights for hybrid scenarios
            mode_weights = self._calculate_mode_weights(criteria, selected_mode)

            # Update portfolio for retroactive tasking
            if portfolio_data:
                self._update_portfolio_assets(portfolio_data)

            # Create result
            result = ModeSelectionResult(
                selected_mode=selected_mode,
                confidence_score=confidence,
                reasoning=reasoning,
                market_condition=market_condition,
                mode_weights=mode_weights,
                metadata={
                    'phase_488': self.phase_488_active,
                    'phase_42': self.phase_42_active,
                    'timeframe': timeframe,
                    'volatility': criteria.volatility_score,
                    'trend_strength': criteria.trend_strength
                }
            )

            safe_safe_print(f"🎯 Mode Selected: {selected_mode.value} (confidence: {confidence:.3f})")
            safe_safe_print(f"   Market Condition: {market_condition.value}")
            safe_safe_print(f"   Reasoning: {', '.join(reasoning[:2])}")

            return result

        except Exception as e:
            safe_safe_print(f"❌ Mode selection failed: {safe_format_error(e, 'mode_selection')}")
            # Emergency fallback
            return ModeSelectionResult(
                selected_mode=TradingMode.EMERGENCY_FALLBACK,
                confidence_score=1.0,
                reasoning=["Emergency fallback due to selection error"],
                market_condition=MarketCondition.CRISIS
            )

    def _analyze_market_condition(self, market_data: Dict[str, Any]) -> MarketCondition:
        """Analyze current market condition."""
        try:
            trend_strength = market_data.get('trend_strength', 0.0)
            volatility = market_data.get('volatility', 0.5)
            price_change = market_data.get('price_change_24h', 0.0)

            # Check for crisis conditions
            if volatility > 0.9 or unified_math.abs(price_change) > 0.2:
                return MarketCondition.CRISIS

            # Check for bull run
            if trend_strength > self.bull_run_threshold and price_change > 0.05:
                return MarketCondition.BULL_RUN

            # Check for bear market
            if trend_strength < self.bear_market_threshold and price_change < -0.05:
                return MarketCondition.BEAR_MARKET

            # Check for high volatility
            if volatility > self.high_volatility_threshold:
                return MarketCondition.HIGH_VOLATILITY

            # Check for low volatility
            if volatility < self.low_volatility_threshold:
                return MarketCondition.LOW_VOLATILITY

            # Default to sideways
            return MarketCondition.SIDEWAYS

        except Exception as e:
            safe_safe_print(f"⚠️ Market condition analysis failed: {safe_format_error(e, 'market_analysis')}")
            return MarketCondition.SIDEWAYS

    def _calculate_selection_criteria(
        self,
        market_data: Dict[str, Any],
        market_condition: MarketCondition,
        timeframe: str
    ) -> ModeSelectionCriteria:
        """Calculate mode selection criteria."""
        try:
            volatility_score = market_data.get('volatility', 0.5)
            trend_strength = market_data.get('trend_strength', 0.0)
            profit_performance = market_data.get('profit_performance', 0.0)

            # Check ZPE availability
            zpe_available = ZPE_MODULES_AVAILABLE and self.zpe_core is not None

            # Check reactive availability (always available as fallback)
            reactive_available = True

            # Check for emergency conditions
            emergency_triggered = market_condition == MarketCondition.CRISIS

            return ModeSelectionCriteria(
                market_condition=market_condition,
                volatility_score=volatility_score,
                trend_strength=trend_strength,
                timeframe=timeframe,
                profit_performance=profit_performance,
                zpe_availability=zpe_available,
                reactive_availability=reactive_available,
                emergency_triggered=emergency_triggered
            )

        except Exception as e:
            safe_safe_print(f"⚠️ Criteria calculation failed: {safe_format_error(e, 'criteria_calculation')}")
            return ModeSelectionCriteria(
                market_condition=MarketCondition.SIDEWAYS,
                volatility_score=0.5,
                trend_strength=0.0,
                timeframe=timeframe,
                profit_performance=0.0
            )

    def _update_phase_logic(self, market_data: Dict[str, Any]) -> None:
        """Update 488 and 42-bit phase logic."""
        try:
            # Simple phase switching logic based on market conditions
            current_time = time.time()

            # Phase 488: High-frequency, high-volatility conditions
            if market_data.get('volatility', 0.0) > 0.7:
                self.phase_488_active = True
                self.phase_42_active = False
            # Phase 42: Low-frequency, stable conditions
            elif market_data.get('volatility', 0.0) < 0.3:
                self.phase_488_active = False
                self.phase_42_active = True
            else:
                # Mixed phase
                self.phase_488_active = False
                self.phase_42_active = False

            # Update phase switch counter
            if self.phase_488_active or self.phase_42_active:
                self.phase_switch_counter += 1

        except Exception as e:
            safe_safe_print(f"⚠️ Phase logic update failed: {safe_format_error(e, 'phase_logic')}")

    def _select_mode_by_criteria(
        self,
        criteria: ModeSelectionCriteria
    ) -> Tuple[TradingMode, float, List[str]]:
        """Select mode based on criteria."""
        reasoning = []
        mode_scores = {}

        try:
            # Emergency fallback check
            if criteria.emergency_triggered:
                reasoning.append("Emergency conditions detected")
                return TradingMode.EMERGENCY_FALLBACK, 1.0, reasoning

            # Calculate base scores for each mode
            mode_scores[TradingMode.ZPE_RECURSIVE] = self._calculate_zpe_score(criteria)
            mode_scores[TradingMode.REACTIVE_TASKING] = self._calculate_reactive_score(criteria)
            mode_scores[TradingMode.HYBRID_BLEND] = self._calculate_hybrid_score(criteria)

            # Apply availability constraints
            if not criteria.zpe_availability:
                mode_scores[TradingMode.ZPE_RECURSIVE] = 0.0
                mode_scores[TradingMode.HYBRID_BLEND] *= 0.5
                reasoning.append("ZPE modules unavailable")

            if not criteria.reactive_availability:
                mode_scores[TradingMode.REACTIVE_TASKING] = 0.0
                mode_scores[TradingMode.HYBRID_BLEND] *= 0.5
                reasoning.append("Reactive modules unavailable")

            # Apply phase logic adjustments
            if self.phase_488_active:
                mode_scores[TradingMode.REACTIVE_TASKING] *= 1.2  # Boost reactive in 488
                reasoning.append("Phase 488 active - boosting reactive")

            if self.phase_42_active:
                mode_scores[TradingMode.ZPE_RECURSIVE] *= 1.2  # Boost ZPE in 42
                reasoning.append("Phase 42 active - boosting ZPE")

            # Select best mode
            best_mode = unified_math.max(mode_scores, key=mode_scores.get)
            best_score = mode_scores[best_mode]

            # Apply confidence thresholds
            if best_mode == TradingMode.ZPE_RECURSIVE and best_score < self.zpe_threshold:
                reasoning.append(f"ZPE score {best_score:.3f} below threshold {self.zpe_threshold}")
                best_mode = TradingMode.HYBRID_BLEND
                best_score = mode_scores[TradingMode.HYBRID_BLEND]

            if best_mode == TradingMode.REACTIVE_TASKING and best_score < self.reactive_threshold:
                reasoning.append(f"Reactive score {best_score:.3f} below threshold {self.reactive_threshold}")
                best_mode = TradingMode.HYBRID_BLEND
                best_score = mode_scores[TradingMode.HYBRID_BLEND]

            # Add reasoning
            reasoning.append(f"Selected {best_mode.value} with score {best_score:.3f}")

            return best_mode, best_score, reasoning

        except Exception as e:
            safe_safe_print(f"⚠️ Mode selection failed: {safe_format_error(e, 'mode_selection_logic')}")
            return TradingMode.EMERGENCY_FALLBACK, 1.0, ["Emergency fallback due to selection error"]

    def _calculate_zpe_score(self, criteria: ModeSelectionCriteria) -> float:
        """Calculate ZPE mode score."""
        score = 0.0

        # Market condition scoring
        if criteria.market_condition == MarketCondition.BULL_RUN:
            score += 0.4
        elif criteria.market_condition == MarketCondition.LOW_VOLATILITY:
            score += 0.3
        elif criteria.market_condition == MarketCondition.SIDEWAYS:
            score += 0.2

        # Trend strength scoring
        if criteria.trend_strength > 0.5:
            score += 0.2
        elif criteria.trend_strength > 0.0:
            score += 0.1

        # Volatility scoring (ZPE prefers lower volatility)
        if criteria.volatility_score < 0.3:
            score += 0.2
        elif criteria.volatility_score < 0.5:
            score += 0.1

        # Timeframe scoring
        timeframe_weights = self.timeframe_weights.get(criteria.timeframe, {"zpe": 0.5, "reactive": 0.5})
        score += timeframe_weights["zpe"] * 0.2

        # Profit performance scoring
        if criteria.profit_performance > 0.1:
            score += 0.1

        return unified_math.min(1.0, score)

    def _calculate_reactive_score(self, criteria: ModeSelectionCriteria) -> float:
        """Calculate reactive mode score."""
        score = 0.0

        # Market condition scoring
        if criteria.market_condition == MarketCondition.BEAR_MARKET:
            score += 0.4
        elif criteria.market_condition == MarketCondition.HIGH_VOLATILITY:
            score += 0.3
        elif criteria.market_condition == MarketCondition.CRISIS:
            score += 0.5

        # Trend strength scoring (reactive prefers weak trends)
        if criteria.trend_strength < -0.3:
            score += 0.2
        elif unified_math.abs(criteria.trend_strength) < 0.2:
            score += 0.1

        # Volatility scoring (reactive handles high volatility well)
        if criteria.volatility_score > 0.7:
            score += 0.3
        elif criteria.volatility_score > 0.5:
            score += 0.2

        # Timeframe scoring
        timeframe_weights = self.timeframe_weights.get(criteria.timeframe, {"zpe": 0.5, "reactive": 0.5})
        score += timeframe_weights["reactive"] * 0.2

        # Profit performance scoring (reactive for poor performance)
        if criteria.profit_performance < -0.1:
            score += 0.2

        return unified_math.min(1.0, score)

    def _calculate_hybrid_score(self, criteria: ModeSelectionCriteria) -> float:
        """Calculate hybrid mode score."""
        # Hybrid is good for mixed conditions
        score = 0.3  # Base score

        # Mixed market conditions
        if criteria.market_condition == MarketCondition.SIDEWAYS:
            score += 0.2

        # Moderate volatility
        if 0.3 <= criteria.volatility_score <= 0.7:
            score += 0.2

        # Mixed trend strength
        if -0.3 <= criteria.trend_strength <= 0.3:
            score += 0.2

        # Both modes available
        if criteria.zpe_availability and criteria.reactive_availability:
            score += 0.1

        return unified_math.min(1.0, score)

    def _calculate_mode_weights(
        self,
        criteria: ModeSelectionCriteria,
        selected_mode: TradingMode
    ) -> Dict[TradingMode, float]:
        """Calculate mode weights for hybrid scenarios."""
        weights = {
            TradingMode.ZPE_RECURSIVE: 0.0,
            TradingMode.REACTIVE_TASKING: 0.0,
            TradingMode.HYBRID_BLEND: 0.0,
            TradingMode.EMERGENCY_FALLBACK: 0.0
        }

        if selected_mode == TradingMode.HYBRID_BLEND:
            # Calculate hybrid weights based on conditions
            zpe_weight = self._calculate_zpe_score(criteria)
            reactive_weight = self._calculate_reactive_score(criteria)

            # Normalize weights
            total_weight = zpe_weight + reactive_weight
            if total_weight > 0:
                weights[TradingMode.ZPE_RECURSIVE] = zpe_weight / total_weight
                weights[TradingMode.REACTIVE_TASKING] = reactive_weight / total_weight
            else:
                weights[TradingMode.ZPE_RECURSIVE] = 0.5
                weights[TradingMode.REACTIVE_TASKING] = 0.5
        else:
            weights[selected_mode] = 1.0

        return weights

    def _update_portfolio_assets(self, portfolio_data: Dict[str, Any]) -> None:
        """Update portfolio assets for retroactive tasking."""
        try:
            assets = portfolio_data.get('assets', {})

            for symbol, asset_data in assets.items():
                if symbol not in self.portfolio_assets:
                    self.portfolio_assets[symbol] = PortfolioAsset(
                        symbol=symbol,
                        current_value=asset_data.get('value', 0.0),
                        allocation_percentage=asset_data.get('allocation', 0.0),
                        last_trade_time=datetime.now(),
                        profit_performance=asset_data.get('performance', 0.0),
                        volatility=asset_data.get('volatility', 0.5)
                    )
                else:
                    # Update existing asset
                    asset = self.portfolio_assets[symbol]
                    asset.current_value = asset_data.get('value', asset.current_value)
                    asset.allocation_percentage = asset_data.get('allocation', asset.allocation_percentage)
                    asset.profit_performance = asset_data.get('performance', asset.profit_performance)
                    asset.volatility = asset_data.get('volatility', asset.volatility)

        except Exception as e:
            safe_safe_print(f"⚠️ Portfolio update failed: {safe_format_error(e, 'portfolio_update')}")

    def get_retroactive_tasking_candidates(self) -> List[PortfolioAsset]:
        """Get portfolio assets that can be retroactively tasked."""
        candidates = []

        for asset in self.portfolio_assets.values():
            if asset.can_retroactive_task:
                # Check if asset meets retroactive tasking criteria
                if (asset.profit_performance < -0.05 or  # Poor performance
                    asset.volatility > 0.7 or           # High volatility
                    asset.allocation_percentage > 0.2):  # High allocation
                    candidates.append(asset)

        # Sort by priority (poor performance first)
        candidates.sort(key=lambda x: x.profit_performance)
        return candidates

    def record_mode_performance(
        self,
        mode: TradingMode,
        performance: float,
        market_condition: MarketCondition
    ) -> None:
        """Record performance for mode selection learning."""
        try:
            if mode not in self.mode_performance_history:
                self.mode_performance_history[mode] = []

            self.mode_performance_history[mode].append(performance)

            # Keep only recent performance (last 100)
            if len(self.mode_performance_history[mode]) > 100:
                self.mode_performance_history[mode] = self.mode_performance_history[mode][-100:]

        except Exception as e:
            safe_safe_print(f"⚠️ Performance recording failed: {safe_format_error(e, 'performance_recording')}")

    def get_mode_statistics(self) -> Dict[str, Any]:
        """Get mode selection statistics."""
        try:
            stats = {
                'total_selections': sum(len(perf) for perf in self.mode_performance_history.values()),
                'mode_performance': {},
                'phase_488_active': self.phase_488_active,
                'phase_42_active': self.phase_42_active,
                'phase_switch_counter': self.phase_switch_counter,
                'portfolio_assets': len(self.portfolio_assets),
                'retroactive_candidates': len(self.get_retroactive_tasking_candidates())
            }

            for mode, performance_list in self.mode_performance_history.items():
                if performance_list:
                    stats['mode_performance'][mode.value] = {
                        'count': len(performance_list),
                        'average': sum(performance_list) / len(performance_list),
                        'recent': performance_list[-10:] if len(performance_list) >= 10 else performance_list
                    }

            return stats

        except Exception as e:
            safe_safe_print(f"⚠️ Statistics calculation failed: {safe_format_error(e, 'statistics')}")
            return {}


# Global mode selector instance
hybrid_mode_selector = ZPEHybridModeSelector()


# Convenience functions for external access
def select_trading_mode(
    market_data: Dict[str, Any],
    portfolio_data: Optional[Dict[str, Any]] = None,
    timeframe: str = "daily"
) -> ModeSelectionResult:
    """Select trading mode using global selector."""
    return hybrid_mode_selector.select_mode(market_data, portfolio_data, timeframe)


def get_retroactive_candidates() -> List[PortfolioAsset]:
    """Get retroactive tasking candidates."""
    return hybrid_mode_selector.get_retroactive_tasking_candidates()


def record_performance(mode: TradingMode, performance: float, market_condition: MarketCondition) -> None:
    """Record mode performance."""
    hybrid_mode_selector.record_mode_performance(mode, performance, market_condition)


def get_mode_stats() -> Dict[str, Any]:
    """Get mode selection statistics."""
    return hybrid_mode_selector.get_mode_statistics()


# Example usage
if __name__ == "__main__":
    # Test mode selection
    test_market_data = {
        'trend_strength': 0.8,
        'volatility': 0.3,
        'price_change_24h': 0.08,
        'profit_performance': 0.15
    }

    result = select_trading_mode(test_market_data, timeframe="daily")
    safe_print(f"Selected Mode: {result.selected_mode.value}")
    safe_print(f"Confidence: {result.confidence_score:.3f}")
    safe_print(f"Market Condition: {result.market_condition.value}")
    safe_print(f"Reasoning: {result.reasoning}")
