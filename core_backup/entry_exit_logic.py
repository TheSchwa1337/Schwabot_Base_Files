# -*- coding: utf-8 -*-
""""""
Entry/Exit Logic Module.

Implements trading entry and exit logic based on:
- Drift shell vectorized signals
- Ferris wheel phase alignment
- Ghost overlay analysis
- Multi-bit strategy integration

Provides comprehensive trading signal generation with
risk management and performance tracking.
""""""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from core.order_book_vectorizer import OrderBookVectorizer
from core.strategy_bit_mapper import StrategyBitMapper
from core.api_bridge import APIBridge
from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


class EntryExitLogic:
    """"""
    Entry/Exit Logic Engine for trading signal generation.

    Integrates multiple data sources and mathematical frameworks
    to generate comprehensive trading signals with risk management.
    """"""

    def __init__()
        self,
            entry_threshold: float = 0.42,
                exit_threshold: float = -0.38,
                risk_management_enabled: bool = True,
                position_sizing_enabled: bool = True,
                max_position_size: float = 0.25,
                stop_loss_pct: float = 0.2,
                take_profit_pct: float = 0.4,
                enable_ghost_overlay: bool = True,
                enable_ferris_phase: bool = True,
                ):
        """Initialize the entry/exit logic engine."""

        Args:
            entry_threshold: Signal strength threshold for entry
            exit_threshold: Signal strength threshold for exit
            risk_management_enabled: Enable risk management features
            position_sizing_enabled: Enable position sizing
            max_position_size: Maximum position size as fraction of capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            enable_ghost_overlay: Enable Ghost overlay analysis
            enable_ferris_phase: Enable Ferris wheel phase analysis
        """"""
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.risk_management_enabled = risk_management_enabled
        self.position_sizing_enabled = position_sizing_enabled
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.enable_ghost_overlay = enable_ghost_overlay
        self.enable_ferris_phase = enable_ferris_phase

        # Initialize components
        self.order_book_vectorizer = OrderBookVectorizer()
        self.strategy_bit_mapper = StrategyBitMapper()
        self.api_bridge = APIBridge()

        # Performance tracking
        self.trading_stats = {}
            "total_signals": 0,
                "entry_signals": 0,
                    "exit_signals": 0,
                    "hold_signals": 0,
                    "successful_entries": 0,
                    "successful_exits": 0,
                    "avg_signal_strength": 0.0,
                    "avg_processing_time": 0.0,
}
        # Signal history for analysis
        self.signal_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        logger.info()
            f"EntryExitLogic initialized: "
            f"entry_threshold={entry_threshold}, "
            f"exit_threshold={exit_threshold}, "
            f"risk_management={risk_management_enabled}, "
            f"ghost_overlay={enable_ghost_overlay}, "
            f"ferris_phase={enable_ferris_phase}"
        )

    def compute_entry_signal()
        self,
            vector: np.ndarray,
                ferris_phase: float,
                ghost_input: float,
                metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """"""
        Compute entry signal based on drift shell vectorized signal + Ferris phase + Ghost overlay.

        Args:
            vector: Vectorized order book data
            ferris_phase: Ferris wheel phase value
            ghost_input: Ghost overlay input value
            metadata: Additional metadata for signal computation

        Returns:
            Dictionary with entry signal results
        """"""
        start_time = time.time()

        try:
            # Compute basic signal components
            vector_mean = np.mean(vector)
            volatility = np.std(vector)

            # Phase weight based on Ferris wheel alignment
            if self.enable_ferris_phase:
                phase_weight = np.cos(ferris_phase)  # Peak alignment
            else:
                phase_weight = 1.0

            # Ghost gate function
            if self.enable_ghost_overlay:
                ghost_gate = np.tanh(ghost_input)
            else:
                ghost_gate = 1.0

            # Compute signal strength
            signal_strength = (vector_mean * ghost_gate) - (volatility * (1 - phase_weight))

            # Determine entry decision
            should_enter = signal_strength > self.entry_threshold

            # Compute additional metrics
            signal_metrics = self._compute_signal_metrics()
                vector, ferris_phase, ghost_input, signal_strength
            )

            # Risk assessment
            risk_assessment = self._assess_risk(vector, signal_strength, metadata)

            # Position sizing recommendation
            position_size = self._compute_position_size(signal_strength, risk_assessment)

            # Create result dictionary
            result = {
                "should_enter": should_enter,
                "signal_strength": signal_strength,
                "vector_mean": vector_mean,
                "volatility": volatility,
                "phase_weight": phase_weight,
                "ghost_gate": ghost_gate,
                "signal_metrics": signal_metrics,
                "risk_assessment": risk_assessment,
                "position_size": position_size,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time,
}
}
            # Update statistics
            self._update_stats("entry" if should_enter else "hold", result["processing_time"])

            # Store in history
            self._store_signal_history(result, "entry")

            logger.debug()
                f"Entry signal computed: strength={signal_strength:.4f}, "
                f"enter={should_enter}, time={result['processing_time']:.6f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Entry signal computation failed: {e}")
            return self._generate_fallback_signal("entry", time.time() - start_time)

    def compute_exit_signal()
        self,
            vector: np.ndarray,
                ferris_phase: float,
                ghost_input: float,
                entry_price: float,
                current_price: float,
                time_held: float,
                metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """"""
        Compute exit signal based on current market conditions and position metrics.

        Args:
            vector: Vectorized order book data
            ferris_phase: Ferris wheel phase value
            ghost_input: Ghost overlay input value
            entry_price: Entry price of the position
            current_price: Current market price
            time_held: Time position has been held
            metadata: Additional metadata for signal computation

        Returns:
            Dictionary with exit signal results
        """"""
        start_time = time.time()

        try:
            # Compute basic signal components
            vector_mean = np.mean(vector)
            volatility = np.std(vector)

            # Phase weight (inverted for exit logic)
            if self.enable_ferris_phase:
                phase_weight = -np.cos(ferris_phase)  # Inverted for exit
            else:
                phase_weight = 0.0

            # Ghost gate function
            if self.enable_ghost_overlay:
                ghost_gate = np.tanh(ghost_input)
            else:
                ghost_gate = 1.0

            # Compute signal strength (inverted for exit)
            signal_strength = -(vector_mean * ghost_gate) + (volatility * (1 + phase_weight))

            # Determine exit decision
            should_exit = signal_strength < self.exit_threshold

            # Check stop loss and take profit
            price_change_pct = (current_price - entry_price) / entry_price

            stop_loss_triggered = price_change_pct < -self.stop_loss_pct
            take_profit_triggered = price_change_pct > self.take_profit_pct

            # Force exit if stop loss or take profit triggered
            if stop_loss_triggered or take_profit_triggered:
                should_exit = True
                signal_strength = self.exit_threshold - 0.1  # Force exit signal

            # Compute additional metrics
            signal_metrics = self._compute_signal_metrics()
                vector, ferris_phase, ghost_input, signal_strength
            )

            # Risk assessment
            risk_assessment = self._assess_risk(vector, signal_strength, metadata)

            # Create result dictionary
            result = {
                "should_exit": should_exit,
                "signal_strength": signal_strength,
                "vector_mean": vector_mean,
                "volatility": volatility,
                "phase_weight": phase_weight,
                "ghost_gate": ghost_gate,
                "price_change_pct": price_change_pct,
                "stop_loss_triggered": stop_loss_triggered,
                "take_profit_triggered": take_profit_triggered,
                "time_held": time_held,
                "signal_metrics": signal_metrics,
                "risk_assessment": risk_assessment,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time,
}
}
            # Update statistics
            self._update_stats("exit" if should_exit else "hold", result["processing_time"])

            # Store in history
            self._store_signal_history(result, "exit")

            logger.debug()
                f"Exit signal computed: strength={signal_strength:.4f}, "
                f"exit={should_exit}, price_change={price_change_pct:.4f}, "
                f"time={result['processing_time']:.6f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Exit signal computation failed: {e}")
            return self._generate_fallback_signal("exit", time.time() - start_time)

    def _compute_signal_metrics()
        self,
            vector: np.ndarray,
                ferris_phase: float,
                ghost_input: float,
                signal_strength: float
    ) -> Dict[str, float]:
        """Compute additional signal metrics."""
        try:
            metrics = {
                "vector_entropy": self._compute_entropy(vector),
                "vector_skewness": self._compute_skewness(vector),
                "phase_alignment": np.cos(ferris_phase),
                "ghost_intensity": np.abs(ghost_input),
                "signal_confidence": np.tanh(np.abs(signal_strength)),
                "vector_diversity": len(set(vector)) / len(vector),
}
}
            return metrics

        except Exception as e:
            logger.error(f"Signal metrics computation failed: {e}")
            return {}

    def _assess_risk()
        self,
            vector: np.ndarray,
                signal_strength: float,
                metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Assess risk for the trading signal."""
        try:
            # Basic risk metrics
            volatility = np.std(vector)
            signal_volatility = np.std([signal_strength])

            # Market risk (based on vector characteristics)
            market_risk = volatility * (1 - np.abs(signal_strength))

            # Signal risk (based on signal strength)
            signal_risk = 1 - np.abs(signal_strength)

            # Overall risk score
            overall_risk = (market_risk + signal_risk) / 2

            risk_assessment = {
                "market_risk": market_risk,
                "signal_risk": signal_risk,
                "overall_risk": overall_risk,
                "volatility": volatility,
                "signal_volatility": signal_volatility,
                "risk_score": overall_risk,  # Normalized risk score
}
}
            return risk_assessment

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"overall_risk": 0.5, "risk_score": 0.5}

    def _compute_position_size()
        self,
            signal_strength: float,
                risk_assessment: Dict[str, float]
    ) -> float:
        """Compute recommended position size based on signal and risk."""
        if not self.position_sizing_enabled:
            return self.max_position_size

        try:
            # Base position size from signal strength
            base_size = np.abs(signal_strength) * self.max_position_size

            # Adjust for risk
            risk_factor = 1 - risk_assessment.get("risk_score", 0.5)
            adjusted_size = base_size * risk_factor

            # Apply limits
            position_size = np.clip(adjusted_size, 0.1, self.max_position_size)

            return position_size

        except Exception as e:
            logger.error(f"Position size computation failed: {e}")
            return self.max_position_size * 0.5  # Conservative fallback

    def _compute_entropy(self, vector: np.ndarray) -> float:
        """Compute Shannon entropy of the vector."""
        try:
            # Normalize to probability distribution
            vector_norm = np.abs(vector).astype(float)
            if np.sum(vector_norm) == 0:
                return 0.0
            vector_norm = vector_norm / np.sum(vector_norm)

            # Compute entropy
            entropy = -np.sum(vector_norm * np.log2(vector_norm + 1e-12))
            return entropy
        except Exception:
            return 0.0

    def _compute_skewness(self, vector: np.ndarray) -> float:
        """Compute skewness of the vector."""
        try:
            mean = np.mean(vector)
            std = np.std(vector)
            if std == 0:
                return 0.0
            skewness = np.mean(((vector - mean) / std) ** 3)
            return skewness
        except Exception:
            return 0.0

    def _update_stats(self, signal_type: str, processing_time: float) -> None:
        """Update trading statistics."""
        self.trading_stats["total_signals"] += 1

        if signal_type == "entry":
            self.trading_stats["entry_signals"] += 1
        elif signal_type == "exit":
            self.trading_stats["exit_signals"] += 1
        else:
            self.trading_stats["hold_signals"] += 1

        # Update average processing time
        total_time = self.trading_stats["avg_processing_time"] * ()
            self.trading_stats["total_signals"] - 1
        )
        self.trading_stats["avg_processing_time"] = ()
            (total_time + processing_time) / self.trading_stats["total_signals"]
        )

    def _store_signal_history(self, signal_result: Dict[str, Any], signal_type: str) -> None:
        """Store signal in history for analysis."""
        history_entry = {
            "timestamp": signal_result["timestamp"],
            "signal_type": signal_type,
            "signal_strength": signal_result["signal_strength"],
            "should_enter": signal_result.get("should_enter", False),
            "should_exit": signal_result.get("should_exit", False),
            "processing_time": signal_result["processing_time"],
            "signal_hash": hash(str(signal_result)),
}
}
        self.signal_history.append(history_entry)

        # Maintain history size
        if len(self.signal_history) > self.max_history_size:
            self.signal_history.pop(0)

    def _generate_fallback_signal(self, signal_type: str, processing_time: float) -> Dict[str, Any]:
        """Generate fallback signal when computation fails."""
        return {}
            "should_enter": False,
                "should_exit": False,
                    "signal_strength": 0.0,
                    "vector_mean": 0.0,
                    "volatility": 0.0,
                    "phase_weight": 1.0,
                    "ghost_gate": 1.0,
                    "signal_metrics": {},
                    "risk_assessment": {"overall_risk": 0.5, "risk_score": 0.5},
                    "position_size": self.max_position_size * 0.1,
                    "timestamp": time.time(),
                    "processing_time": processing_time,
}
    def get_trading_performance_summary(self) -> Dict[str, Union[int, float]]:
        """Get trading performance summary."""
        return self.trading_stats.copy()

    def get_signal_history_summary(self) -> Dict[str, Any]:
        """Get summary of signal history."""
        if not self.signal_history:
            return {"total_signals": 0, "avg_signal_strength": 0.0}

        signal_strengths = [entry["signal_strength"] for entry in self.signal_history]

        return {}
            "total_signals": len(self.signal_history),
                "avg_signal_strength": np.mean(signal_strengths),
                    "max_signal_strength": np.max(signal_strengths),
                    "min_signal_strength": np.min(signal_strengths),
                    "signal_std": np.std(signal_strengths),
}
    def clear_history(self) -> None:
        """Clear signal history."""
        self.signal_history.clear()
        logger.info("Signal history cleared")


# Global instance for easy access
entry_exit_logic = EntryExitLogic()


def compute_entry_signal()
    vector: np.ndarray,
        ferris_phase: float,
            ghost_input: float
) -> bool:
    """"""
    Standalone function for computing entry signal.

    Args:
        vector: Vectorized order book data
        ferris_phase: Ferris wheel phase value
        ghost_input: Ghost overlay input value

    Returns:
        True if should enter position, False otherwise
    """"""
    result = entry_exit_logic.compute_entry_signal(vector, ferris_phase, ghost_input)
    return result["should_enter"]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create test data
    test_vector = np.random.rand(16)  # 16-bit vector
    test_ferris_phase = np.pi / 4  # 45 degrees
    test_ghost_input = 0.5

    print(f"Test vector: {test_vector}")
    print(f"Ferris phase: {test_ferris_phase}")
    print(f"Ghost input: {test_ghost_input}")

    # Test entry signal
    entry_result = entry_exit_logic.compute_entry_signal()
        test_vector, test_ferris_phase, test_ghost_input
    )
    print(f"\nEntry signal result: {entry_result}")

    # Test exit signal
    exit_result = entry_exit_logic.compute_exit_signal()
        test_vector, test_ferris_phase, test_ghost_input,
            entry_price=62000.0, current_price=62500.0, time_held=3600.0
    )
    print(f"\nExit signal result: {exit_result}")

    # Performance summary
    performance = entry_exit_logic.get_trading_performance_summary()
    print(f"\nTrading performance: {performance}")

    # Signal history summary
    history_summary = entry_exit_logic.get_signal_history_summary()
    print(f"Signal history summary: {history_summary}")