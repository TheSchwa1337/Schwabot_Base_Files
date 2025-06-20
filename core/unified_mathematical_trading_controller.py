#!/usr/bin/env python3
"""
Unified Mathematical Trading Controller - Schwabot Framework.

Central mathematical trading system that unifies all mathematical components
including ghost data recovery, profit routing, ferris wheel matrix operations,
thermal processing, and constraint validation with Windows CLI compatibility.

Key Features:
- Decimal precision financial calculations
- Ghost swap signal detection and recovery
- Profit vector routing and optimization
- Thermal-aware BTC processing
- Ferris wheel matrix cycle management
- Mode-aware constraint validation
- Real-time mathematical analysis
"""

from __future__ import annotations

import logging
from decimal import Decimal, getcontext
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from .ghost_profit_tracker import register_profit, profit_summary

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class SafeDecimalHandler:
    """Handles safe conversion between float and Decimal types."""

    @staticmethod
    def safe_float(x: float | Decimal) -> float:
        """Safely convert Decimal to float."""
        return float(x) if isinstance(x, Decimal) else x

    @staticmethod
    def safe_decimal(x: float | str) -> Decimal:
        """Safely convert float to Decimal."""
        return Decimal(str(x))

    @staticmethod
    def safe_cast_funds(x: float | str) -> Decimal:
        """Safely cast funds with high precision."""
        return Decimal(str(x))


class MathematicalConstraints:
    """Mathematical bounds and validation constraints."""

    def __init__(self: MathematicalConstraints) -> None:
        """Initialize constraints."""
        self.epsilon = Decimal("1e-12")
        self.max_position_size = Decimal("1.0")
        self.max_leverage = Decimal("2.0")
        self.min_thermal_bound = Decimal("-0.05")
        self.max_thermal_bound = Decimal("0.10")

    def bounded_profit(
        self: MathematicalConstraints,
        x: float | Decimal,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> Decimal:
        """Apply thermal bounds to profit values."""
        value = self.safe_decimal(x) if not isinstance(x, Decimal) else x

        min_bound = (
            Decimal(str(min_val)) if min_val is not None
            else self.min_thermal_bound
        )
        max_bound = (
            Decimal(str(max_val)) if max_val is not None
            else self.max_thermal_bound
        )

        return max(min(value, max_bound), min_bound)

    def safe_decimal(self: MathematicalConstraints, x: float | str) -> Decimal:
        """Safely convert to Decimal."""
        return SafeDecimalHandler.safe_decimal(x)


class TradingVector:
    """Represents a single trading operation vector."""

    def __init__(
        self: TradingVector,
        asset: str,
        entry_price: float,
        exit_price: float,
        volume: float,
        thermal_index: float,
        timestamp: float
    ) -> None:
        """Initialize trading vector."""
        self.asset = asset
        self.entry_price = Decimal(str(entry_price))
        self.exit_price = Decimal(str(exit_price))
        self.volume = Decimal(str(volume))
        self.thermal_index = Decimal(str(thermal_index))
        self.timestamp = Decimal(str(timestamp))
        self.profit = self._calculate_profit()
        self.efficiency = self._calculate_efficiency()

    def _calculate_profit(self: TradingVector) -> Decimal:
        """Calculate profit from price difference and volume."""
        return (self.exit_price - self.entry_price) * self.volume

    def _calculate_efficiency(self: TradingVector) -> Decimal:
        """Calculate efficiency ratio (profit per thermal cost)."""
        if self.thermal_index == 0:
            return Decimal("0.0")
        return self.profit / self.thermal_index


class GhostSwapDetector:
    """Detects and manages ghost swap signals."""

    def __init__(self: GhostSwapDetector) -> None:
        """Initialize ghost swap detector."""
        self.phantom_triggers = []
        self.signal_registry = {}

    def detect_phantom_trigger(
        self: GhostSwapDetector,
        delta_t: Decimal,
        delta_price: Decimal,
        delta_volume: Decimal
    ) -> bool:
        """Detect phantom swap triggers based on delta patterns."""
        # Phantom trigger: rapid price movement with low volume
        rapid_price = (
            delta_t < Decimal("0.5") and
            abs(delta_price) > Decimal("50")
        )
        low_volume = delta_volume < Decimal("0.1")

        return rapid_price and low_volume

    def register_ghost_signal(
        self: GhostSwapDetector,
        strategy: str,
        asset_pair: str,
        timestamp: Decimal
    ) -> str:
        """Register a ghost swap signal and return its ID."""
        import hashlib

        signal_data = f"{timestamp}{strategy}{asset_pair}"
        signal_id = hashlib.sha256(signal_data.encode()).hexdigest()

        self.signal_registry[signal_id] = {
            'strategy': strategy,
            'asset_pair': asset_pair,
            'timestamp': timestamp,
            'active': True
        }

        return signal_id


class FerrisWheelCycleEngine:
    """Manages cyclic trading patterns with matrix-based feedback."""

    def __init__(self: FerrisWheelCycleEngine) -> None:
        """Initialize Ferris wheel cycle engine."""
        self.cycles = {}
        self.feedback_stabilizer = Decimal("0.0")

    def create_cycle(
        self: FerrisWheelCycleEngine,
        cycle_name: str,
        base_thermal: float
    ) -> None:
        """Create a new trading cycle."""
        self.cycles[cycle_name] = {
            'thermal_base': Decimal(str(base_thermal)),
            'vectors': [],
            'total_profit': Decimal("0.0"),
            'cycle_position': 0,
            'stabilizer_delta': Decimal("0.0")
        }

    def add_vector_to_cycle(
        self: FerrisWheelCycleEngine,
        cycle_name: str,
        vector: TradingVector
    ) -> None:
        """Add a trading vector to a cycle."""
        if cycle_name not in self.cycles:
            self.create_cycle(cycle_name, float(vector.thermal_index))

        cycle = self.cycles[cycle_name]
        cycle['vectors'].append(vector)
        cycle['total_profit'] += vector.profit
        cycle['cycle_position'] += 1

        # Apply feedback stabilization
        self._apply_feedback_stabilization(cycle_name)

    def _apply_feedback_stabilization(
        self: FerrisWheelCycleEngine,
        cycle_name: str
    ) -> None:
        """Apply stabilization feedback to cycle."""
        cycle = self.cycles[cycle_name]

        if len(cycle['vectors']) < 2:
            return

        # Calculate stabilizer based on profit variance
        profits = [v.profit for v in cycle['vectors']]
        profit_variance = Decimal(str(np.var([float(p) for p in profits])))

        # Stabilizer reduces excessive variance
        stabilizer_strength = Decimal("0.1")
        cycle['stabilizer_delta'] = stabilizer_strength * profit_variance

        # Apply bounded stabilization
        constraints = MathematicalConstraints()
        cycle['stabilizer_delta'] = constraints.bounded_profit(
            cycle['stabilizer_delta'], -0.02, 0.02
        )

    def get_cycle_thermal_signature(
        self: FerrisWheelCycleEngine,
        cycle_name: str
    ) -> Dict[str, Decimal]:
        """Get thermal signature for a cycle."""
        if cycle_name not in self.cycles:
            return {}

        cycle = self.cycles[cycle_name]
        base_thermal = cycle['thermal_base']
        total_profit = cycle['total_profit']
        stabilizer_delta = cycle['stabilizer_delta']

        # Calculate thermal drift
        thermal_drift = (
            total_profit / base_thermal if base_thermal != 0
            else Decimal("0.0")
        )

        return {
            'base_thermal': base_thermal,
            'current_thermal': base_thermal + thermal_drift,
            'thermal_drift': thermal_drift,
            'total_profit': total_profit,
            'stabilizer_delta': stabilizer_delta,
            'vector_count': len(cycle['vectors'])
        }


class UnifiedMathematicalTradingController:
    """Unified mathematical trading controller with all components."""

    def __init__(self: UnifiedMathematicalTradingController) -> None:
        """Initialize unified trading controller."""
        self.version = "1.0.0"
        self.constraints = MathematicalConstraints()
        self.safe_decimal = SafeDecimalHandler()
        self.ghost_detector = GhostSwapDetector()
        self.ferris_engine = FerrisWheelCycleEngine()
        self.trading_vectors: list[TradingVector] = []
        self.profit_memory: Dict[str, Dict[str, Any]] = {}

    def process_trade_signal(
        self: UnifiedMathematicalTradingController,
        signal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a trade signal and return analysis results."""
        try:
            # Extract signal data
            asset = signal_data.get('asset', 'UNKNOWN')
            entry_price = signal_data.get('entry_price', 0.0)
            exit_price = signal_data.get('exit_price', 0.0)
            volume = signal_data.get('volume', 0.0)
            thermal_index = signal_data.get('thermal_index', 0.0)
            timestamp = signal_data.get('timestamp', 0.0)
            strategy = signal_data.get('strategy', 'default')

            # Create trading vector
            vector = TradingVector(
                asset=asset,
                entry_price=entry_price,
                exit_price=exit_price,
                volume=volume,
                thermal_index=thermal_index,
                timestamp=timestamp
            )

            # Apply constraints
            bounded_profit = self.constraints.bounded_profit(vector.profit)

            # Track profit in global tracker
            register_profit(float(bounded_profit))

            # Check for ghost signals
            delta_t = Decimal("1.0")  # Default time delta
            delta_price = vector.exit_price - vector.entry_price
            delta_volume = vector.volume

            is_phantom = self.ghost_detector.detect_phantom_trigger(
                delta_t, delta_price, delta_volume
            )

            ghost_signal_id = None
            if is_phantom:
                ghost_signal_id = self.ghost_detector.register_ghost_signal(
                    strategy, asset, vector.timestamp
                )

            # Add to Ferris wheel cycle
            cycle_name = f"{strategy}_{asset}"
            self.ferris_engine.add_vector_to_cycle(cycle_name, vector)

            # Store in profit memory
            profit_key = f"{asset}_{strategy}_{int(timestamp)}"
            self.profit_memory[profit_key] = {
                'profit': bounded_profit,
                'efficiency': vector.efficiency,
                'thermal_signature': (
                    self.ferris_engine.get_cycle_thermal_signature(cycle_name)
                ),
                'ghost_signal': ghost_signal_id
            }

            # Store vector
            self.trading_vectors.append(vector)

            return {
                'status': 'success',
                'vector_id': len(self.trading_vectors) - 1,
                'profit': float(bounded_profit),
                'efficiency': float(vector.efficiency),
                'is_phantom_trigger': is_phantom,
                'ghost_signal_id': ghost_signal_id,
                'cycle_name': cycle_name,
                'thermal_signature': {
                    k: float(v)
                    for k, v in self.ferris_engine.get_cycle_thermal_signature(
                        cycle_name
                    ).items()
                },
                'tracked_profit_total': profit_summary()[0]
            }

        except Exception as e:
            logger.error(f"Error processing trade signal: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'signal_data': signal_data
            }

    def get_optimal_allocation(
        self: UnifiedMathematicalTradingController,
        available_capital: float,
        risk_tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """Calculate optimal capital allocation across trading vectors."""
        if not self.trading_vectors:
            return {'status': 'no_vectors', 'allocation': {}}

        capital = self.safe_decimal.safe_decimal(available_capital)

        # Calculate efficiency scores for all vectors
        efficiency_scores = [v.efficiency for v in self.trading_vectors]
        total_efficiency = sum(efficiency_scores)

        if total_efficiency <= 0:
            return {'status': 'negative_efficiency', 'allocation': {}}

        # Allocate capital proportional to efficiency
        allocations = {}
        for i, vector in enumerate(self.trading_vectors):
            if vector.efficiency > 0:
                allocation_ratio = vector.efficiency / total_efficiency
                allocated_amount = capital * allocation_ratio

                # Apply risk constraints
                max_allocation = capital * Decimal(str(risk_tolerance))
                final_allocation = min(allocated_amount, max_allocation)

                allocations[f"{vector.asset}_{i}"] = {
                    'amount': float(final_allocation),
                    'efficiency': float(vector.efficiency),
                    'thermal_index': float(vector.thermal_index),
                    'expected_profit': float(vector.profit * allocation_ratio)
                }

        return {
            'status': 'success',
            'total_capital': float(capital),
            'allocated_capital': float(
                sum(Decimal(str(a['amount'])) for a in allocations.values())
            ),
            'allocation': allocations
        }

    def analyze_thermal_zones(
        self: UnifiedMathematicalTradingController
    ) -> Dict[str, Any]:
        """Analyze thermal patterns across all trading zones."""
        thermal_analysis = {}

        for cycle_name, cycle_data in self.ferris_engine.cycles.items():
            signature = (
                self.ferris_engine.get_cycle_thermal_signature(cycle_name)
            )

            thermal_analysis[cycle_name] = {
                'thermal_stability': float(
                    abs(signature.get('thermal_drift', 0))
                ),
                'profit_thermal_ratio': (
                    float(signature.get('total_profit', 0)) /
                    float(signature.get('current_thermal', 1))
                ),
                'stabilizer_impact': float(signature.get('stabilizer_delta', 0)),
                'vector_count': len(cycle_data['vectors']),
                'thermal_efficiency': (
                    float(signature.get('total_profit', 0)) /
                    float(signature.get('base_thermal', 1))
                )
            }

        return {
            'thermal_zones': thermal_analysis,
            'total_zones': len(thermal_analysis),
            'most_stable_zone': (
                max(
                    thermal_analysis.keys(),
                    key=lambda x: thermal_analysis[x]['thermal_stability']
                ) if thermal_analysis else None
            )
        }

    def get_system_status(
        self: UnifiedMathematicalTradingController
    ) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'version': self.version,
            'total_vectors': len(self.trading_vectors),
            'active_cycles': len(self.ferris_engine.cycles),
            'ghost_signals': len(self.ghost_detector.signal_registry),
            'profit_memory_entries': len(self.profit_memory),
            'total_profit': float(sum(v.profit for v in self.trading_vectors)),
            'tracked_profit_total': profit_summary()[0],
            'average_efficiency': float(
                sum(v.efficiency for v in self.trading_vectors) /
                len(self.trading_vectors)
            ) if self.trading_vectors else 0.0,
            'thermal_analysis': self.analyze_thermal_zones()
        }


def main() -> None:
    """Demo of unified mathematical trading controller."""
    try:
        controller = UnifiedMathematicalTradingController()
        print("✅ UnifiedMathematicalTradingController v{} initialized".format(
            controller.version
        ))

        # Demo trade signals
        demo_signals = [
            {
                'asset': 'BTC',
                'entry_price': 26000.0,
                'exit_price': 27200.0,
                'volume': 0.5,
                'thermal_index': 1.2,
                'timestamp': 1640995200.0,
                'strategy': 'momentum'
            },
            {
                'asset': 'ETH',
                'entry_price': 1700.0,
                'exit_price': 1850.0,
                'volume': 2.0,
                'thermal_index': 0.9,
                'timestamp': 1640995260.0,
                'strategy': 'arbitrage'
            },
            {
                'asset': 'BTC',
                'entry_price': 27200.0,
                'exit_price': 26800.0,  # Loss trade
                'volume': 0.3,
                'thermal_index': 2.1,
                'timestamp': 1640995320.0,
                'strategy': 'momentum'
            }
        ]

        # Process signals
        for signal in demo_signals:
            result = controller.process_trade_signal(signal)
            print(f"📊 Processed {signal['asset']} signal: "
                  f"Profit ${result.get('profit', 0):.2f}, "
                  f"Efficiency {result.get('efficiency', 0):.3f}")

        # Get optimal allocation
        allocation = controller.get_optimal_allocation(10000.0, 0.15)
        print(f"💰 Optimal allocation status: {allocation['status']}")
        if allocation['status'] == 'success':
            print(f"📈 Total allocated: ${allocation['allocated_capital']:.2f}")

        # System status
        status = controller.get_system_status()
        print("🎯 System Status:")
        print(f"   Vectors: {status['total_vectors']}")
        print(f"   Cycles: {status['active_cycles']}")
        print(f"   Ghost signals: {status['ghost_signals']}")
        print(f"   Total profit: ${status['total_profit']:.2f}")
        print(f"   Tracked profit total: ${status['tracked_profit_total']:.2f}")
        print(f"   Avg efficiency: {status['average_efficiency']:.3f}")

        print("🎉 Unified mathematical trading controller demo completed!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main() 