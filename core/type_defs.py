#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Type Definitions

Type definitions for the Schwabot unified mathematics and trading system.
Provides consistent type annotations across all modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np

# Basic mathematical types
Vector = NewType("Vector", np.ndarray)
Matrix = NewType("Matrix", np.ndarray)
Tensor = NewType("Tensor", np.ndarray)
Scalar = Union[int, float, np.number]


# Trading types
class TradingAction(Enum):
    """Trading action types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(Enum):
    """Order types for trading."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


# Entropy and information types
@dataclass
class Entropy:
    """Entropy value with metadata."""

    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.value


@dataclass
class PricePoint:
    """Price point with timestamp."""

    price: float
    timestamp: float
    volume: Optional[float] = None


@dataclass
class MarketData:
    """Market data container."""

    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    timestamp: Optional[float] = None


@dataclass
class TradeSignal:
    """Trading signal container."""

    action: TradingAction
    confidence: float
    price: Optional[float] = None
    quantity: Optional[float] = None
    reason: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class Position:
    """Trading position."""

    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    timestamp: Optional[float] = None


@dataclass
class RiskMetrics:
    """Risk assessment metrics."""

    var_95: float  # Value at Risk at 95% confidence
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: Optional[float] = None


# Strategy types
StrategyFunction = Callable[[MarketData], TradeSignal]
RiskFunction = Callable[[Position], RiskMetrics]
SignalProcessor = Callable[[List[TradeSignal]], TradeSignal]


# Mathematical operation types
class MathOperation(Enum):
    """Mathematical operation types."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    LOG = "log"
    EXP = "exp"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"


@dataclass
class CalculationResult:
    """Result of a mathematical calculation."""

    value: Union[Scalar, Vector, Matrix, Tensor]
    operation: MathOperation
    inputs: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# Quantum and advanced types
@dataclass
class QuantumState:
    """Quantum state representation."""

    amplitude: complex
    phase: float
    coherence: float
    entanglement: Optional[float] = None


@dataclass
class WaveFunction:
    """Wave function representation."""

    states: List[QuantumState]
    normalization: float = 1.0

    def collapse(self) -> QuantumState:
        """Collapse wave function to single state."""
        if not self.states:
            return QuantumState(amplitude=0 + 0j, phase=0.0, coherence=0.0)
        return self.states[0]


# Error and status types
class ComponentStatus(Enum):
    """Component status types."""

    OPERATIONAL = "OPERATIONAL"
    WARNING = "WARNING"
    ERROR = "ERROR"
    OFFLINE = "OFFLINE"
    INITIALIZING = "INITIALIZING"


@dataclass
class SystemStatus:
    """System status container."""

    component_name: str
    status: ComponentStatus
    message: Optional[str] = None
    timestamp: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


# Configuration types
@dataclass
class TradingConfig:
    """Trading configuration."""

    symbol: str
    max_position_size: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_per_trade: float = 0.02


@dataclass
class MathConfig:
    """Mathematical configuration."""

    precision: int = 8
    use_numpy: bool = True
    enable_caching: bool = True
    cache_size: int = 1000


@dataclass
class SystemConfig:
    """System configuration."""
    log_level: str = "INFO"
    enable_debug: bool = False
    max_memory_usage: int = 1024  # MB
    enable_profiling: bool = False


# Unified types for backward compatibility
TradingData = Union[MarketData, TradeSignal, Position]
MathData = Union[Vector, Matrix, Tensor, Scalar]
StatusData = Union[SystemStatus, ComponentStatus]
ConfigData = Union[TradingConfig, MathConfig, SystemConfig]


# Type aliases for complex structures
TensorOperation = Callable[[Tensor, Tensor], Tensor]
StrategyPipeline = List[StrategyFunction]
RiskPipeline = List[RiskFunction]
ValidationFunction = Callable[[Any], bool]


# Advanced mathematical structures
@dataclass
class ComplexMatrix:
    """Complex-valued matrix."""

    real_part: Matrix
    imaginary_part: Matrix

    def to_complex(self) -> np.ndarray:
        """Convert to complex numpy array."""
        return self.real_part + 1j * self.imaginary_part


@dataclass
class SparseTensor:
    """Sparse tensor representation."""

    indices: List[Tuple[int, ...]]
    values: List[Scalar]
    shape: Tuple[int, ...]

    def to_dense(self) -> Tensor:
        """Convert to dense tensor."""
        dense = np.zeros(self.shape)
        for idx, val in zip(self.indices, self.values):
            dense[idx] = val
        return Tensor(dense)


# Profit and performance types
@dataclass
class ProfitMetrics:
    """Profit and performance metrics."""

    total_return: float
    annual_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    calmar_ratio: Optional[float] = None


@dataclass
class TradeRecord:
    """Individual trade record."""

    symbol: str
    action: TradingAction
    quantity: float
    price: float
    timestamp: float
    commission: float = 0.0
    slippage: float = 0.0


# Integration and pipeline types
IntegrationResult = Dict[str, Any]
PipelineStage = Callable[[Any], Any]
ValidationResult = Tuple[bool, str]
ProcessingPipeline = List[PipelineStage]


# Export list for easy imports
__all__ = [
    # Basic types
    "Vector",
    "Matrix",
    "Tensor",
    "Scalar",
    # Enums
    "TradingAction",
    "OrderType",
    "MathOperation",
    "ComponentStatus",
    # Dataclasses
    "Entropy",
    "PricePoint",
    "MarketData",
    "TradeSignal",
    "Position",
    "RiskMetrics",
    "CalculationResult",
    "QuantumState",
    "WaveFunction",
    "SystemStatus",
    "TradingConfig",
    "MathConfig",
    "SystemConfig",
    "ComplexMatrix",
    "SparseTensor",
    "ProfitMetrics",
    "TradeRecord",
    # Type aliases
    "TradingData",
    "MathData",
    "StatusData",
    "ConfigData",
    "TensorOperation",
    "StrategyPipeline",
    "RiskPipeline",
    "ValidationFunction",
    "IntegrationResult",
    "PipelineStage",
    "ValidationResult",
    "ProcessingPipeline",
    # Function types
    "StrategyFunction",
    "RiskFunction",
    "SignalProcessor",
]


def create_default_market_data(symbol: str = "BTC/USDC") -> MarketData:
    """Create default market data for testing."""
    return MarketData(
        symbol=symbol, price=50000.0, bid=49999.0, ask=50001.0, volume=1000.0, timestamp=None
    )


def create_default_trade_signal() -> TradeSignal:
    """Create default trade signal for testing."""
    return TradeSignal(action=TradingAction.HOLD, confidence=0.5, reason="Default signal")


def validate_trading_data(data: TradingData) -> ValidationResult:
    """Validate trading data structure."""
    try:
        if isinstance(data, MarketData):
            if data.price <= 0:
                return False, "Invalid price"
        elif isinstance(data, TradeSignal):
            if not 0 <= data.confidence <= 1:
                return False, "Invalid confidence range"
        elif isinstance(data, Position):
            if data.size <= 0:
                return False, "Invalid position size"

        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"


if __name__ == "__main__":
    # Test the type definitions
    print("Testing Schwabot Type Definitions...")

    # Test market data
    market_data = create_default_market_data()
    print(f"Market data: {market_data}")

    # Test trade signal
    signal = create_default_trade_signal()
    print(f"Trade signal: {signal}")

    # Test validation
    valid, msg = validate_trading_data(market_data)
    print(f"Market data validation: {valid} - {msg}")

    print("Type definitions test completed!")
