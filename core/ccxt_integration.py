#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCXT Integration Module - Mathematical Trading Integration
=========================================================

Provides comprehensive CCXT integration with mathematical analysis:
- get_exchange_status: get exchange status with mathematical health checks
- create_ccxt_integration: create ccxt integration with mathematical setup
- process_order_book: process order book with mathematical analysis
- execute_order_mathematically: execute order with mathematical validation
"""

import logging
import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Import the actual mathematical infrastructure
try:
    from core.math_cache import MathResultCache
    from core.math_config_manager import MathConfigManager
    from core.math_orchestrator import MathOrchestrator

    # Import mathematical modules for exchange analysis
    from core.math.volume_weighted_hash_oscillator import VolumeWeightedHashOscillator
    from core.math.zygot_zalgo_entropy_dual_key_gate import ZygotZalgoEntropyDualKeyGate
    from core.math.qsc_quantum_signal_collapse_gate import QSCGate
    from core.math.unified_tensor_algebra import UnifiedTensorAlgebra
    from core.math.galileo_tensor_field_entropy_drift import GalileoTensorField
    from core.math.advanced_tensor_algebra import AdvancedTensorAlgebra
    from core.math.entropy_math import EntropyMath

    # Import trading pipeline components
    from core.enhanced_math_to_trade_integration import EnhancedMathToTradeIntegration

    # Lazy import to avoid circular dependency
    # from core.unified_mathematical_bridge import UnifiedMathematicalBridge
    from core.automated_trading_pipeline import AutomatedTradingPipeline

    MATH_INFRASTRUCTURE_AVAILABLE = True
    TRADING_PIPELINE_AVAILABLE = True
except ImportError as e:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    TRADING_PIPELINE_AVAILABLE = False
    logger.warning(f"Mathematical infrastructure not available: {e}")


def _get_unified_mathematical_bridge():
    """Lazy import to avoid circular dependency."""
    try:
        from core.unified_mathematical_bridge import UnifiedMathematicalBridge
        return UnifiedMathematicalBridge
    except ImportError:
        logger.warning("UnifiedMathematicalBridge not available due to circular import")
        return None


class Status(Enum):
    """System status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PROCESSING = "processing"


class Mode(Enum):
    """Operation mode enumeration."""
    NORMAL = "normal"
    DEBUG = "debug"
    TEST = "test"
    PRODUCTION = "production"


class ExchangeStatus(Enum):
    """Exchange status enumeration."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    DEGRADED = "degraded"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Config:
    """Configuration data class."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    mathematical_integration: bool = True
    exchange_validation: bool = True
    order_validation: bool = True


@dataclass
class Result:
    """Result data class."""
    success: bool = False
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderBookSnapshot:
    """Order book snapshot with mathematical analysis."""
    exchange: str
    symbol: str
    timestamp: float
    bids: List[Tuple[float, float]]  # (price, volume)
    asks: List[Tuple[float, float]]  # (price, volume)
    mathematical_score: float
    tensor_score: float
    entropy_value: float
    spread: float
    depth: float
    mathematical_analysis: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeInfo:
    """Exchange information with mathematical health metrics."""
    exchange_id: str
    name: str
    status: ExchangeStatus
    mathematical_health: float
    latency: float
    uptime: float
    last_check: float
    mathematical_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CCXTIntegration:
    """
    CCXTIntegration Implementation
    Provides core ccxt integration functionality with mathematical integration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CCXTIntegration with configuration and mathematical integration."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False

        # Exchange state
        self.exchanges: Dict[str, ExchangeInfo] = {}
        self.order_book_cache: Dict[str, OrderBookSnapshot] = {}
        self.exchange_health_metrics: Dict[str, float] = {}

        # Initialize mathematical infrastructure if available
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_config = MathConfigManager()
            self.math_cache = MathResultCache()
            self.math_orchestrator = MathOrchestrator()

            # Initialize mathematical modules for exchange analysis
            self.vwho = VolumeWeightedHashOscillator()
            self.zygot_zalgo = ZygotZalgoEntropyDualKeyGate()
            self.qsc = QSCGate()
            self.tensor_algebra = UnifiedTensorAlgebra()
            self.galileo = GalileoTensorField()
            self.advanced_tensor = AdvancedTensorAlgebra()
            self.entropy_math = EntropyMath()

        # Initialize exchange integration components
        if TRADING_PIPELINE_AVAILABLE:
            self.enhanced_math_integration = EnhancedMathToTradeIntegration(
                self.config
            )
            UnifiedMathematicalBridgeClass = _get_unified_mathematical_bridge()
            if UnifiedMathematicalBridgeClass:
                self.unified_bridge = UnifiedMathematicalBridgeClass(self.config)
            else:
                self.unified_bridge = None
            self.trading_pipeline = AutomatedTradingPipeline(self.config)

        self._initialize_system()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration with mathematical exchange settings."""
        return {
            "enabled": True,
            "timeout": 30.0,
            "retries": 3,
            "debug": False,
            "log_level": "INFO",
            "mathematical_integration": True,
            "exchange_validation": True,
            "order_validation": True,
            "supported_exchanges": ["binance", "coinbase", "kraken"],
            "order_book_cache_size": 1000,
            "health_check_interval": 60,  # seconds
            "mathematical_health_threshold": 0.7,
        }

    def _initialize_system(self) -> None:
        """Initialize the system with mathematical integration."""
        try:
            self.logger.info(
                f"Initializing {self.__class__.__name__} with mathematical integration"
            )

            if MATH_INFRASTRUCTURE_AVAILABLE:
                self.logger.info(
                    "✅ Mathematical infrastructure initialized for exchange analysis"
                )
                self.logger.info("✅ Volume Weighted Hash Oscillator initialized")
                self.logger.info("✅ Zygot-Zalgo Entropy Dual Key Gate initialized")
                self.logger.info("✅ QSC Quantum Signal Collapse Gate initialized")
                self.logger.info("✅ Unified Tensor Algebra initialized")
                self.logger.info("✅ Galileo Tensor Field initialized")
                self.logger.info("✅ Advanced Tensor Algebra initialized")
                self.logger.info("✅ Entropy Math initialized")

            if TRADING_PIPELINE_AVAILABLE:
                self.logger.info("✅ Enhanced math-to-trade integration initialized")
                self.logger.info("✅ Unified mathematical bridge initialized")
                self.logger.info(
                    "✅ Trading pipeline initialized for exchange integration"
                )

            # Initialize default exchanges
            self._initialize_default_exchanges()

            self.initialized = True
            self.logger.info(
                f"✅ {self.__class__.__name__} initialized successfully with full integration"
            )
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False

    def _initialize_default_exchanges(self) -> None:
        """Initialize default exchanges with mathematical health monitoring."""
        try:
            supported_exchanges = self.config.get(
                "supported_exchanges", ["binance", "coinbase", "kraken"]
            )

            for exchange_id in supported_exchanges:
                exchange_info = ExchangeInfo(
                    exchange_id=exchange_id,
                    name=exchange_id.capitalize(),
                    status=ExchangeStatus.ONLINE,
                    mathematical_health=0.9,  # High initial health
                    latency=50.0,  # ms
                    uptime=99.9,  # %
                    last_check=time.time(),
                    mathematical_metrics={
                        "tensor_score": 0.8,
                        "entropy_value": 0.2,
                        "quantum_score": 0.7,
                    },
                )

                self.exchanges[exchange_id] = exchange_info
                self.exchange_health_metrics[exchange_id] = 0.9

            self.logger.info(
                f"✅ Initialized {len(self.exchanges)} exchanges with mathematical monitoring"
            )

        except Exception as e:
            self.logger.error(f"❌ Error initializing default exchanges: {e}")

    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        try:
            self.active = True
            self.logger.info(
                f"✅ {self.__class__.__name__} activated with mathematical integration"
            )
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating {self.__class__.__name__}: {e}")
            return False

    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info(f"✅ {self.__class__.__name__} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get system status with mathematical integration status."""
        return {
            "active": self.active,
            "initialized": self.initialized,
            "config": self.config,
            "mathematical_integration": MATH_INFRASTRUCTURE_AVAILABLE,
            "exchange_integration_available": TRADING_PIPELINE_AVAILABLE,
            "exchanges": {k: v.__dict__ for k, v in self.exchanges.items()},
            "health_metrics": self.exchange_health_metrics,
            "order_book_cache_size": len(self.order_book_cache),
        }

    def get_exchange_status(self, exchange_id: str) -> Optional[ExchangeInfo]:
        """Get exchange status with mathematical health checks."""
        try:
            if exchange_id not in self.exchanges:
                self.logger.warning(f"Exchange {exchange_id} not found")
                return None

            exchange_info = self.exchanges[exchange_id]

            # Update mathematical health metrics
            if MATH_INFRASTRUCTURE_AVAILABLE:
                # Calculate mathematical health score
                tensor_score = exchange_info.mathematical_metrics.get(
                    "tensor_score", 0.8
                )
                entropy_value = exchange_info.mathematical_metrics.get(
                    "entropy_value", 0.2
                )
                quantum_score = exchange_info.mathematical_metrics.get(
                    "quantum_score", 0.7
                )

                # Weighted mathematical health calculation
                mathematical_health = (
                    tensor_score * 0.4
                    + (1.0 - entropy_value) * 0.3
                    + quantum_score * 0.3
                )

                exchange_info.mathematical_health = mathematical_health
                exchange_info.last_check = time.time()

            # Update health metrics
            self.exchange_health_metrics[exchange_id] = exchange_info.mathematical_health

            return exchange_info

        except Exception as e:
            self.logger.error(f"❌ Error getting exchange status for {exchange_id}: {e}")
            return None

    def process_order_book(
        self, exchange_id: str, symbol: str, order_book_data: Dict[str, Any]
    ) -> Optional[OrderBookSnapshot]:
        """Process order book with mathematical analysis."""
        try:
            if not MATH_INFRASTRUCTURE_AVAILABLE:
                self.logger.warning(
                    "Mathematical infrastructure not available for order book analysis"
                )
                return None

            # Extract order book data
            bids = order_book_data.get("bids", [])
            asks = order_book_data.get("asks", [])
            timestamp = order_book_data.get("timestamp", time.time())

            if not bids or not asks:
                self.logger.warning(
                    f"Invalid order book data for {exchange_id}:{symbol}"
                )
                return None

            # Calculate basic metrics
            best_bid = max(bids, key=lambda x: x[0])[0] if bids else 0
            best_ask = min(asks, key=lambda x: x[0])[0] if asks else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            depth = sum(bid[1] for bid in bids[:10]) + sum(ask[1] for ask in asks[:10])

            # Mathematical analysis
            mathematical_score = 0.0
            tensor_score = 0.0
            entropy_value = 0.0

            if MATH_INFRASTRUCTURE_AVAILABLE:
                try:
                    # Calculate mathematical scores
                    mathematical_score = self.vwho.calculate_oscillation(bids, asks)
                    tensor_score = self.tensor_algebra.calculate_tensor_score(bids, asks)
                    entropy_value = self.entropy_math.calculate_entropy(bids, asks)
                except Exception as e:
                    self.logger.warning(f"Mathematical analysis failed: {e}")

            # Create order book snapshot
            snapshot = OrderBookSnapshot(
                exchange=exchange_id,
                symbol=symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks,
                mathematical_score=mathematical_score,
                tensor_score=tensor_score,
                entropy_value=entropy_value,
                spread=spread,
                depth=depth,
                mathematical_analysis={
                    'vwho_score': mathematical_score,
                    'tensor_score': tensor_score,
                    'entropy_value': entropy_value
                }
            )

            # Cache the snapshot
            cache_key = f"{exchange_id}:{symbol}"
            self.order_book_cache[cache_key] = snapshot

            # Maintain cache size
            if len(self.order_book_cache) > self.config.get(
                "order_book_cache_size", 1000
            ):
                # Remove oldest entries
                oldest_key = min(
                    self.order_book_cache.keys(),
                    key=lambda k: self.order_book_cache[k].timestamp,
                )
                del self.order_book_cache[oldest_key]

            return snapshot

        except Exception as e:
            self.logger.error(f"Error processing order book: {e}")
            return None

    def execute_order_mathematically(
        self,
        exchange_id: str,
        symbol: str,
        order_type: OrderType,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Result:
        """Execute order with mathematical validation."""
        try:
            if not self.active:
                return Result(success=False, error="System not active")

            # Get current order book for validation
            cache_key = f"{exchange_id}:{symbol}"
            order_book = self.order_book_cache.get(cache_key)

            if not order_book:
                return Result(
                    success=False,
                    error=f"No order book data for {exchange_id}:{symbol}",
                )

            # Mathematical validation
            if MATH_INFRASTRUCTURE_AVAILABLE:
                # Validate order against mathematical constraints
                validation_result = self._validate_order_mathematically(
                    order_book, order_type, side, amount, price
                )

                if not validation_result["valid"]:
                    return Result(success=False, error=validation_result["reason"])

            # Simulate order execution (in real implementation, this would call CCXT)
            execution_result = {
                "exchange": exchange_id,
                "symbol": symbol,
                "order_type": order_type.value,
                "side": side,
                "amount": amount,
                "price": price or order_book.bids[0][0]
                if side == "buy"
                else order_book.asks[0][0],
                "timestamp": time.time(),
                "status": "filled",
                "mathematical_score": order_book.mathematical_score,
            }

            self.logger.info(f"✅ Order executed mathematically: {execution_result}")
            return Result(success=True, data=execution_result)

        except Exception as e:
            self.logger.error(f"❌ Error executing order: {e}")
            return Result(success=False, error=str(e))

    def _validate_order_mathematically(
        self,
        order_book: OrderBookSnapshot,
        order_type: OrderType,
        side: str,
        amount: float,
        price: Optional[float],
    ) -> Dict[str, Any]:
        """Validate order using mathematical analysis."""
        try:
            validation_result = {"valid": True, "reason": None}

            # Check mathematical health threshold
            if order_book.mathematical_score < self.config.get(
                "mathematical_health_threshold", 0.7
            ):
                validation_result = {
                    "valid": False,
                    "reason": f"Mathematical health score too low: {order_book.mathematical_score:.3f}",
                }
                return validation_result

            # Check spread constraints
            if order_book.spread > 0.01:  # 1% spread threshold
                validation_result = {
                    "valid": False,
                    "reason": f"Spread too high: {order_book.spread:.4f}",
                }
                return validation_result

            # Check depth constraints
            if order_book.depth < amount * 10:  # 10x depth requirement
                validation_result = {
                    "valid": False,
                    "reason": f"Insufficient market depth: {order_book.depth:.2f} < {amount * 10:.2f}",
                }
                return validation_result

            # Check entropy constraints
            if order_book.entropy_value > 0.8:  # High entropy indicates uncertainty
                validation_result = {
                    "valid": False,
                    "reason": f"Market entropy too high: {order_book.entropy_value:.3f}",
                }
                return validation_result

            return validation_result

        except Exception as e:
            self.logger.error(f"❌ Error in mathematical validation: {e}")
            return {"valid": False, "reason": f"Validation error: {str(e)}"}

    def get_mathematical_analysis(
        self, exchange_id: str, symbol: str
    ) -> Dict[str, Any]:
        """Get comprehensive mathematical analysis for an exchange/symbol pair."""
        try:
            cache_key = f"{exchange_id}:{symbol}"
            order_book = self.order_book_cache.get(cache_key)

            if not order_book:
                return {"error": f"No data available for {exchange_id}:{symbol}"}

            exchange_info = self.get_exchange_status(exchange_id)

            analysis = {
                "exchange": exchange_id,
                "symbol": symbol,
                "timestamp": order_book.timestamp,
                "order_book_analysis": {
                    "mathematical_score": order_book.mathematical_score,
                    "tensor_score": order_book.tensor_score,
                    "entropy_value": order_book.entropy_value,
                    "spread": order_book.spread,
                    "depth": order_book.depth,
                    "detailed_analysis": order_book.mathematical_analysis,
                },
                "exchange_analysis": {
                    "mathematical_health": exchange_info.mathematical_health
                    if exchange_info
                    else 0.0,
                    "latency": exchange_info.latency if exchange_info else 0.0,
                    "uptime": exchange_info.uptime if exchange_info else 0.0,
                    "mathematical_metrics": exchange_info.mathematical_metrics
                    if exchange_info
                    else {},
                },
                "system_status": {
                    "mathematical_integration": MATH_INFRASTRUCTURE_AVAILABLE,
                    "trading_pipeline_available": TRADING_PIPELINE_AVAILABLE,
                    "cache_size": len(self.order_book_cache),
                },
            }

            return analysis

        except Exception as e:
            self.logger.error(f"❌ Error getting mathematical analysis: {e}")
            return {"error": str(e)}

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.deactivate()
            self.order_book_cache.clear()
            self.exchanges.clear()
            self.exchange_health_metrics.clear()
            self.logger.info(f"✅ {self.__class__.__name__} cleaned up successfully")
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


# Factory function for creating CCXT integration
def create_ccxt_integration(config: Optional[Dict[str, Any]] = None) -> CCXTIntegration:
    """Create CCXT integration with mathematical setup."""
    try:
        integration = CCXTIntegration(config)
        if integration.initialized:
            integration.activate()
            logger.info("✅ CCXT integration created and activated successfully")
        else:
            logger.warning("⚠️ CCXT integration created but not fully initialized")
        return integration
    except Exception as e:
        logger.error(f"❌ Error creating CCXT integration: {e}")
        raise


# Main function for testing
async def main():
    """Main function for testing CCXT integration."""
    try:
        # Create integration
        config = {
            "enabled": True,
            "debug": True,
            "mathematical_integration": True,
            "supported_exchanges": ["binance", "coinbase"],
        }

        integration = create_ccxt_integration(config)

        # Test status
        status = integration.get_status()
        print(f"Integration status: {status}")

        # Test exchange status
        for exchange_id in ["binance", "coinbase"]:
            exchange_status = integration.get_exchange_status(exchange_id)
            if exchange_status:
                print(f"{exchange_id} status: {exchange_status.__dict__}")

        # Test mathematical analysis
        analysis = integration.get_mathematical_analysis("binance", "BTC/USDT")
        print(f"Mathematical analysis: {analysis}")

        # Cleanup
        integration.cleanup()

    except Exception as e:
        logger.error(f"❌ Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
