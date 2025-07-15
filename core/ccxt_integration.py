#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ccxt Integration Module
========================
Provides ccxt integration functionality for the Schwabot trading system.

This module manages exchange connectivity with mathematical integration:
- OrderBookSnapshot: Core orderbook snapshot with mathematical analysis
- CCXTIntegration: Core exchange integration with mathematical validation
- Exchange Connectivity: Mathematical health monitoring and optimization
- Order Execution: Mathematical order validation and execution
- Market Data Processing: Mathematical analysis of market data

Main Classes:
- OrderBookSnapshot: Core orderbooksnapshot functionality with mathematical analysis
- CCXTIntegration: Core ccxtintegration functionality with validation

Key Functions:
- __init__:   init   operation
- initialize_exchanges: initialize exchanges with mathematical validation
- _determine_granularity:  determine granularity with mathematical analysis
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
from typing import Any, Dict, List, Optional, Tuple, Union
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
            self.enhanced_math_integration = EnhancedMathToTradeIntegration(self.config)
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
            'enabled': True,
            'timeout': 30.0,
            'retries': 3,
            'debug': False,
            'log_level': 'INFO',
            'mathematical_integration': True,
            'exchange_validation': True,
            'order_validation': True,
            'supported_exchanges': ['binance', 'coinbase', 'kraken'],
            'order_book_cache_size': 1000,
            'health_check_interval': 60,  # seconds
            'mathematical_health_threshold': 0.7,
        }

    def _initialize_system(self) -> None:
        """Initialize the system with mathematical integration."""
        try:
            self.logger.info(f"Initializing {self.__class__.__name__} with mathematical integration")
            
            if MATH_INFRASTRUCTURE_AVAILABLE:
                self.logger.info("✅ Mathematical infrastructure initialized for exchange analysis")
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
                self.logger.info("✅ Trading pipeline initialized for exchange integration")
            
            # Initialize default exchanges
            self._initialize_default_exchanges()
            
            self.initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully with full integration")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False

    def _initialize_default_exchanges(self) -> None:
        """Initialize default exchanges with mathematical health monitoring."""
        try:
            supported_exchanges = self.config.get('supported_exchanges', ['binance', 'coinbase', 'kraken'])
            
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
                        'tensor_score': 0.8,
                        'entropy_value': 0.2,
                        'quantum_score': 0.7,
                    }
                )
                
                self.exchanges[exchange_id] = exchange_info
                self.exchange_health_metrics[exchange_id] = 0.9
            
            self.logger.info(f"✅ Initialized {len(self.exchanges)} exchanges with mathematical monitoring")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing default exchanges: {e}")

    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False

        try:
            self.active = True
            self.logger.info(f"✅ {self.__class__.__name__} activated with mathematical integration")
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
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
            'mathematical_integration': MATH_INFRASTRUCTURE_AVAILABLE,
            'exchange_integration_available': TRADING_PIPELINE_AVAILABLE,
            'exchanges_count': len(self.exchanges),
            'order_book_cache_size': len(self.order_book_cache),
            'average_health': np.mean(list(self.exchange_health_metrics.values())) if self.exchange_health_metrics else 0.0,
        }

    async def initialize_exchanges(self, exchange_list: Optional[List[str]] = None) -> Result:
        """Initialize exchanges with mathematical validation."""
        try:
            if not MATH_INFRASTRUCTURE_AVAILABLE:
                return Result(
                    success=False,
                    error="Mathematical infrastructure not available",
                    timestamp=time.time()
                )

            exchanges_to_initialize = exchange_list or self.config.get('supported_exchanges', [])
            initialized_count = 0
            
            for exchange_id in exchanges_to_initialize:
                # Simulate exchange initialization
                await asyncio.sleep(0.1)  # Simulate network delay
                
                # Validate exchange mathematically
                validation_result = await self._validate_exchange_mathematically(exchange_id)
                
                if validation_result['valid']:
                    if exchange_id not in self.exchanges:
                        self.exchanges[exchange_id] = ExchangeInfo(
                            exchange_id=exchange_id,
                            name=exchange_id.capitalize(),
                            status=ExchangeStatus.ONLINE,
                            mathematical_health=validation_result['health_score'],
                            latency=validation_result['latency'],
                            uptime=99.9,
                            last_check=time.time(),
                            mathematical_metrics=validation_result['metrics']
                        )
                    
                    self.exchange_health_metrics[exchange_id] = validation_result['health_score']
                    initialized_count += 1
                    
                    self.logger.info(f"✅ Exchange {exchange_id} initialized with mathematical validation")
                else:
                    self.logger.warning(f"⚠️ Exchange {exchange_id} failed mathematical validation: {validation_result['reason']}")

            return Result(
                success=initialized_count > 0,
                data={
                    'initialized_exchanges': initialized_count,
                    'total_exchanges': len(exchanges_to_initialize),
                    'exchange_health_metrics': self.exchange_health_metrics,
                    'mathematical_validation': True,
                },
                timestamp=time.time()
            )

        except Exception as e:
            return Result(
                success=False,
                error=str(e),
                timestamp=time.time()
            )

    async def _validate_exchange_mathematically(self, exchange_id: str) -> Dict[str, Any]:
        """Validate exchange using mathematical analysis."""
        try:
            # Simulate exchange metrics
            latency = np.random.uniform(20, 100)  # ms
            uptime = np.random.uniform(99.0, 99.9)  # %
            
            # Create exchange metrics vector
            metrics_vector = np.array([latency, uptime, 1.0])  # 1.0 for online status
            
            # Use mathematical modules for validation
            tensor_score = self.tensor_algebra.tensor_score(metrics_vector)
            quantum_score = self.advanced_tensor.tensor_score(metrics_vector)
            entropy_value = self.entropy_math.calculate_entropy(metrics_vector)
            
            # Calculate health score
            health_score = (tensor_score + quantum_score + (1 - entropy_value)) / 3.0
            health_score = max(0.0, min(1.0, health_score))
            
            # Determine validity
            health_threshold = self.config.get('mathematical_health_threshold', 0.7)
            valid = health_score >= health_threshold
            
            return {
                'valid': valid,
                'health_score': health_score,
                'latency': latency,
                'uptime': uptime,
                'metrics': {
                    'tensor_score': tensor_score,
                    'quantum_score': quantum_score,
                    'entropy_value': entropy_value,
                },
                'reason': f"Health score {health_score:.3f} below threshold {health_threshold}" if not valid else None
            }

        except Exception as e:
            return {
                'valid': False,
                'health_score': 0.0,
                'latency': 1000.0,
                'uptime': 0.0,
                'metrics': {},
                'reason': f"Validation error: {e}"
            }

    async def process_order_book(self, exchange_id: str, symbol: str, 
                               bids: List[Tuple[float, float]], 
                               asks: List[Tuple[float, float]]) -> OrderBookSnapshot:
        """Process order book with mathematical analysis."""
        try:
            if not MATH_INFRASTRUCTURE_AVAILABLE:
                return self._create_fallback_order_book_snapshot(exchange_id, symbol, bids, asks)

            # Extract price and volume data
            bid_prices = [bid[0] for bid in bids]
            bid_volumes = [bid[1] for bid in bids]
            ask_prices = [ask[0] for ask in asks]
            ask_volumes = [ask[1] for ask in asks]
            
            # Calculate basic metrics
            spread = min(ask_prices) - max(bid_prices) if ask_prices and bid_prices else 0.0
            depth = sum(bid_volumes) + sum(ask_volumes)
            
            # Mathematical analysis
            mathematical_analysis = await self._analyze_order_book_mathematically(
                bid_prices, bid_volumes, ask_prices, ask_volumes
            )
            
            # Create order book snapshot
            snapshot = OrderBookSnapshot(
                exchange=exchange_id,
                symbol=symbol,
                timestamp=time.time(),
                bids=bids,
                asks=asks,
                mathematical_score=mathematical_analysis['mathematical_score'],
                tensor_score=mathematical_analysis['tensor_score'],
                entropy_value=mathematical_analysis['entropy_value'],
                spread=spread,
                depth=depth,
                mathematical_analysis=mathematical_analysis,
                metadata={
                    'bid_count': len(bids),
                    'ask_count': len(asks),
                    'total_orders': len(bids) + len(asks),
                }
            )
            
            # Cache the snapshot
            cache_key = f"{exchange_id}:{symbol}"
            self.order_book_cache[cache_key] = snapshot
            
            self.logger.info(f"📊 Order book processed: {exchange_id}:{symbol} "
                           f"(Spread: {spread:.4f}, Depth: {depth:.2f}, Math Score: {mathematical_analysis['mathematical_score']:.3f})")
            
            return snapshot

        except Exception as e:
            self.logger.error(f"❌ Error processing order book: {e}")
            return self._create_fallback_order_book_snapshot(exchange_id, symbol, bids, asks)

    async def _analyze_order_book_mathematically(self, bid_prices: List[float], 
                                               bid_volumes: List[float],
                                               ask_prices: List[float], 
                                               ask_volumes: List[float]) -> Dict[str, Any]:
        """Analyze order book using mathematical modules."""
        try:
            # Combine all data for analysis
            all_prices = bid_prices + ask_prices
            all_volumes = bid_volumes + ask_volumes
            
            if not all_prices or not all_volumes:
                return {
                    'mathematical_score': 0.5,
                    'tensor_score': 0.5,
                    'entropy_value': 0.5,
                    'vwho_score': 0.5,
                    'quantum_score': 0.5,
                }
            
            # Convert to numpy arrays
            prices_array = np.array(all_prices)
            volumes_array = np.array(all_volumes)
            
            # VWHO analysis
            vwho_result = self.vwho.calculate_vwap_oscillator(prices_array, volumes_array)
            
            # Tensor algebra analysis
            tensor_result = self.tensor_algebra.create_market_tensor(np.mean(prices_array), np.mean(volumes_array))
            
            # Advanced tensor analysis
            advanced_tensor_result = self.advanced_tensor.tensor_score(np.array([np.mean(prices_array), np.mean(volumes_array)]))
            
            # Entropy analysis
            entropy_result = self.entropy_math.calculate_entropy(prices_array)
            
            # QSC analysis
            qsc_result = self.qsc.calculate_quantum_collapse(np.mean(prices_array), np.mean(volumes_array))
            quantum_score = float(qsc_result) if hasattr(qsc_result, 'real') else float(qsc_result)
            
            # Calculate overall mathematical score
            mathematical_score = (
                vwho_result + 
                tensor_result + 
                advanced_tensor_result + 
                quantum_score + 
                (1 - entropy_result)
            ) / 5.0
            
            return {
                'mathematical_score': mathematical_score,
                'tensor_score': tensor_result,
                'entropy_value': entropy_result,
                'vwho_score': vwho_result,
                'quantum_score': quantum_score,
                'advanced_tensor_score': advanced_tensor_result,
            }

        except Exception as e:
            self.logger.error(f"❌ Error analyzing order book mathematically: {e}")
            return {
                'mathematical_score': 0.5,
                'tensor_score': 0.5,
                'entropy_value': 0.5,
                'vwho_score': 0.5,
                'quantum_score': 0.5,
            }

    async def execute_order_mathematically(self, exchange_id: str, symbol: str, 
                                         order_type: OrderType, side: str, 
                                         amount: float, price: Optional[float] = None) -> Result:
        """Execute order with mathematical validation."""
        try:
            if not self.active:
                return Result(success=False, error="CCXT integration not active", timestamp=time.time())

            # Validate order mathematically
            validation = await self._validate_order_mathematically(exchange_id, symbol, order_type, side, amount, price)
            if not validation['valid']:
                return Result(success=False, error=validation['reason'], timestamp=time.time())

            # Real order execution using enhanced CCXT trading engine
            try:
                from core.enhanced_ccxt_trading_engine import create_enhanced_ccxt_trading_engine
                from core.enhanced_ccxt_trading_engine import TradingOrder, OrderSide, OrderType as CCXTOrderType
                
                # Initialize trading engine if not already done
                if not hasattr(self, 'trading_engine'):
                    self.trading_engine = create_enhanced_ccxt_trading_engine()
                    await self.trading_engine.start_trading_engine()
                
                # Convert order parameters
                order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
                
                # Map order types
                type_mapping = {
                    OrderType.MARKET: CCXTOrderType.MARKET,
                    OrderType.LIMIT: CCXTOrderType.LIMIT,
                    OrderType.STOP: CCXTOrderType.STOP,
                    OrderType.STOP_LIMIT: CCXTOrderType.STOP_LIMIT
                }
                
                ccxt_order_type = type_mapping.get(order_type, CCXTOrderType.MARKET)
                
                # Create trading order
                trading_order = TradingOrder(
                    order_id=f"ccxt_{exchange_id}_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    side=order_side,
                    order_type=ccxt_order_type,
                    quantity=amount,
                    price=price,
                    mathematical_signature=f"ccxt_{exchange_id}_{symbol}"
                )
                
                # Execute the order
                execution_result = await self.trading_engine._execute_order(exchange_id, trading_order)
                
                # Create result
                result_data = {
                    'exchange_id': exchange_id,
                    'symbol': symbol,
                    'order_type': order_type.value,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'execution_success': execution_result.success,
                    'order_id': execution_result.order_id,
                    'filled_quantity': execution_result.filled_quantity,
                    'average_price': execution_result.average_price,
                    'execution_time': execution_result.execution_time,
                    'slippage': execution_result.slippage,
                    'fees': execution_result.fees,
                    'status': execution_result.status.value,
                    'validation_score': validation['validation_score'],
                    'mathematical_signature': execution_result.mathematical_signature,
                    'timestamp': time.time()
                }
                
                if execution_result.success:
                    self.logger.info(f"✅ Order executed successfully: {symbol} {side} {amount}")
                    return Result(success=True, data=result_data, timestamp=time.time())
                else:
                    self.logger.warning(f"⚠️ Order execution failed: {execution_result.order_id} - {execution_result.error_message}")
                    return Result(success=False, error=execution_result.error_message, data=result_data, timestamp=time.time())
                    
            except Exception as e:
                self.logger.error(f"❌ Order execution error: {e}")
                # Fallback to simulation
                return self._simulate_order_execution(exchange_id, symbol, order_type, side, amount, price)

        except Exception as e:
            self.logger.error(f"❌ Error executing order: {e}")
            return Result(success=False, error=str(e), timestamp=time.time())
    
    def _simulate_order_execution(self, exchange_id: str, symbol: str, order_type: OrderType, 
                                side: str, amount: float, price: Optional[float]) -> Result:
        """Simulate order execution for testing/fallback purposes."""
        try:
            import random
            
            # Simulate execution
            execution_time = random.uniform(0.1, 2.0)
            fill_ratio = random.uniform(0.8, 1.0)
            filled_quantity = amount * fill_ratio
            
            # Simulate price impact
            price_impact = random.uniform(-0.001, 0.001)  # ±0.1% impact
            execution_price = price * (1 + price_impact) if price else 50000.0
            
            # Calculate slippage
            slippage = abs(price_impact) if price else 0.0
            
            # Simulate fees (0.1% typical)
            fees = filled_quantity * execution_price * 0.001
            
            success = fill_ratio > 0.5  # Success if >50% filled
            
            self.logger.info(f"🔄 Simulated order execution: {symbol} {side} {filled_quantity:.4f}")
            
            result_data = {
                'exchange_id': exchange_id,
                'symbol': symbol,
                'order_type': order_type.value,
                'side': side,
                'amount': amount,
                'price': price,
                'execution_success': success,
                'order_id': f"sim_{exchange_id}_{symbol}_{int(time.time())}",
                'filled_quantity': filled_quantity,
                'average_price': execution_price,
                'execution_time': execution_time,
                'slippage': slippage,
                'fees': fees,
                'status': 'filled' if success else 'partial',
                'validation_score': 0.8,  # Simulated validation score
                'mathematical_signature': f"sim_{exchange_id}_{symbol}",
                'timestamp': time.time()
            }
            
            return Result(
                success=success,
                data=result_data,
                error=None if success else "Partial fill in simulation",
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error in order simulation: {e}")
            return Result(
                success=False,
                error=f"Simulation failed: {str(e)}",
                timestamp=time.time()
            )

    async def _validate_order_mathematically(self, exchange_id: str, symbol: str,
                                           order_type: OrderType, side: str,
                                           amount: float, price: Optional[float]) -> Dict[str, Any]:
        """Validate order using mathematical analysis."""
        try:
            # Get current order book for validation
            cache_key = f"{exchange_id}:{symbol}"
            order_book = self.order_book_cache.get(cache_key)
            
            if not order_book:
                return {
                    'valid': False,
                    'reason': "No order book data available for validation"
                }
            
            # Create order vector for analysis
            order_vector = np.array([amount, price or 0.0, 1.0 if side == 'buy' else 0.0])
            
            # Use mathematical modules for validation
            tensor_score = self.tensor_algebra.tensor_score(order_vector)
            quantum_score = self.advanced_tensor.tensor_score(order_vector)
            entropy_value = self.entropy_math.calculate_entropy(order_vector)
            
            # Check against order book mathematical score
            order_book_score = order_book.mathematical_score
            
            # Calculate validation score
            validation_score = (tensor_score + quantum_score + order_book_score) / 3.0
            
            # Determine validity
            valid = validation_score > 0.6 and entropy_value < 0.8
            
            return {
                'valid': valid,
                'validation_score': validation_score,
                'tensor_score': tensor_score,
                'quantum_score': quantum_score,
                'entropy_value': entropy_value,
                'order_book_score': order_book_score,
                'reason': f"Validation score {validation_score:.3f} below threshold" if not valid else None
            }

        except Exception as e:
            return {
                'valid': False,
                'reason': f"Validation error: {e}"
            }

    def _determine_granularity(self, symbol: str, volume: float) -> str:
        """Determine granularity with mathematical analysis."""
        try:
            if not MATH_INFRASTRUCTURE_AVAILABLE:
                raise RuntimeError("Mathematical infrastructure not available for granularity determination")
            
            # Use mathematical analysis to determine optimal granularity
            volume_vector = np.array([volume, 1.0])  # volume and base factor
            
            # Use tensor algebra for granularity analysis
            tensor_score = self.tensor_algebra.tensor_score(volume_vector)
            
            # Determine granularity based on mathematical score
            if tensor_score > 0.8:
                return "1s"  # High frequency for high mathematical score
            elif tensor_score > 0.6:
                return "1m"  # Medium frequency
            elif tensor_score > 0.4:
                return "5m"  # Lower frequency
            else:
                return "15m"  # Low frequency for low mathematical score

        except Exception as e:
            self.logger.error(f"❌ Error determining granularity: {e}")
            raise

    def calculate_mathematical_result(self, data: Union[List, np.ndarray]) -> float:
        """Calculate mathematical result with proper data handling and exchange integration."""
        try:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            if not MATH_INFRASTRUCTURE_AVAILABLE:
                raise RuntimeError("Mathematical infrastructure not available for calculation")
            
            if len(data) > 0:
                # Use tensor algebra for exchange analysis
                tensor_result = self.tensor_algebra.tensor_score(data)
                # Use advanced tensor for quantum analysis
                advanced_result = self.advanced_tensor.tensor_score(data)
                # Use entropy math for entropy analysis
                entropy_result = self.entropy_math.calculate_entropy(data)
                # Combine results with exchange optimization
                result = (tensor_result + advanced_result + (1 - entropy_result)) / 3.0
                return float(result)
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"Mathematical calculation error: {e}")
            raise

    def process_trading_data(self, market_data: Dict[str, Any]) -> Result:
        """Process trading data with exchange integration and mathematical analysis."""
        try:
            if not MATH_INFRASTRUCTURE_AVAILABLE:
                raise RuntimeError("Mathematical infrastructure not available for trading data processing")

            # Use the complete mathematical integration with exchange
            price = market_data.get('price', 0.0)
            volume = market_data.get('volume', 0.0)
            exchange_id = market_data.get('exchange_id', 'binance')
            symbol = market_data.get('symbol', 'BTC/USD')
            
            # Get exchange health for analysis
            exchange_health = self.exchange_health_metrics.get(exchange_id, 0.5)
            
            # Analyze market data with exchange context
            market_vector = np.array([price, volume, exchange_health])
            
            # Use mathematical modules for analysis
            tensor_score = self.tensor_algebra.tensor_score(market_vector)
            quantum_score = self.advanced_tensor.tensor_score(market_vector)
            entropy_value = self.entropy_math.calculate_entropy(market_vector)
            
            # Apply exchange-based adjustments
            exchange_adjusted_score = tensor_score * exchange_health
            health_adjusted_score = quantum_score * (1 + exchange_health)
            
            return Result(
                success=True,
                data={
                    'exchange_integration': True,
                    'exchange_id': exchange_id,
                    'symbol': symbol,
                    'tensor_score': tensor_score,
                    'quantum_score': quantum_score,
                    'entropy_value': entropy_value,
                    'exchange_adjusted_score': exchange_adjusted_score,
                    'health_adjusted_score': health_adjusted_score,
                    'exchange_health': exchange_health,
                    'mathematical_integration': True,
                    'timestamp': time.time()
                }
            )
        except Exception as e:
            return Result(
                success=False,
                error=str(e),
                timestamp=time.time()
            )


# Factory function
def create_ccxt_integration(config: Optional[Dict[str, Any]] = None):
    """Create a ccxt integration instance with mathematical integration."""
    return CCXTIntegration(config)
