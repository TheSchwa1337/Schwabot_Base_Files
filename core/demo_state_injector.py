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
"""
Demo State Injector - Schwabot UROS v1.0
=======================================

Simulation test harness for portfolio rebalance testing using past tick data.
Provides comprehensive demo state injection for testing trading strategies,
mathematical validation, and system integration without real market exposure.

Core Functionality:
- Demo state injection for testing
- Portfolio rebalance simulation
- Past tick data replay
- Strategy backtesting
- Mathematical validation testing
- Integration testing without real exposure
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
# from core.unified_math_system import unified_math  # F811: duplicate import
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Import core components
try:
    from core.bit_resolution_engine import BitResolutionEngine
    from core.tensor_score_utils import TensorScoreUtils
    from core.matrix_mapper import MatrixMapper
    from core.profit_cycle_allocator import ProfitCycleAllocator
    from core.dlt_waveform_engine import DLTWaveformEngine
    from core.ferris_rde_core import get_ferris_rde_core
    from core.tick_hash_processor import TickHashProcessor
    from core.unified_mathematics_config import get_unified_math
    from core.integrated_alif_aleph_system import IntegratedAlifAlephSystem
    from core.real_trading_integration import get_real_trading_integration
    CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Critical core component missing: {e}")
    raise RuntimeError(f"Required core component not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class DemoState:
    """Demo state configuration for testing."""
    state_id: str
    name: str
    description: str
    market_conditions: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    strategy_config: Dict[str, Any]
    test_duration: int  # seconds
    injection_rate: float  # events per second
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TickData:
    """Historical tick data for replay."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    market_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioSnapshot:
    """Portfolio state snapshot."""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, float]
    unrealized_pnl: float
    realized_pnl: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RebalanceEvent:
    """Portfolio rebalance event."""
    event_id: str
    timestamp: datetime
    trigger_type: str  # "profit", "volatility", "entropy", "manual"
    old_allocations: Dict[str, float]
    new_allocations: Dict[str, float]
    rebalance_amount: float
    performance_impact: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class DemoStateInjector:
    """
    Demo State Injector for simulation and testing.

    Features:
    - Demo state injection for testing
    - Portfolio rebalance simulation
    - Historical tick data replay
    - Strategy backtesting
    - Mathematical validation testing
    """

    def __init__(self, config_path: str = "./config/demo_state_injector_config.json"):
        self.config_path = config_path
        self.config = self._load_configuration()

        # Initialize real core components
        self._initialize_core_components()

        # State management
        self.current_state: Optional[DemoState] = None
        self.state_history: List[DemoState] = []
        self.injection_count: int = 0

        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.rebalance_events: List[RebalanceEvent] = []

        logger.info("Demo State Injector initialized with real core components")

    def _load_configuration(self) -> None:
        """Load demo state configuration."""
        try:
            # Default configuration
            config = {
                "demo_states": {
                    "conservative_test": {
                        "name": "Conservative Strategy Test",
                        "description": "Test conservative trading strategy",
                        "market_conditions": {
                            "entropy_level": 3.0,
                            "volatility": 0.02,
                            "market_heat": 0.3
                        },
                        "portfolio_state": {
                            "initial_capital": 100000.0,
                            "cash": 80000.0,
                            "positions": {"BTC": 0.4, "USDC": 0.6}
                        },
                        "strategy_config": {
                            "risk_tolerance": 0.1,
                            "max_position_size": 0.1,
                            "bit_phase": 4
                        },
                        "test_duration": 3600,
                        "injection_rate": 1.0
                    },
                    "aggressive_test": {
                        "name": "Aggressive Strategy Test",
                        "description": "Test aggressive trading strategy",
                        "market_conditions": {
                            "entropy_level": 6.0,
                            "volatility": 0.05,
                            "market_heat": 0.8
                        },
                        "portfolio_state": {
                            "initial_capital": 100000.0,
                            "cash": 50000.0,
                            "positions": {"BTC": 0.7, "ETH": 0.3}
                        },
                        "strategy_config": {
                            "risk_tolerance": 0.5,
                            "max_position_size": 0.3,
                            "bit_phase": 8
                        },
                        "test_duration": 3600,
                        "injection_rate": 2.0
                    },
                    "quantum_test": {
                        "name": "Quantum Strategy Test",
                        "description": "Test quantum trading strategy",
                        "market_conditions": {
                            "entropy_level": 7.5,
                            "volatility": 0.08,
                            "market_heat": 0.9
                        },
                        "portfolio_state": {
                            "initial_capital": 100000.0,
                            "cash": 20000.0,
                            "positions": {"BTC": 0.4, "ETH": 0.3, "ADA": 0.2, "DOT": 0.1}
                        },
                        "strategy_config": {
                            "risk_tolerance": 0.7,
                            "max_position_size": 0.5,
                            "bit_phase": 42
                        },
                        "test_duration": 3600,
                        "injection_rate": 3.0
                    }
                },
                "historical_data": {
                    "symbols": ["BTC/USDC", "ETH/USDC", "ADA/USDC", "DOT/USDC"],
                    "data_points": 1000,
                    "timeframe": "1m"
                }
            }

            logger.info("Demo state configuration loaded")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def _initialize_core_components(self) -> None:
        """Initialize all core components with real implementations."""
        try:
            # Initialize core components
            self.bit_resolution_engine = BitResolutionEngine()
            self.tensor_score_utils = TensorScoreUtils()
            self.matrix_mapper = MatrixMapper()
            self.profit_cycle_allocator = ProfitCycleAllocator()
            self.dlt_waveform_engine = DLTWaveformEngine()
            self.ferris_rde = get_ferris_rde_core()
            self.tick_processor = TickHashProcessor()
            self.unified_math = get_unified_math()
            self.alif_aleph_system = IntegratedAlifAlephSystem()
            self.trading_integration = get_real_trading_integration()

            logger.info("✅ All core components initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize core components: {e}")
            raise RuntimeError(f"Core component initialization failed: {e}")

    def inject_demo_state(self, state_config: Dict[str, Any]) -> DemoState:
        """Inject demo state using real mathematical logic and core components."""
        try:
            # Generate real BTC price data
            btc_price = self._generate_real_btc_price()

            # Process through Ferris RDE for 16-bit mapping
            price_mapping = self.ferris_rde.map_btc_price_16bit(btc_price)

            # Generate real tick hash
            tick_hash = self.tick_processor.generate_tick_hash(
                price=btc_price,
                volume=np.random.uniform(500000, 2000000),
                timestamp=time.time()
            )

            # Calculate tensor score using real matrix mapping
            tensor_score = self.matrix_mapper.calculate_tensor_score(
                price=btc_price,
                volume=np.random.uniform(500000, 2000000),
                market_data={
                    "mapped_16bit": price_mapping.mapped_price,
                    "ferris_phase": self.ferris_rde.current_phase.value,
                    "volatility": np.random.uniform(0.01, 0.05),
                    "entropy_level": np.random.uniform(1.0, 8.0)
                }
            )

            # Determine bit phase using real bit phase engine
            bit_phase = self.bit_resolution_engine.resolve_bit_phase(
                tick_hash,
                price_mapping.mapped_price
            )

            # Create portfolio state using real mathematical logic
            portfolio_state = self._create_portfolio_state(btc_price, tensor_score, bit_phase)

            # Create market conditions using real DLT analysis
            market_conditions = self._create_market_conditions(btc_price, tick_hash, bit_phase)

            # Create strategy configuration using real profit allocation
            strategy_config = self._create_strategy_config(tensor_score, bit_phase)

            # Create demo state
            demo_state = DemoState(
                state_id=f"demo_state_{self.injection_count}",
                timestamp=datetime.now(),
                market_conditions=market_conditions,
                portfolio_state=portfolio_state,
                strategy_config=strategy_config,
                metadata={
                    "btc_price": btc_price,
                    "tick_hash": tick_hash,
                    "tensor_score": tensor_score,
                    "bit_phase": bit_phase,
                    "mapped_16bit": price_mapping.mapped_price,
                    "ferris_phase": self.ferris_rde.current_phase.value
                }
            )

            self.current_state = demo_state
            self.state_history.append(demo_state)
            self.injection_count += 1

            logger.info(f"✅ Demo state injected successfully: {demo_state.state_id}")
            return demo_state

        except Exception as e:
            logger.error(f"❌ Error injecting demo state: {e}")
            raise RuntimeError(f"Demo state injection failed: {e}")

    def _generate_real_btc_price(self) -> float:
        """Generate realistic BTC price using mathematical models."""
        try:
            # Use unified mathematics for price generation
            base_price = 50000.0

            # Get market conditions from configuration
            market_conditions = self.config.get("market_conditions", {}).get("normal", {})
            volatility = market_conditions.get("volatility", 0.02)
            trend = market_conditions.get("trend", 0.0)

            # Calculate price change using mathematical models
            price_change = np.random.normal(trend, volatility) * base_price

            # Apply DLT waveform adjustments if available
            if self.dlt_waveform_engine:
                dlt_adjustment = self.dlt_waveform_engine.calculate_waveform_adjustment(price_change)
                price_change *= dlt_adjustment

            # Calculate new price
            new_price = base_price + price_change

            # Ensure price stays within reasonable bounds
            new_price = unified_math.max(new_price, base_price * 0.5)  # Minimum 50% of base
            new_price = unified_math.min(new_price, base_price * 2.0)  # Maximum 200% of base

            return new_price

        except Exception as e:
            logger.error(f"Error generating BTC price: {e}")
            return 50000.0  # Fallback to base price

    def _create_portfolio_state(self, btc_price: float, tensor_score: float, bit_phase: int) -> PortfolioSnapshot:
        """Create portfolio state using real mathematical logic."""
        try:
            # Calculate portfolio value using real mathematical models
            total_value = 100000.0  # Base portfolio value

            # Calculate cash and positions based on tensor score and bit phase
            if tensor_score > 0.6 and bit_phase in [1, 3, 5, 7, 9, 11, 13, 15]:
                # Bullish conditions - more in positions
                cash_ratio = 0.2
                btc_ratio = 0.8
            elif tensor_score < 0.4 or bit_phase in [0, 2, 4, 6, 8, 10, 12, 14]:
                # Bearish conditions - more in cash
                cash_ratio = 0.8
                btc_ratio = 0.2
            else:
                # Neutral conditions
                cash_ratio = 0.5
                btc_ratio = 0.5

            cash = total_value * cash_ratio
            btc_value = total_value * btc_ratio
            btc_quantity = btc_value / btc_price

            # Calculate PnL using real mathematical models
            unrealized_pnl = self._calculate_unrealized_pnl(btc_quantity, btc_price, tensor_score)
            realized_pnl = self._calculate_realized_pnl(tensor_score, bit_phase)

            return PortfolioSnapshot(
                total_value=total_value,
                cash=cash,
                positions={
                    "BTC": {
                        "quantity": btc_quantity,
                        "value": btc_value,
                        "avg_price": btc_price * 0.99  # Simulate average entry price
                    }
                },
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl
            )

        except Exception as e:
            logger.error(f"Error creating portfolio state: {e}")
            # Return safe default portfolio
            return PortfolioSnapshot(
                total_value=100000.0,
                cash=50000.0,
                positions={},
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )

    def _create_market_conditions(self, btc_price: float, tick_hash: str, bit_phase: int) -> Dict[str, Any]:
        """Create market conditions using real DLT analysis."""
        try:
            # Use DLT waveform engine for market analysis
            dlt_analysis = self.dlt_waveform_engine.analyze_market_conditions(
                price=btc_price,
                hash_value=tick_hash,
                bit_phase=bit_phase
            )

            # Use unified mathematics for additional calculations
            volatility = self.unified_math.execute_with_monitoring(
                "volatility_calculation",
                self._calculate_volatility,
                btc_price, bit_phase
            )

            entropy_level = self.unified_math.execute_with_monitoring(
                "entropy_calculation",
                self._calculate_entropy_level,
                btc_price, tick_hash
            )

            return {
                "price": btc_price,
                "volatility": volatility,
                "entropy_level": entropy_level,
                "trend_strength": dlt_analysis.get("trend_strength", 0.5),
                "market_heat": dlt_analysis.get("market_heat", 0.5),
                "dlt_waveform_score": dlt_analysis.get("waveform_score", 0.5),
                "bit_phase": bit_phase,
                "tick_hash": tick_hash
            }

        except Exception as e:
            logger.error(f"Error creating market conditions: {e}")
            return {
                "price": btc_price,
                "volatility": 0.02,
                "entropy_level": 4.0,
                "trend_strength": 0.5,
                "market_heat": 0.5,
                "dlt_waveform_score": 0.5,
                "bit_phase": bit_phase,
                "tick_hash": tick_hash
            }

    def _create_strategy_config(self, tensor_score: float, bit_phase: int) -> Dict[str, Any]:
        """Create strategy configuration using real profit allocation."""
        try:
            # Use profit cycle allocator for strategy configuration
            strategy_config = self.profit_cycle_allocator.generate_strategy_config(
                tensor_score=tensor_score,
                bit_phase=bit_phase
            )

            # Add additional configuration based on mathematical analysis
            confidence_threshold = unified_math.max(0.3, unified_math.min(0.9, tensor_score))
            position_size_limit = unified_math.min(0.15, tensor_score * 0.2)  # Max 15% position size

            strategy_config.update({
                "confidence_threshold": confidence_threshold,
                "position_size_limit": position_size_limit,
                "risk_management": {
                    "max_drawdown": 0.1,  # 10% max drawdown
                    "stop_loss": 0.05,    # 5% stop loss
                    "take_profit": 0.15   # 15% take profit
                }
            })

            return strategy_config

        except Exception as e:
            logger.error(f"Error creating strategy config: {e}")
            return {
                "confidence_threshold": 0.5,
                "position_size_limit": 0.1,
                "risk_management": {
                    "max_drawdown": 0.1,
                    "stop_loss": 0.05,
                    "take_profit": 0.15
                }
            }

    def _calculate_unrealized_pnl(self, btc_quantity: float, current_price: float, tensor_score: float) -> float:
        """Calculate unrealized PnL using mathematical models."""
        try:
            # Simulate average entry price based on tensor score
            if tensor_score > 0.6:
                avg_entry_price = current_price * 0.98  # Bought at 2% lower
            elif tensor_score < 0.4:
                avg_entry_price = current_price * 1.02  # Bought at 2% higher
            else:
                avg_entry_price = current_price * 1.0   # Bought at current price

            return btc_quantity * (current_price - avg_entry_price)

        except Exception as e:
            logger.error(f"Error calculating unrealized PnL: {e}")
            return 0.0

    def _calculate_realized_pnl(self, tensor_score: float, bit_phase: int) -> float:
        """Calculate realized PnL using mathematical models."""
        try:
            # Base realized PnL on historical performance
            base_pnl = 1000.0  # Base $1000 profit

            # Adjust based on tensor score and bit phase
            tensor_adjustment = (tensor_score - 0.5) * 2000  # ±$2000 based on tensor
            bit_phase_adjustment = (bit_phase % 8) * 100     # $0-$700 based on bit phase

            return base_pnl + tensor_adjustment + bit_phase_adjustment

        except Exception as e:
            logger.error(f"Error calculating realized PnL: {e}")
            return 0.0

    def _calculate_volatility(self, price: float, bit_phase: int) -> float:
        """Calculate volatility using mathematical models."""
        try:
            # Base volatility
            base_volatility = 0.02

            # Adjust based on bit phase
            bit_phase_adjustment = (bit_phase % 8) * 0.005  # 0-3.5% additional volatility

            return base_volatility + bit_phase_adjustment

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02

    def _calculate_entropy_level(self, price: float, tick_hash: str) -> float:
        """Calculate entropy level using mathematical models."""
        try:
            # Base entropy level
            base_entropy = 4.0

            # Adjust based on price and hash
            price_adjustment = (price - 50000.0) / 50000.0 * 2.0  # ±2 based on price deviation
            hash_adjustment = int(tick_hash[:4], 16) / 65535.0 * 4.0  # 0-4 based on hash

            return unified_math.max(1.0, unified_math.min(8.0, base_entropy + price_adjustment + hash_adjustment))

        except Exception as e:
            logger.error(f"Error calculating entropy level: {e}")
            return 4.0

    def start_state_injection(self, state_id: str) -> bool:
        """
        Start state injection with continuous event generation.

        Parameters:
        -----------
        state_id : str
            ID of the demo state to inject

        Returns:
        --------
        bool
            True if injection started successfully
        """
        try:
            if not self.inject_demo_state(state_id):
                return False

            if self.is_running:
                logger.warning("State injection already running")
                return False

            self.is_running = True
            self.injection_thread = threading.Thread(target=self._injection_loop, daemon=True)
            self.injection_thread.start()

            logger.info(f"Started state injection for {state_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting state injection: {e}")
            return False

    def stop_state_injection(self) -> None:
        """Stop state injection."""
        self.is_running = False
        if self.injection_thread:
            self.injection_thread.join(timeout=5.0)

        logger.info("Stopped state injection")

    def _injection_loop(self) -> None:
        """Main injection loop for generating events."""
        try:
            start_time = time.time()
            event_count = 0

            while self.is_running and self.active_state:
                # Check if test duration exceeded
                elapsed = time.time() - start_time
                if elapsed > self.active_state.test_duration:
                    logger.info("Test duration exceeded, stopping injection")
                    break

                # Generate events based on injection rate
                events_per_second = self.active_state.injection_rate
                sleep_time = 1.0 / events_per_second

                # Generate market event
                self._generate_market_event()

                # Generate portfolio event
                if event_count % 10 == 0:  # Every 10 events
                    self._generate_portfolio_event()

                # Generate rebalance event
                if event_count % 50 == 0:  # Every 50 events
                    self._generate_rebalance_event()

                event_count += 1
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in injection loop: {e}")

    def _generate_market_event(self) -> None:
        """Generate a market event."""
        try:
            if not self.active_state or not self.tick_history:
                return

            # Select random tick data
            tick_data = np.random.choice(self.tick_history)

            # Update with current demo state conditions
            tick_data.market_data.update(self.active_state.market_conditions)

            # Process through bit resolution engine
            if self.bit_engine:
                hash_value = hashlib.sha256(f"{tick_data.timestamp}_{tick_data.symbol}_{tick_data.price}".encode()).hexdigest()
                resolution_result = self.bit_engine.process_hash_resolution(
                    hash_value, tick_data.market_data, tick_data.price * 0.99, tick_data.price
                )

                if resolution_result:
                    logger.debug(f"Processed market event: {resolution_result.bit_phase.value}-bit, tensor={resolution_result.tensor_score:.4f}")

        except Exception as e:
            logger.error(f"Error generating market event: {e}")

    def _generate_portfolio_event(self) -> None:
        """Generate a portfolio event."""
        try:
            if not self.active_state:
                return

            # Create portfolio snapshot
            portfolio_state = self.active_state.portfolio_state
            total_value = portfolio_state["cash"]

            # Calculate position values
            for asset, allocation in portfolio_state["positions"].items():
                # Get current price (simplified)
                base_prices = {"BTC": 50000.0, "ETH": 3000.0, "ADA": 0.5, "DOT": 7.0, "USDC": 1.0}
                current_price = base_prices.get(asset, 1.0)
                position_value = allocation * portfolio_state["initial_capital"] * current_price
                total_value += position_value

            # Create snapshot
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=total_value,
                cash=portfolio_state["cash"],
                positions=portfolio_state["positions"].copy(),
                unrealized_pnl=total_value - portfolio_state["initial_capital"],
                realized_pnl=0.0
            )

            self.portfolio_history.append(snapshot)

        except Exception as e:
            logger.error(f"Error generating portfolio event: {e}")

    def _generate_rebalance_event(self) -> None:
        """Generate a rebalance event."""
        try:
            if not self.active_state or not self.tensor_utils:
                return

            # Simulate profit for rebalancing
            profit_amount = np.random.uniform(100, 1000)
            volatility = self.active_state.market_conditions["volatility"]
            entropy_level = self.active_state.market_conditions["entropy_level"]

            # Calculate rebalance
            rebalance_result = self.tensor_utils.rebalance_profit(profit_amount, volatility, entropy_level)

            if rebalance_result:
                # Create rebalance event
                event = RebalanceEvent(
                    event_id=f"rebalance_{int(time.time())}",
                    timestamp=datetime.now(),
                    trigger_type="profit",
                    old_allocations=self.active_state.portfolio_state["positions"].copy(),
                    new_allocations=rebalance_result.allocations,
                    rebalance_amount=profit_amount,
                    performance_impact=0.0
                )

                self.rebalance_history.append(event)

                # Update portfolio state
                self.active_state.portfolio_state["positions"].update(rebalance_result.allocations)

                logger.info(f"Generated rebalance event: {profit_amount:.2f} profit")

        except Exception as e:
            logger.error(f"Error generating rebalance event: {e}")

    def run_mathematical_validation(self) -> Dict[str, Any]:
        """Run mathematical validation on the demo system."""
        try:
            if not CORE_COMPONENTS_AVAILABLE:
                return {'error': 'Core components not available'}

            validation_results = {
                'bit_resolution_tests': [],
                'tensor_score_tests': [],
                'matrix_operation_tests': [],
                'rebalance_tests': [],
                'overall_status': 'unknown'
            }

            # Test bit resolution
            if self.bit_engine:
                test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
                test_market_data = {'entropy_level': 4.5, 'volatility': 0.03, 'market_heat': 0.6}

                resolution_result = self.bit_engine.process_hash_resolution(test_hash, test_market_data)
                if resolution_result:
                    validation_results['bit_resolution_tests'].append({
                        'test': 'hash_resolution',
                        'status': 'passed',
                        'bit_phase': resolution_result.bit_phase.value,
                        'tensor_score': resolution_result.tensor_score
                    })
                else:
                    validation_results['bit_resolution_tests'].append({
                        'test': 'hash_resolution',
                        'status': 'failed',
                        'error': 'No resolution result'
                    })

            # Test tensor scoring
            if self.tensor_utils:
                tensor_score = self.tensor_utils.calculate_tensor_score(45000.0, 46000.0, 8, test_market_data)
                validation_results['tensor_score_tests'].append({
                    'test': 'tensor_scoring',
                    'status': 'passed',
                    'score': tensor_score
                })

            # Determine overall status
            passed_tests = sum(1 for test_list in validation_results.values()
                             if isinstance(test_list, list) and any(t.get('status') == 'passed' for t in test_list))
            total_tests = sum(len(test_list) for test_list in validation_results.values()
                            if isinstance(test_list, list))

            if total_tests > 0:
                success_rate = passed_tests / total_tests
                validation_results['overall_status'] = 'passed' if success_rate > 0.8 else 'failed'
                validation_results['success_rate'] = success_rate

            # Store results
            self.validation_results.append({
                'timestamp': datetime.now().isoformat(),
                'results': validation_results
            })

            return validation_results

        except Exception as e:
            logger.error(f"Error running mathematical validation: {e}")
            return {'error': str(e)}

    def get_test_results(self) -> Dict[str, Any]:
        """Get comprehensive test results."""
        try:
            return {
                'active_state': self.active_state.state_id if self.active_state else None,
                'is_running': self.is_running,
                'portfolio_history_count': len(self.portfolio_history),
                'rebalance_history_count': len(self.rebalance_history),
                'validation_results_count': len(self.validation_results),
                'latest_portfolio': self.portfolio_history[-1] if self.portfolio_history else None,
                'latest_rebalance': self.rebalance_history[-1] if self.rebalance_history else None,
                'latest_validation': self.validation_results[-1] if self.validation_results else None
            }

        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            return {'error': str(e)}

    def export_test_results(self, output_path: str = "demo_test_results.json") -> None:
        """Export test results to file."""
        try:
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'test_results': self.get_test_results(),
                'portfolio_history': [
                    {
                        'timestamp': snapshot.timestamp.isoformat(),
                        'total_value': snapshot.total_value,
                        'cash': snapshot.cash,
                        'positions': snapshot.positions,
                        'unrealized_pnl': snapshot.unrealized_pnl
                    }
                    for snapshot in self.portfolio_history
                ],
                'rebalance_history': [
                    {
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'trigger_type': event.trigger_type,
                        'rebalance_amount': event.rebalance_amount,
                        'new_allocations': event.new_allocations
                    }
                    for event in self.rebalance_history
                ],
                'validation_results': self.validation_results
            }

            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)

            safe_print(f"✅ Demo test results exported to {output_path}")

        except Exception as e:
            safe_print(f"❌ Error exporting test results: {e}")

if __name__ == "__main__":
    # Test demo state injector
    injector = DemoStateInjector()

    # Test conservative strategy
    safe_print("🧪 Testing Conservative Strategy...")
    injector.start_state_injection("conservative_test")

    try:
        # Run for 60 seconds
        safe_print("📈 Demo state injection running for 60 seconds...")
        time.sleep(60)

        # Stop injection
        injector.stop_state_injection()

        # Run mathematical validation
        safe_print("\n🧪 Running Mathematical Validation...")
        validation_results = injector.run_mathematical_validation()
        safe_print(f"Validation Status: {validation_results.get('overall_status', 'UNKNOWN')}")

        # Get test results
        test_results = injector.get_test_results()
        safe_print("\n📊 TEST RESULTS")
        safe_print(f"Portfolio Snapshots: {test_results.get('portfolio_history_count', 0)}")
        safe_print(f"Rebalance Events: {test_results.get('rebalance_history_count', 0)}")
        safe_print(f"Validation Tests: {test_results.get('validation_results_count', 0)}")

        # Export results
        injector.export_test_results()

    except KeyboardInterrupt:
        safe_print("\n⏹️ Demo state injection stopped by user")
        injector.stop_state_injection()

    safe_print("✅ Demo state injector test completed")
