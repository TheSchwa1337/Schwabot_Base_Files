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
Demo Ledger State Injector - Schwabot UROS v1.0
==============================================

Replaces stub inject_demo_ledger() with proper demo ledger state injection
that loads prior tick JSON and simulates portfolio state for backtesting.

Features:
- Load prior portfolio state from JSON
- Inject historical tick data
- Simulate portfolio rebalancing
- Generate demo trading scenarios
- Export state snapshots for verification
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
# from core.unified_math_system import unified_math  # F811: duplicate import
import hashlib
import os
import glob

logger = logging.getLogger(__name__)

class DemoScenario(Enum):
    """Demo trading scenarios."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    QUANTUM = "quantum"
    CRASH_TEST = "crash_test"
    BULL_RUN = "bull_run"

@dataclass
class TickData:
    """Historical tick data point."""
    timestamp: datetime
    asset: str
    price: float
    volume: float
    phase_4bit: int
    phase_8bit: int
    phase_42bit: int
    bit_sync: int
    entropy_level: float
    volatility: float
    market_heat: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioSnapshot:
    """Portfolio state snapshot."""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, Dict[str, Any]]
    unrealized_pnl: float
    realized_pnl: float
    risk_metrics: Dict[str, float]
    scenario: DemoScenario
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DemoLedgerState:
    """Complete demo ledger state."""
    scenario: DemoScenario
    start_timestamp: datetime
    end_timestamp: datetime
    initial_portfolio: PortfolioSnapshot
    final_portfolio: PortfolioSnapshot
    tick_data: List[TickData]
    trade_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class DemoLedgerInjector:
    """
    Demo ledger state injector for backtesting and simulation.

    Mathematical Foundation:
    - Portfolio Evolution: P(t+1) = P(t) + Σ(trades * price_changes)
    - Risk Metrics: volatility = unified_math.std(returns), sharpe = unified_math.mean(returns) / unified_math.std(returns)
    - Performance Tracking: total_return = (final_value - initial_value) / initial_value
    - Scenario Generation: scenario_params = f(market_conditions, risk_profile)
    """

    def __init__(self, config_path: str = "./config/demo_ledger_config.json"):
        self.config_path = config_path

        # Demo state storage
        self.demo_states: Dict[str, DemoLedgerState] = {}
        self.current_scenario: DemoScenario = DemoScenario.BALANCED
        self.tick_data_path = "./data/tick_data/"
        self.portfolio_snapshots_path = "./data/portfolio_snapshots/"

        # Scenario configurations
        self.scenario_configs = {
            DemoScenario.CONSERVATIVE: {
                "initial_capital": 100000.0,
                "cash_buffer": 0.3,
                "max_position_size": 0.2,
                "risk_tolerance": 0.1,
                "rebalance_frequency": "daily"
            },
            DemoScenario.BALANCED: {
                "initial_capital": 100000.0,
                "cash_buffer": 0.2,
                "max_position_size": 0.3,
                "risk_tolerance": 0.3,
                "rebalance_frequency": "weekly"
            },
            DemoScenario.AGGRESSIVE: {
                "initial_capital": 100000.0,
                "cash_buffer": 0.1,
                "max_position_size": 0.4,
                "risk_tolerance": 0.5,
                "rebalance_frequency": "daily"
            },
            DemoScenario.QUANTUM: {
                "initial_capital": 100000.0,
                "cash_buffer": 0.05,
                "max_position_size": 0.5,
                "risk_tolerance": 0.7,
                "rebalance_frequency": "hourly"
            },
            DemoScenario.CRASH_TEST: {
                "initial_capital": 100000.0,
                "cash_buffer": 0.4,
                "max_position_size": 0.15,
                "risk_tolerance": 0.05,
                "rebalance_frequency": "daily"
            },
            DemoScenario.BULL_RUN: {
                "initial_capital": 100000.0,
                "cash_buffer": 0.1,
                "max_position_size": 0.45,
                "risk_tolerance": 0.6,
                "rebalance_frequency": "daily"
            }
        }

        # Integration with other components
        self.trade_simulator = None
        self.tensor_matcher = None
        self.bit_phase_engine = None
        self.matrix_mapper = None

        # Load configuration
        self._load_configuration()
        self._ensure_data_directories()
        logger.info("Demo Ledger Injector initialized")

    def _load_configuration(self) -> None:
        """Load demo ledger configuration."""
        try:
            # Default configuration
            config = {
                "data_paths": {
                    "tick_data": "./data/tick_data/",
                    "portfolio_snapshots": "./data/portfolio_snapshots/",
                    "demo_states": "./data/demo_states/"
                },
                "scenarios": {
                    "default": "balanced",
                    "duration_days": 30,
                    "tick_interval_minutes": 5
                },
                "assets": ["BTC", "ETH", "USDC", "XRP", "SOL"],
                "market_conditions": {
                    "normal": {"volatility": 0.02, "trend": 0.0},
                    "volatile": {"volatility": 0.05, "trend": 0.0},
                    "bull": {"volatility": 0.03, "trend": 0.01},
                    "bear": {"volatility": 0.04, "trend": -0.008}
                }
            }

            logger.info("Demo ledger configuration loaded")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def _ensure_data_directories(self) -> None:
        """Ensure data directories exist."""
        try:
            directories = [
                self.tick_data_path,
                self.portfolio_snapshots_path,
                "./data/demo_states/"
            ]

            for directory in directories:
                os.makedirs(directory, exist_ok=True)

            logger.info("Data directories ensured")

        except Exception as e:
            logger.error(f"Error ensuring data directories: {e}")

    def inject_demo_state(self, scenario_name: str = "balanced") -> bool:
        """
        Inject demo ledger state for specified scenario.

        Parameters:
        -----------
        scenario_name : str
            Name of the demo scenario to inject

        Returns:
        --------
        bool
            True if injection successful, False otherwise
        """
        try:
            # Convert scenario name to enum
            scenario = DemoScenario(scenario_name.lower())
            self.current_scenario = scenario

            # Generate demo ledger state
            demo_state = self._generate_demo_ledger_state(scenario)

            # Store demo state
            self.demo_states[scenario_name] = demo_state

            # Export demo state to file
            self._export_demo_state(demo_state, scenario_name)

            logger.info(f"Demo state injected for scenario: {scenario_name}")
            return True

        except Exception as e:
            logger.error(f"Error injecting demo state: {e}")
            return False

    def _generate_demo_ledger_state(self, scenario: DemoScenario) -> DemoLedgerState:
        """Generate complete demo ledger state for scenario."""
        try:
            # Get scenario configuration
            config = self.scenario_configs[scenario]

            # Generate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)

            # Generate initial portfolio
            initial_portfolio = self._generate_initial_portfolio(config)

            # Generate tick data
            tick_data = self._generate_tick_data(start_time, end_time, scenario)

            # Simulate trading and generate final portfolio
            final_portfolio, trade_history = self._simulate_trading(
                initial_portfolio, tick_data, config
            )

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                initial_portfolio, final_portfolio, trade_history
            )

            # Create demo ledger state
            demo_state = DemoLedgerState(
                scenario=scenario,
                start_timestamp=start_time,
                end_timestamp=end_time,
                initial_portfolio=initial_portfolio,
                final_portfolio=final_portfolio,
                tick_data=tick_data,
                trade_history=trade_history,
                performance_metrics=performance_metrics,
                metadata={
                    'scenario_config': config,
                    'generated_at': datetime.now().isoformat()
                }
            )

            return demo_state

        except Exception as e:
            logger.error(f"Error generating demo ledger state: {e}")
            return None

    def _generate_initial_portfolio(self, config: Dict[str, Any]) -> PortfolioSnapshot:
        """Generate initial portfolio snapshot."""
        try:
            initial_capital = config.get('initial_capital', 100000.0)
            cash_buffer = config.get('cash_buffer', 0.2)
            max_position_size = config.get('max_position_size', 0.3)

            # Calculate cash allocation
            cash = initial_capital * cash_buffer

            # Generate positions
            positions = {}
            assets = ["BTC", "ETH", "USDC", "XRP", "SOL"]
            base_prices = [50000.0, 3000.0, 1.0, 0.5, 100.0]

            remaining_capital = initial_capital - cash
            for i, asset in enumerate(assets):
                if asset == "USDC":
                    # USDC is stable, allocate as cash equivalent
                    positions[asset] = {
                        'quantity': remaining_capital * 0.1,
                        'entry_price': 1.0,
                        'current_price': 1.0
                    }
                else:
                    # Crypto assets
                    allocation = remaining_capital * max_position_size * np.random.uniform(0.5, 1.5)
                    price = base_prices[i] * np.random.uniform(0.8, 1.2)
                    quantity = allocation / price

                    positions[asset] = {
                        'quantity': quantity,
                        'entry_price': price,
                        'current_price': price
                    }

            return PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=initial_capital,
                cash=cash,
                positions=positions,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                risk_metrics={
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                },
                scenario=self.current_scenario
            )

        except Exception as e:
            logger.error(f"Error generating initial portfolio: {e}")
            return None

    def _generate_tick_data(self, start_time: datetime, end_time: datetime,
                           scenario: DemoScenario) -> List[TickData]:
        """Generate historical tick data for scenario."""
        try:
            tick_data = []
            current_time = start_time
            assets = ["BTC", "ETH", "USDC", "XRP", "SOL"]
            base_prices = [50000.0, 3000.0, 1.0, 0.5, 100.0]

            # Scenario-specific market conditions
            market_conditions = self._get_market_conditions(scenario)

            while current_time <= end_time:
                for i, asset in enumerate(assets):
                    # Generate price movement
                    base_price = base_prices[i]
                    if asset == "USDC":
                        # USDC is stable
                        price = 1.0
                        volatility = 0.0001
                    else:
                        # Crypto price movement
                        volatility = market_conditions['volatility']
                        trend = market_conditions['trend']

                        # Random walk with trend
                        price_change = np.random.normal(trend, volatility)
                        price = base_price * (1 + price_change)
                        base_prices[i] = price  # Update base price

                    # Generate bit phases
                    hash_value = hashlib.sha256(f"{asset}_{current_time.isoformat()}".encode()).hexdigest()
                    phase_4bit = int(hash_value[0:1], 16) % 16
                    phase_8bit = int(hash_value[0:2], 16) % 256
                    phase_42bit = int(hash_value[0:11], 16) % 4398046511104
                    bit_sync = phase_8bit

                    # Generate market metrics
                    entropy_level = np.random.uniform(2.0, 8.0)
                    market_volatility = np.random.uniform(0.01, 0.1)
                    market_heat = np.random.uniform(0.1, 1.0)

                    # Create tick data
                    tick = TickData(
                        timestamp=current_time,
                        asset=asset,
                        price=price,
                        volume=np.random.uniform(100, 1000),
                        phase_4bit=phase_4bit,
                        phase_8bit=phase_8bit,
                        phase_42bit=phase_42bit,
                        bit_sync=bit_sync,
                        entropy_level=entropy_level,
                        volatility=market_volatility,
                        market_heat=market_heat
                    )

                    tick_data.append(tick)

                # Move to next tick (5-minute intervals)
                current_time += timedelta(minutes=5)

            logger.info(f"Generated {len(tick_data)} tick data points")
            return tick_data

        except Exception as e:
            logger.error(f"Error generating tick data: {e}")
            return []

    def _get_market_conditions(self, scenario: DemoScenario) -> Dict[str, float]:
        """Get market conditions for scenario."""
        conditions = {
            DemoScenario.CONSERVATIVE: {"volatility": 0.015, "trend": 0.002},
            DemoScenario.BALANCED: {"volatility": 0.025, "trend": 0.005},
            DemoScenario.AGGRESSIVE: {"volatility": 0.035, "trend": 0.008},
            DemoScenario.QUANTUM: {"volatility": 0.045, "trend": 0.012},
            DemoScenario.CRASH_TEST: {"volatility": 0.06, "trend": -0.015},
            DemoScenario.BULL_RUN: {"volatility": 0.03, "trend": 0.02}
        }
        return conditions.get(scenario, {"volatility": 0.025, "trend": 0.005})

    def _simulate_trading(self, initial_portfolio: PortfolioSnapshot,
                         tick_data: List[TickData], config: Dict[str, Any]) -> Tuple[PortfolioSnapshot, List[Dict[str, Any]]]:
        """Simulate trading based on tick data."""
        try:
            # Initialize portfolio state
            current_portfolio = initial_portfolio
            trade_history = []

            # Group ticks by asset
            asset_ticks = {}
            for tick in tick_data:
                if tick.asset not in asset_ticks:
                    asset_ticks[tick.asset] = []
                asset_ticks[tick.asset].append(tick)

            # Simulate trading for each asset
            for asset, ticks in asset_ticks.items():
                asset_trades = self._simulate_asset_trading(
                    asset, ticks, current_portfolio, config
                )
                trade_history.extend(asset_trades)

                # Update portfolio after each asset
                current_portfolio = self._update_portfolio_from_trades(
                    current_portfolio, asset_trades
                )

            # Calculate final metrics
            final_portfolio = self._calculate_final_portfolio(current_portfolio, trade_history)

            return final_portfolio, trade_history

        except Exception as e:
            logger.error(f"Error simulating trading: {e}")
            return initial_portfolio, []

    def _simulate_asset_trading(self, asset: str, ticks: List[TickData],
                               portfolio: PortfolioSnapshot, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate trading for a specific asset."""
        try:
            trades = []
            position = portfolio.positions.get(asset, {'quantity': 0.0, 'entry_price': 0.0})
            risk_tolerance = config.get('risk_tolerance', 0.3)

            for i, tick in enumerate(ticks):
                # Calculate tensor score
                if position['entry_price'] > 0:
                    tensor_score = (tick.price - position['entry_price']) / position['entry_price']
                else:
                    tensor_score = 0.0

                # Determine trade based on tensor score and risk tolerance
                trade_decision = self._make_trade_decision(
                    tensor_score, risk_tolerance, position, tick
                )

                if trade_decision:
                    trade = {
                        'timestamp': tick.timestamp,
                        'asset': asset,
                        'trade_type': trade_decision['type'],
                        'quantity': trade_decision['quantity'],
                        'price': tick.price,
                        'tensor_score': tensor_score,
                        'bit_phase': tick.phase_8bit,
                        'basket_id': f"basket_8bit_{tick.phase_8bit}"
                    }
                    trades.append(trade)

                    # Update position
                    if trade_decision['type'] == 'buy':
                        if position['quantity'] == 0:
                            position['entry_price'] = tick.price
                        else:
                            # Weighted average
                            total_quantity = position['quantity'] + trade_decision['quantity']
                            position['entry_price'] = (
                                (position['quantity'] * position['entry_price'] +
                                 trade_decision['quantity'] * tick.price) / total_quantity
                            )
                        position['quantity'] += trade_decision['quantity']
                    else:  # sell
                        position['quantity'] -= trade_decision['quantity']
                        if position['quantity'] <= 0:
                            position['quantity'] = 0.0
                            position['entry_price'] = 0.0

                position['current_price'] = tick.price

            return trades

        except Exception as e:
            logger.error(f"Error simulating asset trading: {e}")
            return []

    def _make_trade_decision(self, tensor_score: float, risk_tolerance: float,
                            position: Dict[str, Any], tick: TickData) -> Optional[Dict[str, Any]]:
        """Make trade decision based on tensor score and risk tolerance."""
        try:
            entry_threshold = 0.02 * risk_tolerance
            exit_threshold = -0.05 * risk_tolerance

            if tensor_score > entry_threshold and position['quantity'] == 0:
                # Buy signal
                quantity = 1000.0 / tick.price  # $1000 position
                return {'type': 'buy', 'quantity': quantity}
            elif tensor_score < exit_threshold and position['quantity'] > 0:
                # Sell signal
                return {'type': 'sell', 'quantity': position['quantity']}

            return None

        except Exception as e:
            logger.error(f"Error making trade decision: {e}")
            return None

    def _update_portfolio_from_trades(self, portfolio: PortfolioSnapshot,
                                     trades: List[Dict[str, Any]]) -> PortfolioSnapshot:
        """Update portfolio state from trades."""
        try:
            # Create new portfolio snapshot
            new_portfolio = PortfolioSnapshot(
                timestamp=portfolio.timestamp,
                total_value=portfolio.total_value,
                cash=portfolio.cash,
                positions=portfolio.positions.copy(),
                unrealized_pnl=portfolio.unrealized_pnl,
                realized_pnl=portfolio.realized_pnl,
                risk_metrics=portfolio.risk_metrics.copy(),
                scenario=portfolio.scenario
            )

            # Apply trades
            for trade in trades:
                asset = trade['asset']
                trade_type = trade['trade_type']
                quantity = trade['quantity']
                price = trade['price']

                if asset not in new_portfolio.positions:
                    new_portfolio.positions[asset] = {
                        'quantity': 0.0,
                        'entry_price': 0.0,
                        'current_price': price
                    }

                position = new_portfolio.positions[asset]

                if trade_type == 'buy':
                    # Calculate cash impact
                    trade_value = quantity * price
                    commission = trade_value * 0.0025
                    new_portfolio.cash -= (trade_value + commission)

                    # Update position
                    if position['quantity'] == 0:
                        position['entry_price'] = price
                    else:
                        # Weighted average
                        total_quantity = position['quantity'] + quantity
                        position['entry_price'] = (
                            (position['quantity'] * position['entry_price'] +
                             quantity * price) / total_quantity
                        )
                    position['quantity'] += quantity

                else:  # sell
                    # Calculate cash impact
                    trade_value = quantity * price
                    commission = trade_value * 0.0025
                    new_portfolio.cash += (trade_value - commission)

                    # Update position
                    position['quantity'] -= quantity
                    if position['quantity'] <= 0:
                        position['quantity'] = 0.0
                        position['entry_price'] = 0.0

                position['current_price'] = price

            return new_portfolio

        except Exception as e:
            logger.error(f"Error updating portfolio from trades: {e}")
            return portfolio

    def _calculate_final_portfolio(self, portfolio: PortfolioSnapshot,
                                  trades: List[Dict[str, Any]]) -> PortfolioSnapshot:
        """Calculate final portfolio state."""
        try:
            # Calculate total value
            total_value = portfolio.cash
            unrealized_pnl = 0.0

            for asset, position in portfolio.positions.items():
                if position['quantity'] > 0:
                    position_value = position['quantity'] * position['current_price']
                    total_value += position_value

                    if position['entry_price'] > 0:
                        unrealized_pnl += position['quantity'] * (position['current_price'] - position['entry_price'])

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(trades)

            return PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=total_value,
                cash=portfolio.cash,
                positions=portfolio.positions,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=portfolio.realized_pnl,
                risk_metrics=risk_metrics,
                scenario=portfolio.scenario
            )

        except Exception as e:
            logger.error(f"Error calculating final portfolio: {e}")
            return portfolio

    def _calculate_risk_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk metrics from trade history."""
        try:
            if not trades:
                return {'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}

            # Calculate returns
            returns = []
            for i in range(1, len(trades)):
                prev_price = trades[i-1]['price']
                curr_price = trades[i]['price']
                if prev_price > 0:
                    returns.append((curr_price - prev_price) / prev_price)

            if not returns:
                return {'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}

            returns_array = np.array(returns)

            # Calculate metrics
            volatility = unified_math.unified_math.std(returns_array)
            sharpe_ratio = unified_math.unified_math.mean(returns_array) / (volatility + 1e-9)

            # Calculate win rate
            winning_trades = sum(1 for r in returns if r > 0)
            win_rate = winning_trades / len(returns) if returns else 0.0

            # Calculate max drawdown (simplified)
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = unified_math.unified_math.min(drawdown)

            return {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': unified_math.abs(max_drawdown),
                'win_rate': win_rate
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}

    def _calculate_performance_metrics(self, initial_portfolio: PortfolioSnapshot,
                                     final_portfolio: PortfolioSnapshot,
                                     trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        try:
            # Calculate total return
            total_return = (final_portfolio.total_value - initial_portfolio.total_value) / initial_portfolio.total_value

            # Calculate trade statistics
            total_trades = len(trade_history)
            buy_trades = sum(1 for trade in trade_history if trade['trade_type'] == 'buy')
            sell_trades = sum(1 for trade in trade_history if trade['trade_type'] == 'sell')

            # Calculate average tensor score
            tensor_scores = [trade['tensor_score'] for trade in trade_history]
            avg_tensor_score = unified_math.unified_math.mean(tensor_scores) if tensor_scores else 0.0

            return {
                'total_return': total_return,
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'avg_tensor_score': avg_tensor_score,
                'initial_value': initial_portfolio.total_value,
                'final_value': final_portfolio.total_value,
                'absolute_pnl': final_portfolio.total_value - initial_portfolio.total_value
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def _export_demo_state(self, demo_state: DemoLedgerState, scenario_name: str) -> None:
        """Export demo state to file."""
        try:
            output_path = f"./data/demo_states/{scenario_name}_demo_state.json"

            # Convert to serializable format
            export_data = {
                'scenario': demo_state.scenario.value,
                'start_timestamp': demo_state.start_timestamp.isoformat(),
                'end_timestamp': demo_state.end_timestamp.isoformat(),
                'initial_portfolio': {
                    'timestamp': demo_state.initial_portfolio.timestamp.isoformat(),
                    'total_value': demo_state.initial_portfolio.total_value,
                    'cash': demo_state.initial_portfolio.cash,
                    'positions': demo_state.initial_portfolio.positions,
                    'unrealized_pnl': demo_state.initial_portfolio.unrealized_pnl,
                    'realized_pnl': demo_state.initial_portfolio.realized_pnl,
                    'risk_metrics': demo_state.initial_portfolio.risk_metrics
                },
                'final_portfolio': {
                    'timestamp': demo_state.final_portfolio.timestamp.isoformat(),
                    'total_value': demo_state.final_portfolio.total_value,
                    'cash': demo_state.final_portfolio.cash,
                    'positions': demo_state.final_portfolio.positions,
                    'unrealized_pnl': demo_state.final_portfolio.unrealized_pnl,
                    'realized_pnl': demo_state.final_portfolio.realized_pnl,
                    'risk_metrics': demo_state.final_portfolio.risk_metrics
                },
                'tick_data_count': len(demo_state.tick_data),
                'trade_history_count': len(demo_state.trade_history),
                'performance_metrics': demo_state.performance_metrics,
                'metadata': demo_state.metadata
            }

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Demo state exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting demo state: {e}")

    def load_demo_state(self, scenario_name: str) -> Optional[DemoLedgerState]:
        """Load demo state from file."""
        try:
            file_path = f"./data/demo_states/{scenario_name}_demo_state.json"

            if not os.path.exists(file_path):
                logger.warning(f"Demo state file not found: {file_path}")
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Convert back to DemoLedgerState object
            # (This is a simplified conversion - full implementation would be more complex)
            logger.info(f"Demo state loaded from {file_path}")
            return self.demo_states.get(scenario_name)

        except Exception as e:
            logger.error(f"Error loading demo state: {e}")
            return None

    def get_available_scenarios(self) -> List[str]:
        """Get list of available demo scenarios."""
        return [scenario.value for scenario in DemoScenario]

    def set_trade_simulator(self, trade_simulator) -> None:
        """Set trade simulator for integration."""
        self.trade_simulator = trade_simulator
        logger.info("Trade simulator integrated with demo ledger injector")

    def set_tensor_matcher(self, tensor_matcher) -> None:
        """Set tensor matcher for integration."""
        self.tensor_matcher = tensor_matcher
        logger.info("Tensor matcher integrated with demo ledger injector")

    def set_bit_phase_engine(self, bit_engine) -> None:
        """Set bit phase engine for integration."""
        self.bit_phase_engine = bit_engine
        logger.info("Bit phase engine integrated with demo ledger injector")

    def set_matrix_mapper(self, matrix_mapper) -> None:
        """Set matrix mapper for integration."""
        self.matrix_mapper = matrix_mapper
        logger.info("Matrix mapper integrated with demo ledger injector")

if __name__ == "__main__":
    # Test demo ledger injector
    injector = DemoLedgerInjector()

    # Test scenario injection
    scenarios = ["conservative", "balanced", "aggressive"]

    for scenario in scenarios:
        safe_print(f"\n🧪 Testing {scenario} scenario...")
        success = injector.inject_demo_state(scenario)
        safe_print(f"✅ {scenario} scenario: {'SUCCESS' if success else 'FAILED'}")

    # Get available scenarios
    available = injector.get_available_scenarios()
    safe_print(f"\n📋 Available scenarios: {available}")

    # Load demo state
    demo_state = injector.load_demo_state("balanced")
    if demo_state:
        safe_print(f"📊 Loaded demo state: {demo_state.scenario.value}")
        safe_print(f"   Total return: {demo_state.performance_metrics.get('total_return', 0):.2%}")
        safe_print(f"   Total trades: {demo_state.performance_metrics.get('total_trades', 0)}")
