#!/usr/bin/env python3
"""
Schwabot Trading Dashboard
Advanced Algorithmic Trading Intelligence System
Matches the visual interface at http://127.0.0.1:5000/
"""

import json
import logging
import os
import sqlite3
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.automated_strategy_engine import AutomatedStrategyEngine
from core.enhanced_ccxt_trading_engine import EnhancedCCXTTradingEngine, create_enhanced_ccxt_engine
from core.soulprint_registry import SoulprintRegistry

# Setup logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[]
        logging.FileHandler('schwabot_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
    class DashboardConfig:
    """Configuration for Schwabot Trading Dashboard."""
    session_id: str
    exchange_name: str = 'coinbase'
    sandbox_mode: bool = True
    api_key: str = ''
    api_secret: str = ''
    symbols: List[str] = None
    portfolio_value: float = 10000.0
    demo_mode: bool = True

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTC/USDC', 'ETH/USDC', 'SOL/USDC']

@dataclass
    class DashboardState:
    """Current state of the trading dashboard."""
    session_id: str
    start_time: datetime
    config: DashboardConfig
    trading_engine: Optional[EnhancedCCXTTradingEngine] = None
    strategy_engine: Optional[AutomatedStrategyEngine] = None
    soulprint_registry: Optional[SoulprintRegistry] = None
    portfolio_data: Dict = None
    trade_history: List[Dict] = None
    active_orders: Dict = None
    tensor_state: Dict = None
    total_profit: float = 0.0
    win_rate: float = 0.0
    active_trades: int = 0

    def __post_init__(self):
        if self.portfolio_data is None:
            self.portfolio_data = {}
        if self.trade_history is None:
            self.trade_history = []
        if self.active_orders is None:
            self.active_orders = {}
        if self.tensor_state is None:
            self.tensor_state = {}

class SchwabotTradingDashboard:
    """Schwabot Trading Dashboard - Main dashboard interface."""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.state = DashboardState()
            session_id=config.session_id,
            start_time=datetime.now(),
            config=config
        )
        self.running = False
        self.background_threads = []

        logger.info(f"Initialized Schwabot Trading Dashboard for session {config.session_id}")

    def initialize_components(self):
        """Initialize all dashboard components."""
        try:
            logger.info("Initializing Schwabot Trading Dashboard components...")

            # Initialize enhanced CCXT trading engine
            exchange_config = {}
                'name': self.config.exchange_name,
                'sandbox': self.config.sandbox_mode
            }

            self.state.trading_engine = create_enhanced_ccxt_engine()
                exchange_config,
                self.config.api_key,
                self.config.api_secret
            )

            # Initialize automated strategy engine
            self.state.strategy_engine = AutomatedStrategyEngine()
                self.state.trading_engine
            )

            # Initialize soulprint registry
            registry_path = os.path.join('data', f'soulprint_registry_{self.config.session_id}.json')
            self.state.soulprint_registry = SoulprintRegistry(registry_path)

            # Add symbols to tracking
            for symbol in self.config.symbols:
                self.state.trading_engine.add_symbol_to_tracking(symbol)

            logger.info("All dashboard components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Dashboard component initialization failed: {e}")
            return False

    def start_background_processors(self):
        """Start background data processing for dashboard."""
        if not self.running:
            self.running = True

            # Portfolio monitoring thread
            portfolio_thread = threading.Thread()
                target=self._portfolio_monitor,
                daemon=True,
                name="DashboardPortfolioMonitor"
            )
            portfolio_thread.start()
            self.background_threads.append(portfolio_thread)

            # Trade history monitoring thread
            trade_thread = threading.Thread()
                target=self._trade_history_monitor,
                daemon=True,
                name="DashboardTradeHistoryMonitor"
            )
            trade_thread.start()
            self.background_threads.append(trade_thread)

            # Tensor state monitoring thread
            tensor_thread = threading.Thread()
                target=self._tensor_state_monitor,
                daemon=True,
                name="DashboardTensorStateMonitor"
            )
            tensor_thread.start()
            self.background_threads.append(tensor_thread)

            logger.info("Dashboard background processors started")

    def _portfolio_monitor(self):
        """Monitor and update portfolio data for dashboard."""
        while self.running:
            try:
                if self.state.trading_engine:
                    portfolio = self.state.trading_engine.get_portfolio()
                    self.state.portfolio_data = portfolio

                    # Update portfolio value
                    try:
                        total_value = sum()
                            float(balance.get('total', 0)) * 1.0
                            for balance in portfolio.get('total', {}).values()
                        )
                        self.state.config.portfolio_value = total_value
                    except Exception:
                        pass

                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Dashboard portfolio monitoring error: {e}")
                time.sleep(60)

    def _trade_history_monitor(self):
        """Monitor and update trade history for dashboard."""
        while self.running:
            try:
                if self.state.trading_engine:
                    orders = self.state.trading_engine.get_all_orders()

                    # Update active trades count
                    self.state.active_trades = len(orders)

                    # Check for new completed trades
                    for order_id, order_data in orders.items():
                        if order_id not in self.state.active_orders:
                            # New order
                            self.state.active_orders[order_id] = order_data
                        elif order_data.get('status') in ['closed', 'canceled']:
                            # Completed order
                            self.state.trade_history.append(order_data)
                            del self.state.active_orders[order_id]

                    # Calculate win rate
                    if self.state.trade_history:
                        winning_trades = sum(1 for trade in self.state.trade_history)
                                           if trade.get('profit', 0) > 0)
                        self.state.win_rate = (winning_trades / len(self.state.trade_history)) * 100

                time.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Dashboard trade history monitoring error: {e}")
                time.sleep(30)

    def _tensor_state_monitor(self):
        """Monitor and update tensor state for dashboard."""
        while self.running:
            try:
                if self.state.trading_engine:
                    tensor_state = self.state.trading_engine.get_tensor_state()
                    self.state.tensor_state = tensor_state

                time.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Dashboard tensor state monitoring error: {e}")
                time.sleep(120)

    def execute_trade(self, symbol: str, side: str, quantity: float, price: float = None) -> Dict:
        """Execute a trade through the dashboard."""
        try:
            if self.state.trading_engine:
                if side.lower() == 'buy':
                    order = self.state.trading_engine.create_market_buy_order(symbol, quantity, price)
                else:
                    order = self.state.trading_engine.create_market_sell_order(symbol, quantity, price)

                logger.info(f"Dashboard executed {side} trade for {symbol}: {quantity} @ {price}")
                return {}
                    'success': True,
                    'order_id': order.get('id'),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'status': 'executed'
                }

            return {'success': False, 'error': 'Trading engine not available'}

        except Exception as e:
            logger.error(f"Dashboard trade execution failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_dashboard_data(self) -> Dict:
        """Get all dashboard data for the web interface."""
        try:
            return {}
                'session_id': self.config.session_id,
                'portfolio_value': self.config.portfolio_value,
                'total_profit': self.state.total_profit,
                'win_rate': self.state.win_rate,
                'active_trades': self.state.active_trades,
                'symbols': self.config.symbols,
                'exchange': self.config.exchange_name,
                'demo_mode': self.config.demo_mode,
                'trade_history': self.state.trade_history[-10:],  # Last 10 trades
                'active_orders': self.state.active_orders,
                'tensor_state': self.state.tensor_state,
                'running': self.running
            }

        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {'error': str(e)}

    def calculate_math_score(self, symbol: str, price: float, volume: float, confidence: float) -> Dict:
        """Calculate mathematical score for trading decision."""
        try:
            if self.state.strategy_engine:
                score = self.state.strategy_engine.calculate_decision_score()
                    symbol, price, volume, confidence
                )
                return {}
                    'success': True,
                    'math_score': score,
                    'symbol': symbol,
                    'price': price,
                    'volume': volume,
                    'confidence': confidence
                }

            return {'success': False, 'error': 'Strategy engine not available'}

        except Exception as e:
            logger.error(f"Math score calculation failed: {e}")
            return {'success': False, 'error': str(e)}

    def save_to_backlog(self, trade_data: Dict) -> Dict:
        """Save trade data to backlog for later execution."""
        try:
            if self.state.soulprint_registry:
                # Generate hash for the trade data
                trade_hash = self.state.soulprint_registry.generate_soulprint_hash(trade_data)

                # Save to registry
                self.state.soulprint_registry.register_soulprint()
                    trade_hash,
                    trade_data,
                    'dashboard_backlog'
                )

                return {}
                    'success': True,
                    'hash': trade_hash,
                    'message': 'Trade saved to backlog'
                }

            return {'success': False, 'error': 'Soulprint registry not available'}

        except Exception as e:
            logger.error(f"Failed to save to backlog: {e}")
            return {'success': False, 'error': str(e)}

    def shutdown(self):
        """Graceful shutdown of the dashboard."""
        logger.info("Shutting down Schwabot Trading Dashboard...")

        self.running = False

        # Wait for background threads
        for thread in self.background_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        # Shutdown trading engine
        if self.state.trading_engine:
            self.state.trading_engine.shutdown()

        logger.info("Schwabot Trading Dashboard shutdown complete")

# Factory function for creating dashboard instances
    def create_schwabot_dashboard(session_id: str = None, **kwargs) -> SchwabotTradingDashboard:
    """Create a new Schwabot Trading Dashboard instance."""
    if session_id is None:
        session_id = f"dashboard_session_{int(time.time())}"

    config = DashboardConfig(session_id=session_id, **kwargs)
    return SchwabotTradingDashboard(config)

# Demo function
    def demo_trading_dashboard():
    """Demonstrate the trading dashboard."""
    print("🚀 Schwabot Trading Dashboard Demo")
    print("=" * 50)

    # Create dashboard instance
    dashboard = create_schwabot_dashboard()
        session_id="demo_dashboard_001",
        exchange_name="coinbase",
        sandbox_mode=True,
        symbols=['BTC/USDC', 'ETH/USDC'],
        demo_mode=True
    )

    try:
        # Initialize components
        if dashboard.initialize_components():
            print("✅ Dashboard components initialized successfully")

            # Start background processors
            dashboard.start_background_processors()
            print("✅ Background processors started")

            # Get dashboard data
            data = dashboard.get_dashboard_data()
            print(f"📊 Dashboard Data: {data}")

            # Calculate math score
            score = dashboard.calculate_math_score('BTC/USDC', 60000, 1000, 0.5)
            print(f"🧮 Math Score: {score}")

            # Wait a bit for data collection
            time.sleep(5)

            # Get updated data
            data = dashboard.get_dashboard_data()
            print(f"📊 Updated Dashboard Data: {data}")

        else:
            print("❌ Dashboard initialization failed")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

    finally:
        # Shutdown
        dashboard.shutdown()
        print("✅ Demo completed")

if __name__ == "__main__":
    demo_trading_dashboard() 