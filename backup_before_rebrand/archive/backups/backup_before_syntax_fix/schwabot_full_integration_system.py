#!/usr/bin/env python3
"""
Schwabot Full Integration System
Complete integration of all components with proper naming and persistence
"""

import json
import logging
import os
import pickle
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
        logging.FileHandler('schwabot_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
    class IntegrationConfig:
    """Configuration for full system integration."""
    session_id: str
    exchange_name: str = 'coinbase'
    sandbox_mode: bool = True
    api_key: str = ''
    api_secret: str = ''
    symbols: List[str] = None
    data_persistence: bool = True
    strategy_persistence: bool = True
    real_time_updates: bool = True

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTC/USDC', 'ETH/USDC', 'SOL/USDC']

@dataclass
    class TradingSession:
    """Trading session with all components."""
    session_id: str
    start_time: datetime
    config: IntegrationConfig
    trading_engine: Optional[EnhancedCCXTTradingEngine] = None
    strategy_engine: Optional[AutomatedStrategyEngine] = None
    soulprint_registry: Optional[SoulprintRegistry] = None
    portfolio_data: Dict = None
    trade_history: List[Dict] = None
    active_orders: Dict = None
    tensor_state: Dict = None

    def __post_init__(self):
        if self.portfolio_data is None:
            self.portfolio_data = {}
        if self.trade_history is None:
            self.trade_history = []
        if self.active_orders is None:
            self.active_orders = {}
        if self.tensor_state is None:
            self.tensor_state = {}

class SchwabotDataPersistence:
    """Data persistence layer for Schwabot system."""

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, 'schwabot.db')
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for data persistence."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create sessions table
                cursor.execute(''')
                    CREATE TABLE IF NOT EXISTS trading_sessions ()
                        session_id TEXT PRIMARY KEY,
                        start_time TEXT,
                        config TEXT,
                        status TEXT DEFAULT 'active'
                    )
                ''')'

                # Create portfolio table
                cursor.execute(''')
                    CREATE TABLE IF NOT EXISTS portfolio_history ()
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        timestamp TEXT,
                        portfolio_data TEXT,
                        FOREIGN KEY (session_id) REFERENCES trading_sessions (session_id)
                    )
                ''')'

                # Create trade history table
                cursor.execute(''')
                    CREATE TABLE IF NOT EXISTS trade_history ()
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        timestamp TEXT,
                        order_id TEXT,
                        symbol TEXT,
                        side TEXT,
                        quantity REAL,
                        price REAL,
                        status TEXT,
                        trade_data TEXT,
                        FOREIGN KEY (session_id) REFERENCES trading_sessions (session_id)
                    )
                ''')'

                # Create tensor state table
                cursor.execute(''')
                    CREATE TABLE IF NOT EXISTS tensor_states ()
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        timestamp TEXT,
                        symbol TEXT,
                        tensor_data TEXT,
                        FOREIGN KEY (session_id) REFERENCES trading_sessions (session_id)
                    )
                ''')'

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def save_session(self, session: TradingSession):
        """Save trading session to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(''')
                    INSERT OR REPLACE INTO trading_sessions 
                    (session_id, start_time, config, status)
                    VALUES (?, ?, ?, ?)
                ''', (')
                    session.session_id,
                    session.start_time.isoformat(),
                    json.dumps(asdict(session.config)),
                    'active'
                ))
                conn.commit()
                logger.info(f"Session {session.session_id} saved to database")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def load_session(self, session_id: str) -> Optional[TradingSession]:
        """Load trading session from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(''')
                    SELECT start_time, config FROM trading_sessions 
                    WHERE session_id = ?
                ''', (session_id,))'

                result = cursor.fetchone()
                if result:
                    start_time = datetime.fromisoformat(result[0])
                    config_dict = json.loads(result[1])
                    config = IntegrationConfig(**config_dict)

                    session = TradingSession()
                        session_id=session_id,
                        start_time=start_time,
                        config=config
                    )

                    # Load additional data
                    self.load_session_data(session)
                    return session

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
        return None

    def load_session_data(self, session: TradingSession):
        """Load additional session data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Load portfolio history
                cursor.execute(''')
                    SELECT portfolio_data FROM portfolio_history 
                    WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1
                ''', (session.session_id,))'
                portfolio_result = cursor.fetchone()
                if portfolio_result:
                    session.portfolio_data = json.loads(portfolio_result[0])

                # Load trade history
                cursor.execute(''')
                    SELECT trade_data FROM trade_history 
                    WHERE session_id = ? ORDER BY timestamp DESC
                ''', (session.session_id,))'
                trade_results = cursor.fetchall()
                session.trade_history = [json.loads(row[0]) for row in trade_results]

                # Load tensor states
                cursor.execute(''')
                    SELECT symbol, tensor_data FROM tensor_states 
                    WHERE session_id = ? ORDER BY timestamp DESC
                ''', (session.session_id,))'
                tensor_results = cursor.fetchall()
                for symbol, tensor_data in tensor_results:
                    session.tensor_state[symbol] = json.loads(tensor_data)

        except Exception as e:
            logger.error(f"Failed to load session data: {e}")

    def save_portfolio(self, session_id: str, portfolio_data: Dict):
        """Save portfolio data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(''')
                    INSERT INTO portfolio_history 
                    (session_id, timestamp, portfolio_data)
                    VALUES (?, ?, ?)
                ''', (')
                    session_id,
                    datetime.now().isoformat(),
                    json.dumps(portfolio_data)
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")

    def save_trade(self, session_id: str, trade_data: Dict):
        """Save trade data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(''')
                    INSERT INTO trade_history 
                    (session_id, timestamp, order_id, symbol, side, quantity, price, status, trade_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (')
                    session_id,
                    datetime.now().isoformat(),
                    trade_data.get('order_id'),
                    trade_data.get('symbol'),
                    trade_data.get('side'),
                    trade_data.get('quantity'),
                    trade_data.get('price'),
                    trade_data.get('status'),
                    json.dumps(trade_data)
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    def save_tensor_state(self, session_id: str, symbol: str, tensor_data: Dict):
        """Save tensor state data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(''')
                    INSERT INTO tensor_states 
                    (session_id, timestamp, symbol, tensor_data)
                    VALUES (?, ?, ?, ?)
                ''', (')
                    session_id,
                    datetime.now().isoformat(),
                    symbol,
                    json.dumps(tensor_data)
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save tensor state: {e}")

class SchwabotFullIntegration:
    """Complete Schwabot integration system."""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session = TradingSession()
            session_id=config.session_id,
            start_time=datetime.now(),
            config=config
        )
        self.persistence = SchwabotDataPersistence()
        self.running = False
        self.background_threads = []

        logger.info(f"Initialized Schwabot Full Integration for session {config.session_id}")

    def initialize_components(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing Schwabot components...")

            # Initialize enhanced CCXT trading engine
            exchange_config = {}
                'name': self.config.exchange_name,
                'sandbox': self.config.sandbox_mode
            }

            self.session.trading_engine = create_enhanced_ccxt_engine()
                exchange_config,
                self.config.api_key,
                self.config.api_secret
            )

            # Initialize automated strategy engine
            self.session.strategy_engine = AutomatedStrategyEngine()
                self.session.trading_engine
            )

            # Initialize soulprint registry
            registry_path = os.path.join('data', f'soulprint_registry_{self.config.session_id}.json')
            self.session.soulprint_registry = SoulprintRegistry(registry_path)

            # Add symbols to tracking
            for symbol in self.config.symbols:
                self.session.trading_engine.add_symbol_to_tracking(symbol)

            # Save session to database
            if self.config.data_persistence:
                self.persistence.save_session(self.session)

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False

    def start_background_processors(self):
        """Start background data processing and persistence."""
        if not self.running:
            self.running = True

            # Portfolio monitoring thread
            portfolio_thread = threading.Thread()
                target=self._portfolio_monitor,
                daemon=True,
                name="PortfolioMonitor"
            )
            portfolio_thread.start()
            self.background_threads.append(portfolio_thread)

            # Trade history monitoring thread
            trade_thread = threading.Thread()
                target=self._trade_history_monitor,
                daemon=True,
                name="TradeHistoryMonitor"
            )
            trade_thread.start()
            self.background_threads.append(trade_thread)

            # Tensor state monitoring thread
            tensor_thread = threading.Thread()
                target=self._tensor_state_monitor,
                daemon=True,
                name="TensorStateMonitor"
            )
            tensor_thread.start()
            self.background_threads.append(tensor_thread)

            logger.info("Background processors started")

    def _portfolio_monitor(self):
        """Monitor and persist portfolio data."""
        while self.running:
            try:
                if self.session.trading_engine:
                    portfolio = self.session.trading_engine.get_portfolio()
                    self.session.portfolio_data = portfolio

                    if self.config.data_persistence:
                        self.persistence.save_portfolio()
                            self.config.session_id,
                            portfolio
                        )

                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Portfolio monitoring error: {e}")
                time.sleep(60)

    def _trade_history_monitor(self):
        """Monitor and persist trade history."""
        while self.running:
            try:
                if self.session.trading_engine:
                    orders = self.session.trading_engine.get_all_orders()

                    # Check for new completed trades
                    for order_id, order_data in orders.items():
                        if order_id not in self.session.active_orders:
                            # New order
                            self.session.active_orders[order_id] = order_data
                        elif order_data.get('status') in ['closed', 'canceled']:
                            # Completed order
                            if self.config.data_persistence:
                                self.persistence.save_trade()
                                    self.config.session_id,
                                    order_data
                                )
                            self.session.trade_history.append(order_data)
                            del self.session.active_orders[order_id]

                time.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Trade history monitoring error: {e}")
                time.sleep(30)

    def _tensor_state_monitor(self):
        """Monitor and persist tensor state data."""
        while self.running:
            try:
                if self.session.trading_engine:
                    tensor_state = self.session.trading_engine.get_tensor_state()
                    self.session.tensor_state = tensor_state

                    if self.config.data_persistence:
                        for symbol, data in tensor_state.items():
                            self.persistence.save_tensor_state()
                                self.config.session_id,
                                symbol,
                                data
                            )

                time.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Tensor state monitoring error: {e}")
                time.sleep(120)

    def create_batch_order(self, symbol: str, side: str, total_quantity: float,)
                          price_range: tuple, batch_count: int = 10, spread_seconds: int = 30) -> str:
        """Create a batch order with full integration."""
        try:
            if side.lower() == 'buy':
                batch_id = self.session.trading_engine.create_enhanced_buy_wall()
                    symbol, total_quantity, price_range, batch_count, spread_seconds
                )
            else:
                batch_id = self.session.trading_engine.create_enhanced_sell_wall()
                    symbol, total_quantity, price_range, batch_count, spread_seconds
                )

            logger.info(f"Created {side} batch order {batch_id} for {symbol}")
            return batch_id

        except Exception as e:
            logger.error(f"Failed to create batch order: {e}")
            raise

    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        try:
            status = {}
                'session_id': self.config.session_id,
                'running': self.running,
                'exchange': self.config.exchange_name,
                'symbols_tracking': self.config.symbols,
                'components_initialized': {}
                    'trading_engine': self.session.trading_engine is not None,
                    'strategy_engine': self.session.strategy_engine is not None,
                    'soulprint_registry': self.session.soulprint_registry is not None
                },
                'data_persistence': self.config.data_persistence,
                'portfolio_value': 0.0,
                'active_orders_count': len(self.session.active_orders),
                'trade_history_count': len(self.session.trade_history),
                'tensor_symbols_count': len(self.session.tensor_state)
            }

            # Get portfolio value if available
            if self.session.portfolio_data:
                try:
                    total_value = sum()
                        float(balance.get('total', 0)) * 1.0  # Simplified calculation
                        for balance in self.session.portfolio_data.get('total', {}).values()
                    )
                    status['portfolio_value'] = total_value
                except:
                    pass

            return status

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    def shutdown(self):
        """Graceful shutdown of the integration system."""
        logger.info("Shutting down Schwabot Full Integration...")

        self.running = False

        # Wait for background threads
        for thread in self.background_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        # Shutdown trading engine
        if self.session.trading_engine:
            self.session.trading_engine.shutdown()

        # Save final session state
        if self.config.data_persistence:
            self.persistence.save_session(self.session)

        logger.info("Schwabot Full Integration shutdown complete")

# Factory function for creating integration instances
    def create_schwabot_integration(session_id: str = None, **kwargs) -> SchwabotFullIntegration:
    """Create a new Schwabot integration instance."""
    if session_id is None:
        session_id = f"schwabot_session_{int(time.time())}"

    config = IntegrationConfig(session_id=session_id, **kwargs)
    return SchwabotFullIntegration(config)

# Demo function
    def demo_full_integration():
    """Demonstrate full integration system."""
    print("🚀 Schwabot Full Integration Demo")
    print("=" * 50)

    # Create integration instance
    integration = create_schwabot_integration()
        session_id="demo_session_001",
        exchange_name="coinbase",
        sandbox_mode=True,
        symbols=['BTC/USDC', 'ETH/USDC'],
        data_persistence=True
    )

    try:
        # Initialize components
        if integration.initialize_components():
            print("✅ Components initialized successfully")

            # Start background processors
            integration.start_background_processors()
            print("✅ Background processors started")

            # Get system status
            status = integration.get_system_status()
            print(f"📊 System Status: {status}")

            # Wait a bit for data collection
            time.sleep(5)

            # Get updated status
            status = integration.get_system_status()
            print(f"📊 Updated Status: {status}")

        else:
            print("❌ Component initialization failed")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

    finally:
        # Shutdown
        integration.shutdown()
        print("✅ Demo completed")

if __name__ == "__main__":
    demo_full_integration() 