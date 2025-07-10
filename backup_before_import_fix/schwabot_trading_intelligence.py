#!/usr/bin/env python3
"""
Schwabot Trading Intelligence
Advanced Algorithmic Trading Intelligence System
Core intelligence engine for the trading dashboard
"""

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.automated_strategy_engine import AutomatedStrategyEngine
from core.enhanced_ccxt_trading_engine import EnhancedCCXTTradingEngine
from core.soulprint_registry import SoulprintRegistry

# Setup logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[]
        logging.FileHandler('schwabot_intelligence.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
    class IntelligenceConfig:
    """Configuration for Schwabot Trading Intelligence."""
    session_id: str
    exchange_name: str = 'coinbase'
    sandbox_mode: bool = True
    api_key: str = ''
    api_secret: str = ''
    symbols: List[str] = None
    enable_learning: bool = True
    enable_automation: bool = True

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTC/USDC', 'ETH/USDC', 'SOL/USDC']

class SchwabotTradingIntelligence:
    """Schwabot Trading Intelligence - Core intelligence engine."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config
        self.trading_engine = None
        self.strategy_engine = None
        self.soulprint_registry = None
        self.running = False
        self.background_threads = []

        logger.info(f"Initialized Schwabot Trading Intelligence for session {config.session_id}")

    def initialize_components(self):
        """Initialize all intelligence components."""
        try:
            logger.info("Initializing Schwabot Trading Intelligence components...")

            # Initialize enhanced CCXT trading engine
            exchange_config = {}
                'name': self.config.exchange_name,
                'sandbox': self.config.sandbox_mode
            }

            self.trading_engine = EnhancedCCXTTradingEngine()
                exchange_config,
                self.config.api_key,
                self.config.api_secret
            )

            # Initialize automated strategy engine
            self.strategy_engine = AutomatedStrategyEngine()
                self.trading_engine
            )

            # Initialize soulprint registry
            registry_path = os.path.join('data', f'intelligence_registry_{self.config.session_id}.json')
            self.soulprint_registry = SoulprintRegistry(registry_path)

            # Add symbols to tracking
            for symbol in self.config.symbols:
                self.trading_engine.add_symbol_to_tracking(symbol)

            logger.info("All intelligence components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Intelligence component initialization failed: {e}")
            return False

    def start_intelligence_engine(self):
        """Start the intelligence engine."""
        if not self.running:
            self.running = True

            # Market analysis thread
            analysis_thread = threading.Thread()
                target=self._market_analysis_loop,
                daemon=True,
                name="MarketAnalysis"
            )
            analysis_thread.start()
            self.background_threads.append(analysis_thread)

            # Strategy learning thread
            if self.config.enable_learning:
                learning_thread = threading.Thread()
                    target=self._strategy_learning_loop,
                    daemon=True,
                    name="StrategyLearning"
                )
                learning_thread.start()
                self.background_threads.append(learning_thread)

            # Automated trading thread
            if self.config.enable_automation:
                automation_thread = threading.Thread()
                    target=self._automated_trading_loop,
                    daemon=True,
                    name="AutomatedTrading"
                )
                automation_thread.start()
                self.background_threads.append(automation_thread)

            logger.info("Schwabot Trading Intelligence engine started")

    def _market_analysis_loop(self):
        """Continuous market analysis loop."""
        while self.running:
            try:
                if self.trading_engine:
                    # Analyze all tracked symbols
                    for symbol in self.config.symbols:
                        analysis = self.strategy_engine.analyze_market_conditions(symbol)

                        # Store analysis results
                        if self.soulprint_registry:
                            self.soulprint_registry.register_soulprint()
                                f"analysis_{symbol}_{int(time.time())}",
                                analysis,
                                'market_analysis'
                            )

                time.sleep(60)  # Analyze every minute

            except Exception as e:
                logger.error(f"Market analysis error: {e}")
                time.sleep(120)

    def _strategy_learning_loop(self):
        """Continuous strategy learning loop."""
        while self.running:
            try:
                if self.strategy_engine:
                    # Learn from recent trades and market data
                    self.strategy_engine.learn_from_recent_data()

                    # Optimize strategies
                    self.strategy_engine.optimize_strategies()

                time.sleep(300)  # Learn every 5 minutes

            except Exception as e:
                logger.error(f"Strategy learning error: {e}")
                time.sleep(600)

    def _automated_trading_loop(self):
        """Continuous automated trading loop."""
        while self.running:
            try:
                if self.trading_engine and self.strategy_engine:
                    # Get automated trading decisions
                    decisions = self.strategy_engine.get_automated_decisions()

                    # Execute decisions
                    for decision in decisions:
                        if decision.get('confidence', 0) > 0.7:  # High confidence threshold
                            self._execute_automated_trade(decision)

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Automated trading error: {e}")
                time.sleep(60)

    def _execute_automated_trade(self, decision: Dict):
        """Execute an automated trade decision."""
        try:
            symbol = decision.get('symbol')
            side = decision.get('side')
            quantity = decision.get('quantity')
            price = decision.get('price')

            if side == 'buy':
                order = self.trading_engine.create_market_buy_order(symbol, quantity, price)
            else:
                order = self.trading_engine.create_market_sell_order(symbol, quantity, price)

            logger.info(f"Automated {side} trade executed for {symbol}: {quantity} @ {price}")

            # Store decision in registry
            if self.soulprint_registry:
                self.soulprint_registry.register_soulprint()
                    f"auto_trade_{int(time.time())}",
                    {**decision, 'order_id': order.get('id')},
                    'automated_trading'
                )

        except Exception as e:
            logger.error(f"Automated trade execution failed: {e}")

    def get_intelligence_status(self) -> Dict:
        """Get intelligence system status."""
        try:
            return {}
                'session_id': self.config.session_id,
                'running': self.running,
                'exchange': self.config.exchange_name,
                'symbols_tracking': self.config.symbols,
                'components_initialized': {}
                    'trading_engine': self.trading_engine is not None,
                    'strategy_engine': self.strategy_engine is not None,
                    'soulprint_registry': self.soulprint_registry is not None
                },
                'features_enabled': {}
                    'learning': self.config.enable_learning,
                    'automation': self.config.enable_automation
                },
                'background_threads': len(self.background_threads)
            }

        except Exception as e:
            logger.error(f"Failed to get intelligence status: {e}")
            return {'error': str(e)}

    def shutdown(self):
        """Graceful shutdown of the intelligence system."""
        logger.info("Shutting down Schwabot Trading Intelligence...")

        self.running = False

        # Wait for background threads
        for thread in self.background_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        # Shutdown trading engine
        if self.trading_engine:
            self.trading_engine.shutdown()

        logger.info("Schwabot Trading Intelligence shutdown complete")

# Factory function for creating intelligence instances
    def create_schwabot_intelligence(session_id: str = None, **kwargs) -> SchwabotTradingIntelligence:
    """Create a new Schwabot Trading Intelligence instance."""
    if session_id is None:
        session_id = f"intelligence_session_{int(time.time())}"

    config = IntelligenceConfig(session_id=session_id, **kwargs)
    return SchwabotTradingIntelligence(config)

# Demo function
    def demo_trading_intelligence():
    """Demonstrate the trading intelligence system."""
    print("🧠 Schwabot Trading Intelligence Demo")
    print("=" * 50)

    # Create intelligence instance
    intelligence = create_schwabot_intelligence()
        session_id="demo_intelligence_001",
        exchange_name="coinbase",
        sandbox_mode=True,
        symbols=['BTC/USDC', 'ETH/USDC'],
        enable_learning=True,
        enable_automation=True
    )

    try:
        # Initialize components
        if intelligence.initialize_components():
            print("✅ Intelligence components initialized successfully")

            # Start intelligence engine
            intelligence.start_intelligence_engine()
            print("✅ Intelligence engine started")

            # Get status
            status = intelligence.get_intelligence_status()
            print(f"📊 Intelligence Status: {status}")

            # Wait a bit for processing
            time.sleep(10)

            # Get updated status
            status = intelligence.get_intelligence_status()
            print(f"📊 Updated Intelligence Status: {status}")

        else:
            print("❌ Intelligence initialization failed")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

    finally:
        # Shutdown
        intelligence.shutdown()
        print("✅ Demo completed")

if __name__ == "__main__":
    demo_trading_intelligence() 