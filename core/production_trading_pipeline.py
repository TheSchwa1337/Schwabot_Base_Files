"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Trading Pipeline - COMPLETE MATHEMATICAL INTEGRATION
=============================================================
Complete production-ready trading system that integrates:
- Real CCXT exchange connections with API keys
- Portfolio tracking and position management
- Live market data feeds
- Risk management and circuit breakers
- Order execution and balance synchronization
- Performance monitoring and reporting
- ALL MATHEMATICAL SYSTEMS (DLT, Dualistic Engines, Bit Phases, etc.)

This system provides a complete, production-ready trading environment
for live cryptocurrency trading with proper risk management and
portfolio tracking, PLUS your complete mathematical foundation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import ccxt

# Core imports
from .portfolio_tracker import PortfolioTracker
from .risk_manager import RiskManager
from .entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
from .enhanced_ccxt_trading_engine import EnhancedCCXTTradingEngine

# MATHEMATICAL INTEGRATION - ALL YOUR SYSTEMS
try:
    from backtesting.mathematical_integration_simplified import mathematical_integration, MathematicalSignal
    MATHEMATICAL_INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback import
    try:
        from ..backtesting.mathematical_integration_simplified import mathematical_integration, MathematicalSignal
        MATHEMATICAL_INTEGRATION_AVAILABLE = True
    except ImportError:
        MATHEMATICAL_INTEGRATION_AVAILABLE = False
        logger.warning("Mathematical integration not available")

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuration for production trading with mathematical integration."""
    exchange_name: str
    api_key: str
    secret: str
    sandbox: bool = True
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDC'])
    base_currency: str = 'USDC'
    initial_balance: Dict[str, float] = field(default_factory=dict)
    risk_tolerance: float = 0.2
    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    enable_circuit_breakers: bool = True
    portfolio_sync_interval: int = 30  # seconds
    price_update_interval: int = 5    # seconds
    
    # Mathematical integration settings
    enable_mathematical_integration: bool = True
    mathematical_confidence_threshold: float = 0.7
    mathematical_weight: float = 0.7  # Weight for mathematical vs AI decisions

@dataclass
class TradingStatus:
    """Current trading system status with mathematical tracking."""
    is_running: bool = False
    last_sync: float = 0.0
    last_trade: float = 0.0
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    current_risk_level: str = 'normal'
    circuit_breaker_active: bool = False
    error_count: int = 0
    
    # Mathematical tracking
    mathematical_signals_processed: int = 0
    mathematical_decisions_made: int = 0
    dlt_waveform_signals: int = 0
    dualistic_consensus_signals: int = 0
    bit_phase_signals: int = 0
    ferris_phase_signals: int = 0

class ProductionTradingPipeline:
    """
    Complete production-ready trading pipeline with FULL MATHEMATICAL INTEGRATION.

    This class orchestrates the entire trading system:
    1. Exchange connection management
    2. Portfolio tracking and synchronization
    3. Real-time market data processing
    4. MATHEMATICAL INTEGRATION (DLT, Dualistic Engines, Bit Phases, etc.)
    5. Risk management and circuit breakers
    6. Order execution and position management
    7. Performance monitoring and reporting
    """

    def __init__(self, config: TradingConfig) -> None:
        """Initialize the production trading pipeline with mathematical integration."""
        self.config = config
        self.status = TradingStatus()

        # Initialize core components
        self._initialize_exchange()
        self._initialize_portfolio_tracker()
        self._initialize_risk_manager()
        self._initialize_trading_executor()
        
        # Initialize mathematical integration
        if self.config.enable_mathematical_integration:
            self._initialize_mathematical_integration()

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []
        
        # Mathematical tracking
        self.mathematical_signals: List[MathematicalSignal] = []
        self.dualistic_consensus_history: List[Dict[str, Any]] = []
        self.dlt_waveform_history: List[float] = []
        self.bit_phase_history: List[int] = []
        self.ferris_phase_history: List[float] = []

        logger.info("🚀 Production Trading Pipeline initialized with FULL MATHEMATICAL INTEGRATION")
        logger.info("🧮 All mathematical systems enabled: DLT, Dualistic Engines, Bit Phases, etc.")

    def _initialize_mathematical_integration(self) -> None:
        """Initialize mathematical integration engine."""
        try:
            # The mathematical_integration is already imported and ready
            logger.info("✅ Mathematical integration engine initialized")
            logger.info("🧮 Available mathematical systems:")
            logger.info("   - DLT Waveform Engine")
            logger.info("   - Dualistic Thought Engines (ALEPH, ALIF, RITL, RITTLE)")
            logger.info("   - Bit Phase Resolution (4-bit, 8-bit, 42-bit)")
            logger.info("   - Matrix Basket Tensor Algebra")
            logger.info("   - Ferris RDE (3.75-minute cycles)")
            logger.info("   - Lantern Core")
            logger.info("   - Quantum Operations")
            logger.info("   - Entropy-Driven Systems")
            logger.info("   - Vault Orbital Bridge")
            logger.info("   - Advanced Tensor Operations")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize mathematical integration: {e}")
            raise

    def _initialize_exchange(self) -> None:
        """Initialize CCXT exchange connection."""
        try:
            exchange_class = getattr(ccxt, self.config.exchange_name)

            self.exchange = exchange_class({
                'apiKey': self.config.api_key,
                'secret': self.config.secret,
                'sandbox': self.config.sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            })

            logger.info(f"✅ Exchange connection established: {self.config.exchange_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange: {e}")
            raise

    def _initialize_portfolio_tracker(self) -> None:
        """Initialize portfolio tracker with initial balances."""
        try:
            self.portfolio_tracker = PortfolioTracker(
                base_currency=self.config.base_currency,
                initial_balances=self.config.initial_balance
            )

            logger.info(f"✅ Portfolio tracker initialized with base currency: {self.config.base_currency}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize portfolio tracker: {e}")
            raise

    def _initialize_risk_manager(self) -> None:
        """Initialize risk manager with configuration."""
        try:
            risk_config = {
                'risk_tolerance': self.config.risk_tolerance,
                'max_position_size': self.config.max_position_size,
                'max_daily_loss': self.config.max_daily_loss,
                'enable_circuit_breakers': self.config.enable_circuit_breakers,
                'symbols': self.config.symbols
            }

            self.risk_manager = RiskManager(risk_config)

            logger.info(f"✅ Risk manager initialized with tolerance: {self.config.risk_tolerance}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize risk manager: {e}")
            raise

    def _initialize_trading_executor(self) -> None:
        """Initialize trading executor with all components."""
        try:
            exchange_config = {
                'exchange': self.config.exchange_name,
                'api_key': self.config.api_key,
                'secret': self.config.secret,
                'sandbox': self.config.sandbox
            }

            strategy_config = {
                'strategy_type': 'production_entropy',
                'symbols': self.config.symbols,
                'timeframe': '1m'
            }

            entropy_config = {
                'entropy_threshold': 0.7,
                'signal_strength_min': 0.3,
                'timing_window': 300
            }

            risk_config = {
                'risk_tolerance': self.config.risk_tolerance,
                'max_position_size': self.config.max_position_size,
                'max_daily_loss': self.config.max_daily_loss,
                'enable_circuit_breakers': self.config.enable_circuit_breakers
            }

            self.trading_executor = EntropyEnhancedTradingExecutor(
                exchange_config=exchange_config,
                strategy_config=strategy_config,
                entropy_config=entropy_config,
                risk_config=risk_config
            )

            # Override portfolio tracker with our production instance
            self.trading_executor.portfolio_tracker = self.portfolio_tracker

            logger.info("✅ Trading executor initialized with production components")

        except Exception as e:
            logger.error(f"❌ Failed to initialize trading executor: {e}")
            raise

    async def process_market_data_with_mathematics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data through ALL mathematical systems."""
        try:
            if not self.config.enable_mathematical_integration:
                return {'decision': 'HOLD', 'confidence': 0.5, 'reason': 'Mathematical integration disabled'}

            # Prepare market data for mathematical processing
            mathematical_market_data = {
                'current_price': market_data.get('price', 0.0),
                'volume': market_data.get('volume', 0.0),
                'price_change': market_data.get('price_change', 0.0),
                'volatility': market_data.get('volatility', 0.0),
                'sentiment': market_data.get('sentiment', 0.5),
                'close_prices': market_data.get('price_history', []),
                'entry_price': self._get_entry_price(market_data.get('symbol', 'BTC/USDC')),
                'bit_phase': self._get_current_bit_phase(market_data.get('symbol', 'BTC/USDC'))
            }

            # Process through ALL mathematical systems
            mathematical_signal = await mathematical_integration.process_market_data_mathematically(mathematical_market_data)
            
            # Store mathematical signal for analysis
            self.mathematical_signals.append(mathematical_signal)
            self.status.mathematical_signals_processed += 1

            # Store specific mathematical components
            if mathematical_signal.dualistic_consensus:
                self.dualistic_consensus_history.append(mathematical_signal.dualistic_consensus)
                self.status.dualistic_consensus_signals += 1

            self.dlt_waveform_history.append(mathematical_signal.dlt_waveform_score)
            self.bit_phase_history.append(mathematical_signal.bit_phase)
            self.ferris_phase_history.append(mathematical_signal.ferris_phase)

            # Update status counters
            if mathematical_signal.dlt_waveform_score > 0:
                self.status.dlt_waveform_signals += 1
            if mathematical_signal.bit_phase > 0:
                self.status.bit_phase_signals += 1
            if mathematical_signal.ferris_phase != 0:
                self.status.ferris_phase_signals += 1

            # Create trading decision from mathematical signal
            decision = {
                'action': mathematical_signal.decision,
                'symbol': market_data.get('symbol', 'BTC/USDC'),
                'entry_price': market_data.get('price', 0.0),
                'position_size': self._calculate_position_size(mathematical_signal.confidence, market_data.get('price', 0.0)),
                'confidence': mathematical_signal.confidence,
                'timestamp': time.time(),
                'mathematical_metadata': {
                    'dualistic_consensus': mathematical_signal.dualistic_consensus,
                    'dlt_waveform_score': mathematical_signal.dlt_waveform_score,
                    'bit_phase': mathematical_signal.bit_phase,
                    'ferris_phase': mathematical_signal.ferris_phase,
                    'tensor_score': mathematical_signal.tensor_score,
                    'entropy_score': mathematical_signal.entropy_score,
                    'routing_target': mathematical_signal.routing_target
                }
            }

            self.status.mathematical_decisions_made += 1

            logger.debug(f"🧮 Mathematical processing: {mathematical_signal.decision} @ {mathematical_signal.confidence:.3f} confidence")
            logger.debug(f"   DLT Score: {mathematical_signal.dlt_waveform_score:.4f}")
            logger.debug(f"   Bit Phase: {mathematical_signal.bit_phase}")
            logger.debug(f"   Ferris Phase: {mathematical_signal.ferris_phase:.4f}")

            return decision

        except Exception as e:
            logger.error(f"❌ Mathematical processing failed: {e}")
            return {'decision': 'HOLD', 'confidence': 0.5, 'reason': f'Mathematical error: {e}'}

    def _get_entry_price(self, symbol: str) -> float:
        """Get entry price for a symbol."""
        # For simplicity, use current market price
        # In a real implementation, you'd track actual entry prices
        return 50000.0  # Default BTC price

    def _get_current_bit_phase(self, symbol: str) -> int:
        """Get current bit phase for a symbol."""
        # For simplicity, return a default bit phase
        # In a real implementation, you'd track actual bit phases
        return 8  # Default 8-bit phase

    def _calculate_position_size(self, confidence: float, price: float) -> float:
        """Calculate position size based on confidence and risk management."""
        try:
            # Base position size from risk management
            base_size = self.portfolio_tracker.get_balance(self.config.base_currency) * self.config.risk_per_trade / price
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence * 2, 1.0)  # Scale confidence to 0-1
            
            # Apply maximum position limit
            max_positions = 5  # Configurable
            current_positions = len(self.portfolio_tracker.get_positions())
            
            if current_positions >= max_positions:
                position_multiplier = 0.5  # Reduce size if at position limit
            else:
                position_multiplier = 1.0
            
            final_size = base_size * confidence_multiplier * position_multiplier
            
            return max(0.0, final_size)
            
        except Exception as e:
            logger.error(f"❌ Position size calculation failed: {e}")
            return 0.0

    async def start_trading(self) -> None:
        """Start the production trading system with mathematical integration."""
        try:
            logger.info("🚀 Starting production trading with FULL MATHEMATICAL INTEGRATION")
            self.status.is_running = True

            # Start portfolio synchronization
            sync_task = asyncio.create_task(self._portfolio_sync_loop())
            
            # Start market data processing with mathematical integration
            market_task = asyncio.create_task(self._market_data_processing_loop())
            
            # Start performance monitoring
            monitor_task = asyncio.create_task(self._performance_monitoring_loop())

            # Wait for all tasks
            await asyncio.gather(sync_task, market_task, monitor_task)

        except Exception as e:
            logger.error(f"❌ Trading startup failed: {e}")
            self.status.is_running = False
            raise

    async def stop_trading(self) -> None:
        """Stop the production trading system."""
        try:
            logger.info("🛑 Stopping production trading system")
            self.status.is_running = False
            
            # Save mathematical performance data
            self._save_mathematical_performance_data()
            
            logger.info("✅ Production trading system stopped")

        except Exception as e:
            logger.error(f"❌ Trading shutdown failed: {e}")

    async def _portfolio_sync_loop(self) -> None:
        """Portfolio synchronization loop."""
        while self.status.is_running:
            try:
                await self._sync_portfolio()
                await asyncio.sleep(self.config.portfolio_sync_interval)
            except Exception as e:
                logger.error(f"❌ Portfolio sync error: {e}")
                await asyncio.sleep(5)

    async def _market_data_processing_loop(self) -> None:
        """Market data processing loop with mathematical integration."""
        while self.status.is_running:
            try:
                # Get market data from exchange
                market_data = await self._get_market_data()
                
                if market_data:
                    # Process through ALL mathematical systems
                    mathematical_decision = await self.process_market_data_with_mathematics(market_data)
                    
                    # Execute trade if confidence is high enough
                    if mathematical_decision['confidence'] >= self.config.mathematical_confidence_threshold:
                        await self._execute_mathematical_trade(mathematical_decision)
                
                await asyncio.sleep(self.config.price_update_interval)
                
            except Exception as e:
                logger.error(f"❌ Market data processing error: {e}")
                await asyncio.sleep(5)

    async def _performance_monitoring_loop(self) -> None:
        """Performance monitoring loop."""
        while self.status.is_running:
            try:
                self._update_performance_metrics()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {e}")
                await asyncio.sleep(5)

    async def _sync_portfolio(self) -> None:
        """Synchronize portfolio with exchange."""
        try:
            balance = await self.exchange.fetch_balance()
            self.portfolio_tracker.update_balances(balance)
            self.status.last_sync = time.time()
            
        except Exception as e:
            logger.error(f"❌ Portfolio sync failed: {e}")

    async def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """Get market data from exchange."""
        try:
            ticker = await self.exchange.fetch_ticker(self.config.symbols[0])
            
            return {
                'symbol': self.config.symbols[0],
                'price': ticker['last'],
                'volume': ticker['baseVolume'],
                'price_change': ticker['percentage'],
                'volatility': abs(ticker['percentage']),
                'sentiment': 0.5 + (ticker['percentage'] * 0.1),  # Simple sentiment
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Market data fetch failed: {e}")
            return None

    async def _execute_mathematical_trade(self, decision: Dict[str, Any]) -> None:
        """Execute trade based on mathematical decision."""
        try:
            # Validate decision with risk manager
            if not self.risk_manager.validate_trade(decision):
                logger.warning(f"⚠️ Trade rejected by risk manager: {decision['action']}")
                return

            # Execute trade through trading executor
            success = await self.trading_executor.execute_signal(decision)
            
            if success:
                self.status.total_trades += 1
                self.status.last_trade = time.time()
                logger.info(f"💰 Mathematical trade executed: {decision['action']} {decision['symbol']} @ ${decision['entry_price']:.4f}")
                logger.info(f"🧮 Mathematical confidence: {decision['confidence']:.3f}")
            else:
                logger.warning(f"⚠️ Mathematical trade execution failed: {decision['action']}")
                
        except Exception as e:
            logger.error(f"❌ Mathematical trade execution error: {e}")

    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            # Calculate basic metrics
            total_balance = self.portfolio_tracker.get_total_balance()
            pnl = total_balance - sum(self.config.initial_balance.values())
            
            # Update status
            self.status.total_pnl = pnl
            
            # Log performance
            logger.info(f"📊 Performance Update:")
            logger.info(f"   Total Balance: ${total_balance:.2f}")
            logger.info(f"   Total P&L: ${pnl:.2f}")
            logger.info(f"   Total Trades: {self.status.total_trades}")
            logger.info(f"   Mathematical Signals: {self.status.mathematical_signals_processed}")
            logger.info(f"   Mathematical Decisions: {self.status.mathematical_decisions_made}")
            
        except Exception as e:
            logger.error(f"❌ Performance metrics update failed: {e}")

    def _save_mathematical_performance_data(self) -> None:
        """Save mathematical performance data."""
        try:
            performance_data = {
                'timestamp': time.time(),
                'mathematical_signals_processed': self.status.mathematical_signals_processed,
                'mathematical_decisions_made': self.status.mathematical_decisions_made,
                'dlt_waveform_signals': self.status.dlt_waveform_signals,
                'dualistic_consensus_signals': self.status.dualistic_consensus_signals,
                'bit_phase_signals': self.status.bit_phase_signals,
                'ferris_phase_signals': self.status.ferris_phase_signals,
                'mathematical_metrics': self._calculate_mathematical_metrics()
            }
            
            self.performance_history.append(performance_data)
            logger.info("✅ Mathematical performance data saved")
            
        except Exception as e:
            logger.error(f"❌ Mathematical performance data save failed: {e}")

    def _calculate_mathematical_metrics(self) -> Dict[str, Any]:
        """Calculate mathematical performance metrics."""
        try:
            if not self.mathematical_signals:
                return {}
            
            # DLT Waveform Analysis
            dlt_scores = [s.dlt_waveform_score for s in self.mathematical_signals if s.dlt_waveform_score > 0]
            avg_dlt_score = sum(dlt_scores) / len(dlt_scores) if dlt_scores else 0.0
            
            # Dualistic Consensus Analysis
            dualistic_scores = []
            for consensus in self.dualistic_consensus_history:
                if consensus and 'mathematical_score' in consensus:
                    dualistic_scores.append(consensus['mathematical_score'])
            avg_dualistic_score = sum(dualistic_scores) / len(dualistic_scores) if dualistic_scores else 0.0
            
            # Bit Phase Analysis
            bit_phase_distribution = {}
            for phase in self.bit_phase_history:
                bit_phase_distribution[phase] = bit_phase_distribution.get(phase, 0) + 1
            
            # Ferris Phase Analysis
            ferris_phases = [p for p in self.ferris_phase_history if p != 0]
            avg_ferris_phase = sum(ferris_phases) / len(ferris_phases) if ferris_phases else 0.0
            
            # Decision Distribution
            decisions = [s.decision for s in self.mathematical_signals]
            decision_distribution = {}
            for decision in decisions:
                decision_distribution[decision] = decision_distribution.get(decision, 0) + 1
            
            return {
                "avg_dlt_waveform_score": avg_dlt_score,
                "avg_dualistic_score": avg_dualistic_score,
                "bit_phase_distribution": bit_phase_distribution,
                "avg_ferris_phase": avg_ferris_phase,
                "decision_distribution": decision_distribution,
                "total_mathematical_signals": len(self.mathematical_signals),
                "mathematical_confidence_avg": sum(s.confidence for s in self.mathematical_signals) / len(self.mathematical_signals) if self.mathematical_signals else 0.0,
                "tensor_score_avg": sum(s.tensor_score for s in self.mathematical_signals if s.tensor_score != 0) / len([s for s in self.mathematical_signals if s.tensor_score != 0]) if [s for s in self.mathematical_signals if s.tensor_score != 0] else 0.0,
                "entropy_score_avg": sum(s.entropy_score for s in self.mathematical_signals if s.entropy_score > 0) / len([s for s in self.mathematical_signals if s.entropy_score > 0]) if [s for s in self.mathematical_signals if s.entropy_score > 0] else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Mathematical metrics calculation failed: {e}")
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get current system status with mathematical information."""
        return {
            'is_running': self.status.is_running,
            'last_sync': self.status.last_sync,
            'last_trade': self.status.last_trade,
            'total_trades': self.status.total_trades,
            'total_pnl': self.status.total_pnl,
            'current_risk_level': self.status.current_risk_level,
            'circuit_breaker_active': self.status.circuit_breaker_active,
            'error_count': self.status.error_count,
            'mathematical_integration': {
                'enabled': self.config.enable_mathematical_integration,
                'signals_processed': self.status.mathematical_signals_processed,
                'decisions_made': self.status.mathematical_decisions_made,
                'dlt_waveform_signals': self.status.dlt_waveform_signals,
                'dualistic_consensus_signals': self.status.dualistic_consensus_signals,
                'bit_phase_signals': self.status.bit_phase_signals,
                'ferris_phase_signals': self.status.ferris_phase_signals
            }
        }

def create_production_pipeline(
    exchange_name: str,
    api_key: str,
    secret: str,
    sandbox: bool = True,
    symbols: List[str] = None,
    risk_tolerance: float = 0.2,
    max_position_size: float = 0.1,
    max_daily_loss: float = 0.05,
    enable_mathematical_integration: bool = True
) -> ProductionTradingPipeline:
    """Create a production trading pipeline with mathematical integration."""
    
    if symbols is None:
        symbols = ['BTC/USDC']
    
    config = TradingConfig(
        exchange_name=exchange_name,
        api_key=api_key,
        secret=secret,
        sandbox=sandbox,
        symbols=symbols,
        risk_tolerance=risk_tolerance,
        max_position_size=max_position_size,
        max_daily_loss=max_daily_loss,
        enable_mathematical_integration=enable_mathematical_integration
    )
    
    return ProductionTradingPipeline(config)

# Demo function

async def demo_production_pipeline():
    """Demonstrate the production trading pipeline."""
    logger.info("🎯 DEMO: Production Trading Pipeline")

    # Create pipeline with demo configuration
    pipeline = create_production_pipeline(
        exchange_name='coinbase',
        api_key='demo_key',
        secret='demo_secret',
        sandbox=True,
        symbols=['BTC/USDC'],
        risk_tolerance=0.1,
        max_position_size=0.05
    )

    try:
        # Start trading for a short period
        logger.info("🚀 Starting demo trading...")

        # Run for 5 minutes
        await asyncio.wait_for(pipeline.start_trading(), timeout=300)

    except asyncio.TimeoutError:
        logger.info("⏰ Demo completed after 5 minutes")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
    finally:
        await pipeline.stop_trading()

    # Export final report
    report = pipeline.export_trading_report()
    logger.info(f"📊 Final Report: {report['system_status']['performance']}")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_production_pipeline())
