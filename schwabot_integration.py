"""
Schwabot Integration Module
==========================

Integrates the enhanced fault bus, profit cycle navigator, and DLT waveform engine
to provide comprehensive profit-fault correlation with recursive loop detection.
This is the main orchestrator for Schwabot's profit cycle navigation.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import ccxt
import numpy as np

from core.fault_bus import FaultBus, FaultType, FaultBusEvent
from profit_cycle_navigator import ProfitCycleNavigator, ProfitVector, ProfitCycleState
from dlt_waveform_engine import DLTWaveformEngine, PhaseDomain
from schwabot_stop import SchwabotStopBook, StopPatternState

class SchwabotOrchestrator:
    """
    Main orchestrator for Schwabot's profit cycle navigation system.
    Coordinates fault detection, profit analysis, and trade execution.
    """
    
    def __init__(self, 
                 exchange_id: str = "binance",
                 initial_portfolio: float = 10000.0,
                 log_level: int = logging.INFO):
        """
        Initialize the Schwabot orchestrator
        
        Args:
            exchange_id: CCXT exchange identifier
            initial_portfolio: Initial portfolio value in USD
            log_level: Logging level
        """
        # Setup logging
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.fault_bus = FaultBus()
        self.profit_navigator = ProfitCycleNavigator(self.fault_bus, initial_portfolio)
        self.waveform_engine = DLTWaveformEngine()
        self.stop_book = SchwabotStopBook()
        
        # Exchange and trading setup
        self.exchange_id = exchange_id
        self.exchange = None
        self.trading_pairs = ["BTC/USDT", "ETH/USDT"]
        self.current_positions = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0
        }
        
        # Orchestration state
        self.is_running = False
        self.last_update = datetime.now()
        self.update_interval = timedelta(seconds=30)  # 30-second updates
        
        # Risk management
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_total_exposure = 0.3   # 30% max total exposure
        
        # Initialize hooks between components
        self._setup_component_hooks()
        
    def _setup_component_hooks(self):
        """Setup communication hooks between components"""
        
        # Waveform engine hooks
        self.waveform_engine.register_hook(
            "on_entropy_vector_generated",
            self._handle_entropy_update
        )
        
        self.waveform_engine.register_hook(
            "on_bitmap_cascade_updated",
            self._handle_bitmap_update
        )
        
        # Fault bus hooks for critical events
        @self.fault_bus.register_handler("recursive_loop")
        def handle_loop_detection(event):
            self.logger.warning(f"Recursive loop detected: {event.metadata}")
            self._emergency_stop()
        
        @self.fault_bus.register_handler("profit_anomaly")
        def handle_profit_anomaly(event):
            self.logger.info(f"Profit anomaly detected: {event.metadata}")
            self._investigate_anomaly(event)
    
    async def initialize_exchange(self, api_key: str = None, secret: str = None, sandbox: bool = True):
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            config = {
                'apiKey': api_key,
                'secret': secret,
                'sandbox': sandbox,
                'enableRateLimit': True,
            }
            
            self.exchange = exchange_class(config)
            await self.exchange.load_markets()
            self.logger.info(f"Exchange {self.exchange_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def start_trading(self):
        """Start the main trading loop"""
        if not self.exchange:
            raise RuntimeError("Exchange not initialized. Call initialize_exchange() first.")
        
        self.is_running = True
        self.logger.info("Schwabot trading started")
        
        try:
            while self.is_running:
                await self._trading_cycle()
                await asyncio.sleep(self.update_interval.total_seconds())
                
        except Exception as e:
            self.logger.error(f"Trading loop error: {e}")
            await self.stop_trading()
    
    async def _trading_cycle(self):
        """Main trading cycle - orchestrates all components"""
        try:
            current_time = datetime.now()
            
            # Update market data for all trading pairs
            market_data = {}
            for symbol in self.trading_pairs:
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    market_data[symbol] = {
                        'price': ticker['last'],
                        'volume': ticker['baseVolume'],
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'timestamp': current_time
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
                    continue
            
            # Process each trading pair
            for symbol, data in market_data.items():
                await self._process_trading_pair(symbol, data, current_time)
            
            # Dispatch fault bus events
            await self.fault_bus.dispatch(severity_threshold=0.3)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            self.last_update = current_time
            
        except Exception as e:
            self.logger.error(f"Trading cycle error: {e}")
    
    async def _process_trading_pair(self, symbol: str, market_data: Dict, timestamp: datetime):
        """Process a single trading pair through the full pipeline"""
        
        # Update profit navigator with current market state
        profit_vector = self.profit_navigator.update_market_state(
            current_price=market_data['price'],
            current_volume=market_data['volume'],
            timestamp=timestamp
        )
        
        # Update waveform engine with profit context
        self.waveform_engine.profit_navigator = self.profit_navigator
        
        # Check for trade signals
        trade_signal = self.profit_navigator.get_trade_signal()
        
        if trade_signal:
            await self._execute_trade_signal(symbol, trade_signal, market_data)
        
        # Update stop loss patterns
        if symbol in self.current_positions:
            position = self.current_positions[symbol]
            current_pnl = self._calculate_pnl(position, market_data['price'])
            
            stop_state = self.stop_book.update_pattern(
                pattern_id=f"{symbol}_position",
                value=current_pnl,
                timestamp=timestamp,
                metadata={'symbol': symbol, 'price': market_data['price']}
            )
            
            # Handle stop loss triggers
            if stop_state == StopPatternState.TRIGGERED:
                await self._emergency_exit(symbol, "Stop loss triggered")
    
    async def _execute_trade_signal(self, symbol: str, signal: Dict, market_data: Dict):
        """Execute a trade signal with proper risk management"""
        try:
            action = signal['action']
            
            if action == 'ENTER':
                await self._enter_position(symbol, signal, market_data)
            elif action == 'EXIT':
                await self._exit_position(symbol, signal.get('reason', 'Signal exit'))
                
        except Exception as e:
            self.logger.error(f"Trade execution error for {symbol}: {e}")
            
            # Create fault event for trade execution failure
            fault_event = FaultBusEvent(
                tick=int(time.time()),
                module="trade_executor",
                type=FaultType.PROFIT_CRITICAL,
                severity=0.8,
                metadata={
                    'symbol': symbol,
                    'error': str(e),
                    'signal': signal
                }
            )
            self.fault_bus.push(fault_event)
    
    async def _enter_position(self, symbol: str, signal: Dict, market_data: Dict):
        """Enter a new position with risk management"""
        
        if symbol in self.current_positions:
            self.logger.warning(f"Already have position in {symbol}")
            return
        
        # Calculate position size based on risk management
        portfolio_value = self._get_portfolio_value()
        risk_amount = portfolio_value * self.max_risk_per_trade
        
        # Adjust for signal confidence and volume weight
        confidence_multiplier = signal.get('confidence', 0.5)
        volume_weight = signal.get('volume_weight', 0.1)
        
        adjusted_risk = risk_amount * confidence_multiplier * volume_weight
        
        # Calculate quantity
        entry_price = market_data['price']
        stop_loss_price = signal.get('stop_loss', entry_price * 0.98)
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit > 0:
            quantity = adjusted_risk / risk_per_unit
        else:
            self.logger.warning("Invalid risk calculation - skipping trade")
            return
        
        # Check total exposure limit
        current_exposure = sum(pos['quantity'] * pos['entry_price'] 
                             for pos in self.current_positions.values())
        if (current_exposure + quantity * entry_price) / portfolio_value > self.max_total_exposure:
            self.logger.warning("Total exposure limit reached - skipping trade")
            return
        
        try:
            # Execute the trade (simplified for demo)
            self.logger.info(f"ENTERING {signal['direction']} position: {symbol} @ {entry_price:.2f}")
            
            # Store position information
            self.current_positions[symbol] = {
                'entry_price': entry_price,
                'quantity': quantity,
                'direction': signal['direction'],
                'entry_time': datetime.now(),
                'stop_loss': stop_loss_price,
                'take_profit': signal.get('take_profit'),
                'sha_signature': signal.get('sha_signature'),
                'confidence': confidence_multiplier
            }
            
            self.performance_metrics['total_trades'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to enter position in {symbol}: {e}")
    
    async def _exit_position(self, symbol: str, reason: str):
        """Exit a position and update performance metrics"""
        
        if symbol not in self.current_positions:
            self.logger.warning(f"No position to exit in {symbol}")
            return
        
        try:
            position = self.current_positions[symbol]
            
            # Get current market price
            ticker = await self.exchange.fetch_ticker(symbol)
            exit_price = ticker['last']
            
            # Calculate PnL
            pnl = self._calculate_pnl(position, exit_price)
            
            self.logger.info(f"EXITING position: {symbol} @ {exit_price:.2f} | PnL: {pnl:.2f} | Reason: {reason}")
            
            # Update performance metrics
            self.performance_metrics['total_profit'] += pnl
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            
            # Remove position
            del self.current_positions[symbol]
            
            # Update profit context in fault bus
            self.fault_bus.update_profit_context(pnl, int(time.time()))
            
        except Exception as e:
            self.logger.error(f"Failed to exit position in {symbol}: {e}")
    
    def _calculate_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate profit/loss for a position"""
        entry_price = position['entry_price']
        quantity = position['quantity']
        direction = 1 if position['direction'] == 'LONG' else -1
        
        price_diff = (current_price - entry_price) * direction
        return price_diff * quantity
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        # Simplified - should include current positions value
        return self.profit_navigator.portfolio_value
    
    def _handle_entropy_update(self, entropy: List[float], **kwargs):
        """Handle entropy updates from waveform engine"""
        if 'profit_vector' in kwargs:
            profit_vector = kwargs['profit_vector']
            self.logger.debug(f"Entropy update - Profit magnitude: {profit_vector.magnitude:.4f}")
    
    def _handle_bitmap_update(self, cascade_state: Dict, **kwargs):
        """Handle bitmap cascade updates"""
        self.logger.debug(f"Bitmap cascade updated: {cascade_state}")
    
    def _investigate_anomaly(self, event: FaultBusEvent):
        """Investigate profit anomalies for potential genuine profit opportunities"""
        metadata = event.metadata or {}
        anomaly_strength = metadata.get('anomaly_strength', 0.0)
        
        if anomaly_strength > 0.7:
            self.logger.info("High-strength anomaly detected - potential genuine profit tier")
            # Could trigger additional analysis or position sizing adjustments
    
    async def _emergency_stop(self):
        """Emergency stop all trading activities"""
        self.logger.critical("EMERGENCY STOP activated")
        
        # Exit all positions
        for symbol in list(self.current_positions.keys()):
            await self._exit_position(symbol, "Emergency stop")
        
        # Pause trading
        self.is_running = False
    
    async def _emergency_exit(self, symbol: str, reason: str):
        """Emergency exit for a specific symbol"""
        self.logger.warning(f"Emergency exit triggered for {symbol}: {reason}")
        await self._exit_position(symbol, f"Emergency: {reason}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if self.performance_metrics['total_trades'] > 0:
            win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            self.performance_metrics['win_rate'] = win_rate
    
    async def stop_trading(self):
        """Stop trading and cleanup"""
        self.is_running = False
        
        # Exit all positions
        for symbol in list(self.current_positions.keys()):
            await self._exit_position(symbol, "System shutdown")
        
        self.logger.info("Schwabot trading stopped")
    
    def export_comprehensive_report(self, file_path: str = None) -> str:
        """Export comprehensive trading report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics,
            'current_positions': self.current_positions,
            'fault_bus_log': json.loads(self.fault_bus.export_memory_log()),
            'correlation_matrix': json.loads(self.fault_bus.export_correlation_matrix()),
            'navigation_log': json.loads(self.profit_navigator.export_navigation_log()),
            'stop_patterns': {
                'active_patterns': {k: v.__dict__ for k, v in self.stop_book.get_active_patterns().items()},
                'pattern_history': [p.__dict__ for p in self.stop_book.get_pattern_history()[-10:]]
            }
        }
        
        output = json.dumps(report, indent=2, default=str)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(output)
        return output

# Example usage and testing
async def main():
    """Example usage of the Schwabot orchestrator"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize orchestrator
    schwabot = SchwabotOrchestrator(
        exchange_id="binance",
        initial_portfolio=10000.0
    )
    
    try:
        # Initialize exchange (in sandbox mode)
        await schwabot.initialize_exchange(sandbox=True)
        
        # Start trading for a short demo period
        print("Starting Schwabot demo trading...")
        
        # Run for 2 minutes as demo
        trading_task = asyncio.create_task(schwabot.start_trading())
        await asyncio.sleep(120)  # 2 minutes
        
        # Stop trading
        await schwabot.stop_trading()
        trading_task.cancel()
        
        # Export report
        report = schwabot.export_comprehensive_report("schwabot_demo_report.json")
        print("\n=== Demo Trading Report ===")
        print(report[:1000] + "..." if len(report) > 1000 else report)
        
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        await schwabot.stop_trading()

if __name__ == "__main__":
    asyncio.run(main()) 