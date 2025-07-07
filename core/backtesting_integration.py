from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging
import numpy as np
import time

from core.gpu_shader_integration import GPUShaderIntegration
from core.secure_exchange_manager import SecureExchangeManager
from core.system_state_profiler import SystemStateProfiler
from core.vectorized_profit_orchestrator import VectorizedProfitOrchestrator
from core.unified_market_data_pipeline import UnifiedMarketDataPipeline
from core.phantom_registry import PhantomRegistry
from core.phantom_logger import PhantomLogger
from core.risk_manager import RiskManager
from core.backtest_visualization import BacktestVisualizer
from utils.safe_print import safe_print
from utils.cuda_helper import get_cuda_status, USING_CUDA

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    trading_pairs: List[str]
    use_gpu: bool = True
    data_source: str = "historical"  # historical, synthetic, or live
    risk_profile: str = "moderate"
    enable_visualization: bool = True
    save_results: bool = True
    results_dir: str = "backtest_results"
    
    # Advanced settings
    gpu_batch_size: int = 1024
    price_data_interval: str = "1m"
    strategy_update_interval: int = 60  # seconds
    max_open_positions: int = 5
    max_leverage: float = 1.0
    
    # System integration
    enable_phantom_logging: bool = True
    enable_risk_management: bool = True
    enable_gpu_acceleration: bool = True
    
    timestamp: float = field(default_factory=time.time)

class BacktestingIntegration:
    """Integrates backtesting capabilities with core system components."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.safe_print = safe_print
        
        # Initialize core components
        self.market_data = UnifiedMarketDataPipeline()
        self.risk_manager = RiskManager()
        self.profit_orchestrator = VectorizedProfitOrchestrator()
        self.system_profiler = SystemStateProfiler()
        self.phantom_registry = PhantomRegistry()
        self.phantom_logger = PhantomLogger()
        
        # Initialize visualization
        if config.enable_visualization:
            self.visualizer = BacktestVisualizer(config.results_dir)
        
        # GPU acceleration
        if config.use_gpu and USING_CUDA:
            self.gpu_integration = GPUShaderIntegration()
            safe_print("GPU acceleration enabled for backtesting")
        else:
            safe_print("Running in CPU-only mode")
            
        # State tracking
        self.portfolio_value_history: List[Decimal] = [config.initial_capital]
        self.trade_history: List[Dict[str, Any]] = []
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.metrics: Dict[str, float] = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0
        }
        
    async def initialize(self):
        """Initialize all system components for backtesting."""
        try:
            # Initialize market data pipeline
            await self.market_data.initialize(
                trading_pairs=self.config.trading_pairs,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                interval=self.config.price_data_interval
            )
            
            # Initialize GPU resources if enabled
            if self.config.use_gpu and hasattr(self, 'gpu_integration'):
                await self.gpu_integration.initialize_resources(
                    batch_size=self.config.gpu_batch_size
                )
                
            # Initialize risk management
            if self.config.enable_risk_management:
                self.risk_manager.initialize(
                    max_positions=self.config.max_open_positions,
                    max_leverage=self.config.max_leverage,
                    risk_profile=self.config.risk_profile
                )
                
            # Initialize phantom components
            if self.config.enable_phantom_logging:
                self.phantom_registry.initialize()
                self.phantom_logger.initialize(
                    log_dir=self.config.results_dir
                )
                
            safe_print("✅ Backtesting system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize backtesting system: {e}")
            return False
            
    async def run_backtest(self) -> Dict[str, Any]:
        """Execute the backtest simulation."""
        try:
            safe_print(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
            
            async for market_data in self.market_data.stream_historical_data():
                # Process market data through GPU if enabled
                if self.config.use_gpu and hasattr(self, 'gpu_integration'):
                    processed_data = await self.gpu_integration.process_market_data(
                        market_data, 
                        batch_size=self.config.gpu_batch_size
                    )
                else:
                    processed_data = market_data
                    
                # Generate trading signals
                signals = await self.profit_orchestrator.generate_signals(
                    processed_data,
                    self.current_positions
                )
                
                # Apply risk management
                if self.config.enable_risk_management:
                    signals = self.risk_manager.filter_signals(signals)
                    
                # Execute approved signals
                for signal in signals:
                    execution_result = await self._execute_trade(signal)
                    if execution_result["executed"]:
                        self.trade_history.append(execution_result)
                        
                # Update system state
                await self._update_system_state(processed_data)
                
                # Log phantom data if enabled
                if self.config.enable_phantom_logging:
                    self.phantom_logger.log_system_state({
                        "market_data": processed_data,
                        "signals": signals,
                        "portfolio": self.current_positions,
                        "metrics": self.metrics
                    })
                    
            # Calculate final metrics
            self._calculate_performance_metrics()
            
            # Generate visualizations if enabled
            if self.config.enable_visualization and hasattr(self, 'visualizer'):
                self.visualizer.create_performance_charts(
                    portfolio_history=self.portfolio_value_history,
                    trade_history=self.trade_history,
                    metrics=self.metrics
                )
                self.visualizer.create_trade_analysis(
                    trade_history=self.trade_history
                )
                if self.config.save_results:
                    self.visualizer.save_trade_log(self.trade_history)
            
            return {
                "metrics": self.metrics,
                "trade_history": self.trade_history,
                "portfolio_history": self.portfolio_value_history
            }
            
        except Exception as e:
            logger.error(f"Error during backtest execution: {e}")
            return {"error": str(e)}
            
    async def _execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade signal in the backtesting environment."""
        try:
            # Simulate trade execution
            execution_price = signal["price"]
            execution_size = signal["size"]
            execution_type = signal["type"]
            
            trade_result = {
                "executed": True,
                "timestamp": time.time(),
                "pair": signal["pair"],
                "type": execution_type,
                "price": execution_price,
                "size": execution_size,
                "value": execution_price * execution_size,
                "fees": execution_price * execution_size * Decimal("0.001")  # Simulated fees
            }
            
            # Update positions
            if execution_type == "buy":
                self.current_positions[signal["pair"]] = {
                    "size": execution_size,
                    "entry_price": execution_price
                }
            else:
                self.current_positions.pop(signal["pair"], None)
                
            return trade_result
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {"executed": False, "error": str(e)}
            
    async def _update_system_state(self, market_data: Dict[str, Any]):
        """Update system state with latest market data."""
        try:
            # Calculate current portfolio value
            portfolio_value = self.config.initial_capital
            for pair, position in self.current_positions.items():
                current_price = market_data[pair]["price"]
                position_value = position["size"] * current_price
                portfolio_value += position_value
                
            self.portfolio_value_history.append(portfolio_value)
            
            # Update system profiler
            self.system_profiler.update_state({
                "portfolio_value": portfolio_value,
                "open_positions": len(self.current_positions),
                "market_data": market_data
            })
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
            
    def _calculate_performance_metrics(self):
        """Calculate final performance metrics."""
        try:
            portfolio_values = np.array(self.portfolio_value_history)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Calculate metrics
            self.metrics["total_return"] = (
                float(self.portfolio_value_history[-1] / self.config.initial_capital) - 1
            )
            self.metrics["sharpe_ratio"] = float(
                np.mean(returns) / np.std(returns) * np.sqrt(252)
            ) if len(returns) > 0 else 0
            self.metrics["max_drawdown"] = float(
                np.min(portfolio_values / np.maximum.accumulate(portfolio_values)) - 1
            )
            
            # Calculate trade-specific metrics
            if self.trade_history:
                winning_trades = sum(
                    1 for trade in self.trade_history 
                    if trade["type"] == "sell" and trade["price"] > trade.get("entry_price", 0)
                )
                total_trades = len([t for t in self.trade_history if t["type"] == "sell"])
                
                self.metrics["win_rate"] = winning_trades / total_trades if total_trades > 0 else 0
                
                # Calculate profit factor
                profits = sum(
                    trade["value"] - trade["fees"]
                    for trade in self.trade_history
                    if trade["type"] == "sell" and trade["price"] > trade.get("entry_price", 0)
                )
                losses = sum(
                    trade["value"] - trade["fees"]
                    for trade in self.trade_history
                    if trade["type"] == "sell" and trade["price"] <= trade.get("entry_price", 0)
                )
                
                self.metrics["profit_factor"] = float(profits / abs(losses)) if losses != 0 else float('inf')
                
            safe_print(f"Final portfolio value: ${float(self.portfolio_value_history[-1]):,.2f}")
            safe_print(f"Total return: {self.metrics['total_return']:.2%}")
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}") 