import asyncio
import logging
from typing import Any, Dict, List, Optional

from .core.clean_math_foundation import CleanMathFoundation
from .core.clean_profit_vectorization import CleanProfitVectorization
from .lantern_core import LanternEye
from .ferris_rde import FerrisRDE
from .vortex_security import get_vortex_security
from .visualizer import (
    SchwabotVisualizer,
    DataAggregator,
    PerformanceMonitor,
    TradingVisualizer,
    MathVisualizer
)

class SchwabotIntegrationHub:
    """
    Central integration hub for Schwabot, coordinating between 
    different system components and managing complex workflows.
    """
    
    def __init__(
        self, 
        initial_capital: float = 100000.0, 
        debug: bool = False,
        enable_visualization: bool = True
    ):
        """
        Initialize the Schwabot Integration Hub.
        
        Args:
            initial_capital: Starting capital for trading operations
            debug: Enable detailed logging
            enable_visualization: Enable visualization components
        """
        self.logger = logging.getLogger('SchwabotIntegrationHub')
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Core mathematical foundations
        self.math_foundation = CleanMathFoundation()
        self.profit_vectorizer = CleanProfitVectorization()
        
        # External system integrations
        self.lantern_eye = LanternEye()
        self.ferris_rde = FerrisRDE()
        
        # Security and authentication
        self.vortex_security = get_vortex_security()
        
        # Visualization components
        self.enable_visualization = enable_visualization
        if enable_visualization:
            self.visualizer = SchwabotVisualizer()
            self.data_aggregator = DataAggregator(self.visualizer)
            self.performance_monitor = PerformanceMonitor()
            self.trading_visualizer = TradingVisualizer()
            self.math_visualizer = MathVisualizer()
        else:
            self.visualizer = None
            self.data_aggregator = None
            self.performance_monitor = None
            self.trading_visualizer = None
            self.math_visualizer = None
        
        # State tracking
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.integration_state: Dict[str, Any] = {}
        
        # Tensor and data collection
        self.collected_tensors: List[Any] = []
        self.processed_data: Dict[str, Any] = {}
    
    async def initialize_systems(self):
        """
        Asynchronously initialize all connected systems.
        Provides a coordinated startup sequence.
        """
        try:
            # Parallel initialization of systems
            init_tasks = [
                self.lantern_eye.initialize(),
                self.ferris_rde.startup(),
                self.vortex_security.validate_systems()
            ]
            
            # Add visualization initialization if enabled
            if self.enable_visualization:
                init_tasks.extend([
                    self.visualizer.start(),
                    self.data_aggregator.start(),
                    self.performance_monitor.start(),
                    self.trading_visualizer.start(),
                    self.math_visualizer.start()
                ])
            
            await asyncio.gather(*init_tasks)
            
            # Perform initial mathematical calibration
            self.math_foundation.calibrate()
            self.profit_vectorizer.initialize(self.initial_capital)
            
            self.logger.info("All systems initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    async def collect_tensor_data(self) -> List[Any]:
        """
        Collect and aggregate tensor data from multiple sources.
        
        Returns:
            List of collected tensor data
        """
        try:
            # Collect from Lantern Eye
            lantern_tensors = await self.lantern_eye.extract_tensors()
            
            # Collect from Ferris RDE
            ferris_tensors = await self.ferris_rde.generate_tensors()
            
            # Combine and process tensors
            self.collected_tensors = lantern_tensors + ferris_tensors
            
            # Optional: Apply mathematical filtering
            processed_tensors = self.math_foundation.filter_tensors(self.collected_tensors)
            
            # Add to visualizer if enabled
            if self.enable_visualization and self.data_aggregator:
                for tensor in processed_tensors:
                    self.data_aggregator.add_math_data('tensor_processing', tensor, None)
            
            return processed_tensors
        except Exception as e:
            self.logger.error(f"Tensor data collection failed: {e}")
            return []
    
    async def process_market_intelligence(self) -> Dict[str, Any]:
        """
        Process and synthesize market intelligence from various sources.
        
        Returns:
            Processed market intelligence dictionary
        """
        try:
            # Collect tensor data
            tensors = await self.collect_tensor_data()
            
            # Generate market insights
            market_insights = self.profit_vectorizer.analyze_market(tensors)
            
            # Security validation
            validated_insights = await self.vortex_security.validate_insights(market_insights)
            
            # Store processed data
            self.processed_data = validated_insights
            
            # Add to visualizer if enabled
            if self.enable_visualization and self.data_aggregator:
                self.data_aggregator.add_math_data('market_analysis', validated_insights, None)
            
            return validated_insights
        except Exception as e:
            self.logger.error(f"Market intelligence processing failed: {e}")
            return {}
    
    async def execute_trading_strategy(self, strategy_params: Dict[str, Any]):
        """
        Execute a trading strategy based on collected intelligence.
        
        Args:
            strategy_params: Parameters for the trading strategy
        """
        try:
            # Validate strategy through security layer
            validated_strategy = await self.vortex_security.validate_strategy(strategy_params)
            
            # Execute strategy through Ferris RDE
            trade_results = await self.ferris_rde.execute_strategy(validated_strategy)
            
            # Update capital based on trade results
            self.current_capital += trade_results.get('profit', 0)
            
            # Add to visualizer if enabled
            if self.enable_visualization:
                if self.trading_visualizer and 'trade' in trade_results:
                    trade = trade_results['trade']
                    self.trading_visualizer.add_trade_execution(
                        trade.get('symbol', 'UNKNOWN'),
                        trade.get('side', 'unknown'),
                        trade.get('amount', 0),
                        trade.get('price', 0),
                        trade.get('order_id', 'unknown')
                    )
                
                if self.data_aggregator:
                    self.data_aggregator.add_strategy_data(
                        validated_strategy.get('strategy_id', 'unknown'),
                        'strategy_execution',
                        'completed',
                        trade_results
                    )
            
            self.logger.info(f"Strategy executed. Profit: {trade_results.get('profit', 0)}")
            return trade_results
        except Exception as e:
            self.logger.error(f"Trading strategy execution failed: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive system status.
        
        Returns:
            Dictionary with system status details
        """
        status = {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "collected_tensors": len(self.collected_tensors),
            "processed_data_keys": list(self.processed_data.keys()),
            "systems": {
                "lantern": self.lantern_eye.get_status(),
                "ferris_rde": self.ferris_rde.get_status(),
                "security": self.vortex_security.get_status()
            }
        }
        
        # Add visualization status if enabled
        if self.enable_visualization:
            status["visualization"] = {
                "visualizer": self.visualizer.get_aggregated_data() if self.visualizer else {},
                "performance": self.performance_monitor.get_current_metrics() if self.performance_monitor else {},
                "trading": self.trading_visualizer.get_all_trading_data() if self.trading_visualizer else {},
                "math": self.math_visualizer.get_all_math_data() if self.math_visualizer else {}
            }
        
        return status
    
    async def shutdown(self):
        """Shutdown all systems gracefully"""
        try:
            # Stop visualization components if enabled
            if self.enable_visualization:
                shutdown_tasks = []
                if self.visualizer:
                    shutdown_tasks.append(self.visualizer.stop())
                if self.data_aggregator:
                    shutdown_tasks.append(self.data_aggregator.stop())
                if self.performance_monitor:
                    shutdown_tasks.append(self.performance_monitor.stop())
                if self.trading_visualizer:
                    shutdown_tasks.append(self.trading_visualizer.stop())
                if self.math_visualizer:
                    shutdown_tasks.append(self.math_visualizer.stop())
                
                if shutdown_tasks:
                    await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            self.logger.info("Schwabot Integration Hub shutdown completed")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

async def create_schwabot_hub(initial_capital: float = 100000.0, debug: bool = False, enable_visualization: bool = True):
    """
    Async factory method to create and initialize the Schwabot Integration Hub.
    
    Args:
        initial_capital: Starting capital for trading operations
        debug: Enable detailed logging
        enable_visualization: Enable visualization components
    
    Returns:
        Initialized SchwabotIntegrationHub instance
    """
    hub = SchwabotIntegrationHub(initial_capital, debug, enable_visualization)
    await hub.initialize_systems()
    return hub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) 