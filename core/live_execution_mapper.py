# -*- coding: utf-8 -*-
"""Live Execution Mapper for Schwabot's Glyph-Driven Trading System.

Orchestrates the end-to-end process of translating glyph strategy outputs
into live or simulated trade actions, integrating risk management and
portfolio tracking. This module acts as the central hub for the
glyph-execution mapping pipeline.

Key Responsibilities:
- Receive glyph-based trade signals from the Entry/Exit Portal.
- Coordinate with Risk Manager for pre-trade risk assessment and position sizing.
- Coordinate with Trade Executor for order placement and management.
- Update Portfolio Tracker with executed trade details.
- Implement robust error handling and logging for the entire execution flow.
- Maintain a clear state of the execution pipeline for debugging and monitoring.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import sys
import os

# Add the parent directory to sys.path to allow imports from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components from the core package
try:
    from core.strategy.glyph_strategy_core import GlyphStrategyCore, GlyphStrategyResult
    from core.strategy.entry_exit_portal import EntryExitPortal, TradeSignal, SignalDirection
    from core.trade_executor import TradeExecutor, Order
    from core.risk_manager import RiskManager, RiskMetric
    from core.portfolio_tracker import PortfolioTracker, Position
except ImportError as e:
    # This block is for robustness in isolated testing environments or partial deployments
    # In a fully integrated system, these imports are expected to succeed.
    logging.error(f"Failed to import core trading components: {e}")
    GlyphStrategyCore = None
    GlyphStrategyResult = None
    EntryExitPortal = None
    TradeSignal = None
    SignalDirection = None
    TradeExecutor = None
    Order = None
    RiskManager = None
    RiskMetric = None
    PortfolioTracker = None
    Position = None

logger = logging.getLogger(__name__)

@dataclass
class ExecutionState:
    """Represents the current state of a trade execution request."""
    trade_id: str
    glyph: str
    asset: str
    initial_signal: TradeSignal
    status: str = "pending" # pending, signal_processed, risk_checked, sized, ordered, executed, failed, canceled
    timestamp: float = field(default_factory=time.time)
    risk_assessment: Optional[Dict[str, Any]] = None
    position_sizing_details: Optional[Dict[str, Any]] = None
    order_details: Optional[Dict[str, Any]] = None
    execution_details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class LiveExecutionMapper:
    """
    Orchestrates the live (or simulated) execution of glyph-driven trade signals.
    """

    def __init__(self,
                 simulation_mode: bool = True,
                 initial_portfolio_cash: Union[float, str] = 100000.0,
                 enable_risk_manager: bool = True,
                 enable_portfolio_tracker: bool = True):
        """
        Initialize the LiveExecutionMapper.

        Args:
            simulation_mode: If True, all trades are simulated.
            initial_portfolio_cash: Starting cash for the portfolio.
            enable_risk_manager: Whether to use the RiskManager for assessments.
            enable_portfolio_tracker: Whether to use the PortfolioTracker for updates.
        """
        self.simulation_mode = simulation_mode
        self.enable_risk_manager = enable_risk_manager
        self.enable_portfolio_tracker = enable_portfolio_tracker

        # Initialize core components
        if not all([GlyphStrategyCore, EntryExitPortal, TradeExecutor, RiskManager, PortfolioTracker]):
            logger.critical("One or more core trading components failed to import. LiveExecutionMapper may not function correctly.")
            # Depending on desired behavior, could raise an exception or run in a very limited mode
            raise ImportError("Critical trading components are missing. Please check your core package imports.")

        self.glyph_core = GlyphStrategyCore()
        self.portal = EntryExitPortal(
            glyph_core=self.glyph_core,
            enable_risk_management=enable_risk_manager,
            enable_portfolio_tracking=enable_portfolio_tracker
        )
        self.trade_executor = TradeExecutor(simulation_mode=simulation_mode)
        self.risk_manager = RiskManager() if enable_risk_manager else None
        self.portfolio_tracker = PortfolioTracker(initial_cash=initial_portfolio_cash) if enable_portfolio_tracker else None

        # State management
        self.execution_states: Dict[str, ExecutionState] = {}
        self.trade_id_counter = 0

        # Performance metrics
        self.stats = {
            "total_execution_requests": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "risk_rejected_executions": 0,
            "rejected_by_signal_threshold": 0, # Added for clarity
            "rejected_by_sizing": 0, # Added for clarity
            "avg_execution_flow_time": 0.0
        }

        logger.info(f"LiveExecutionMapper initialized in {'simulation' if self.simulation_mode else 'live'} mode.")

    def _generate_trade_id(self) -> str:
        """Generates a unique trade ID."""
        self.trade_id_counter += 1
        return f"TRADE-{self.trade_id_counter}-{int(time.time() * 1000)}"

    def execute_glyph_trade(self, glyph: str, volume: float, asset: str, price: float,
                            confidence_boost: float = 0.0) -> ExecutionState:
        """
        Executes a trade based on a glyph signal through the full pipeline.

        Args:
            glyph: The input glyph.
            volume: Current market volume.
            asset: The trading asset (e.g., "BTC/USD").
            price: Current asset price.
            confidence_boost: Optional confidence boost for signal generation.

        Returns:
            An ExecutionState object detailing the outcome of the trade request.
        """
        trade_id = self._generate_trade_id()
        current_time = time.time()
        self.stats["total_execution_requests"] += 1

        execution_state = ExecutionState(
            trade_id=trade_id,
            glyph=glyph,
            asset=asset,
            initial_signal=None # Will be populated if signal is generated
        )
        self.execution_states[trade_id] = execution_state

        logger.info(f"[{trade_id}] Initiating glyph trade for {glyph} on {asset} @ {price} with volume {volume}")

        try:
            # Step 1: Process Glyph Signal via EntryExitPortal
            trade_signal = self.portal.process_glyph_signal(
                glyph, volume, asset, price, confidence_boost
            )
            execution_state.initial_signal = trade_signal
            execution_state.status = "signal_processed"

            if not trade_signal:
                execution_state.status = "rejected_by_signal_threshold"
                execution_state.error_message = "Signal confidence too low or no signal generated."
                self.stats["rejected_by_signal_threshold"] += 1 # Updated stat
                logger.warning(f"[{trade_id}] Signal rejected: {execution_state.error_message}")
                return execution_state

            # Step 2: Risk Assessment and Position Sizing
            portfolio_value = self.portfolio_tracker.get_portfolio_summary()["total_value"] if self.portfolio_tracker else 0.0
            if portfolio_value <= 0:
                execution_state.status = "failed"
                execution_state.error_message = "Portfolio value is zero or negative. Cannot execute trades."
                self.stats["failed_executions"] += 1
                logger.error(f"[{trade_id}] {execution_state.error_message}")
                return execution_state

            position_sizing = self.portal.calculate_position_size(trade_signal, portfolio_value)
            execution_state.position_sizing_details = position_sizing.__dict__ # Convert dataclass to dict
            execution_state.status = "position_sized"

            size_to_execute = position_sizing.risk_adjusted_size

            if size_to_execute <= 0:
                execution_state.status = "rejected_by_sizing"
                execution_state.error_message = f"Calculated position size is zero ({size_to_execute:.4f}). Trade not executed."
                self.stats["rejected_by_sizing"] += 1 # Updated stat
                logger.warning(f"[{trade_id}] {execution_state.error_message}")
                return execution_state

            # Step 3: Execute Trade via TradeExecutor
            order_result = self.trade_executor.place_order(
                asset,
                trade_signal.direction.value,
                size_to_execute,
                price
            )
            execution_state.order_details = order_result
            execution_state.status = "order_placed"

            if order_result.get("status") not in ["filled", "dry_run_success"]:
                execution_state.status = "failed"
                execution_state.error_message = f"Order placement failed: {order_result.get('error', 'Unknown error')}"
                self.stats["failed_executions"] += 1
                logger.error(f"[{trade_id}] {execution_state.error_message}")
                return execution_state

            # Step 4: Update Portfolio Tracker
            if self.portfolio_tracker:
                self.portfolio_tracker.update_position(
                    asset,
                    trade_signal.direction.value,
                    order_result.get("executed_quantity", size_to_execute), # Use actual executed quantity if available
                    order_result.get("executed_price", price), # Use actual executed price if available
                    order_result.get("fees", 0.0)
                )
                execution_state.status = "portfolio_updated"
                logger.info(f"[{trade_id}] Portfolio updated for {asset}. Current cash: {self.portfolio_tracker.cash:.2f}")

            execution_state.status = "executed_successfully"
            execution_state.execution_details = order_result
            self.stats["successful_executions"] += 1
            logger.info(f"[{trade_id}] Trade executed successfully for {glyph} ({asset} {trade_signal.direction.value} {size_to_execute:.4f} @ {price})")

        except Exception as e:
            execution_state.status = "failed"
            execution_state.error_message = f"An unexpected error occurred during trade execution: {str(e)}"
            self.stats["failed_executions"] += 1
            logger.critical(f"[{trade_id}] CRITICAL ERROR: {execution_state.error_message}", exc_info=True)

        finally:
            # Update average execution flow time
            execution_flow_time = time.time() - current_time
            total_completed = self.stats["successful_executions"] + self.stats["failed_executions"] + self.stats["rejected_by_signal_threshold"] + self.stats["rejected_by_sizing"] + self.stats["failed_executions"]
            if total_completed > 0:
                self.stats["avg_execution_flow_time"] = (
                    (self.stats["avg_execution_flow_time"] * (total_completed - 1) + execution_flow_time) / total_completed
                )
            execution_state.metadata["total_flow_time"] = execution_flow_time
            logger.debug(f"[{trade_id}] Execution flow completed in {execution_flow_time:.4f} seconds with status: {execution_state.status}")

        return execution_state

    def get_execution_state(self, trade_id: str) -> Optional[ExecutionState]:
        """Retrieves the state of a specific trade execution."""
        return self.execution_states.get(trade_id)

    def get_all_execution_states(self) -> Dict[str, ExecutionState]:
        """Returns all tracked execution states."""
        return self.execution_states.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Returns the overall performance statistics of the mapper."""
        stats = self.stats.copy()
        if self.portfolio_tracker:
            stats["portfolio_summary"] = self.portfolio_tracker.get_portfolio_summary()
        if self.trade_executor:
            stats["trade_executor_stats"] = self.trade_executor.get_performance_stats()
        if self.risk_manager:
            stats["risk_manager_stats"] = self.risk_manager.get_performance_stats()
        return stats

    def reset_system(self):
        """Resets all integrated components and mapper state."""
        self.glyph_core.reset_memory()
        self.portal.clear_signals()
        if self.portfolio_tracker:
            self.portfolio_tracker.reset_portfolio()
        # Note: TradeExecutor and RiskManager don't have direct reset methods defined,
        # so they'll retain their state unless re-instantiated.
        # If needed, add reset methods to those classes.

        self.execution_states = {}
        self.trade_id_counter = 0
        self.stats = {
            "total_execution_requests": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "risk_rejected_executions": 0,
            "rejected_by_signal_threshold": 0,
            "rejected_by_sizing": 0,
            "avg_execution_flow_time": 0.0
        }
        logger.info("LiveExecutionMapper and integrated components reset.")

def main():
    """Demonstrate LiveExecutionMapper functionality."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("\n--- Live Execution Mapper Demo (Simulation Mode) ---")

    # Initialize mapper in simulation mode
    mapper = LiveExecutionMapper(simulation_mode=True, initial_portfolio_cash=100000.0)

    # Scenario 1: Successful trade
    print("\nScenario 1: Successful Trade (Glyph 'brain')")
    state1 = mapper.execute_glyph_trade(
        glyph='brain',
        volume=3.5e6,
        asset="BTC/USD",
        price=50000.0,
        confidence_boost=0.1
    )
    print(f"Trade ID: {state1.trade_id}, Final Status: {state1.status}")
    if state1.execution_details:
        print(f"  Executed Size: {state1.execution_details.get('executed_quantity'):.4f}, "
              f"Price: {state1.execution_details.get('executed_price'):.2f}")
    print(f"  Error: {state1.error_message}")
    print(f"  Portfolio Cash: {mapper.portfolio_tracker.cash:.2f}")

    # Scenario 2: Signal rejected due to low confidence (simulate by setting very low confidence boost)
    print("\nScenario 2: Signal Rejected (Glyph 'skull', low confidence)")
    state2 = mapper.execute_glyph_trade(
        glyph='skull',
        volume=1.0e6,
        asset="ETH/USD",
        price=3000.0,
        confidence_boost=-0.5 # Force low confidence
    )
    print(f"Trade ID: {state2.trade_id}, Final Status: {state2.status}")
    print(f"  Error: {state2.error_message}")
    print(f"  Portfolio Cash: {mapper.portfolio_tracker.cash:.2f}") # Should be unchanged

    # Scenario 3: Trade rejected by position sizing (simulate by having 0 portfolio value)
    print("\nScenario 3: Trade Rejected by Sizing (Glyph 'fire', no funds)")
    mapper_no_funds = LiveExecutionMapper(simulation_mode=True, initial_portfolio_cash=0.0)
    state3 = mapper_no_funds.execute_glyph_trade(
        glyph='fire',
        volume=4.0e6,
        asset="LTC/USD",
        price=200.0
    )
    print(f"Trade ID: {state3.trade_id}, Final Status: {state3.status}")
    print(f"  Error: {state3.error_message}")
    print(f"  Portfolio Cash (no funds): {mapper_no_funds.portfolio_tracker.cash:.2f}")

    # Scenario 4: Multiple trades and check performance stats
    print("\nScenario 4: Multiple Trades and Performance Stats")
    mapper_multi = LiveExecutionMapper(simulation_mode=True, initial_portfolio_cash=50000.0)
    mapper_multi.execute_glyph_trade('hourglass', 2.0e6, 'ADA/USD', 0.5)
    mapper_multi.execute_glyph_trade('tornado', 5.0e6, 'SOL/USD', 150.0)
    mapper_multi.execute_glyph_trade('lightning', 0.5e6, 'XRP/USD', 0.6, confidence_boost=0.01) # Low volume, may reject

    print("\n--- Overall Performance Statistics ---")
    stats = mapper_multi.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (float, type(Decimal('1.0')))):
                    print(f"    {sub_key}: {sub_value:.2f}")
                else:
                    print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n--- All Execution States ---")
    for trade_id, state in mapper_multi.get_all_execution_states().items():
        print(f"  [{trade_id}] Glyph: {state.glyph}, Asset: {state.asset}, Status: {state.status}, Error: {state.error_message}")

    # Reset system
    print("\n--- Resetting the system ---")
    mapper.reset_system()
    print(f"Initial portfolio cash after reset: {mapper.portfolio_tracker.cash:.2f}")
    print(f"Total execution requests after reset: {mapper.get_performance_stats()['total_execution_requests']}")


if __name__ == "__main__":
    main() 