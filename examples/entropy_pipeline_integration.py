#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy Pipeline Integration Example

This example demonstrates how to integrate the entropy signal flow into your
main trading pipeline, showing how to use the timing cycles and signal processing
for automated trading decisions.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

# Import the entropy signal integration
from entropy_signal_integration import (
    get_entropy_integrator,
    process_entropy_signal,
    should_execute_routing,
    should_execute_tick,
)

# Import other core components
try:
    from dual_state_router import get_dual_state_router
    from neural_processing_engine import NeuralProcessingEngine
    from order_book_analyzer import OrderBookAnalyzer
except ImportError as e:
    print(f"Warning: Could not import some core components: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EntropyTradingPipeline:
    """
    Example trading pipeline that integrates entropy signals.

    This demonstrates how to use the entropy signal integration in a real
    trading scenario with timing cycles and automated decision making.
    """

    def __init__(self):
        """Initialize the entropy trading pipeline."""
        # Get the entropy integrator
        self.entropy_integrator = get_entropy_integrator()

        # Initialize core components
        self.order_book_analyzer = None
        self.dual_state_router = None
        self.neural_engine = None

        # Trading state
        self.current_position = 0.0  # Current position size
        self.available_balance = 10000.0  # Available USDC
        self.btc_price = 50000.0  # Current BTC price

        # Performance tracking
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0

        # Initialize components
        self._initialize_components()

        logger.info("Entropy Trading Pipeline initialized")

    def _initialize_components(self):
        """Initialize core trading components."""
        try:
            # Initialize order book analyzer
            self.order_book_analyzer = OrderBookAnalyzer()

            # Initialize dual state router
            self.dual_state_router = get_dual_state_router()

            # Initialize neural processing engine
            self.neural_engine = NeuralProcessingEngine()

            logger.info("All core components initialized")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")

    def generate_mock_order_book(
        self,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Generate mock order book data for demonstration."""
        try:
            # Generate realistic order book data
            base_price = self.btc_price

            # Generate bids (price, volume)
            bids = []
            for i in range(20):
                price = base_price * (1 - 0.001 * (i + 1))  # Decreasing prices
                volume = np.random.uniform(0.1, 2.0)  # Random volume
                bids.append((price, volume))

            # Generate asks (price, volume)
            asks = []
            for i in range(20):
                price = base_price * (1 + 0.001 * (i + 1))  # Increasing prices
                volume = np.random.uniform(0.1, 2.0)  # Random volume
                asks.append((price, volume))

            # Add some entropy by varying the spread
            if np.random.random() < 0.3:  # 30% chance of high entropy
                # Increase spread volatility
                spread_factor = np.random.uniform(1.5, 3.0)
                asks = [(price * spread_factor, volume) for price, volume in asks]

            return bids, asks

        except Exception as e:
            logger.error(f"Error generating mock order book: {e}")
            return [], []

    def process_tick_cycle(self):
        """Process a single tick cycle with entropy analysis."""
        try:
            # Generate mock order book data
            bids, asks = self.generate_mock_order_book()

            if not bids or not asks:
                logger.warning("No order book data available")
                return

            # Process entropy signal through the complete pipeline
            entropy_signal = process_entropy_signal(bids, asks)

            # Log the entropy signal
            logger.info(
                f"Tick Cycle - Entropy: {entropy_signal.entropy_value:.6f}, "
                f"Routing: {entropy_signal.routing_state}, "
                f"Quantum: {entropy_signal.quantum_state}, "
                f"Confidence: {entropy_signal.confidence:.3f}"
            )

            # Make trading decision based on entropy signal
            self._make_trading_decision(entropy_signal)

            # Update performance metrics
            self._update_performance_metrics()

        except Exception as e:
            logger.error(f"Error in tick cycle: {e}")

    def process_routing_cycle(self):
        """Process a single routing cycle."""
        try:
            # Get current system state
            current_state = self.entropy_integrator.get_current_state()

            # Get performance summary
            performance = self.entropy_integrator.get_performance_summary()

            logger.info(
                f"Routing Cycle - State: {current_state.get('current_entropy_state', 'UNKNOWN')}, "
                f"Detection Rate: {performance.get('average_detection_rate', 0):.3f}, "
                f"Latency: {performance.get('average_latency_ms', 0):.1f}ms"
            )

            # Adjust trading parameters based on performance
            self._adjust_trading_parameters(performance)

        except Exception as e:
            logger.error(f"Error in routing cycle: {e}")

    def _make_trading_decision(self, entropy_signal):
        """Make trading decision based on entropy signal."""
        try:
            # Get routing decision matrix from config
            routing_config = (
                self.entropy_integrator.config.get("timing_cycles", {})
                .get("routing_cycle", {})
                .get("routing_decisions", {})
            )

            # Determine trading parameters based on routing state
            if entropy_signal.routing_state == "ROUTE_ACTIVE":
                # Aggressive mode
                params = routing_config.get("AGGRESSIVE", {})
                risk_tolerance = params.get("risk_tolerance", 0.8)
                position_sizing = params.get("position_sizing", 1.2)
                action = self._determine_action(entropy_signal, risk_tolerance, position_sizing)

            elif entropy_signal.routing_state == "ROUTE_PASSIVE":
                # Passive mode
                params = routing_config.get("PASSIVE", {})
                risk_tolerance = params.get("risk_tolerance", 0.2)
                position_sizing = params.get("position_sizing", 0.8)
                action = self._determine_action(entropy_signal, risk_tolerance, position_sizing)

            else:
                # Neutral mode
                params = routing_config.get("NEUTRAL", {})
                risk_tolerance = params.get("risk_tolerance", 0.5)
                position_sizing = params.get("position_sizing", 1.0)
                action = self._determine_action(entropy_signal, risk_tolerance, position_sizing)

            # Execute the action
            if action:
                self._execute_trade(action)

        except Exception as e:
            logger.error(f"Error making trading decision: {e}")

    def _determine_action(self, entropy_signal, risk_tolerance: float, position_sizing: float) -> Dict[str, Any]:
        """Determine trading action based on entropy signal and parameters."""
        try:
            # Use neural engine for prediction if available
            if self.neural_engine:
                # Create mock market data for neural prediction
                market_data = np.random.randn(10)  # Mock market features
                historical_data = np.random.randn(20, 5)  # Mock historical data

                # Get neural profit optimization
                neural_result = self.neural_engine.neural_profit_optimization(
                    self.btc_price, self.available_balance, market_data, historical_data
                )

                recommended_action = neural_result.get("recommended_action", "Hold")
                confidence = neural_result.get("ensemble_confidence", 0.5)

            else:
                # Fallback decision logic
                if entropy_signal.entropy_value > 0.020 and entropy_signal.confidence > 0.7:
                    recommended_action = "Buy"
                    confidence = entropy_signal.confidence
                elif entropy_signal.entropy_value < 0.010 and entropy_signal.confidence > 0.7:
                    recommended_action = "Sell"
                    confidence = entropy_signal.confidence
                else:
                    recommended_action = "Hold"
                    confidence = 0.5

            # Apply risk tolerance and position sizing
            if confidence < risk_tolerance:
                recommended_action = "Hold"

            # Calculate position size
            if recommended_action != "Hold":
                base_size = self.available_balance * 0.1  # 10% of balance
                position_size = base_size * position_sizing * confidence
            else:
                position_size = 0.0

            return {
                "action": recommended_action,
                "position_size": position_size,
                "confidence": confidence,
                "entropy_value": entropy_signal.entropy_value,
                "routing_state": entropy_signal.routing_state,
                "quantum_state": entropy_signal.quantum_state,
            }

        except Exception as e:
            logger.error(f"Error determining action: {e}")
            return {"action": "Hold", "position_size": 0.0, "confidence": 0.0}

    def _execute_trade(self, action: Dict[str, Any]):
        """Execute a trade based on the action."""
        try:
            action_type = action.get("action", "Hold")
            position_size = action.get("position_size", 0.0)
            confidence = action.get("confidence", 0.0)

            if action_type == "Hold" or position_size <= 0:
                logger.debug("No trade executed - holding position")
                return

            # Calculate BTC amount
            btc_amount = position_size / self.btc_price

            if action_type == "Buy":
                if position_size <= self.available_balance:
                    # Execute buy
                    self.available_balance -= position_size
                    self.current_position += btc_amount
                    self.total_trades += 1

                    logger.info(
                        f"BUY executed: {btc_amount:.6f} BTC at ${self.btc_price:.2f}, "
                        f"Confidence: {confidence:.3f}, Entropy: {action.get('entropy_value', 0):.6f}"
                    )

            elif action_type == "Sell":
                if btc_amount <= self.current_position:
                    # Execute sell
                    self.available_balance += position_size
                    self.current_position -= btc_amount
                    self.total_trades += 1

                    logger.info(
                        f"SELL executed: {btc_amount:.6f} BTC at ${self.btc_price:.2f}, "
                        f"Confidence: {confidence:.3f}, Entropy: {action.get('entropy_value', 0):.6f}"
                    )

            # Update BTC price (simulate market movement)
            self._simulate_price_movement(action)

        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    def _simulate_price_movement(self, action: Dict[str, Any]):
        """Simulate price movement based on the action."""
        try:
            # Simple price movement simulation
            base_movement = 0.001  # 0.1% base movement

            if action.get("action") == "Buy":
                # Price tends to go up after buy
                movement = base_movement * (1 + action.get("confidence", 0.5))
                self.btc_price *= 1 + movement
            elif action.get("action") == "Sell":
                # Price tends to go down after sell
                movement = base_movement * (1 + action.get("confidence", 0.5))
                self.btc_price *= 1 - movement
            else:
                # Random small movement for hold
                movement = base_movement * np.random.uniform(-0.5, 0.5)
                self.btc_price *= 1 + movement

        except Exception as e:
            logger.error(f"Error simulating price movement: {e}")

    def _adjust_trading_parameters(self, performance: Dict[str, Any]):
        """Adjust trading parameters based on performance."""
        try:
            detection_rate = performance.get("average_detection_rate", 0.5)
            latency = performance.get("average_latency_ms", 50)

            # Adjust parameters based on performance
            if detection_rate < 0.8:
                logger.warning(f"Low detection rate ({detection_rate:.3f}) - reducing position sizes")
                # Could implement parameter adjustment here

            if latency > 100:
                logger.warning(f"High latency ({latency:.1f}ms) - optimizing processing")
                # Could implement optimization here

        except Exception as e:
            logger.error(f"Error adjusting trading parameters: {e}")

    def _update_performance_metrics(self):
        """Update trading performance metrics."""
        try:
            # Calculate current portfolio value
            portfolio_value = self.available_balance + (self.current_position * self.btc_price)

            # Calculate PnL
            initial_value = 10000.0  # Initial balance
            current_pnl = portfolio_value - initial_value

            # Update total PnL
            self.total_pnl = current_pnl

            # Log performance periodically
            if self.total_trades % 10 == 0 and self.total_trades > 0:
                win_rate = self.profitable_trades / self.total_trades
                logger.info(
                    f"Performance Update - Trades: {self.total_trades}, "
                    f"Win Rate: {win_rate:.3f}, PnL: ${current_pnl:.2f}, "
                    f"Portfolio: ${portfolio_value:.2f}"
                )

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def run_trading_loop(self, duration_seconds: int = 60):
        """Run the main trading loop for a specified duration."""
        try:
            logger.info(f"Starting entropy trading pipeline for {duration_seconds} seconds")

            start_time = time.time()
            tick_count = 0
            routing_count = 0

            while time.time() - start_time < duration_seconds:
                current_time = time.time()

                # Check if tick cycle should execute
                if should_execute_tick():
                    self.process_tick_cycle()
                    tick_count += 1

                # Check if routing cycle should execute
                if should_execute_routing():
                    self.process_routing_cycle()
                    routing_count += 1

                # Small sleep to prevent excessive CPU usage
                time.sleep(0.001)  # 1ms sleep

            # Final performance report
            self._print_final_report(tick_count, routing_count, start_time)

        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")

    def _print_final_report(self, tick_count: int, routing_count: int, start_time: float):
        """Print final performance report."""
        try:
            duration = time.time() - start_time
            portfolio_value = self.available_balance + (self.current_position * self.btc_price)

            print("\n" + "=" * 60)
            print("ENTROPY TRADING PIPELINE - FINAL REPORT")
            print("=" * 60)
            print(f"Duration: {duration:.1f} seconds")
            print(f"Tick Cycles: {tick_count}")
            print(f"Routing Cycles: {routing_count}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Final Portfolio Value: ${portfolio_value:.2f}")
            print(f"Total PnL: ${self.total_pnl:.2f}")
            print(f"Return: {(self.total_pnl / 10000.0) * 100:.2f}%")

            # Get entropy integrator performance
            entropy_performance = self.entropy_integrator.get_performance_summary()
            print(f"\nEntropy Signal Performance:")
            print(f"Detection Rate: {entropy_performance.get('average_detection_rate', 0):.3f}")
            print(f"Average Latency: {entropy_performance.get('average_latency_ms', 0):.1f}ms")
            print(f"Routing Accuracy: {entropy_performance.get('average_routing_accuracy', 0):.3f}")
            print(f"Quantum Activation Rate: {entropy_performance.get('average_activation_rate', 0):.3f}")

            print("=" * 60)

        except Exception as e:
            logger.error(f"Error printing final report: {e}")


def main():
    """Main function to run the entropy trading pipeline example."""
    try:
        # Create and run the trading pipeline
        pipeline = EntropyTradingPipeline()

        # Run for 60 seconds (adjust as needed)
        pipeline.run_trading_loop(duration_seconds=60)

    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
