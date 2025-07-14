import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from schwabot_unified_math import UnifiedTradingMathematics

from core.brain_trading_engine import BrainTradingEngine
from core.ghost_core import GhostCore, StrategyBranch
from core.matrix_math_utils import analyze_price_matrix
from core.profit_vector_forecast import ProfitVectorForecastEngine
from core.risk_manager import RiskManager
from core.strategy_logic import StrategyLogic

    #!/usr/bin/env python3
    """
🧠 Schwabot Complete Integration Demo
=====================================

Demonstrates the complete integration of all Schwabot components:

1. Ghost Core - Hash-based strategy switching
2. CCXT Integration - Exchange connectivity
3. Matrix Math - Mathematical analysis
4. Brain Trading Engine - Signal processing
5. Risk Management - Position control
6. Profit Vector System - Profit optimization
7. Unified Trading Pipeline - Complete integration

This demo shows how all components work together to create a
comprehensive, internalized trading system with proper mathematical
foundations and profit optimization.
"""

    # Configure logging
    logging.basicConfig()
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger=logging.getLogger(__name__)

    # Import all components
    try:
    UnifiedProfitVectorizationSystem,
)

# Note: CCXT integration requires actual exchange API keys for live trading
# For demo purposes, we'll simulate the CCXT functionality'

ALL_COMPONENTS_AVAILABLE = True
logger.info("✅ All core components imported successfully")

except ImportError as e:
    logger.warning(f"Some components not available: {e}")
    ALL_COMPONENTS_AVAILABLE = False


class SimulatedCCXTIntegration:
    """Simulated CCXT integration for demo purposes."""

    def __init__(self):
        self.order_books = {}
        self.base_price = 50000.0

    def generate_order_book(): -> Dict[str, Any]:
        """Generate simulated order book data."""
        # Simulate bid/ask spread
        spread = price * 0.001  # 0.1% spread
        bid_price = price - spread / 2
        ask_price = price + spread / 2

        # Generate order levels
        bids = []
        asks = []

        for i in range(10):
            bid_level = bid_price - (i * spread / 10)
            ask_level = ask_price + (i * spread / 10)

            bid_volume = np.random.uniform(0.1, 2.0)
            ask_volume = np.random.uniform(0.1, 2.0)

            bids.append([bid_level, bid_volume])
            asks.append([ask_level, ask_volume])

        return {}
            "bids": bids,
            "asks": asks,
            "spread": spread,
            "mid_price": price,
            "timestamp": time.time() * 1000,
        }

    def detect_buy_sell_walls(): -> List[Dict[str, Any]]:
        """Detect buy/sell walls in order book."""
        walls = []

        # Analyze bids for buy walls
        for price, volume in order_book["bids"][:5]:
            if volume > 1.5:  # Large volume threshold
                walls.append()
                    {}
                        "side": "buy",
                        "price": price,
                        "volume": volume,
                        "strength": volume / 2.0,
                    }
                )

        # Analyze asks for sell walls
        for price, volume in order_book["asks"][:5]:
            if volume > 1.5:  # Large volume threshold
                walls.append()
                    {}
                        "side": "sell",
                        "price": price,
                        "volume": volume,
                        "strength": volume / 2.0,
                    }
                )

        return walls

    def calculate_profit_vector(): -> Dict[str, Any]:
        """Calculate profit vector from order book and walls."""
        spread = order_book["spread"]
        mid_price = order_book["mid_price"]

        # Calculate wall pressure
        buy_pressure = sum(w["strength"] for w in walls if w["side"] == "buy")
        sell_pressure = sum(w["strength"] for w in walls if w["side"] == "sell")

        pressure_ratio = buy_pressure / sell_pressure if sell_pressure > 0 else 1.0

        # Calculate profit potential
        base_profit = spread * mid_price
        wall_enhanced_profit = base_profit * pressure_ratio

        return {}
            "base_profit": base_profit,
            "wall_enhanced_profit": wall_enhanced_profit,
            "pressure_ratio": pressure_ratio,
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "spread": spread,
            "mid_price": mid_price,
        }


class CompleteIntegrationDemo:
    """Complete integration demo showing all components working together."""

    def __init__(self, initial_capital: float = 100_000.0):
        """Initialize the complete integration demo."""
        if not ALL_COMPONENTS_AVAILABLE:
            raise ImportError("Not all required components are available")

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trade_history = []
        self.price_history = []

        # Initialize all components
        self._initialize_components()

        logger.info()
            "🚀 Complete Integration Demo initialized with capital: $%.2f", "
            initial_capital,
        )

    def _initialize_components(self):
        """Initialize all trading components."""
        # Ghost Core for strategy switching
        self.ghost_core = GhostCore(memory_depth=100)

        # Brain Trading Engine for signal processing
        self.brain_engine = BrainTradingEngine()
            {}
                "base_profit_rate": 0.02,
                "confidence_threshold": 0.6,
                "enhancement_range": (0.8, 1.5),
            }
        )

        # Risk Manager
        self.risk_manager = RiskManager()
            {}
                "max_position_size": 0.1,
                "stop_loss_threshold": 0.2,
                "max_drawdown_percent": 0.1,
                "max_exposure_per_asset": 0.2,
                "volatility_threshold": 0.3,
            }
        )

        # Profit Vector System
        self.profit_system = UnifiedProfitVectorizationSystem()

        # Strategy Logic
        self.strategy_logic = StrategyLogic()

        # Profit Vector Forecast
        self.profit_forecast = ProfitVectorForecastEngine()

        # Unified Trading Mathematics
        self.unified_math = UnifiedTradingMathematics()

        # Simulated CCXT Integration
        self.ccxt_sim = SimulatedCCXTIntegration()

        logger.info("✅ All components initialized successfully")

    def process_market_tick(): -> Dict[str, Any]:
        """Process a single market tick through the complete pipeline."""
        try:
            # 1. Update price history
            self.price_history.append(price)
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-100:]

            # 2. Calculate mathematical state
            mathematical_state = self._calculate_mathematical_state()

            # 3. Generate Ghost Core hash and switch strategy
            hash_signature = self.ghost_core.generate_strategy_hash()
                price=price,
                volume=volume,
                granularity=2,  # 2 decimal places for BTC
                tick_index=tick_index,
                mathematical_state=mathematical_state,
            )

            # 4. Analyze market conditions
            market_conditions = self._analyze_market_conditions(price, volume)

            # 5. Switch strategy
            ghost_state = self.ghost_core.switch_strategy()
                hash_signature=hash_signature,
                market_conditions=market_conditions,
                mathematical_state=mathematical_state,
            )

            # 6. Generate order book data
            order_book = self.ccxt_sim.generate_order_book(symbol, price)

            # 7. Detect buy/sell walls
            walls = self.ccxt_sim.detect_buy_sell_walls(order_book)

            # 8. Calculate profit vector
            profit_vector = self.ccxt_sim.calculate_profit_vector(order_book, walls)

            # 9. Process brain signal
            brain_signal = self.brain_engine.process_brain_signal()
                price=price, volume=volume, symbol=symbol
            )

            # 10. Get brain trading decision
            brain_decision = self.brain_engine.get_trading_decision(brain_signal)

            # 11. Calculate risk metrics
            position_size = self.risk_manager.calculate_position_size()
                entry_price=price,
                stop_loss_price=price * 0.98,  # 2% stop loss
                portfolio_value=self.current_capital,
                volatility=market_conditions.get("volatility", 0.2),
            )

            risk_metrics = {}
                "position_size": position_size,
                "entry_price": price,
                "stop_loss_price": price * 0.98,
                "portfolio_value": self.current_capital,
                "volatility": market_conditions.get("volatility", 0.2),
            }

            # 12. Generate final trading decision
            trading_decision = self._generate_trading_decision()
                symbol=symbol,
                price=price,
                volume=volume,
                ghost_state=ghost_state,
                brain_decision=brain_decision,
                profit_vector=profit_vector,
                risk_metrics=risk_metrics,
                market_conditions=market_conditions,
                mathematical_state=mathematical_state,
            )

            # 13. Execute trade if decision made
            trade_result = None
            if trading_decision and trading_decision.get("action") != "HOLD":
                trade_result = self._execute_trade(trading_decision)

            # 14. Update Ghost Core performance
            if trade_result:
                self.ghost_core.update_strategy_performance()
                    ghost_state.current_branch, trade_result
                )

            # Return comprehensive results
            return {}
                "timestamp": time.time(),
                "tick_index": tick_index,
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "ghost_state": {}
                    "branch": ghost_state.current_branch.value,
                    "confidence": ghost_state.confidence,
                    "profit_potential": ghost_state.profit_potential,
                    "hash_signature": hash_signature[:8],
                },
                "brain_signal": {}
                    "signal_strength": brain_signal.signal_strength,
                    "enhancement_factor": brain_signal.enhancement_factor,
                    "profit_score": brain_signal.profit_score,
                    "confidence": brain_signal.confidence,
                },
                "brain_decision": brain_decision,
                "profit_vector": profit_vector,
                "risk_metrics": risk_metrics,
                "market_conditions": market_conditions,
                "mathematical_state": mathematical_state,
                "trading_decision": trading_decision,
                "trade_result": trade_result,
                "walls_detected": len(walls),
                "order_book_summary": {}
                    "spread": order_book["spread"],
                    "mid_price": order_book["mid_price"],
                    "bid_levels": len(order_book["bids"]),
                    "ask_levels": len(order_book["asks"]),
                },
            }

        except Exception as e:
            logger.error("Error processing market tick: %s", e)
            return {"error": str(e)}

    def _calculate_mathematical_state(): -> Dict[str, Any]:
        """Calculate mathematical state from price history."""
        try:
            # Calculate matrix analysis only if enough data
            if len(self.price_history) >= 2:
                price_matrix = np.array(self.price_history[-20:]).reshape(-1, 1)
                matrix_analysis = analyze_price_matrix(price_matrix)
            else:
                matrix_analysis = {}
                    "stability_score": 0.5,
                    "condition_number": 1.0,
                    "eigenvalues": np.array([1.0]),
                    "volatility": 0.2,
                }

            # Calculate volatility
            if len(self.price_history) >= 2:
                returns = np.diff(np.log(self.price_history[-20:]))
                volatility = ()
                    np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.2
                )
            else:
                volatility = 0.2  # Default volatility

            return {}
                "complexity": 1.0
                - matrix_analysis.get("stability_score", 0.5),  # Inverse of stability
                "stability": matrix_analysis.get("stability_score", 0.5),
                "volatility": matrix_analysis.get("volatility", volatility),
                "condition_number": matrix_analysis.get("condition_number", 1.0),
                "eigenvalues": matrix_analysis.get()
                    "eigenvalues", np.array([1.0])
                ).tolist(),
            }

        except Exception as e:
            logger.error("Error calculating mathematical state: %s", e)
            return {"complexity": 0.5, "stability": 0.5, "volatility": 0.2}

    def _analyze_market_conditions(): -> Dict[str, Any]:
        """Analyze current market conditions."""
        try:
            if len(self.price_history) < 5:
                return {"volatility": 0.2, "momentum": 0.0, "volume_profile": 1.0}

            # Calculate volatility
            returns = np.diff(np.log(self.price_history[-10:]))
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.2

            # Calculate momentum
            momentum = ()
                (price - self.price_history[0]) / self.price_history[0]
                if self.price_history[0] > 0
                else 0.0
            )

            # Calculate volume profile
            avg_volume = 1000.0  # Simulated average volume
            volume_profile = volume / avg_volume if avg_volume > 0 else 1.0

            return {}
                "volatility": volatility,
                "momentum": momentum,
                "volume_profile": volume_profile,
                "price_range": ()
                    min(self.price_history[-10:]),
                    max(self.price_history[-10:]),
                ),
            }

        except Exception as e:
            logger.error("Error analyzing market conditions: %s", e)
            return {"volatility": 0.2, "momentum": 0.0, "volume_profile": 1.0}

    def _generate_trading_decision(): -> Optional[Dict[str, Any]]:
        """Generate final trading decision."""
        try:
            brain_action = brain_decision.get("action", "HOLD")
            brain_confidence = brain_decision.get("confidence", 0.0)

            # Combined decision logic
            if brain_action == "HOLD" or brain_confidence < 0.3:
                return {"action": "HOLD", "reason": "Low confidence"}

            # High confidence scenarios
            if brain_confidence > 0.7 and ghost_state.confidence > 0.6:
                profit_potential = profit_vector.get("wall_enhanced_profit", 0.0)
                if profit_potential > 0.01:  # 0.1% profit potential
                    return {}
                        "action": brain_action,
                        "confidence": brain_confidence,
                        "quantity": risk_metrics.get("position_size", 0.0),
                        "price": price,
                        "reason": "High confidence with profit potential",
                    }

            # Moderate confidence with strong profit potential
            if brain_confidence > 0.5:
                profit_potential = profit_vector.get("wall_enhanced_profit", 0.0)
                if profit_potential > 0.02:  # 0.2% profit potential
                    return {}
                        "action": brain_action,
                        "confidence": brain_confidence,
                        "quantity": risk_metrics.get("position_size", 0.0)
                        * 0.5,  # Reduced size
                        "price": price,
                        "reason": "Moderate confidence with strong profit potential",
                    }

            return {"action": "HOLD", "reason": "Insufficient conditions"}

        except Exception as e:
            logger.error("Error generating trading decision: %s", e)
            return {"action": "HOLD", "reason": "Error in decision generation"}

    def _execute_trade(): -> Dict[str, Any]:
        """Execute a trading decision."""
        try:
            action = decision.get("action")
            quantity = decision.get("quantity", 0.0)
            price = decision.get("price", 0.0)

            if action == "HOLD" or quantity <= 0:
                return {"executed": False, "reason": "No trade to execute"}

            # Calculate trade value
            trade_value = quantity * price
            commission = trade_value * 0.01  # 0.1% commission

            # Update capital
            if action == "BUY":
                self.current_capital -= trade_value + commission
            else:  # SELL
                self.current_capital += trade_value - commission

            # Calculate profit (simplified)
            profit = 0.0
            if action == "SELL":
                profit = trade_value * 0.1  # 1% profit assumption

            # Store trade result
            trade_result = {}
                "timestamp": time.time(),
                "action": action,
                "quantity": quantity,
                "price": price,
                "trade_value": trade_value,
                "commission": commission,
                "profit": profit,
                "new_capital": self.current_capital,
            }

            self.trade_history.append(trade_result)

            logger.info()
                "✅ Trade executed: %s %.4f BTC @ $%.2f (profit: $%.2f, capital: $%.2f)",
                action,
                quantity,
                price,
                profit,
                self.current_capital,
            )

            return trade_result

        except Exception as e:
            logger.error("Error executing trade: %s", e)
            return {"executed": False, "error": str(e)}

    def run_demo(): -> Dict[str, Any]:
        """Run the complete integration demo."""
        logger.info("Starting complete integration demo with %d ticks", num_ticks)

        results = []

        # Initial price within the desired range
        base_price = np.random.uniform(25000.0, 45000.0)
        current_time = time.time()

        for i in range(num_ticks):
            # Simulate price movement with a tendency to revert but also trend
            # Introduce some random walk, with a slight drift towards the mean of the range (35000)
            drift = (35000.0 - base_price) * 0.0005  # Gentle drift back to middle
            volatility_factor = np.random.normal(0, 0.01)  # Daily volatility 0.1%

            base_price *= ()
                1 + drift + volatility_factor + np.random.uniform(-0.005, 0.005)
            )

            # Clamp price within the 20,00 - 50,00 range
            base_price = max(20000.0, min(50000.0, base_price))

            # Generate volume (more, dynamic)
            volume = max()
                100, 500 + np.random.uniform(-400, 1000) * (base_price / 35000.0)
            )

            current_time += 60 * 5  # Simulate ticks every 5 minutes

            # Process market tick
            tick_result = self.process_market_tick()
                symbol="BTC/USDT", price=base_price, volume=volume, tick_index=i
            )

            results.append(tick_result)

            # Log progress
            if i % 10 == 0:
                logger.info()
                    "Demo progress: %d/%d (%.1f%%)", i, num_ticks, i / num_ticks * 100
                )

        # Calculate summary statistics
        total_trades = len()
            [r for r in results if r.get("trade_result", {}).get("executed", False)]
        )
        winning_trades = len()
            [r for r in results if r.get("trade_result", {}).get("profit", 0) > 0]
        )
        total_profit = sum(r.get("trade_result", {}).get("profit", 0) for r in results)
        total_return = ()
            self.current_capital - self.initial_capital
        ) / self.initial_capital

        # Get component statuses
        ghost_status = self.ghost_core.get_system_status()
        brain_metrics = self.brain_engine.get_metrics_summary()

        summary = {}
            "initial_capital": self.initial_capital,
            "final_capital": self.current_capital,
            "total_return": total_return,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / max(total_trades, 1),
            "total_profit": total_profit,
            "ghost_core_status": ghost_status,
            "brain_engine_metrics": brain_metrics,
            "tick_results": results,
        }

        logger.info()
            "Demo completed: %.2f%% return, %d trades, %.1f%% win rate",
            total_return * 100,
            total_trades,
            summary["win_rate"] * 100,
        )

        return summary

    def print_detailed_results(self, results: Dict[str, Any]):
        """Print detailed demo results."""
        print("\n" + "=" * 60)
        print("🧠 SCHWABOT COMPLETE INTEGRATION DEMO RESULTS")
        print("=" * 60)

        # Financial Summary
        print("\n💰 FINANCIAL SUMMARY:")
        print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"  Final Capital:   ${results['final_capital']:,.2f}")
        print(f"  Total Return:    {results['total_return']:.2%}")
        print(f"  Total Profit:    ${results['total_profit']:,.2f}")

        # Trading Performance
        print("\n📊 TRADING PERFORMANCE:")
        print(f"  Total Trades:    {results['total_trades']}")
        print(f"  Winning Trades:  {results['winning_trades']}")
        print(f"  Win Rate:        {results['win_rate']:.1%}")

        # Ghost Core Status
        ghost_status = results["ghost_core_status"]
        print("\n👻 GHOST CORE STATUS:")
        print(f"  Current Branch:  {ghost_status.get('current_branch', 'N/A')}")
        print(f"  Memory Depth:    {ghost_status.get('memory_depth', 0)}")
        print(f"  Hash History:    {ghost_status.get('hash_history_size', 0)}")

        # Strategy Performance
        strategy_perf = ghost_status.get("strategy_performance", {})
        if strategy_perf:
            print("  Strategy Performance:")
            for strategy, perf in strategy_perf.items():
                if perf["total_trades"] > 0:
                    print()
                        f"    {strategy}: {"}
                            perf['success_rate']:.1%} win rate, {
                            perf['total_trades']} trades")"

        # Brain Engine Metrics
        brain_metrics = results["brain_engine_metrics"]
        print("\n🧠 BRAIN ENGINE METRICS:")
        print(f"  Total Signals:   {brain_metrics.get('total_signals', 0)}")
        print(f"  Win Rate:        {brain_metrics.get('win_rate', 0):.1%}")
        print()
            f"  Avg Profit:      ${brain_metrics.get('avg_profit_per_signal', 0):.4f}"
        )

        # Sample Tick Analysis
        print("\n📈 SAMPLE TICK ANALYSIS:")
        sample_ticks = results["tick_results"][:5]  # First 5 ticks
        for i, tick in enumerate(sample_ticks):
            if "ghost_state" in tick:
                print()
                    f"  Tick {i}: {tick['ghost_state']['branch']} "
                    f"(conf: {tick['ghost_state']['confidence']:.2f}, ")
                    f"hash: {tick['ghost_state']['hash_signature']})"
                )

        print("\n✅ Demo completed successfully!")


def main():
    """Main demo function."""
    print("🧠 Schwabot Complete Integration Demo")
    print("=" * 50)

    try:
        # Initialize demo
        demo = CompleteIntegrationDemo(initial_capital=100_000.0)

        # Run demo
        results = demo.run_demo(num_ticks=200)

        # Print results
        demo.print_detailed_results(results)

    except Exception as e:
        logger.error("Demo failed: %s", e)
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
