from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from core.brain_trading_engine import register_risk_manager, update_risk_threshold
from core.ccxt_integration import CCXTIntegration
from core.ghost_core import GhostCore
from core.matrix_math_utils import analyze_price_matrix
from demo_integrated_trading_system import IntegratedTradingSystem

"""Internal Backtester for Schwabot
===================================

Runs the `IntegratedTradingSystem` on a supplied price-time series and
periodically applies self-corrective adjustments based on matrix
analysis (see `core.matrix_math_utils`).

The goal is *continuous calibration* – the risk manager thresholds and
strategy aggressiveness are adapted in-flight according to market
stability metrics.
"""





class InternalBacktester:
    """Simple sliding-window back-tester with self-correction."""

    def __init__():base_price: float = 50_000.0,
        window: int = 100,
    ) -> None:
        self.price_series = price_series
        self.window = window
        self.system = IntegratedTradingSystem(initial_capital=100_000.0)
        register_risk_manager(self.system.risk_manager)

        # Initialize Ghost Core and CCXT Integration for micro-trend analysis
        self.ghost_core = GhostCore(memory_depth=50)  # Smaller memory for faster demo
        self.ccxt_integration = CCXTIntegration(
            config={"exchanges": ["binance"]}
        )  # Simulate one exchange

        # Buffer for matrix analysis
        self.price_buffer: List[float] = []

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _create_market_data():-> Dict[str, Any]:
        """Construct a market-data dict compatible with system API."""
        if len(self.price_buffer) >= 20:
            returns = np.diff(np.array(self.price_buffer[-20:])) / np.array(
                self.price_buffer[-20:-1]
            )
            volatility = float(np.std(returns) * np.sqrt(252))
        else:
            volatility = 0.25

        volume = random.uniform(800, 1_200)
        return {
            "asset": "BTC/USD",
            "price": price,
            "volume": volume,
            "timestamp": time.time(),
            "price_history": self.price_buffer.copy(),
            "market_trend": "neutral",
            "volatility": volatility,
        }

    # ------------------------------------------------------------------
    def run():-> None:
        """Execute back-test over the full price series."""
        for idx, (_ts, price) in enumerate(self.price_series, start=1):
            self.price_buffer.append(price)
            market_data = self._create_market_data(price)
            self.system.run_trading_cycle(market_data)

            # Every *window* steps – perform matrix analysis & adjust
            if idx >= self.window and idx % self.window == 0:
                self._self_correct(idx)

            # Call micro-trend analysis at every step, allowing it to handle its own data requirements
            self.analyze_micro_trends_and_apply_strategy_hooks(self.price_buffer, idx)

        # Final summary
        self.system.print_performance_summary()

    # ------------------------------------------------------------------
    def _self_correct():-> None:
        """Analyse price matrix and tune risk parameters."""
        # Ensure enough data for matrix analysis
        if len(self.price_buffer) < self.window:
            print(
                f"⚠️ Not enough data for matrix analysis at step {idx}. Need at least {self.window} samples, have {len(self.price_buffer)}."
            )
            return

        matrix_window = np.array(self.price_buffer[-self.window :]).reshape(-1, 1)
        analysis = analyze_price_matrix(matrix_window)

        if analysis is None or analysis.get("stability_score") is None:
            print(
                f"❌ Matrix analysis failed or returned incomplete data at step {idx}."
            )
            return

        stability = analysis["stability_score"]
        eigen_max = float(np.max(analysis["eigenvalues"]))

        # Adapt volatility threshold inversely to stability (bounded).
        new_vol_thresh = float(max(0.01, min(0.05, 0.03 * (1.0 / stability))))
        update_risk_threshold(new_vol_thresh)

        # Optionally log adjustment
        print(
            f"⚙️  Self-correct @ step {idx}: stability={stability:.3f} | "
            f"e_max={eigen_max:.4f} | new_vol_thresh={new_vol_thresh:.3f}"
        )

    def analyze_micro_trends_and_apply_strategy_hooks(
        self, all_prices: List[float], tick_index: int
    ):
        """
        Hook for future CCXT, Ghost Core, and Profit Vector logic integration over micro-decimal feeds.
        - Targets high-res feed strategy (e.g. 8-decimal BTC precision)
        - Implements hash-switch logic from Ghost Core
        - Ensures swap triggers are confluence-driven and not arbitrary
        """
        decimal_granularities = [8, 6, 2]

        # Use a more recent window for micro-trend analysis, but ensure it's not empty
        micro_trend_window_size = 20  # Can be adjusted
        if len(all_prices) < micro_trend_window_size:
            return  # Not enough data for micro-trend analysis

        window_prices = all_prices[-micro_trend_window_size:]

        for gran in decimal_granularities:
            rounded_prices = np.round(window_prices, decimals=gran)

            if (
                len(rounded_prices) < 2
            ):  # Need at least 2 points for std_dev and momentum
                continue

            # Ensure the input to analyze_price_matrix is 2D
            prices_for_math = rounded_prices.reshape(-1, 1)
            mathematical_state_analysis = analyze_price_matrix(prices_for_math)

            if (
                mathematical_state_analysis is None
                or mathematical_state_analysis.get("stability_score") is None
            ):
                # print(f"❌ Micro-trend matrix analysis failed for granularity {gran}.")
                continue

            std_dev = mathematical_state_analysis.get(
                "volatility", np.std(rounded_prices)
            )  # Use volatility if present, else calculate

            # Simulate when we would consider switching strategies
            # Using a dynamic threshold based on granularity and price range
            price_range_factor = (max(rounded_prices) - min(rounded_prices)) / max(
                rounded_prices, 1e-9
            )
            dynamic_threshold = (
                0.0001 * (10 ** (8 - gran)) * (1 + price_range_factor * 0.5)
            )

            if std_dev < dynamic_threshold:
                # Simulate current market conditions for Ghost Core
                market_conditions = {
                    "volatility": std_dev,
                    "momentum": (rounded_prices[-1] - rounded_prices[0])
                    / rounded_prices[0]
                    if rounded_prices[0] != 0
                    else 0.0,
                    "volume_profile": 1.0,  # Placeholder
                }

                # Directly pass the results from analyze_price_matrix
                mathematical_state = mathematical_state_analysis

                # Generate ghost hash and trigger strategy switch
                # Use a dummy price/volume for hash generation as real-time data is not available here
                dummy_price = window_prices[-1] if window_prices else 50000.0
                dummy_volume = 1000.0

                ghost_hash = self.ghost_core.generate_strategy_hash(
                    price=dummy_price,
                    volume=dummy_volume,
                    granularity=gran,
                    tick_index=tick_index,
                    mathematical_state=mathematical_state,
                )

                # Use Ghost Core to switch strategy
                current_ghost_state = self.ghost_core.switch_strategy(
                    hash_signature=ghost_hash,
                    market_conditions=market_conditions,
                    mathematical_state=mathematical_state,
                )

                # Simulate CCXT integration for order book analysis and profit vector
                # This part would typically involve async calls, but for backtesting demo, we simulate
                # order_book_snapshot = await self.ccxt_integration.fetch_order_book(symbol, limit)
                # walls = self.ccxt_integration.detect_buy_sell_walls(order_book_snapshot)
                # profit_vector = self.ccxt_integration.calculate_profit_vector(order_book_snapshot, walls)

                print(
                    f"?? Triggering strategy switch via Ghost Core at gran={gran} | "
                    f"hash={ghost_hash[:8]}... | New Strategy: {current_ghost_state.current_branch.value}"
                )

        # In future: pull from live CCXT order book depth, calculate spread walls, etc.
        pass
