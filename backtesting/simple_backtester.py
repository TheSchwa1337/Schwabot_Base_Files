#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Backtester for Schwabot Trading System.

This module provides a basic framework for backtesting trading strategies
using historical price data and the CCXTTradingExecutor.
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .historical_data_manager import HistoricalDataManager
from core.ccxt_trading_executor import CCXTTradingExecutor, ExecutionResult, IntegratedTradingSignal, TradingPair

logger = logging.getLogger(__name__)


class SimpleBacktester:
    """A simple backtesting engine to simulate trading strategies."""

    def __init__(
        self,
        initial_capital: Decimal = Decimal("10000"),
        start_date: datetime = datetime(2023, 1, 1),
        end_date: datetime = datetime(2023, 1, 31),
        trading_pair: TradingPair = TradingPair.BTC_USDC,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.trading_pair = trading_pair
        self.config = config or {}

        # Initialize CCXTTradingExecutor with a mock config for backtesting
        # In a real scenario, this config would be more elaborate, including API keys if needed for live testing.
        executor_config = {
            "unified_api_config": {
                "ccxt_config": {
                    "timeout": 30000,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "spot",
                    },
                }
            },
            "COINMARKETCAP_API_KEY": "mock_cmc_key",  # Mock key
            # No actual API keys needed for backtesting if using simulated data
        }
        self.ccxt_executor = CCXTTradingExecutor(executor_config)
        self.ccxt_executor.portfolio_balance["USDC"] = initial_capital

        # Initialize HistoricalDataManager
        self.historical_data_manager = HistoricalDataManager(
            start_date=self.start_date,
            end_date=self.end_date,
            interval_minutes=1440,  # Daily data for backtesting, can be configured
            initial_price=Decimal("40000.0"),  # Initial price for mock data generation
        )

        self.current_date = start_date
        self.trade_results: List[ExecutionResult] = []
        self.portfolio_value_history: List[Decimal] = [self.initial_capital]

        logger.info(
            f"SimpleBacktester initialized for {trading_pair.value} from {start_date.date()} to {end_date.date()}"
        )

    async def run_backtest(
        self, data_source: str = "mock", data_file_path: Optional[str] = None
    ):
        """Runs the backtest simulation."""
        logger.info("Starting backtest...")

        # Start price monitoring in the executor (will use prices from backtest data)
        self.ccxt_executor.start_price_monitoring()

        try:
            # Use the HistoricalDataManager to get data
            async for data_point in self.historical_data_manager.get_historical_data(
                source=data_source, file_path=data_file_path
            ):
                self.current_date = datetime.fromisoformat(data_point["datetime"])
                current_price = Decimal(
                    str(data_point["close"])
                )  # Use close price for current price
                open_price = Decimal(str(data_point["open"]))
                Decimal(str(data_point["high"]))
                Decimal(str(data_point["low"]))
                volume = Decimal(str(data_point["volume"]))

                logger.info(
                    f"Processing data for {self.current_date.date()}: Close Price={current_price:.2f}, Volume={volume:.2f}"
                )

                # Simulate updating the executor's price data for the trading pair
                # This mimics the price monitoring loop feeding updated prices
                self.ccxt_executor.price_data[self.trading_pair] = current_price
                # Also update for stablecoin if relevant
                if self.trading_pair in [
                    TradingPair.BTC_USDC,
                    TradingPair.ETH_USDC,
                    TradingPair.XRP_USDC,
                ]:
                    self.ccxt_executor.price_data[TradingPair.USDC_USD] = Decimal("1.0")
                if self.trading_pair in [TradingPair.BTC_USDT, TradingPair.ETH_USDT]:
                    self.ccxt_executor.price_data[TradingPair.USDT_USD] = Decimal("1.0")

                # Update portfolio history for drawdown calculation
                # Assuming all non-USDC assets are converted to USDC value for total portfolio calculation
                current_portfolio_value = self.ccxt_executor.portfolio_balance["USDC"]
                if self.trading_pair == TradingPair.BTC_USDC:
                    current_portfolio_value += (
                        self.ccxt_executor.portfolio_balance["BTC"] * current_price
                    )
                elif self.trading_pair == TradingPair.ETH_USDC:
                    current_portfolio_value += (
                        self.ccxt_executor.portfolio_balance["ETH"] * current_price
                    )
                elif self.trading_pair == TradingPair.XRP_USDC:
                    current_portfolio_value += (
                        self.ccxt_executor.portfolio_balance["XRP"] * current_price
                    )
                # Add logic for other pairs if they hold non-USDC base assets

                self.portfolio_value_history.append(current_portfolio_value)

                # Simple trading strategy: Buy if price dropped significantly, Sell if gained significantly
                # This is a placeholder and would be replaced by a real trading strategy.
                # For demonstration, let's use a simple moving average crossover or similar logic here
                # For now, keep the random signal generation, but enhance its data usage

                # Simple price action strategy: if current price is significantly above/below open
                signal_action = None
                confidence = 0.0
                profit_potential = 0.0
                risk = 0.0

                price_change_pct = (current_price - open_price) / open_price

                if price_change_pct < Decimal(
                    "-0.0075"
                ):  # Price dropped by more than 0.75%
                    signal_action = "buy"
                    confidence = Decimal(str(random.uniform(0.7, 0.95)))
                    profit_potential = Decimal(str(random.uniform(0.02, 0.08)))
                    risk = Decimal(str(random.uniform(0.05, 0.2)))
                elif price_change_pct > Decimal(
                    "0.0075"
                ):  # Price increased by more than 0.75%
                    signal_action = "sell"
                    confidence = Decimal(str(random.uniform(0.6, 0.9)))
                    profit_potential = Decimal(str(random.uniform(0.01, 0.05)))
                    risk = Decimal(str(random.uniform(0.02, 0.1)))

                if signal_action:
                    signal = IntegratedTradingSignal(
                        signal_id=f"mock_signal_{int(time.time() * 1000)}",
                        recommended_action=signal_action,
                        target_pair=self.trading_pair,
                        confidence_score=confidence,
                        profit_potential=profit_potential,
                        risk_assessment={"overall_risk": risk},
                        ghost_route="ghost_trade",
                    )
                    logger.debug(
                        f"Generated mock signal: {signal.recommended_action} {signal.target_pair.value}"
                    )
                    execution_result = await self.ccxt_executor.execute_signal(signal)
                    self.trade_results.append(execution_result)
                    if execution_result.executed:
                        logger.info(
                            f"Trade executed: {execution_result.strategy.value} {execution_result.fill_amount:.6f} {execution_result.pair.value} @ {execution_result.fill_price:.2f}"
                        )
                    else:
                        logger.warning(
                            f"Trade not executed: {execution_result.error_message}"
                        )

                # Simulate a delay for realism in backtesting
                await asyncio.sleep(0.001)  # Very small sleep to allow async operations

        finally:
            self.ccxt_executor.stop_price_monitoring()
            logger.info("Backtest finished.")
            self._report_results()

    def _report_results(self):
        """Reports the results of the backtest."""
        logger.info("\n--- Backtest Results ---")

        # Calculate final portfolio value more accurately using actual holdings and current prices
        final_portfolio_value = self.ccxt_executor.portfolio_balance["USDC"]
        if TradingPair.BTC_USDC in self.ccxt_executor.price_data:
            final_portfolio_value += (
                self.ccxt_executor.portfolio_balance["BTC"]
                * self.ccxt_executor.price_data[TradingPair.BTC_USDC]
            )
        if TradingPair.ETH_USDC in self.ccxt_executor.price_data:
            final_portfolio_value += (
                self.ccxt_executor.portfolio_balance["ETH"]
                * self.ccxt_executor.price_data[TradingPair.ETH_USDC]
            )
        if TradingPair.XRP_USDC in self.ccxt_executor.price_data:
            final_portfolio_value += (
                self.ccxt_executor.portfolio_balance["XRP"]
                * self.ccxt_executor.price_data[TradingPair.XRP_USDC]
            )
        # Add other assets if necessary, using their current prices

        net_profit_loss = final_portfolio_value - self.initial_capital

        total_realized_profit = Decimal("0")
        for result in self.trade_results:
            if result.executed and result.profit_realized is not None:
                total_realized_profit += result.profit_realized

        total_return_pct = (
            (net_profit_loss / self.initial_capital * Decimal("100"))
            if self.initial_capital
            else Decimal("0")
        )

        # Calculate Maximum Drawdown
        peak = self.initial_capital
        max_drawdown = Decimal("0")
        for value in self.portfolio_value_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak else Decimal("0")
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        max_drawdown_pct = max_drawdown * Decimal("100")

        # Calculate winning and losing trades
        winning_trades = [
            r.profit_realized
            for r in self.trade_results
            if r.executed and r.profit_realized is not None and r.profit_realized > 0
        ]
        losing_trades = [
            r.profit_realized
            for r in self.trade_results
            if r.executed and r.profit_realized is not None and r.profit_realized < 0
        ]

        num_winning_trades = len(winning_trades)
        num_losing_trades = len(losing_trades)

        avg_win = (
            sum(winning_trades) / num_winning_trades
            if num_winning_trades > 0
            else Decimal("0")
        )
        avg_loss = (
            sum(losing_trades) / num_losing_trades
            if num_losing_trades > 0
            else Decimal("0")
        )

        total_wins = sum(winning_trades) if winning_trades else Decimal("0")
        total_losses = sum(losing_trades) if losing_trades else Decimal("0")

        profit_factor = (
            total_wins / abs(total_losses)
            if total_losses != Decimal("0")
            else Decimal("Inf")
        )

        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        logger.info(f"Net Profit/Loss (including unrealized): ${net_profit_loss:.2f}")
        logger.info(f"Total Return: {total_return_pct:.2f}%")
        logger.info(f"Maximum Drawdown: {max_drawdown_pct:.2f}%")
        logger.info(f"Total Trades: {self.ccxt_executor.total_trades}")
        logger.info(f"Successful Trades: {self.ccxt_executor.successful_trades}")
        logger.info(f"Win Rate: {self.ccxt_executor.win_rate:.2f}%")
        logger.info(f"Total Realized Profit/Loss: ${total_realized_profit:.2f}")
        logger.info(f"Number of Winning Trades: {num_winning_trades}")
        logger.info(f"Number of Losing Trades: {num_losing_trades}")
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info("------------------------")


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    backtester = SimpleBacktester(
        initial_capital=Decimal("10000"),
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        trading_pair=TradingPair.BTC_USDC,
    )
    # Run backtest using mock data from HistoricalDataManager
    await backtester.run_backtest(data_source="mock")


if __name__ == "__main__":
    asyncio.run(main())
