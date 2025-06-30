# -*- coding: utf-8 -*-
"""Portfolio Tracker for Schwabot Trading System.

Monitors and tracks the state of the trading portfolio, including current holdings,
realized and unrealized PnL, and various portfolio-level metrics.

Integrates with: [Other modules that execute trades or manage risk]
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from decimal import Decimal, getcontext

# Set high precision for financial calculations
getcontext().prec = 18

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single asset position in the portfolio."""

    asset: str
    quantity: Decimal
    avg_price: Decimal
    current_price: Decimal
    last_update: float = field(default_factory=time.time)

    @property
    def value(self) -> Decimal:
        """Current market value of the position."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized Profit and Loss."""
        return (self.current_price - self.avg_price) * self.quantity


class PortfolioTracker:
    """Tracks and manages the trading portfolio."""

    def __init__(self, initial_cash: Union[float, str, Decimal] = "100000.0"):
        """Initialize the portfolio tracker.

        Args:
            initial_cash: Starting cash balance.
        """
        self.cash = Decimal(str(initial_cash))
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = Decimal("0.0")
        self.transaction_history: List[Dict[str, Any]] = []

        # Performance metrics
        self.portfolio_stats = {
            "total_deposits": Decimal("0.0"),
            "total_withdrawals": Decimal("0.0"),
            "trade_count": 0,
            "total_fees": Decimal("0.0"),
            "last_update_time": time.time(),
        }
        self.portfolio_stats["total_deposits"] += self.cash

        logger.info(f"PortfolioTracker initialized with cash: {self.cash:.2f}")

    def deposit(self, amount: Union[float, str, Decimal]) -> None:
        """Deposit cash into the portfolio."""
        amount_dec = Decimal(str(amount))
        if amount_dec <= 0:
            logger.warning(
                f"Attempted to deposit non-positive amount: {amount_dec}")
            return
        self.cash += amount_dec
        self.portfolio_stats["total_deposits"] += amount_dec
        self.portfolio_stats["last_update_time"] = time.time()
        logger.info(f"Deposited {amount_dec:.2f}. New cash: {self.cash:.2f}")

    def withdraw(self, amount: Union[float, str, Decimal]) -> None:
        """Withdraw cash from the portfolio."""
        amount_dec = Decimal(str(amount))
        if amount_dec <= 0 or amount_dec > self.cash:
            logger.warning(
                f"Attempted to withdraw invalid amount {
                    amount_dec:.2f} or insufficient cash.")
            return
        self.cash -= amount_dec
        self.portfolio_stats["total_withdrawals"] += amount_dec
        self.portfolio_stats["last_update_time"] = time.time()
        logger.info(f"Withdrew {amount_dec:.2f}. New cash: {self.cash:.2f}")

    def update_position(
        self,
        asset: str,
        direction: str,
        quantity: float,
        price: float,
        fees: float = 0.0,
    ) -> None:
        """Update an asset position based on a trade execution.

        Args:
            asset: The asset traded (e.g., "BTC/USD").
            direction: "buy" or "sell".
            quantity: The quantity of the asset traded.
            price: The price at which the trade occurred.
            fees: Any fees incurred in the trade.
        """
        qty_dec = Decimal(str(quantity))
        price_dec = Decimal(str(price))
        fees_dec = Decimal(str(fees))
        trade_value = qty_dec * price_dec

        self.portfolio_stats["trade_count"] += 1
        self.portfolio_stats["total_fees"] += fees_dec
        self.portfolio_stats["last_update_time"] = time.time()

        if direction == "buy":
            if self.cash < (trade_value + fees_dec):
                logger.error(
                    f"Insufficient cash to buy {qty_dec} of {asset}. Required: {
                        trade_value +
                        fees_dec:.2f}, Available: {
                        self.cash:.2f}")
                return
            self.cash -= (trade_value + fees_dec)

            if asset in self.positions:
                pos = self.positions[asset]
                new_total_qty = pos.quantity + qty_dec
                new_avg_price = (
                    (pos.quantity * pos.avg_price) + trade_value) / new_total_qty
                pos.quantity = new_total_qty
                pos.avg_price = new_avg_price
                pos.current_price = price_dec
                pos.last_update = time.time()
            else:
                self.positions[asset] = Position(
                    asset, qty_dec, price_dec, price_dec)
            logger.info(
                f"Bought {qty_dec:.4f} {asset} @ {price_dec:.2f}. New cash: {self.cash:.2f}")

        elif direction == "sell":
            if asset not in self.positions or self.positions[asset].quantity < qty_dec:
                logger.error(
                    f"Insufficient {asset} to sell {qty_dec}. Available: {
                        self.positions.get(
                            asset,
                            Position(
                                asset,
                                Decimal("0"),
                                Decimal("0"),
                                Decimal("0"))).quantity}")
                return
            pos = self.positions[asset]
            pnl = (price_dec - pos.avg_price) * qty_dec
            self.realized_pnl += pnl
            self.cash += (trade_value - fees_dec)
            pos.quantity -= qty_dec
            pos.current_price = price_dec
            pos.last_update = time.time()

            if pos.quantity == 0:
                del self.positions[asset]
            logger.info(
                f"Sold {
                    qty_dec:.4f} {asset} @ {
                    price_dec:.2f}. Realized PnL: {
                    pnl:.2f}. New cash: {
                    self.cash:.2f}")

        self.transaction_history.append({
            "timestamp": time.time(),
            "asset": asset,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "fees": fees,
            "trade_value": float(trade_value),
            "realized_pnl": float(self.realized_pnl),
            "cash_after_trade": float(self.cash),
        })

    def update_market_prices(self, current_prices: Dict[str, float]) -> None:
        """Update current market prices for all held positions."""
        for asset, price in current_prices.items():
            if asset in self.positions:
                self.positions[asset].current_price = Decimal(str(price))
                self.positions[asset].last_update = time.time()
        self.portfolio_stats["last_update_time"] = time.time()

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the current portfolio state."""
        total_assets_value = Decimal("0.0")
        unrealized_pnl = Decimal("0.0")
        positions_summary = {}

        for asset, pos in self.positions.items():
            total_assets_value += pos.value
            unrealized_pnl += pos.unrealized_pnl
            positions_summary[asset] = {
                "quantity": float(pos.quantity),
                "avg_price": float(pos.avg_price),
                "current_price": float(pos.current_price),
                "value": float(pos.value),
                "unrealized_pnl": float(pos.unrealized_pnl),
            }

        total_value = self.cash + total_assets_value
        total_pnl = self.realized_pnl + unrealized_pnl

        return {
            "cash": float(self.cash),
            "total_assets_value": float(total_assets_value),
            "total_value": float(total_value),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(unrealized_pnl),
            "total_pnl": float(total_pnl),
            "positions": positions_summary,
            "last_update_time": time.time(),
        }

    def get_transaction_history(
            self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve a portion of the transaction history."""
        return list(self.transaction_history)[-limit:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Return the performance statistics of the portfolio tracker."""
        stats = self.portfolio_stats.copy()
        summary = self.get_portfolio_summary()
        stats["current_total_value"] = summary["total_value"]
        stats["current_realized_pnl"] = summary["realized_pnl"]
        stats["current_unrealized_pnl"] = summary["unrealized_pnl"]
        stats["current_total_pnl"] = summary["total_pnl"]
        return stats

    def reset_portfolio(
            self, initial_cash: Union[float, str, Decimal] = "100000.0") -> None:
        """Reset the portfolio to an initial state."""
        self.cash = Decimal(str(initial_cash))
        self.positions = {}
        self.realized_pnl = Decimal("0.0")
        self.transaction_history = []
        self.portfolio_stats = {
            "total_deposits": Decimal(str(initial_cash)),
            "total_withdrawals": Decimal("0.0"),
            "trade_count": 0,
            "total_fees": Decimal("0.0"),
            "last_update_time": time.time(),
        }
        logger.info(f"Portfolio reset to initial cash: {self.cash:.2f}")


def main():
    """Demonstrate PortfolioTracker functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    tracker = PortfolioTracker(initial_cash=10000.0)

    print("\n--- Portfolio Tracker Demo ---")

    # Initial state
    print("\nInitial Portfolio:")
    print(tracker.get_portfolio_summary())

    # Deposit more cash
    tracker.deposit(5000.0)
    print("\nAfter Deposit:")
    print(tracker.get_portfolio_summary())

    # Simulate a buy trade
    tracker.update_position("BTC/USD", "buy", 0.1, 50000.0, fees=5.0)
    print("\nAfter BTC Buy:")
    print(tracker.get_portfolio_summary())

    # Simulate another buy trade for the same asset
    tracker.update_position("BTC/USD", "buy", 0.05, 51000.0, fees=2.5)
    print("\nAfter Second BTC Buy:")
    print(tracker.get_portfolio_summary())

    # Update market price
    tracker.update_market_prices({"BTC/USD": 52000.0})
    print("\nAfter Market Price Update (BTC):")
    print(tracker.get_portfolio_summary())

    # Simulate a sell trade
    tracker.update_position("BTC/USD", "sell", 0.07, 52500.0, fees=3.0)
    print("\nAfter BTC Sell:")
    print(tracker.get_portfolio_summary())

    # Simulate an ETH buy
    tracker.update_position("ETH/USD", "buy", 0.5, 3000.0, fees=1.5)
    print("\nAfter ETH Buy:")
    print(tracker.get_portfolio_summary())

    # Get performance stats
    print("\n--- Performance Statistics ---")
    stats = tracker.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get transaction history
    print("\n--- Transaction History (Last 5) ---")
    for tx in tracker.get_transaction_history(5):
        print(f"  {tx}")

    # Reset portfolio
    tracker.reset_portfolio(initial_cash=20000.0)
    print("\n--- After Reset ---")
    print(tracker.get_portfolio_summary())


if __name__ == "__main__":
    main()
