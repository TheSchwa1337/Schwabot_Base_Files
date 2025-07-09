"""
trade_executor.py
-----------------
Responsible for executing (or simulating) trades once all security gates pass.

Two modes:
• live  – sends a real order to the exchange via CCXT
• dry   – prints/logs what *would* have happened (safe for testing)
"""

from __future__ import annotations

import time
from typing import Any, Dict

try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ccxt package is required for trade execution.\n" "Install via `pip install ccxt`.\n") from exc

DEFAULT_EXCHANGE = "binance"


class TradeExecutor:
    """Minimal wrapper around *ccxt* for market orders."""

    def __init__(self,   api_keys: Dict[str, str], mode: str = "dry") -> None:
        self.mode = mode.lower()
        if self.mode not in {"dry", "live"}:
            raise ValueError("mode must be 'dry' or 'live'")

        self.exchange_id = api_keys.get("exchange", DEFAULT_EXCHANGE)
        self.symbol = api_keys.get("symbol", "BTC/USDC")
        self.amount = float(api_keys.get("amount", 0.001))  # default 0.001 BTC

        params: Dict[str, Any] = {"enableRateLimit": True}
        if self.mode == "live":
            params.update(
                {
                    "apiKey": api_keys.get("api_key"),
                    "secret": api_keys.get("api_secret"),
                    "password": api_keys.get("passphrase"),
                }
            )
        self.exchange = getattr(ccxt, self.exchange_id)(params)

    # ------------------------------------------------------------------
    def execute(self,   side: str = "buy") -> None:
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        if self.mode == "dry":
            price = self._fetch_price()
            cost = price * self.amount
            print(
                f"[TradeExecutor] 🟣 DRY RUN: ENTRY {side.upper()} {self.amount} {self.symbol} @ {price} (cost ~{cost:."
    2f})"
            )
            # Simulate exit if exit_strategy set on instance
            if hasattr(self, 'exit_strategy') and self.exit_strategy:
                exit_price, exit_time = self.exit_strategy
                opp = 'sell' if side == 'buy' else 'buy'
                print(
                    f"[TradeExecutor] 🟣 DRY RUN: EXIT {opp.upper()} {self.amount} {self.symbol} @ {exit_price} after {exit"
    "
    _time}s"
                )
            return

        # live mode entry
        try:
            order = self.exchange.create_market_order(self.symbol, side, self.amount)
            print(f"[TradeExecutor] ✅ LIVE ENTRY ORDER SENT: {order.get('id', '?')}")
            entry_time = time.time()
            entry_price = float(order.get('price', self._fetch_price()))
        except Exception as exc:
            print(f"[TradeExecutor] ❌ Live entry order failed: {exc}")
            return

        # If exit strategy provided, monitor for exit
        if hasattr(self, 'exit_strategy') and self.exit_strategy:
            exit_price, exit_time = self.exit_strategy
            opp_side = 'sell' if side == 'buy' else 'buy'
            print(f"[TradeExecutor] 🟢 Monitoring for exit @ {exit_price} within {exit_time}s")
            start = time.time()
            while True:
                current_price = self._fetch_price()
                elapsed = time.time() - start
                # Check profit target
                if (side == 'buy' and current_price >= exit_price) or (side == 'sell' and current_price <= exit_price):
                    try:
                        exit_order = self.exchange.create_market_order(self.symbol, opp_side, self.amount)
                        print(f"[TradeExecutor] ✅ LIVE EXIT ORDER SENT: {exit_order.get('id', '?')} @ {current_price}")
                    except Exception as e:
                        print(f"[TradeExecutor] ❌ Live exit order failed: {e}")
                    break
                # Check timeout
                if elapsed >= exit_time:
                    print(
                        f"[TradeExecutor] ⚠️  EXIT TTL reached ({elapsed:.0f}s). Placing limit exit to preserve profit."
                    )
                    # Decide limit price: lock current profit or break-even
                    current_price = self._fetch_price()
                    if current_price >= entry_price:
                        limit_price = current_price
                        print(f"[TradeExecutor] 🛡️  Locking profit via limit exit at current price {limit_price}")
                    else:
                        limit_price = entry_price
                        print(f"[TradeExecutor] 🛡️  Setting break-even limit at entry price {limit_price}")
                    try:
                        # Place a limit order to exit without taking a loss
                        exit_order = self.exchange.create_order(
                            self.symbol, 'limit', opp_side, self.amount, limit_price
                        )
                        print(f"[TradeExecutor] ✅ LIMIT EXIT ORDER SENT: {exit_order.get('id', '?')} @ {limit_price}")
                    except Exception as e:
                        print(f"[TradeExecutor] ❌ Limit exit failed: {e}")
                    break
                time.sleep(1)

    # ------------------------------------------------------------------
    def _fetch_price(self) -> float:
        ticker = self.exchange.fetch_ticker(self.symbol)
        return float(ticker.get("last"))
