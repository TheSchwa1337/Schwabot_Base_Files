"""
Data Feed System
===============

Lightweight abstraction over CCXT to fetch the latest ticker snapshot for a
symbol from a chosen exchange. Returns the data in a minimal tick_blob
string that the rest of Schwabot understands.

Mathematical Components:
- Timestamp conversion (ms to epoch)
- Standardized data format: "{symbol},price={price},time={epoch}"
"""

import time
from typing import Any, Dict

try:
    import ccxt
except ImportError as exc:
    raise RuntimeError("ccxt package is required for live data feeds.\n" "Install via `pip install ccxt`.\n") from exc


DEFAULT_EXCHANGE = "binance"
DEFAULT_SYMBOL = "BTC/USDC"


class DataFeed:
    """Manages market data feeds from various exchanges."""

    def __init__(self,  exchange_id: str = DEFAULT_EXCHANGE) -> None:
        """Initialize data feed with specified exchange."""
        self.exchange_id = exchange_id
        self.exchange = self._build_exchange(exchange_id)

    def _build_exchange(self,  exchange_id: str) -> ccxt.Exchange:
        """Return a rate-limited ccxt exchange instance."""
        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unsupported exchange id: {exchange_id}")
        return exchange_cls({"enableRateLimit": True})

    def fetch_latest_tick(self,  symbol: str = DEFAULT_SYMBOL) -> str:
        """Fetch symbol ticker and return standardized tick_blob string.

        Returns format: "{symbol},price={last_price},time={epoch}"
        """
        try:
            ticker: Dict[str, Any] = self.exchange.fetch_ticker(symbol)

            last_price = ticker.get("last")
            if last_price is None:
                raise ValueError(f"No last price found for {symbol}")

            timestamp_ms = ticker.get("timestamp") or int(time.time() * 1000)
            epoch = int(timestamp_ms / 1000)

            return f"{symbol},price={last_price},time={epoch}"

        except Exception as e:
            raise RuntimeError(f"Failed to fetch tick for {symbol}: {e}")

    def fetch_ohlcv(self,  symbol: str, timeframe: str = "1m", limit: int = 100) -> list:
        """Fetch OHLCV data for technical analysis."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: {e}")

    def fetch_order_book(self,  symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch order book data."""
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            raise RuntimeError(f"Failed to fetch order book for {symbol}: {e}")

    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information and capabilities."""
        try:
            return {
                "exchange_id": self.exchange_id,
                "has_fetch_ticker": hasattr(self.exchange, 'fetch_ticker'),
                "has_fetch_ohlcv": hasattr(self.exchange, 'fetch_ohlcv'),
                "has_fetch_order_book": hasattr(self.exchange, 'fetch_order_book'),
                "rate_limit": getattr(self.exchange, 'rateLimit', None),
            }
        except Exception as e:
            return {"error": str(e)}


def fetch_latest_tick(symbol: str = DEFAULT_SYMBOL, exchange_id: str = DEFAULT_EXCHANGE) -> str:
    """Convenience function to fetch latest tick data."""
    feed = DataFeed(exchange_id)
    return feed.fetch_latest_tick(symbol)


def _build_exchange(exchange_id: str) -> ccxt.Exchange:
    """Return a rate-limited ccxt exchange instance."""
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unsupported exchange id: {exchange_id}")
    return exchange_cls({"enableRateLimit": True})
