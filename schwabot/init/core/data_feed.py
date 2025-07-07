"""
data_feed.py
------------
Lightweight abstraction over *ccxt* to fetch the latest ticker snapshot for a
symbol from a chosen exchange.  Returns the data in a minimal *tick_blob*
string that the rest of Schwabot understands.
"""
from __future__ import annotations

import time
from typing import Dict, Any

try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ccxt package is required for live data feeds.\n"
                       "Install via `pip install ccxt`.\n") from exc


DEFAULT_EXCHANGE = "binance"
DEFAULT_SYMBOL = "BTC/USDC"


def _build_exchange(exchange_id: str) -> "ccxt.Exchange":
    """Return a rate-limited ccxt exchange instance."""
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unsupported exchange id: {exchange_id}")
    return exchange_cls({"enableRateLimit": True})


def fetch_latest_tick(symbol: str = DEFAULT_SYMBOL,
                      exchange_id: str = DEFAULT_EXCHANGE) -> str:
    """Fetch *symbol* ticker from *exchange_id* and return tick_blob string.

    The returned blob format: "{symbol},price={last_price},time={epoch}"
    """
    exchange = _build_exchange(exchange_id)
    ticker: Dict[str, Any] = exchange.fetch_ticker(symbol)

    last_price = ticker.get("last")
    timestamp_ms = ticker.get("timestamp") or int(time.time() * 1000)
    epoch = int(timestamp_ms / 1000)
    return f"{symbol},price={last_price},time={epoch}" 