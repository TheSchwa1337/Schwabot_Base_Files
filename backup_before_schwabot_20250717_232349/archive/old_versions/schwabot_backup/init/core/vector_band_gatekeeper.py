"""
vector_band_gatekeeper.py
------------------------
This module analyses market context across multiple temporal bands (short,
mid, long) using real TA indicators (EMA crossover, RSI, MACD) and decides
whether each band is aligned for trade execution.
"""

from __future__ import annotations

import ccxt
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

# Rate-limited exchange instance for fetching historical OHLCV
exchange = ccxt.binance({"enableRateLimit": True})


def fetch_candles(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV candles and return as DataFrame."""
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA, RSI, and MACD indicators on close price."""
    df["ema_12"] = EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(df["close"], window=26).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_diff"] = macd_obj.macd_diff()
    # Long-term EMA for trend confirmation
    df["ema_200"] = EMAIndicator(df["close"], window=200).ema_indicator()
    return df


def _symbol_from_blob(tick_blob: str) -> str:
    """Parse symbol (e.g. 'BTC/USDC') from tick_blob string."""
    return tick_blob.split(",", 1)[0]


def confirm_short_drift(tick_blob: str) -> bool:
    """Short-term (5m): EMA12>EMA26, RSI9>50, MACD rising."""
    symbol = _symbol_from_blob(tick_blob)
    df = fetch_candles(symbol, timeframe="5m")
    df = apply_indicators(df)
    latest = df.iloc[-1]
    # Compute RSI9 separately
    rsi9 = RSIIndicator(df["close"], window=9).rsi().iloc[-1]
    return latest["ema_12"] > latest["ema_26"] and rsi9 > 50 and latest["macd_diff"] > 0


def confirm_mid_vector(tick_blob: str) -> bool:
    """Mid-term (1h): RSI14>50 and MACD rising."""
    symbol = _symbol_from_blob(tick_blob)
    df = fetch_candles(symbol, timeframe="1h")
    df = apply_indicators(df)
    latest = df.iloc[-1]
    return latest["rsi"] > 50 and latest["macd_diff"] > 0


def confirm_long_trend(tick_blob: str) -> bool:
    """Long-term (4h): Price above EMA200 and MACD rising."""
    symbol = _symbol_from_blob(tick_blob)
    df = fetch_candles(symbol, timeframe="4h")
    df = apply_indicators(df)
    latest = df.iloc[-1]
    return latest["close"] > latest["ema_200"] and latest["macd_diff"] > 0
