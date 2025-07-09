"""
risk_manager.py
---------------
Handles trailing stop-loss, profit locking, TTL exits, and fail-safe fallbacks.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional


class RiskManager:
    """Manages risk rules for active trades."""

    def __init__(
        self,
        default_trailing_pct: float = 0.015,
        default_lock_profit_pct: float = 0.025,
        default_ttl: int = 300,
        profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        profile_path: Optional[str] = None,
    ) -> None:
        """
        default_trailing_pct: global stop-loss fraction below high-water mark
        default_lock_profit_pct: global profit threshold to begin locking
        default_ttl: global max trade duration in seconds before forced exit
        profiles: optional dict mapping symbol -> {trailing_pct, lock_profit_pct, ttl}
        profile_path: optional JSON file path to load per-symbol profiles
        """
        self.default_trailing_pct = default_trailing_pct
        self.default_lock_profit_pct = default_lock_profit_pct
        self.default_ttl = default_ttl
        # Load per-symbol profiles
        self.profiles: Dict[str, Dict[str, Any]] = {}
        if profile_path:
            try:
                with open(profile_path, 'r') as f:
                    self.profiles = json.load(f)
            except Exception as e:
                print(f"[RiskManager] Failed to load profile_path: {e}")
        if profiles:
            self.profiles.update(profiles)
        # active_trades maps trade_id -> {'entry','high','time','symbol'}
        self.active_trades: Dict[str, Dict[str, Any]] = {}

    def register_trade(
        self,
        trade_id: str,
        entry_price: float,
        timestamp: Optional[float] = None,
        symbol: Optional[str] = None,
    ) -> None:
        """Register a new trade with entry price, start time, and symbol."""
        self.active_trades[trade_id] = {
            'entry': entry_price,
            'high': entry_price,
            'time': timestamp or time.time(),
            'symbol': symbol,
        }

    def update_price(self,  trade_id: str, current_price: float) -> str:
        """Update the trade with current price and return action:

        - 'HOLD': no action
        - 'STOP': trailing stop-loss hit
        - 'LOCK': profit-lock threshold reached
        - 'TTL_EXIT': time-to-live exceeded
        - 'NO_TRADE': trade_id not found
        """
        trade = self.active_trades.get(trade_id)
        if not trade:
            return 'NO_TRADE'

        # Determine per-symbol config or use defaults
        symbol = trade.get('symbol')
        cfg = self.profiles.get(symbol, {})
        trailing_pct = cfg.get('trailing_pct', self.default_trailing_pct)
        lock_profit_pct = cfg.get('lock_profit_pct', self.default_lock_profit_pct)
        ttl = cfg.get('ttl', self.default_ttl)

        # Update high-water mark
        if current_price > trade['high']:
            trade['high'] = current_price

        # Trailing stop check
        if current_price < trade['high'] * (1 - trailing_pct):
            return 'STOP'

        # Profit lock trigger
        if current_price > trade['entry'] * (1 + lock_profit_pct):
            return 'LOCK'

        # TTL exit
        if time.time() - trade['time'] > ttl:
            return 'TTL_EXIT'

        return 'HOLD'

    def cancel_trade(self,  trade_id: str) -> None:
        """Cancel and remove a trade from active tracking."""
        self.active_trades.pop(trade_id, None)
