"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Tracker Module
=========================
Provides portfolio tracker functionality for the Schwabot trading system.

Main Classes:
- PositionType: Core positiontype functionality
- RiskLevel: Core risklevel functionality
- Position: Core position functionality

Key Functions:
- __post_init__:   post init   operation
- __init__:   init   operation
- _initialize_balances:  initialize balances operation
- update_balance: update balance operation
- get_balance: get balance operation

"""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)
getcontext().prec = 18

@dataclass
class Position:
    """Class for Schwabot trading functionality."""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    current_price: Decimal
    side: str  # 'buy' or 'sell'
    entry_time: float
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    closed: bool = False
    close_time: Optional[float] = None
    close_price: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_unrealized(self, price: Decimal) -> None:
        self.current_price = price
        if self.side == 'buy':
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_price - price) * self.quantity

    def close(self, price: Decimal) -> None:
        self.closed = True
        self.close_time = time.time()
        self.close_price = price
        if self.side == 'buy':
            self.realized_pnl = (price - self.avg_price) * self.quantity
        else:
            self.realized_pnl = (self.avg_price - price) * self.quantity
        self.unrealized_pnl = Decimal('0')


class PortfolioTracker:
    """Class for Schwabot trading functionality."""
    
    def __init__(self, base_currency: str = 'USDC', initial_balances: Optional[Dict[str, Union[float, Decimal]]] = None) -> None:
        self.base_currency = base_currency
        self.balances: Dict[str, Decimal] = {base_currency: Decimal('0')}
        if initial_balances:
            for k, v in initial_balances.items():
                self.balances[k] = Decimal(str(v))
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.transaction_history: List[Dict[str, Any]] = []
        self.last_update = time.time()
        self.realized_pnl = Decimal('0')
        self.unrealized_pnl = Decimal('0')

    def sync_balances(self, ccxt_balance: Dict[str, Any]) -> None:
        """Update balances from a CCXT fetch_balance() result."""
        for asset, info in ccxt_balance.items():
            if isinstance(info, dict) and 'free' in info:
                self.balances[asset] = Decimal(str(info['free']))
            elif isinstance(info, (int, float, Decimal)):
                self.balances[asset] = Decimal(str(info))
        self.last_update = time.time()
        logger.info(f"Portfolio balances synchronized: {self.balances}")

    def open_position(self, symbol: str, quantity: Union[float, Decimal], price: Union[float, Decimal], side: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        quantity = Decimal(str(quantity))
        price = Decimal(str(price))
        pos_id = f"{symbol}_{int(time.time())}_{side}"
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=price,
            current_price=price,
            side=side,
            entry_time=time.time(),
            stop_loss=Decimal(str(stop_loss)) if stop_loss else None,
            take_profit=Decimal(str(take_profit)) if take_profit else None,
            metadata=metadata or {}
        )
        self.positions[pos_id] = position
        # Adjust balances
        base, quote = symbol.split('/')
        if side == 'buy':
            cost = quantity * price
            self.balances[quote] = self.balances.get(quote, Decimal('0')) - cost
            self.balances[base] = self.balances.get(base, Decimal('0')) + quantity
        else:
            self.balances[base] = self.balances.get(base, Decimal('0')) - quantity
            self.balances[quote] = self.balances.get(quote, Decimal('0')) + (quantity * price)
        self.last_update = time.time()
        logger.info(f"Opened position: {pos_id} {position}")
        return pos_id

    def close_position(self, pos_id: str, price: Union[float, Decimal]) -> None:
        if pos_id not in self.positions:
            logger.warning(f"Position {pos_id} not found")
            return None
        position = self.positions[pos_id]
        price = Decimal(str(price))
        position.close(price)
        # Adjust balances
        base, quote = position.symbol.split('/')
        if position.side == 'buy':
            self.balances[base] = self.balances.get(base, Decimal('0')) - position.quantity
            self.balances[quote] = self.balances.get(quote, Decimal('0')) + (position.quantity * price)
        else:
            self.balances[base] = self.balances.get(base, Decimal('0')) + position.quantity
            self.balances[quote] = self.balances.get(quote, Decimal('0')) - (position.quantity * price)
        self.realized_pnl += position.realized_pnl
        self.closed_positions.append(position)
        del self.positions[pos_id]
        self.last_update = time.time()
        logger.info(f"Closed position: {pos_id} {position}")
        return position

    def update_prices(self, price_map: Dict[str, Union[float, Decimal]]) -> None:
        for pos_id, position in self.positions.items():
            if position.symbol in price_map:
                position.update_unrealized(Decimal(str(price_map[position.symbol])))
        self.unrealized_pnl = sum((p.unrealized_pnl for p in self.positions.values()), Decimal('0'))
        self.last_update = time.time()

    def get_portfolio_summary(self) -> Dict[str, Any]:
        total_value = sum(self.balances.values())
        open_positions = [vars(p) for p in self.positions.values()]
        closed_positions = [vars(p) for p in self.closed_positions]
        return {
            'balances': {k: float(v) for k, v in self.balances.items()},
            'open_positions': open_positions,
            'closed_positions': closed_positions,
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_value': float(total_value + self.unrealized_pnl),
            'last_update': self.last_update
        }

    def record_transaction(self, tx: Dict[str, Any]) -> None:
        self.transaction_history.append(tx)
        logger.info(f"Transaction recorded: {tx}")

    def reset(self) -> None:
        self.positions.clear()
        self.closed_positions.clear()
        self.transaction_history.clear()
        self.realized_pnl = Decimal('0')
        self.unrealized_pnl = Decimal('0')
        self.last_update = time.time()
        logger.info("Portfolio tracker reset.")
