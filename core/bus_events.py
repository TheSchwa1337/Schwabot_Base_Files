# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
# #!/usr/bin/env python3
"""
Bus Events - Core Event Bus System for Schwabot
==============================================

This module implements the event bus system for Schwabot, supporting event
definition, dispatch, subscription, and logging. It is designed for extensibility
and type safety, and integrates with the main trading pipeline.

Core Functionality:
- Event definition and typing
- Event dispatch and subscription
- Event logging and history
- Extensible event types for trading, system, and error events
"""

import logging
from typing import Callable, Dict, List, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Event base class
@dataclass
class BusEvent:
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    payload: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

# Example event types
@dataclass
class TradeEvent(BusEvent):
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    side: Optional[str] = None  # 'buy' or 'sell'

@dataclass
class SystemEvent(BusEvent):
    system_status: Optional[str] = None
    message: Optional[str] = None

@dataclass
class ErrorEvent(BusEvent):
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    severity: Optional[str] = None

# Event bus implementation
class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[BusEvent], None]]] = {}
        self._event_history: List[BusEvent] = []
        logger.info("EventBus initialized")

    def subscribe(self, event_type: str, handler: Callable[[BusEvent], None]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Handler subscribed to event type: {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable[[BusEvent], None]) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type] = [h for h in self._subscribers[event_type] if h != handler]
            logger.debug(f"Handler unsubscribed from event type: {event_type}")

    def dispatch(self, event: BusEvent) -> None:
        self._event_history.append(event)
        handlers = self._subscribers.get(event.event_type, [])
        logger.info(f"Dispatching event: {event.event_type} at {event.timestamp}")
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")

    def get_event_history(self, event_type: Optional[str] = None) -> List[BusEvent]:
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type]
        return list(self._event_history)

    def clear_history(self) -> None:
        self._event_history.clear()
        logger.info("Event history cleared")

# Example usage

if __name__ == "__main__":
    bus = EventBus()

    def print_trade(event: TradeEvent):
        safe_print(f"Trade Event: {event.trade_id} {event.symbol} {event.price} {event.volume} {event.side}")

    bus.subscribe("trade", print_trade)
    trade_event = TradeEvent(event_type="trade", trade_id="T123", symbol="BTCUSD", price=45000.0, volume=1.5, side="buy")
    bus.dispatch(trade_event)
    safe_print("Event history:", bus.get_event_history("trade"))
