"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""


from utils.safe_print import safe_print, info, warn, error, success, debug
Bus Core - Central Communication and Routing Layer for Schwabot
== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

This module implements the core bus system for Schwabot, providing the central
communication and routing layer for all event, message, and data flows. It supports
message types, routing rules, middleware, and diagnostics, and is designed for
extensibility and robust logging.

Core Functionality:
- Central message / event routing
- Message type registration
- Routing rules and middleware
- Diagnostics and logging
- Extensible for new message types and protocols
"""
"""
"""

import logging
from typing import Callable, Dict, List, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BusMessage:

    message_type: str
    timestamp: datetime = field(default_factory = datetime.now)
    payload: Dict[str, Any] = field(default_factory = dict)
    source: Optional[str] = None
    destination: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory = dict)


class BusCore:

    def __init__(self):

        self._routes: Dict[str, List[Callable[[BusMessage], None]]] = {}
        self._middleware: List[Callable[[BusMessage], BusMessage]] = []
        self._message_history: List[BusMessage] = []
        logger.info("BusCore initialized")

    def register_route(self, message_type: str, handler: Callable[[BusMessage], None]) -> None:

        if message_type not in self._routes:
            self._routes[message_type] = []
        self._routes[message_type].append(handler)
        logger.debug(f"Handler registered for message type: {message_type}")

    def unregister_route(self, message_type: str, handler: Callable[[BusMessage], None]) -> None:

        if message_type in self._routes:
            self._routes[message_type] = [h for h in self._routes[message_type] if h != handler]
            logger.debug(f"Handler unregistered for message type: {message_type}")

    def add_middleware(self, middleware: Callable[[BusMessage], BusMessage]) -> None:

        self._middleware.append(middleware)
        logger.debug("Middleware added to BusCore")

    def send(self, message: BusMessage) -> None:

# Apply middleware
        for mw in self._middleware:
            try:
                message = mw(message)
            except Exception as e:
                logger.error(f"Error in middleware: {e}")
        self._message_history.append(message)
        handlers = self._routes.get(message.message_type, [])
        logger.info(f"Routing message: {message.message_type} from {message.source} to {message.destination}")
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Error in handler for {message.message_type}: {e}")

    def get_message_history(self, message_type: Optional[str] = None) -> List[BusMessage]:

        if message_type:
            return [m for m in self._message_history if m.message_type == message_type]
        return list(self._message_history)

    def clear_history(self) -> None:

        self._message_history.clear()
        logger.info("BusCore message history cleared")


if __name__ == "__main__":
    bus = BusCore()

    def print_message(msg: BusMessage):

        safe_print(f"Message: {msg.message_type} from {msg.source} to {msg.destination} payload: {msg.payload}")
    bus.register_route("trade", print_message)
    msg = BusMessage(message_type="trade", source="engine", destination="logger",
                        payload={"trade_id": "T123", "amount": 1.5})
    bus.send(msg)
    safe_print("Message history:", bus.get_message_history("trade"))
