# -*- coding: utf-8 -*-
""""""
QuickTime Event Manager
======================

Monitors for rapid market changes, hash frequency shifts, and priority events.
Captures event snapshots, logs them, and notifies TrendStateManager, BasketRelayEngine, and MathematicalBacklogManager.
""""""

import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class QuickTimeEventManager:
    def __init__()
        self,
            trend_manager: Any,
                basket_engine: Any,
                backlog_manager: Any,
                event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                ):
        self.trend_manager = trend_manager
        self.basket_engine = basket_engine
        self.backlog_manager = backlog_manager
        self.event_callback = event_callback
        self.lock = threading.RLock()
        self.event_log: list = []

    def detect_and_log_event(self, event_type: str, context: Dict[str, Any]) -> None:
        with self.lock:
            event = {"event_type": event_type, "context": context, "timestamp": datetime.now().isoformat()}
            self.event_log.append(event)
            logger.info(f"QuickTime event detected: {event_type}")
            # Log to backlog
            self.backlog_manager.add_issue()
                issue=f"QuickTime event: {event_type}", module="QuickTimeEventManager", severity="high"
            )
            # Notify trend manager
            if hasattr(self.trend_manager, "trigger_event"):
                self.trend_manager.trigger_event(event_type, context.get("sub_ring", 0), context)
            # Notify basket engine
            if hasattr(self.basket_engine, "process_basket_event") and "basket_id" in context:
                self.basket_engine.process_basket_event()
                    basket_id=context["basket_id"],
                        tier=context.get("tier", "unknown"),
                            bit_depth=context.get("bit_depth", 32),
                            channel=context.get("channel", "primary"),
                            price=context.get("price", 0.0),
                            volume=context.get("volume", 0.0),
                            sub_ring=context.get("sub_ring", 0),
                            )
            # Custom callback (e.g., ghost logic)
            if self.event_callback:
                self.event_callback(event)

    def get_event_log(self, limit: int = 100) -> list:
        with self.lock:
            return self.event_log[-limit:]
