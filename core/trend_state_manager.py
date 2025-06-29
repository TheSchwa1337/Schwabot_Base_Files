# -*- coding: utf-8 -*-
"""
Trend State Manager
==================

Manages trend state logging, sub-ring caches, and event triggers for mathematical relay navigation.
Tracks time states for price pulls and event sequencing.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrendState:
    trend_id: str
    bit_depth: int
    channel: str
    sub_ring: int
    event_type: str
    price: float
    volume: float
    timestamp: datetime
    triggered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrendStateManager:
    def __init__(self):
        self.trend_states: List[TrendState] = []
        self.sub_ring_cache: Dict[int, List[TrendState]] = {}
        self.event_log: List[Dict[str, Any]] = []
        self.lock = threading.RLock()

    def log_trend_state(self, trend_state: TrendState) -> None:
        with self.lock:
            self.trend_states.append(trend_state)
            if trend_state.sub_ring not in self.sub_ring_cache:
                self.sub_ring_cache[trend_state.sub_ring] = []
            self.sub_ring_cache[trend_state.sub_ring].append(trend_state)
            logger.info(f"Trend state logged: {trend_state.trend_id}")

    def trigger_event(self, event_type: str, sub_ring: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self.lock:
            event = {
                "event_type": event_type,
                "sub_ring": sub_ring,
                "timestamp": datetime.now(),
                "metadata": metadata or {},
            }
            self.event_log.append(event)
            logger.info(f"Event triggered: {event_type} on sub_ring {sub_ring}")

    def get_recent_trend_states(self, minutes: int = 10) -> List[TrendState]:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        with self.lock:
            return [ts for ts in self.trend_states if ts.timestamp >= cutoff]

    def get_sub_ring_states(self, sub_ring: int) -> List[TrendState]:
        with self.lock:
            return self.sub_ring_cache.get(sub_ring, [])

    def get_event_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            return self.event_log[-limit:]
