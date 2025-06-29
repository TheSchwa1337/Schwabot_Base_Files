# -*- coding: utf-8 -*-
"""
Basket Relay Engine
==================

Handles basket-tier navigation, sub-ring logic, and profit relay for mathematical relay navigation.
Integrates with TrendStateManager.
"""

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.trend_state_manager import TrendState, TrendStateManager

logger = logging.getLogger(__name__)


class BasketRelayEngine:
    def __init__(self, trend_manager: TrendStateManager):
        self.trend_manager = trend_manager
        self.basket_states: List[Dict[str, Any]] = []
        self.lock = threading.RLock()

    def process_basket_event(
        self, basket_id: str, tier: str, bit_depth: int, channel: str, price: float, volume: float, sub_ring: int
    ) -> None:
        with self.lock:
            event = {
                "basket_id": basket_id,
                "tier": tier,
                "bit_depth": bit_depth,
                "channel": channel,
                "price": price,
                "volume": volume,
                "sub_ring": sub_ring,
                "timestamp": datetime.now(),
            }
            self.basket_states.append(event)
            # Log as trend state
            trend_state = TrendState(
                trend_id=f"basket_{basket_id}_{int(datetime.now().timestamp())}",
                bit_depth=bit_depth,
                channel=channel,
                sub_ring=sub_ring,
                event_type="basket_event",
                price=price,
                volume=volume,
                timestamp=datetime.now(),
            )
            self.trend_manager.log_trend_state(trend_state)
            logger.info(f"Basket event processed: {basket_id}, tier={tier}")

    def get_basket_states(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            return self.basket_states[-limit:]
