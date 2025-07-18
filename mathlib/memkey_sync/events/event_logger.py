import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class EventLogger:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log(self, event_type: str, details: Dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "details": details
        }
        self.events.append(event)
        logger.info(f"Event logged: {event_type} - {details}")

    def get_all_events(self) -> List[Dict[str, Any]]:
        return self.events 