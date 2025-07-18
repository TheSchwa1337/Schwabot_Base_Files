import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class ActionTracker:
    def __init__(self):
        self.actions: List[Dict[str, Any]] = []

    def track_action(self, action_type: str, details: Dict[str, Any]) -> None:
        action = {
            "timestamp": datetime.now(),
            "action_type": action_type,
            "details": details
        }
        self.actions.append(action)
        logger.info(f"Tracked action: {action_type} - {details}")

    def get_actions(self) -> List[Dict[str, Any]]:
        return self.actions 