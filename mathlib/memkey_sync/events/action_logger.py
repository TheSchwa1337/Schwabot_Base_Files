import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class ActionLogger:
    def __init__(self):
        self.actions: List[Dict[str, Any]] = []

    def log_action(self, action_type: str, details: Dict[str, Any]) -> None:
        action = {
            "timestamp": datetime.now(),
            "action_type": action_type,
            "details": details
        }
        self.actions.append(action)
        logger.info(f"Action logged: {action_type} - {details}")

    def get_all_actions(self) -> List[Dict[str, Any]]:
        return self.actions 