import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class TradeLogger:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []

    def log_trade(self, trade_type: str, details: Dict[str, Any]) -> None:
        trade = {
            "timestamp": datetime.now(),
            "trade_type": trade_type,
            "details": details
        }
        self.trades.append(trade)
        logger.info(f"Trade logged: {trade_type} - {details}")

    def get_all_trades(self) -> List[Dict[str, Any]]:
        return self.trades 