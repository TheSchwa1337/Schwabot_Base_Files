"""
Order Wall Analysis Module - Analyze buy/sell walls to provide trading signals.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OrderWallAnalyzer:
    """
    Analyzes order book walls and returns metrics.
    """
    def analyze_order_book(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns dict with 'buy_wall_strength' and 'sell_wall_strength'.
        """
        buy_strength = sum([vol for price, vol in order_book.get('bids', [])[:5]])
        sell_strength = sum([vol for price, vol in order_book.get('asks', [])[:5]])
        logger.debug(f"OrderWallAnalyzer: buy_strength={buy_strength}, sell_strength={sell_strength}")
        return {'buy_wall_strength': buy_strength, 'sell_wall_strength': sell_strength} 