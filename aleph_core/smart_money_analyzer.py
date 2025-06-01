"""
Smart Money Analysis Module for Schwabot
Implements advanced market microstructure analysis including spoof detection,
wall analysis, and liquidity resonance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SmartMoneyMetrics:
    """Container for Smart Money analysis metrics"""
    spoof_score: float = 0.0  # 0-1 score indicating likelihood of spoofing
    wall_score: float = 0.0   # 0-1 score for bid/ask wall presence
    velocity: str = "NORMAL"  # Price/volume velocity state
    liquidity_resonance: str = "NORMAL"  # Liquidity interaction state
    smart_money_score: float = 0.0  # Aggregated Smart Money score

class SmartMoneyAnalyzer:
    """
    Analyzes market microstructure for Smart Money patterns including:
    - Spoof detection (fake orders)
    - Wall analysis (large bid/ask walls)
    - Liquidity resonance (order book interaction)
    - Velocity analysis (price/volume movement)
    """
    
    def __init__(self,
                 spoof_threshold: float = 0.7,
                 wall_size_multiplier: float = 2.0,
                 velocity_period: int = 5,
                 liquidity_window: int = 10):
        self.spoof_threshold = spoof_threshold
        self.wall_size_multiplier = wall_size_multiplier
        self.velocity_period = velocity_period
        self.liquidity_window = liquidity_window
        
        # Internal state
        self.order_book_history: List[Dict] = []
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.trade_history: List[Dict] = []
        
    def calculate_spoof_score(self, current_book: Dict, previous_book: Dict) -> float:
        """
        Calculate spoofing score based on order book changes.
        Higher score indicates higher likelihood of spoofing.
        """
        if not previous_book:
            return 0.0
            
        score = 0.0
        
        # Check for large orders that disappear quickly
        for side in ['bids', 'asks']:
            for price, volume in current_book.get(side, {}).items():
                if price in previous_book.get(side, {}):
                    prev_volume = previous_book[side][price]
                    if volume < prev_volume * 0.1 and prev_volume > 1000:
                        score += 0.2
                        
        return min(score, 1.0)
        
    def detect_walls(self, order_book: Dict, avg_volume: float) -> Tuple[bool, bool, Optional[float], Optional[float]]:
        """
        Detect significant bid/ask walls in the order book.
        Returns (bid_wall_present, ask_wall_present, bid_wall_price, ask_wall_price)
        """
        if not order_book:
            return False, False, None, None
            
        max_bid_volume = max(order_book.get('bids', {}).values()) if order_book.get('bids') else 0
        max_ask_volume = max(order_book.get('asks', {}).values()) if order_book.get('asks') else 0
        
        bid_wall = max_bid_volume > avg_volume * self.wall_size_multiplier
        ask_wall = max_ask_volume > avg_volume * self.wall_size_multiplier
        
        bid_wall_price = max(order_book['bids'].keys(), key=(lambda k: order_book['bids'][k])) if bid_wall else None
        ask_wall_price = min(order_book['asks'].keys(), key=(lambda k: order_book['asks'][k])) if ask_wall else None
        
        return bid_wall, ask_wall, bid_wall_price, ask_wall_price
        
    def calculate_velocity(self, prices: List[float], volumes: List[float]) -> str:
        """Calculate price/volume velocity state"""
        if len(prices) < self.velocity_period:
            return "NORMAL"
            
        recent_prices = prices[-self.velocity_period:]
        price_change = recent_prices[-1] - recent_prices[0]
        
        if price_change > prices[0] * 0.001:  # 0.1% change
            return "HIGH_UP"
        elif price_change < -prices[0] * 0.001:
            return "HIGH_DOWN"
        return "NORMAL"
        
    def analyze_liquidity_resonance(self, trades: List[Dict]) -> str:
        """
        Analyze how trades interact with available liquidity.
        Returns resonance state: SWEEP_RESONANCE, CONSOLIDATION_RESONANCE, or NORMAL_RESONANCE
        """
        if not trades:
            return "NORMAL_RESONANCE"
            
        total_volume = sum(t['volume'] for t in trades)
        if total_volume == 0:
            return "NORMAL_RESONANCE"
            
        # Check for sweep patterns
        num_distinct_prices = len(set(t['price'] for t in trades))
        if num_distinct_prices > 5 and total_volume > 1000:
            return "SWEEP_RESONANCE"
        elif num_distinct_prices <= 2 and total_volume < 100:
            return "CONSOLIDATION_RESONANCE"
            
        return "NORMAL_RESONANCE"
        
    def analyze_tick(self,
                    price: float,
                    volume: float,
                    order_book: Dict,
                    trades: List[Dict]) -> SmartMoneyMetrics:
        """
        Analyze a single tick for Smart Money patterns.
        Returns SmartMoneyMetrics with all analysis results.
        """
        # Update history
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.order_book_history.append(order_book)
        self.trade_history.extend(trades)
        
        # Keep history limited
        max_history = max(self.velocity_period, self.liquidity_window)
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.order_book_history = self.order_book_history[-max_history:]
            self.trade_history = self.trade_history[-max_history:]
            
        # Calculate metrics
        previous_book = self.order_book_history[-2] if len(self.order_book_history) > 1 else {}
        avg_volume = np.mean(self.volume_history) if self.volume_history else 1.0
        
        spoof_score = self.calculate_spoof_score(order_book, previous_book)
        bid_wall, ask_wall, _, _ = self.detect_walls(order_book, avg_volume)
        velocity = self.calculate_velocity(self.price_history, self.volume_history)
        liquidity_resonance = self.analyze_liquidity_resonance(self.trade_history)
        
        # Calculate aggregated Smart Money score
        smart_money_score = 0.0
        if spoof_score > 0.5:
            smart_money_score += 0.3
        if bid_wall or ask_wall:
            smart_money_score += 0.2
        if "HIGH" in velocity:
            smart_money_score += 0.1
        if "SWEEP" in liquidity_resonance:
            smart_money_score += 0.4
            
        return SmartMoneyMetrics(
            spoof_score=spoof_score,
            wall_score=float(bid_wall or ask_wall),
            velocity=velocity,
            liquidity_resonance=liquidity_resonance,
            smart_money_score=smart_money_score
        ) 