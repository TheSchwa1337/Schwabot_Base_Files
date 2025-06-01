"""
Strategy Replay Module for Schwabot
Analyzes past trades with Smart Money context for learning and optimization.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import json
import hashlib

from .smart_money_analyzer import SmartMoneyAnalyzer, SmartMoneyMetrics

@dataclass
class ReplayTrade:
    """Container for a single trade with Smart Money context"""
    trade_id: str
    timestamp: datetime
    price: float
    volume: float
    side: str  # "BUY" or "SELL"
    profit: float
    smart_money_metrics: SmartMoneyMetrics
    pattern_id: str
    confidence: float

class StrategyReplayer:
    """
    Analyzes past trades with Smart Money context to:
    1. Learn from successful/failed trades
    2. Optimize strategy parameters
    3. Generate insights about market conditions
    """
    
    def __init__(self,
                 replay_window: int = 1000,  # Number of trades to keep in memory
                 profit_threshold: float = 0.001,  # Minimum profit to consider successful
                 smart_money_threshold: float = 0.6):  # Minimum Smart Money score
        self.replay_window = replay_window
        self.profit_threshold = profit_threshold
        self.smart_money_threshold = smart_money_threshold
        self.smart_money = SmartMoneyAnalyzer()
        self.trade_history: List[ReplayTrade] = []
        
    def add_trade(self,
                 trade_id: str,
                 timestamp: datetime,
                 price: float,
                 volume: float,
                 side: str,
                 profit: float,
                 order_book: Dict,
                 trades: List[Dict],
                 pattern_id: str,
                 confidence: float) -> None:
        """
        Add a trade to the replay history with Smart Money analysis.
        """
        # Get Smart Money metrics for this trade
        smart_money_metrics = self.smart_money.analyze_tick(
            price=price,
            volume=volume,
            order_book=order_book,
            trades=trades
        )
        
        # Create replay trade
        replay_trade = ReplayTrade(
            trade_id=trade_id,
            timestamp=timestamp,
            price=price,
            volume=volume,
            side=side,
            profit=profit,
            smart_money_metrics=smart_money_metrics,
            pattern_id=pattern_id,
            confidence=confidence
        )
        
        # Add to history
        self.trade_history.append(replay_trade)
        
        # Keep history limited
        if len(self.trade_history) > self.replay_window:
            self.trade_history = self.trade_history[-self.replay_window:]
            
    def analyze_trade(self, trade_id: str) -> Dict:
        """
        Analyze a specific trade with Smart Money context.
        Returns detailed analysis of the trade conditions.
        """
        trade = next((t for t in self.trade_history if t.trade_id == trade_id), None)
        if not trade:
            return {"error": "Trade not found"}
            
        return {
            'trade_id': trade.trade_id,
            'timestamp': trade.timestamp.isoformat(),
            'price': trade.price,
            'volume': trade.volume,
            'side': trade.side,
            'profit': trade.profit,
            'pattern_id': trade.pattern_id,
            'confidence': trade.confidence,
            'smart_money_metrics': {
                'spoof_score': trade.smart_money_metrics.spoof_score,
                'wall_score': trade.smart_money_metrics.wall_score,
                'velocity': trade.smart_money_metrics.velocity,
                'liquidity_resonance': trade.smart_money_metrics.liquidity_resonance,
                'smart_money_score': trade.smart_money_metrics.smart_money_score
            },
            'success': trade.profit > self.profit_threshold,
            'smart_money_aligned': trade.smart_money_metrics.smart_money_score > self.smart_money_threshold
        }
        
    def get_pattern_insights(self, pattern_id: str) -> Dict:
        """
        Get insights about a specific pattern's performance with Smart Money context.
        """
        pattern_trades = [t for t in self.trade_history if t.pattern_id == pattern_id]
        if not pattern_trades:
            return {"error": "No trades found for pattern"}
            
        successful_trades = [t for t in pattern_trades if t.profit > self.profit_threshold]
        smart_money_aligned = [t for t in pattern_trades if t.smart_money_metrics.smart_money_score > self.smart_money_threshold]
        
        return {
            'pattern_id': pattern_id,
            'total_trades': len(pattern_trades),
            'successful_trades': len(successful_trades),
            'success_rate': len(successful_trades) / len(pattern_trades),
            'avg_profit': np.mean([t.profit for t in pattern_trades]),
            'smart_money_aligned_trades': len(smart_money_aligned),
            'smart_money_alignment_rate': len(smart_money_aligned) / len(pattern_trades),
            'velocity_distribution': {
                'HIGH_UP': sum(1 for t in pattern_trades if t.smart_money_metrics.velocity == "HIGH_UP"),
                'HIGH_DOWN': sum(1 for t in pattern_trades if t.smart_money_metrics.velocity == "HIGH_DOWN"),
                'NORMAL': sum(1 for t in pattern_trades if t.smart_money_metrics.velocity == "NORMAL")
            },
            'liquidity_distribution': {
                'SWEEP': sum(1 for t in pattern_trades if "SWEEP" in t.smart_money_metrics.liquidity_resonance),
                'CONSOLIDATION': sum(1 for t in pattern_trades if "CONSOLIDATION" in t.smart_money_metrics.liquidity_resonance),
                'NORMAL': sum(1 for t in pattern_trades if "NORMAL" in t.smart_money_metrics.liquidity_resonance)
            }
        }
        
    def get_optimization_suggestions(self) -> Dict:
        """
        Generate suggestions for strategy optimization based on Smart Money analysis.
        """
        if not self.trade_history:
            return {"error": "No trade history available"}
            
        successful_trades = [t for t in self.trade_history if t.profit > self.profit_threshold]
        if not successful_trades:
            return {"error": "No successful trades found"}
            
        # Analyze conditions for successful trades
        successful_conditions = {
            'avg_smart_money_score': np.mean([t.smart_money_metrics.smart_money_score for t in successful_trades]),
            'avg_spoof_score': np.mean([t.smart_money_metrics.spoof_score for t in successful_trades]),
            'avg_wall_score': np.mean([t.smart_money_metrics.wall_score for t in successful_trades]),
            'velocity_distribution': {
                'HIGH_UP': sum(1 for t in successful_trades if t.smart_money_metrics.velocity == "HIGH_UP"),
                'HIGH_DOWN': sum(1 for t in successful_trades if t.smart_money_metrics.velocity == "HIGH_DOWN"),
                'NORMAL': sum(1 for t in successful_trades if t.smart_money_metrics.velocity == "NORMAL")
            },
            'liquidity_distribution': {
                'SWEEP': sum(1 for t in successful_trades if "SWEEP" in t.smart_money_metrics.liquidity_resonance),
                'CONSOLIDATION': sum(1 for t in successful_trades if "CONSOLIDATION" in t.smart_money_metrics.liquidity_resonance),
                'NORMAL': sum(1 for t in successful_trades if "NORMAL" in t.smart_money_metrics.liquidity_resonance)
            }
        }
        
        # Generate suggestions
        suggestions = []
        
        # Smart Money score threshold
        if successful_conditions['avg_smart_money_score'] > self.smart_money_threshold:
            suggestions.append({
                'type': 'smart_money_threshold',
                'current': self.smart_money_threshold,
                'suggested': successful_conditions['avg_smart_money_score'],
                'reason': 'Higher Smart Money scores correlate with successful trades'
            })
            
        # Spoof score threshold
        if successful_conditions['avg_spoof_score'] < 0.5:
            suggestions.append({
                'type': 'spoof_threshold',
                'current': 0.5,
                'suggested': successful_conditions['avg_spoof_score'],
                'reason': 'Lower spoof scores correlate with successful trades'
            })
            
        # Velocity conditions
        if successful_conditions['velocity_distribution']['HIGH_UP'] > len(successful_trades) * 0.3:
            suggestions.append({
                'type': 'velocity_condition',
                'condition': 'HIGH_UP',
                'reason': 'High upward velocity appears frequently in successful trades'
            })
            
        # Liquidity conditions
        if successful_conditions['liquidity_distribution']['SWEEP'] > len(successful_trades) * 0.3:
            suggestions.append({
                'type': 'liquidity_condition',
                'condition': 'SWEEP_RESONANCE',
                'reason': 'Sweep resonance appears frequently in successful trades'
            })
            
        return {
            'successful_conditions': successful_conditions,
            'suggestions': suggestions
        }
        
    def export_replay_data(self, filepath: str) -> None:
        """
        Export replay data to a JSON file for external analysis.
        """
        data = {
            'trades': [
                {
                    'trade_id': t.trade_id,
                    'timestamp': t.timestamp.isoformat(),
                    'price': t.price,
                    'volume': t.volume,
                    'side': t.side,
                    'profit': t.profit,
                    'pattern_id': t.pattern_id,
                    'confidence': t.confidence,
                    'smart_money_metrics': {
                        'spoof_score': t.smart_money_metrics.spoof_score,
                        'wall_score': t.smart_money_metrics.wall_score,
                        'velocity': t.smart_money_metrics.velocity,
                        'liquidity_resonance': t.smart_money_metrics.liquidity_resonance,
                        'smart_money_score': t.smart_money_metrics.smart_money_score
                    }
                }
                for t in self.trade_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2) 