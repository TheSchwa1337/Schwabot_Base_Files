"""
Strategy Replay Module for Schwabot
Analyzes past trades with Smart Money context for learning and optimization.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import json
import hashlib
import logging

from .smart_money_analyzer import SmartMoneyAnalyzer, SmartMoneyMetrics
from .entropy_analyzer import EntropyAnalyzer
from .paradox_visualizer import ParadoxVisualizer
from .unitizer import AlephUnitizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    trade_hash: str  # Added for faster lookup
    entropy_tag: Optional[int] = None  # Added for entropy analysis
    paradox_state: Optional[Dict] = None  # Added for paradox visualization

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
        self.entropy_analyzer = EntropyAnalyzer()
        self.paradox_visualizer = ParadoxVisualizer()
        self.unitizer = AlephUnitizer()
        self.trade_history: List[ReplayTrade] = []
        self.price_history: List[float] = []  # Added for RSI calculation
        logger.info("StrategyReplayer initialized with replay window %d", self.replay_window)
        
    def calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI based on price history."""
        if len(self.price_history) < period:
            return 50.0
        deltas = np.diff(self.price_history[-period:])
        gains = deltas[deltas > 0].sum()
        losses = -deltas[deltas < 0].sum()
        rs = (gains / period) / (losses / period + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return float(np.clip(rsi, 0, 100))
        
    def _calculate_trade_hash(self, trade_id: str, timestamp: datetime) -> str:
        """Generate a unique hash for the trade."""
        data = f"{trade_id}{timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()
        
    def _analyze_entropy(self, order_book: Dict, trades: List[Dict]) -> int:
        """Analyze entropy for the trade using EntropyAnalyzer."""
        try:
            # Convert order book and trades to entropy values
            entropy_values = self.entropy_analyzer.analyze_entropy_distribution(
                [hash(str(item)) % 144 for item in trades]
            )
            return entropy_values.get('mean', 0)
        except Exception as e:
            logger.error(f"Failed to analyze entropy: {e}")
            return 0
            
    def _get_paradox_state(self, price: float, volume: float, smart_money_metrics: SmartMoneyMetrics) -> Dict:
        """Get paradox visualization state for the trade."""
        try:
            market_data = {
                'price': price,
                'volume': volume,
                'rsi': self.calculate_rsi(),  # Use calculated RSI instead of hardcoded value
                'drift': smart_money_metrics.velocity,
                'entropy': smart_money_metrics.smart_money_score
            }
            return self.paradox_visualizer.get_visualization_data()
        except Exception as e:
            logger.error(f"Failed to get paradox state: {e}")
            return {}
        
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
        try:
            # Update price history for RSI calculation
            self.price_history.append(price)
            if len(self.price_history) > self.replay_window:
                self.price_history = self.price_history[-self.replay_window:]
            
            # Get Smart Money metrics for this trade
            smart_money_metrics = self.smart_money.analyze_tick(
                price=price,
                volume=volume,
                order_book=order_book,
                trades=trades
            )
            
            # Calculate trade hash
            trade_hash = self._calculate_trade_hash(trade_id, timestamp)
            
            # Analyze entropy
            entropy_tag = self._analyze_entropy(order_book, trades)
            
            # Get paradox state
            paradox_state = self._get_paradox_state(price, volume, smart_money_metrics)
            
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
                confidence=confidence,
                trade_hash=trade_hash,
                entropy_tag=entropy_tag,
                paradox_state=paradox_state
            )
            
            # Add to history
            self.trade_history.append(replay_trade)
            
            # Keep history limited
            if len(self.trade_history) > self.replay_window:
                self.trade_history = self.trade_history[-self.replay_window:]
                
            logger.info(f"Added trade {trade_id} to replay history")
            
        except Exception as e:
            logger.error(f"Failed to add trade {trade_id}: {e}")
            
    def analyze_trade(self, trade_id: str) -> Dict:
        """
        Analyze a specific trade with Smart Money context.
        Returns detailed analysis of the trade conditions.
        """
        try:
            trade = next((t for t in self.trade_history if t.trade_id == trade_id), None)
            if not trade:
                logger.warning(f"Trade {trade_id} not found")
                return {"error": "Trade not found"}
                
            # Get unitizer analysis
            unitizer_analysis = self.unitizer.analyze_trade({
                'price': trade.price,
                'volume': trade.volume,
                'side': trade.side,
                'profit': trade.profit,
                'pattern_id': trade.pattern_id,
                'confidence': trade.confidence
            })
            
            return {
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp.isoformat(),
                'price': trade.price,
                'volume': trade.volume,
                'side': trade.side,
                'profit': trade.profit,
                'pattern_id': trade.pattern_id,
                'confidence': trade.confidence,
                'trade_hash': trade.trade_hash,
                'entropy_tag': trade.entropy_tag,
                'paradox_state': trade.paradox_state,
                'smart_money_metrics': {
                    'spoof_score': trade.smart_money_metrics.spoof_score,
                    'wall_score': trade.smart_money_metrics.wall_score,
                    'velocity': trade.smart_money_metrics.velocity,
                    'liquidity_resonance': trade.smart_money_metrics.liquidity_resonance,
                    'smart_money_score': trade.smart_money_metrics.smart_money_score
                },
                'success': trade.profit > self.profit_threshold,
                'smart_money_aligned': trade.smart_money_metrics.smart_money_score > self.smart_money_threshold,
                'unitizer_analysis': unitizer_analysis
            }
        except Exception as e:
            logger.error(f"Failed to analyze trade {trade_id}: {e}")
            return {"error": str(e)}
        
    def get_pattern_insights(self, pattern_id: str) -> Dict:
        """
        Get insights about a specific pattern's performance with Smart Money context.
        """
        try:
            pattern_trades = [t for t in self.trade_history if t.pattern_id == pattern_id]
            if not pattern_trades:
                logger.warning(f"No trades found for pattern {pattern_id}")
                return {"error": "No trades found for pattern"}
                
            successful_trades = [t for t in pattern_trades if t.profit > self.profit_threshold]
            smart_money_aligned = [t for t in pattern_trades if t.smart_money_metrics.smart_money_score > self.smart_money_threshold]
            
            # Get entropy analysis for the pattern
            entropy_values = [t.entropy_tag for t in pattern_trades if t.entropy_tag is not None]
            entropy_analysis = self.entropy_analyzer.analyze_entropy_distribution(entropy_values) if entropy_values else {}
            
            # Get paradox visualization for the pattern
            paradox_states = [t.paradox_state for t in pattern_trades if t.paradox_state]
            paradox_analysis = self.paradox_visualizer.calculate_tpf_metrics() if paradox_states else {}
            
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
                },
                'entropy_analysis': entropy_analysis,
                'paradox_analysis': paradox_analysis
            }
        except Exception as e:
            logger.error(f"Failed to get pattern insights for {pattern_id}: {e}")
            return {"error": str(e)}
        
    def get_optimization_suggestions(self) -> Dict:
        """
        Generate suggestions for strategy optimization based on Smart Money analysis.
        """
        try:
            if not self.trade_history:
                logger.warning("No trade history available for optimization suggestions")
                return {"error": "No trade history available"}
                
            successful_trades = [t for t in self.trade_history if t.profit > self.profit_threshold]
            if not successful_trades:
                logger.warning("No successful trades found for optimization suggestions")
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
            
            # Get entropy analysis for successful trades
            entropy_values = [t.entropy_tag for t in successful_trades if t.entropy_tag is not None]
            entropy_analysis = self.entropy_analyzer.analyze_entropy_distribution(entropy_values) if entropy_values else {}
            
            # Get paradox visualization for successful trades
            paradox_states = [t.paradox_state for t in successful_trades if t.paradox_state]
            paradox_analysis = self.paradox_visualizer.calculate_tpf_metrics() if paradox_states else {}
            
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
                
            # Add entropy-based suggestions
            if entropy_analysis:
                suggestions.append({
                    'type': 'entropy_condition',
                    'analysis': entropy_analysis,
                    'reason': 'Entropy analysis suggests optimal trading conditions'
                })
                
            # Add paradox-based suggestions
            if paradox_analysis:
                suggestions.append({
                    'type': 'paradox_condition',
                    'analysis': paradox_analysis,
                    'reason': 'Paradox visualization suggests optimal trading conditions'
                })
                
            return {
                'successful_conditions': successful_conditions,
                'suggestions': suggestions,
                'entropy_analysis': entropy_analysis,
                'paradox_analysis': paradox_analysis
            }
        except Exception as e:
            logger.error(f"Failed to generate optimization suggestions: {e}")
            return {"error": str(e)}
        
    def export_replay_data(self, filepath: str) -> None:
        """
        Export replay data to a JSON file for external analysis.
        """
        try:
            data = {
                'replay_window_size': self.replay_window,
                'profit_threshold': self.profit_threshold,
                'smart_money_threshold': self.smart_money_threshold,
                'generated_at': datetime.now().isoformat(),
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
                        'trade_hash': t.trade_hash,
                        'entropy_tag': t.entropy_tag,
                        'paradox_state': t.paradox_state,
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
            logger.info(f"Successfully exported replay data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export replay data to {filepath}: {e}")
            raise 