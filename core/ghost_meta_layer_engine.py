"""
Ghost Meta-Layer Weighting Engine
==================================

Dynamically adjusts signal layer weights based on market conditions, entropy analysis,
and historical performance patterns. Implements sophisticated weighting algorithms
that adapt ghost signal processing to current market microstructure.

Mathematical Framework:
- Entropy-based weight adjustment using Shannon and Renyi entropy
- Volatility-aware dynamic scaling
- Historical performance correlation weighting
- Market microstructure adaptation algorithms
"""

import numpy as np
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import yaml

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_RANGING = "sideways_ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT_PENDING = "breakout_pending"
    NEWS_DRIVEN = "news_driven"
    THIN_LIQUIDITY = "thin_liquidity"

class MetaWeightingStrategy(Enum):
    """Meta-layer weighting strategies"""
    ENTROPY_ADAPTIVE = "entropy_adaptive"
    PERFORMANCE_BASED = "performance_based"
    VOLATILITY_SCALED = "volatility_scaled"
    REGIME_DEPENDENT = "regime_dependent"
    HYBRID_DYNAMIC = "hybrid_dynamic"

@dataclass
class MarketCondition:
    """Current market condition assessment"""
    regime: MarketRegime
    volatility_percentile: float      # 0-100 percentile of recent volatility
    entropy_score: float              # Shannon entropy of recent price moves
    liquidity_depth: float            # Order book depth metric
    spoofing_intensity: float         # Smart money spoofing activity level
    momentum_strength: float          # Directional momentum measure
    timestamp: float = field(default_factory=time.time)

@dataclass
class LayerWeights:
    """Signal layer weight configuration"""
    geometric: float = 0.25
    smart_money: float = 0.35
    depth: float = 0.20
    timeband: float = 0.20
    
    def normalize(self) -> 'LayerWeights':
        """Normalize weights to sum to 1.0"""
        total = self.geometric + self.smart_money + self.depth + self.timeband
        if total > 0:
            return LayerWeights(
                geometric=self.geometric / total,
                smart_money=self.smart_money / total,
                depth=self.depth / total,
                timeband=self.timeband / total
            )
        return LayerWeights()
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {
            'geometric': self.geometric,
            'smart_money': self.smart_money,
            'depth': self.depth,
            'timeband': self.timeband
        }

class GhostMetaLayerEngine:
    """Dynamic meta-layer weighting engine for ghost signals"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.default_weights = LayerWeights()
        self.current_weights = LayerWeights()
        self.weight_history: List[Tuple[float, LayerWeights]] = []
        self.market_conditions: List[MarketCondition] = []
        self.performance_tracker: Dict[str, List[float]] = {
            'geometric': [],
            'smart_money': [],
            'depth': [],
            'timeband': []
        }
        self.entropy_analyzer = EntropyAnalyzer()
        self.volatility_tracker = VolatilityTracker()
        self.regime_detector = MarketRegimeDetector()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load meta-layer configuration"""
        default_config = {
            'weighting_strategy': MetaWeightingStrategy.HYBRID_DYNAMIC.value,
            'adaptation_speed': 0.1,
            'min_history_length': 100,
            'entropy_window_size': 50,
            'volatility_lookback_hours': 24,
            'weight_bounds': {
                'geometric': {'min': 0.10, 'max': 0.50},
                'smart_money': {'min': 0.15, 'max': 0.60},
                'depth': {'min': 0.05, 'max': 0.40},
                'timeband': {'min': 0.05, 'max': 0.40}
            },
            'regime_mappings': {
                MarketRegime.BULL_TRENDING.value: {
                    'geometric': 0.35, 'smart_money': 0.25, 'depth': 0.20, 'timeband': 0.20
                },
                MarketRegime.BEAR_TRENDING.value: {
                    'geometric': 0.30, 'smart_money': 0.40, 'depth': 0.15, 'timeband': 0.15
                },
                MarketRegime.SIDEWAYS_RANGING.value: {
                    'geometric': 0.20, 'smart_money': 0.30, 'depth': 0.30, 'timeband': 0.20
                },
                MarketRegime.HIGH_VOLATILITY.value: {
                    'geometric': 0.15, 'smart_money': 0.50, 'depth': 0.25, 'timeband': 0.10
                },
                MarketRegime.THIN_LIQUIDITY.value: {
                    'geometric': 0.25, 'smart_money': 0.25, 'depth': 0.40, 'timeband': 0.10
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def update_market_conditions(self, price_data: List[float], 
                                volume_data: List[float],
                                order_book_data: Optional[Dict[str, Any]] = None,
                                smart_money_metrics: Optional[Dict[str, float]] = None) -> MarketCondition:
        """Update current market condition assessment"""
        
        # Calculate market metrics
        volatility = self.volatility_tracker.calculate_volatility(price_data)
        entropy = self.entropy_analyzer.calculate_entropy(price_data)
        momentum = self._calculate_momentum(price_data)
        
        # Assess liquidity depth
        liquidity_depth = self._assess_liquidity_depth(order_book_data) if order_book_data else 0.5
        
        # Extract spoofing intensity
        spoofing_intensity = smart_money_metrics.get('spoofing_intensity', 0.0) if smart_money_metrics else 0.0
        
        # Detect market regime
        regime = self.regime_detector.detect_regime(price_data, volume_data, volatility, entropy)
        
        # Calculate volatility percentile
        vol_percentile = self.volatility_tracker.get_volatility_percentile(volatility)
        
        condition = MarketCondition(
            regime=regime,
            volatility_percentile=vol_percentile,
            entropy_score=entropy,
            liquidity_depth=liquidity_depth,
            spoofing_intensity=spoofing_intensity,
            momentum_strength=momentum
        )
        
        self.market_conditions.append(condition)
        
        # Maintain reasonable history size
        if len(self.market_conditions) > 1000:
            self.market_conditions = self.market_conditions[-500:]
        
        return condition
    
    def calculate_dynamic_weights(self, market_condition: Optional[MarketCondition] = None) -> LayerWeights:
        """Calculate dynamic weights based on current market conditions"""
        
        if market_condition is None and self.market_conditions:
            market_condition = self.market_conditions[-1]
        elif market_condition is None:
            return self.default_weights
        
        strategy = MetaWeightingStrategy(self.config['weighting_strategy'])
        
        if strategy == MetaWeightingStrategy.ENTROPY_ADAPTIVE:
            weights = self._calculate_entropy_adaptive_weights(market_condition)
        elif strategy == MetaWeightingStrategy.PERFORMANCE_BASED:
            weights = self._calculate_performance_based_weights(market_condition)
        elif strategy == MetaWeightingStrategy.VOLATILITY_SCALED:
            weights = self._calculate_volatility_scaled_weights(market_condition)
        elif strategy == MetaWeightingStrategy.REGIME_DEPENDENT:
            weights = self._calculate_regime_dependent_weights(market_condition)
        else:  # HYBRID_DYNAMIC
            weights = self._calculate_hybrid_dynamic_weights(market_condition)
        
        # Apply bounds and normalization
        weights = self._apply_weight_bounds(weights)
        weights = weights.normalize()
        
        # Update current weights with adaptation smoothing
        adaptation_speed = self.config['adaptation_speed']
        if self.current_weights:
            self.current_weights = LayerWeights(
                geometric=self.current_weights.geometric * (1 - adaptation_speed) + weights.geometric * adaptation_speed,
                smart_money=self.current_weights.smart_money * (1 - adaptation_speed) + weights.smart_money * adaptation_speed,
                depth=self.current_weights.depth * (1 - adaptation_speed) + weights.depth * adaptation_speed,
                timeband=self.current_weights.timeband * (1 - adaptation_speed) + weights.timeband * adaptation_speed
            )
        else:
            self.current_weights = weights
        
        # Store in history
        self.weight_history.append((time.time(), self.current_weights))
        
        return self.current_weights
    
    def _calculate_entropy_adaptive_weights(self, condition: MarketCondition) -> LayerWeights:
        """Calculate weights based on entropy analysis"""
        
        base_weights = LayerWeights()
        entropy_factor = condition.entropy_score
        
        # High entropy favors smart money and depth analysis
        if entropy_factor > 0.8:
            return LayerWeights(
                geometric=0.15,
                smart_money=0.45,
                depth=0.30,
                timeband=0.10
            )
        elif entropy_factor > 0.6:
            return LayerWeights(
                geometric=0.20,
                smart_money=0.40,
                depth=0.25,
                timeband=0.15
            )
        else:
            # Low entropy favors geometric patterns
            return LayerWeights(
                geometric=0.40,
                smart_money=0.25,
                depth=0.20,
                timeband=0.15
            )
    
    def _calculate_performance_based_weights(self, condition: MarketCondition) -> LayerWeights:
        """Calculate weights based on historical layer performance"""
        
        if not any(self.performance_tracker.values()):
            return self.default_weights
        
        # Calculate recent performance for each layer
        layer_performance = {}
        for layer, performance_history in self.performance_tracker.items():
            if performance_history:
                # Use recent performance with exponential decay weighting
                recent_performance = performance_history[-20:]  # Last 20 trades
                weights = np.exp(np.linspace(-1, 0, len(recent_performance)))
                weighted_performance = np.average(recent_performance, weights=weights)
                layer_performance[layer] = max(0.01, weighted_performance)
            else:
                layer_performance[layer] = 0.25  # Default
        
        # Convert performance to weights (softmax normalization)
        performance_values = np.array(list(layer_performance.values()))
        performance_exp = np.exp(performance_values - np.max(performance_values))
        performance_weights = performance_exp / np.sum(performance_exp)
        
        return LayerWeights(
            geometric=performance_weights[0],
            smart_money=performance_weights[1],
            depth=performance_weights[2],
            timeband=performance_weights[3]
        )
    
    def _calculate_volatility_scaled_weights(self, condition: MarketCondition) -> LayerWeights:
        """Calculate weights scaled by volatility regime"""
        
        vol_percentile = condition.volatility_percentile
        
        if vol_percentile > 80:  # High volatility
            # Favor smart money detection and depth analysis
            return LayerWeights(
                geometric=0.15,
                smart_money=0.50,
                depth=0.25,
                timeband=0.10
            )
        elif vol_percentile > 60:  # Medium-high volatility
            return LayerWeights(
                geometric=0.20,
                smart_money=0.40,
                depth=0.25,
                timeband=0.15
            )
        elif vol_percentile > 40:  # Medium volatility
            return LayerWeights(
                geometric=0.25,
                smart_money=0.35,
                depth=0.20,
                timeband=0.20
            )
        else:  # Low volatility
            # Favor geometric patterns and timeband analysis
            return LayerWeights(
                geometric=0.35,
                smart_money=0.25,
                depth=0.15,
                timeband=0.25
            )
    
    def _calculate_regime_dependent_weights(self, condition: MarketCondition) -> LayerWeights:
        """Calculate weights based on detected market regime"""
        
        regime_mapping = self.config['regime_mappings'].get(
            condition.regime.value, 
            self.default_weights.to_dict()
        )
        
        return LayerWeights(**regime_mapping)
    
    def _calculate_hybrid_dynamic_weights(self, condition: MarketCondition) -> LayerWeights:
        """Calculate weights using hybrid approach combining multiple strategies"""
        
        # Get weights from different strategies
        entropy_weights = self._calculate_entropy_adaptive_weights(condition)
        volatility_weights = self._calculate_volatility_scaled_weights(condition)
        regime_weights = self._calculate_regime_dependent_weights(condition)
        
        # Performance weights if available
        if any(self.performance_tracker.values()):
            performance_weights = self._calculate_performance_based_weights(condition)
        else:
            performance_weights = self.default_weights
        
        # Combine with weighted average based on market conditions
        # High entropy/volatility → more weight to entropy/volatility strategies
        entropy_factor = min(1.0, condition.entropy_score * 2)
        volatility_factor = min(1.0, condition.volatility_percentile / 50.0)
        regime_factor = 0.3  # Constant baseline
        performance_factor = max(0.1, 1.0 - entropy_factor - volatility_factor)
        
        # Normalize factors
        total_factor = entropy_factor + volatility_factor + regime_factor + performance_factor
        if total_factor > 0:
            entropy_factor /= total_factor
            volatility_factor /= total_factor
            regime_factor /= total_factor
            performance_factor /= total_factor
        
        # Weighted combination
        hybrid_weights = LayerWeights(
            geometric=(
                entropy_weights.geometric * entropy_factor +
                volatility_weights.geometric * volatility_factor +
                regime_weights.geometric * regime_factor +
                performance_weights.geometric * performance_factor
            ),
            smart_money=(
                entropy_weights.smart_money * entropy_factor +
                volatility_weights.smart_money * volatility_factor +
                regime_weights.smart_money * regime_factor +
                performance_weights.smart_money * performance_factor
            ),
            depth=(
                entropy_weights.depth * entropy_factor +
                volatility_weights.depth * volatility_factor +
                regime_weights.depth * regime_factor +
                performance_weights.depth * performance_factor
            ),
            timeband=(
                entropy_weights.timeband * entropy_factor +
                volatility_weights.timeband * volatility_factor +
                regime_weights.timeband * regime_factor +
                performance_weights.timeband * performance_factor
            )
        )
        
        return hybrid_weights
    
    def _apply_weight_bounds(self, weights: LayerWeights) -> LayerWeights:
        """Apply configured bounds to layer weights"""
        
        bounds = self.config['weight_bounds']
        
        return LayerWeights(
            geometric=np.clip(weights.geometric, 
                            bounds['geometric']['min'], 
                            bounds['geometric']['max']),
            smart_money=np.clip(weights.smart_money, 
                              bounds['smart_money']['min'], 
                              bounds['smart_money']['max']),
            depth=np.clip(weights.depth, 
                        bounds['depth']['min'], 
                        bounds['depth']['max']),
            timeband=np.clip(weights.timeband, 
                           bounds['timeband']['min'], 
                           bounds['timeband']['max'])
        )
    
    def update_layer_performance(self, layer: str, performance_score: float) -> None:
        """Update performance tracking for a specific layer"""
        
        if layer in self.performance_tracker:
            self.performance_tracker[layer].append(performance_score)
            
            # Keep reasonable history size
            if len(self.performance_tracker[layer]) > 200:
                self.performance_tracker[layer] = self.performance_tracker[layer][-100:]
    
    def get_weight_adaptation_metrics(self) -> Dict[str, Any]:
        """Get metrics about weight adaptation behavior"""
        
        if len(self.weight_history) < 2:
            return {}
        
        # Calculate weight stability
        recent_weights = [w for _, w in self.weight_history[-20:]]
        weight_arrays = {
            'geometric': [w.geometric for w in recent_weights],
            'smart_money': [w.smart_money for w in recent_weights],
            'depth': [w.depth for w in recent_weights],
            'timeband': [w.timeband for w in recent_weights]
        }
        
        stability_metrics = {
            layer: np.std(values) for layer, values in weight_arrays.items()
        }
        
        # Calculate adaptation frequency
        adaptation_changes = 0
        for i in range(1, len(recent_weights)):
            prev_weights = recent_weights[i-1]
            curr_weights = recent_weights[i]
            
            change_magnitude = (
                abs(curr_weights.geometric - prev_weights.geometric) +
                abs(curr_weights.smart_money - prev_weights.smart_money) +
                abs(curr_weights.depth - prev_weights.depth) +
                abs(curr_weights.timeband - prev_weights.timeband)
            )
            
            if change_magnitude > 0.05:  # Significant change threshold
                adaptation_changes += 1
        
        adaptation_frequency = adaptation_changes / len(recent_weights) if recent_weights else 0
        
        return {
            'weight_stability': stability_metrics,
            'adaptation_frequency': adaptation_frequency,
            'current_weights': self.current_weights.to_dict(),
            'weight_history_length': len(self.weight_history),
            'last_market_regime': self.market_conditions[-1].regime.value if self.market_conditions else None
        }
    
    def _calculate_momentum(self, price_data: List[float]) -> float:
        """Calculate momentum strength from price data"""
        if len(price_data) < 10:
            return 0.0
        
        # Simple momentum calculation using price differences
        recent_prices = price_data[-10:]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Normalize to [0, 1] range
        return min(1.0, max(0.0, abs(momentum) * 100))
    
    def _assess_liquidity_depth(self, order_book_data: Dict[str, Any]) -> float:
        """Assess liquidity depth from order book data"""
        try:
            bids = order_book_data.get('bids', [])
            asks = order_book_data.get('asks', [])
            
            if not bids or not asks:
                return 0.5
            
            # Calculate depth within 1% of mid price
            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
            depth_range = mid_price * 0.01
            
            bid_depth = sum(float(qty) for price, qty in bids 
                          if float(price) >= mid_price - depth_range)
            ask_depth = sum(float(qty) for price, qty in asks 
                          if float(price) <= mid_price + depth_range)
            
            total_depth = bid_depth + ask_depth
            
            # Normalize to [0, 1] range (assuming 100 BTC is very high depth)
            return min(1.0, total_depth / 100.0)
            
        except (KeyError, ValueError, IndexError):
            return 0.5  # Default moderate liquidity

class EntropyAnalyzer:
    """Analyzes entropy patterns in market data"""
    
    def calculate_entropy(self, data: List[float], window_size: int = 50) -> float:
        """Calculate Shannon entropy of price movements"""
        if len(data) < window_size:
            return 0.5
        
        # Calculate price returns
        recent_data = data[-window_size:]
        returns = [recent_data[i] / recent_data[i-1] - 1 for i in range(1, len(recent_data))]
        
        if not returns:
            return 0.5
        
        # Discretize returns into bins
        bins = 20
        hist, _ = np.histogram(returns, bins=bins)
        hist = hist + 1e-10  # Avoid log(0)
        
        # Calculate probabilities
        probs = hist / np.sum(hist)
        
        # Calculate Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize to [0, 1] range
        max_entropy = np.log2(bins)
        return min(1.0, max(0.0, entropy / max_entropy))

class VolatilityTracker:
    """Tracks and analyzes volatility patterns"""
    
    def __init__(self):
        self.volatility_history: List[Tuple[float, float]] = []  # (timestamp, volatility)
    
    def calculate_volatility(self, price_data: List[float], window_size: int = 24) -> float:
        """Calculate rolling volatility"""
        if len(price_data) < window_size:
            return 0.0
        
        recent_prices = price_data[-window_size:]
        returns = [recent_prices[i] / recent_prices[i-1] - 1 for i in range(1, len(recent_prices))]
        
        if not returns:
            return 0.0
        
        volatility = np.std(returns) * np.sqrt(len(returns))  # Annualized volatility
        
        # Store in history
        self.volatility_history.append((time.time(), volatility))
        
        # Maintain reasonable history size
        if len(self.volatility_history) > 1000:
            self.volatility_history = self.volatility_history[-500:]
        
        return volatility
    
    def get_volatility_percentile(self, current_volatility: float) -> float:
        """Get percentile rank of current volatility"""
        if len(self.volatility_history) < 10:
            return 50.0  # Default to median
        
        historical_vols = [vol for _, vol in self.volatility_history]
        percentile = (sum(1 for vol in historical_vols if vol <= current_volatility) / 
                     len(historical_vols)) * 100
        
        return percentile

class MarketRegimeDetector:
    """Detects current market regime"""
    
    def detect_regime(self, price_data: List[float], volume_data: List[float], 
                     volatility: float, entropy: float) -> MarketRegime:
        """Detect current market regime based on multiple factors"""
        
        if len(price_data) < 50:
            return MarketRegime.SIDEWAYS_RANGING
        
        # Calculate trend strength
        recent_prices = price_data[-20:]
        trend_strength = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Calculate volatility level
        vol_threshold_high = 0.05
        vol_threshold_low = 0.01
        
        # Detect regime based on multiple factors
        if volatility > vol_threshold_high:
            if entropy > 0.8:
                return MarketRegime.NEWS_DRIVEN
            else:
                return MarketRegime.HIGH_VOLATILITY
        
        elif volatility < vol_threshold_low:
            return MarketRegime.LOW_VOLATILITY
        
        elif abs(trend_strength) > 0.1:
            if trend_strength > 0:
                return MarketRegime.BULL_TRENDING
            else:
                return MarketRegime.BEAR_TRENDING
        
        elif entropy > 0.7 and volatility > 0.03:
            return MarketRegime.BREAKOUT_PENDING
        
        else:
            return MarketRegime.SIDEWAYS_RANGING 