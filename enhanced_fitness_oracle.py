"""
Enhanced Fitness Oracle - Central Orchestrator
==============================================

This is the redesigned FitnessOracle that acts as the central nervous system
for Schwabot, connecting all mathematical engines in a proper hierarchy:

ProfitOracle -> RegimeOracle -> RittleGEMM -> FaultBus -> DLT Engine
                    ↓
              FitnessOracle (Central Decision Maker)
                    ↓
              Unified Trading Recommendations
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import yaml
import logging
import asyncio
from collections import deque
from pathlib import Path

# Import all our mathematical engines
from schwabot.ai_oracles.profit_oracle import ProfitOracle, ProfitSignal
from schwabot.ai_oracles.reigime_oracle import RegimeDetector, MarketRegime
from rittle_gemm import RittleGEMM, RingLayer
from core.fault_bus import FaultBus, FaultType, FaultBusEvent
from profit_cycle_navigator import ProfitCycleNavigator, ProfitVector

logger = logging.getLogger(__name__)

@dataclass
class MarketSnapshot:
    """Complete market snapshot from all engines"""
    timestamp: datetime
    
    # Raw Market Data
    price: float
    volume: float
    price_series: List[float]
    volume_series: List[float]
    
    # Engine Outputs
    regime: Optional[MarketRegime] = None
    profit_signals: List[ProfitSignal] = None
    profit_vector: Optional[ProfitVector] = None
    ring_state: Dict = None
    fault_correlations: List = None
    
    # Calculated Metrics
    volatility: float = 0.0
    trend_strength: float = 0.0
    momentum: float = 0.0
    anomaly_strength: float = 0.0

@dataclass
class UnifiedFitnessScore:
    """Unified fitness score with clear actionable recommendations"""
    timestamp: datetime
    
    # Core Scores (all normalized -1 to +1)
    overall_fitness: float
    profit_fitness: float
    regime_fitness: float
    pattern_fitness: float
    risk_fitness: float
    
    # Trading Decision
    action: str  # "BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"
    position_size: float  # 0.0 to 1.0 of available capital
    confidence: float  # 0.0 to 1.0
    
    # Risk Management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_time: Optional[timedelta] = None
    
    # Analysis Details
    dominant_factors: Dict[str, float] = None
    market_regime: str = "unknown"
    profit_tier_detected: bool = False
    loop_warning: bool = False

class EnhancedFitnessOracle:
    """
    Central orchestrator that connects all mathematical engines
    into a unified profit-seeking navigation system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize all mathematical engines in proper hierarchy
        self.fault_bus = FaultBus()
        self.profit_navigator = ProfitCycleNavigator(self.fault_bus)
        self.profit_oracle = ProfitOracle()
        self.regime_detector = RegimeDetector()
        self.rittle_gemm = RittleGEMM(ring_size=1000)
        
        # Central memory and state tracking
        self.market_history = deque(maxlen=1000)
        self.fitness_history = deque(maxlen=500)
        self.engine_performance = {}
        
        # Adaptive weight system
        self.regime_weights = self.config.get("regime_weights", {})
        self.adaptation_rate = self.config.get("adaptation_rate", 0.05)
        
        # Navigation state
        self.current_regime = "unknown"
        self.profit_tier_active = False
        self.loop_detected = False
        
        logger.info("Enhanced FitnessOracle initialized with integrated engine hierarchy")

    def _load_config(self, path: Optional[str]) -> Dict:
        """Load configuration with intelligent defaults"""
        default_config = {
            "regime_weights": {
                "trending": {"profit": 0.4, "momentum": 0.3, "regime": 0.2, "pattern": 0.1},
                "ranging": {"profit": 0.2, "momentum": 0.1, "regime": 0.4, "pattern": 0.3},
                "volatile": {"profit": 0.3, "momentum": 0.2, "regime": 0.3, "pattern": 0.2}
            },
            "fitness_thresholds": {
                "strong_buy": 0.75, "buy": 0.35, "hold": 0.15, 
                "sell": -0.35, "strong_sell": -0.75
            },
            "risk_parameters": {
                "max_position_size": 0.8, "base_stop_loss": 0.02, 
                "base_take_profit": 0.04, "volatility_multiplier": 2.0
            },
            "adaptation_rate": 0.05,
            "min_confidence": 0.6
        }
        
        if path and Path(path).exists():
            try:
                with open(path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Deep merge configs
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                logger.warning(f"Config load failed: {e}. Using defaults.")
        
        return default_config

    async def capture_market_snapshot(self, market_data: Dict[str, Any]) -> MarketSnapshot:
        """
        Capture complete market snapshot using all engines
        This is where the engine hierarchy coordination happens
        """
        timestamp = datetime.now()
        
        # Extract basic market data
        price = market_data.get("price", 0.0)
        volume = market_data.get("volume", 0.0)
        price_series = market_data.get("price_series", [price])
        volume_series = market_data.get("volume_series", [volume])
        
        # STEP 1: REGIME DETECTION (Foundation layer)
        regime = self.regime_detector.detect(price_series, volume_series)
        if regime:
            self.current_regime = regime.regime_id
        
        # STEP 2: PROFIT SIGNAL ANALYSIS (Opportunity layer)
        profit_signals = []
        try:
            # Convert to format expected by profit oracle
            from schwabot.schemas.trade_models import TradeSnapshot
            snapshots = [
                TradeSnapshot(price=p, volume=v, timestamp=timestamp - timedelta(minutes=i))
                for i, (p, v) in enumerate(zip(price_series[-10:], volume_series[-10:]))
            ]
            profit_signal = self.profit_oracle.detect_profit_signal(snapshots)
            if profit_signal:
                profit_signals.append(profit_signal)
        except Exception as e:
            logger.warning(f"Profit oracle failed: {e}")
        
        # STEP 3: PROFIT VECTOR CALCULATION (Navigation layer)
        profit_vector = None
        try:
            profit_vector = self.profit_navigator.update_market_state(
                current_price=price,
                current_volume=volume,
                timestamp=timestamp
            )
        except Exception as e:
            logger.warning(f"Profit navigator failed: {e}")
        
        # STEP 4: RING PATTERN ANALYSIS (Pattern layer)
        ring_state = {}
        try:
            tick_data = {
                "timestamp": int(timestamp.timestamp()),
                "profit": profit_signals[0].projected_gain if profit_signals else 0.0,
                "return": (price_series[-1] - price_series[0]) / price_series[0] if len(price_series) > 1 and price_series[0] != 0 else 0.0,
                "volume": volume,
                "hash_rec": hash(str(market_data)) % 1000 / 1000.0,
                "z_score": 0.0,
                "drift": 0.0,
                "executed": 0,
                "rebuy": 0
            }
            ring_state = self.rittle_gemm.process_tick(tick_data)
        except Exception as e:
            logger.warning(f"RittleGEMM failed: {e}")
        
        # STEP 5: FAULT CORRELATION ANALYSIS (Risk layer)
        fault_correlations = []
        try:
            fault_correlations = self.fault_bus.get_profit_correlations()
        except Exception as e:
            logger.warning(f"Fault correlation failed: {e}")
        
        # STEP 6: CALCULATE TECHNICAL METRICS
        volatility = self._calculate_volatility(price_series)
        trend_strength = self._calculate_trend_strength(price_series)
        momentum = self._calculate_momentum(price_series)
        anomaly_strength = self._detect_anomaly_strength(price_series, profit_signals)
        
        # Create market snapshot
        snapshot = MarketSnapshot(
            timestamp=timestamp,
            price=price,
            volume=volume,
            price_series=price_series,
            volume_series=volume_series,
            regime=regime,
            profit_signals=profit_signals,
            profit_vector=profit_vector,
            ring_state=ring_state,
            fault_correlations=fault_correlations,
            volatility=volatility,
            trend_strength=trend_strength,
            momentum=momentum,
            anomaly_strength=anomaly_strength
        )
        
        # Store in history
        self.market_history.append(snapshot)
        
        return snapshot

    def calculate_unified_fitness(self, snapshot: MarketSnapshot) -> UnifiedFitnessScore:
        """
        Calculate unified fitness score from market snapshot
        This is the core decision-making algorithm
        """
        timestamp = snapshot.timestamp
        
        # COMPONENT FITNESS CALCULATIONS
        
        # 1. PROFIT FITNESS - How strong are the profit opportunities?
        profit_fitness = self._calculate_profit_fitness(snapshot)
        
        # 2. REGIME FITNESS - How well do conditions match the regime?
        regime_fitness = self._calculate_regime_fitness(snapshot)
        
        # 3. PATTERN FITNESS - What do the patterns tell us?
        pattern_fitness = self._calculate_pattern_fitness(snapshot)
        
        # 4. RISK FITNESS - What's the risk-adjusted opportunity?
        risk_fitness = self._calculate_risk_fitness(snapshot)
        
        # ADAPTIVE WEIGHT APPLICATION
        weights = self.regime_weights.get(self.current_regime, self.regime_weights.get("trending", {}))
        
        overall_fitness = (
            profit_fitness * weights.get("profit", 0.3) +
            regime_fitness * weights.get("regime", 0.3) +
            pattern_fitness * weights.get("pattern", 0.2) +
            risk_fitness * weights.get("risk", 0.2)
        )
        
        # Normalize to [-1, 1]
        overall_fitness = np.tanh(overall_fitness)
        
        # GENERATE TRADING DECISION
        action, position_size, confidence = self._generate_trading_decision(
            overall_fitness, snapshot
        )
        
        # CALCULATE RISK MANAGEMENT LEVELS
        stop_loss, take_profit, max_hold_time = self._calculate_risk_levels(
            snapshot, action
        )
        
        # DETECT SPECIAL CONDITIONS
        profit_tier_detected = self._detect_profit_tier(snapshot)
        loop_warning = self._detect_recursive_loop(snapshot)
        
        # Create unified fitness score
        fitness_score = UnifiedFitnessScore(
            timestamp=timestamp,
            overall_fitness=overall_fitness,
            profit_fitness=profit_fitness,
            regime_fitness=regime_fitness,
            pattern_fitness=pattern_fitness,
            risk_fitness=risk_fitness,
            action=action,
            position_size=position_size,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_time=max_hold_time,
            dominant_factors={
                "profit": profit_fitness,
                "regime": regime_fitness,
                "pattern": pattern_fitness,
                "risk": risk_fitness
            },
            market_regime=self.current_regime,
            profit_tier_detected=profit_tier_detected,
            loop_warning=loop_warning
        )
        
        # Store in history
        self.fitness_history.append(fitness_score)
        
        # Adapt weights based on performance
        self._adapt_weights()
        
        return fitness_score

    def _calculate_profit_fitness(self, snapshot: MarketSnapshot) -> float:
        """Calculate profit opportunity fitness"""
        fitness = 0.0
        
        # Profit signals contribution
        if snapshot.profit_signals:
            signal_strength = np.mean([
                s.projected_gain * s.confidence 
                for s in snapshot.profit_signals
            ])
            fitness += signal_strength * 0.4
        
        # Profit vector contribution
        if snapshot.profit_vector:
            vector_strength = snapshot.profit_vector.magnitude * snapshot.profit_vector.confidence
            fitness += vector_strength * 0.4
        
        # Anomaly strength contribution
        fitness += snapshot.anomaly_strength * 0.2
        
        return np.clip(fitness, -1.0, 1.0)

    def _calculate_regime_fitness(self, snapshot: MarketSnapshot) -> float:
        """Calculate how well conditions match the identified regime"""
        if not snapshot.regime:
            return 0.0
        
        regime = snapshot.regime
        
        # Regime-specific logic
        if "trend" in regime.regime_id.lower():
            # Trending regime: strong trend + reasonable volatility
            fitness = abs(regime.trend) * (1.0 - min(regime.volatility, 0.8)) * regime.confidence
        elif "rang" in regime.regime_id.lower():
            # Ranging regime: weak trend + moderate volatility
            fitness = (1.0 - abs(regime.trend)) * min(regime.volatility * 2, 1.0) * regime.confidence
        else:
            # Unknown regime: use confidence only
            fitness = regime.confidence * 0.5
        
        return np.clip(fitness, -1.0, 1.0)

    def _calculate_pattern_fitness(self, snapshot: MarketSnapshot) -> float:
        """Calculate pattern-based fitness from RITTLE-GEMM"""
        if not snapshot.ring_state:
            return 0.0
        
        fitness = 0.0
        
        # Get ring snapshot
        try:
            ring_snapshot = self.rittle_gemm.get_ring_snapshot()
            
            # Key ring indicators
            fitness += ring_snapshot.get('R1', 0.0) * 0.25  # Profit ring
            fitness += ring_snapshot.get('R3', 0.0) * 0.25  # EMA profit
            fitness += ring_snapshot.get('R8', 0.0) * 0.25  # Executed profit
            
            # Pattern triggers
            should_trigger, strategy_id = self.rittle_gemm.check_strategy_trigger()
            if should_trigger:
                fitness += 0.25
                
        except Exception as e:
            logger.warning(f"Pattern fitness calculation failed: {e}")
        
        return np.clip(fitness, -1.0, 1.0)

    def _calculate_risk_fitness(self, snapshot: MarketSnapshot) -> float:
        """Calculate risk-adjusted fitness"""
        base_fitness = 0.5
        
        # Volatility risk (high vol = lower fitness)
        vol_penalty = min(snapshot.volatility * 2, 1.0)
        
        # Volume risk (low volume = lower fitness)
        volume_ratio = snapshot.volume / max(np.mean(snapshot.volume_series), 1.0)
        volume_bonus = min(volume_ratio, 2.0) / 2.0
        
        # Regime confidence (low confidence = lower fitness)
        regime_confidence = snapshot.regime.confidence if snapshot.regime else 0.5
        
        risk_fitness = base_fitness * (1.0 - vol_penalty * 0.5) * volume_bonus * regime_confidence
        
        return np.clip(risk_fitness * 2 - 1, -1.0, 1.0)

    def _generate_trading_decision(self, fitness: float, snapshot: MarketSnapshot) -> Tuple[str, float, float]:
        """Generate actionable trading decision"""
        thresholds = self.config["fitness_thresholds"]
        
        # Determine action based on fitness
        if fitness >= thresholds["strong_buy"]:
            action = "STRONG_BUY"
            base_size = 0.8
        elif fitness >= thresholds["buy"]:
            action = "BUY"
            base_size = 0.5
        elif fitness >= thresholds["hold"]:
            action = "HOLD"
            base_size = 0.0
        elif fitness >= thresholds["sell"]:
            action = "SELL"
            base_size = 0.4
        else:
            action = "STRONG_SELL"
            base_size = 0.8
        
        # Adjust position size based on confidence and volatility
        regime_confidence = snapshot.regime.confidence if snapshot.regime else 0.5
        volatility_adjustment = 1.0 - min(snapshot.volatility * 1.5, 0.8)
        
        position_size = base_size * regime_confidence * volatility_adjustment
        position_size = np.clip(position_size, 0.0, self.config["risk_parameters"]["max_position_size"])
        
        # Overall confidence
        confidence = regime_confidence * volatility_adjustment
        
        return action, position_size, confidence

    def _calculate_risk_levels(self, snapshot: MarketSnapshot, action: str) -> Tuple[Optional[float], Optional[float], Optional[timedelta]]:
        """Calculate stop loss, take profit, and max hold time"""
        if action == "HOLD":
            return None, None, None
        
        risk_params = self.config["risk_parameters"]
        current_price = snapshot.price
        volatility = snapshot.volatility
        
        # Dynamic risk levels based on volatility
        vol_multiplier = 1.0 + volatility * risk_params["volatility_multiplier"]
        stop_loss_pct = risk_params["base_stop_loss"] * vol_multiplier
        take_profit_pct = risk_params["base_take_profit"] * vol_multiplier
        
        if action in ["BUY", "STRONG_BUY"]:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        elif action in ["SELL", "STRONG_SELL"]:
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        else:
            stop_loss = take_profit = None
        
        # Max hold time based on volatility and regime
        if volatility > 0.5:
            max_hold_time = timedelta(hours=1)
        elif self.current_regime == "trending":
            max_hold_time = timedelta(hours=12)
        else:
            max_hold_time = timedelta(hours=4)
        
        return stop_loss, take_profit, max_hold_time

    def _detect_profit_tier(self, snapshot: MarketSnapshot) -> bool:
        """Detect if current conditions represent a genuine profit tier"""
        if len(self.fitness_history) < 20:
            return False
        
        recent_fitness = [f.overall_fitness for f in list(self.fitness_history)[-20:]]
        current_fitness = self.fitness_history[-1].overall_fitness if self.fitness_history else 0.0
        
        # Statistical anomaly detection (JuMBO-style)
        mean_fitness = np.mean(recent_fitness)
        std_fitness = np.std(recent_fitness)
        
        if std_fitness == 0:
            return False
        
        z_score = abs(current_fitness - mean_fitness) / std_fitness
        
        # Check for clustering of high-fitness readings
        high_fitness_count = sum(1 for f in recent_fitness[-5:] if f > mean_fitness + std_fitness)
        
        return z_score > 2.0 and high_fitness_count >= 3

    def _detect_recursive_loop(self, snapshot: MarketSnapshot) -> bool:
        """Detect recursive patterns that might indicate false cycles"""
        if len(self.market_history) < 10:
            return False
        
        # Simple pattern detection based on price and fitness similarity
        recent_snapshots = list(self.market_history)[-10:]
        current_price = snapshot.price
        
        similar_conditions = sum(
            1 for s in recent_snapshots 
            if abs(s.price - current_price) / current_price < 0.01
        )
        
        return similar_conditions >= 5  # Too many similar conditions = potential loop

    def _adapt_weights(self):
        """Adapt regime weights based on recent performance"""
        if len(self.fitness_history) < 50:
            return
        
        # Simple adaptation based on confidence levels
        recent_reports = list(self.fitness_history)[-20:]
        avg_confidence = np.mean([r.confidence for r in recent_reports])
        
        if avg_confidence < self.config["min_confidence"]:
            # Low confidence = need to adjust weights
            # This is a placeholder for more sophisticated adaptation
            logger.info(f"Low confidence detected ({avg_confidence:.3f}), considering weight adaptation")

    # Technical calculation helpers
    def _calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns))

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return float(slope / np.mean(prices))

    def _calculate_momentum(self, prices: List[float]) -> float:
        if len(prices) < 5:
            return 0.0
        short_ma = np.mean(prices[-3:])
        long_ma = np.mean(prices[-5:])
        return float((short_ma - long_ma) / long_ma)

    def _detect_anomaly_strength(self, prices: List[float], profit_signals: List) -> float:
        """Detect anomaly strength in current conditions"""
        if not profit_signals:
            return 0.0
        
        # Simple anomaly detection based on profit signal strength
        signal_strength = np.mean([s.projected_gain for s in profit_signals])
        return min(abs(signal_strength) * 2, 1.0)

    async def run_continuous_fitness_calculation(self, market_data_stream):
        """Run continuous fitness calculation loop"""
        logger.info("Starting continuous fitness calculation")
        
        async for market_data in market_data_stream:
            try:
                # Capture market snapshot
                snapshot = await self.capture_market_snapshot(market_data)
                
                # Calculate unified fitness
                fitness_score = self.calculate_unified_fitness(snapshot)
                
                # Log results
                logger.info(
                    f"Fitness: {fitness_score.overall_fitness:.3f} | "
                    f"Action: {fitness_score.action} | "
                    f"Position: {fitness_score.position_size:.2f} | "
                    f"Confidence: {fitness_score.confidence:.3f} | "
                    f"Regime: {fitness_score.market_regime}"
                )
                
                # Yield fitness score for external consumption
                yield fitness_score
                
            except Exception as e:
                logger.error(f"Fitness calculation failed: {e}")
                await asyncio.sleep(1)

# Example usage
async def demo_enhanced_fitness_oracle():
    """Demonstrate the enhanced fitness oracle"""
    
    # Initialize oracle
    oracle = EnhancedFitnessOracle()
    
    # Simulate market data stream
    async def mock_market_stream():
        for i in range(100):
            yield {
                "price": 100.0 + np.random.normal(0, 2),
                "volume": 1000 + np.random.normal(0, 200),
                "price_series": [100.0 + np.random.normal(0, 2) for _ in range(20)],
                "volume_series": [1000 + np.random.normal(0, 200) for _ in range(20)]
            }
            await asyncio.sleep(0.1)
    
    # Run fitness calculation
    async for fitness_score in oracle.run_continuous_fitness_calculation(mock_market_stream()):
        print(f"Action: {fitness_score.action}, Fitness: {fitness_score.overall_fitness:.3f}")
        
        if fitness_score.profit_tier_detected:
            print("🎯 PROFIT TIER DETECTED!")
        if fitness_score.loop_warning:
            print("⚠️  RECURSIVE LOOP WARNING!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_enhanced_fitness_oracle()) 