"""
Strategy Execution Mapper v1.0
==============================

Bridges pattern recognition from Hash Affinity Vault to actual trade execution.
Integrates with existing vault_router, hash_profit_matrix, and profit_navigator.

Key Integration Points:
- HashAffinityVault profit tier analysis
- VaultRouter for strategy routing  
- ProfitNavigator for position sizing
- GPU/CPU dynamic allocation based on thermal state
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Core system imports
from .hash_affinity_vault import HashAffinityVault, TickSignature

# Optional imports with fallbacks
try:
    from .vault_router import VaultRouter
except ImportError:
    VaultRouter = None

try:
    from .profit_navigator import ProfitNavigator
except ImportError:
    ProfitNavigator = None

try:
    from .thermal_zone_manager import ThermalZoneManager
except ImportError:
    ThermalZoneManager = None

try:
    from .gpu_metrics import GPUMetrics
except ImportError:
    GPUMetrics = None

class StrategyType(Enum):
    """Available strategy types based on Anti-Pole Theory"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    MOMENTUM = "momentum"
    ANTI_POLE = "anti_pole"
    VAULT_LOCK = "vault_lock"

@dataclass
class TradeSignal:
    """Complete trade signal with execution parameters"""
    signal_id: str
    strategy_type: StrategyType
    pair: str
    side: str  # 'buy' or 'sell'
    confidence: float
    volume_fraction: float
    max_slippage: float
    profit_tier: str
    hash_correlation: str
    thermal_budget: float
    execution_urgency: int  # 1-10 scale
    
@dataclass
class ExecutionResult:
    """Result of strategy execution attempt"""
    signal_id: str
    executed: bool
    fill_price: Optional[float]
    fill_quantity: Optional[float]
    execution_time: datetime
    error_message: Optional[str]
    vault_impact: Dict[str, Any]

class StrategyExecutionMapper:
    """
    Advanced strategy execution engine that converts hash-based signals 
    into executable trades with risk management and thermal awareness.
    """
    
    def __init__(self, vault: HashAffinityVault, 
                 vault_router=None,
                 profit_navigator=None,
                 thermal_manager=None):
        """
        Initialize with core system components
        
        Args:
            vault: Hash affinity vault for pattern analysis
            vault_router: Vault routing for strategy selection (optional)
            profit_navigator: Position sizing and profit optimization (optional)
            thermal_manager: GPU/CPU thermal state management (optional)
        """
        self.vault = vault
        
        # Use provided components or create mocks
        self.vault_router = vault_router or MockVaultRouter()
        self.profit_navigator = profit_navigator or MockProfitNavigator()
        self.thermal_manager = thermal_manager or MockThermalZoneManager()
        
        # Strategy configuration
        self.strategy_confidence_thresholds = {
            StrategyType.ACCUMULATION: 0.6,
            StrategyType.DISTRIBUTION: 0.7,
            StrategyType.BREAKOUT: 0.8,
            StrategyType.REVERSAL: 0.75,
            StrategyType.MOMENTUM: 0.65,
            StrategyType.ANTI_POLE: 0.9,
            StrategyType.VAULT_LOCK: 0.95
        }
        
        # Volume allocation limits based on strategy risk
        self.max_volume_fractions = {
            StrategyType.ACCUMULATION: 0.15,
            StrategyType.DISTRIBUTION: 0.12,
            StrategyType.BREAKOUT: 0.08,
            StrategyType.REVERSAL: 0.10,
            StrategyType.MOMENTUM: 0.06,
            StrategyType.ANTI_POLE: 0.20,
            StrategyType.VAULT_LOCK: 0.25
        }
        
        # Execution tracking
        self.active_signals: Dict[str, TradeSignal] = {}
        self.execution_history: List[ExecutionResult] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze current market conditions for strategy selection"""
        
        # Get recent vault analysis
        vault_stats = self.vault.export_comprehensive_state()
        profit_analysis = vault_stats['profit_analysis']
        
        # Get thermal state
        thermal_state = self.thermal_manager.get_thermal_state()
        
        # Get anomalies
        anomalies = self.vault.detect_anomalies()
        
        # Calculate market volatility from recent ticks
        recent_ticks = list(self.vault.recent_ticks)[-20:] if len(self.vault.recent_ticks) >= 20 else list(self.vault.recent_ticks)
        
        if len(recent_ticks) < 3:
            volatility = 0.5  # Neutral default
        else:
            prices = [t.btc_price for t in recent_ticks]
            returns = np.diff(np.log(prices))
            volatility = np.std(returns)
        
        return {
            'volatility': volatility,
            'profit_tier_success_rate': profit_analysis.get('success_rate', 0.5),
            'thermal_budget': 1.0 - thermal_state.get('gpu_utilization', 0.0),
            'anomaly_count': len(anomalies),
            'most_profitable_backend': profit_analysis.get('most_profitable_backend', 'CPU_STANDARD'),
            'recent_profit_correlation': max([
                perf.get('profit_correlation', 0.0) 
                for perf in vault_stats['backend_performance'].values()
            ]) if vault_stats['backend_performance'] else 0.0
        }
    
    def determine_strategy_type(self, tick_signature: TickSignature, 
                              market_conditions: Dict[str, Any]) -> Optional[StrategyType]:
        """Determine optimal strategy type based on tick signature and market conditions"""
        
        # Extract key metrics
        signal_strength = tick_signature.signal_strength
        profit_tier = tick_signature.profit_tier
        correlation_score = tick_signature.correlation_score
        volatility = market_conditions['volatility']
        anomaly_count = market_conditions['anomaly_count']
        
        # Strategy selection logic based on Anti-Pole Theory
        if anomaly_count > 3:
            # High anomaly environment - use anti-pole strategies
            if signal_strength > 0.9:
                return StrategyType.ANTI_POLE
            else:
                return None  # Too risky
        
        if profit_tier == 'PLATINUM':
            if volatility < 0.02:  # Low volatility
                return StrategyType.ACCUMULATION
            elif volatility > 0.08:  # High volatility
                return StrategyType.VAULT_LOCK
            else:
                return StrategyType.MOMENTUM
        
        elif profit_tier == 'GOLD':
            if signal_strength > 0.8:
                return StrategyType.BREAKOUT if volatility > 0.05 else StrategyType.MOMENTUM
            else:
                return StrategyType.ACCUMULATION
        
        elif profit_tier == 'SILVER':
            if correlation_score > 0.7:
                return StrategyType.REVERSAL
            else:
                return StrategyType.DISTRIBUTION
        
        else:  # BRONZE or NEUTRAL
            if signal_strength > 0.6:
                return StrategyType.DISTRIBUTION
            else:
                return None  # No strategy recommended
    
    def calculate_position_sizing(self, strategy_type: StrategyType, 
                                confidence: float, 
                                market_conditions: Dict[str, Any]) -> float:
        """Calculate optimal position size based on strategy and conditions"""
        
        # Base allocation from strategy limits
        base_fraction = self.max_volume_fractions[strategy_type]
        
        # Adjust for confidence
        confidence_multiplier = min(confidence * 1.5, 1.0)
        
        # Adjust for thermal budget
        thermal_multiplier = market_conditions['thermal_budget']
        
        # Adjust for volatility (higher volatility = smaller positions)
        volatility_multiplier = max(0.3, 1.0 - market_conditions['volatility'] * 5)
        
        # Calculate final fraction
        volume_fraction = (
            base_fraction * 
            confidence_multiplier * 
            thermal_multiplier * 
            volatility_multiplier
        )
        
        return min(volume_fraction, 0.25)  # Never exceed 25% of capital
    
    async def generate_trade_signal(self, tick_signature: TickSignature) -> Optional[TradeSignal]:
        """Generate executable trade signal from tick signature"""
        
        # Analyze current market conditions
        market_conditions = await self.analyze_market_conditions()
        
        # Determine strategy type
        strategy_type = self.determine_strategy_type(tick_signature, market_conditions)
        if not strategy_type:
            self.logger.debug(f"No strategy recommended for tick {tick_signature.tick_id}")
            return None
        
        # Check confidence threshold
        required_confidence = self.strategy_confidence_thresholds[strategy_type]
        if tick_signature.correlation_score < required_confidence:
            self.logger.debug(f"Confidence {tick_signature.correlation_score} below threshold {required_confidence}")
            return None
        
        # Calculate position sizing
        volume_fraction = self.calculate_position_sizing(
            strategy_type, tick_signature.correlation_score, market_conditions
        )
        
        # Determine trade side based on strategy and signal
        side = self._determine_trade_side(strategy_type, tick_signature, market_conditions)
        
        # Calculate execution parameters
        max_slippage = self._calculate_max_slippage(strategy_type, market_conditions['volatility'])
        execution_urgency = self._calculate_urgency(strategy_type, market_conditions)
        
        return TradeSignal(
            signal_id=f"{strategy_type.value}_{tick_signature.tick_id}_{datetime.now().strftime('%H%M%S')}",
            strategy_type=strategy_type,
            pair="BTC/USDC",  # Primary trading pair
            side=side,
            confidence=tick_signature.correlation_score,
            volume_fraction=volume_fraction,
            max_slippage=max_slippage,
            profit_tier=tick_signature.profit_tier,
            hash_correlation=tick_signature.sha256_hash[:8],
            thermal_budget=market_conditions['thermal_budget'],
            execution_urgency=execution_urgency
        )
    
    def _determine_trade_side(self, strategy_type: StrategyType, 
                            tick_signature: TickSignature, 
                            market_conditions: Dict[str, Any]) -> str:
        """Determine buy/sell side based on strategy logic"""
        
        signal_strength = tick_signature.signal_strength
        volatility = market_conditions['volatility']
        
        if strategy_type == StrategyType.ACCUMULATION:
            return 'buy'  # Always accumulate on good signals
        
        elif strategy_type == StrategyType.DISTRIBUTION:
            return 'sell'  # Distribute positions
        
        elif strategy_type == StrategyType.BREAKOUT:
            # Buy on strong upward signals, sell on weak ones
            return 'buy' if signal_strength > 0.8 else 'sell'
        
        elif strategy_type == StrategyType.REVERSAL:
            # Contrarian approach
            return 'sell' if signal_strength > 0.7 else 'buy'
        
        elif strategy_type == StrategyType.MOMENTUM:
            # Follow the signal direction
            return 'buy' if signal_strength > 0.6 else 'sell'
        
        elif strategy_type == StrategyType.ANTI_POLE:
            # Anti-pole logic - prevent value sinks
            return 'buy' if volatility < 0.05 else 'sell'
        
        elif strategy_type == StrategyType.VAULT_LOCK:
            # Secure high-value positions
            return 'buy'
        
        return 'buy'  # Default fallback
    
    def _calculate_max_slippage(self, strategy_type: StrategyType, volatility: float) -> float:
        """Calculate maximum acceptable slippage"""
        base_slippage = {
            StrategyType.ACCUMULATION: 0.003,
            StrategyType.DISTRIBUTION: 0.002,
            StrategyType.BREAKOUT: 0.005,
            StrategyType.REVERSAL: 0.004,
            StrategyType.MOMENTUM: 0.006,
            StrategyType.ANTI_POLE: 0.001,
            StrategyType.VAULT_LOCK: 0.002
        }
        
        # Adjust for volatility
        volatility_adjustment = min(volatility * 2, 0.003)
        
        return base_slippage[strategy_type] + volatility_adjustment
    
    def _calculate_urgency(self, strategy_type: StrategyType, 
                          market_conditions: Dict[str, Any]) -> int:
        """Calculate execution urgency (1-10 scale)"""
        
        base_urgency = {
            StrategyType.ACCUMULATION: 3,
            StrategyType.DISTRIBUTION: 4,
            StrategyType.BREAKOUT: 8,
            StrategyType.REVERSAL: 6,
            StrategyType.MOMENTUM: 7,
            StrategyType.ANTI_POLE: 9,
            StrategyType.VAULT_LOCK: 10
        }
        
        urgency = base_urgency[strategy_type]
        
        # Increase urgency for high anomaly environments
        if market_conditions['anomaly_count'] > 2:
            urgency += 2
        
        # Decrease urgency if thermal budget is low
        if market_conditions['thermal_budget'] < 0.3:
            urgency -= 1
        
        return max(1, min(10, urgency))
    
    async def execute_signal(self, signal: TradeSignal) -> ExecutionResult:
        """Execute a trade signal and return results"""
        
        start_time = datetime.utcnow()
        self.logger.info(f"Executing signal {signal.signal_id}: {signal.strategy_type.value} {signal.side} {signal.volume_fraction:.3f}")
        
        try:
            # Store active signal
            self.active_signals[signal.signal_id] = signal
            
            # Route through vault router for strategy execution
            vault_route = await self.vault_router.route_strategy({
                'strategy_type': signal.strategy_type.value,
                'confidence': signal.confidence,
                'volume_fraction': signal.volume_fraction,
                'pair': signal.pair,
                'side': signal.side
            })
            
            # Calculate position size through profit navigator
            position_size = await self.profit_navigator.calculate_position_size(
                signal.volume_fraction, 
                signal.confidence
            )
            
            # Simulate execution (replace with real exchange integration)
            execution_result = await self._simulate_execution(signal, position_size)
            
            # Update vault with execution result
            await self._update_vault_memory(signal, execution_result)
            
            # Store execution history
            self.execution_history.append(execution_result)
            
            # Clean up active signal
            if signal.signal_id in self.active_signals:
                del self.active_signals[signal.signal_id]
            
            self.logger.info(f"Signal {signal.signal_id} executed successfully")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Failed to execute signal {signal.signal_id}: {e}")
            
            error_result = ExecutionResult(
                signal_id=signal.signal_id,
                executed=False,
                fill_price=None,
                fill_quantity=None,
                execution_time=start_time,
                error_message=str(e),
                vault_impact={}
            )
            
            self.execution_history.append(error_result)
            return error_result
    
    async def _simulate_execution(self, signal: TradeSignal, position_size: float) -> ExecutionResult:
        """Simulate trade execution (replace with real exchange API)"""
        
        # Simulate market price with slippage
        base_price = 45000.0  # BTC/USDC base price
        slippage = np.random.uniform(-signal.max_slippage, signal.max_slippage)
        fill_price = base_price * (1 + slippage)
        
        # Simulate partial or full fill based on urgency
        fill_ratio = min(1.0, 0.5 + (signal.execution_urgency / 20.0))
        fill_quantity = position_size * fill_ratio
        
        return ExecutionResult(
            signal_id=signal.signal_id,
            executed=True,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            execution_time=datetime.utcnow(),
            error_message=None,
            vault_impact={
                'strategy_type': signal.strategy_type.value,
                'profit_tier': signal.profit_tier,
                'hash_correlation': signal.hash_correlation
            }
        )
    
    async def _update_vault_memory(self, signal: TradeSignal, result: ExecutionResult):
        """Update vault memory with execution results"""
        
        if result.executed:
            # Log successful execution to vault
            self.vault.log_tick(
                tick_id=f"exec_{signal.signal_id}",
                signal_strength=signal.confidence,
                backend="EXECUTION_ENGINE",
                matrix_id=signal.hash_correlation,
                btc_price=result.fill_price,
                volume=result.fill_quantity,
                profit_tier=signal.profit_tier
            )
            
            self.logger.debug(f"Updated vault memory for successful execution {signal.signal_id}")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        
        if not self.execution_history:
            return {'error': 'No execution history available'}
        
        successful_executions = [r for r in self.execution_history if r.executed]
        failed_executions = [r for r in self.execution_history if not r.executed]
        
        # Strategy performance analysis
        strategy_performance = {}
        for result in successful_executions:
            signal = next((s for s in self.active_signals.values() 
                         if s.signal_id == result.signal_id), None)
            if signal:
                strategy = signal.strategy_type.value
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(result.fill_quantity)
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'failed_executions': len(failed_executions),
            'success_rate': len(successful_executions) / len(self.execution_history),
            'strategy_performance': {
                strategy: {
                    'count': len(quantities),
                    'avg_size': np.mean(quantities),
                    'total_volume': np.sum(quantities)
                }
                for strategy, quantities in strategy_performance.items()
            },
            'active_signals': len(self.active_signals)
        }

# Mock classes for missing dependencies
class MockVaultRouter:
    async def route_strategy(self, params):
        return {'status': 'routed', 'params': params}

class MockProfitNavigator:
    async def calculate_position_size(self, volume_fraction, confidence):
        return volume_fraction * 10000.0  # Mock $10k position

class MockThermalZoneManager:
    def get_thermal_state(self):
        return {'gpu_utilization': 0.3, 'cpu_utilization': 0.2, 'gpu_temperature': 65, 'cpu_temperature': 55} 