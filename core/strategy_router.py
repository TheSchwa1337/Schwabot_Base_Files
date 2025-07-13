#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Router Module
======================
Provides strategy routing functionality for the Schwabot trading system.

Mathematical Core:
R(s_i) = argmax_j {w_j * f_j(s_i) + λ_j * g_j(s_i)}
Where:
- w_j: strategy weights
- f_j: strategy performance functions
- λ_j: risk adjustment factors
- g_j: market condition functions

This module intelligently routes trading signals to optimal execution strategies
based on mathematical optimization and market conditions.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import mathematical infrastructure
try:
    from core.unified_mathematical_bridge import UnifiedMathematicalBridge
    from core.unified_mathematical_integration_methods import UnifiedMathematicalIntegrationMethods
    from core.unified_mathematical_performance_monitor import UnifiedMathematicalPerformanceMonitor
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Mathematical infrastructure not available - using fallback")


class RoutingStrategy(Enum):
    """Routing strategy types."""
    PERFORMANCE_BASED = "performance_based"
    RISK_ADJUSTED = "risk_adjusted"
    MARKET_CONDITION = "market_condition"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class MarketCondition(Enum):
    """Market condition types."""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CALM = "calm"
    CRISIS = "crisis"


class SignalPriority(Enum):
    """Signal priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RoutingDecision:
    """Routing decision with mathematical analysis."""
    signal_id: str
    selected_strategy: str
    routing_score: float
    confidence: float
    reasoning: str
    mathematical_analysis: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    routing_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Strategy performance metrics for routing."""
    strategy_name: str
    total_signals: int
    successful_routes: int
    average_score: float
    win_rate: float
    risk_score: float
    mathematical_signature: str = ""


@dataclass
class MarketConditionData:
    """Market condition data for routing."""
    condition: MarketCondition
    volatility: float
    trend_strength: float
    liquidity_score: float
    mathematical_signature: str = ""


@dataclass
class StrategyRouterConfig:
    """Configuration for strategy router."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    max_concurrent_routes: int = 20
    routing_threshold: float = 0.6  # Minimum routing score
    mathematical_analysis_enabled: bool = True
    adaptive_routing_enabled: bool = True
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 0.25,
        'mean_reversion': 0.20,
        'scalping': 0.15,
        'arbitrage': 0.15,
        'grid': 0.10,
        'quantum': 0.10,
        'phantom': 0.05
    })
    risk_factors: Dict[str, float] = field(default_factory=lambda: {
        'volatility_penalty': 0.1,
        'liquidity_penalty': 0.05,
        'trend_alignment_bonus': 0.15
    })


class StrategyRouter:
    """
    Strategy Router System
    
    Implements intelligent signal routing:
    R(s_i) = argmax_j {w_j * f_j(s_i) + λ_j * g_j(s_i)}
    
    Intelligently routes trading signals to optimal execution strategies
    based on mathematical optimization and market conditions.
    """
    
    def __init__(self, config: Optional[StrategyRouterConfig] = None):
        """Initialize the strategy router system."""
        self.config = config or StrategyRouterConfig()
        self.logger = logging.getLogger(__name__)
        
        # Routing state
        self.active_routes: Dict[str, RoutingDecision] = {}
        self.routing_history: List[RoutingDecision] = []
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.market_conditions: Dict[str, MarketConditionData] = {}
        
        # Signal processing
        self.signal_queue: asyncio.Queue = asyncio.Queue()
        self.routing_queue: asyncio.Queue = asyncio.Queue()
        
        # Mathematical infrastructure
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_bridge = UnifiedMathematicalBridge()
            self.math_integration = UnifiedMathematicalIntegrationMethods()
            self.math_monitor = UnifiedMathematicalPerformanceMonitor()
        else:
            self.math_bridge = None
            self.math_integration = None
            self.math_monitor = None
        
        # Performance tracking
        self.performance_metrics = {
            'signals_routed': 0,
            'successful_routes': 0,
            'routing_errors': 0,
            'average_routing_time': 0.0,
            'routing_accuracy': 0.0
        }
        
        # System state
        self.initialized = False
        self.active = False
        
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the strategy router system."""
        try:
            self.logger.info("Initializing Strategy Router System")
            
            # Initialize strategy performance tracking
            for strategy_name in self.config.strategy_weights.keys():
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    total_signals=0,
                    successful_routes=0,
                    average_score=0.0,
                    win_rate=0.0,
                    risk_score=0.0
                )
            
            self.initialized = True
            self.logger.info("✅ Strategy Router System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing Strategy Router System: {e}")
            self.initialized = False
    
    async def start_router(self) -> bool:
        """Start the strategy router."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            
            # Start processing tasks
            asyncio.create_task(self._process_signal_queue())
            asyncio.create_task(self._process_routing_queue())
            
            self.logger.info("✅ Strategy Router started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting strategy router: {e}")
            return False
    
    async def stop_router(self) -> bool:
        """Stop the strategy router."""
        try:
            self.active = False
            self.logger.info("✅ Strategy Router stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping strategy router: {e}")
            return False
    
    async def route_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Route a trading signal to optimal strategy."""
        if not self.active:
            self.logger.error("Strategy router not active")
            return False
        
        try:
            # Validate signal data
            if not self._validate_signal_data(signal_data):
                self.logger.error(f"Invalid signal data: {signal_data}")
                return False
            
            # Add mathematical analysis
            if self.config.mathematical_analysis_enabled:
                await self._analyze_signal_mathematically(signal_data)
            
            # Queue for processing
            await self.signal_queue.put(signal_data)
            
            self.logger.info(f"✅ Signal queued for routing: {signal_data.get('signal_id', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error routing signal: {e}")
            return False
    
    def _validate_signal_data(self, signal_data: Dict[str, Any]) -> bool:
        """Validate signal data for routing."""
        try:
            required_fields = ['signal_id', 'symbol', 'signal_type', 'confidence']
            
            for field in required_fields:
                if field not in signal_data:
                    return False
            
            # Check confidence range
            confidence = signal_data.get('confidence', 0.0)
            if confidence < 0.0 or confidence > 1.0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating signal data: {e}")
            return False
    
    async def _analyze_signal_mathematically(self, signal_data: Dict[str, Any]) -> None:
        """Perform mathematical analysis on signal."""
        try:
            if not self.math_bridge:
                return
            
            # Prepare signal data for mathematical analysis
            analysis_data = {
                'signal_id': signal_data.get('signal_id'),
                'symbol': signal_data.get('symbol'),
                'signal_type': signal_data.get('signal_type'),
                'confidence': signal_data.get('confidence'),
                'timestamp': time.time(),
                'metadata': signal_data.get('metadata', {})
            }
            
            # Perform mathematical integration
            result = self.math_bridge.integrate_all_mathematical_systems(
                analysis_data, {}
            )
            
            # Update signal data with mathematical analysis
            signal_data['mathematical_analysis'] = {
                'confidence': result.overall_confidence,
                'connections': len(result.connections),
                'performance_metrics': result.performance_metrics,
                'mathematical_signature': result.mathematical_signature
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing signal mathematically: {e}")
    
    async def _process_signal_queue(self) -> None:
        """Process signals from the queue."""
        try:
            while self.active:
                try:
                    # Get signal from queue
                    signal_data = await asyncio.wait_for(
                        self.signal_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process signal
                    await self._process_signal(signal_data)
                    
                    # Mark task as done
                    self.signal_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Error processing signal: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error in signal processing loop: {e}")
    
    async def _process_signal(self, signal_data: Dict[str, Any]) -> None:
        """Process a signal for routing."""
        try:
            start_time = time.time()
            
            # Update performance metrics
            self.performance_metrics['signals_routed'] += 1
            
            # Make routing decision
            decision = await self._make_routing_decision(signal_data)
            
            # Store decision
            self.routing_history.append(decision)
            
            # Update strategy performance
            self._update_strategy_performance(decision)
            
            # Queue for execution
            await self.routing_queue.put(decision)
            
            # Update performance metrics
            routing_time = time.time() - start_time
            self.performance_metrics['average_routing_time'] = (
                (self.performance_metrics['average_routing_time'] * (self.performance_metrics['signals_routed'] - 1) + routing_time) / 
                self.performance_metrics['signals_routed']
            )
            
            self.logger.info(f"✅ Signal routed: {signal_data.get('signal_id')} -> {decision.selected_strategy}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing signal: {e}")
            self.performance_metrics['routing_errors'] += 1
    
    async def _make_routing_decision(self, signal_data: Dict[str, Any]) -> RoutingDecision:
        """Make routing decision based on mathematical optimization."""
        try:
            signal_id = signal_data.get('signal_id', 'unknown')
            symbol = signal_data.get('symbol', '')
            
            # Get market conditions
            market_condition = self._get_market_condition(symbol)
            
            # Calculate routing scores for each strategy
            routing_scores = {}
            for strategy_name in self.config.strategy_weights.keys():
                score = await self._calculate_routing_score(
                    signal_data, strategy_name, market_condition
                )
                routing_scores[strategy_name] = score
            
            # Select optimal strategy
            selected_strategy = max(routing_scores.items(), key=lambda x: x[1])[0]
            routing_score = routing_scores[selected_strategy]
            
            # Determine if routing should proceed
            should_route = routing_score >= self.config.routing_threshold
            
            if not should_route:
                selected_strategy = "hold"
                routing_score = 0.0
            
            # Generate routing parameters
            routing_parameters = self._generate_routing_parameters(
                signal_data, selected_strategy, market_condition
            )
            
            # Perform mathematical analysis on decision
            mathematical_analysis = await self._analyze_decision_mathematically(
                signal_data, selected_strategy, routing_score, market_condition
            )
            
            # Create reasoning
            reasoning = self._generate_routing_reasoning(
                signal_data, selected_strategy, routing_score, market_condition
            )
            
            return RoutingDecision(
                signal_id=signal_id,
                selected_strategy=selected_strategy,
                routing_score=routing_score,
                confidence=signal_data.get('confidence', 0.0),
                reasoning=reasoning,
                mathematical_analysis=mathematical_analysis,
                routing_parameters=routing_parameters
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error making routing decision: {e}")
            return RoutingDecision(
                signal_id=signal_data.get('signal_id', 'unknown'),
                selected_strategy="hold",
                routing_score=0.0,
                confidence=0.0,
                reasoning=f"Error in routing decision: {e}",
                routing_parameters={}
            )
    
    def _get_market_condition(self, symbol: str) -> MarketConditionData:
        """Get current market condition for symbol."""
        try:
            # Use cached condition or create default
            if symbol in self.market_conditions:
                return self.market_conditions[symbol]
            
            # Create default market condition
            default_condition = MarketConditionData(
                condition=MarketCondition.SIDEWAYS,
                volatility=0.02,  # 2% volatility
                trend_strength=0.0,
                liquidity_score=0.8
            )
            
            self.market_conditions[symbol] = default_condition
            return default_condition
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market condition: {e}")
            return MarketConditionData(
                condition=MarketCondition.SIDEWAYS,
                volatility=0.02,
                trend_strength=0.0,
                liquidity_score=0.8
            )
    
    async def _calculate_routing_score(self, signal_data: Dict[str, Any], 
                                     strategy_name: str, 
                                     market_condition: MarketConditionData) -> float:
        """Calculate routing score for a strategy using mathematical optimization."""
        try:
            # Get strategy weight
            strategy_weight = self.config.strategy_weights.get(strategy_name, 0.1)
            
            # Get strategy performance
            performance = self.strategy_performance.get(strategy_name)
            if not performance:
                return 0.0
            
            # Calculate performance function f_j(s_i)
            performance_score = self._calculate_performance_function(
                signal_data, performance, market_condition
            )
            
            # Calculate market condition function g_j(s_i)
            market_score = self._calculate_market_condition_function(
                signal_data, strategy_name, market_condition
            )
            
            # Calculate risk adjustment factor λ_j
            risk_factor = self._calculate_risk_factor(
                signal_data, strategy_name, market_condition
            )
            
            # Apply mathematical optimization formula: R(s_i) = argmax_j {w_j * f_j(s_i) + λ_j * g_j(s_i)}
            routing_score = (strategy_weight * performance_score + 
                           risk_factor * market_score)
            
            return max(0.0, min(1.0, routing_score))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating routing score: {e}")
            return 0.0
    
    def _calculate_performance_function(self, signal_data: Dict[str, Any], 
                                      performance: StrategyPerformance, 
                                      market_condition: MarketConditionData) -> float:
        """Calculate performance function f_j(s_i)."""
        try:
            # Base performance score
            base_score = performance.win_rate * performance.average_score
            
            # Adjust for signal confidence
            confidence = signal_data.get('confidence', 0.5)
            confidence_adjustment = confidence * 0.3
            
            # Adjust for market condition alignment
            market_alignment = self._calculate_market_alignment(
                signal_data, market_condition
            )
            
            # Combine scores
            performance_score = (base_score * 0.5 + 
                               confidence_adjustment * 0.3 + 
                               market_alignment * 0.2)
            
            return max(0.0, min(1.0, performance_score))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance function: {e}")
            return 0.0
    
    def _calculate_market_condition_function(self, signal_data: Dict[str, Any], 
                                           strategy_name: str, 
                                           market_condition: MarketConditionData) -> float:
        """Calculate market condition function g_j(s_i)."""
        try:
            # Base market score
            market_score = 0.5
            
            # Adjust based on strategy type and market condition
            if strategy_name == 'momentum':
                if market_condition.condition in [MarketCondition.BULL_TRENDING, MarketCondition.BEAR_TRENDING]:
                    market_score += 0.3
                elif market_condition.condition == MarketCondition.SIDEWAYS:
                    market_score -= 0.2
                    
            elif strategy_name == 'mean_reversion':
                if market_condition.condition == MarketCondition.SIDEWAYS:
                    market_score += 0.3
                elif market_condition.condition in [MarketCondition.BULL_TRENDING, MarketCondition.BEAR_TRENDING]:
                    market_score -= 0.2
                    
            elif strategy_name == 'scalping':
                if market_condition.condition == MarketCondition.VOLATILE:
                    market_score += 0.3
                elif market_condition.condition == MarketCondition.CALM:
                    market_score -= 0.2
                    
            elif strategy_name == 'arbitrage':
                if market_condition.liquidity_score > 0.8:
                    market_score += 0.3
                else:
                    market_score -= 0.2
            
            # Adjust for volatility
            volatility_penalty = market_condition.volatility * self.config.risk_factors['volatility_penalty']
            market_score -= volatility_penalty
            
            # Adjust for liquidity
            liquidity_bonus = market_condition.liquidity_score * 0.2
            market_score += liquidity_bonus
            
            return max(0.0, min(1.0, market_score))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating market condition function: {e}")
            return 0.0
    
    def _calculate_risk_factor(self, signal_data: Dict[str, Any], 
                             strategy_name: str, 
                             market_condition: MarketConditionData) -> float:
        """Calculate risk adjustment factor λ_j."""
        try:
            # Base risk factor
            risk_factor = 0.5
            
            # Adjust based on strategy risk profile
            strategy_risk_profiles = {
                'momentum': 0.6,
                'mean_reversion': 0.4,
                'scalping': 0.8,
                'arbitrage': 0.3,
                'grid': 0.5,
                'quantum': 0.7,
                'phantom': 0.9
            }
            
            risk_factor = strategy_risk_profiles.get(strategy_name, 0.5)
            
            # Adjust for market volatility
            volatility_adjustment = market_condition.volatility * 0.5
            risk_factor += volatility_adjustment
            
            # Adjust for signal confidence
            confidence = signal_data.get('confidence', 0.5)
            confidence_adjustment = (1.0 - confidence) * 0.3
            risk_factor += confidence_adjustment
            
            return max(0.0, min(1.0, risk_factor))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk factor: {e}")
            return 0.5
    
    def _calculate_market_alignment(self, signal_data: Dict[str, Any], 
                                  market_condition: MarketConditionData) -> float:
        """Calculate market alignment score."""
        try:
            # Simple market alignment based on trend strength
            trend_alignment = market_condition.trend_strength * self.config.risk_factors['trend_alignment_bonus']
            
            # Adjust for volatility
            volatility_alignment = 1.0 - market_condition.volatility
            
            # Combine alignments
            alignment_score = (trend_alignment + volatility_alignment) / 2.0
            
            return max(0.0, min(1.0, alignment_score))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating market alignment: {e}")
            return 0.5
    
    def _generate_routing_parameters(self, signal_data: Dict[str, Any], 
                                   selected_strategy: str, 
                                   market_condition: MarketConditionData) -> Dict[str, Any]:
        """Generate routing parameters."""
        try:
            base_params = {
                'strategy': selected_strategy,
                'symbol': signal_data.get('symbol'),
                'signal_type': signal_data.get('signal_type'),
                'confidence': signal_data.get('confidence'),
                'market_condition': market_condition.condition.value,
                'volatility': market_condition.volatility,
                'liquidity_score': market_condition.liquidity_score
            }
            
            # Add strategy-specific parameters
            if selected_strategy == 'momentum':
                base_params.update({
                    'trend_following': True,
                    'stop_loss': 0.02,
                    'take_profit': 0.04
                })
            elif selected_strategy == 'mean_reversion':
                base_params.update({
                    'reversion_threshold': 0.03,
                    'position_sizing': 'conservative'
                })
            elif selected_strategy == 'scalping':
                base_params.update({
                    'timeout': 300,  # 5 minutes
                    'position_size': 'small',
                    'high_frequency': True
                })
            elif selected_strategy == 'arbitrage':
                base_params.update({
                    'min_spread': 0.001,
                    'execution_speed': 'ultra_fast'
                })
            
            return base_params
            
        except Exception as e:
            self.logger.error(f"❌ Error generating routing parameters: {e}")
            return {'error': str(e)}
    
    async def _analyze_decision_mathematically(self, signal_data: Dict[str, Any], 
                                             selected_strategy: str, 
                                             routing_score: float, 
                                             market_condition: MarketConditionData) -> Dict[str, Any]:
        """Perform mathematical analysis on routing decision."""
        try:
            if not self.math_bridge:
                return {}
            
            # Prepare decision data for mathematical analysis
            decision_data = {
                'signal_id': signal_data.get('signal_id'),
                'selected_strategy': selected_strategy,
                'routing_score': routing_score,
                'market_condition': market_condition.condition.value,
                'volatility': market_condition.volatility,
                'trend_strength': market_condition.trend_strength,
                'liquidity_score': market_condition.liquidity_score,
                'mathematical_signature': signal_data.get('mathematical_analysis', {}).get('mathematical_signature', '')
            }
            
            # Perform mathematical integration
            result = self.math_bridge.integrate_all_mathematical_systems(
                decision_data, {}
            )
            
            return {
                'confidence': result.overall_confidence,
                'connections': len(result.connections),
                'performance_metrics': result.performance_metrics,
                'mathematical_signature': result.mathematical_signature
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing decision mathematically: {e}")
            return {}
    
    def _generate_routing_reasoning(self, signal_data: Dict[str, Any], 
                                  selected_strategy: str, 
                                  routing_score: float, 
                                  market_condition: MarketConditionData) -> str:
        """Generate human-readable routing reasoning."""
        try:
            reasoning_parts = []
            
            # Strategy selection reason
            if selected_strategy == "hold":
                reasoning_parts.append(f"Routing score {routing_score:.3f} below threshold {self.config.routing_threshold}")
            else:
                reasoning_parts.append(f"Selected {selected_strategy} strategy with routing score {routing_score:.3f}")
            
            # Market condition context
            reasoning_parts.append(f"Market condition: {market_condition.condition.value}")
            reasoning_parts.append(f"Volatility: {market_condition.volatility:.3f}")
            
            # Signal confidence
            confidence = signal_data.get('confidence', 0.0)
            reasoning_parts.append(f"Signal confidence: {confidence:.3f}")
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating routing reasoning: {e}")
            return f"Error generating reasoning: {e}"
    
    def _update_strategy_performance(self, decision: RoutingDecision) -> None:
        """Update strategy performance metrics."""
        try:
            strategy_name = decision.selected_strategy
            if strategy_name not in self.strategy_performance:
                return
            
            performance = self.strategy_performance[strategy_name]
            
            # Update metrics
            performance.total_signals += 1
            performance.average_score = (
                (performance.average_score * (performance.total_signals - 1) + decision.routing_score) / 
                performance.total_signals
            )
            
            # Update mathematical signature
            performance.mathematical_signature = decision.mathematical_analysis.get('mathematical_signature', '')
            
            # Note: win rate and successful routes would be updated after execution results
            
        except Exception as e:
            self.logger.error(f"❌ Error updating strategy performance: {e}")
    
    async def _process_routing_queue(self) -> None:
        """Process routing decisions from the queue."""
        try:
            while self.active:
                try:
                    # Get decision from queue
                    decision = await asyncio.wait_for(
                        self.routing_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process decision (send to execution engine)
                    await self._execute_routing_decision(decision)
                    
                    # Mark task as done
                    self.routing_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Error processing routing decision: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error in routing processing loop: {e}")
    
    async def _execute_routing_decision(self, decision: RoutingDecision) -> None:
        """Execute a routing decision (send to execution engine)."""
        try:
            # Update performance metrics
            self.performance_metrics['successful_routes'] += 1
            
            # Log execution
            self.logger.info(f"🚀 Executing routing decision: {decision.signal_id} -> {decision.selected_strategy}")
            
            # Here you would send the decision to the execution engine
            # For now, we'll just log it
            execution_data = {
                'decision_id': decision.signal_id,
                'selected_strategy': decision.selected_strategy,
                'routing_score': decision.routing_score,
                'confidence': decision.confidence,
                'parameters': decision.routing_parameters,
                'timestamp': decision.timestamp
            }
            
            self.logger.info(f"Routing execution data: {json.dumps(execution_data, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error executing routing decision: {e}")
    
    def update_market_condition(self, symbol: str, condition_data: Dict[str, Any]) -> bool:
        """Update market condition for a symbol."""
        try:
            condition = MarketCondition(condition_data.get('condition', 'sideways'))
            
            market_condition = MarketConditionData(
                condition=condition,
                volatility=condition_data.get('volatility', 0.02),
                trend_strength=condition_data.get('trend_strength', 0.0),
                liquidity_score=condition_data.get('liquidity_score', 0.8),
                mathematical_signature=condition_data.get('mathematical_signature', '')
            )
            
            self.market_conditions[symbol] = market_condition
            
            self.logger.info(f"✅ Updated market condition for {symbol}: {condition.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating market condition: {e}")
            return False
    
    def get_strategy_performance(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        try:
            if strategy_name:
                performance = self.strategy_performance.get(strategy_name)
                if not performance:
                    return {}
                
                return {
                    'strategy_name': performance.strategy_name,
                    'total_signals': performance.total_signals,
                    'successful_routes': performance.successful_routes,
                    'average_score': performance.average_score,
                    'win_rate': performance.win_rate,
                    'risk_score': performance.risk_score,
                    'mathematical_signature': performance.mathematical_signature
                }
            else:
                return {
                    name: {
                        'total_signals': perf.total_signals,
                        'successful_routes': perf.successful_routes,
                        'average_score': perf.average_score,
                        'win_rate': perf.win_rate,
                        'risk_score': perf.risk_score
                    }
                    for name, perf in self.strategy_performance.items()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error getting strategy performance: {e}")
            return {}
    
    def get_recent_routing_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent routing decisions."""
        try:
            recent_decisions = self.routing_history[-limit:]
            return [
                {
                    'signal_id': decision.signal_id,
                    'selected_strategy': decision.selected_strategy,
                    'routing_score': decision.routing_score,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning,
                    'timestamp': decision.timestamp,
                    'routing_parameters': decision.routing_parameters
                }
                for decision in recent_decisions
            ]
        except Exception as e:
            self.logger.error(f"❌ Error getting recent routing decisions: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Calculate routing accuracy
        total_routes = metrics['signals_routed']
        if total_routes > 0:
            metrics['routing_accuracy'] = metrics['successful_routes'] / total_routes
        else:
            metrics['routing_accuracy'] = 0.0
        
        return metrics
    
    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.logger.info("✅ Strategy Router System activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating Strategy Router System: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info("✅ Strategy Router System deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating Strategy Router System: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'signals_queued': self.signal_queue.qsize(),
            'routing_queued': self.routing_queue.qsize(),
            'active_routes': len(self.active_routes),
            'total_routing_decisions': len(self.routing_history),
            'symbols_tracked': len(self.market_conditions),
            'performance_metrics': self.performance_metrics,
            'config': {
                'enabled': self.config.enabled,
                'max_concurrent_routes': self.config.max_concurrent_routes,
                'routing_threshold': self.config.routing_threshold,
                'mathematical_analysis_enabled': self.config.mathematical_analysis_enabled,
                'adaptive_routing_enabled': self.config.adaptive_routing_enabled
            }
        }


def create_strategy_router(config: Optional[StrategyRouterConfig] = None) -> StrategyRouter:
    """Factory function to create StrategyRouter instance."""
    return StrategyRouter(config)


async def main():
    """Main function for testing."""
    # Create configuration
    config = StrategyRouterConfig(
        enabled=True,
        debug=True,
        max_concurrent_routes=10,
        routing_threshold=0.6,
        mathematical_analysis_enabled=True,
        adaptive_routing_enabled=True
    )
    
    # Create router
    router = create_strategy_router(config)
    
    # Activate system
    router.activate()
    
    # Start router
    await router.start_router()
    
    # Update market conditions
    router.update_market_condition("BTCUSDT", {
        'condition': 'bull_trending',
        'volatility': 0.025,
        'trend_strength': 0.7,
        'liquidity_score': 0.9
    })
    
    # Submit test signals
    test_signals = [
        {
            'signal_id': 'signal_001',
            'symbol': 'BTCUSDT',
            'signal_type': 'buy',
            'confidence': 0.85,
            'metadata': {'price': 50000.0}
        },
        {
            'signal_id': 'signal_002',
            'symbol': 'ETHUSDT',
            'signal_type': 'sell',
            'confidence': 0.75,
            'metadata': {'price': 3000.0}
        },
        {
            'signal_id': 'signal_003',
            'symbol': 'BTCUSDT',
            'signal_type': 'hold',
            'confidence': 0.4,
            'metadata': {'price': 50100.0}
        }
    ]
    
    # Route signals
    for signal_data in test_signals:
        await router.route_signal(signal_data)
    
    # Wait for processing
    await asyncio.sleep(5)
    
    # Get status
    status = router.get_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    
    # Get strategy performance
    performance = router.get_strategy_performance()
    print(f"Strategy Performance: {json.dumps(performance, indent=2)}")
    
    # Get recent routing decisions
    decisions = router.get_recent_routing_decisions()
    print(f"Recent Routing Decisions: {json.dumps(decisions, indent=2)}")
    
    # Stop router
    await router.stop_router()
    
    # Deactivate system
    router.deactivate()


if __name__ == "__main__":
    asyncio.run(main())
