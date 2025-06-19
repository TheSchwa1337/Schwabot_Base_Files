"""
Profit Routing Engine
====================

Implements Step 4: Comprehensive profit maximization strategies that integrate
with your unified mathematical framework, phase gate system, and sustainment principles.

This connects:
- SustainmentMathLib profit optimization
- PhaseGateController execution routing  
- CCXTExecutionManager trade execution
- Mathematical validation from all previous steps
- Enhanced with Windows CLI compatibility for cross-platform reliability

Mathematical Foundation:
π(t) = ∑ᵢ Wᵢ · Pᵢ(t) · SI(t) · G(t) · R(t)

Where:
- π(t): Total profit function
- Wᵢ: Dynamic routing weights  
- Pᵢ(t): Profit vectors from phase gates
- SI(t): Sustainment index
- G(t): Graded profit vector
- R(t): Risk adjustment factor
"""

import asyncio
import logging
import time
import platform
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import threading
from collections import deque, defaultdict

# Import your existing mathematical systems
from .mathlib_v3 import SustainmentMathLib, GradedProfitVector, SustainmentVector, MathematicalContext
from .phase_gate_controller import PhaseGateController, MathematicalTradeSignal, PhaseGateDecision
from .ccxt_execution_manager import CCXTExecutionManager, ExecutionResult
from .bit_operations import BitOperations, PhaseState
from ..enhanced_fitness_oracle import EnhancedFitnessOracle, UnifiedFitnessScore

logger = logging.getLogger(__name__)

# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """
    Handles Windows CLI compatibility issues including emoji rendering
    and ASIC implementation for plain text output explanations
    
    Addresses the CLI error issues mentioned in the comprehensive testing:
    - Emoji characters causing encoding errors on Windows
    - Need for ASIC plain text output
    - Cross-platform compatibility for error messages
    """
    
    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return (platform.system() == "Windows" and 
                ("cmd" in os.environ.get("COMSPEC", "").lower() or
                 "powershell" in os.environ.get("PSModulePath", "").lower()))
    
    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """
        Print message safely with Windows CLI compatibility
        Implements ASIC plain text output for Windows environments
        
        ASIC Implementation: Application-Specific Integrated Circuit approach
        provides specialized text rendering for Windows CLI environments
        """
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            # ASIC plain text markers for Windows CLI compatibility
            emoji_to_asic_mapping = {
                '✅': '[SUCCESS]',    # Success indicator
                '❌': '[ERROR]',      # Error indicator  
                '🔧': '[PROCESSING]', # Processing indicator
                '🚀': '[LAUNCH]',     # Launch/start indicator
                '🎉': '[COMPLETE]',   # Completion indicator
                '💥': '[CRITICAL]',   # Critical alert
                '⚡': '[FAST]',       # Fast execution
                '🔍': '[SEARCH]',     # Search/analysis
                '📊': '[DATA]',       # Data processing
                '🧪': '[TEST]',       # Testing indicator
                '🛠️': '[TOOLS]',      # Tools/utilities
                '⚖️': '[BALANCE]',    # Balance/measurement
                '🔄': '[CYCLE]',      # Cycle/loop
                '🎯': '[TARGET]',     # Target/goal
                '📈': '[PROFIT]',     # Profit indicator
                '🔥': '[HOT]',        # High activity
                '❄️': '[COOL]',       # Cool/low activity
                '⭐': '[STAR]',       # Important/featured
            }
            
            safe_message = message
            for emoji, asic_replacement in emoji_to_asic_mapping.items():
                safe_message = safe_message.replace(emoji, asic_replacement)
            
            return safe_message
        
        return message
    
    @staticmethod
    def log_safe(logger, level: str, message: str):
        """Log message safely with Windows CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            # Emergency ASCII fallback for Windows CLI
            ascii_message = safe_message.encode('ascii', errors='replace').decode('ascii')
            getattr(logger, level.lower())(ascii_message)
    
    @staticmethod
    def safe_format_error(error: Exception, context: str = "") -> str:
        """Format error messages safely for Windows CLI"""
        error_message = f"Error: {str(error)}"
        if context:
            error_message += f" | Context: {context}"
        
        return WindowsCliCompatibilityHandler.safe_print(error_message)

class ProfitRouteType(Enum):
    """Types of profit routing strategies"""
    MICRO_SCALP = "micro_scalp"           # 4b fast micro profits
    HARMONIC_SWING = "harmonic_swing"     # 8b harmonic profit waves
    STRATEGIC_HOLD = "strategic_hold"     # 42b strategic positions
    DIVERSIFIED_BLEND = "diversified_blend" # Multi-route optimization

class ProfitOptimizationMode(Enum):
    """Profit optimization modes"""
    MAXIMIZE_TOTAL = "maximize_total"     # Maximize absolute profit
    MAXIMIZE_RATIO = "maximize_ratio"     # Maximize profit/risk ratio
    MAXIMIZE_SUSTAINED = "maximize_sustained" # Maximize sustained profit
    MAXIMIZE_VELOCITY = "maximize_velocity"   # Maximize profit velocity

class RoutePerformanceLevel(Enum):
    """Performance levels for profit routes"""
    FAILING = "failing"        # Route is losing money
    MARGINAL = "marginal"      # Route is barely profitable
    PERFORMING = "performing"  # Route is meeting targets
    EXCELLENT = "excellent"    # Route is exceeding targets

@dataclass
class ProfitRoute:
    """Individual profit route configuration"""
    route_id: str
    route_type: ProfitRouteType
    phase_gate: str  # 4b, 8b, 42b
    
    # Mathematical parameters
    weight: float = 1.0
    target_profit_rate: float = 0.05  # 5% target
    max_risk_ratio: float = 0.02      # 2% max risk
    sustainment_requirement: float = 0.65  # Minimum SI
    
    # Performance tracking
    total_profit: float = 0.0
    total_trades: int = 0
    success_rate: float = 0.0
    average_return: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Current state
    active: bool = True
    performance_level: RoutePerformanceLevel = RoutePerformanceLevel.PERFORMING
    last_execution: Optional[datetime] = None
    
    # Mathematical validation
    graded_vectors: List[GradedProfitVector] = field(default_factory=list)
    sustainment_history: List[float] = field(default_factory=list)

@dataclass
class ProfitRoutingDecision:
    """Decision from profit routing analysis"""
    selected_routes: List[str]
    route_allocations: Dict[str, float]  # Route ID -> allocation percentage
    total_position_size: float
    expected_profit: float
    risk_assessment: float
    
    # Mathematical validation
    sustainment_index: float
    mathematical_validity: bool
    profit_vector: GradedProfitVector
    
    # Execution details
    execution_priority: float
    routing_confidence: float
    timing_recommendation: str  # immediate, delayed, queued
    
    # Supporting data
    reasoning: str = ""
    alternative_routes: List[str] = field(default_factory=list)

@dataclass
class RoutingPerformanceMetrics:
    """Performance metrics for the routing system"""
    total_profit: float = 0.0
    total_volume_traded: float = 0.0
    average_profit_per_trade: float = 0.0
    profit_velocity: float = 0.0  # Profit per time unit
    
    # Route-specific metrics
    route_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Mathematical metrics
    average_sustainment_index: float = 0.0
    mathematical_validation_rate: float = 0.0
    profit_vector_consistency: float = 0.0
    
    # Risk metrics
    total_risk_taken: float = 0.0
    profit_to_risk_ratio: float = 0.0
    maximum_drawdown: float = 0.0

class ProfitRoutingEngine:
    """
    Comprehensive profit routing engine that optimizes profit maximization
    across multiple mathematical validated strategies and phase gates.
    """
    
    def __init__(self,
                 sustainment_lib: Optional[SustainmentMathLib] = None,
                 phase_controller: Optional[PhaseGateController] = None,
                 execution_manager: Optional[CCXTExecutionManager] = None,
                 fitness_oracle: Optional[EnhancedFitnessOracle] = None,
                 bit_operations: Optional[BitOperations] = None):
        """
        Initialize profit routing engine with mathematical foundations
        
        Args:
            sustainment_lib: Your existing SustainmentMathLib
            phase_controller: Phase gate controller from Step 3
            execution_manager: CCXT execution manager from Step 2
            fitness_oracle: Your existing EnhancedFitnessOracle
            bit_operations: Your existing BitOperations
        """
        # Windows CLI compatibility handler
        self.cli_handler = WindowsCliCompatibilityHandler()
        
        # Initialize mathematical systems
        self.sustainment_lib = sustainment_lib or SustainmentMathLib()
        self.phase_controller = phase_controller
        self.execution_manager = execution_manager
        self.fitness_oracle = fitness_oracle or EnhancedFitnessOracle()
        self.bit_operations = bit_operations or BitOperations()
        
        # Initialize profit routes
        self.profit_routes = self._initialize_profit_routes()
        
        # Routing configuration
        self.optimization_mode = ProfitOptimizationMode.MAXIMIZE_SUSTAINED
        self.rebalance_frequency = timedelta(minutes=5)  # Rebalance every 5 minutes
        self.route_evaluation_window = 100  # Trades to evaluate performance
        
        # Mathematical parameters
        self.min_sustainment_index = 0.65
        self.max_total_risk = 0.10  # 10% maximum total risk
        self.profit_velocity_weight = 0.3
        self.consistency_weight = 0.4
        self.growth_weight = 0.3
        
        # Performance tracking
        self.routing_history = deque(maxlen=1000)
        self.performance_metrics = RoutingPerformanceMetrics()
        self.route_allocations = {route_id: 1.0/len(self.profit_routes) 
                                for route_id in self.profit_routes.keys()}
        
        # State management
        self.last_rebalance = datetime.now(timezone.utc)
        self.active_routes = set(self.profit_routes.keys())
        self.suspended_routes = set()
        
        # Mathematical validation cache
        self.validation_cache = {}
        self.sustainment_cache = deque(maxlen=50)
        
        self.cli_handler.log_safe(logger, 'info', "ProfitRoutingEngine initialized with mathematical foundation")
    
    def _initialize_profit_routes(self) -> Dict[str, ProfitRoute]:
        """Initialize the profit routing strategies"""
        routes = {}
        
        # Micro scalping route (4b phase gate)
        routes['micro_scalp_4b'] = ProfitRoute(
            route_id='micro_scalp_4b',
            route_type=ProfitRouteType.MICRO_SCALP,
            phase_gate='4b',
            weight=0.25,
            target_profit_rate=0.01,  # 1% quick profits
            max_risk_ratio=0.005,     # 0.5% max risk
            sustainment_requirement=0.75  # Higher requirement for fast trades
        )
        
        # Harmonic swing route (8b phase gate)
        routes['harmonic_swing_8b'] = ProfitRoute(
            route_id='harmonic_swing_8b',
            route_type=ProfitRouteType.HARMONIC_SWING,
            phase_gate='8b',
            weight=0.40,
            target_profit_rate=0.03,  # 3% swing profits
            max_risk_ratio=0.015,     # 1.5% max risk
            sustainment_requirement=0.65  # Standard requirement
        )
        
        # Strategic hold route (42b phase gate)
        routes['strategic_hold_42b'] = ProfitRoute(
            route_id='strategic_hold_42b',
            route_type=ProfitRouteType.STRATEGIC_HOLD,
            phase_gate='42b',
            weight=0.25,
            target_profit_rate=0.08,  # 8% strategic profits
            max_risk_ratio=0.04,      # 4% max risk
            sustainment_requirement=0.60  # Lower requirement for long-term
        )
        
        # Diversified blend route (multi-gate)
        routes['diversified_blend'] = ProfitRoute(
            route_id='diversified_blend',
            route_type=ProfitRouteType.DIVERSIFIED_BLEND,
            phase_gate='8b',  # Default to harmonic
            weight=0.10,
            target_profit_rate=0.05,  # 5% blended profits
            max_risk_ratio=0.02,      # 2% max risk
            sustainment_requirement=0.70  # Higher requirement for diversity
        )
        
        return routes
    
    async def analyze_profit_routing_opportunity(self, 
                                               trade_signal: MathematicalTradeSignal,
                                               market_data: Dict[str, Any]) -> ProfitRoutingDecision:
        """
        Analyze profit routing opportunities for a trade signal
        
        Args:
            trade_signal: Mathematical trade signal from execution manager
            market_data: Current market data
            
        Returns:
            ProfitRoutingDecision with optimal routing strategy
        """
        try:
            # STEP 1: Calculate mathematical context
            self.cli_handler.log_safe(logger, 'info', "📊 Analyzing profit routing opportunity...")
            
            math_context = self._create_mathematical_context(trade_signal, market_data)
            
            # STEP 2: Calculate sustainment vector
            sustainment_vector = self.sustainment_lib.calculate_sustainment_vector(math_context)
            sustainment_index = sustainment_vector.sustainment_index()
            
            # STEP 3: Evaluate each profit route
            route_scores = await self._evaluate_profit_routes(
                trade_signal, market_data, sustainment_vector
            )
            
            # STEP 4: Select optimal routes based on optimization mode
            selected_routes, allocations = self._select_optimal_routes(
                route_scores, sustainment_index
            )
            
            # STEP 5: Calculate total position sizing
            total_position_size = self._calculate_total_position_size(
                trade_signal, selected_routes, allocations, sustainment_index
            )
            
            # STEP 6: Calculate expected profit and risk
            expected_profit = self._calculate_expected_profit(
                selected_routes, allocations, trade_signal
            )
            risk_assessment = self._calculate_route_risk(
                selected_routes, allocations, market_data
            )
            
            # STEP 7: Create graded profit vector
            profit_vector = self._create_profit_vector(
                trade_signal, expected_profit, risk_assessment
            )
            
            # STEP 8: Determine execution timing
            timing_recommendation = self._determine_execution_timing(
                sustainment_index, risk_assessment, route_scores
            )
            
            # STEP 9: Create routing decision
            routing_decision = ProfitRoutingDecision(
                selected_routes=selected_routes,
                route_allocations=allocations,
                total_position_size=total_position_size,
                expected_profit=expected_profit,
                risk_assessment=risk_assessment,
                sustainment_index=sustainment_index,
                mathematical_validity=sustainment_index >= self.min_sustainment_index,
                profit_vector=profit_vector,
                execution_priority=self._calculate_execution_priority(route_scores),
                routing_confidence=self._calculate_route_confidence(route_scores[selected_routes[0]] if selected_routes else {}),
                timing_recommendation=timing_recommendation,
                reasoning=self._generate_routing_reasoning(selected_routes, allocations, sustainment_index, route_scores)
            )
            
            # Store in routing history
            self.routing_history.append({
                'timestamp': datetime.now(timezone.utc),
                'decision': routing_decision,
                'trade_signal_id': trade_signal.signal_id
            })
            
            self.cli_handler.log_safe(logger, 'info', f"✅ Profit routing analysis complete: {len(selected_routes)} routes selected")
            return routing_decision
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "analyze_profit_routing_opportunity")
            self.cli_handler.log_safe(logger, 'error', error_message)
            return self._create_fallback_decision(trade_signal)
    
    async def execute_profit_routing(self,
                                   routing_decision: ProfitRoutingDecision,
                                   trade_signal: MathematicalTradeSignal,
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute profit routing decision through selected routes
        
        Args:
            routing_decision: Profit routing decision
            trade_signal: Original trade signal
            market_data: Current market data
            
        Returns:
            Execution results with profit routing information
        """
        try:
            execution_results = []
            total_executed_volume = 0.0
            total_expected_profit = 0.0
            
            # Execute through each selected route
            for route_id in routing_decision.selected_routes:
                route = self.profit_routes[route_id]
                allocation = routing_decision.route_allocations[route_id]
                
                # Create route-specific trade signal
                route_signal = self._create_route_signal(
                    trade_signal, route, allocation, routing_decision
                )
                
                # Execute through appropriate phase gate
                if self.phase_controller:
                    route_result = await self.phase_controller.execute_through_phase_gate(
                        route_signal, market_data
                    )
                else:
                    # Direct execution if no phase controller
                    if self.execution_manager:
                        exec_result = await self.execution_manager.execute_signal(route_signal)
                        route_result = {
                            'status': 'executed',
                            'execution_result': exec_result,
                            'gate_type': route.phase_gate,
                            'route_id': route_id
                        }
                    else:
                        route_result = {
                            'status': 'simulated',
                            'route_id': route_id,
                            'allocation': allocation
                        }
                
                # Track route performance
                await self._update_route_performance(route_id, route_result, route_signal)
                
                execution_results.append({
                    'route_id': route_id,
                    'allocation': allocation,
                    'result': route_result
                })
                
                if route_result['status'] == 'executed':
                    total_executed_volume += route_signal.position_size
                    # Estimate profit (simplified)
                    estimated_profit = route_signal.position_size * route.target_profit_rate
                    total_expected_profit += estimated_profit
            
            # Update overall performance metrics
            self._update_performance_metrics(
                routing_decision, execution_results, total_expected_profit
            )
            
            # Check if rebalancing is needed
            if self._should_rebalance():
                await self._rebalance_routes()
            
            return {
                'status': 'completed',
                'executed_routes': len(execution_results),
                'total_volume': total_executed_volume,
                'expected_profit': total_expected_profit,
                'route_results': execution_results,
                'routing_decision': routing_decision,
                'performance_metrics': asdict(self.performance_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error executing profit routing: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'routing_decision': routing_decision
            }
    
    async def _evaluate_profit_routes(self,
                                    trade_signal: MathematicalTradeSignal,
                                    market_data: Dict[str, Any],
                                    sustainment_vector: SustainmentVector) -> Dict[str, Dict[str, float]]:
        """Evaluate each profit route for the current opportunity"""
        route_scores = {}
        
        for route_id, route in self.profit_routes.items():
            if not route.active or route_id in self.suspended_routes:
                continue
            
            # Calculate route-specific scores
            scores = {
                'profit_potential': self._calculate_profit_potential(route, trade_signal, market_data),
                'risk_alignment': self._calculate_risk_alignment(route, trade_signal),
                'sustainment_compatibility': self._calculate_sustainment_compatibility(route, sustainment_vector),
                'performance_track_record': self._calculate_performance_score(route),
                'market_suitability': self._calculate_market_suitability(route, market_data),
                'phase_gate_alignment': self._calculate_phase_gate_alignment(route, trade_signal),
                'timing_appropriateness': self._calculate_timing_score(route, market_data)
            }
            
            # Calculate weighted overall score
            overall_score = (
                scores['profit_potential'] * 0.25 +
                scores['risk_alignment'] * 0.20 +
                scores['sustainment_compatibility'] * 0.15 +
                scores['performance_track_record'] * 0.15 +
                scores['market_suitability'] * 0.10 +
                scores['phase_gate_alignment'] * 0.10 +
                scores['timing_appropriateness'] * 0.05
            )
            
            scores['overall_score'] = overall_score
            scores['confidence'] = self._calculate_route_confidence(scores)
            
            route_scores[route_id] = scores
        
        return route_scores
    
    def _select_optimal_routes(self,
                             route_scores: Dict[str, Dict[str, float]],
                             sustainment_index: float) -> Tuple[List[str], Dict[str, float]]:
        """Select optimal routes based on optimization mode"""
        
        if self.optimization_mode == ProfitOptimizationMode.MAXIMIZE_TOTAL:
            return self._select_for_total_profit(route_scores)
        elif self.optimization_mode == ProfitOptimizationMode.MAXIMIZE_RATIO:
            return self._select_for_profit_ratio(route_scores)
        elif self.optimization_mode == ProfitOptimizationMode.MAXIMIZE_SUSTAINED:
            return self._select_for_sustained_profit(route_scores, sustainment_index)
        elif self.optimization_mode == ProfitOptimizationMode.MAXIMIZE_VELOCITY:
            return self._select_for_profit_velocity(route_scores)
        else:
            return self._select_for_sustained_profit(route_scores, sustainment_index)
    
    def _select_for_sustained_profit(self,
                                   route_scores: Dict[str, Dict[str, float]],
                                   sustainment_index: float) -> Tuple[List[str], Dict[str, float]]:
        """Select routes optimized for sustained profit with mathematical validation"""
        
        # Filter routes that meet sustainment requirements
        valid_routes = {
            route_id: scores for route_id, scores in route_scores.items()
            if (scores['sustainment_compatibility'] >= 0.6 and 
                scores['overall_score'] >= 0.5 and
                self.profit_routes[route_id].sustainment_requirement <= sustainment_index * 1.1)
        }
        
        if not valid_routes:
            # Fallback to best available route
            best_route = max(route_scores.items(), key=lambda x: x[1]['overall_score'])
            return [best_route[0]], {best_route[0]: 1.0}
        
        # Sort by combined score (sustainment + performance + consistency)
        sorted_routes = sorted(
            valid_routes.items(),
            key=lambda x: (
                x[1]['sustainment_compatibility'] * 0.4 +
                x[1]['overall_score'] * 0.4 +
                x[1]['performance_track_record'] * 0.2
            ),
            reverse=True
        )
        
        # Select top routes with diversification
        selected_routes = []
        allocations = {}
        total_allocation = 0.0
        
        for route_id, scores in sorted_routes[:3]:  # Top 3 routes max
            # Calculate allocation based on score and diversification
            base_allocation = scores['overall_score'] * scores['sustainment_compatibility']
            
            # Diversification bonus
            route_type = self.profit_routes[route_id].route_type
            type_count = sum(1 for r in selected_routes 
                           if self.profit_routes[r].route_type == route_type)
            diversification_factor = 1.0 - (type_count * 0.2)
            
            allocation = base_allocation * diversification_factor
            
            if total_allocation + allocation <= 1.0 and allocation >= 0.1:
                selected_routes.append(route_id)
                allocations[route_id] = allocation
                total_allocation += allocation
        
        # Normalize allocations
        if total_allocation > 0:
            allocations = {route_id: alloc / total_allocation 
                         for route_id, alloc in allocations.items()}
        
        return selected_routes, allocations 

    # === CALCULATION METHODS ===
    
    def _calculate_profit_potential(self, route: ProfitRoute, 
                                  trade_signal: MathematicalTradeSignal,
                                  market_data: Dict[str, Any]) -> float:
        """Calculate profit potential for a route"""
        # Base profit potential from trade signal
        base_potential = trade_signal.confidence * 0.6
        
        # Route-specific adjustments
        route_multiplier = {
            ProfitRouteType.MICRO_SCALP: 0.8,      # Lower but faster
            ProfitRouteType.HARMONIC_SWING: 1.0,   # Balanced
            ProfitRouteType.STRATEGIC_HOLD: 1.3,   # Higher potential
            ProfitRouteType.DIVERSIFIED_BLEND: 0.9  # Diversified
        }.get(route.route_type, 1.0)
        
        # Market volatility factor
        volatility = self._calculate_market_volatility(market_data)
        volatility_factor = min(1.5, 1.0 + volatility * 0.5)
        
        # Historical performance factor
        if route.total_trades > 10:
            performance_factor = min(1.3, max(0.7, route.average_return + 1.0))
        else:
            performance_factor = 1.0
        
        profit_potential = base_potential * route_multiplier * volatility_factor * performance_factor
        return min(1.0, max(0.0, profit_potential))
    
    def _calculate_risk_alignment(self, route: ProfitRoute, 
                                trade_signal: MathematicalTradeSignal) -> float:
        """Calculate how well route risk aligns with trade signal"""
        signal_risk = 1.0 - trade_signal.confidence
        route_max_risk = route.max_risk_ratio
        
        # Check if signal risk is within route tolerance
        if signal_risk <= route_max_risk:
            # Perfect alignment
            alignment = 1.0
        elif signal_risk <= route_max_risk * 1.5:
            # Acceptable with penalty
            alignment = 0.7
        else:
            # Poor alignment
            alignment = 0.3
        
        # Adjust for trade signal risk level
        risk_adjustments = {
            'low': 1.0,
            'medium': 0.8,
            'high': 0.6,
            'critical': 0.3
        }
        risk_factor = risk_adjustments.get(trade_signal.risk_level.value, 0.8)
        
        return alignment * risk_factor
    
    def _calculate_sustainment_compatibility(self, route: ProfitRoute,
                                           sustainment_vector: SustainmentVector) -> float:
        """Calculate route compatibility with sustainment principles"""
        current_si = sustainment_vector.sustainment_index()
        required_si = route.sustainment_requirement
        
        if current_si >= required_si:
            # Calculate how much above requirement
            excess = current_si - required_si
            compatibility = min(1.0, 0.8 + excess * 2.0)
        else:
            # Calculate penalty for being below requirement
            deficit = required_si - current_si
            compatibility = max(0.2, 0.8 - deficit * 3.0)
        
        # Bonus for economy principle (profit-focused)
        economy_score = sustainment_vector.principles[4]  # Economy is index 4
        economy_bonus = (economy_score - 0.5) * 0.2
        
        return min(1.0, max(0.0, compatibility + economy_bonus))
    
    def _calculate_performance_score(self, route: ProfitRoute) -> float:
        """Calculate historical performance score for route"""
        if route.total_trades < 5:
            return 0.5  # Neutral for new routes
        
        # Success rate component
        success_component = route.success_rate
        
        # Return component
        return_component = min(1.0, max(0.0, route.average_return + 0.5))
        
        # Sharpe ratio component
        sharpe_component = min(1.0, max(0.0, route.sharpe_ratio / 2.0 + 0.5))
        
        # Volume component (more trades = more confidence)
        volume_component = min(1.0, route.total_trades / 100.0)
        
        performance_score = (
            success_component * 0.4 +
            return_component * 0.3 +
            sharpe_component * 0.2 +
            volume_component * 0.1
        )
        
        return performance_score
    
    def _calculate_market_suitability(self, route: ProfitRoute,
                                    market_data: Dict[str, Any]) -> float:
        """Calculate how suitable current market is for route"""
        volatility = self._calculate_market_volatility(market_data)
        volume = market_data.get('volume', 1000.0)
        trend_strength = self._calculate_trend_strength(market_data)
        
        # Route-specific market preferences
        if route.route_type == ProfitRouteType.MICRO_SCALP:
            # Prefers high volume, low volatility
            suitability = (
                (1.0 - volatility) * 0.4 +  # Low volatility
                min(1.0, volume / 10000.0) * 0.4 +  # High volume
                trend_strength * 0.2  # Some trend
            )
        elif route.route_type == ProfitRouteType.HARMONIC_SWING:
            # Prefers medium volatility, good trends
            suitability = (
                (1.0 - abs(volatility - 0.5) * 2) * 0.3 +  # Medium volatility
                trend_strength * 0.5 +  # Strong trends
                min(1.0, volume / 5000.0) * 0.2  # Decent volume
            )
        elif route.route_type == ProfitRouteType.STRATEGIC_HOLD:
            # Prefers strong trends, less sensitive to volatility
            suitability = (
                trend_strength * 0.6 +  # Strong trends most important
                min(1.0, volume / 2000.0) * 0.3 +  # Some volume
                min(1.0, volatility + 0.3) * 0.1  # Volatility acceptable
            )
        else:  # DIVERSIFIED_BLEND
            # Balanced approach
            suitability = (
                (1.0 - abs(volatility - 0.4) * 1.5) * 0.3 +
                trend_strength * 0.4 +
                min(1.0, volume / 7500.0) * 0.3
            )
        
        return min(1.0, max(0.0, suitability))
    
    def _calculate_phase_gate_alignment(self, route: ProfitRoute,
                                      trade_signal: MathematicalTradeSignal) -> float:
        """Calculate alignment between route and trade signal phase requirements"""
        # Get trade signal's optimal phase gate from entropy/coherence
        entropy = trade_signal.entropy_score
        coherence = trade_signal.coherence_score
        
        # Determine optimal gate for trade signal
        if entropy <= 0.3 and coherence >= 0.8:
            optimal_gate = '4b'
        elif entropy <= 0.7 and coherence >= 0.5:
            optimal_gate = '8b'
        else:
            optimal_gate = '42b'
        
        # Check alignment with route's preferred gate
        if route.phase_gate == optimal_gate:
            return 1.0
        elif route.route_type == ProfitRouteType.DIVERSIFIED_BLEND:
            return 0.8  # Diversified can adapt
        else:
            # Calculate penalty based on distance
            gate_order = {'4b': 0, '8b': 1, '42b': 2}
            route_pos = gate_order.get(route.phase_gate, 1)
            optimal_pos = gate_order.get(optimal_gate, 1)
            distance = abs(route_pos - optimal_pos)
            return max(0.3, 1.0 - distance * 0.35)
    
    def _calculate_timing_score(self, route: ProfitRoute, 
                              market_data: Dict[str, Any]) -> float:
        """Calculate timing appropriateness for route"""
        current_time = datetime.now(timezone.utc)
        
        # Time since last execution
        if route.last_execution:
            time_since_last = (current_time - route.last_execution).total_seconds()
            
            # Route-specific timing preferences
            if route.route_type == ProfitRouteType.MICRO_SCALP:
                # Can execute frequently
                optimal_interval = 300  # 5 minutes
            elif route.route_type == ProfitRouteType.HARMONIC_SWING:
                # Moderate frequency
                optimal_interval = 1800  # 30 minutes
            else:  # STRATEGIC_HOLD
                # Less frequent
                optimal_interval = 7200  # 2 hours
            
            # Calculate timing score based on interval
            if time_since_last >= optimal_interval:
                timing_score = 1.0
            else:
                timing_score = time_since_last / optimal_interval
        else:
            timing_score = 1.0  # First execution
        
        # Market timing factors
        market_volatility = self._calculate_market_volatility(market_data)
        
        # Adjust based on market conditions
        if route.route_type == ProfitRouteType.MICRO_SCALP:
            # Prefers stable markets
            market_timing = 1.0 - market_volatility
        else:
            # Other routes can benefit from volatility
            market_timing = 0.5 + market_volatility * 0.5
        
        return (timing_score * 0.7 + market_timing * 0.3)
    
    def _calculate_route_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate overall confidence in route selection"""
        # Weight the scores for confidence calculation
        confidence_weights = {
            'profit_potential': 0.3,
            'risk_alignment': 0.25,
            'sustainment_compatibility': 0.2,
            'performance_track_record': 0.15,
            'market_suitability': 0.1
        }
        
        weighted_confidence = sum(
            scores.get(metric, 0.5) * weight
            for metric, weight in confidence_weights.items()
        )
        
        # Penalize if any critical score is too low
        critical_scores = ['sustainment_compatibility', 'risk_alignment']
        critical_penalty = 0.0
        for score_name in critical_scores:
            if scores.get(score_name, 0.5) < 0.4:
                critical_penalty += 0.2
        
        confidence = max(0.1, weighted_confidence - critical_penalty)
        return min(1.0, confidence)
    
    # === HELPER METHODS ===
    
    def _create_mathematical_context(self, trade_signal: MathematicalTradeSignal,
                                   market_data: Dict[str, Any]) -> MathematicalContext:
        """Create mathematical context for sustainment calculations"""
        return MathematicalContext(
            current_state={
                'price': market_data.get('price', 100.0),
                'volume': market_data.get('volume', 1000.0),
                'entropy': trade_signal.entropy_score,
                'coherence': trade_signal.coherence_score
            },
            system_metrics={
                'profit_delta': self.performance_metrics.average_profit_per_trade,
                'cpu_cost': 1.0,
                'gpu_cost': 2.0,
                'memory_cost': 0.5,
                'latency_ms': 50.0,
                'operations_count': 100,
                'active_strategies': len(self.active_routes)
            },
            historical_data=[
                {'profit': entry['decision'].expected_profit,
                 'timestamp': entry['timestamp']}
                for entry in list(self.routing_history)[-10:]
            ]
        )
    
    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate current market volatility"""
        price_series = market_data.get('price_series', [])
        if len(price_series) < 2:
            return 0.5  # Default volatility
        
        returns = np.diff(np.log(price_series))
        volatility = float(np.std(returns))
        return min(1.0, volatility * 20)  # Scale to 0-1
    
    def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> float:
        """Calculate trend strength"""
        price_series = market_data.get('price_series', [])
        if len(price_series) < 3:
            return 0.5
        
        # Simple trend calculation
        short_avg = np.mean(price_series[-3:])
        long_avg = np.mean(price_series[-6:] if len(price_series) >= 6 else price_series)
        
        trend_ratio = short_avg / long_avg if long_avg > 0 else 1.0
        trend_strength = abs(trend_ratio - 1.0) * 10  # Scale trend signal
        
        return min(1.0, trend_strength)
    
    def _calculate_total_position_size(self, trade_signal: MathematicalTradeSignal,
                                     selected_routes: List[str],
                                     allocations: Dict[str, float],
                                     sustainment_index: float) -> float:
        """Calculate total position size considering sustainment"""
        base_position = trade_signal.position_size
        
        # Sustainment adjustment
        if sustainment_index >= self.min_sustainment_index:
            sustainment_multiplier = min(1.2, sustainment_index / self.min_sustainment_index)
        else:
            sustainment_multiplier = max(0.5, sustainment_index / self.min_sustainment_index)
        
        # Route diversification bonus
        diversification_bonus = 1.0 + (len(selected_routes) - 1) * 0.1
        
        # Risk constraint
        total_risk = sum(
            allocations[route_id] * self.profit_routes[route_id].max_risk_ratio
            for route_id in selected_routes
        )
        
        if total_risk > self.max_total_risk:
            risk_multiplier = self.max_total_risk / total_risk
        else:
            risk_multiplier = 1.0
        
        total_position = (base_position * sustainment_multiplier * 
                         diversification_bonus * risk_multiplier)
        
        return min(trade_signal.position_size * 1.5, total_position)  # Cap at 1.5x original
    
    def _calculate_expected_profit(self, selected_routes: List[str],
                                 allocations: Dict[str, float],
                                 trade_signal: MathematicalTradeSignal) -> float:
        """Calculate expected profit from selected routes"""
        total_expected_profit = 0.0
        
        for route_id in selected_routes:
            route = self.profit_routes[route_id]
            allocation = allocations[route_id]
            
            # Base expected profit from route target
            route_profit = route.target_profit_rate * allocation
            
            # Adjust for trade signal confidence
            confidence_adjustment = trade_signal.confidence
            
            # Adjust for route performance
            if route.total_trades > 5:
                performance_adjustment = min(1.3, max(0.7, route.average_return + 1.0))
            else:
                performance_adjustment = 1.0
            
            adjusted_profit = route_profit * confidence_adjustment * performance_adjustment
            total_expected_profit += adjusted_profit
        
        return total_expected_profit
    
    def _calculate_route_risk(self, selected_routes: List[str],
                            allocations: Dict[str, float],
                            market_data: Dict[str, Any]) -> float:
        """Calculate total risk from selected routes"""
        total_risk = 0.0
        market_volatility = self._calculate_market_volatility(market_data)
        
        for route_id in selected_routes:
            route = self.profit_routes[route_id]
            allocation = allocations[route_id]
            
            # Base route risk
            base_risk = route.max_risk_ratio * allocation
            
            # Market volatility adjustment
            volatility_multiplier = 1.0 + market_volatility * 0.5
            
            # Route-specific risk adjustments
            route_risk_multiplier = {
                ProfitRouteType.MICRO_SCALP: 0.8,      # Lower risk
                ProfitRouteType.HARMONIC_SWING: 1.0,   # Standard risk
                ProfitRouteType.STRATEGIC_HOLD: 1.2,   # Higher risk
                ProfitRouteType.DIVERSIFIED_BLEND: 0.9  # Diversified risk
            }.get(route.route_type, 1.0)
            
            route_risk = base_risk * volatility_multiplier * route_risk_multiplier
            total_risk += route_risk
        
        # Diversification benefit
        if len(selected_routes) > 1:
            diversification_factor = 1.0 - (len(selected_routes) - 1) * 0.1
            total_risk *= max(0.7, diversification_factor)
        
        return total_risk
    
    def _create_profit_vector(self, trade_signal: MathematicalTradeSignal,
                            expected_profit: float,
                            risk_assessment: float) -> GradedProfitVector:
        """Create graded profit vector for the routing decision"""
        return GradedProfitVector(
            profit=expected_profit,
            volume_allocated=trade_signal.position_size,
            time_held=0.0,  # Will be updated after execution
            signal_strength=trade_signal.confidence,
            smart_money_score=risk_assessment  # Use risk as smart money indicator
        )
    
    def _determine_execution_timing(self, sustainment_index: float,
                                  risk_assessment: float,
                                  route_scores: Dict[str, Dict[str, float]]) -> str:
        """Determine optimal execution timing"""
        avg_confidence = np.mean([scores['confidence'] for scores in route_scores.values()])
        
        if sustainment_index >= 0.8 and risk_assessment <= 0.3 and avg_confidence >= 0.8:
            return "immediate"
        elif sustainment_index >= 0.65 and risk_assessment <= 0.5:
            return "delayed"
        else:
            return "queued"
    
    def _calculate_execution_priority(self, route_scores: Dict[str, Dict[str, float]]) -> float:
        """Calculate execution priority based on route scores"""
        if not route_scores:
            return 0.5
        
        max_overall_score = max(scores['overall_score'] for scores in route_scores.values())
        avg_confidence = np.mean([scores['confidence'] for scores in route_scores.values()])
        
        return (max_overall_score * 0.6 + avg_confidence * 0.4)
    
    def _generate_routing_reasoning(self, selected_routes: List[str],
                                  allocations: Dict[str, float],
                                  sustainment_index: float,
                                  route_scores: Dict[str, Dict[str, float]]) -> str:
        """Generate human-readable reasoning for routing decision"""
        reasons = []
        
        # Sustainment reasoning
        if sustainment_index >= 0.8:
            reasons.append("Strong sustainment index supports aggressive routing")
        elif sustainment_index >= 0.65:
            reasons.append("Adequate sustainment index allows moderate routing")
        else:
            reasons.append("Low sustainment index requires conservative routing")
        
        # Route selection reasoning
        if len(selected_routes) == 1:
            route = self.profit_routes[selected_routes[0]]
            reasons.append(f"Single route strategy using {route.route_type.value}")
        else:
            reasons.append(f"Diversified strategy across {len(selected_routes)} routes")
        
        # Top route reasoning
        if route_scores:
            best_route = max(route_scores.items(), key=lambda x: x[1]['overall_score'])
            reasons.append(f"Primary route {best_route[0]} scored {best_route[1]['overall_score']:.3f}")
        
        return "; ".join(reasons)
    
    def _create_fallback_decision(self, trade_signal: MathematicalTradeSignal) -> ProfitRoutingDecision:
        """Create fallback decision when analysis fails"""
        # Select the most conservative route
        conservative_route = 'harmonic_swing_8b'  # Default to harmonic
        
        return ProfitRoutingDecision(
            selected_routes=[conservative_route],
            route_allocations={conservative_route: 1.0},
            total_position_size=trade_signal.position_size * 0.5,  # Reduce size for safety
            expected_profit=0.02,  # Conservative estimate
            risk_assessment=0.05,  # Conservative risk
            sustainment_index=0.6,
            mathematical_validity=False,
            profit_vector=GradedProfitVector(
                profit=0.02,
                signal_strength=0.5,
                smart_money_score=0.5
            ),
            execution_priority=0.3,
            routing_confidence=0.4,
            timing_recommendation="delayed",
            reasoning="Fallback decision due to analysis error"
        )

    # === ADDITIONAL ROUTE SELECTION METHODS ===
    
    def _select_for_total_profit(self, route_scores: Dict[str, Dict[str, float]]) -> Tuple[List[str], Dict[str, float]]:
        """Select routes optimized for maximum total profit"""
        # Sort by profit potential
        sorted_routes = sorted(
            route_scores.items(),
            key=lambda x: x[1]['profit_potential'] * x[1]['overall_score'],
            reverse=True
        )
        
        selected_routes = []
        allocations = {}
        total_allocation = 0.0
        
        for route_id, scores in sorted_routes:
            if total_allocation >= 1.0:
                break
            
            # Allocate based on profit potential
            allocation = min(1.0 - total_allocation, scores['profit_potential'] * 0.5)
            
            if allocation >= 0.1:  # Minimum 10% allocation
                selected_routes.append(route_id)
                allocations[route_id] = allocation
                total_allocation += allocation
        
        # Normalize allocations
        if total_allocation > 0:
            allocations = {route_id: alloc / total_allocation 
                         for route_id, alloc in allocations.items()}
        
        return selected_routes, allocations
    
    def _select_for_profit_ratio(self, route_scores: Dict[str, Dict[str, float]]) -> Tuple[List[str], Dict[str, float]]:
        """Select routes optimized for profit/risk ratio"""
        # Calculate profit/risk ratios
        ratios = {}
        for route_id, scores in route_scores.items():
            route = self.profit_routes[route_id]
            profit_score = scores['profit_potential']
            risk_score = route.max_risk_ratio
            ratios[route_id] = profit_score / max(risk_score, 0.001)  # Avoid division by zero
        
        # Sort by ratio
        sorted_routes = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        
        # Select top routes
        selected_routes = [route_id for route_id, _ in sorted_routes[:2]]
        
        # Equal allocation by default, adjusted by performance
        if len(selected_routes) == 1:
            allocations = {selected_routes[0]: 1.0}
        else:
            total_ratio = sum(ratios[route_id] for route_id in selected_routes)
            allocations = {route_id: ratios[route_id] / total_ratio 
                         for route_id in selected_routes}
        
        return selected_routes, allocations
    
    def _select_for_profit_velocity(self, route_scores: Dict[str, Dict[str, float]]) -> Tuple[List[str], Dict[str, float]]:
        """Select routes optimized for profit velocity (profit per time)"""
        # Calculate velocity scores
        velocity_scores = {}
        for route_id, scores in route_scores.items():
            route = self.profit_routes[route_id]
            
            # Time factor based on route type
            time_factors = {
                ProfitRouteType.MICRO_SCALP: 4.0,      # Fast trades
                ProfitRouteType.HARMONIC_SWING: 2.0,   # Medium speed
                ProfitRouteType.STRATEGIC_HOLD: 1.0,   # Slow trades
                ProfitRouteType.DIVERSIFIED_BLEND: 2.5  # Mixed speed
            }
            
            time_factor = time_factors.get(route.route_type, 2.0)
            velocity_scores[route_id] = scores['profit_potential'] * time_factor
        
        # Sort by velocity
        sorted_routes = sorted(velocity_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prefer faster routes
        selected_routes = []
        allocations = {}
        total_velocity = 0.0
        
        for route_id, velocity in sorted_routes[:3]:
            if velocity >= max(velocity_scores.values()) * 0.5:  # At least 50% of max velocity
                selected_routes.append(route_id)
                allocations[route_id] = velocity
                total_velocity += velocity
        
        # Normalize allocations
        if total_velocity > 0:
            allocations = {route_id: alloc / total_velocity 
                         for route_id, alloc in allocations.items()}
        
        return selected_routes, allocations
    
    # === PERFORMANCE TRACKING AND MANAGEMENT ===
    
    async def _update_route_performance(self, route_id: str, 
                                      execution_result: Dict[str, Any],
                                      trade_signal: MathematicalTradeSignal):
        """Update performance metrics for a specific route"""
        route = self.profit_routes[route_id]
        
        # Update execution timestamp
        route.last_execution = datetime.now(timezone.utc)
        
        # Extract execution data
        if execution_result['status'] == 'executed' and 'execution_result' in execution_result:
            exec_result = execution_result['execution_result']
            
            # Calculate profit (simplified - in real system would track actual P&L)
            if hasattr(exec_result, 'executed_price') and exec_result.executed_price:
                estimated_profit = trade_signal.position_size * route.target_profit_rate
                route.total_profit += estimated_profit
                
                # Update success tracking
                route.total_trades += 1
                if estimated_profit > 0:
                    route.success_rate = ((route.success_rate * (route.total_trades - 1)) + 1) / route.total_trades
                else:
                    route.success_rate = (route.success_rate * (route.total_trades - 1)) / route.total_trades
                
                # Update average return
                route.average_return = ((route.average_return * (route.total_trades - 1)) + 
                                      (estimated_profit / trade_signal.position_size)) / route.total_trades
                
                # Add to graded vectors for analysis
                profit_vector = GradedProfitVector(
                    profit=estimated_profit,
                    volume_allocated=trade_signal.position_size,
                    signal_strength=trade_signal.confidence,
                    smart_money_score=route.average_return
                )
                route.graded_vectors.append(profit_vector)
                
                # Keep limited history
                if len(route.graded_vectors) > 100:
                    route.graded_vectors = route.graded_vectors[-100:]
        
        # Update performance level
        route.performance_level = self._assess_route_performance_level(route)
        
        logger.info(f"Updated performance for route {route_id}: {route.performance_level.value}")
    
    def _assess_route_performance_level(self, route: ProfitRoute) -> RoutePerformanceLevel:
        """Assess performance level of a route"""
        if route.total_trades < 5:
            return RoutePerformanceLevel.PERFORMING  # Neutral for new routes
        
        # Performance criteria
        success_threshold = 0.6
        return_threshold = 0.02  # 2% average return
        
        if (route.success_rate >= success_threshold and 
            route.average_return >= return_threshold and
            route.total_profit > 0):
            return RoutePerformanceLevel.EXCELLENT
        elif (route.success_rate >= success_threshold * 0.8 and 
              route.average_return >= return_threshold * 0.5):
            return RoutePerformanceLevel.PERFORMING
        elif route.total_profit >= 0:
            return RoutePerformanceLevel.MARGINAL
        else:
            return RoutePerformanceLevel.FAILING
    
    def _update_performance_metrics(self, routing_decision: ProfitRoutingDecision,
                                  execution_results: List[Dict[str, Any]],
                                  estimated_profit: float):
        """Update overall performance metrics"""
        # Update total metrics
        self.performance_metrics.total_profit += estimated_profit
        self.performance_metrics.total_volume_traded += routing_decision.total_position_size
        
        # Update average profit per trade
        total_trades = sum(route.total_trades for route in self.profit_routes.values())
        if total_trades > 0:
            self.performance_metrics.average_profit_per_trade = (
                self.performance_metrics.total_profit / total_trades
            )
        
        # Update profit velocity (profit per hour)
        if self.routing_history:
            first_trade_time = self.routing_history[0]['timestamp']
            time_diff = (datetime.now(timezone.utc) - first_trade_time).total_seconds() / 3600.0  # Hours
            if time_diff > 0:
                self.performance_metrics.profit_velocity = (
                    self.performance_metrics.total_profit / time_diff
                )
        
        # Update route-specific performance
        for route_id, route in self.profit_routes.items():
            if route_id not in self.performance_metrics.route_performance:
                self.performance_metrics.route_performance[route_id] = {}
            
            route_perf = self.performance_metrics.route_performance[route_id]
            route_perf.update({
                'total_profit': route.total_profit,
                'success_rate': route.success_rate,
                'average_return': route.average_return,
                'total_trades': route.total_trades,
                'performance_level': route.performance_level.value
            })
        
        # Update mathematical metrics
        self.performance_metrics.sustainment_index = routing_decision.sustainment_index
        self.performance_metrics.mathematical_validation_rate = (
            1.0 if routing_decision.mathematical_validity else 0.0
        )
        
        # Update risk metrics
        self.performance_metrics.total_risk_taken += routing_decision.risk_assessment
        if self.performance_metrics.total_risk_taken > 0:
            self.performance_metrics.profit_to_risk_ratio = (
                self.performance_metrics.total_profit / self.performance_metrics.total_risk_taken
            )
    
    def _should_rebalance(self) -> bool:
        """Check if route rebalancing is needed"""
        current_time = datetime.now(timezone.utc)
        time_since_rebalance = current_time - self.last_rebalance
        
        # Time-based rebalancing
        if time_since_rebalance >= self.rebalance_frequency:
            return True
        
        # Performance-based rebalancing
        failing_routes = sum(1 for route in self.profit_routes.values() 
                           if route.performance_level == RoutePerformanceLevel.FAILING)
        
        if failing_routes > len(self.profit_routes) * 0.3:  # More than 30% failing
            return True
        
        # Allocation drift check
        current_allocations = list(self.route_allocations.values())
        if len(current_allocations) > 1:
            allocation_variance = np.var(current_allocations)
            if allocation_variance > 0.1:  # High variance in allocations
                return True
        
        return False
    
    async def _rebalance_routes(self):
        """Rebalance route allocations based on performance"""
        self.cli_handler.log_safe(logger, 'info', "🔄 Rebalancing profit routes...")
        
        # Calculate new allocations based on performance
        performance_scores = {}
        for route_id, route in self.profit_routes.items():
            if route.total_trades >= 5:
                # Performance-based score
                performance_score = (
                    route.success_rate * 0.4 +
                    min(1.0, max(0.0, route.average_return + 0.5)) * 0.3 +
                    (1.0 if route.performance_level != RoutePerformanceLevel.FAILING else 0.0) * 0.3
                )
            else:
                # Default score for new routes
                performance_score = 0.6
            
            performance_scores[route_id] = performance_score
        
        # Normalize scores to allocations
        total_score = sum(performance_scores.values())
        if total_score > 0:
            new_allocations = {
                route_id: score / total_score
                for route_id, score in performance_scores.items()
            }
        else:
            # Equal allocation fallback
            new_allocations = {
                route_id: 1.0 / len(self.profit_routes)
                for route_id in self.profit_routes.keys()
            }
        
        # Update allocations
        self.route_allocations = new_allocations
        self.last_rebalance = datetime.now(timezone.utc)
        
        # Suspend consistently failing routes
        for route_id, route in self.profit_routes.items():
            if (route.performance_level == RoutePerformanceLevel.FAILING and 
                route.total_trades >= 10):
                self.suspended_routes.add(route_id)
                route.active = False
                self.cli_handler.log_safe(logger, 'warning', f"⚠️ Suspended failing route: {route_id}")
        
        self.cli_handler.log_safe(logger, 'info', f"✅ Route rebalancing complete. New allocations: {new_allocations}")
    
    def _create_route_signal(self, base_signal: MathematicalTradeSignal,
                           route: ProfitRoute,
                           allocation: float,
                           routing_decision: ProfitRoutingDecision) -> MathematicalTradeSignal:
        """Create route-specific trade signal"""
        # Adjust position size for allocation
        adjusted_position_size = routing_decision.total_position_size * allocation
        
        # Adjust targets based on route type
        route_adjustments = {
            ProfitRouteType.MICRO_SCALP: {
                'stop_loss_multiplier': 0.5,
                'take_profit_multiplier': 0.5,
                'max_hold_time_multiplier': 0.2
            },
            ProfitRouteType.HARMONIC_SWING: {
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'max_hold_time_multiplier': 1.0
            },
            ProfitRouteType.STRATEGIC_HOLD: {
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 2.0,
                'max_hold_time_multiplier': 5.0
            },
            ProfitRouteType.DIVERSIFIED_BLEND: {
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 1.3,
                'max_hold_time_multiplier': 2.0
            }
        }
        
        adjustments = route_adjustments.get(route.route_type, route_adjustments[ProfitRouteType.HARMONIC_SWING])
        
        # Create route-specific signal
        from copy import deepcopy
        route_signal = deepcopy(base_signal)
        
        # Update signal parameters
        route_signal.signal_id = f"{base_signal.signal_id}_{route.route_id}"
        route_signal.position_size = adjusted_position_size
        route_signal.phase_gate = route.phase_gate
        
        # Adjust risk parameters
        if route_signal.stop_loss:
            current_price = base_signal.unified_analysis.data.get('price', 100.0)
            stop_distance = abs(current_price - route_signal.stop_loss)
            new_stop_distance = stop_distance * adjustments['stop_loss_multiplier']
            route_signal.stop_loss = (current_price - new_stop_distance 
                                    if base_signal.side == 'buy' 
                                    else current_price + new_stop_distance)
        
        if route_signal.take_profit:
            current_price = base_signal.unified_analysis.data.get('price', 100.0)
            profit_distance = abs(route_signal.take_profit - current_price)
            new_profit_distance = profit_distance * adjustments['take_profit_multiplier']
            route_signal.take_profit = (current_price + new_profit_distance 
                                      if base_signal.side == 'buy' 
                                      else current_price - new_profit_distance)
        
        if route_signal.max_hold_time:
            route_signal.max_hold_time = int(route_signal.max_hold_time * adjustments['max_hold_time_multiplier'])
        
        return route_signal
    
    # === SYSTEM MANAGEMENT AND REPORTING ===
    
    def get_profit_routing_summary(self) -> Dict[str, Any]:
        """Get comprehensive profit routing summary"""
        return {
            'routing_engine_status': {
                'optimization_mode': self.optimization_mode.value,
                'active_routes': len(self.active_routes),
                'suspended_routes': len(self.suspended_routes),
                'last_rebalance': self.last_rebalance.isoformat(),
                'total_routing_decisions': len(self.routing_history)
            },
            'performance_metrics': asdict(self.performance_metrics),
            'route_details': {
                route_id: {
                    'type': route.route_type.value,
                    'phase_gate': route.phase_gate,
                    'performance_level': route.performance_level.value,
                    'total_profit': route.total_profit,
                    'success_rate': route.success_rate,
                    'total_trades': route.total_trades,
                    'active': route.active,
                    'allocation': self.route_allocations.get(route_id, 0.0)
                }
                for route_id, route in self.profit_routes.items()
            },
            'recent_decisions': [
                {
                    'timestamp': entry['timestamp'].isoformat(),
                    'selected_routes': entry['decision'].selected_routes,
                    'expected_profit': entry['decision'].expected_profit,
                    'sustainment_index': entry['decision'].sustainment_index,
                    'confidence': entry['decision'].routing_confidence
                }
                for entry in list(self.routing_history)[-5:]  # Last 5 decisions
            ]
        }
    
    def set_optimization_mode(self, mode: ProfitOptimizationMode):
        """Set the profit optimization mode"""
        self.optimization_mode = mode
        logger.info(f"Profit optimization mode set to: {mode.value}")
    
    def activate_route(self, route_id: str):
        """Activate a specific route"""
        if route_id in self.profit_routes:
            self.profit_routes[route_id].active = True
            self.active_routes.add(route_id)
            self.suspended_routes.discard(route_id)
            logger.info(f"Route {route_id} activated")
    
    def suspend_route(self, route_id: str):
        """Suspend a specific route"""
        if route_id in self.profit_routes:
            self.profit_routes[route_id].active = False
            self.suspended_routes.add(route_id)
            self.active_routes.discard(route_id)
            logger.info(f"Route {route_id} suspended")
    
    def reset_route_performance(self, route_id: str):
        """Reset performance metrics for a specific route"""
        if route_id in self.profit_routes:
            route = self.profit_routes[route_id]
            route.total_profit = 0.0
            route.total_trades = 0
            route.success_rate = 0.0
            route.average_return = 0.0
            route.sharpe_ratio = 0.0
            route.graded_vectors = []
            route.sustainment_history = []
            route.performance_level = RoutePerformanceLevel.PERFORMING
            logger.info(f"Performance reset for route {route_id}")

# === INTEGRATION FUNCTION ===

def create_profit_routing_system(phase_controller: PhaseGateController,
                                execution_manager: CCXTExecutionManager) -> ProfitRoutingEngine:
    """
    Create a complete profit routing system integrated with existing components
    
    Args:
        phase_controller: Phase gate controller from Step 3
        execution_manager: CCXT execution manager from Step 2
        
    Returns:
        Fully integrated ProfitRoutingEngine
    """
    # Initialize mathematical components
    sustainment_lib = SustainmentMathLib()
    fitness_oracle = EnhancedFitnessOracle()
    bit_operations = BitOperations()
    
    # Create profit routing engine
    routing_engine = ProfitRoutingEngine(
        sustainment_lib=sustainment_lib,
        phase_controller=phase_controller,
        execution_manager=execution_manager,
        fitness_oracle=fitness_oracle,
        bit_operations=bit_operations
    )
    
    # Use cli_handler for safe logging
    cli_handler = WindowsCliCompatibilityHandler()
    cli_handler.log_safe(logger, 'info', "🚀 Profit routing system created with mathematical integration")
    return routing_engine

if __name__ == "__main__":
    # Basic testing
    print("Testing ProfitRoutingEngine...")
    
    # Create test components
    sustainment_lib = SustainmentMathLib()
    
    # Create routing engine
    routing_engine = ProfitRoutingEngine(sustainment_lib=sustainment_lib)
    
    print(f"Initialized with {len(routing_engine.profit_routes)} profit routes")
    print(f"Optimization mode: {routing_engine.optimization_mode.value}")
    
    # Get summary
    summary = routing_engine.get_profit_routing_summary()
    print(f"Active routes: {summary['routing_engine_status']['active_routes']}")
    print(f"Performance metrics initialized: {len(summary['performance_metrics'])}")
    
    print("✅ ProfitRoutingEngine basic test completed") 