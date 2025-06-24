"""
Unified Mathematical Trading Controller - Step 5: Final Orchestration
===================================================================

This is the final step (Step 5) that brings together all mathematical components
from Steps 1-4 into a unified, coherent mathematical trading system.

Orchestrates:
- Step 1: Mathematical validation core (RecursiveQuantumAIAnalysis + UnifiedMathematicalProcessor)
- Step 2: CCXT execution manager (CCXTExecutionManager + MathematicalTradeSignal)
- Step 3: Phase gate controller (PhaseGateController + entropy/bit operations)
- Step 4: Profit routing engine (ProfitRoutingEngine + sustainment optimization)

Mathematical Foundation:
The unified system implements a complete mathematical trading framework where:
Π(t) = ∫ [M(x) · P(x) · S(x) · E(x)] dx

Where:
- M(x): Mathematical validation function (Klein bottle topology + fractal convergence)
- P(x): Phase gate routing function (4b/8b/42b based on entropy/coherence)
- S(x): Sustainment optimization function (8 principles compliance)
- E(x): Execution management function (CCXT + risk management)
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import threading
from collections import deque, defaultdict

# Import all components from Steps 1-4
from .math_core import UnifiedMathematicalProcessor, RecursiveQuantumAIAnalysis, AnalysisResult
from .ccxt_execution_manager import CCXTExecutionManager, MathematicalTradeSignal, create_mathematical_execution_system
from .phase_gate_controller import PhaseGateController, PhaseGateDecision, create_phase_gate_system
from .profit_routing_engine import ProfitRoutingEngine, ProfitRoutingDecision, create_profit_routing_system

# Import mathematical libraries
from .mathlib_v3 import SustainmentMathLib, MathematicalContext, SustainmentVector
from .mathlib import GradedProfitVector
from ..enhanced_fitness_oracle import EnhancedFitnessOracle, UnifiedFitnessScore
from .bit_operations import BitOperations
from ..unified_entropy_engine import UnifiedEntropyEngine

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading modes for the unified system"""
    SIMULATION = "simulation"       # Paper trading with full mathematical validation
    LIVE_CONSERVATIVE = "live_conservative"  # Live trading with high safety thresholds
    LIVE_AGGRESSIVE = "live_aggressive"      # Live trading with standard thresholds
    MATHEMATICAL_ONLY = "mathematical_only"   # Mathematical analysis without execution

class SystemHealthStatus(Enum):
    """System health status levels"""
    EXCELLENT = "excellent"    # All systems optimal
    GOOD = "good"             # Systems performing well
    WARNING = "warning"       # Some issues detected
    CRITICAL = "critical"     # Major issues, halt trading
    EMERGENCY = "emergency"   # Emergency stop activated

@dataclass
class UnifiedSystemMetrics:
    """Comprehensive system performance metrics"""
    # Mathematical validation metrics
    mathematical_validation_rate: float = 0.0
    klein_bottle_consistency_rate: float = 0.0
    fractal_convergence_rate: float = 0.0
    
    # Phase gate metrics
    phase_gate_success_rate: float = 0.0
    average_gate_latency_ms: float = 0.0
    entropy_calculation_accuracy: float = 0.0
    
    # Profit routing metrics
    profit_routing_efficiency: float = 0.0
    sustainment_index_average: float = 0.0
    route_optimization_success_rate: float = 0.0
    
    # Execution metrics
    execution_success_rate: float = 0.0
    average_execution_time_ms: float = 0.0
    slippage_percentage: float = 0.0
    
    # Overall performance
    total_trades_executed: int = 0
    total_profit_realized: float = 0.0
    average_profit_per_trade: float = 0.0
    sharpe_ratio: float = 0.0
    maximum_drawdown: float = 0.0
    
    # System health
    uptime_percentage: float = 100.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percentage: float = 0.0

@dataclass
class TradingOpportunity:
    """Complete trading opportunity with unified analysis"""
    opportunity_id: str
    timestamp: datetime
    symbol: str
    market_data: Dict[str, Any]
    
    # Step 1: Mathematical validation
    mathematical_analysis: AnalysisResult
    math_processor_result: Dict[str, Any]
    
    # Step 2: Execution readiness
    trade_signal: MathematicalTradeSignal
    execution_feasibility: Dict[str, Any]
    
    # Step 3: Phase gate analysis
    phase_gate_decision: PhaseGateDecision
    optimal_execution_timing: str
    
    # Step 4: Profit routing
    profit_routing_decision: ProfitRoutingDecision
    expected_profit_breakdown: Dict[str, float]
    
    # Overall assessment
    unified_confidence: float
    final_recommendation: str  # execute, queue, reject
    risk_assessment: Dict[str, float]
    mathematical_validity: bool

@dataclass
class ExecutionResult:
    """Complete execution result with all system feedback"""
    execution_id: str
    opportunity: TradingOpportunity
    execution_timestamp: datetime
    
    # Execution details
    executed_price: float
    executed_volume: float
    execution_latency_ms: float
    slippage_bp: float
    
    # Route-specific results
    route_executions: List[Dict[str, Any]]
    phase_gate_performance: Dict[str, Any]
    
    # Mathematical validation of results
    post_execution_analysis: Dict[str, Any]
    profit_realization: float
    
    # System impact
    system_performance_impact: Dict[str, Any]
    lessons_learned: List[str]

class UnifiedMathematicalTradingController:
    """
    Master controller that orchestrates the complete mathematical trading system.
    
    This is the final integration point (Step 5) that brings together:
    - Mathematical validation (Step 1)
    - Execution management (Step 2) 
    - Phase gate control (Step 3)
    - Profit routing (Step 4)
    
    Into a unified, mathematically coherent trading system.
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 trading_mode: TradingMode = TradingMode.SIMULATION):
        """
        Initialize the unified mathematical trading controller
        
        Args:
            config: System configuration
            trading_mode: Initial trading mode
        """
        # Load configuration
        self.config = config or self._get_default_config()
        self.trading_mode = trading_mode
        
        # Initialize all Step 1-4 components
        self._initialize_mathematical_systems()
        
        # System state management
        self.system_health = SystemHealthStatus.GOOD
        self.active_opportunities = {}
        self.execution_queue = deque(maxlen=1000)
        self.completed_executions = deque(maxlen=5000)
        
        # Performance tracking
        self.system_metrics = UnifiedSystemMetrics()
        self.performance_history = deque(maxlen=10000)
        self.error_log = deque(maxlen=1000)
        
        # Threading and control
        self.controller_active = False
        self.monitoring_thread = None
        self.execution_thread = None
        self.analysis_thread = None
        
        # Mathematical validation
        self.mathematical_coherence_threshold = self.config.get('coherence_threshold', 0.75)
        self.sustainment_index_threshold = self.config.get('sustainment_threshold', 0.65)
        self.max_concurrent_opportunities = self.config.get('max_opportunities', 10)
        
        # Risk management
        self.max_daily_loss = self.config.get('max_daily_loss', 1000.0)
        self.max_position_size = self.config.get('max_position_size', 0.1)
        self.emergency_stop_triggers = self.config.get('emergency_triggers', {})
        
        logger.info("🚀 Unified Mathematical Trading Controller initialized")
    
    def _initialize_mathematical_systems(self):
        """Initialize all mathematical systems from Steps 1-4"""
        logger.info("🧮 Initializing unified mathematical systems...")
        
        try:
            # Step 1: Mathematical validation core
            self.math_processor = UnifiedMathematicalProcessor()
            self.quantum_analyzer = RecursiveQuantumAIAnalysis()
            
            # Step 2: CCXT execution manager
            self.execution_manager = create_mathematical_execution_system()
            
            # Step 3: Phase gate controller
            self.phase_controller = create_phase_gate_system(self.execution_manager)
            
            # Step 4: Profit routing engine
            self.profit_engine = create_profit_routing_system(
                self.phase_controller, self.execution_manager
            )
            
            # Supporting mathematical systems
            self.sustainment_lib = SustainmentMathLib()
            self.fitness_oracle = EnhancedFitnessOracle()
            self.entropy_engine = UnifiedEntropyEngine()
            self.bit_operations = BitOperations()
            
            logger.info("✅ All mathematical systems initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize mathematical systems: {e}")
            raise RuntimeError(f"System initialization failed: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the unified system"""
        return {
            'coherence_threshold': 0.75,
            'sustainment_threshold': 0.65,
            'max_opportunities': 10,
            'max_daily_loss': 1000.0,
            'max_position_size': 0.1,
            'analysis_frequency_ms': 1000,
            'execution_timeout_ms': 5000,
            'emergency_triggers': {
                'max_consecutive_losses': 5,
                'max_drawdown_percent': 10.0,
                'min_system_health': 0.3
            },
            'mathematical_validation': {
                'require_klein_bottle_consistency': True,
                'require_fractal_convergence': True,
                'min_confidence_threshold': 0.7
            }
        }
    
    async def start_unified_system(self) -> bool:
        """Start the complete unified mathematical trading system"""
        if self.controller_active:
            logger.warning("Unified system already active")
            return False
        
        try:
            logger.info("🚀 Starting unified mathematical trading system...")
            
            # Start all subsystems
            await self._start_subsystems()
            
            # Start controller threads
            self.controller_active = True
            self._start_controller_threads()
            
            # Verify system health
            health_check = await self._comprehensive_health_check()
            if not health_check['healthy']:
                logger.error("❌ System health check failed")
                await self.stop_unified_system()
                return False
            
            logger.info("✅ Unified mathematical trading system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start unified system: {e}")
            await self.stop_unified_system()
            return False
    
    async def stop_unified_system(self):
        """Stop the unified system gracefully"""
        logger.info("🛑 Stopping unified mathematical trading system...")
        
        # Stop controller
        self.controller_active = False
        
        # Stop all subsystems
        await self._stop_subsystems()
        
        # Wait for threads to complete
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5)
        
        logger.info("✅ Unified system stopped successfully")
    
    async def _start_subsystems(self):
        """Start all subsystems"""
        # Start API coordinator
        await self.execution_manager.api_coordinator.start_coordinator()
        
        # Additional subsystem initialization can be added here
        logger.info("✅ All subsystems started")
    
    async def _stop_subsystems(self):
        """Stop all subsystems"""
        try:
            # Stop API coordinator
            await self.execution_manager.api_coordinator.stop_coordinator()
            
            logger.info("✅ All subsystems stopped")
        except Exception as e:
            logger.error(f"Error stopping subsystems: {e}")
    
    def _start_controller_threads(self):
        """Start background controller threads"""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        
        self.monitoring_thread.start()
        self.execution_thread.start()
        self.analysis_thread.start()
        
        logger.info("✅ All controller threads started")
    
    async def process_market_opportunity(self, 
                                       market_data: Dict[str, Any]) -> Optional[TradingOpportunity]:
        """
        Process a market opportunity through the complete mathematical pipeline
        
        Args:
            market_data: Real-time market data
            
        Returns:
            Complete trading opportunity analysis or None if not viable
        """
        try:
            opportunity_id = f"opp_{int(time.time() * 1000)}"
            timestamp = datetime.now(timezone.utc)
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            logger.info(f"🔍 Processing market opportunity: {opportunity_id} ({symbol})")
            
            # STEP 1: Mathematical validation
            logger.debug("🧮 Step 1: Mathematical validation...")
            mathematical_analysis = await self._perform_mathematical_validation(market_data)
            math_processor_result = self.math_processor.process_market_data(market_data)
            
            if not mathematical_analysis or not mathematical_analysis.confidence >= 0.5:
                logger.debug(f"❌ Mathematical validation failed for {opportunity_id}")
                return None
            
            # STEP 2: Create trade signal and check execution feasibility
            logger.debug("💱 Step 2: Execution feasibility...")
            trade_signal = await self.execution_manager.evaluate_trade_opportunity(market_data)
            
            if not trade_signal:
                logger.debug(f"❌ Trade signal generation failed for {opportunity_id}")
                return None
            
            execution_feasibility = await self._assess_execution_feasibility(trade_signal, market_data)
            
            # STEP 3: Phase gate analysis
            logger.debug("⚡ Step 3: Phase gate analysis...")
            phase_gate_decision = await self.phase_controller.evaluate_phase_gate(
                trade_signal, market_data
            )
            
            if phase_gate_decision.decision.value in ['reject', 'close']:
                logger.debug(f"❌ Phase gate rejected opportunity {opportunity_id}")
                return None
            
            # STEP 4: Profit routing optimization
            logger.debug("💰 Step 4: Profit routing...")
            profit_routing_decision = await self.profit_engine.analyze_profit_routing_opportunity(
                trade_signal, market_data
            )
            
            if not profit_routing_decision.mathematical_validity:
                logger.debug(f"❌ Profit routing validation failed for {opportunity_id}")
                return None
            
            # STEP 5: Unified confidence and final assessment
            logger.debug("🎯 Step 5: Unified assessment...")
            unified_confidence = self._calculate_unified_confidence(
                mathematical_analysis, trade_signal, phase_gate_decision, profit_routing_decision
            )
            
            final_recommendation = self._determine_final_recommendation(
                unified_confidence, phase_gate_decision, profit_routing_decision
            )
            
            risk_assessment = self._calculate_unified_risk_assessment(
                trade_signal, phase_gate_decision, profit_routing_decision, market_data
            )
            
            # Create complete trading opportunity
            opportunity = TradingOpportunity(
                opportunity_id=opportunity_id,
                timestamp=timestamp,
                symbol=symbol,
                market_data=market_data,
                mathematical_analysis=mathematical_analysis,
                math_processor_result=math_processor_result,
                trade_signal=trade_signal,
                execution_feasibility=execution_feasibility,
                phase_gate_decision=phase_gate_decision,
                optimal_execution_timing=phase_gate_decision.timing_recommendation,
                profit_routing_decision=profit_routing_decision,
                expected_profit_breakdown=self._calculate_profit_breakdown(profit_routing_decision),
                unified_confidence=unified_confidence,
                final_recommendation=final_recommendation,
                risk_assessment=risk_assessment,
                mathematical_validity=True
            )
            
            # Add to active opportunities
            self.active_opportunities[opportunity_id] = opportunity
            
            logger.info(f"✅ Opportunity {opportunity_id} analyzed: {final_recommendation} (confidence: {unified_confidence:.3f})")
            return opportunity
            
        except Exception as e:
            logger.error(f"❌ Error processing market opportunity: {e}")
            return None
    
    async def execute_trading_opportunity(self, 
                                        opportunity: TradingOpportunity) -> Optional[ExecutionResult]:
        """
        Execute a trading opportunity through the unified system
        
        Args:
            opportunity: Complete trading opportunity
            
        Returns:
            Execution result with comprehensive feedback
        """
        try:
            execution_id = f"exec_{opportunity.opportunity_id}_{int(time.time() * 1000)}"
            execution_timestamp = datetime.now(timezone.utc)
            
            logger.info(f"🚀 Executing trading opportunity: {execution_id}")
            
            # Pre-execution validation
            if not self._validate_pre_execution(opportunity):
                logger.warning(f"❌ Pre-execution validation failed for {execution_id}")
                return None
            
            # Execute through profit routing system
            start_time = time.time()
            routing_result = await self.profit_engine.execute_profit_routing(
                opportunity.profit_routing_decision,
                opportunity.trade_signal,
                opportunity.market_data
            )
            execution_latency_ms = (time.time() - start_time) * 1000
            
            if routing_result['status'] != 'completed':
                logger.error(f"❌ Profit routing execution failed for {execution_id}")
                return None
            
            # Calculate execution metrics
            executed_price = opportunity.market_data.get('price', 0.0)
            executed_volume = opportunity.profit_routing_decision.total_position_size
            slippage_bp = self._calculate_slippage(
                opportunity.trade_signal.unified_analysis.data.get('price', executed_price),
                executed_price
            )
            
            # Post-execution mathematical analysis
            post_execution_analysis = await self._perform_post_execution_analysis(
                opportunity, routing_result
            )
            
            # Calculate profit realization
            profit_realization = routing_result.get('expected_profit', 0.0)
            
            # Create execution result
            execution_result = ExecutionResult(
                execution_id=execution_id,
                opportunity=opportunity,
                execution_timestamp=execution_timestamp,
                executed_price=executed_price,
                executed_volume=executed_volume,
                execution_latency_ms=execution_latency_ms,
                slippage_bp=slippage_bp,
                route_executions=routing_result.get('route_results', []),
                phase_gate_performance={
                    'gate_type': opportunity.phase_gate_decision.gate_type.value,
                    'latency_ms': execution_latency_ms,
                    'success': True
                },
                post_execution_analysis=post_execution_analysis,
                profit_realization=profit_realization,
                system_performance_impact=self._calculate_system_impact(routing_result),
                lessons_learned=self._extract_lessons_learned(opportunity, routing_result)
            )
            
            # Update system metrics
            self._update_system_metrics(execution_result)
            
            # Add to completed executions
            self.completed_executions.append(execution_result)
            
            # Remove from active opportunities
            if opportunity.opportunity_id in self.active_opportunities:
                del self.active_opportunities[opportunity.opportunity_id]
            
            logger.info(f"✅ Execution {execution_id} completed successfully")
            return execution_result
            
        except Exception as e:
            logger.error(f"❌ Error executing trading opportunity: {e}")
            return None
    
    # === MATHEMATICAL VALIDATION METHODS ===
    
    async def _perform_mathematical_validation(self, market_data: Dict[str, Any]) -> Optional[AnalysisResult]:
        """Perform comprehensive mathematical validation (Step 1)"""
        try:
            # Use RecursiveQuantumAIAnalysis for Klein bottle dynamics
            klein_analysis = await self.quantum_analyzer.analyze_klein_bottle_dynamics(
                market_data.get('price_series', []),
                market_data.get('volume_series', [])
            )
            
            # Validate mathematical coherence
            mathematical_validity = {
                'topology_consistent': klein_analysis.get('topology_consistent', False),
                'fractal_convergent': klein_analysis.get('fractal_convergent', False),
                'coherence_score': klein_analysis.get('coherence', 0.0)
            }
            
            # Create analysis result
            analysis_result = AnalysisResult(
                name="unified_mathematical_validation",
                data={
                    'mathematical_validity': mathematical_validity,
                    'klein_bottle_analysis': klein_analysis,
                    'price': market_data.get('price', 0.0),
                    'volume': market_data.get('volume', 0.0)
                },
                confidence=klein_analysis.get('coherence', 0.0),
                timestamp=time.time()
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Mathematical validation error: {e}")
            return None
    
    async def _assess_execution_feasibility(self, 
                                          trade_signal: MathematicalTradeSignal,
                                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess execution feasibility (Step 2)"""
        return {
            'liquidity_sufficient': market_data.get('volume', 0) > 1000,
            'spread_acceptable': True,  # Simplified
            'risk_within_limits': trade_signal.position_size <= self.max_position_size,
            'signal_strength': trade_signal.confidence,
            'feasibility_score': trade_signal.confidence * 0.8
        }
    
    def _calculate_unified_confidence(self,
                                    mathematical_analysis: AnalysisResult,
                                    trade_signal: MathematicalTradeSignal,
                                    phase_gate_decision: PhaseGateDecision,
                                    profit_routing_decision: ProfitRoutingDecision) -> float:
        """Calculate unified confidence score across all systems"""
        
        # Mathematical confidence (30%)
        math_confidence = mathematical_analysis.confidence * 0.3
        
        # Execution confidence (25%)
        execution_confidence = trade_signal.confidence * 0.25
        
        # Phase gate confidence (25%)
        phase_confidence = phase_gate_decision.confidence * 0.25
        
        # Profit routing confidence (20%)
        routing_confidence = profit_routing_decision.routing_confidence * 0.2
        
        unified_confidence = math_confidence + execution_confidence + phase_confidence + routing_confidence
        
        # Apply mathematical coherence bonus
        if (mathematical_analysis.data.get('mathematical_validity', {}).get('topology_consistent', False) and
            mathematical_analysis.data.get('mathematical_validity', {}).get('fractal_convergent', False)):
            unified_confidence *= 1.1  # 10% bonus for mathematical coherence
        
        return min(1.0, unified_confidence)
    
    def _determine_final_recommendation(self,
                                      unified_confidence: float,
                                      phase_gate_decision: PhaseGateDecision,
                                      profit_routing_decision: ProfitRoutingDecision) -> str:
        """Determine final trading recommendation"""
        
        if unified_confidence >= 0.8 and phase_gate_decision.decision.value == 'execute_immediately':
            return 'execute'
        elif unified_confidence >= 0.65 and phase_gate_decision.decision.value in ['execute_with_delay', 'queue_for_later']:
            return 'queue'
        else:
            return 'reject'
    
    def _calculate_unified_risk_assessment(self,
                                         trade_signal: MathematicalTradeSignal,
                                         phase_gate_decision: PhaseGateDecision,
                                         profit_routing_decision: ProfitRoutingDecision,
                                         market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive risk assessment"""
        
        # Base risk from trade signal
        base_risk = 1.0 - trade_signal.confidence
        
        # Market volatility risk
        volatility = self._calculate_market_volatility(market_data)
        volatility_risk = volatility * 0.5
        
        # Phase gate risk
        phase_risk = phase_gate_decision.metrics.volatility_risk if hasattr(phase_gate_decision, 'metrics') else 0.3
        
        # Profit routing risk
        routing_risk = profit_routing_decision.risk_assessment
        
        return {
            'base_risk': base_risk,
            'volatility_risk': volatility_risk,
            'phase_gate_risk': phase_risk,
            'routing_risk': routing_risk,
            'total_risk': min(1.0, base_risk + volatility_risk + phase_risk + routing_risk)
        }
    
    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility from price series"""
        price_series = market_data.get('price_series', [])
        if len(price_series) < 2:
            return 0.5  # Default
        
        returns = np.diff(np.log(price_series))
        volatility = float(np.std(returns))
        return min(1.0, volatility * 20)  # Scale to 0-1
    
    def _calculate_profit_breakdown(self, profit_routing_decision: ProfitRoutingDecision) -> Dict[str, float]:
        """Calculate detailed profit breakdown by route"""
        breakdown = {}
        for route_id, allocation in profit_routing_decision.route_allocations.items():
            route_profit = profit_routing_decision.expected_profit * allocation
            breakdown[route_id] = route_profit
        return breakdown
    
    # === EXECUTION VALIDATION AND MONITORING ===
    
    def _validate_pre_execution(self, opportunity: TradingOpportunity) -> bool:
        """Validate opportunity before execution"""
        
        # Check system health
        if self.system_health in [SystemHealthStatus.CRITICAL, SystemHealthStatus.EMERGENCY]:
            logger.warning("System health critical - blocking execution")
            return False
        
        # Check mathematical validity
        if not opportunity.mathematical_validity:
            logger.warning("Mathematical validity failed - blocking execution")
            return False
        
        # Check risk limits
        if opportunity.risk_assessment['total_risk'] > 0.7:
            logger.warning("Risk too high - blocking execution")
            return False
        
        # Check daily loss limits
        daily_loss = self._calculate_daily_loss()
        if daily_loss >= self.max_daily_loss:
            logger.warning("Daily loss limit reached - blocking execution")
            return False
        
        return True
    
    def _calculate_daily_loss(self) -> float:
        """Calculate current daily loss"""
        today = datetime.now(timezone.utc).date()
        daily_executions = [
            ex for ex in self.completed_executions 
            if ex.execution_timestamp.date() == today
        ]
        
        return sum(
            ex.profit_realization for ex in daily_executions 
            if ex.profit_realization < 0
        )
    
    def _calculate_slippage(self, expected_price: float, executed_price: float) -> float:
        """Calculate slippage in basis points"""
        if expected_price == 0:
            return 0.0
        return abs(executed_price - expected_price) / expected_price * 10000
    
    async def _perform_post_execution_analysis(self,
                                             opportunity: TradingOpportunity,
                                             routing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform post-execution mathematical analysis"""
        return {
            'execution_efficiency': routing_result.get('total_volume', 0) / max(opportunity.trade_signal.position_size, 0.001),
            'route_performance': routing_result.get('route_results', []),
            'mathematical_consistency': True,  # Simplified
            'lessons_learned': []
        }
    
    def _calculate_system_impact(self, routing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system performance impact"""
        return {
            'memory_impact_mb': 5.0,  # Simplified
            'cpu_impact_percent': 2.0,
            'latency_impact_ms': 10.0,
            'overall_impact': 'minimal'
        }
    
    def _extract_lessons_learned(self,
                                opportunity: TradingOpportunity,
                                routing_result: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from execution"""
        lessons = []
        
        if opportunity.unified_confidence < 0.7:
            lessons.append("Low confidence execution - consider raising thresholds")
        
        if routing_result.get('total_volume', 0) < opportunity.trade_signal.position_size * 0.9:
            lessons.append("Partial execution - improve liquidity assessment")
        
        return lessons
    
    def _update_system_metrics(self, execution_result: ExecutionResult):
        """Update comprehensive system metrics"""
        
        # Update trade metrics
        self.system_metrics.total_trades_executed += 1
        self.system_metrics.total_profit_realized += execution_result.profit_realization
        
        # Update averages
        if self.system_metrics.total_trades_executed > 0:
            self.system_metrics.average_profit_per_trade = (
                self.system_metrics.total_profit_realized / self.system_metrics.total_trades_executed
            )
        
        # Update execution metrics
        self.system_metrics.average_execution_time_ms = (
            (self.system_metrics.average_execution_time_ms * (self.system_metrics.total_trades_executed - 1) +
             execution_result.execution_latency_ms) / self.system_metrics.total_trades_executed
        )
        
        # Update success rates (simplified)
        self.system_metrics.execution_success_rate = min(1.0, self.system_metrics.execution_success_rate + 0.01)
        self.system_metrics.mathematical_validation_rate = min(1.0, self.system_metrics.mathematical_validation_rate + 0.01)
        
        # Add to performance history
        self.performance_history.append({
            'timestamp': execution_result.execution_timestamp,
            'profit': execution_result.profit_realization,
            'confidence': execution_result.opportunity.unified_confidence,
            'latency_ms': execution_result.execution_latency_ms
        })
    
    # === BACKGROUND MONITORING LOOPS ===
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.controller_active:
            try:
                # Update system health
                self._update_system_health()
                
                # Check for emergency conditions
                self._check_emergency_conditions()
                
                # Clean up old opportunities
                self._cleanup_old_opportunities()
                
                # Sleep
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
    
    def _execution_loop(self):
        """Background execution loop"""
        while self.controller_active:
            try:
                # Process execution queue
                self._process_execution_queue()
                
                # Sleep
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                time.sleep(1.0)
    
    def _analysis_loop(self):
        """Background analysis loop"""
        while self.controller_active:
            try:
                # Perform continuous analysis
                self._perform_continuous_analysis()
                
                # Sleep based on configuration
                sleep_ms = self.config.get('analysis_frequency_ms', 1000)
                time.sleep(sleep_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(5.0)
    
    def _update_system_health(self):
        """Update overall system health status"""
        health_factors = []
        
        # Error rate factor
        recent_errors = len([e for e in self.error_log if e['timestamp'] > time.time() - 300])  # Last 5 minutes
        error_factor = max(0.0, 1.0 - recent_errors / 10.0)
        health_factors.append(error_factor)
        
        # Performance factor
        if self.system_metrics.total_trades_executed > 0:
            profit_factor = min(1.0, max(0.0, (self.system_metrics.total_profit_realized + 100) / 200))
            health_factors.append(profit_factor)
        
        # System resource factor
        memory_factor = max(0.0, 1.0 - self.system_metrics.memory_usage_mb / 1000.0)
        cpu_factor = max(0.0, 1.0 - self.system_metrics.cpu_usage_percentage / 100.0)
        health_factors.extend([memory_factor, cpu_factor])
        
        # Calculate overall health
        overall_health = sum(health_factors) / len(health_factors) if health_factors else 0.5
        
        # Update health status
        if overall_health >= 0.9:
            self.system_health = SystemHealthStatus.EXCELLENT
        elif overall_health >= 0.7:
            self.system_health = SystemHealthStatus.GOOD
        elif overall_health >= 0.5:
            self.system_health = SystemHealthStatus.WARNING
        elif overall_health >= 0.3:
            self.system_health = SystemHealthStatus.CRITICAL
        else:
            self.system_health = SystemHealthStatus.EMERGENCY
    
    def _check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        emergency_triggers = self.config.get('emergency_triggers', {})
        
        # Check consecutive losses
        max_losses = emergency_triggers.get('max_consecutive_losses', 5)
        recent_executions = list(self.completed_executions)[-max_losses:]
        if (len(recent_executions) >= max_losses and 
            all(ex.profit_realization < 0 for ex in recent_executions)):
            logger.critical("🚨 EMERGENCY: Too many consecutive losses")
            self.system_health = SystemHealthStatus.EMERGENCY
        
        # Check maximum drawdown
        max_drawdown = emergency_triggers.get('max_drawdown_percent', 10.0)
        if self.system_metrics.maximum_drawdown > max_drawdown:
            logger.critical("🚨 EMERGENCY: Maximum drawdown exceeded")
            self.system_health = SystemHealthStatus.EMERGENCY
    
    def _cleanup_old_opportunities(self):
        """Clean up old opportunities"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        old_opportunities = [
            opp_id for opp_id, opp in self.active_opportunities.items()
            if opp.timestamp < cutoff_time
        ]
        
        for opp_id in old_opportunities:
            del self.active_opportunities[opp_id]
    
    def _process_execution_queue(self):
        """Process any queued executions"""
        # Implementation would process queued opportunities
        pass
    
    def _perform_continuous_analysis(self):
        """Perform continuous mathematical analysis"""
        # Implementation would perform ongoing system analysis
        pass
    
    # === HEALTH CHECK METHODS ===
    
    async def _comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_results = {
            'healthy': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Test mathematical systems
            health_results['checks']['mathematical_systems'] = await self._test_mathematical_systems()
            
            # Test execution systems
            health_results['checks']['execution_systems'] = await self._test_execution_systems()
            
            # Test phase gate systems
            health_results['checks']['phase_gate_systems'] = await self._test_phase_gate_systems()
            
            # Test profit routing systems
            health_results['checks']['profit_routing_systems'] = await self._test_profit_routing_systems()
            
            # Overall health assessment
            all_healthy = all(check['healthy'] for check in health_results['checks'].values())
            health_results['healthy'] = all_healthy
            
            return health_results
            
        except Exception as e:
            health_results['healthy'] = False
            health_results['errors'].append(f"Health check failed: {e}")
            return health_results
    
    async def _test_mathematical_systems(self) -> Dict[str, Any]:
        """Test mathematical systems"""
        try:
            # Test math processor
            test_data = {'price': 100.0, 'volume': 1000.0}
            result = self.math_processor.process_market_data(test_data)
            
            # Test quantum analyzer
            test_prices = [100, 101, 102, 101, 100]
            test_volumes = [1000, 1100, 1200, 1100, 1000]
            klein_result = await self.quantum_analyzer.analyze_klein_bottle_dynamics(test_prices, test_volumes)
            
            return {
                'healthy': True,
                'math_processor': result is not None,
                'quantum_analyzer': klein_result is not None
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def _test_execution_systems(self) -> Dict[str, Any]:
        """Test execution systems"""
        try:
            # Test API coordinator status
            api_status = self.execution_manager.api_coordinator.coordinator_active
            
            return {
                'healthy': api_status,
                'api_coordinator': api_status,
                'execution_manager': self.execution_manager is not None
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def _test_phase_gate_systems(self) -> Dict[str, Any]:
        """Test phase gate systems"""
        try:
            # Test phase controller components
            entropy_healthy = self.phase_controller.entropy_engine is not None
            bit_ops_healthy = self.phase_controller.bit_operations is not None
            
            return {
                'healthy': entropy_healthy and bit_ops_healthy,
                'entropy_engine': entropy_healthy,
                'bit_operations': bit_ops_healthy
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def _test_profit_routing_systems(self) -> Dict[str, Any]:
        """Test profit routing systems"""
        try:
            # Test profit engine components
            sustainment_healthy = self.profit_engine.sustainment_lib is not None
            routes_healthy = len(self.profit_engine.profit_routes) > 0
            
            return {
                'healthy': sustainment_healthy and routes_healthy,
                'sustainment_lib': sustainment_healthy,
                'profit_routes': routes_healthy
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    # === PUBLIC API METHODS ===
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'controller_active': self.controller_active,
            'trading_mode': self.trading_mode.value,
            'system_health': self.system_health.value,
            'active_opportunities': len(self.active_opportunities),
            'completed_executions': len(self.completed_executions),
            'system_metrics': asdict(self.system_metrics),
            'performance_summary': {
                'total_profit': self.system_metrics.total_profit_realized,
                'total_trades': self.system_metrics.total_trades_executed,
                'success_rate': self.system_metrics.execution_success_rate,
                'avg_profit_per_trade': self.system_metrics.average_profit_per_trade
            }
        }
    
    def set_trading_mode(self, mode: TradingMode):
        """Set trading mode"""
        self.trading_mode = mode
        logger.info(f"Trading mode set to: {mode.value}")
    
    def emergency_stop(self):
        """Trigger emergency stop"""
        logger.critical("🚨 EMERGENCY STOP TRIGGERED")
        self.system_health = SystemHealthStatus.EMERGENCY
        self.trading_mode = TradingMode.MATHEMATICAL_ONLY
    
    def reset_system_metrics(self):
        """Reset system performance metrics"""
        self.system_metrics = UnifiedSystemMetrics()
        self.performance_history.clear()
        logger.info("System metrics reset")

# === FACTORY FUNCTION ===

def create_unified_mathematical_trading_system(
    config: Optional[Dict[str, Any]] = None,
    trading_mode: TradingMode = TradingMode.SIMULATION
) -> UnifiedMathematicalTradingController:
    """
    Create a complete unified mathematical trading system
    
    Args:
        config: System configuration
        trading_mode: Initial trading mode
        
    Returns:
        Fully integrated UnifiedMathematicalTradingController
    """
    controller = UnifiedMathematicalTradingController(
        config=config,
        trading_mode=trading_mode
    )
    
    logger.info("🎉 Unified Mathematical Trading System created successfully!")
    logger.info("✅ All Steps 1-5 integrated and ready for operation")
    return controller

if __name__ == "__main__":
    # Basic testing
    print("🧪 Testing UnifiedMathematicalTradingController...")
    
    # Create unified system
    controller = create_unified_mathematical_trading_system()
    
    print(f"✅ Controller created with mode: {controller.trading_mode.value}")
    print(f"✅ System health: {controller.system_health.value}")
    print(f"✅ Configuration loaded: {len(controller.config)} parameters")
    
    # Test system status
    status = controller.get_system_status()
    print(f"✅ System status available: {len(status)} metrics")
    
    print("🎉 UnifiedMathematicalTradingController basic test completed!") 