"""
Phase Gate Controller
====================

Integrates your existing entropy, coherence, and bit-level analysis systems
to control trading execution phases based on mathematical validation.

This connects:
- UnifiedEntropyEngine calculations
- BitOperations phase extraction (4b, 8b, 42b)
- Mathematical coherence analysis
- CCXTExecutionManager trade signals
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Import your existing mathematical systems
from .entropy_engine import UnifiedEntropyEngine, EntropyConfig
from .bit_operations import BitOperations, PhaseState
from .math_core import UnifiedMathematicalProcessor, AnalysisResult
from .ccxt_execution_manager import CCXTExecutionManager, MathematicalTradeSignal
from ..enhanced_fitness_oracle import EnhancedFitnessOracle, UnifiedFitnessScore

logger = logging.getLogger(__name__)

class PhaseGateType(Enum):
    """Phase gate types based on mathematical analysis"""
    MICRO_4B = "4b"      # Fast micro trades (low entropy, high convergence)
    HARMONIC_8B = "8b"   # Harmonic trades (medium entropy, good convergence)
    STRATEGIC_42B = "42b" # Strategic long-term (high entropy, emerging patterns)

class PhaseGateStatus(Enum):
    """Phase gate execution status"""
    OPEN = "open"           # Gate is open for execution
    CLOSED = "closed"       # Gate is closed (mathematical rejection)
    THROTTLED = "throttled" # Gate is throttled (risk management)
    PAUSED = "paused"       # Gate is paused (system maintenance)

class GateDecision(Enum):
    """Phase gate decisions"""
    EXECUTE_IMMEDIATELY = "execute_immediately"
    EXECUTE_WITH_DELAY = "execute_with_delay"
    QUEUE_FOR_LATER = "queue_for_later"
    REJECT = "reject"

@dataclass
class PhaseGateMetrics:
    """Metrics for phase gate analysis"""
    entropy_score: float
    coherence_score: float
    bit_density: float
    pattern_strength: float
    convergence_rate: float
    mathematical_validity: bool
    
    # Phase-specific metrics
    micro_suitability: float   # 0.0-1.0 for 4b suitability
    harmonic_suitability: float # 0.0-1.0 for 8b suitability
    strategic_suitability: float # 0.0-1.0 for 42b suitability
    
    # Risk metrics
    volatility_risk: float
    liquidity_risk: float
    timing_risk: float

@dataclass
class PhaseGateDecision:
    """Decision from phase gate analysis"""
    gate_type: PhaseGateType
    decision: GateDecision
    confidence: float
    delay_seconds: float = 0.0
    
    # Supporting data
    metrics: PhaseGateMetrics = None
    reasoning: str = ""
    mathematical_validation: Dict = None
    
    # Execution parameters
    position_size_adjustment: float = 1.0  # Multiplier for position size
    urgency_level: float = 0.5  # 0.0-1.0 urgency
    risk_adjustment: float = 1.0  # Risk multiplier

class PhaseGateController:
    """
    Controls trading execution phases based on unified mathematical analysis.
    Integrates entropy, coherence, bit operations, and execution management.
    """
    
    def __init__(self,
                 entropy_engine: Optional[UnifiedEntropyEngine] = None,
                 bit_operations: Optional[BitOperations] = None,
                 math_processor: Optional[UnifiedMathematicalProcessor] = None,
                 execution_manager: Optional[CCXTExecutionManager] = None,
                 fitness_oracle: Optional[EnhancedFitnessOracle] = None):
        """
        Initialize phase gate controller with mathematical components
        
        Args:
            entropy_engine: Your existing UnifiedEntropyEngine
            bit_operations: Your existing BitOperations
            math_processor: Your existing UnifiedMathematicalProcessor
            execution_manager: CCXT execution manager from Step 2
            fitness_oracle: Your existing EnhancedFitnessOracle
        """
        # Initialize your existing mathematical systems
        self.entropy_engine = entropy_engine or UnifiedEntropyEngine()
        self.bit_operations = bit_operations or BitOperations()
        self.math_processor = math_processor or UnifiedMathematicalProcessor()
        self.execution_manager = execution_manager
        self.fitness_oracle = fitness_oracle or EnhancedFitnessOracle()
        
        # Phase gate configuration
        self.gate_config = {
            PhaseGateType.MICRO_4B: {
                'entropy_range': (0.0, 0.3),
                'convergence_min': 0.8,
                'density_range': (0.2, 0.6),
                'execution_delay': 0.1,  # 100ms delay
                'position_size_max': 0.05,  # 5% max position
                'min_confidence': 0.85
            },
            PhaseGateType.HARMONIC_8B: {
                'entropy_range': (0.3, 0.7),
                'convergence_min': 0.5,
                'density_range': (0.4, 0.8),
                'execution_delay': 1.0,  # 1 second delay
                'position_size_max': 0.15,  # 15% max position
                'min_confidence': 0.70
            },
            PhaseGateType.STRATEGIC_42B: {
                'entropy_range': (0.7, 1.0),
                'convergence_min': 0.3,
                'density_range': (0.6, 1.0),
                'execution_delay': 5.0,  # 5 second delay
                'position_size_max': 0.25,  # 25% max position
                'min_confidence': 0.60
            }
        }
        
        # Gate status tracking
        self.gate_status = {
            PhaseGateType.MICRO_4B: PhaseGateStatus.OPEN,
            PhaseGateType.HARMONIC_8B: PhaseGateStatus.OPEN,
            PhaseGateType.STRATEGIC_42B: PhaseGateStatus.OPEN
        }
        
        # Performance tracking
        self.gate_stats = {
            'total_evaluations': 0,
            'gate_decisions': {
                PhaseGateType.MICRO_4B: {'open': 0, 'closed': 0, 'executed': 0},
                PhaseGateType.HARMONIC_8B: {'open': 0, 'closed': 0, 'executed': 0},
                PhaseGateType.STRATEGIC_42B: {'open': 0, 'closed': 0, 'executed': 0}
            },
            'mathematical_validations': 0,
            'coherence_rejections': 0,
            'entropy_rejections': 0
        }
        
        # Queue for delayed executions
        self.delayed_execution_queue = []
        
        logger.info("PhaseGateController initialized with mathematical foundation")
    
    async def evaluate_phase_gate(self, 
                                 trade_signal: MathematicalTradeSignal,
                                 market_data: Dict[str, Any]) -> PhaseGateDecision:
        """
        Evaluate which phase gate to use and whether execution should proceed
        
        Args:
            trade_signal: Mathematical trade signal from Step 2
            market_data: Current market data
            
        Returns:
            PhaseGateDecision with gate type and execution decision
        """
        self.gate_stats['total_evaluations'] += 1
        
        try:
            # STEP 1: Calculate unified entropy
            logger.info("🌀 Calculating unified entropy...")
            entropy_context = {
                'price_data': market_data.get('price_series', []),
                'volume_data': market_data.get('volume_series', []),
                'timestamp': time.time()
            }
            
            entropy_result = await self.entropy_engine.request_entropy(
                context=entropy_context,
                method="wavelet"
            )
            entropy_score = entropy_result.get('entropy', 0.5)
            
            # STEP 2: Perform bit operations analysis
            logger.info("🔢 Performing bit operations analysis...")
            bit_pattern = self.bit_operations.calculate_42bit_float(entropy_score)
            phase_state = self.bit_operations.create_phase_state(bit_pattern, entropy_result)
            bit_analysis = self.bit_operations.analyze_bit_pattern(bit_pattern)
            
            # STEP 3: Calculate coherence from mathematical analysis
            logger.info("🧮 Calculating mathematical coherence...")
            coherence_score = trade_signal.coherence_score
            
            # Additional coherence from unified analysis
            if hasattr(trade_signal.unified_analysis.data, 'mathematical_validity'):
                math_validity = trade_signal.unified_analysis.data.get('mathematical_validity', {})
                topology_coherence = 1.0 if math_validity.get('topology_consistent', False) else 0.5
                fractal_coherence = 1.0 if math_validity.get('fractal_convergent', False) else 0.6
                coherence_score = float(np.mean([coherence_score, topology_coherence, fractal_coherence]))
            
            # STEP 4: Create comprehensive metrics
            metrics = PhaseGateMetrics(
                entropy_score=entropy_score,
                coherence_score=coherence_score,
                bit_density=phase_state.density,
                pattern_strength=bit_analysis.get('pattern_strength', 0.0),
                convergence_rate=trade_signal.unified_analysis.data.get('fractal_convergence', {}).get('convergence_rate', 0.5),
                mathematical_validity=trade_signal.mathematical_validity,
                micro_suitability=self._calculate_micro_suitability(entropy_score, coherence_score, phase_state),
                harmonic_suitability=self._calculate_harmonic_suitability(entropy_score, coherence_score, phase_state),
                strategic_suitability=self._calculate_strategic_suitability(entropy_score, coherence_score, phase_state),
                volatility_risk=self._calculate_volatility_risk(market_data),
                liquidity_risk=self._calculate_liquidity_risk(market_data),
                timing_risk=self._calculate_timing_risk(trade_signal, entropy_score)
            )
            
            # STEP 5: Determine optimal phase gate
            optimal_gate = self._determine_optimal_gate(metrics)
            
            # STEP 6: Make gate decision
            gate_decision = self._make_gate_decision(optimal_gate, metrics, trade_signal)
            
            # STEP 7: Update statistics
            self._update_gate_stats(optimal_gate, gate_decision)
            
            logger.info(f"✅ Phase gate evaluation complete: {optimal_gate.value} -> {gate_decision.decision.value}")
            return gate_decision
            
        except Exception as e:
            logger.error(f"Error evaluating phase gate: {e}")
            # Return safe fallback decision
            return PhaseGateDecision(
                gate_type=PhaseGateType.HARMONIC_8B,
                decision=GateDecision.REJECT,
                confidence=0.0,
                reasoning=f"Error in phase gate evaluation: {str(e)}"
            )
    
    async def execute_through_phase_gate(self, 
                                        trade_signal: MathematicalTradeSignal,
                                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade through appropriate phase gate with mathematical validation
        
        Args:
            trade_signal: Mathematical trade signal
            market_data: Current market data
            
        Returns:
            Execution result with phase gate information
        """
        try:
            # Evaluate phase gate
            gate_decision = await self.evaluate_phase_gate(trade_signal, market_data)
            
            # Check if execution should proceed
            if gate_decision.decision == GateDecision.REJECT:
                return {
                    'status': 'rejected',
                    'reason': gate_decision.reasoning,
                    'gate_type': gate_decision.gate_type.value,
                    'confidence': gate_decision.confidence
                }
            
            # Apply phase gate adjustments to trade signal
            adjusted_signal = self._apply_phase_gate_adjustments(trade_signal, gate_decision)
            
            # Handle execution based on decision
            if gate_decision.decision == GateDecision.EXECUTE_IMMEDIATELY:
                if self.execution_manager:
                    execution_result = await self.execution_manager.execute_signal(adjusted_signal)
                    return {
                        'status': 'executed',
                        'execution_result': execution_result,
                        'gate_type': gate_decision.gate_type.value,
                        'confidence': gate_decision.confidence,
                        'phase_metrics': gate_decision.metrics
                    }
                else:
                    return {
                        'status': 'simulated',
                        'reason': 'No execution manager available',
                        'gate_type': gate_decision.gate_type.value,
                        'adjusted_signal': adjusted_signal
                    }
            
            elif gate_decision.decision == GateDecision.EXECUTE_WITH_DELAY:
                # Schedule delayed execution
                await self._schedule_delayed_execution(adjusted_signal, gate_decision.delay_seconds)
                return {
                    'status': 'scheduled',
                    'delay_seconds': gate_decision.delay_seconds,
                    'gate_type': gate_decision.gate_type.value,
                    'confidence': gate_decision.confidence
                }
            
            elif gate_decision.decision == GateDecision.QUEUE_FOR_LATER:
                # Add to queue for later processing
                self.delayed_execution_queue.append({
                    'signal': adjusted_signal,
                    'gate_decision': gate_decision,
                    'timestamp': datetime.now(timezone.utc)
                })
                return {
                    'status': 'queued',
                    'gate_type': gate_decision.gate_type.value,
                    'queue_position': len(self.delayed_execution_queue)
                }
            
        except Exception as e:
            logger.error(f"Error executing through phase gate: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'gate_type': 'unknown'
            }
    
    def _calculate_micro_suitability(self, entropy: float, coherence: float, phase_state: PhaseState) -> float:
        """Calculate suitability for 4-bit micro phase gate"""
        config = self.gate_config[PhaseGateType.MICRO_4B]
        
        # Entropy suitability (lower is better for micro)
        entropy_suit = 1.0 - max(0, min(1, (entropy - config['entropy_range'][0]) / 
                                       (config['entropy_range'][1] - config['entropy_range'][0])))
        
        # Coherence suitability (higher is better)
        coherence_suit = coherence
        
        # Density suitability
        density_suit = 1.0 if config['density_range'][0] <= phase_state.density <= config['density_range'][1] else 0.5
        
        # Variance suitability (lower variance is better for micro)
        variance_suit = 1.0 - min(1.0, phase_state.variance_short)
        
        return float(np.mean([entropy_suit, coherence_suit, density_suit, variance_suit]))
    
    def _calculate_harmonic_suitability(self, entropy: float, coherence: float, phase_state: PhaseState) -> float:
        """Calculate suitability for 8-bit harmonic phase gate"""
        config = self.gate_config[PhaseGateType.HARMONIC_8B]
        
        # Entropy suitability (medium range is optimal)
        entropy_in_range = config['entropy_range'][0] <= entropy <= config['entropy_range'][1]
        entropy_suit = 1.0 if entropy_in_range else 0.6
        
        # Coherence suitability
        coherence_suit = coherence
        
        # Density suitability
        density_suit = 1.0 if config['density_range'][0] <= phase_state.density <= config['density_range'][1] else 0.5
        
        # Balanced variance (moderate variance is good for harmonic)
        variance_suit = 1.0 - abs(phase_state.variance_mid - 0.5)
        
        return float(np.mean([entropy_suit, coherence_suit, density_suit, variance_suit]))
    
    def _calculate_strategic_suitability(self, entropy: float, coherence: float, phase_state: PhaseState) -> float:
        """Calculate suitability for 42-bit strategic phase gate"""
        config = self.gate_config[PhaseGateType.STRATEGIC_42B]
        
        # Entropy suitability (higher is better for strategic)
        entropy_suit = max(0, min(1, (entropy - config['entropy_range'][0]) / 
                                 (config['entropy_range'][1] - config['entropy_range'][0])))
        
        # Coherence suitability (still important but less critical)
        coherence_suit = max(0.3, coherence)
        
        # Density suitability
        density_suit = 1.0 if config['density_range'][0] <= phase_state.density <= config['density_range'][1] else 0.5
        
        # Higher variance is acceptable for strategic
        variance_suit = min(1.0, phase_state.variance_long + 0.3)
        
        return float(np.mean([entropy_suit, coherence_suit, density_suit, variance_suit]))
    
    def _calculate_volatility_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate volatility-based risk"""
        prices = market_data.get('price_series', [])
        if len(prices) < 2:
            return 0.5
        
        returns = np.diff(np.log(prices))
        volatility = float(np.std(returns))
        
        # Normalize volatility to 0-1 risk scale
        return min(1.0, volatility * 50)  # Arbitrary scaling factor
    
    def _calculate_liquidity_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate liquidity-based risk"""
        volume = market_data.get('volume', 1000.0)
        avg_volume = np.mean(market_data.get('volume_series', [volume]))
        
        # Lower volume = higher risk
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        liquidity_risk = max(0.0, 1.0 - volume_ratio)
        
        return min(1.0, liquidity_risk)
    
    def _calculate_timing_risk(self, trade_signal: MathematicalTradeSignal, entropy: float) -> float:
        """Calculate timing-based risk"""
        # Higher entropy = higher timing risk for immediate execution
        entropy_risk = entropy
        
        # Confidence risk (lower confidence = higher timing risk)
        confidence_risk = 1.0 - trade_signal.confidence
        
        # Risk level factor
        risk_multipliers = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 1.0
        }
        risk_level_risk = risk_multipliers.get(trade_signal.risk_level.value, 0.5)
        
        return float(np.mean([entropy_risk, confidence_risk, risk_level_risk]))
    
    def _determine_optimal_gate(self, metrics: PhaseGateMetrics) -> PhaseGateType:
        """Determine optimal phase gate based on metrics"""
        suitability_scores = {
            PhaseGateType.MICRO_4B: metrics.micro_suitability,
            PhaseGateType.HARMONIC_8B: metrics.harmonic_suitability,
            PhaseGateType.STRATEGIC_42B: metrics.strategic_suitability
        }
        
        # Find gate with highest suitability
        optimal_gate = max(suitability_scores.items(), key=lambda x: x[1])[0]
        
        # Check if gate is open
        if self.gate_status[optimal_gate] != PhaseGateStatus.OPEN:
            # Fall back to harmonic gate if available
            if self.gate_status[PhaseGateType.HARMONIC_8B] == PhaseGateStatus.OPEN:
                return PhaseGateType.HARMONIC_8B
            # Otherwise use the optimal gate anyway (may be rejected later)
        
        return optimal_gate
    
    def _make_gate_decision(self, 
                          gate_type: PhaseGateType, 
                          metrics: PhaseGateMetrics,
                          trade_signal: MathematicalTradeSignal) -> PhaseGateDecision:
        """Make execution decision for the selected gate"""
        config = self.gate_config[gate_type]
        
        # Check mathematical validity
        if not metrics.mathematical_validity:
            self.gate_stats['mathematical_validations'] += 1
            return PhaseGateDecision(
                gate_type=gate_type,
                decision=GateDecision.REJECT,
                confidence=0.0,
                reasoning="Mathematical validation failed"
            )
        
        # Check coherence threshold
        if metrics.coherence_score < 0.4:
            self.gate_stats['coherence_rejections'] += 1
            return PhaseGateDecision(
                gate_type=gate_type,
                decision=GateDecision.REJECT,
                confidence=metrics.coherence_score,
                reasoning=f"Coherence too low: {metrics.coherence_score:.3f}"
            )
        
        # Check entropy range
        entropy_in_range = config['entropy_range'][0] <= metrics.entropy_score <= config['entropy_range'][1]
        if not entropy_in_range and gate_type != PhaseGateType.HARMONIC_8B:  # Harmonic is more flexible
            self.gate_stats['entropy_rejections'] += 1
            return PhaseGateDecision(
                gate_type=gate_type,
                decision=GateDecision.REJECT,
                confidence=0.0,
                reasoning=f"Entropy {metrics.entropy_score:.3f} outside range {config['entropy_range']}"
            )
        
        # Check minimum confidence
        if trade_signal.confidence < config['min_confidence']:
            return PhaseGateDecision(
                gate_type=gate_type,
                decision=GateDecision.REJECT,
                confidence=trade_signal.confidence,
                reasoning=f"Confidence {trade_signal.confidence:.3f} below minimum {config['min_confidence']}"
            )
        
        # Calculate overall risk
        total_risk = np.mean([metrics.volatility_risk, metrics.liquidity_risk, metrics.timing_risk])
        
        # Make execution decision based on risk and gate type
        if total_risk < 0.3:
            # Low risk - execute immediately
            decision = GateDecision.EXECUTE_IMMEDIATELY
            delay = 0.0
        elif total_risk < 0.6:
            # Medium risk - execute with delay
            decision = GateDecision.EXECUTE_WITH_DELAY
            delay = config['execution_delay']
        elif total_risk < 0.8:
            # High risk - queue for later
            decision = GateDecision.QUEUE_FOR_LATER
            delay = config['execution_delay'] * 3
        else:
            # Critical risk - reject
            decision = GateDecision.REJECT
            delay = 0.0
        
        # Calculate position size adjustment based on risk
        position_adjustment = max(0.1, 1.0 - total_risk)
        
        # Calculate urgency based on gate type and entropy
        urgency = {
            PhaseGateType.MICRO_4B: 0.9,
            PhaseGateType.HARMONIC_8B: 0.6,
            PhaseGateType.STRATEGIC_42B: 0.3
        }[gate_type]
        
        return PhaseGateDecision(
            gate_type=gate_type,
            decision=decision,
            confidence=trade_signal.confidence,
            delay_seconds=delay,
            metrics=metrics,
            reasoning=f"Risk: {total_risk:.3f}, Gate: {gate_type.value}",
            position_size_adjustment=position_adjustment,
            urgency_level=urgency,
            risk_adjustment=1.0 + total_risk
        )
    
    def _apply_phase_gate_adjustments(self, 
                                    trade_signal: MathematicalTradeSignal,
                                    gate_decision: PhaseGateDecision) -> MathematicalTradeSignal:
        """Apply phase gate adjustments to trade signal"""
        # Create adjusted signal
        adjusted_signal = MathematicalTradeSignal(
            signal_id=f"{trade_signal.signal_id}_adjusted",
            timestamp=trade_signal.timestamp,
            unified_analysis=trade_signal.unified_analysis,
            fitness_score=trade_signal.fitness_score,
            graded_vector=trade_signal.graded_vector,
            symbol=trade_signal.symbol,
            side=trade_signal.side,
            position_size=trade_signal.position_size * gate_decision.position_size_adjustment,
            confidence=trade_signal.confidence,
            stop_loss=trade_signal.stop_loss,
            take_profit=trade_signal.take_profit,
            max_hold_time=trade_signal.max_hold_time,
            risk_level=trade_signal.risk_level,
            phase_gate=gate_decision.gate_type.value,
            entropy_score=gate_decision.metrics.entropy_score if gate_decision.metrics else 0.0,
            coherence_score=gate_decision.metrics.coherence_score if gate_decision.metrics else 0.0,
            mathematical_validity=trade_signal.mathematical_validity,
            klein_bottle_consistent=trade_signal.klein_bottle_consistent,
            fractal_convergent=trade_signal.fractal_convergent
        )
        
        return adjusted_signal
    
    async def _schedule_delayed_execution(self, signal: MathematicalTradeSignal, delay_seconds: float):
        """Schedule delayed execution of trade signal"""
        await asyncio.sleep(delay_seconds)
        
        if self.execution_manager:
            try:
                result = await self.execution_manager.execute_signal(signal)
                logger.info(f"Delayed execution completed for {signal.signal_id}: {result.status.value}")
            except Exception as e:
                logger.error(f"Delayed execution failed for {signal.signal_id}: {e}")
    
    def _update_gate_stats(self, gate_type: PhaseGateType, decision: PhaseGateDecision):
        """Update gate statistics"""
        if decision.decision == GateDecision.REJECT:
            self.gate_stats['gate_decisions'][gate_type]['closed'] += 1
        else:
            self.gate_stats['gate_decisions'][gate_type]['open'] += 1
            
        if decision.decision in [GateDecision.EXECUTE_IMMEDIATELY, GateDecision.EXECUTE_WITH_DELAY]:
            self.gate_stats['gate_decisions'][gate_type]['executed'] += 1
    
    def get_phase_gate_summary(self) -> Dict[str, Any]:
        """Get comprehensive phase gate summary"""
        return {
            'gate_status': {gate.value: status.value for gate, status in self.gate_status.items()},
            'statistics': self.gate_stats.copy(),
            'queue_length': len(self.delayed_execution_queue),
            'configuration': {gate.value: config for gate, config in self.gate_config.items()}
        }
    
    def set_gate_status(self, gate_type: PhaseGateType, status: PhaseGateStatus):
        """Set status for a specific phase gate"""
        self.gate_status[gate_type] = status
        logger.info(f"Phase gate {gate_type.value} status set to {status.value}")

# Integration function to create complete phase gate system
def create_phase_gate_system(execution_manager: CCXTExecutionManager) -> PhaseGateController:
    """Create a complete phase gate system integrated with execution manager"""
    
    # Initialize your existing mathematical components
    entropy_engine = UnifiedEntropyEngine()
    bit_operations = BitOperations()
    math_processor = UnifiedMathematicalProcessor()
    fitness_oracle = EnhancedFitnessOracle()
    
    # Create phase gate controller
    phase_controller = PhaseGateController(
        entropy_engine=entropy_engine,
        bit_operations=bit_operations,
        math_processor=math_processor,
        execution_manager=execution_manager,
        fitness_oracle=fitness_oracle
    )
    
    logger.info("🚀 Phase gate system created with mathematical integration")
    return phase_controller 