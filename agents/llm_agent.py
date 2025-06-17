#!/usr/bin/env python3
"""
Offline LLM Agent for Schwabot
==============================

Provides local CPU-based and CUDA-based agents for:
- Hash validation and pattern analysis
- Profit optimization strategies
- Risk assessment and decision support

Uses ZeroMQ REP socket for communication with main system.
Can run multiple instances (CPU + GPU) for parallel processing.
"""

import json
import logging
import os
import time
import zmq
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

# Try to import GPU libraries
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Local imports for analysis
try:
    from ..core.bit_operations import BitOperations
    from ..core.entropy_tracker import EntropyTracker
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("⚠️ Core modules not available - using simplified analysis")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentRequest:
    """Request structure for agent processing"""
    request_id: str
    request_type: str  # 'hash_validate', 'profit_optimize', 'risk_assess'
    hash_value: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    historical_data: Optional[List[Dict]] = None
    constraints: Optional[Dict[str, Any]] = None

@dataclass
class AgentResponse:
    """Response structure from agent"""
    request_id: str
    success: bool
    strategy_json: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    reasoning: str = ""
    processing_time_ms: float = 0.0
    agent_type: str = ""

class OfflineLLMAgent:
    """
    Offline agent for hash validation and profit optimization
    
    This agent processes requests via ZeroMQ and provides:
    - Hash pattern validation
    - Profit optimization strategies
    - Risk assessment
    - Pattern recognition and analysis
    """
    
    def __init__(self, 
                 port: int = 5555,
                 agent_type: str = "cpu",
                 use_cuda: bool = False):
        """
        Initialize the offline agent
        
        Args:
            port: ZeroMQ port to listen on
            agent_type: "cpu" or "gpu" for identification
            use_cuda: Whether to use CUDA acceleration
        """
        
        self.port = port
        self.agent_type = agent_type
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        # Analysis components
        self.bit_operations = None
        self.entropy_tracker = None
        
        if CORE_AVAILABLE:
            self.bit_operations = BitOperations()
            self.entropy_tracker = EntropyTracker()
        
        # Performance tracking
        self.requests_processed = 0
        self.start_time = time.time()
        self.total_processing_time = 0.0
        
        logger.info(f"🤖 {agent_type.upper()} Agent initialized on port {port} (CUDA: {self.use_cuda})")

    def run(self):
        """Main agent loop"""
        logger.info(f"🚀 Agent listening on tcp://*:{self.port}")
        
        try:
            while True:
                # Wait for request
                message = self.socket.recv_string()
                
                try:
                    # Parse request
                    request_data = json.loads(message)
                    request = AgentRequest(**request_data)
                    
                    # Process request
                    response = self.process_request(request)
                    
                    # Send response
                    self.socket.send_string(response.to_json())
                    
                    self.requests_processed += 1
                    
                except Exception as e:
                    # Send error response
                    error_response = AgentResponse(
                        request_id=request_data.get('request_id', 'unknown'),
                        success=False,
                        reasoning=f"Processing error: {str(e)}",
                        agent_type=self.agent_type
                    )
                    self.socket.send_string(error_response.to_json())
                    logger.error(f"Request processing error: {e}")
                
        except KeyboardInterrupt:
            logger.info("Agent shutdown requested")
        finally:
            self.cleanup()

    def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming request and return response"""
        start_time = time.time()
        
        try:
            if request.request_type == "hash_validate":
                response = self._validate_hash(request)
            elif request.request_type == "profit_optimize":
                response = self._optimize_profit(request)
            elif request.request_type == "risk_assess":
                response = self._assess_risk(request)
            else:
                response = AgentResponse(
                    request_id=request.request_id,
                    success=False,
                    reasoning=f"Unknown request type: {request.request_type}",
                    agent_type=self.agent_type
                )
        
        except Exception as e:
            response = AgentResponse(
                request_id=request.request_id,
                success=False,
                reasoning=f"Processing error: {str(e)}",
                agent_type=self.agent_type
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        response.processing_time_ms = processing_time
        self.total_processing_time += processing_time
        
        return response

    def _validate_hash(self, request: AgentRequest) -> AgentResponse:
        """Validate hash patterns and provide confidence assessment"""
        
        if not request.hash_value:
            return AgentResponse(
                request_id=request.request_id,
                success=False,
                reasoning="No hash value provided",
                agent_type=self.agent_type
            )
        
        try:
            hash_value = request.hash_value
            context = request.context or {}
            
            # Basic hash analysis
            bit_pattern = self._extract_bit_pattern(hash_value)
            pattern_strength = self._analyze_bit_strength(bit_pattern)
            
            # Advanced analysis if core available
            confidence = 0.5  # Base confidence
            analysis = {}
            
            if CORE_AVAILABLE and self.bit_operations:
                analysis = self.bit_operations.analyze_bit_pattern(bit_pattern)
                confidence = analysis.get('pattern_strength', 0.5)
            
            # Context-based adjustments
            if 'market_volatility' in context:
                volatility = context['market_volatility']
                if volatility > 0.8:  # High volatility
                    confidence *= 0.8  # Reduce confidence
                elif volatility < 0.2:  # Low volatility
                    confidence *= 1.2  # Increase confidence
            
            # Generate strategy recommendation
            strategy = self._generate_hash_strategy(hash_value, analysis, context)
            
            return AgentResponse(
                request_id=request.request_id,
                success=True,
                strategy_json=strategy,
                confidence=min(confidence, 1.0),
                reasoning=f"Hash pattern analysis completed. Strength: {pattern_strength:.3f}",
                agent_type=self.agent_type
            )
            
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                success=False,
                reasoning=f"Hash validation error: {str(e)}",
                agent_type=self.agent_type
            )

    def _optimize_profit(self, request: AgentRequest) -> AgentResponse:
        """Optimize profit strategies based on historical data"""
        
        try:
            context = request.context or {}
            historical_data = request.historical_data or []
            constraints = request.constraints or {}
            
            # Analyze historical performance
            profit_analysis = self._analyze_profit_patterns(historical_data)
            
            # Generate optimization strategy
            strategy = {
                'strategy_type': 'profit_optimization',
                'recommended_action': self._determine_optimal_action(profit_analysis, context),
                'position_sizing': self._calculate_optimal_position_size(profit_analysis, constraints),
                'risk_parameters': self._calculate_risk_parameters(profit_analysis),
                'entry_conditions': self._generate_entry_conditions(profit_analysis),
                'exit_conditions': self._generate_exit_conditions(profit_analysis),
                'confidence_multiplier': self._calculate_confidence_multiplier(profit_analysis)
            }
            
            # Calculate overall confidence
            confidence = self._calculate_strategy_confidence(profit_analysis, strategy)
            
            return AgentResponse(
                request_id=request.request_id,
                success=True,
                strategy_json=strategy,
                confidence=confidence,
                reasoning=f"Profit optimization based on {len(historical_data)} historical points",
                agent_type=self.agent_type
            )
            
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                success=False,
                reasoning=f"Profit optimization error: {str(e)}",
                agent_type=self.agent_type
            )

    def _assess_risk(self, request: AgentRequest) -> AgentResponse:
        """Assess risk levels and provide recommendations"""
        
        try:
            context = request.context or {}
            historical_data = request.historical_data or []
            
            # Risk analysis
            risk_metrics = self._calculate_risk_metrics(historical_data, context)
            
            # Generate risk assessment
            strategy = {
                'strategy_type': 'risk_assessment',
                'risk_level': risk_metrics['overall_risk'],
                'max_position_size': risk_metrics['max_safe_position'],
                'stop_loss_suggestion': risk_metrics['recommended_stop_loss'],
                'diversification_score': risk_metrics['diversification_needed'],
                'volatility_assessment': risk_metrics['volatility_level'],
                'correlation_warnings': risk_metrics['correlation_risks'],
                'recommended_actions': risk_metrics['risk_actions']
            }
            
            # Risk-adjusted confidence
            confidence = max(0.1, 1.0 - risk_metrics['overall_risk'])
            
            return AgentResponse(
                request_id=request.request_id,
                success=True,
                strategy_json=strategy,
                confidence=confidence,
                reasoning=f"Risk assessment: {risk_metrics['overall_risk']:.1%} risk level",
                agent_type=self.agent_type
            )
            
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                success=False,
                reasoning=f"Risk assessment error: {str(e)}",
                agent_type=self.agent_type
            )

    def _extract_bit_pattern(self, hash_value: int) -> int:
        """Extract bit pattern from hash value"""
        # Use lower 42 bits for pattern analysis
        return hash_value & ((1 << 42) - 1)

    def _analyze_bit_strength(self, bit_pattern: int) -> float:
        """Analyze bit pattern strength"""
        if bit_pattern == 0:
            return 0.0
        
        # Count set bits
        set_bits = bin(bit_pattern).count('1')
        total_bits = 42
        
        # Calculate entropy-like measure
        density = set_bits / total_bits
        
        # Optimal density is around 0.5 (balanced)
        strength = 1.0 - abs(density - 0.5) * 2
        return max(0.0, strength)

    def _generate_hash_strategy(self, hash_value: int, analysis: Dict, context: Dict) -> Dict:
        """Generate strategy based on hash analysis"""
        
        strategy = {
            'hash_value': hash_value,
            'pattern_type': analysis.get('dominant_pattern', 'unknown'),
            'strength_score': analysis.get('pattern_strength', 0.5),
            'recommended_timeframe': self._suggest_timeframe(analysis),
            'signal_weight': self._calculate_signal_weight(analysis, context),
            'filters': self._generate_filters(analysis),
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'agent_type': self.agent_type,
                'cuda_accelerated': self.use_cuda
            }
        }
        
        return strategy

    def _analyze_profit_patterns(self, historical_data: List[Dict]) -> Dict:
        """Analyze profit patterns from historical data"""
        if not historical_data:
            return {'patterns': [], 'avg_profit': 0.0, 'volatility': 1.0}
        
        profits = [d.get('profit', 0.0) for d in historical_data]
        
        analysis = {
            'avg_profit': np.mean(profits),
            'profit_volatility': np.std(profits),
            'win_rate': sum(1 for p in profits if p > 0) / len(profits),
            'max_profit': max(profits),
            'max_loss': min(profits),
            'profit_trend': self._calculate_trend(profits),
            'consistency_score': self._calculate_consistency(profits)
        }
        
        return analysis

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0.0
        return np.clip(slope, -1.0, 1.0)

    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency score (0 to 1)"""
        if len(values) < 2:
            return 0.5
        
        # Coefficient of variation (inverted)
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
        
        cv = np.std(values) / abs(mean_val)
        return 1.0 / (1.0 + cv)

    def _determine_optimal_action(self, profit_analysis: Dict, context: Dict) -> str:
        """Determine optimal trading action"""
        
        avg_profit = profit_analysis.get('avg_profit', 0.0)
        trend = profit_analysis.get('profit_trend', 0.0)
        win_rate = profit_analysis.get('win_rate', 0.5)
        
        if avg_profit > 0 and trend > 0.2 and win_rate > 0.6:
            return "strong_long"
        elif avg_profit > 0 and win_rate > 0.55:
            return "moderate_long"
        elif avg_profit < 0 and trend < -0.2 and win_rate < 0.4:
            return "avoid"
        else:
            return "neutral"

    def _calculate_optimal_position_size(self, profit_analysis: Dict, constraints: Dict) -> float:
        """Calculate optimal position size using Kelly criterion approximation"""
        
        win_rate = profit_analysis.get('win_rate', 0.5)
        avg_profit = profit_analysis.get('avg_profit', 0.0)
        max_loss = abs(profit_analysis.get('max_loss', 0.05))
        
        if max_loss == 0:
            return 0.0
        
        # Simplified Kelly fraction
        kelly_fraction = (win_rate * abs(avg_profit) - (1 - win_rate) * max_loss) / max_loss
        
        # Apply constraints
        max_position = constraints.get('max_position_size', 0.1)
        optimal_size = min(max(kelly_fraction * 0.5, 0.0), max_position)  # Conservative Kelly
        
        return optimal_size

    def _calculate_risk_parameters(self, profit_analysis: Dict) -> Dict:
        """Calculate risk management parameters"""
        
        volatility = profit_analysis.get('profit_volatility', 0.05)
        max_loss = abs(profit_analysis.get('max_loss', 0.05))
        
        return {
            'stop_loss_pct': min(max_loss * 1.5, 0.1),  # 1.5x historical max loss, capped at 10%
            'take_profit_pct': volatility * 3,  # 3x volatility
            'trailing_stop_pct': volatility * 2,  # 2x volatility
            'max_holding_time': 24 * 60,  # 24 hours in minutes
        }

    def _generate_entry_conditions(self, profit_analysis: Dict) -> List[str]:
        """Generate entry condition recommendations"""
        
        conditions = []
        
        win_rate = profit_analysis.get('win_rate', 0.5)
        consistency = profit_analysis.get('consistency_score', 0.5)
        
        if win_rate > 0.6:
            conditions.append("high_confidence_signal")
        
        if consistency > 0.7:
            conditions.append("consistent_pattern")
        
        conditions.append("volume_confirmation")
        conditions.append("trend_alignment")
        
        return conditions

    def _generate_exit_conditions(self, profit_analysis: Dict) -> List[str]:
        """Generate exit condition recommendations"""
        
        conditions = []
        
        conditions.append("stop_loss_hit")
        conditions.append("take_profit_hit")
        conditions.append("pattern_invalidation")
        conditions.append("time_limit_reached")
        
        volatility = profit_analysis.get('profit_volatility', 0.05)
        if volatility > 0.1:
            conditions.append("volatility_spike")
        
        return conditions

    def _calculate_confidence_multiplier(self, profit_analysis: Dict) -> float:
        """Calculate confidence multiplier for strategies"""
        
        win_rate = profit_analysis.get('win_rate', 0.5)
        consistency = profit_analysis.get('consistency_score', 0.5)
        trend = profit_analysis.get('profit_trend', 0.0)
        
        # Base multiplier
        multiplier = 1.0
        
        # Adjust based on metrics
        if win_rate > 0.7:
            multiplier *= 1.2
        elif win_rate < 0.4:
            multiplier *= 0.8
        
        if consistency > 0.8:
            multiplier *= 1.1
        elif consistency < 0.3:
            multiplier *= 0.9
        
        if abs(trend) > 0.5:
            multiplier *= 1.1
        
        return np.clip(multiplier, 0.5, 2.0)

    def _calculate_strategy_confidence(self, profit_analysis: Dict, strategy: Dict) -> float:
        """Calculate overall strategy confidence"""
        
        base_confidence = 0.5
        
        # Adjust based on profit analysis
        win_rate = profit_analysis.get('win_rate', 0.5)
        consistency = profit_analysis.get('consistency_score', 0.5)
        
        confidence = base_confidence * (1 + win_rate) * (1 + consistency) / 2
        
        # Apply confidence multiplier
        multiplier = strategy.get('confidence_multiplier', 1.0)
        confidence *= multiplier
        
        return np.clip(confidence, 0.0, 1.0)

    def _calculate_risk_metrics(self, historical_data: List[Dict], context: Dict) -> Dict:
        """Calculate comprehensive risk metrics"""
        
        if not historical_data:
            return {
                'overall_risk': 0.5,
                'max_safe_position': 0.05,
                'recommended_stop_loss': 0.05,
                'diversification_needed': 0.8,
                'volatility_level': 0.5,
                'correlation_risks': [],
                'risk_actions': ['reduce_position_size', 'increase_diversification']
            }
        
        profits = [d.get('profit', 0.0) for d in historical_data]
        volatility = np.std(profits)
        downside_deviation = np.std([p for p in profits if p < 0])
        max_drawdown = self._calculate_max_drawdown(profits)
        
        # Overall risk score
        risk_factors = [
            volatility * 2,  # Volatility component
            max_drawdown,    # Drawdown component
            downside_deviation * 1.5,  # Downside risk
        ]
        
        overall_risk = np.clip(np.mean(risk_factors), 0.0, 1.0)
        
        return {
            'overall_risk': overall_risk,
            'max_safe_position': max(0.01, 0.2 * (1 - overall_risk)),
            'recommended_stop_loss': max(0.02, volatility * 2),
            'diversification_needed': min(0.9, overall_risk + 0.2),
            'volatility_level': np.clip(volatility * 10, 0.0, 1.0),
            'correlation_risks': self._assess_correlation_risks(context),
            'risk_actions': self._generate_risk_actions(overall_risk)
        }

    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not profits:
            return 0.0
        
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / (running_max + 1e-8)
        
        return np.max(drawdown)

    def _assess_correlation_risks(self, context: Dict) -> List[str]:
        """Assess correlation risks"""
        risks = []
        
        if context.get('market_correlation', 0.0) > 0.8:
            risks.append("high_market_correlation")
        
        if context.get('sector_concentration', 0.0) > 0.6:
            risks.append("sector_concentration")
        
        return risks

    def _generate_risk_actions(self, overall_risk: float) -> List[str]:
        """Generate risk mitigation actions"""
        actions = []
        
        if overall_risk > 0.8:
            actions.extend(["reduce_position_size", "increase_stop_loss", "exit_positions"])
        elif overall_risk > 0.6:
            actions.extend(["reduce_position_size", "increase_diversification"])
        elif overall_risk > 0.4:
            actions.append("monitor_closely")
        
        return actions

    def _suggest_timeframe(self, analysis: Dict) -> str:
        """Suggest optimal timeframe based on analysis"""
        strength = analysis.get('pattern_strength', 0.5)
        
        if strength > 0.8:
            return "1h"  # Strong patterns work on shorter timeframes
        elif strength > 0.6:
            return "4h"
        else:
            return "1d"  # Weak patterns need longer timeframes

    def _calculate_signal_weight(self, analysis: Dict, context: Dict) -> float:
        """Calculate signal weight for strategy"""
        base_weight = analysis.get('pattern_strength', 0.5)
        
        # Adjust based on context
        if context.get('market_volatility', 0.5) > 0.8:
            base_weight *= 0.8  # Reduce weight in high volatility
        
        return np.clip(base_weight, 0.1, 1.0)

    def _generate_filters(self, analysis: Dict) -> List[str]:
        """Generate signal filters"""
        filters = ["volume_filter", "trend_filter"]
        
        strength = analysis.get('pattern_strength', 0.5)
        if strength < 0.6:
            filters.append("confirmation_filter")
        
        return filters

    def get_status(self) -> Dict:
        """Get agent status"""
        uptime = time.time() - self.start_time
        avg_processing_time = (self.total_processing_time / self.requests_processed 
                             if self.requests_processed > 0 else 0.0)
        
        return {
            'agent_type': self.agent_type,
            'port': self.port,
            'cuda_enabled': self.use_cuda,
            'uptime_seconds': uptime,
            'requests_processed': self.requests_processed,
            'avg_processing_time_ms': avg_processing_time,
            'status': 'running'
        }

    def cleanup(self):
        """Cleanup resources"""
        self.socket.close()
        self.context.term()
        logger.info(f"Agent {self.agent_type} cleaned up")

# Add to_json method to AgentResponse
def agent_response_to_json(self) -> str:
    """Convert AgentResponse to JSON string"""
    return json.dumps({
        'request_id': self.request_id,
        'success': self.success,
        'strategy_json': self.strategy_json,
        'confidence': self.confidence,
        'reasoning': self.reasoning,
        'processing_time_ms': self.processing_time_ms,
        'agent_type': self.agent_type
    })

AgentResponse.to_json = agent_response_to_json

def main():
    """Main entry point for agent"""
    parser = argparse.ArgumentParser(description="Schwabot Offline LLM Agent")
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ port")
    parser.add_argument("--type", default="cpu", choices=["cpu", "gpu"], help="Agent type")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA acceleration")
    
    args = parser.parse_args()
    
    # Set CUDA device if specified
    if args.cuda and CUDA_AVAILABLE:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Create and run agent
    agent = OfflineLLMAgent(
        port=args.port,
        agent_type=args.type,
        use_cuda=args.cuda
    )
    
    try:
        agent.run()
    except KeyboardInterrupt:
        logger.info("Agent shutdown requested")
    finally:
        agent.cleanup()

if __name__ == "__main__":
    main() 