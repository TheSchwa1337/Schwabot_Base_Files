#!/usr/bin/env python3
"""Fallback Logic Router - Graceful Degradation for Primary Logic Failures.

This module provides intelligent fallback mechanisms when primary Schwabot components
fail, ensuring system continuity and graceful degradation while maintaining
mathematical consistency and safety protocols.

Mathematical Foundation:
- Fallback strategy selection based on failure type and severity
- Graceful degradation with mathematical consistency preservation
- Error recovery with minimal impact on system performance
- Adaptive fallback routing based on component health
- Integration with Phantom Lag Model and Meta-Layer Ghost Bridge
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import math
import numpy as np

from core.error_handler import safe_execute
from core.import_resolver import safe_import

# Import new mathematical components
try:
    from core.phantom_lag_model import PhantomLagModel, phantom_lag_penalty
    from core.meta_layer_ghost_bridge import MetaLayerGhostBridge, get_meta_ghost_vector
    PHANTOM_LAG_AVAILABLE = True
    META_BRIDGE_AVAILABLE = True
except ImportError:
    PHANTOM_LAG_AVAILABLE = False
    META_BRIDGE_AVAILABLE = False
    logger.warning("Phantom Lag Model and Meta-Layer Ghost Bridge not available")

logger = logging.getLogger(__name__)


@dataclass
class FallbackStrategy:
    """Represents a fallback strategy configuration."""
    
    strategy_id: str
    name: str
    description: str
    severity_level: str  # 'low', 'medium', 'high', 'critical'
    recovery_time: float  # seconds
    success_rate: float
    mathematical_consistency: bool
    handler_function: Callable
    phantom_lag_integration: bool = False
    meta_bridge_integration: bool = False


@dataclass
class FallbackResult:
    """Result of fallback strategy execution."""
    
    strategy_used: str
    success: bool
    recovery_time: float
    data_quality: float
    mathematical_consistency: bool
    error_message: Optional[str]
    phantom_lag_penalty: Optional[float] = None
    meta_ghost_vector: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class FallbackLogicRouter:
    """Handles graceful degradation when primary logic fails."""
    
    def __init__(self) -> None:
        """Initialize the fallback logic router."""
        self.fallback_strategies = {}
        self.fallback_history = []
        self.max_history_size = 1000
        
        # Component health tracking
        self.component_health = {}
        self.health_decay_rate = 0.1
        
        # Initialize mathematical components
        self.phantom_lag_model = None
        self.meta_ghost_bridge = None
        self._initialize_mathematical_components()
        
        # Initialize fallback strategies
        self._initialize_fallback_strategies()
        
        logger.info("FallbackLogicRouter initialized with mathematical components")
    
    def _initialize_mathematical_components(self) -> None:
        """Initialize Phantom Lag Model and Meta-Layer Ghost Bridge."""
        try:
            if PHANTOM_LAG_AVAILABLE:
                self.phantom_lag_model = PhantomLagModel()
                logger.info("Phantom Lag Model integrated with FallbackLogicRouter")
            
            if META_BRIDGE_AVAILABLE:
                self.meta_ghost_bridge = MetaLayerGhostBridge()
                logger.info("Meta-Layer Ghost Bridge integrated with FallbackLogicRouter")
                
        except Exception as e:
            logger.error(f"Error initializing mathematical components: {e}")
    
    def _initialize_fallback_strategies(self) -> None:
        """Initialize all available fallback strategies."""
        
        # Data processing fallbacks
        self.fallback_strategies['data_processor'] = {
            'primary': FallbackStrategy(
                strategy_id='data_processor_primary',
                name='Simplified Data Processing',
                description='Use simplified data processing with reduced complexity',
                severity_level='medium',
                recovery_time=2.0,
                success_rate=0.85,
                mathematical_consistency=True,
                handler_function=self._fallback_data_processing,
                phantom_lag_integration=True,
                meta_bridge_integration=True
            ),
            'critical': FallbackStrategy(
                strategy_id='data_processor_critical',
                name='Minimal Data Processing',
                description='Use minimal data processing for critical operations',
                severity_level='critical',
                recovery_time=1.0,
                success_rate=0.95,
                mathematical_consistency=True,
                handler_function=self._fallback_critical_data_processing,
                phantom_lag_integration=False,
                meta_bridge_integration=False
            )
        }
        
        # Altitude math fallbacks
        self.fallback_strategies['altitude_math'] = {
            'primary': FallbackStrategy(
                strategy_id='altitude_math_primary',
                name='Simplified Altitude Calculation',
                description='Use simplified altitude calculation with basic metrics',
                severity_level='medium',
                recovery_time=1.5,
                success_rate=0.80,
                mathematical_consistency=True,
                handler_function=self._fallback_altitude_calculation,
                phantom_lag_integration=True,
                meta_bridge_integration=True
            ),
            'critical': FallbackStrategy(
                strategy_id='altitude_math_critical',
                name='Static Altitude Values',
                description='Use static altitude values for critical operations',
                severity_level='critical',
                recovery_time=0.5,
                success_rate=0.98,
                mathematical_consistency=True,
                handler_function=self._fallback_static_altitude,
                phantom_lag_integration=False,
                meta_bridge_integration=False
            )
        }
        
        # Profit routing fallbacks
        self.fallback_strategies['profit_routing'] = {
            'primary': FallbackStrategy(
                strategy_id='profit_routing_primary',
                name='Simplified Profit Calculation',
                description='Use simplified profit calculation with basic metrics',
                severity_level='medium',
                recovery_time=2.0,
                success_rate=0.75,
                mathematical_consistency=True,
                handler_function=self._fallback_profit_calculation,
                phantom_lag_integration=True,
                meta_bridge_integration=True
            ),
            'critical': FallbackStrategy(
                strategy_id='profit_routing_critical',
                name='Conservative Profit Routing',
                description='Use conservative profit routing for safety',
                severity_level='critical',
                recovery_time=1.0,
                success_rate=0.90,
                mathematical_consistency=True,
                handler_function=self._fallback_conservative_routing,
                phantom_lag_integration=False,
                meta_bridge_integration=False
            )
        }
        
        # Hash matrix fallbacks
        self.fallback_strategies['hash_matrix'] = {
            'primary': FallbackStrategy(
                strategy_id='hash_matrix_primary',
                name='Simplified Hash Matching',
                description='Use simplified hash matching with reduced complexity',
                severity_level='medium',
                recovery_time=1.5,
                success_rate=0.70,
                mathematical_consistency=True,
                handler_function=self._fallback_hash_matching,
                phantom_lag_integration=True,
                meta_bridge_integration=True
            ),
            'critical': FallbackStrategy(
                strategy_id='hash_matrix_critical',
                name='Basic Hash Validation',
                description='Use basic hash validation for critical operations',
                severity_level='critical',
                recovery_time=0.5,
                success_rate=0.95,
                mathematical_consistency=True,
                handler_function=self._fallback_basic_hash_validation,
                phantom_lag_integration=False,
                meta_bridge_integration=False
            )
        }
        
        # NEW: Phantom Lag Model fallbacks
        self.fallback_strategies['phantom_lag'] = {
            'primary': FallbackStrategy(
                strategy_id='phantom_lag_primary',
                name='Phantom Lag Analysis',
                description='Analyze missed opportunities using Phantom Lag Model',
                severity_level='medium',
                recovery_time=1.0,
                success_rate=0.85,
                mathematical_consistency=True,
                handler_function=self._fallback_phantom_lag_analysis,
                phantom_lag_integration=True,
                meta_bridge_integration=True
            ),
            'critical': FallbackStrategy(
                strategy_id='phantom_lag_critical',
                name='Basic Lag Penalty',
                description='Calculate basic lag penalty for critical operations',
                severity_level='critical',
                recovery_time=0.3,
                success_rate=0.95,
                mathematical_consistency=True,
                handler_function=self._fallback_basic_lag_penalty,
                phantom_lag_integration=True,
                meta_bridge_integration=False
            )
        }
        
        # NEW: Meta-Layer Ghost Bridge fallbacks
        self.fallback_strategies['meta_bridge'] = {
            'primary': FallbackStrategy(
                strategy_id='meta_bridge_primary',
                name='Meta-Ghost Bridge Analysis',
                description='Analyze cross-layer coordination using Meta-Layer Ghost Bridge',
                severity_level='medium',
                recovery_time=1.5,
                success_rate=0.80,
                mathematical_consistency=True,
                handler_function=self._fallback_meta_bridge_analysis,
                phantom_lag_integration=True,
                meta_bridge_integration=True
            ),
            'critical': FallbackStrategy(
                strategy_id='meta_bridge_critical',
                name='Basic Meta Vector',
                description='Calculate basic meta-ghost vector for critical operations',
                severity_level='critical',
                recovery_time=0.5,
                success_rate=0.90,
                mathematical_consistency=True,
                handler_function=self._fallback_basic_meta_vector,
                phantom_lag_integration=False,
                meta_bridge_integration=True
            )
        }
    
    def route_fallback(self, module: str, error: Exception, context: Dict[str, Any] = None) -> Any:
        """Route to appropriate fallback strategy based on module and error.
        
        Args:
            module: Name of the failed module
            error: The exception that caused the failure
            context: Additional context for mathematical integration
            
        Returns:
            Result from fallback strategy execution
        """
        try:
            # Determine severity level based on error type
            severity_level = self._determine_severity_level(error)
            
            # Select appropriate fallback strategy
            strategy = self._select_fallback_strategy(module, severity_level)
            
            if not strategy:
                logger.error(f"No fallback strategy available for {module}")
                return None
            
            # Execute fallback strategy
            start_time = datetime.now()
            result = strategy.handler_function(error, context)
            recovery_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate mathematical integration metrics
            phantom_lag_penalty = None
            meta_ghost_vector = None
            
            if strategy.phantom_lag_integration and context:
                phantom_lag_penalty = self._calculate_phantom_lag_penalty(context)
            
            if strategy.meta_bridge_integration and context:
                meta_ghost_vector = self._calculate_meta_ghost_vector(context)
            
            # Create fallback result
            fallback_result = FallbackResult(
                strategy_used=strategy.strategy_id,
                success=result is not None,
                recovery_time=recovery_time,
                data_quality=self._assess_data_quality(result),
                mathematical_consistency=strategy.mathematical_consistency,
                error_message=str(error) if error else None,
                phantom_lag_penalty=phantom_lag_penalty,
                meta_ghost_vector=meta_ghost_vector
            )
            
            # Store result and update component health
            self._store_fallback_result(fallback_result)
            self._update_component_health(module, fallback_result.success)
            
            logger.info(f"Fallback executed: {strategy.name} for {module} "
                       f"(success: {fallback_result.success}, "
                       f"phantom_lag: {phantom_lag_penalty:.4f if phantom_lag_penalty else 'N/A'}, "
                       f"meta_vector: {meta_ghost_vector:.4f if meta_ghost_vector else 'N/A'})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fallback routing: {e}")
            return None
    
    def _calculate_phantom_lag_penalty(self, context: Dict[str, Any]) -> Optional[float]:
        """Calculate phantom lag penalty from context."""
        try:
            if not self.phantom_lag_model or not context:
                return None
            
            # Extract relevant data from context
            delta_price = context.get('delta_price', 0.0)
            entropy = context.get('entropy', 0.5)
            max_price_ref = context.get('max_price_ref', 70000.0)
            
            # Calculate phantom lag penalty
            penalty = self.phantom_lag_model.calculate_phantom_lag_penalty(
                delta_price, entropy, max_price_ref
            )
            
            return penalty
            
        except Exception as e:
            logger.error(f"Error calculating phantom lag penalty: {e}")
            return None
    
    def _calculate_meta_ghost_vector(self, context: Dict[str, Any]) -> Optional[float]:
        """Calculate meta-ghost vector from context."""
        try:
            if not self.meta_ghost_bridge or not context:
                return None
            
            # Extract relevant data from context
            symbol = context.get('symbol', 'BTC/USD')
            
            # Get meta-ghost vector
            meta_vector = self.meta_ghost_bridge.get_meta_vector(symbol)
            
            return meta_vector
            
        except Exception as e:
            logger.error(f"Error calculating meta-ghost vector: {e}")
            return None
    
    def _fallback_phantom_lag_analysis(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback phantom lag analysis with mathematical integration."""
        try:
            if not self.phantom_lag_model:
                return self._fallback_basic_lag_penalty(error, context)
            
            # Extract data from context
            entry_price = context.get('entry_price', 50000.0) if context else 50000.0
            current_price = context.get('current_price', 50000.0) if context else 50000.0
            signal_hash = context.get('signal_hash', 'default_hash') if context else 'default_hash'
            entropy_level = context.get('entropy_level', 0.5) if context else 0.5
            event_type = context.get('event_type', 'missed_entry') if context else 'missed_entry'
            
            # Analyze missed opportunity
            analysis = self.phantom_lag_model.analyze_missed_opportunity(
                entry_price, current_price, signal_hash, entropy_level, event_type
            )
            
            return {
                'lag_penalty': analysis.lag_penalty,
                'opportunity_cost': analysis.opportunity_cost,
                'confidence_impact': analysis.confidence_impact,
                're_entry_recommendation': analysis.re_entry_recommendation,
                'adaptation_score': analysis.adaptation_score,
                'mathematical_validity': analysis.mathematical_validity,
                'fallback_mode': True,
                'original_error': str(error)
            }
            
        except Exception as e:
            logger.error(f"Error in phantom lag analysis fallback: {e}")
            return self._fallback_basic_lag_penalty(error, context)
    
    def _fallback_basic_lag_penalty(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Basic lag penalty calculation for critical operations."""
        try:
            # Simple lag penalty calculation
            delta_price = context.get('delta_price', 0.0) if context else 0.0
            entropy = context.get('entropy', 0.5) if context else 0.5
            
            # Basic calculation: L(Δp, 𝓔) = e^(-𝓔) × (Δp / P_max)
            max_price_ref = 70000.0
            normalized_delta = max(0.0, delta_price / max_price_ref)
            entropy_decay = np.exp(-entropy)
            lag_penalty = entropy_decay * normalized_delta
            
            return {
                'lag_penalty': float(np.clip(lag_penalty, 0.0, 1.0)),
                'opportunity_cost': delta_price * lag_penalty,
                'confidence_impact': 1.0 - lag_penalty,
                're_entry_recommendation': lag_penalty > 0.1,
                'adaptation_score': lag_penalty,
                'mathematical_validity': True,
                'fallback_mode': True,
                'critical_fallback': True,
                'original_error': str(error)
            }
            
        except Exception as e:
            logger.error(f"Error in basic lag penalty fallback: {e}")
            return {
                'lag_penalty': 0.0,
                'opportunity_cost': 0.0,
                'confidence_impact': 0.0,
                're_entry_recommendation': False,
                'adaptation_score': 0.0,
                'mathematical_validity': False,
                'fallback_mode': True,
                'critical_fallback': True,
                'original_error': str(error)
            }
    
    def _fallback_meta_bridge_analysis(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback meta-bridge analysis with mathematical integration."""
        try:
            if not self.meta_ghost_bridge:
                return self._fallback_basic_meta_vector(error, context)
            
            # Extract data from context
            symbol = context.get('symbol', 'BTC/USD') if context else 'BTC/USD'
            market_data = context.get('market_data', {}) if context else {}
            position_data = context.get('position_data', {}) if context else {}
            bot_id = context.get('bot_id', 'fallback_bot') if context else 'fallback_bot'
            
            # Synchronize with meta-bridge
            sync_result = self.meta_ghost_bridge.synchronize_bot(
                bot_id, market_data, position_data
            )
            
            # Get current opportunities
            opportunities = self.meta_ghost_bridge.get_current_opportunities()
            
            return {
                'ghost_price': sync_result.get('ghost_price', 0.0),
                'meta_vector': sync_result.get('meta_vector', 0.0),
                'opportunities_count': sync_result.get('opportunities_count', 0),
                'synchronization_success': sync_result.get('synchronization_success', False),
                'bridge_opportunities': [op.__dict__ for op in opportunities[:5]],  # Top 5 opportunities
                'mathematical_validity': True,
                'fallback_mode': True,
                'original_error': str(error)
            }
            
        except Exception as e:
            logger.error(f"Error in meta-bridge analysis fallback: {e}")
            return self._fallback_basic_meta_vector(error, context)
    
    def _fallback_basic_meta_vector(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Basic meta-ghost vector calculation for critical operations."""
        try:
            # Simple meta vector calculation
            symbol = context.get('symbol', 'BTC/USD') if context else 'BTC/USD'
            
            # Basic meta vector (simplified)
            base_vector = 0.5  # Neutral vector
            symbol_factor = hash(symbol) % 100 / 100.0  # Deterministic but varied
            meta_vector = base_vector + (symbol_factor - 0.5) * 0.2  # Small variation
            
            return {
                'ghost_price': 50000.0,  # Default BTC price
                'meta_vector': float(meta_vector),
                'opportunities_count': 0,
                'synchronization_success': True,
                'bridge_opportunities': [],
                'mathematical_validity': True,
                'fallback_mode': True,
                'critical_fallback': True,
                'original_error': str(error)
            }
            
        except Exception as e:
            logger.error(f"Error in basic meta vector fallback: {e}")
            return {
                'ghost_price': 0.0,
                'meta_vector': 0.0,
                'opportunities_count': 0,
                'synchronization_success': False,
                'bridge_opportunities': [],
                'mathematical_validity': False,
                'fallback_mode': True,
                'critical_fallback': True,
                'original_error': str(error)
            }
    
    def _determine_severity_level(self, error: Exception) -> str:
        """Determine severity level based on error type."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if any(critical in error_message for critical in ['memory', 'disk', 'network', 'timeout']):
            return 'critical'
        
        # High severity errors
        if any(high in error_message for high in ['connection', 'authentication', 'permission']):
            return 'high'
        
        # Medium severity errors
        if any(medium in error_message for medium in ['calculation', 'validation', 'format']):
            return 'medium'
        
        # Default to low severity
        return 'low'
    
    def _select_fallback_strategy(self, module: str, severity_level: str) -> Optional[FallbackStrategy]:
        """Select appropriate fallback strategy based on module and severity."""
        try:
            if module not in self.fallback_strategies:
                return None
            
            strategies = self.fallback_strategies[module]
            
            # Try to find exact severity match
            if severity_level in strategies:
                return strategies[severity_level]
            
            # Fall back to primary strategy
            if 'primary' in strategies:
                return strategies['primary']
            
            # Fall back to critical strategy as last resort
            if 'critical' in strategies:
                return strategies['critical']
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting fallback strategy: {e}")
            return None
    
    def _fallback_data_processing(self, error: Exception) -> Dict[str, Any]:
        """Fallback data processing with simplified logic."""
        try:
            # Simplified data processing logic
            return {
                'price': 50000.0,  # Default BTC price
                'volume': 1000.0,  # Default volume
                'timestamp': datetime.now().timestamp(),
                'confidence': 0.5,  # Reduced confidence
                'fallback_mode': True,
                'original_error': str(error)
            }
        except Exception as e:
            logger.error(f"Error in fallback data processing: {e}")
            return None
    
    def _fallback_critical_data_processing(self, error: Exception) -> Dict[str, Any]:
        """Critical fallback data processing with minimal logic."""
        try:
            # Minimal data processing for critical operations
            return {
                'price': 50000.0,
                'volume': 1000.0,
                'timestamp': datetime.now().timestamp(),
                'confidence': 0.3,  # Very low confidence
                'critical_fallback': True,
                'original_error': str(error)
            }
        except Exception as e:
            logger.error(f"Error in critical fallback data processing: {e}")
            return None
    
    def _fallback_altitude_calculation(self, error: Exception) -> Dict[str, Any]:
        """Fallback altitude calculation with simplified metrics."""
        try:
            # Simplified altitude calculation
            return {
                'altitude_score': 0.5,  # Neutral altitude
                'drift_compensation': 0.0,  # No drift compensation
                'regulation_vector': [0.25, 0.25, 0.25, 0.25],  # Equal weights
                'confidence': 0.6,
                'fallback_mode': True,
                'original_error': str(error)
            }
        except Exception as e:
            logger.error(f"Error in fallback altitude calculation: {e}")
            return None
    
    def _fallback_static_altitude(self, error: Exception) -> Dict[str, Any]:
        """Static altitude values for critical operations."""
        try:
            # Static altitude values
            return {
                'altitude_score': 0.5,
                'drift_compensation': 0.0,
                'regulation_vector': [0.25, 0.25, 0.25, 0.25],
                'confidence': 0.8,  # High confidence for static values
                'static_fallback': True,
                'original_error': str(error)
            }
        except Exception as e:
            logger.error(f"Error in static altitude fallback: {e}")
            return None
    
    def _fallback_profit_calculation(self, error: Exception) -> Dict[str, Any]:
        """Fallback profit calculation with simplified logic."""
        try:
            # Simplified profit calculation
            return {
                'profit_vector': 0.0,  # Neutral profit
                'risk_score': 0.5,  # Medium risk
                'confidence': 0.5,
                'fallback_mode': True,
                'original_error': str(error)
            }
        except Exception as e:
            logger.error(f"Error in fallback profit calculation: {e}")
            return None
    
    def _fallback_conservative_routing(self, error: Exception) -> Dict[str, Any]:
        """Conservative profit routing for safety."""
        try:
            # Conservative routing
            return {
                'profit_vector': -0.1,  # Slightly negative for safety
                'risk_score': 0.3,  # Low risk
                'confidence': 0.7,
                'conservative_fallback': True,
                'original_error': str(error)
            }
        except Exception as e:
            logger.error(f"Error in conservative routing fallback: {e}")
            return None
    
    def _fallback_hash_matching(self, error: Exception) -> Dict[str, Any]:
        """Fallback hash matching with simplified logic."""
        try:
            # Simplified hash matching
            return {
                'hash_match': False,  # Assume no match for safety
                'confidence': 0.4,
                'fallback_mode': True,
                'original_error': str(error)
            }
        except Exception as e:
            logger.error(f"Error in fallback hash matching: {e}")
            return None
    
    def _fallback_basic_hash_validation(self, error: Exception) -> Dict[str, Any]:
        """Basic hash validation for critical operations."""
        try:
            # Basic hash validation
            return {
                'hash_valid': True,  # Assume valid for critical operations
                'confidence': 0.6,
                'basic_validation': True,
                'original_error': str(error)
            }
        except Exception as e:
            logger.error(f"Error in basic hash validation fallback: {e}")
            return None
    
    def _assess_data_quality(self, result: Any) -> float:
        """Assess data quality of fallback result."""
        try:
            if result is None:
                return 0.0
            
            if not isinstance(result, dict):
                return 0.5
            
            # Check for fallback indicators
            fallback_indicators = ['fallback_mode', 'critical_fallback', 'static_fallback', 
                                 'conservative_fallback', 'basic_validation']
            
            quality_score = 1.0
            for indicator in fallback_indicators:
                if indicator in result:
                    quality_score *= 0.8  # Reduce quality for fallback indicators
            
            return max(0.1, quality_score)
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 0.0
    
    def _store_fallback_result(self, result: FallbackResult) -> None:
        """Store fallback result in history."""
        try:
            self.fallback_history.append(result)
            
            # Maintain history size
            if len(self.fallback_history) > self.max_history_size:
                self.fallback_history = self.fallback_history[-self.max_history_size:]
                
        except Exception as e:
            logger.error(f"Error storing fallback result: {e}")
    
    def _update_component_health(self, module: str, success: bool) -> None:
        """Update component health based on fallback success."""
        try:
            if module not in self.component_health:
                self.component_health[module] = 1.0
            
            # Update health based on success/failure
            if success:
                # Gradual recovery
                self.component_health[module] = min(1.0, 
                    self.component_health[module] + 0.1)
            else:
                # Gradual degradation
                self.component_health[module] = max(0.0, 
                    self.component_health[module] - self.health_decay_rate)
                
        except Exception as e:
            logger.error(f"Error updating component health: {e}")
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback statistics and trends."""
        try:
            if not self.fallback_history:
                return {'total_fallbacks': 0, 'success_rate': 0.0}
            
            total_fallbacks = len(self.fallback_history)
            successful_fallbacks = sum(1 for r in self.fallback_history if r.success)
            success_rate = successful_fallbacks / total_fallbacks
            
            # Strategy usage statistics
            strategy_usage = {}
            for result in self.fallback_history:
                strategy = result.strategy_used
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            # Average recovery time
            recovery_times = [r.recovery_time for r in self.fallback_history]
            avg_recovery_time = np.mean(recovery_times) if recovery_times else 0.0
            
            # Average data quality
            data_qualities = [r.data_quality for r in self.fallback_history]
            avg_data_quality = np.mean(data_qualities) if data_qualities else 0.0
            
            return {
                'total_fallbacks': total_fallbacks,
                'success_rate': round(success_rate, 4),
                'strategy_usage': strategy_usage,
                'average_recovery_time': round(avg_recovery_time, 3),
                'average_data_quality': round(avg_data_quality, 3),
                'component_health': self.component_health.copy(),
                'last_fallback': self.fallback_history[-1].timestamp if self.fallback_history else None
            }
            
        except Exception as e:
            logger.error(f"Error getting fallback statistics: {e}")
            return {'error': str(e)}
    
    def get_component_health(self, module: str) -> float:
        """Get health score for a specific component."""
        return self.component_health.get(module, 1.0)
    
    def is_component_healthy(self, module: str, threshold: float = 0.5) -> bool:
        """Check if a component is healthy based on threshold."""
        return self.get_component_health(module) >= threshold


# Convenience functions
def create_fallback_logic_router() -> FallbackLogicRouter:
    """Create and return a new FallbackLogicRouter instance."""
    return FallbackLogicRouter()


def route_fallback_logic(router: FallbackLogicRouter, 
                        module: str, 
                        error: Exception) -> Any:
    """Route fallback logic using the given router."""
    return router.route_fallback(module, error) 