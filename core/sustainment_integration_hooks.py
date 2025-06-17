"""
Enhanced Sustainment Integration Hooks v3.0
==========================================

Deep mathematical integration layer that connects the 8-principle sustainment framework
with all system controllers. This version provides mathematical continuity across:

- Thermal Zone Management
- Cooldown Control Systems  
- Fractal Processing Core
- Quantum Antipole Engine
- GPU Flash Engine
- Profit Navigation
- Strategy Execution
- Visual Integration

Mathematical Foundation:
SI(t) = Σᵢ wᵢ Pᵢ(t) > S_crit for system sustainability
Each controller receives sustainment-guided corrections to maintain mathematical coherence.
"""

import numpy as np
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from abc import ABC, abstractmethod

# Import sustainment mathematical framework
try:
    from .mathlib_v3 import SustainmentMathLib, SustainmentVector, MathematicalContext, SustainmentPrinciple
    from .sustainment_principles import SustainmentCalculator
    from .sustainment_underlay_controller import SustainmentUnderlayController
except ImportError:
    # Fallback imports for development
    SustainmentMathLib = None
    SustainmentVector = None
    MathematicalContext = None

# Controller imports
try:
    from .thermal_zone_manager import ThermalZoneManager
    from .cooldown_manager import CooldownManager
    from .fractal_core import FractalCore
    from .quantum_antipole_engine import QuantumAntiPoleEngine
    from .gpu_flash_engine import GPUFlashEngine
    from .profit_navigator import AntiPoleProfitNavigator
    from .strategy_execution_mapper import StrategyExecutionMapper
    from .visual_integration_bridge import VisualIntegrationBridge
    from .gan_filter import SustainmentAwareGANFilter
    from .ufs_registry import UFSRegistry
    from .ufs_echo_logger import UFSEchoLogger
    EXTENDED_CONTROLLERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some controllers not available: {e}")
    EXTENDED_CONTROLLERS_AVAILABLE = False
    # Mock classes for testing
    class MockController:
        def __init__(self, *args, **kwargs):
            self.name = self.__class__.__name__
        def get_metrics(self):
            return {}
        def get_sustainment_metrics(self):
            return {}
        def apply_sustainment_correction(self, correction):
            return True
        def get_integration_weights(self):
            return {}

    ThermalZoneManager = MockController
    CooldownManager = MockController
    FractalCore = MockController
    QuantumAntiPoleEngine = MockController
    GPUFlashEngine = MockController
    AntiPoleProfitNavigator = MockController
    StrategyExecutionMapper = MockController
    VisualIntegrationBridge = MockController
    SustainmentAwareGANFilter = MockController
    UFSRegistry = MockController
    UFSEchoLogger = MockController

logger = logging.getLogger(__name__)

@dataclass
class ControllerSustainmentState:
    """Sustainment state for individual controllers"""
    controller_name: str
    sustainment_vector: 'SustainmentVector'
    performance_metrics: Dict[str, float]
    correction_actions: List[Dict[str, Any]]
    integration_weights: Dict[str, float]
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class SustainmentCorrection:
    """Mathematical correction directive for controllers"""
    target_controller: str
    principle: SustainmentPrinciple
    correction_type: str
    magnitude: float
    parameters: Dict[str, Any]
    priority: int = 1
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(seconds=30))

class ControllerIntegrationInterface(ABC):
    """Abstract interface for sustainment-aware controllers"""
    
    @abstractmethod
    def get_sustainment_metrics(self) -> Dict[str, float]:
        """Get current metrics for sustainment calculation"""
        pass
    
    @abstractmethod
    def apply_sustainment_correction(self, correction: SustainmentCorrection) -> bool:
        """Apply sustainment correction and return success status"""
        pass
    
    @abstractmethod
    def get_integration_weights(self) -> Dict[str, float]:
        """Get weights for integration with other controllers"""
        pass

class EnhancedSustainmentIntegrationHooks:
    """
    Enhanced sustainment integration system providing deep mathematical coherence
    across all system controllers through the 8-principle framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced sustainment integration
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize sustainment mathematical library
        if SustainmentMathLib:
            self.math_lib = SustainmentMathLib(
                sustainment_threshold=self.config.get('sustainment_threshold', 0.65),
                adaptation_rate=self.config.get('adaptation_rate', 0.02),
                correction_gain=self.config.get('correction_gain', 0.1)
            )
        else:
            self.math_lib = None
            logger.warning("SustainmentMathLib not available - using mock")
        
        # Controller registry
        self.controllers: Dict[str, Any] = {}
        self.controller_states: Dict[str, ControllerSustainmentState] = {}
        self.integration_interfaces: Dict[str, ControllerIntegrationInterface] = {}
        
        # Sustainment state tracking
        self.global_sustainment_history = deque(maxlen=1000)
        self.correction_history = deque(maxlen=500)
        self.integration_metrics = defaultdict(list)
        
        # Mathematical synthesis parameters
        self.synthesis_interval = self.config.get('synthesis_interval', 5.0)
        self.correction_interval = self.config.get('correction_interval', 2.0)
        self.emergency_threshold = self.config.get('emergency_threshold', 0.3)
        
        # Controller-specific thresholds
        self.controller_thresholds = {
            'thermal_zone': 0.7,
            'cooldown': 0.6,
            'fractal_core': 0.65,
            'quantum_engine': 0.75,
            'gpu_flash': 0.7,
            'profit_navigator': 0.8,
            'strategy_mapper': 0.6,
            'visual_bridge': 0.5,
            'gan_filter': 0.7,  # High threshold for anomaly detection
            'ufs_registry': 0.5,  # Lower threshold for file system
            'ufs_logger': 0.5     # Lower threshold for logging
        }
        
        # Integration weights (how much each controller affects others)
        self.cross_controller_weights = np.array([
            [1.0, 0.8, 0.6, 0.4, 0.7, 0.3, 0.5, 0.2],  # thermal_zone affects
            [0.8, 1.0, 0.5, 0.3, 0.6, 0.4, 0.7, 0.3],  # cooldown affects
            [0.6, 0.5, 1.0, 0.9, 0.4, 0.8, 0.9, 0.6],  # fractal_core affects
            [0.4, 0.3, 0.9, 1.0, 0.5, 0.9, 0.8, 0.7],  # quantum_engine affects
            [0.7, 0.6, 0.4, 0.5, 1.0, 0.6, 0.3, 0.4],  # gpu_flash affects
            [0.3, 0.4, 0.8, 0.9, 0.6, 1.0, 0.9, 0.5],  # profit_navigator affects
            [0.5, 0.7, 0.9, 0.8, 0.3, 0.9, 1.0, 0.8],  # strategy_mapper affects
            [0.2, 0.3, 0.6, 0.7, 0.4, 0.5, 0.8, 1.0]   # visual_bridge affects
        ])
        
        # Active corrections
        self.active_corrections: Dict[str, List[SustainmentCorrection]] = defaultdict(list)
        
        # Threading for continuous operation
        self.synthesis_active = False
        self.synthesis_thread = None
        self.correction_thread = None
        self._lock = threading.RLock()
        
        logger.info("Enhanced Sustainment Integration Hooks initialized")
    
    def register_controller(self, name: str, controller: Any, 
                          interface: Optional[ControllerIntegrationInterface] = None) -> None:
        """
        Register a controller for sustainment integration
        
        Args:
            name: Controller identifier
            controller: Controller instance
            interface: Optional sustainment interface implementation
        """
        with self._lock:
            self.controllers[name] = controller
            if interface:
                self.integration_interfaces[name] = interface
            
            # Initialize controller state
            if self.math_lib and MathematicalContext:
                initial_context = MathematicalContext()
                initial_vector = self.math_lib.calculate_sustainment_vector(initial_context)
            else:
                initial_vector = None
            
            self.controller_states[name] = ControllerSustainmentState(
                controller_name=name,
                sustainment_vector=initial_vector,
                performance_metrics={},
                correction_actions=[],
                integration_weights={}
            )
            
        logger.info(f"Registered controller: {name}")
    
    def start_continuous_integration(self) -> None:
        """Start continuous sustainment integration and correction"""
        if self.synthesis_active:
            logger.warning("Sustainment integration already active")
            return
        
        self.synthesis_active = True
        
        # Start synthesis thread
        self.synthesis_thread = threading.Thread(
            target=self._continuous_synthesis_loop,
            daemon=True
        )
        self.synthesis_thread.start()
        
        # Start correction thread
        self.correction_thread = threading.Thread(
            target=self._continuous_correction_loop,
            daemon=True
        )
        self.correction_thread.start()
        
        logger.info(f"Continuous sustainment integration started")
    
    def stop_continuous_integration(self) -> None:
        """Stop continuous integration"""
        self.synthesis_active = False
        
        if self.synthesis_thread and self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=5.0)
        if self.correction_thread and self.correction_thread.is_alive():
            self.correction_thread.join(timeout=5.0)
            
        logger.info("Sustainment integration stopped")
    
    def _continuous_synthesis_loop(self) -> None:
        """Main synthesis loop for mathematical integration"""
        while self.synthesis_active:
            try:
                start_time = time.time()
                
                # Gather metrics from all controllers
                controller_contexts = self._gather_controller_contexts()
                
                # Calculate cross-controller sustainment synthesis
                global_sustainment = self._calculate_global_sustainment(controller_contexts)
                
                # Update controller states
                self._update_controller_states(controller_contexts, global_sustainment)
                
                # Store history
                with self._lock:
                    self.global_sustainment_history.append({
                        'timestamp': datetime.now(),
                        'global_sustainment': global_sustainment,
                        'controller_count': len(controller_contexts),
                        'synthesis_time_ms': (time.time() - start_time) * 1000
                    })
                
                time.sleep(self.synthesis_interval)
                
            except Exception as e:
                logger.error(f"Error in sustainment synthesis: {e}")
                time.sleep(self.synthesis_interval)
    
    def _continuous_correction_loop(self) -> None:
        """Continuous correction application loop"""
        while self.synthesis_active:
            try:
                # Apply pending corrections
                self._apply_pending_corrections()
                
                # Generate new corrections if needed
                self._generate_sustainment_corrections()
                
                # Clean expired corrections
                self._clean_expired_corrections()
                
                time.sleep(self.correction_interval)
                
            except Exception as e:
                logger.error(f"Error in sustainment correction: {e}")
                time.sleep(self.correction_interval)
    
    def _gather_controller_contexts(self) -> Dict[str, MathematicalContext]:
        """Gather mathematical contexts from all registered controllers"""
        contexts = {}
        
        for name, controller in self.controllers.items():
            try:
                # Get controller metrics
                if hasattr(controller, 'get_sustainment_metrics'):
                    metrics = controller.get_sustainment_metrics()
                elif hasattr(controller, 'get_metrics'):
                    metrics = controller.get_metrics()
                else:
                    metrics = self._extract_basic_metrics(controller)
                
                # Get thermal state if available
                thermal_state = {}
                if hasattr(controller, 'get_thermal_state'):
                    thermal_state = controller.get_thermal_state()
                
                # Get GPU state if available
                gpu_state = {}
                if hasattr(controller, 'get_gpu_metrics'):
                    gpu_state = controller.get_gpu_metrics()
                
                # Create mathematical context
                if MathematicalContext:
                    context = MathematicalContext(
                        current_state=metrics,
                        system_metrics=metrics,
                        thermal_state=thermal_state,
                        gpu_state=gpu_state
                    )
                else:
                    context = None
                
                contexts[name] = context
                
            except Exception as e:
                logger.error(f"Error gathering context for {name}: {e}")
                contexts[name] = None
        
        return contexts
    
    def _extract_basic_metrics(self, controller: Any) -> Dict[str, float]:
        """Extract basic metrics from controller if no specific interface"""
        metrics = {}
        
        # Common attributes to look for
        common_attrs = [
            'temperature', 'cpu_usage', 'memory_usage', 'gpu_usage',
            'latency', 'throughput', 'error_rate', 'uptime',
            'profit', 'efficiency', 'complexity', 'entropy'
        ]
        
        for attr in common_attrs:
            if hasattr(controller, attr):
                value = getattr(controller, attr)
                if isinstance(value, (int, float)):
                    metrics[attr] = float(value)
        
        return metrics
    
    def _calculate_global_sustainment(self, controller_contexts: Dict[str, MathematicalContext]) -> 'SustainmentVector':
        """Calculate global sustainment state across all controllers"""
        if not self.math_lib or not controller_contexts:
            return None
        
        try:
            # Calculate individual sustainment vectors
            individual_vectors = {}
            for name, context in controller_contexts.items():
                if context:
                    vector = self.math_lib.calculate_sustainment_vector(context)
                    individual_vectors[name] = vector
            
            if not individual_vectors:
                return None
            
            # Aggregate into global sustainment vector
            controller_names = list(individual_vectors.keys())
            principle_matrix = np.array([v.principles for v in individual_vectors.values()])
            confidence_matrix = np.array([v.confidence for v in individual_vectors.values()])
            
            # Apply cross-controller weights
            n_controllers = len(controller_names)
            if n_controllers <= len(self.cross_controller_weights):
                weights = self.cross_controller_weights[:n_controllers, :n_controllers]
                
                # Weighted average considering controller interactions
                weighted_principles = np.average(principle_matrix, axis=0, weights=np.mean(weights, axis=1))
                weighted_confidence = np.average(confidence_matrix, axis=0, weights=np.mean(weights, axis=1))
            else:
                # Simple average if too many controllers
                weighted_principles = np.mean(principle_matrix, axis=0)
                weighted_confidence = np.mean(confidence_matrix, axis=0)
            
            # Create global sustainment vector
            if SustainmentVector:
                global_vector = SustainmentVector(
                    principles=weighted_principles,
                    confidence=weighted_confidence,
                    timestamp=datetime.now(),
                    metadata={
                        'controller_count': len(individual_vectors),
                        'individual_vectors': {name: v.sustainment_index() for name, v in individual_vectors.items()}
                    }
                )
            else:
                global_vector = None
            
            return global_vector
            
        except Exception as e:
            logger.error(f"Error calculating global sustainment: {e}")
            return None
    
    def _update_controller_states(self, controller_contexts: Dict[str, MathematicalContext], 
                                global_sustainment: 'SustainmentVector') -> None:
        """Update individual controller sustainment states"""
        with self._lock:
            for name, context in controller_contexts.items():
                if name in self.controller_states and context and self.math_lib:
                    # Calculate individual sustainment
                    individual_vector = self.math_lib.calculate_sustainment_vector(context)
                    
                    # Update controller state
                    self.controller_states[name].sustainment_vector = individual_vector
                    self.controller_states[name].performance_metrics = context.system_metrics
                    self.controller_states[name].last_update = datetime.now()
                    
                    # Calculate integration weights with other controllers
                    if global_sustainment:
                        integration_weights = self._calculate_integration_weights(
                            individual_vector, global_sustainment, name
                        )
                        self.controller_states[name].integration_weights = integration_weights
    
    def _calculate_integration_weights(self, individual_vector: 'SustainmentVector', 
                                     global_vector: 'SustainmentVector', 
                                     controller_name: str) -> Dict[str, float]:
        """Calculate how this controller should integrate with others"""
        if not individual_vector or not global_vector:
            return {}
        
        weights = {}
        
        # Calculate integration need based on principle gaps
        individual_si = individual_vector.sustainment_index()
        global_si = global_vector.sustainment_index()
        
        # Controllers with lower sustainment need more integration
        integration_need = max(0.0, global_si - individual_si)
        
        # Calculate weights for each principle
        for i, principle in enumerate(SustainmentPrinciple):
            principle_gap = global_vector.principles[i] - individual_vector.principles[i]
            principle_weight = max(0.0, principle_gap) * integration_need
            weights[principle.value] = float(principle_weight)
        
        return weights
    
    def _generate_sustainment_corrections(self) -> None:
        """Generate correction actions for controllers below thresholds"""
        with self._lock:
            current_time = datetime.now()
            
            for name, state in self.controller_states.items():
                if not state.sustainment_vector:
                    continue
                
                si = state.sustainment_vector.sustainment_index()
                threshold = self.controller_thresholds.get(name, 0.65)
                
                # Generate corrections if below threshold
                if si < threshold:
                    corrections = self._generate_controller_corrections(name, state, threshold)
                    
                    # Add to active corrections
                    for correction in corrections:
                        correction.expires_at = current_time + timedelta(seconds=30)
                        self.active_corrections[name].append(correction)
                
                # Emergency corrections for critically low sustainment
                if si < self.emergency_threshold:
                    emergency_corrections = self._generate_emergency_corrections(name, state)
                    for correction in emergency_corrections:
                        correction.priority = 10  # High priority
                        correction.expires_at = current_time + timedelta(seconds=60)
                        self.active_corrections[name].append(correction)
    
    def _generate_controller_corrections(self, controller_name: str, 
                                       state: ControllerSustainmentState,
                                       threshold: float) -> List[SustainmentCorrection]:
        """Generate specific corrections for a controller"""
        corrections = []
        
        if not state.sustainment_vector:
            return corrections
        
        failing_principles = state.sustainment_vector.failing_principles(threshold)
        
        for principle in failing_principles:
            correction = self._create_principle_correction(controller_name, principle, state)
            if correction:
                corrections.append(correction)
        
        return corrections
    
    def _create_principle_correction(self, controller_name: str, 
                                   principle: SustainmentPrinciple,
                                   state: ControllerSustainmentState) -> Optional[SustainmentCorrection]:
        """Create specific correction for a principle"""
        if not SustainmentCorrection:
            return None
        
        # Define correction strategies per principle and controller type
        correction_strategies = {
            SustainmentPrinciple.ANTICIPATION: {
                'thermal_zone': ('increase_prediction_window', {'window_size': 1.2}),
                'fractal_core': ('enhance_forecasting', {'depth': 1.1}),
                'quantum_engine': ('improve_state_prediction', {'tau': 0.9}),
                'gan_filter': ('improve_anomaly_prediction', {'sensitivity': 1.1}),
            },
            SustainmentPrinciple.RESPONSIVENESS: {
                'thermal_zone': ('reduce_response_time', {'factor': 0.8}),
                'gpu_flash': ('increase_batch_size', {'multiplier': 1.1}),
                'cooldown': ('decrease_cooldown_time', {'factor': 0.9}),
                'gan_filter': ('reduce_anomaly_threshold', {'factor': 0.9}),
            },
            SustainmentPrinciple.ECONOMY: {
                'gpu_flash': ('optimize_memory_usage', {'efficiency_target': 1.1}),
                'quantum_engine': ('reduce_computation_cost', {'optimization': 1.2}),
                'profit_navigator': ('increase_profit_targeting', {'factor': 1.1}),
                'gan_filter': ('optimize_batch_size', {'factor': 0.9}),
            },
            SustainmentPrinciple.SIMPLICITY: {
                'strategy_mapper': ('reduce_complexity', {'prune_factor': 0.9}),
                'fractal_core': ('simplify_computation', {'terms_reduction': 0.9}),
                'gan_filter': ('cleanup_clusters', {'profit_threshold': 1.2}),
            },
            SustainmentPrinciple.SURVIVABILITY: {
                'thermal_zone': ('emergency_cooling', {'boost_factor': 2.0}),
                'gan_filter': ('emergency_reset', {'reset_optimizers': True}),
                'ufs_registry': ('cleanup_entries', {'max_age_hours': 24}),
            }
        }
        
        strategy = correction_strategies.get(principle, {}).get(controller_name)
        if not strategy:
            return None
        
        correction_type, parameters = strategy
        
        # Calculate magnitude based on principle deficit
        principle_value = state.sustainment_vector.principles[principle.value]
        magnitude = max(0.1, min(1.0, 1.0 - principle_value))
        
        return SustainmentCorrection(
            target_controller=controller_name,
            principle=principle,
            correction_type=correction_type,
            magnitude=magnitude,
            parameters=parameters,
            priority=5
        )
    
    def _generate_emergency_corrections(self, controller_name: str,
                                      state: ControllerSustainmentState) -> List[SustainmentCorrection]:
        """Generate emergency corrections for critically low sustainment"""
        corrections = []
        
        # Emergency actions per controller type
        emergency_actions = {
            'thermal_zone': [
                ('emergency_cooling', {'boost_factor': 2.0}),
                ('reduce_load', {'factor': 0.5})
            ],
            'gpu_flash': [
                ('memory_cleanup', {'aggressive': True}),
                ('reduce_batch_size', {'factor': 0.5})
            ],
            'quantum_engine': [
                ('simplify_calculations', {'emergency_mode': True}),
                ('reduce_field_size', {'factor': 0.7})
            ],
            'profit_navigator': [
                ('conservative_mode', {'risk_reduction': 0.3}),
                ('emergency_exit', {'threshold': 0.1})
            ],
            'gan_filter': [
                ('emergency_reset', {'reset_optimizers': True, 'clear_cache': True}),
                ('reduce_anomaly_threshold', {'factor': 0.5})
            ],
            'ufs_registry': [
                ('emergency_cleanup', {'clear_old_entries': True}),
                ('reduce_registry_size', {'max_entries': 1000})
            ],
            'ufs_logger': [
                ('flush_logs', {'immediate': True}),
                ('reduce_log_level', {'level': 'ERROR'})
            ]
        }
        
        actions = emergency_actions.get(controller_name, [])
        
        for action_type, parameters in actions:
            if SustainmentCorrection:
                correction = SustainmentCorrection(
                    target_controller=controller_name,
                    principle=SustainmentPrinciple.SURVIVABILITY,  # Emergency = survivability
                    correction_type=action_type,
                    magnitude=1.0,  # Maximum magnitude for emergency
                    parameters=parameters,
                    priority=10
                )
                corrections.append(correction)
        
        return corrections
    
    def _apply_pending_corrections(self) -> None:
        """Apply all pending corrections to controllers"""
        with self._lock:
            for controller_name, corrections in self.active_corrections.items():
                if controller_name not in self.controllers:
                    continue
                
                controller = self.controllers[controller_name]
                interface = self.integration_interfaces.get(controller_name)
                
                # Sort by priority (higher priority first)
                corrections.sort(key=lambda c: c.priority, reverse=True)
                
                applied_corrections = []
                
                for correction in corrections:
                    try:
                        success = False
                        
                        # Try interface first if available
                        if interface:
                            success = interface.apply_sustainment_correction(correction)
                        else:
                            # Try direct application
                            success = self._apply_direct_correction(controller, correction)
                        
                        if success:
                            applied_corrections.append(correction)
                            
                            # Log correction
                            self.correction_history.append({
                                'timestamp': datetime.now(),
                                'controller': controller_name,
                                'principle': correction.principle.value,
                                'type': correction.correction_type,
                                'magnitude': correction.magnitude,
                                'success': True
                            })
                        
                    except Exception as e:
                        logger.error(f"Error applying correction to {controller_name}: {e}")
                
                # Remove applied corrections
                for correction in applied_corrections:
                    corrections.remove(correction)
    
    def _apply_direct_correction(self, controller: Any, correction: SustainmentCorrection) -> bool:
        """Apply correction directly to controller if no interface available"""
        try:
            # Try common correction methods
            method_name = f'apply_{correction.correction_type}'
            if hasattr(controller, method_name):
                method = getattr(controller, method_name)
                method(**correction.parameters)
                return True
            
            # Try generic parameter setting
            if hasattr(controller, 'set_parameter'):
                controller.set_parameter(correction.correction_type, correction.magnitude)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Direct correction application failed: {e}")
            return False
    
    def _clean_expired_corrections(self) -> None:
        """Remove expired corrections"""
        current_time = datetime.now()
        
        with self._lock:
            for controller_name in list(self.active_corrections.keys()):
                corrections = self.active_corrections[controller_name]
                active_corrections = [c for c in corrections if c.expires_at > current_time]
                self.active_corrections[controller_name] = active_corrections
                
                # Remove empty lists
                if not active_corrections:
                    del self.active_corrections[controller_name]
    
    # === PUBLIC INTERFACE ===
    
    def get_global_sustainment_state(self) -> Dict[str, Any]:
        """Get current global sustainment state"""
        with self._lock:
            if not self.global_sustainment_history:
                return {'status': 'no_data'}
            
            latest = self.global_sustainment_history[-1]
            global_sustainment = latest['global_sustainment']
            
            if global_sustainment:
                return {
                    'sustainment_index': global_sustainment.sustainment_index(),
                    'is_sustainable': global_sustainment.is_sustainable(),
                    'failing_principles': [p.value for p in global_sustainment.failing_principles()],
                    'principle_values': global_sustainment.principles.tolist(),
                    'confidence_values': global_sustainment.confidence.tolist(),
                    'controller_count': latest['controller_count'],
                    'last_update': latest['timestamp'],
                    'synthesis_time_ms': latest['synthesis_time_ms']
                }
            else:
                return {'status': 'calculation_error'}
    
    def get_controller_sustainment_states(self) -> Dict[str, Dict[str, Any]]:
        """Get sustainment states for all controllers"""
        states = {}
        
        with self._lock:
            for name, state in self.controller_states.items():
                if state.sustainment_vector:
                    states[name] = {
                        'sustainment_index': state.sustainment_vector.sustainment_index(),
                        'is_sustainable': state.sustainment_vector.is_sustainable(),
                        'principle_values': state.sustainment_vector.principles.tolist(),
                        'confidence_values': state.sustainment_vector.confidence.tolist(),
                        'integration_weights': state.integration_weights,
                        'active_corrections': len(self.active_corrections.get(name, [])),
                        'last_update': state.last_update
                    }
                else:
                    states[name] = {'status': 'no_vector'}
        
        return states
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics"""
        with self._lock:
            total_corrections = sum(len(corrections) for corrections in self.active_corrections.values())
            successful_corrections = sum(1 for entry in self.correction_history if entry['success'])
            total_correction_attempts = len(self.correction_history)
            
            success_rate = successful_corrections / total_correction_attempts if total_correction_attempts > 0 else 0.0
            
            return {
                'active_corrections': total_corrections,
                'success_rate': success_rate,
                'total_attempts': total_correction_attempts,
                'registered_controllers': len(self.controllers),
                'synthesis_active': self.synthesis_active,
                'avg_synthesis_time': np.mean([h['synthesis_time_ms'] for h in self.global_sustainment_history]) if self.global_sustainment_history else 0.0
            }
    
    def force_sustainment_correction(self, controller_name: str, 
                                   principle: SustainmentPrinciple,
                                   magnitude: float = 1.0) -> bool:
        """Force immediate sustainment correction for testing/emergency"""
        if controller_name not in self.controllers:
            return False
        
        if not SustainmentCorrection:
            return False
        
        correction = SustainmentCorrection(
            target_controller=controller_name,
            principle=principle,
            correction_type='manual_correction',
            magnitude=magnitude,
            parameters={'manual': True},
            priority=15  # Highest priority
        )
        
        with self._lock:
            self.active_corrections[controller_name].append(correction)
        
        return True

# === TESTING UTILITIES ===

def create_test_integration_system() -> EnhancedSustainmentIntegrationHooks:
    """Create test integration system with mock controllers"""
    config = {
        'sustainment_threshold': 0.65,
        'adaptation_rate': 0.02,
        'correction_gain': 0.1,
        'synthesis_interval': 1.0,
        'correction_interval': 0.5
    }
    
    integration = EnhancedSustainmentIntegrationHooks(config)
    
    # Register mock controllers
    mock_controllers = [
        ('thermal_zone', ThermalZoneManager()),
        ('cooldown', CooldownManager()),
        ('fractal_core', FractalCore()),
        ('quantum_engine', QuantumAntiPoleEngine()),
        ('gpu_flash', GPUFlashEngine()),
        ('gan_filter', SustainmentAwareGANFilter({}, 32)),
        ('ufs_registry', UFSRegistry()),
        ('ufs_logger', UFSEchoLogger())
    ]
    
    for name, controller in mock_controllers:
        try:
            integration.register_controller(name, controller)
        except Exception as e:
            logger.warning(f"Failed to register {name}: {e}")
    
    return integration

if __name__ == "__main__":
    # Basic testing
    integration = create_test_integration_system()
    integration.start_continuous_integration()
    
    time.sleep(10)  # Let it run for 10 seconds
    
    print("Global State:", integration.get_global_sustainment_state())
    print("Integration Metrics:", integration.get_integration_metrics())
    
    integration.stop_continuous_integration() 