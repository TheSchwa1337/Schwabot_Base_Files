"""
GPU Flash Engine v0.5
=====================

Quantum-coherent GPU flash orchestrator with fractal risk awareness.
Integrates with global event lattice for phase-locked decision cascades.

Mathematical Foundation:
Ψ(flash) = ∫(entropy × phase × matrix_stability) dt
Where each flash decision ripples through the event manifold.
"""

from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, asdict
import numpy as np
import time
import json
from pathlib import Path
import logging
import threading
from datetime import datetime

# Core Architecture Imports
from .bus_core import BusCore, BusEvent
from .config_utils import load_yaml_config
from .cursor_math_integration import CursorMath, PhaseShell

logger = logging.getLogger(__name__)

@dataclass
class FlashState:
    """Quantum state snapshot of a GPU flash operation"""
    timestamp: float
    z_score: float
    phase_angle: float
    entropy_class: str
    matrix_state: str
    is_safe: bool
    risk_entropy: float
    fractal_depth: int = 0
    event_id: str = ""
    coherence_score: float = 0.0
    binding_energy: float = 7.5

@dataclass
class GPUFlashConfig:
    """Configuration parameters for GPU Flash Engine"""
    cooldown_period: float = 0.1
    binding_energy_default: float = 7.5
    risk_thresholds: Dict[str, float] = None
    max_cascade_memory: int = 100
    max_history_size: int = 1000
    enable_fractal_corrections: bool = True
    
    def __post_init__(self):
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            }

class GPUFlashEngine:
    """
    Quantum-coherent GPU flash orchestrator.
    
    Subscribes to: entropy.update, phase.drift, risk.cascade, system.shutdown
    Publishes: flash.decision, flash.executed, flash.blocked, anomaly.detected
    """
    
    def __init__(self, config_path: Optional[str] = None, bus_core: Optional[BusCore] = None):
        self._lock = threading.Lock()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Core components
        self.math_core = CursorMath()
        self.bus_core = bus_core or BusCore()
        self.flash_history: List[FlashState] = []
        self.last_flash_time: Optional[float] = None
        
        # Fractal memory fields (Ψ-recursive)
        self.phase_memory: List[float] = []
        self.entropy_cascade: List[float] = []
        self.fractal_depth: int = 0
        self.coherence_history: List[float] = []
        
        # State file path
        self.state_file = Path("data/gpu_flash_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Event bus subscriptions
        self._setup_event_handlers()
        
        # Load previous state if exists
        self._load_state()
        
        logger.info("GPU Flash Engine v0.5 initialized with quantum coherence")
    
    def _load_config(self, config_path: Optional[str]) -> GPUFlashConfig:
        """Load configuration from YAML file or use defaults"""
        try:
            if config_path:
                config_data = load_yaml_config(Path(config_path))
            else:
                # Try to find default config
                default_path = Path(__file__).parent.parent / "config" / "gpu_flash_config.yaml"
                if default_path.exists():
                    config_data = load_yaml_config(default_path)
                else:
                    logger.warning("No config file found, using defaults")
                    config_data = {}
            
            return GPUFlashConfig(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return GPUFlashConfig()
    
    def _setup_event_handlers(self):
        """Wire up event-driven architecture"""
        self.bus_core.register_handler('entropy.update', self._handle_entropy_update)
        self.bus_core.register_handler('phase.drift', self._handle_phase_drift)
        self.bus_core.register_handler('risk.cascade', self._handle_risk_cascade)
        self.bus_core.register_handler('system.shutdown', self._handle_shutdown)
        self.bus_core.register_handler('flash.request', self._handle_flash_request)
    
    def _handle_entropy_update(self, event: BusEvent):
        """React to global entropy changes"""
        with self._lock:
            z_score = event.data.get('z_score', 0) if isinstance(event.data, dict) else 0
            self.entropy_cascade.append(z_score)
            
            # Trim cascade memory
            if len(self.entropy_cascade) > self.config.max_cascade_memory:
                self.entropy_cascade = self.entropy_cascade[-self.config.max_cascade_memory:]
                
            logger.debug(f"Entropy update: z_score={z_score}")
    
    def _handle_phase_drift(self, event: BusEvent):
        """React to phase angle changes"""
        with self._lock:
            phase = event.data.get('phase_angle', 0) if isinstance(event.data, dict) else 0
            self.phase_memory.append(phase)
            
            # Trim phase memory
            if len(self.phase_memory) > self.config.max_cascade_memory:
                self.phase_memory = self.phase_memory[-self.config.max_cascade_memory:]
            
            # Detect phase coherence patterns
            if len(self.phase_memory) >= 3:
                self._check_phase_resonance()
                
            logger.debug(f"Phase drift: phase={phase}")
    
    def _handle_risk_cascade(self, event: BusEvent):
        """React to cascading risk events"""
        with self._lock:
            self.fractal_depth = event.data.get('cascade_depth', 0) if isinstance(event.data, dict) else 0
            logger.debug(f"Risk cascade: depth={self.fractal_depth}")
    
    def _handle_shutdown(self, event: BusEvent):
        """Handle system shutdown - save state"""
        self._save_state()
        logger.info("GPU Flash Engine state saved for shutdown")
    
    def _handle_flash_request(self, event: BusEvent):
        """Handle flash requests from other modules"""
        try:
            data = event.data if isinstance(event.data, dict) else {}
            z_score = data.get('z_score', 0.0)
            phase_angle = data.get('phase_angle', 0.0)
            context = data.get('context', {})
            
            is_permitted, reason, state = self.check_flash_permission(
                z_score, phase_angle, context
            )
            
            # Publish result
            result_event = BusEvent(
                type="flash.result",
                data={
                    'permitted': is_permitted,
                    'reason': reason,
                    'state': asdict(state),
                    'request_id': data.get('request_id', 'unknown')
                },
                timestamp=time.time()
            )
            self.bus_core.dispatch_event(result_event)
            
        except Exception as e:
            logger.error(f"Error handling flash request: {e}")
    
    def check_flash_permission(self, z_score: float, phase_angle: float, 
                             context: Optional[Dict] = None) -> Tuple[bool, str, FlashState]:
        """
        Quantum-coherent flash permission check.
        
        Returns: (is_permitted, reason, flash_state)
        """
        with self._lock:
            current_time = time.time()
            event_id = f"flash_{int(current_time * 1000)}"
            
            # Cooldown check
            if (self.last_flash_time is not None and 
                current_time - self.last_flash_time < self.config.cooldown_period):
                state = self._create_flash_state(
                    current_time, z_score, phase_angle, 
                    'cooldown', 'blocked', False, event_id
                )
                return False, "cooldown_period", state
            
            # Entropy classification
            entropy_class = self.math_core.classify_entropy_shell(z_score)
            if entropy_class == 'critical_bloom':
                state = self._create_flash_state(
                    current_time, z_score, phase_angle,
                    entropy_class, 'critical_entropy', False, event_id
                )
                self._publish_anomaly(state)
                return False, "critical_entropy", state
            
            # Phase shell analysis
            phase_shell = self.math_core.classify_phase_shell(phase_angle)
            if phase_shell.shell_type == 'symmetry':
                state = self._create_flash_state(
                    current_time, z_score, phase_angle,
                    entropy_class, 'symmetry_lockout', False, event_id
                )
                return False, "symmetry_zone", state
            
            # Compute fractal-aware matrix stability
            binding_energy = self._calculate_dynamic_binding_energy(context)
            matrix_state = self.math_core.compute_matrix_stability(
                binding_energy=binding_energy,
                phase_stability=phase_shell.stability,
                entropy_class=entropy_class
            )
            
            # Calculate unified risk metric
            risk_entropy = self._calculate_risk_entropy(
                z_score, phase_angle, phase_shell.stability
            )
            
            is_safe = (matrix_state == 'matrix_safe' and 
                      risk_entropy < self.config.risk_thresholds['high'])
            
            # Create state record
            state = self._create_flash_state(
                current_time, z_score, phase_angle,
                entropy_class, matrix_state, is_safe, 
                event_id, risk_entropy, binding_energy
            )
            
            # Record and publish
            self.flash_history.append(state)
            self._trim_history()
            
            if is_safe:
                self.last_flash_time = current_time
                self._publish_flash_event('flash.executed', state)
            else:
                self._publish_flash_event('flash.blocked', state)
            
            # Persist state periodically
            if len(self.flash_history) % 10 == 0:
                self._save_state()
            
            return is_safe, matrix_state, state
    
    def _create_flash_state(self, timestamp: float, z_score: float, 
                          phase_angle: float, entropy_class: str,
                          matrix_state: str, is_safe: bool, 
                          event_id: str, risk_entropy: float = 0.0,
                          binding_energy: float = 7.5) -> FlashState:
        """Factory for FlashState with computed risk"""
        if risk_entropy == 0.0:
            risk_entropy = self._calculate_risk_entropy(z_score, phase_angle, 0.5)
        
        # Calculate coherence score from recent history
        coherence_score = self._calculate_coherence_score()
        
        return FlashState(
            timestamp=timestamp,
            z_score=z_score,
            phase_angle=phase_angle,
            entropy_class=entropy_class,
            matrix_state=matrix_state,
            is_safe=is_safe,
            risk_entropy=risk_entropy,
            fractal_depth=self.fractal_depth,
            event_id=event_id,
            coherence_score=coherence_score,
            binding_energy=binding_energy
        )
    
    def _calculate_risk_entropy(self, z_score: float, phase_angle: float, 
                              phase_stability: float) -> float:
        """
        Unified risk metric using quantum superposition principle:
        risk = |z|³ + |sin(φ)| + (1 - phase_stability)²
        """
        z_component = abs(z_score) ** 3
        phase_component = abs(np.sin(phase_angle))
        stability_component = (1 - phase_stability) ** 2
        
        # Normalize to [0, 1]
        risk = (z_component + phase_component + stability_component) / 3
        return min(1.0, risk)
    
    def _calculate_dynamic_binding_energy(self, context: Optional[Dict]) -> float:
        """
        Calculate binding energy based on system context and history.
        Implements Ψ-recursive memory field influence.
        """
        base_energy = self.config.binding_energy_default
        
        # Adjust based on recent history
        if len(self.flash_history) > 0:
            recent_failures = sum(1 for s in self.flash_history[-10:] if not s.is_safe)
            history_penalty = recent_failures * 0.5
            base_energy += history_penalty
        
        # Adjust based on cascade depth
        fractal_penalty = self.fractal_depth * 0.3
        base_energy += fractal_penalty
        
        # Context-specific adjustments
        if context:
            if context.get('high_volatility', False):
                base_energy *= 1.2
            if context.get('news_event', False):
                base_energy *= 1.1
            if context.get('market_stress', False):
                base_energy *= 1.15
        
        return base_energy
    
    def _calculate_coherence_score(self) -> float:
        """Calculate quantum coherence based on recent flash history"""
        if len(self.flash_history) < 3:
            return 0.0
        
        recent_states = self.flash_history[-3:]
        phases = [s.phase_angle for s in recent_states]
        
        # Calculate phase variance as inverse of coherence
        phase_variance = np.var(phases)
        coherence = max(0.0, 1.0 - phase_variance)
        
        self.coherence_history.append(coherence)
        if len(self.coherence_history) > 100:
            self.coherence_history = self.coherence_history[-100:]
        
        return coherence
    
    def _check_phase_resonance(self):
        """
        Detect quantum phase resonance patterns.
        When detected, publishes phase.resonance event.
        """
        if len(self.phase_memory) < 3:
            return
        
        # Check for phase locking (consecutive similar phases)
        recent_phases = self.phase_memory[-3:]
        phase_variance = np.var(recent_phases)
        
        if phase_variance < 0.01:  # Phase locked
            resonance_event = BusEvent(
                type='phase.resonance',
                data={
                    'type': 'locked',
                    'phases': recent_phases,
                    'variance': phase_variance,
                    'coherence': 1.0 - phase_variance
                },
                timestamp=time.time()
            )
            self.bus_core.dispatch_event(resonance_event)
    
    def _publish_anomaly(self, state: FlashState):
        """Publish anomaly detection to event bus"""
        anomaly_event = BusEvent(
            type='anomaly.detected',
            data={
                'source': 'gpu_flash',
                'severity': 'critical',
                'state': asdict(state),
                'risk_entropy': state.risk_entropy,
                'fractal_depth': state.fractal_depth
            },
            timestamp=time.time()
        )
        self.bus_core.dispatch_event(anomaly_event)
    
    def _publish_flash_event(self, event_type: str, state: FlashState):
        """Publish flash execution/blocking event"""
        flash_event = BusEvent(
            type=event_type,
            data={
                'state': asdict(state),
                'risk_entropy': state.risk_entropy,
                'coherence_score': state.coherence_score,
                'binding_energy': state.binding_energy
            },
            timestamp=time.time()
        )
        self.bus_core.dispatch_event(flash_event)
    
    def _trim_history(self):
        """Trim flash history to maintain memory bounds"""
        if len(self.flash_history) > self.config.max_history_size:
            self.flash_history = self.flash_history[-self.config.max_history_size:]
    
    def get_quantum_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive quantum statistics including:
        - Entropy distribution
        - Phase coherence metrics  
        - Risk evolution
        - Fractal depth analysis
        """
        if not self.flash_history:
            return {}
        
        # Basic stats
        entropy_stats = self.get_entropy_stats()
        phase_stats = self.get_phase_stats()
        
        # Advanced quantum metrics
        risk_evolution = [s.risk_entropy for s in self.flash_history]
        fractal_depths = [s.fractal_depth for s in self.flash_history]
        coherence_scores = [s.coherence_score for s in self.flash_history]
        
        return {
            'entropy': entropy_stats,
            'phase': phase_stats,
            'risk': {
                'mean': np.mean(risk_evolution) if risk_evolution else 0,
                'std': np.std(risk_evolution) if risk_evolution else 0,
                'trend': np.polyfit(range(len(risk_evolution)), risk_evolution, 1)[0] if len(risk_evolution) > 1 else 0,
                'current': risk_evolution[-1] if risk_evolution else 0
            },
            'fractal': {
                'max_depth': max(fractal_depths) if fractal_depths else 0,
                'mean_depth': np.mean(fractal_depths) if fractal_depths else 0,
                'current_depth': self.fractal_depth
            },
            'coherence': {
                'mean': np.mean(coherence_scores) if coherence_scores else 0,
                'current': coherence_scores[-1] if coherence_scores else 0,
                'trend': np.polyfit(range(len(coherence_scores)), coherence_scores, 1)[0] if len(coherence_scores) > 1 else 0
            },
            'safety_rate': sum(1 for s in self.flash_history if s.is_safe) / len(self.flash_history),
            'total_flashes': len(self.flash_history),
            'memory_usage': {
                'phase_memory': len(self.phase_memory),
                'entropy_cascade': len(self.entropy_cascade),
                'flash_history': len(self.flash_history)
            }
        }
    
    def _save_state(self):
        """Persist flash history and metrics"""
        try:
            state_data = {
                'flash_history': [asdict(s) for s in self.flash_history[-1000:]],  # Keep last 1000
                'phase_memory': self.phase_memory[-100:],  # Keep last 100
                'entropy_cascade': self.entropy_cascade[-100:],
                'coherence_history': self.coherence_history[-100:],
                'last_flash_time': self.last_flash_time,
                'fractal_depth': self.fractal_depth,
                'stats': self.get_quantum_stats(),
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug("GPU Flash state persisted")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Restore previous state if available"""
        try:
            if not self.state_file.exists():
                return
                
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore flash history
            if 'flash_history' in state_data:
                self.flash_history = [FlashState(**s) for s in state_data['flash_history']]
                
            # Restore memory fields
            self.phase_memory = state_data.get('phase_memory', [])
            self.entropy_cascade = state_data.get('entropy_cascade', [])
            self.coherence_history = state_data.get('coherence_history', [])
            self.last_flash_time = state_data.get('last_flash_time')
            self.fractal_depth = state_data.get('fractal_depth', 0)
            
            logger.info(f"Restored {len(self.flash_history)} flash states from {state_data.get('saved_at', 'unknown time')}")
            
        except Exception as e:
            logger.warning(f"Could not load previous state: {e}")
    
    # Backward compatibility methods
    def get_flash_history(self, limit: Optional[int] = None) -> List[FlashState]:
        """Get recent flash history"""
        with self._lock:
            if limit is None:
                return self.flash_history.copy()
            return self.flash_history[-limit:].copy()
    
    def clear_history(self) -> None:
        """Clear flash history"""
        with self._lock:
            self.flash_history.clear()
            self.phase_memory.clear()
            self.entropy_cascade.clear()
            self.coherence_history.clear()
            self.last_flash_time = None
            self.fractal_depth = 0
            
        logger.info("GPU Flash history cleared")
    
    def get_entropy_stats(self) -> Dict[str, float]:
        """Get statistics about entropy classifications"""
        if not self.flash_history:
            return {}
        
        total = len(self.flash_history)
        stats = {
            'critical_bloom': 0.0,
            'unstable': 0.0,
            'stable': 0.0
        }
        
        for state in self.flash_history:
            stats[state.entropy_class] = stats.get(state.entropy_class, 0) + 1
        
        return {k: v/total for k, v in stats.items()}
    
    def get_phase_stats(self) -> Dict[str, float]:
        """Get statistics about phase shell classifications"""
        if not self.flash_history:
            return {}
        
        total = len(self.flash_history)
        stats = {
            'symmetry': 0.0,
            'drift_positive': 0.0,
            'drift_negative': 0.0
        }
        
        for state in self.flash_history:
            shell = self.math_core.classify_phase_shell(state.phase_angle)
            stats[shell.shell_type] = stats.get(shell.shell_type, 0) + 1
        
        return {k: v/total for k, v in stats.items()}


# Standalone test/demo
if __name__ == "__main__":
    # Initialize with test config
    flash_engine = GPUFlashEngine()
    
    # Simulate some events
    flash_engine.bus_core.dispatch_event(BusEvent(
        type='entropy.update',
        data={'z_score': 1.5},
        timestamp=time.time()
    ))
    
    flash_engine.bus_core.dispatch_event(BusEvent(
        type='phase.drift',
        data={'phase_angle': np.pi + 0.2},
        timestamp=time.time()
    ))
    
    # Test flash permission
    is_permitted, reason, state = flash_engine.check_flash_permission(
        z_score=1.5,
        phase_angle=np.pi + 0.2,
        context={'high_volatility': True}
    )
    
    logger.info(f"Flash permitted: {is_permitted}, reason: {reason}")
    logger.info(f"Risk entropy: {state.risk_entropy:.3f}")
    
    # Get quantum statistics
    stats = flash_engine.get_quantum_stats()
    logger.info(f"Quantum stats: {stats}") 