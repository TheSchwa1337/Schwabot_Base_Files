"""
Multi-Phase Strategy Weight Tensor Module

Provides functionality for managing strategy weights across different market phases.
Implements recursive matrix weightings based on phase-encoded strategy signals derived from 
past performance, predictive vector fields, and momentum deviation logic.

Mathematical Framework:
⧈ Phase Tensor Assembly
Let Φᵢ(t) = strategy phase signal at time t
    Wᵢⱼ = weight between strategy i and j
    Tᵢⱼ = full tensor grid of pairwise weight interactions.

Tᵢⱼ(t) = Φᵢ(t) ⋅ ωᵢⱼ(t) + ΔΨ(t)

Where:
- ΔΨ(t) is the momentum drift correction from Ferris tick mapping
- ωᵢⱼ(t) is recursively updated via:
  ωᵢⱼ(t+1) = ωᵢⱼ(t) + α ⋅ (dΦⱼ/dt - dΦᵢ/dt)

⧈ Composite Tensor Evaluation
Final phase-vector for trade strategy execution:
S(t) = Σᵢⱼ Tᵢⱼ(t) ⋅ Pᵢ(t)

Where Pᵢ(t) is the positional profit vector normalized to entropy-corrected time states.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

# Check for mathematical infrastructure availability
try:
    from core.math_config_manager import MathConfigManager
    from core.math_cache import MathResultCache
    from core.math_orchestrator import MathOrchestrator
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    MathConfigManager = None
    MathResultCache = None
    MathOrchestrator = None


class MarketPhase(Enum):
    """Market phase enumeration."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"


class Status(Enum):
    """System status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PROCESSING = "processing"


@dataclass
class PhaseTensorConfig:
    """Configuration data class for phase tensor operations."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    decay_factor: float = 0.95
    learning_rate: float = 0.01
    momentum_drift_coefficient: float = 0.1  # α for momentum drift correction
    entropy_correction_factor: float = 0.05  # For entropy-corrected time states
    ferris_tick_window: int = 100  # Window for Ferris tick mapping


@dataclass
class PhaseTensorResult:
    """Result data class for phase tensor operations."""
    success: bool = False
    phase_tensor: Optional[np.ndarray] = None
    composite_signal: Optional[float] = None
    momentum_drift: Optional[float] = None
    weight_updates: Optional[np.ndarray] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class PhaseTensorCalculator:
    """Phase Tensor Calculator implementing the mathematical framework."""
    
    def __init__(self, config: Optional[PhaseTensorConfig] = None):
        self.config = config or PhaseTensorConfig()
        self.logger = logging.getLogger(f"{__name__}.PhaseTensorCalculator")
        self.previous_phase_signals = None
        self.ferris_tick_history = []
        
    def calculate_momentum_drift_correction(self, current_time: float, 
                                          phase_signals: np.ndarray) -> float:
        """
        Calculate momentum drift correction ΔΨ(t) from Ferris tick mapping.
        
        Args:
            current_time: Current time t
            phase_signals: Current phase signals Φᵢ(t)
            
        Returns:
            Momentum drift correction value
        """
        try:
            # Store Ferris tick data
            tick_data = {
                'time': current_time,
                'signals': phase_signals.copy(),
                'timestamp': time.time()
            }
            self.ferris_tick_history.append(tick_data)
            
            # Keep only recent history
            if len(self.ferris_tick_history) > self.config.ferris_tick_window:
                self.ferris_tick_history.pop(0)
            
            # Calculate momentum drift from recent history
            if len(self.ferris_tick_history) < 2:
                return 0.0
            
            # Calculate signal velocity over recent ticks
            recent_signals = np.array([tick['signals'] for tick in self.ferris_tick_history[-5:]])
            signal_velocity = np.mean(np.diff(recent_signals, axis=0), axis=0)
            
            # Momentum drift correction based on signal velocity variance
            momentum_drift = np.var(signal_velocity) * self.config.momentum_drift_coefficient
            
            self.logger.debug(f"Momentum drift correction: {momentum_drift:.6f}")
            return float(momentum_drift)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum drift: {e}")
            return 0.0
    
    def calculate_phase_tensor(self, phase_signals: np.ndarray, 
                             weight_matrix: np.ndarray,
                             current_time: float) -> np.ndarray:
        """
        Calculate phase tensor Tᵢⱼ(t) = Φᵢ(t) ⋅ ωᵢⱼ(t) + ΔΨ(t)
        
        Args:
            phase_signals: Strategy phase signals Φᵢ(t)
            weight_matrix: Weight matrix ωᵢⱼ(t)
            current_time: Current time t
            
        Returns:
            Phase tensor Tᵢⱼ(t)
        """
        try:
            # Calculate momentum drift correction
            momentum_drift = self.calculate_momentum_drift_correction(current_time, phase_signals)
            
            # Phase tensor calculation: Tᵢⱼ(t) = Φᵢ(t) ⋅ ωᵢⱼ(t) + ΔΨ(t)
            # Outer product of phase signals with weight matrix
            phase_tensor = np.outer(phase_signals, phase_signals) * weight_matrix
            
            # Add momentum drift correction
            phase_tensor += momentum_drift
            
            self.logger.debug(f"Phase tensor calculated: shape {phase_tensor.shape}")
            return phase_tensor
            
        except Exception as e:
            self.logger.error(f"Error calculating phase tensor: {e}")
            return np.zeros((len(phase_signals), len(phase_signals)))
    
    def update_weights_recursively(self, current_weights: np.ndarray,
                                 phase_signals: np.ndarray,
                                 learning_rate: float = None) -> np.ndarray:
        """
        Recursively update weights: ωᵢⱼ(t+1) = ωᵢⱼ(t) + α ⋅ (dΦⱼ/dt - dΦᵢ/dt)
        
        Args:
            current_weights: Current weight matrix ωᵢⱼ(t)
            phase_signals: Current phase signals Φᵢ(t)
            learning_rate: Learning rate α
            
        Returns:
            Updated weight matrix ωᵢⱼ(t+1)
        """
        try:
            if learning_rate is None:
                learning_rate = self.config.learning_rate
            
            # Calculate phase signal derivatives if we have previous signals
            if self.previous_phase_signals is not None:
                # Simple finite difference for derivatives
                dt = 1.0  # Assuming unit time step
                d_phase_dt = (phase_signals - self.previous_phase_signals) / dt
                
                # Calculate weight updates: α ⋅ (dΦⱼ/dt - dΦᵢ/dt)
                # Outer product of derivatives
                derivative_matrix = np.outer(d_phase_dt, d_phase_dt)
                weight_updates = learning_rate * derivative_matrix
                
                # Apply updates
                updated_weights = current_weights + weight_updates
                
                # Ensure weights remain positive
                updated_weights = np.maximum(updated_weights, 0.0)
                
                self.logger.debug(f"Weights updated with learning rate {learning_rate}")
            else:
                updated_weights = current_weights
            
            # Store current signals for next iteration
            self.previous_phase_signals = phase_signals.copy()
            
            return updated_weights
            
        except Exception as e:
            self.logger.error(f"Error updating weights recursively: {e}")
            return current_weights
    
    def calculate_composite_signal(self, phase_tensor: np.ndarray,
                                 profit_vectors: np.ndarray) -> float:
        """
        Calculate composite signal: S(t) = Σᵢⱼ Tᵢⱼ(t) ⋅ Pᵢ(t)
        
        Args:
            phase_tensor: Phase tensor Tᵢⱼ(t)
            profit_vectors: Positional profit vectors Pᵢ(t)
            
        Returns:
            Composite signal S(t)
        """
        try:
            # Apply entropy correction to profit vectors
            entropy_correction = 1.0 + self.config.entropy_correction_factor * np.random.normal(0, 1)
            corrected_profits = profit_vectors * entropy_correction
            
            # Calculate composite signal: S(t) = Σᵢⱼ Tᵢⱼ(t) ⋅ Pᵢ(t)
            # This is equivalent to: phase_tensor * corrected_profits (element-wise) then sum
            composite_signal = np.sum(phase_tensor * corrected_profits)
            
            self.logger.debug(f"Composite signal calculated: {composite_signal:.6f}")
            return float(composite_signal)
            
        except Exception as e:
            self.logger.error(f"Error calculating composite signal: {e}")
            return 0.0


class MultiPhaseStrategyWeightTensor:
    """
    MultiPhaseStrategyWeightTensor Implementation
    Manages strategy weights across different market phases with recursive tensor operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = PhaseTensorConfig(**(config or {}))
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False

        # Strategy and phase management
        self.strategy_ids: List[str] = []
        self.num_strategies: int = 0
        self.num_phases: int = len(MarketPhase)
        self.weight_tensor: Optional[np.ndarray] = None
        self.phase_to_index: Dict[str, int] = {phase.value: i for i, phase in enumerate(MarketPhase)}
        self.current_phase: Optional[MarketPhase] = None
        
        # Phase tensor calculator
        self.phase_calculator = PhaseTensorCalculator(self.config)
        
        # Performance tracking
        self.metrics: Dict[str, Any] = {
            'total_updates': 0,
            'phase_transitions': 0,
            'last_update_time': time.time(),
            'active_phase': None,
            'tensor_operations': 0,
            'composite_signals': 0
        }

        # Initialize math infrastructure if available
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_config = MathConfigManager()
            self.math_cache = MathResultCache()
            self.math_orchestrator = MathOrchestrator()

        self._initialize_system()

    def _initialize_system(self) -> None:
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            self.initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False

    def activate(self) -> bool:
        if not self.initialized:
            self.logger.error("System not initialized")
            return False

        try:
            self.active = True
            self.logger.info(f"✅ {self.__class__.__name__} activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating {self.__class__.__name__}: {e}")
            return False

    def deactivate(self) -> bool:
        try:
            self.active = False
            self.logger.info(f"✅ {self.__class__.__name__} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
            return False

    def initialize_strategies(self, strategy_ids: List[str]) -> None:
        """Initialize the weight tensor with strategy IDs."""
        self.strategy_ids = strategy_ids
        self.num_strategies = len(strategy_ids)
        
        # Initialize weight tensor with equal weights
        self.weight_tensor = np.ones((self.num_strategies, self.num_strategies)) / self.num_strategies
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to ensure they sum to 1 for each phase."""
        if self.weight_tensor is None:
            return
            
        # Avoid division by zero if a column sums to 0
        col_sums = self.weight_tensor.sum(axis=0, keepdims=True)
        
        # Prevent division by zero if a column is all zeros
        col_sums[col_sums == 0] = 1.0
        
        self.weight_tensor = self.weight_tensor / col_sums

    def get_strategy_weights_for_phase(self, phase: MarketPhase) -> Dict[str, float]:
        """Retrieves the weights for all strategies given a specific market phase."""
        if phase.value not in self.phase_to_index:
            raise ValueError(f"Unknown market phase: {phase.value}")
        
        phase_idx = self.phase_to_index[phase.value]
        weights = self.weight_tensor[:, phase_idx]
        
        return {self.strategy_ids[i]: weights[i] for i in range(self.num_strategies)}

    def update_weights(self, identified_phase: MarketPhase, performance_feedback: Dict[str, Dict[str, float]]) -> None:
        """Adjusts strategy weights based on the identified market phase and performance feedback."""
        self.metrics['total_updates'] += 1
        self.metrics['last_update_time'] = time.time()

        if identified_phase != self.current_phase:
            self.metrics['phase_transitions'] += 1
            self.current_phase = identified_phase
            self.metrics['active_phase'] = self.current_phase.value
            self.logger.info(f"Market phase transitioned to: {identified_phase.value}")

        phase_idx = self.phase_to_index[identified_phase.value]

        # Apply decay to existing weights in the current phase
        self.weight_tensor[:, phase_idx] *= self.config.decay_factor

        # Update weights based on performance feedback
        for strategy_id, feedback in performance_feedback.items():
            if strategy_id in self.strategy_ids:
                strategy_idx = self.strategy_ids.index(strategy_id)
                performance_score = feedback.get('performance', 0.0)
                
                # Update weight based on performance
                self.weight_tensor[strategy_idx, phase_idx] += performance_score * self.config.learning_rate

        # Normalize weights after updates
        self._normalize_weights()

    def calculate_phase_tensor_signal(self, phase_signals: Union[List, np.ndarray],
                                    profit_vectors: Union[List, np.ndarray],
                                    current_time: float = None) -> PhaseTensorResult:
        """
        Calculate phase tensor and composite signal according to the mathematical framework.
        
        Args:
            phase_signals: Strategy phase signals Φᵢ(t)
            profit_vectors: Positional profit vectors Pᵢ(t)
            current_time: Current time t
            
        Returns:
            Phase tensor result with all calculations
        """
        try:
            if not self.active:
                return PhaseTensorResult(success=False, error="System not active")
            
            if current_time is None:
                current_time = time.time()
            
            # Convert to numpy arrays
            phase_array = np.array(phase_signals)
            profit_array = np.array(profit_vectors)
            
            # Ensure arrays have correct dimensions
            if len(phase_array) != self.num_strategies:
                return PhaseTensorResult(success=False, error="Phase signals dimension mismatch")
            
            if len(profit_array) != self.num_strategies:
                return PhaseTensorResult(success=False, error="Profit vectors dimension mismatch")
            
            # Calculate phase tensor: Tᵢⱼ(t) = Φᵢ(t) ⋅ ωᵢⱼ(t) + ΔΨ(t)
            phase_tensor = self.phase_calculator.calculate_phase_tensor(
                phase_array, self.weight_tensor, current_time)
            
            # Update weights recursively: ωᵢⱼ(t+1) = ωᵢⱼ(t) + α ⋅ (dΦⱼ/dt - dΦᵢ/dt)
            updated_weights = self.phase_calculator.update_weights_recursively(
                self.weight_tensor, phase_array, self.config.learning_rate)
            
            # Calculate composite signal: S(t) = Σᵢⱼ Tᵢⱼ(t) ⋅ Pᵢ(t)
            composite_signal = self.phase_calculator.calculate_composite_signal(
                phase_tensor, profit_array)
            
            # Update metrics
            self.metrics['tensor_operations'] += 1
            self.metrics['composite_signals'] += 1
            
            # Update weight tensor
            self.weight_tensor = updated_weights
            
            return PhaseTensorResult(
                success=True,
                phase_tensor=phase_tensor,
                composite_signal=composite_signal,
                momentum_drift=self.phase_calculator.calculate_momentum_drift_correction(current_time, phase_array),
                weight_updates=updated_weights - self.weight_tensor,
                data={
                    'phase_signals': phase_signals,
                    'profit_vectors': profit_vectors,
                    'current_time': current_time,
                    'num_strategies': self.num_strategies
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in phase tensor calculation: {e}")
            return PhaseTensorResult(success=False, error=str(e))

    def get_status(self) -> Dict[str, Any]:
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config.__dict__,
            'metrics': self.metrics,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'num_strategies': self.num_strategies,
            'weight_tensor_shape': self.weight_tensor.shape if self.weight_tensor is not None else None,
        }

    def process_strategy_data(self, data: Union[List, Tuple, np.ndarray]) -> float:
        """Process strategy data and return a composite signal."""
        try:
            # This is a simplified interface for backward compatibility
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                phase_signals = data[0]
                profit_vectors = data[1]
                current_time = data[2] if len(data) > 2 else time.time()
                
                result = self.calculate_phase_tensor_signal(phase_signals, profit_vectors, current_time)
                return result.composite_signal if result.success else 0.0
            else:
                self.logger.warning("Invalid data format for process_strategy_data")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error processing strategy data: {e}")
            return 0.0


def create_multi_phase_strategy_weight_tensor(config: Optional[Dict[str, Any]] = None) -> MultiPhaseStrategyWeightTensor:
    """Create a multi-phase strategy weight tensor instance."""
    return MultiPhaseStrategyWeightTensor(config)
