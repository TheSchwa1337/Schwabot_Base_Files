"""
Multi-Phase Strategy Weight Tensor Module

Provides functionality for managing strategy weights across different market phases.
"""

import logging
import logging


import logging
import logging


import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

# Check for mathematical infrastructure availability
try:
    from core.math.mathematical_framework_integrator import MathConfigManager, MathResultCache, MathOrchestrator
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
class Config:
    """Configuration data class."""

    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    decay_factor: float = 0.95
    learning_rate: float = 0.01


@dataclass
class Result:
    """Result data class."""

    success: bool = False
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class MultiPhaseStrategyWeightTensor:
    """
    MultiPhaseStrategyWeightTensor Implementation
    Manages strategy weights across different market phases.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MultiPhaseStrategyWeightTensor with configuration."""
        self.config = config or self._default_config()
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
        
        # Performance tracking
        self.metrics: Dict[str, Any] = {
            'total_updates': 0,
            'phase_transitions': 0,
            'last_update_time': time.time(),
            'active_phase': None
        }

        # Initialize math infrastructure if available
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_config = MathConfigManager()
            self.math_cache = MathResultCache()
            self.math_orchestrator = MathOrchestrator()

        self._initialize_system()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'enabled': True,
            'timeout': 30.0,
            'retries': 3,
            'debug': False,
            'log_level': 'INFO',
            'decay_factor': 0.95,
            'learning_rate': 0.01,
        }

    def _initialize_system(self) -> None:
        """Initialize the system."""
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            self.initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False

    def activate(self) -> bool:
        """Activate the system."""
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
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info(f"✅ {self.__class__.__name__} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
            'metrics': self.metrics,
            'current_phase': self.current_phase.value if self.current_phase else None,
        }

    def initialize_strategies(self, strategy_ids: List[str]) -> None:
        """Initialize the weight tensor with strategy IDs."""
        self.strategy_ids = strategy_ids
        self.num_strategies = len(strategy_ids)
        
        # Initialize weight tensor with equal weights
        self.weight_tensor = np.ones((self.num_strategies, self.num_phases)) / self.num_strategies
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
        self.weight_tensor[:, phase_idx] *= self.config['decay_factor']

        # Update weights based on performance feedback
        for strategy_id, metrics in performance_feedback.items():
            if strategy_id in self.strategy_ids:
                strategy_idx = self.strategy_ids.index(strategy_id)
                pnl = metrics.get('pnl', 0.0)
                
                # Simple weight update based on PnL
                weight_update = pnl * self.config['learning_rate']
                self.weight_tensor[strategy_idx, phase_idx] += weight_update

        # Renormalize weights
        self._normalize_weights()

    def process_strategy_data(self, data: Union[List, Tuple, np.ndarray]) -> float:
        """Process strategy data."""
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise ValueError("Data must be array-like")
        
        data_array = np.array(data)
        # Default mathematical operation
        return np.mean(data_array)


# Factory function
def create_multi_phase_strategy_weight_tensor(config: Optional[Dict[str, Any]] = None) -> MultiPhaseStrategyWeightTensor:
    """Create a multi-phase strategy weight tensor instance."""
    return MultiPhaseStrategyWeightTensor(config)
