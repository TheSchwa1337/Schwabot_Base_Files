"""Strategy Bit Mapper - Handles bitwise strategy expansion and hash-to-matrix matching.

CUDA Integration:
- GPU-accelerated strategy operations with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)
"""

import logging
import random
from datetime import datetime
from typing import Any, Callable, Dict, Optional
import time
import numpy as np
try:
    import cupy as cp
    USING_CUDA = True
    _backend = 'cupy (GPU)'
except ImportError:
    cp = np
    USING_CUDA = False
    _backend = 'numpy (CPU)'
import logging
logger = logging.getLogger(__name__)
logger.info(f"StrategyBitMapper using backend: {_backend}")

# CUDA Helper Integration
try:
    from core.gpu_handlers import get_gpu_memory, select_best_gpu
    from utils.cuda_helper import safe_cuda_operation, xp, USING_CUDA
    CUDA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("⚡ CUDA acceleration enabled in Strategy Bit Mapper")
except ImportError:
    xp = np
    USING_CUDA = False
    CUDA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("🔄 CUDA not available - using CPU-only mode in Strategy Bit Mapper")
    def safe_cuda_operation(cuda_fn, cpu_fn, **kwargs):
        """CPU-only fallback for safe_cuda_operation."""
        return cpu_fn(**kwargs)

# Dual State Router Integration
try:
    from ..system.dual_state_router import (
        ComputeMode,
        StrategyTier,
        get_dual_state_router,
        route_task,
    )
    DUAL_STATE_AVAILABLE = True
    logger.info("🔄 Dual State Router integration enabled in Strategy Bit Mapper")
except ImportError:
    DUAL_STATE_AVAILABLE = False
    logger.warning("⚠️ Dual State Router not available in Strategy Bit Mapper")

from core.advanced_tensor_algebra import (
    AdvancedTensorAlgebra,
    information_geometry,
    spectral_analysis,
    temporal_algebra,
)
from core.fractal_core import fractal_quantize_vector
from core.matrix_mapper import EnhancedMatrixMapper, load_matrix_from_file
from core.schwafit_core import SchwafitCore
from core.strategy_loader import load_strategy
from core.unified_math_system import generate_unified_hash
from .orbital_shell_brain_system import OrbitalBRAINSystem, ShellConsensus, AltitudeVector
from .qutrit_signal_matrix import QutritSignalMatrix, QutritState

# Optional: Import dashboard event emitter if available
try:
    from core.visual_execution_node import emit_dashboard_event
except ImportError:
    def emit_dashboard_event(event: str, data: Any) -> None:
        """Emit dashboard event (no-op fallback)."""
        pass

# Optional: Import profit tick logger if available
try:
    from core.visual_execution_node import log_profit_tick
except ImportError:
    def log_profit_tick(data: Any) -> None:
        """Log profit tick (no-op fallback)."""
        pass

logger = logging.getLogger(__name__)

class ExpansionMode:
    """Expansion modes for strategy bit mapping."""
    FLIP = "flip"
    MIRROR = "mirror"
    RANDOM = "random"
    FERRIS_WHEEL = "ferris_wheel"
    TENSOR_WEIGHTED = "tensor_weighted"
    ORBITAL_ADAPTIVE = "orbital_adaptive"

class StrategyBitMapper:
    """
    Handles bitwise strategy expansion, hash-to-matrix matching, and integration
    for real-time, adaptive trading.
    """
    def __init__(
        self,
        matrix_dir,
        dashboard_hook: Optional[Callable] = None,
        weather_api_key: Optional[str] = None,
    ):
        self.matrix_dir = matrix_dir
        import os
        os.makedirs(self.matrix_dir, exist_ok=True)
        self.dashboard_hook = dashboard_hook or emit_dashboard_event
        self.expansion_history = []
        self.metrics = {
            "total_expansions": 0,
            "successful_mappings": 0,
            "failed_mappings": 0,
            "last_expansion_time": None,
        }
        self.tensor_algebra = AdvancedTensorAlgebra()
        self.temporal_algebra = temporal_algebra
        self.information_geometry = information_geometry
        self.spectral_analysis = spectral_analysis

        if DUAL_STATE_AVAILABLE:
            self.dual_state_router = get_dual_state_router()
        else:
            self.dual_state_router = None

        self.live_handlers: Dict[str, Any] = {}
        self.handler_weights: Dict[str, float] = {}
        self.api_data_cache: Dict[str, Any] = {}

        self.tensor_weights = safe_cuda_operation(
            lambda: xp.ones(64), lambda: np.ones(64)
        )
        self.weight_update_rate = 0.01
        self.rebalancing_threshold = 0.1

        self.matrix_mapper = EnhancedMatrixMapper(matrix_dir, weather_api_key)
        self.schwafit = SchwafitCore(window=64)
        self.orbital_brain = OrbitalBRAINSystem()

    def apply_qutrit_gate(self, strategy_id: str, seed: str, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply qutrit gate to strategy decision
        
        Args:
            strategy_id: Strategy identifier
            seed: Seed for qutrit matrix generation
            market_data: Optional market context
            
        Returns:
            Dictionary with action and metadata
        """
        try:
            # Create qutrit matrix
            qutrit_matrix = QutritSignalMatrix(seed, market_data)
            qutrit_result = qutrit_matrix.get_matrix_result()
            
            # Apply state-based logic
            if qutrit_result.state == QutritState.DEFER:
                action = "defer"
                reason = "Qutrit state indicates hold position"
            elif qutrit_result.state == QutritState.EXECUTE:
                action = "execute"
                reason = "Qutrit state indicates trade execution"
            else:  # RECHECK
                action = "recheck"
                reason = "Qutrit state indicates re-evaluation needed"
            
            return {
                "strategy_id": strategy_id,
                "action": action,
                "reason": reason,
                "qutrit_state": qutrit_result.state.value,
                "confidence": qutrit_result.confidence,
                "hash_segment": qutrit_result.hash_segment,
                "matrix": qutrit_result.matrix.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error applying qutrit gate: {e}")
            return {
                "strategy_id": strategy_id,
                "action": "defer",
                "reason": f"Qutrit gate error: {str(e)}",
                "qutrit_state": QutritState.DEFER.value,
                "confidence": 0.0,
                "hash_segment": "error",
                "matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            }

    def defer(self, strategy_id: str) -> Dict[str, Any]:
        """Defer strategy execution"""
        return {
            "strategy_id": strategy_id,
            "action": "defer",
            "reason": "Strategy deferred by qutrit gate",
            "timestamp": time.time()
        }

    def execute_trade(self, strategy_id: str) -> Dict[str, Any]:
        """Execute trade strategy"""
        return {
            "strategy_id": strategy_id,
            "action": "execute",
            "reason": "Strategy executed by qutrit gate",
            "timestamp": time.time()
        }

    def recheck_later(self, strategy_id: str) -> Dict[str, Any]:
        """Recheck strategy later"""
        return {
            "strategy_id": strategy_id,
            "action": "recheck",
            "reason": "Strategy marked for re-evaluation",
            "timestamp": time.time()
        }

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        norm = safe_cuda_operation(lambda: xp.linalg.norm(v), lambda: np.linalg.norm(v))
        return safe_cuda_operation(
            lambda: v / norm if norm > 0 else v, lambda: v / norm if norm > 0 else v
        )

    def compute_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return float(safe_cuda_operation(lambda: xp.dot(a, b), lambda: np.dot(a, b)))

    def expand_strategy_bits(
        self,
        strategy_id: int,
        target_bits: int = 8,
        mode: str = ExpansionMode.RANDOM,
        market_data: Optional[Dict[str, Any]] = None
    ) -> int:
        if mode == ExpansionMode.FLIP:
            return strategy_id ^ ((1 << target_bits) - 1)
        elif mode == ExpansionMode.MIRROR:
            binary = format(strategy_id, f"0{target_bits}b")
            return int(binary[::-1], 2)
        elif mode == ExpansionMode.RANDOM:
            random.seed(strategy_id)
            return random.randint(0, (1 << target_bits) - 1)
        elif mode == ExpansionMode.FERRIS_WHEEL:
            now = datetime.utcnow()
            hour_angle = (now.hour + now.minute / 60.0) * (2 * np.pi / 24)
            drift = int((np.sin(hour_angle) + 1) * ((1 << (target_bits - 1)) - 1))
            return (strategy_id + drift) % (1 << target_bits)
        elif mode == ExpansionMode.TENSOR_WEIGHTED:
            return self._tensor_weighted_expansion(strategy_id, target_bits)
        elif mode == ExpansionMode.ORBITAL_ADAPTIVE:
            market_data = market_data or self._get_simulated_market_data()
            return self._orbital_adaptive_expansion(strategy_id, target_bits, market_data)
        else:
            raise ValueError(f"Invalid expansion mode: {mode}")

    def _tensor_weighted_expansion(self, strategy_id: int, target_bits: int) -> int:
        strategy_vector = np.array([int(b) for b in format(strategy_id, f"0{target_bits}b")])
        weighted_vector = safe_cuda_operation(
            lambda: strategy_vector * self.tensor_weights[:target_bits],
            lambda: strategy_vector * self.tensor_weights[:target_bits],
        )
        expansion_value = safe_cuda_operation(
            lambda: xp.sum(weighted_vector), lambda: np.sum(weighted_vector)
        )
        return int(np.round(expansion_value)) % (1 << target_bits)

    def _orbital_adaptive_expansion(self, strategy_id: int, target_bits: int, market_data: Dict[str, Any]) -> int:
        altitude_vector = self.orbital_brain.calculate_altitude_vector(market_data)
        shell_consensus = self.orbital_brain.calculate_shell_consensus(market_data)

        altitude_shift = int(altitude_vector.altitude_value * (1 << (target_bits - 2)))
        consensus_shift = int(shell_consensus.consensus_score * (1 << (target_bits - 2)))
        
        adaptive_shift = altitude_shift + consensus_shift
        expanded_strategy = (strategy_id + adaptive_shift) % (1 << target_bits)

        self.dashboard_hook(
            "orbital_expansion",
            {
                "strategy_id": strategy_id,
                "altitude": altitude_vector.altitude_value,
                "consensus": shell_consensus.consensus_score,
                "shift": adaptive_shift,
                "result": expanded_strategy,
            },
        )
        return expanded_strategy

    def match_hash_to_matrix(self, input_hash_vec: np.ndarray, location: Any = None, threshold: float = 0.8):
        return self.matrix_mapper.match_hash_to_matrix(input_hash_vec, location, threshold)

    def select_strategy(self, hash_vec: np.ndarray, asset_hint: Optional[str] = None, location: Any = None):
        return self.matrix_mapper.select_strategy(hash_vec, asset_hint, location)

    def _get_simulated_market_data(self) -> Dict[str, Any]:
        prices = [50000.0 + np.random.normal(0, 1000) for _ in range(20)]
        volumes = [np.random.exponential(1000) for _ in range(20)]
        return {
            'price_history': prices, 'volume_history': volumes, 'current_price': prices[-1],
            'current_volume': volumes[-1], 'timestamp': time.time()
        }
