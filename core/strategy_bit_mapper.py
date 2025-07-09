import logging
import random
from datetime import datetime
from typing import Any, Callable, Dict, Optional
import time
import logging
import os
import numpy as np

from core.backend_math import get_backend, is_gpu
from utils.cuda_helper import safe_cuda_operation

xp = get_backend()

from core.unified_math_system import generate_unified_hash
from .orbital_shell_brain_system import OrbitalBRAINSystem, ShellConsensus, AltitudeVector
from .qutrit_signal_matrix import QutritSignalMatrix, QutritState
from core.gpu_handlers import get_gpu_memory, select_best_gpu
from core.fractal_core import fractal_quantize_vector
from core.matrix_mapper import EnhancedMatrixMapper, load_matrix_from_file
from core.schwafit_core import SchwafitCore
from core.strategy_loader import load_strategy
from core.visual_execution_node import emit_dashboard_event
from core.visual_execution_node import log_profit_tick

# Entropy Signal Integration
try:
    from core.entropy_signal_integration import EntropySignalIntegration
    ENTROPY_AVAILABLE = True
    logger.info("🔄 Entropy Signal Integration enabled in Strategy Bit Mapper")
except ImportError:
    ENTROPY_AVAILABLE = False
    logger.warning("⚠️ Entropy Signal Integration not available in Strategy Bit Mapper")

# Log backend status
logger = logging.getLogger(__name__)
if is_gpu():
    logger.info("⚡ Strategy Bit Mapper using GPU acceleration: CuPy (GPU)")
else:
    logger.info("🔄 Strategy Bit Mapper using CPU fallback: NumPy (CPU)")

"""Strategy Bit Mapper - Handles bitwise strategy expansion and hash-to-matrix matching."

CUDA Integration:
- GPU-accelerated strategy operations with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)
"""

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

try:
    from core.advanced_tensor_algebra import (
        AdvancedTensorAlgebra,
        information_geometry,
        spectral_analysis,
        temporal_algebra,
    )
except ImportError:
    logger.warning("⚠️ Advanced Tensor Algebra not available in Strategy Bit Mapper")

# Optional: Import dashboard event emitter if available
try:
    pass
except ImportError:

    def emit_dashboard_event(event: str, data: Any) -> None:
        """Emit dashboard event (no-op, fallback)."""


# Optional: Import profit tick logger if available
try:
    pass
except ImportError:

    def log_profit_tick(data: Any) -> None:
        """Log profit tick (no-op, fallback)."""


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
        os.makedirs(self.matrix_dir, exist_ok=True)
        self.dashboard_hook = dashboard_hook or emit_dashboard_event
        self.expansion_history = []
        self.metrics = {
            "total_expansions": 0,
            "successful_mappings": 0,
            "failed_mappings": 0,
            "last_expansion_time": None,
        }

        # Initialize tensor algebra components if available
        try:
            self.tensor_algebra = AdvancedTensorAlgebra()
            self.temporal_algebra = temporal_algebra
            self.information_geometry = information_geometry
            self.spectral_analysis = spectral_analysis
        except NameError:
            logger.warning("Tensor algebra components not available")
            self.tensor_algebra = None
            self.temporal_algebra = None
            self.information_geometry = None
            self.spectral_analysis = None

        if DUAL_STATE_AVAILABLE:
            self.dual_state_router = get_dual_state_router()
        else:
            self.dual_state_router = None

        self.live_handlers: Dict[str, Any] = {}
        self.handler_weights: Dict[str, float] = {}
        self.api_data_cache: Dict[str, Any] = {}

        self.tensor_weights = safe_cuda_operation(lambda: xp.ones(64), lambda: xp.ones(64))
        self.weight_update_rate = 0.1
        self.rebalancing_threshold = 0.1

        self.matrix_mapper = EnhancedMatrixMapper(matrix_dir, weather_api_key)
        self.schwafit = SchwafitCore(window=64)
        self.orbital_brain = OrbitalBRAINSystem()

        # Initialize entropy signal integration if available
        if ENTROPY_AVAILABLE:
            self.entropy_integration = EntropySignalIntegration()
            logger.info("🔄 Entropy signal integration initialized in Strategy Bit Mapper")
        else:
            self.entropy_integration = None
            logger.warning("⚠️ Entropy signal integration not available in Strategy Bit Mapper")

    def apply_qutrit_gate(
        self, strategy_id: str, seed: str, market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply qutrit gate to strategy decision with entropy signal integration

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

            # Process entropy signals if available
            entropy_adjustment = 1.0
            entropy_timing = None
            if self.entropy_integration and market_data:
                try:
                    # Extract order book data for entropy processing
                    order_book_data = self._extract_order_book_data(market_data)
                    
                    # Process entropy signals
                    entropy_result = self.entropy_integration.process_entropy_signals(
                        order_book_data=order_book_data,
                        market_context=market_data
                    )
                    
                    # Apply entropy adjustments
                    entropy_adjustment = entropy_result.get('confidence_adjustment', 1.0)
                    entropy_timing = entropy_result.get('timing_cycle', None)
                    
                    logger.info(f"🔄 Entropy adjustment applied: {entropy_adjustment:.3f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Entropy signal processing failed: {e}")
                    entropy_adjustment = 1.0

            # Apply state-based logic with entropy adjustment
            adjusted_confidence = qutrit_result.confidence * entropy_adjustment
            
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
                "confidence": adjusted_confidence,
                "original_confidence": qutrit_result.confidence,
                "entropy_adjustment": entropy_adjustment,
                "entropy_timing": entropy_timing,
                "hash_segment": qutrit_result.hash_segment,
                "matrix": qutrit_result.matrix.tolist(),
            }

        except Exception as e:
            logger.error("Error applying qutrit gate: {0}".format(e))
            return {
                "strategy_id": strategy_id,
                "action": "error",
                "reason": str(e),
                "qutrit_state": "error",
                "confidence": 0.0,
                "original_confidence": 0.0,
                "entropy_adjustment": 1.0,
                "entropy_timing": None,
                "hash_segment": "",
                "matrix": [],
            }

    def defer(self, strategy_id: str) -> Dict[str, Any]:
        """Defer strategy execution."""
        return {"action": "defer", "strategy_id": strategy_id, "reason": "Strategy deferred"}

    def execute_trade(self, strategy_id: str) -> Dict[str, Any]:
        """Execute trade for strategy."""
        return {"action": "execute", "strategy_id": strategy_id, "reason": "Trade executed"}

    def recheck_later(self, strategy_id: str) -> Dict[str, Any]:
        """Recheck strategy later."""
        return {"action": "recheck", "strategy_id": strategy_id, "reason": "Recheck later"}

    def normalize_vector(self, v: xp.ndarray) -> xp.ndarray:
        """Normalize vector using xp backend."""
        norm = xp.linalg.norm(v)
        return v / norm if norm != 0 else v

    def compute_cosine_similarity(self, a: xp.ndarray, b: xp.ndarray) -> float:
        """Compute cosine similarity using xp backend."""
        try:
            a_norm = self.normalize_vector(a)
            b_norm = self.normalize_vector(b)
            return float(xp.dot(a_norm, b_norm))
        except Exception as e:
            logger.error("Error computing cosine similarity: {0}".format(e))
            return 0.0

    def expand_strategy_bits(
        self,
        strategy_id: int,
        target_bits: int = 8,
        mode: str = ExpansionMode.RANDOM,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Expand strategy bits with entropy signal integration for enhanced decision making.
        
        Args:
            strategy_id: Original strategy ID
            target_bits: Target number of bits
            mode: Expansion mode
            market_data: Market data for entropy processing
            
        Returns:
            Expanded strategy ID
        """
        # Process entropy signals if available
        entropy_factor = 1.0
        if self.entropy_integration and market_data:
            try:
                order_book_data = self._extract_order_book_data(market_data)
                entropy_result = self.entropy_integration.process_entropy_signals(
                    order_book_data=order_book_data,
                    market_context=market_data
                )
                
                # Use entropy timing to adjust expansion
                entropy_factor = entropy_result.get('expansion_factor', 1.0)
                logger.info(f"🔄 Entropy expansion factor: {entropy_factor:.3f}")
                
            except Exception as e:
                logger.warning(f"⚠️ Entropy expansion processing failed: {e}")
                entropy_factor = 1.0

        # Apply entropy factor to strategy ID
        adjusted_strategy_id = int(strategy_id * entropy_factor) % (2**32)
        
        if mode == ExpansionMode.FLIP:
            return adjusted_strategy_id ^ ((1 << target_bits) - 1)
        elif mode == ExpansionMode.MIRROR:
            binary = format(adjusted_strategy_id, "0{0}b".format(target_bits))
            return int(binary[::-1], 2)
        elif mode == ExpansionMode.RANDOM:
            random.seed(adjusted_strategy_id)
            return random.randint(0, (1 << target_bits) - 1)
        elif mode == ExpansionMode.FERRIS_WHEEL:
            now = datetime.utcnow()
            hour_angle = (now.hour + now.minute / 60.0) * (2 * np.pi / 24)
            drift = int((np.sin(hour_angle) + 1) * ((1 << (target_bits - 1)) - 1))
            return (adjusted_strategy_id + drift) % (1 << target_bits)
        elif mode == ExpansionMode.TENSOR_WEIGHTED:
            return self._tensor_weighted_expansion(adjusted_strategy_id, target_bits)
        elif mode == ExpansionMode.ORBITAL_ADAPTIVE:
            market_data = market_data or self._get_simulated_market_data()
            return self._orbital_adaptive_expansion(adjusted_strategy_id, target_bits, market_data)
        else:
            raise ValueError("Invalid expansion mode: {0}".format(mode))

    def _tensor_weighted_expansion(self, strategy_id: int, target_bits: int) -> int:
        """Expand strategy using tensor-weighted approach with xp backend."""
        try:
            # Use xp for tensor operations
            weights = self.tensor_weights
            expansion_factor = float(xp.sum(weights[:target_bits]))
            return int(strategy_id * expansion_factor) % (2**target_bits)
        except Exception as e:
            logger.error("Tensor weighted expansion failed: {0}".format(e))
            return strategy_id % (2**target_bits)

    def _orbital_adaptive_expansion(
        self, strategy_id: int, target_bits: int, market_data: Dict[str, Any]
    ) -> int:
        """Expand strategy using orbital adaptive approach with xp backend."""
        try:
            # Use orbital brain for adaptive expansion
            orbital_result = self.orbital_brain.compute_orbital_expansion(strategy_id, market_data)

            # Convert to xp array for processing
            orbital_vector = xp.array(orbital_result.get("expansion_vector", [1.0]))

            # Apply orbital scaling
            expansion_factor = float(xp.mean(orbital_vector))
            return int(strategy_id * expansion_factor) % (2**target_bits)
        except Exception as e:
            logger.error("Orbital adaptive expansion failed: {0}".format(e))
            return strategy_id % (2**target_bits)

    def match_hash_to_matrix(
        self, input_hash_vec: xp.ndarray, location: Any = None, threshold: float = 0.8
    ):
        """Match hash vector to matrix using xp backend."""
        return self.matrix_mapper.match_hash_to_matrix(input_hash_vec, location, threshold)

    def select_strategy(
        self, hash_vec: xp.ndarray, asset_hint: Optional[str] = None, location: Any = None
    ):
        """
        Select strategy based on hash vector with entropy signal integration.
        
        Args:
            hash_vec: Hash vector for strategy selection
            asset_hint: Optional asset hint
            location: Optional location context
            
        Returns:
            Selected strategy information
        """
        try:
            # Get base strategy selection
            base_strategy = self.matrix_mapper.select_strategy(hash_vec, asset_hint, location)
            
            # Apply entropy signal processing if available
            if self.entropy_integration:
                try:
                    # Create market context from available data
                    market_context = {
                        'asset': asset_hint,
                        'timestamp': time.time(),
                        'hash_vector': hash_vec.tolist() if hasattr(hash_vec, 'tolist') else hash_vec
                    }
                    
                    # Process entropy signals
                    entropy_result = self.entropy_integration.process_entropy_signals(
                        order_book_data=self._get_simulated_market_data(),
                        market_context=market_context
                    )
                    
                    # Adjust strategy selection based on entropy
                    entropy_score = entropy_result.get('strategy_score', 1.0)
                    entropy_timing = entropy_result.get('timing_cycle', None)
                    
                    # Enhance base strategy with entropy information
                    if isinstance(base_strategy, dict):
                        base_strategy['entropy_score'] = entropy_score
                        base_strategy['entropy_timing'] = entropy_timing
                        base_strategy['entropy_adjusted'] = True
                        
                        logger.info(f"🔄 Strategy selection enhanced with entropy score: {entropy_score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Entropy strategy selection failed: {e}")
                    if isinstance(base_strategy, dict):
                        base_strategy['entropy_adjusted'] = False
            
            return base_strategy
            
        except Exception as e:
            logger.error(f"Error in strategy selection: {e}")
            return None

    def _get_simulated_market_data(self) -> Dict[str, Any]:
        """Get simulated market data for testing."""
        return {
            "price": 50000.0,
            "volume": 1000.0,
            "timestamp": time.time(),
            "volatility": 0.02,
        }

    def _extract_order_book_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract order book data from market data for entropy processing.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Order book data dictionary
        """
        try:
            # Extract order book data if available
            order_book = market_data.get('order_book', {})
            
            # If no order book data, create simulated data
            if not order_book:
                order_book = {
                    'bids': [[market_data.get('price', 50000) * 0.999, 100]],
                    'asks': [[market_data.get('price', 50000) * 1.001, 100]],
                    'timestamp': market_data.get('timestamp', time.time())
                }
            
            return {
                'bids': order_book.get('bids', []),
                'asks': order_book.get('asks', []),
                'timestamp': order_book.get('timestamp', time.time()),
                'spread': market_data.get('spread', 0.001),
                'depth': market_data.get('depth', 10)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract order book data: {e}")
            # Return minimal order book data
            return {
                'bids': [[50000, 100]],
                'asks': [[50001, 100]],
                'timestamp': time.time(),
                'spread': 0.001,
                'depth': 10
            }
