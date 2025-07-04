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

import numpy as np

# CUDA Helper Integration
try:
    from ..utils.cuda_helper import (
        xp, USING_CUDA, safe_cuda_operation, safe_matrix_multiply,
        safe_tensor_contraction, safe_fft, safe_convolution,
        safe_eigenvalue_decomposition, safe_matrix_inverse, safe_svd,
        get_cuda_status, report_cuda_status
    )
    CUDA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("⚡ CUDA acceleration enabled in Strategy Bit Mapper")
except ImportError:
    # Fallback to CPU-only mode
    xp = np
    USING_CUDA = False
    CUDA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("🔄 CUDA not available - using CPU-only mode in Strategy Bit Mapper")

# Dual State Router Integration
try:
    from ..system.dual_state_router import (
        get_dual_state_router, route_task, StrategyTier, ComputeMode
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
    temporal_algebra
)
from core.fractal_core import fractal_quantize_vector
from core.matrix_mapper import load_matrix_from_file, EnhancedMatrixMapper
from core.schwafit_core import SchwafitCore
from core.strategy_loader import load_strategy
from core.unified_math_system import generate_unified_hash

# Optional: Import dashboard event emitter if available
try:
    from core.visual_execution_node import emit_dashboard_event
except ImportError:
    def emit_dashboard_event(event: str, data: Any) -> None:
        """Emit dashboard event (no-op fallback)."""
        pass  # No-op fallback

# Optional: Import profit tick logger if available
try:
    from core.visual_execution_node import log_profit_tick
except ImportError:
    def log_profit_tick(data: Any) -> None:
        """Log profit tick (no-op fallback)."""
        pass  # No-op fallback

logger = logging.getLogger(__name__)


class ExpansionMode:
    """Expansion modes for strategy bit mapping."""

    FLIP = "flip"
    MIRROR = "mirror"
    RANDOM = "random"
    FERRIS_WHEEL = "ferris_wheel"
    TENSOR_WEIGHTED = "tensor_weighted"  # New tensor-weighted expansion


class StrategyBitMapper:
    """
    StrategyBitMapper: Handles bitwise strategy expansion, hash-to-matrix matching,
    fallback logic, and dashboard/profit logger integration for real-time trading.

    Enhanced with:
    - Live handler feed routing
    - Tensor weight rebalancing
    - API data integration
    - Real-time strategy adaptation
    - Schwafit-driven fit/decision logic
    - CUDA-accelerated mathematical operations
    """

    def __init__(self, matrix_dir, dashboard_hook: Optional[Callable] = None, weather_api_key: Optional[str] = None):
        """Initialize StrategyBitMapper."""
        self.matrix_dir = matrix_dir
        self.dashboard_hook = dashboard_hook or emit_dashboard_event
        self.expansion_history = []
        self.metrics = {
            "total_expansions": 0,
            "successful_mappings": 0,
            "failed_mappings": 0,
            "last_expansion_time": None
        }

        # Mathematical subsystems
        self.tensor_algebra = AdvancedTensorAlgebra()
        self.temporal_algebra = temporal_algebra
        self.information_geometry = information_geometry
        self.spectral_analysis = spectral_analysis

        # Initialize dual state router for profit-tiered orchestration
        if DUAL_STATE_AVAILABLE:
            self.dual_state_router = get_dual_state_router()
            logger.info("🔄 Dual State Router initialized for profit-tiered orchestration")
        else:
            self.dual_state_router = None
            logger.info("⚠️ Using direct operations (no dual state router)")

        # Live handler feed integration
        self.live_handlers: Dict[str, Any] = {}
        self.handler_weights: Dict[str, float] = {}
        self.api_data_cache: Dict[str, Any] = {}

        # Tensor weight rebalancing with CUDA acceleration
        self.tensor_weights = safe_cuda_operation(
            lambda: xp.ones(64),
            lambda: np.ones(64)
        )  # 64-bit strategy space
        self.weight_update_rate = 0.01
        self.rebalancing_threshold = 0.1

        # Schwafit and Matrix Mapper integration
        self.matrix_mapper = EnhancedMatrixMapper(matrix_dir, weather_api_key)
        self.schwafit = SchwafitCore(window=64)

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length with CUDA acceleration."""
        norm = safe_cuda_operation(
            lambda: xp.linalg.norm(v),
            lambda: np.linalg.norm(v)
        )
        return safe_cuda_operation(
            lambda: v / norm if norm != 0 else v,
            lambda: v / norm if norm != 0 else v
        )

    def compute_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors with CUDA acceleration."""
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return float(safe_cuda_operation(
            lambda: xp.dot(a, b),
            lambda: np.dot(a, b)
        ))

    def expand_strategy_bits(
        self, strategy_id: int, target_bits: int = 8, 
        mode: str = ExpansionMode.RANDOM
    ) -> int:
        """Expand strategy bits using various methods."""
        if mode == ExpansionMode.FLIP:
            return strategy_id ^ ((1 << target_bits) - 1)
        elif mode == ExpansionMode.MIRROR:
            binary = format(strategy_id, f'0{target_bits}b')
            mirrored = binary[::-1]
            return int(mirrored, 2)
        elif mode == ExpansionMode.RANDOM:
            random.seed(strategy_id)
            return random.randint(0, (1 << target_bits) - 1)
        elif mode == ExpansionMode.FERRIS_WHEEL:
            # Tune to hour cycle drift (e.g., 24-hour cycle)
            now = datetime.utcnow()
            hour_angle = (now.hour + now.minute / 60.0) * (2 * np.pi / 24)
            drift = int((np.sin(hour_angle) + 1) * ((1 << (target_bits - 1)) - 1))
            return (strategy_id + drift) % (1 << target_bits)
        elif mode == ExpansionMode.TENSOR_WEIGHTED:
            # Use tensor weights for expansion
            return self._tensor_weighted_expansion(strategy_id, target_bits)
        else:
            return strategy_id

    def _tensor_weighted_expansion(self, strategy_id: int, target_bits: int) -> int:
        """Expand strategy bits using tensor weights and API data with CUDA acceleration."""
        try:
            # Use dual state router if available for profit-tiered orchestration
            if self.dual_state_router is not None:
                task_data = {
                    'strategy_id': strategy_id,
                    'target_bits': target_bits,
                    'tensor_weights': self.tensor_weights.tolist(),
                    'operation': 'tensor_weighted_expansion'
                }
                
                result = self.dual_state_router.route(
                    task_id="tensor_weighted_expansion",
                    data=task_data
                )
                
                if result.get('success', False) and 'expanded_id' in result:
                    return result['expanded_id']
                else:
                    # Fallback to direct computation
                    logger.debug("Dual state router returned no result, using direct computation")
            
            # Direct computation (fallback or when dual state router not available)
            # Convert strategy_id to binary vector
            strategy_vector = np.array(
                [int(b) for b in format(strategy_id, f'0{target_bits}b')]
            )

            # Apply tensor weights with CUDA acceleration
            weighted_vector = safe_cuda_operation(
                lambda: strategy_vector * self.tensor_weights[:target_bits],
                lambda: strategy_vector * self.tensor_weights[:target_bits]
            )

            # Apply temporal alignment
            temporal_factor = self.temporal_algebra.ferris_wheel_alignment()
            weighted_vector = safe_cuda_operation(
                lambda: weighted_vector * temporal_factor,
                lambda: weighted_vector * temporal_factor
            )

            # Apply information geometry transformation
            if len(weighted_vector) >= 2:
                # Create Fisher information metric for strategy space
                strategy_data = np.vstack([strategy_vector, weighted_vector])
                fisher_metric = self.information_geometry.fisher_information_metric(
                    strategy_data, "normal"
                )
                # Apply metric transformation with CUDA acceleration
                weighted_vector = safe_matrix_multiply(fisher_metric, weighted_vector)

            # Quantize back to integer
            expanded_id = int(safe_cuda_operation(
                lambda: xp.sum(weighted_vector * (2 ** xp.arange(target_bits))),
                lambda: np.sum(weighted_vector * (2 ** np.arange(target_bits)))
            ))
            return expanded_id % (1 << target_bits)

        except Exception as e:
            logger.error(f"Tensor-weighted expansion failed: {e}")
            return strategy_id

    def match_hash_to_matrix(self, input_hash_vec: np.ndarray, location: Any = None, threshold: float = 0.8):
        """Match hash vector to enhanced matrix using Schwafit and MatrixMapper."""
        # Use Schwafit-driven matrix matching
        result = self.matrix_mapper.match_hash_to_enhanced_matrix(input_hash_vec, location, threshold)
        if result:
            matrix_name, entry, score, schwafit_info = result
            return matrix_name, entry, score, schwafit_info
        return None, None, -1, None

    def select_strategy(self, hash_vec: np.ndarray, asset_hint: Optional[str] = None, location: Any = None):
        """Select strategy based on hash vector, Schwafit fit, and asset hint."""
        matrix_name, entry, score, schwafit_info = self.match_hash_to_matrix(hash_vec, location)
        if matrix_name and entry and schwafit_info:
            quantized = fractal_quantize_vector(entry.hash_vector)
            unified_hash = generate_unified_hash(quantized)
            strategy = load_strategy(asset_hint or unified_hash)

            # Route to live handler feed, include Schwafit info
            self._route_to_live_handler(unified_hash, strategy, score, schwafit_info)

            self.dashboard_hook("strategy_match", {
                "matrix_name": matrix_name,
                "similarity": score,
                "unified_hash": unified_hash,
                "schwafit": schwafit_info
            })
            return {
                "strategy": strategy,
                "matrix_name": matrix_name,
                "similarity": score,
                "unified_hash": unified_hash,
                "schwafit": schwafit_info
            }
        # Fallback: random or default strategy
        fallback_asset = asset_hint or random.choice(
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        )
        strategy = load_strategy(fallback_asset)
        self.dashboard_hook("strategy_fallback", {
            "asset": fallback_asset,
            "reason": "No matrix match",
            "similarity": None
        })
        return {
            "strategy": strategy,
            "matrix_name": None,
            "similarity": None,
            "unified_hash": None,
            "schwafit": None
        }

    def _route_to_live_handler(self, unified_hash: str, strategy: Callable, similarity: float, schwafit_info: Optional[Dict[str, Any]] = None):
        """Route strategy to live handler feed for real-time execution, with Schwafit info."""
        try:
            # Update handler weights based on similarity
            if unified_hash in self.handler_weights:
                self.handler_weights[unified_hash] = (
                    self.handler_weights[unified_hash] * 0.9 + similarity * 0.1
                )
            else:
                self.handler_weights[unified_hash] = similarity

            # Store in live handlers
            self.live_handlers[unified_hash] = {
                "strategy": strategy,
                "similarity": similarity,
                "timestamp": datetime.utcnow(),
                "weight": self.handler_weights[unified_hash],
                "schwafit": schwafit_info
            }

            logger.info(
                f"Routed strategy {unified_hash} to live handler with similarity {similarity:.3f} and Schwafit info: {schwafit_info}"
            )

        except Exception as e:
            logger.error(f"Failed to route to live handler: {e}")

    def update_tensor_weights_from_api_data(self, api_data: Dict[str, Any]):
        """Update tensor weights based on API data for adaptive strategy selection."""
        try:
            # Extract market volatility from API data
            volatility = self._calculate_market_volatility(api_data)

            # Update weights based on volatility
            volatility_factor = np.clip(volatility, 0.1, 2.0)

            # Apply adaptive weight update
            for i in range(len(self.tensor_weights)):
                # Use spectral analysis to determine weight adjustment
                if "price_history" in api_data:
                    price_history = np.array(api_data["price_history"])
                    if len(price_history) > 10:
                        frequencies, power_spectrum = self.spectral_analysis.fourier_spectrum(
                            price_history
                        )
                        # Use high-frequency components for weight adjustment
                        high_freq_power = np.sum(power_spectrum[frequencies > 0.1])
                        spectral_factor = np.log(1.0 + high_freq_power)
                    else:
                        spectral_factor = 1.0
                else:
                    spectral_factor = 1.0

                # Update weight with learning rate
                weight_adjustment = (
                    self.weight_update_rate *
                    volatility_factor *
                    spectral_factor *
                    (1.0 - self.tensor_weights[i])
                )
                self.tensor_weights[i] = np.clip(
                    self.tensor_weights[i] + weight_adjustment, 0.1, 2.0
                )

            # Check if rebalancing is needed
            weight_variance = np.var(self.tensor_weights)
            if weight_variance > self.rebalancing_threshold:
                self._perform_tensor_rebalancing()

            logger.debug(f"Updated tensor weights, volatility: {volatility:.3f}")

        except Exception as e:
            logger.error(f"Failed to update tensor weights from API data: {e}")

    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility from market data."""
        try:
            if "price_history" in market_data:
                prices = np.array(market_data["price_history"])
                if len(prices) > 1:
                    returns = np.diff(np.log(prices))
                    volatility = np.std(returns)
                    return float(volatility)

            # Fallback to default volatility
            return 0.02  # 2% default volatility

        except Exception as e:
            logger.error(f"Failed to calculate market volatility: {e}")
            return 0.02

    def _perform_tensor_rebalancing(self):
        """Perform tensor weight rebalancing to maintain stability."""
        try:
            # Normalize weights to prevent drift
            weight_sum = np.sum(self.tensor_weights)
            if weight_sum > 0:
                self.tensor_weights = self.tensor_weights / weight_sum * len(self.tensor_weights)

            # Apply entropy-based regularization
            entropy = -np.sum(self.tensor_weights * np.log(self.tensor_weights + 1e-10))
            max_entropy = np.log(len(self.tensor_weights))
            entropy_factor = entropy / max_entropy

            # Adjust weights based on entropy
            if entropy_factor < 0.5:  # Low entropy, increase diversity
                self.tensor_weights = self.tensor_weights * 1.1
            elif entropy_factor > 0.9:  # High entropy, increase focus
                self.tensor_weights = self.tensor_weights * 0.9

            logger.info(f"Performed tensor rebalancing, entropy factor: {entropy_factor:.3f}")

        except Exception as e:
            logger.error(f"Failed to perform tensor rebalancing: {e}")

    def get_live_handler_status(self) -> Dict[str, Any]:
        """Get status of live handlers and tensor weights."""
        return {
            "active_handlers": len(self.live_handlers),
            "handler_weights": self.handler_weights.copy(),
            "tensor_weights_mean": float(np.mean(self.tensor_weights)),
            "tensor_weights_std": float(np.std(self.tensor_weights)),
            "rebalancing_threshold": self.rebalancing_threshold,
            "last_update": datetime.utcnow().isoformat()
        }

    def trigger_entry_exit(
        self, 
        signal_packet: Dict[str, Any], 
        market_state: Dict[str, Any], 
        ccxt_executor: Any
    ) -> bool:
        """Trigger entry/exit based on strategy signals."""
        try:
            # Extract strategy information
            strategy = signal_packet.get("strategy")
            if not strategy:
                logger.warning("No strategy found in signal packet")
                return False

            # Execute strategy with market state
            result = strategy(market_state)

            # Log profit tick if available
            if result and "profit" in result:
                log_profit_tick({
                    "timestamp": datetime.utcnow().isoformat(),
                    "profit": result["profit"],
                    "strategy": signal_packet.get("unified_hash", "unknown")
                })

            return result is not None

        except Exception as e:
            logger.error(f"Failed to trigger entry/exit: {e}")
            return False