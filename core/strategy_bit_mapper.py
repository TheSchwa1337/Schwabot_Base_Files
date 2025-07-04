import logging
import random
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, Dict, Any, List

from core.matrix_mapper import load_matrix_from_file
from core.unified_math_system import generate_unified_hash
from core.strategy_loader import load_strategy
from core.fractal_core import fractal_quantize_vector
from core.advanced_tensor_algebra import (
    AdvancedTensorAlgebra,
    temporal_algebra,
    information_geometry,
    spectral_analysis
)

# Optional: Import dashboard event emitter if available
try:
    from core.visual_execution_node import emit_dashboard_event
except ImportError:
    def emit_dashboard_event(event, data):
        pass  # No-op fallback

# Optional: Import profit tick logger if available
try:
    from core.visual_execution_node import log_profit_tick
except ImportError:
    def log_profit_tick(data):
        pass  # No-op fallback

logger = logging.getLogger(__name__)


class ExpansionMode(Enum):
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
    """
    def __init__(self, matrix_dir, dashboard_hook: Optional[Callable] = None):
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
        
        # Live handler feed integration
        self.live_handlers: Dict[str, Any] = {}
        self.handler_weights: Dict[str, float] = {}
        self.api_data_cache: Dict[str, Any] = {}
        
        # Tensor weight rebalancing
        self.tensor_weights: np.ndarray = np.ones(64)  # 64-bit strategy space
        self.weight_update_rate = 0.01
        self.rebalancing_threshold = 0.1

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    def compute_cosine_similarity(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return float(np.dot(a, b))

    def expand_strategy_bits(
        self, strategy_id: int, target_bits: int = 8, 
        mode: ExpansionMode = ExpansionMode.RANDOM
    ) -> int:
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
        """Expand strategy bits using tensor weights and API data."""
        try:
            # Convert strategy_id to binary vector
            strategy_vector = np.array([int(b) for b in format(strategy_id, f'0{target_bits}b')])
            
            # Apply tensor weights
            weighted_vector = strategy_vector * self.tensor_weights[:target_bits]
            
            # Apply temporal alignment
            temporal_factor = self.temporal_algebra.ferris_wheel_alignment()
            weighted_vector *= temporal_factor
            
            # Apply information geometry transformation
            if len(weighted_vector) >= 2:
                # Create Fisher information metric for strategy space
                strategy_data = np.vstack([strategy_vector, weighted_vector])
                fisher_metric = self.information_geometry.fisher_information_metric(
                    strategy_data, "normal"
                )
                # Apply metric transformation
                weighted_vector = fisher_metric @ weighted_vector
            
            # Quantize back to integer
            expanded_id = int(np.sum(weighted_vector * (2 ** np.arange(target_bits))))
            return expanded_id % (1 << target_bits)
            
        except Exception as e:
            logger.error(f"Tensor-weighted expansion failed: {e}")
            return strategy_id

    def match_hash_to_matrix(self, input_hash_vec, threshold=0.8):
        best_score = -1
        best_file = None
        for matrix_file in self.matrix_dir.glob("*.npy"):
            matrix = load_matrix_from_file(matrix_file)
            score = self.compute_cosine_similarity(input_hash_vec, matrix)
            if score > best_score:
                best_score = score
                best_file = matrix_file
        if best_score > threshold and best_file:
            return best_file, best_score
        return None, best_score

    def select_strategy(self, hash_vec, asset_hint=None):
        matrix_file, similarity = self.match_hash_to_matrix(hash_vec)
        if matrix_file:
            matrix = load_matrix_from_file(matrix_file)
            quantized = fractal_quantize_vector(matrix)
            unified_hash = generate_unified_hash(quantized)
            strategy = load_strategy(asset_hint or unified_hash)
            
            # Route to live handler feed
            self._route_to_live_handler(unified_hash, strategy, similarity)
            
            self.dashboard_hook("strategy_match", {
                "matrix_file": matrix_file.name,
                "similarity": similarity,
                "unified_hash": unified_hash
            })
            return {
                "strategy": strategy,
                "matrix_file": matrix_file.name,
                "similarity": similarity,
                "unified_hash": unified_hash
            }
        # Fallback: random or default strategy
        fallback_asset = asset_hint or random.choice(
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        )
        strategy = load_strategy(fallback_asset)
        self.dashboard_hook("strategy_fallback", {
            "asset": fallback_asset,
            "reason": "No matrix match",
            "similarity": similarity
        })
        return {
            "strategy": strategy,
            "matrix_file": None,
            "similarity": similarity,
            "unified_hash": None
        }

    def _route_to_live_handler(self, unified_hash: str, strategy: Callable, similarity: float):
        """Route strategy to live handler feed for real-time execution."""
        try:
            # Create handler entry
            handler_entry = {
                "hash": unified_hash,
                "strategy": strategy,
                "similarity": similarity,
                "timestamp": datetime.utcnow().timestamp(),
                "status": "active",
                "execution_count": 0,
                "success_rate": 0.0
            }
            
            # Store in live handlers
            self.live_handlers[unified_hash] = handler_entry
            
            # Update handler weights based on similarity
            self.handler_weights[unified_hash] = similarity
            
            # Emit dashboard event
            self.dashboard_hook("handler_routed", {
                "hash": unified_hash,
                "similarity": similarity,
                "total_handlers": len(self.live_handlers)
            })
            
            logger.info(f"Strategy {unified_hash} routed to live handler feed")
            
        except Exception as e:
            logger.error(f"Failed to route to live handler: {e}")

    def update_tensor_weights_from_api_data(self, api_data: Dict[str, Any]):
        """Update tensor weights based on API data for dynamic rebalancing."""
        try:
            # Extract relevant API data
            fear_greed_data = api_data.get("fear_greed", {})
            whale_data = api_data.get("whale_alert", {})
            market_data = api_data.get("coingecko", {})
            
            # Calculate weight adjustments
            weight_adjustments = np.zeros_like(self.tensor_weights)
            
            # Fear/Greed influence
            if fear_greed_data:
                sentiment_score = fear_greed_data.get("sentiment_score", 0.0)
                # Adjust weights based on sentiment (fear = conservative, greed = aggressive)
                weight_adjustments[:16] += sentiment_score * 0.1  # First 16 bits
            
            # Whale activity influence
            if whale_data:
                whale_activity = whale_data.get("summary", {}).get("whale_activity_score", 50.0)
                # Normalize whale activity to [0, 1]
                whale_factor = whale_activity / 100.0
                weight_adjustments[16:32] += whale_factor * 0.1  # Bits 16-31
            
            # Market volatility influence
            if market_data:
                # Extract volatility from market data
                volatility = self._calculate_market_volatility(market_data)
                weight_adjustments[32:48] += volatility * 0.1  # Bits 32-47
            
            # Temporal alignment influence
            temporal_factor = self.temporal_algebra.ferris_wheel_alignment()
            weight_adjustments[48:] += temporal_factor * 0.1  # Bits 48-63
            
            # Apply weight updates
            self.tensor_weights += self.weight_update_rate * weight_adjustments
            
            # Normalize weights to prevent explosion
            self.tensor_weights = np.clip(self.tensor_weights, 0.1, 10.0)
            
            # Check if rebalancing is needed
            weight_variance = np.var(self.tensor_weights)
            if weight_variance > self.rebalancing_threshold:
                self._perform_tensor_rebalancing()
            
            logger.debug(f"Tensor weights updated, variance: {weight_variance:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to update tensor weights from API data: {e}")

    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility from market data."""
        try:
            # Extract price data
            prices = market_data.get("prices", {})
            if not prices:
                return 0.5  # Default volatility
            
            # Calculate price changes
            price_values = []
            for coin_data in prices.values():
                if isinstance(coin_data, dict) and "usd" in coin_data:
                    price_values.append(float(coin_data["usd"]))
            
            if len(price_values) < 2:
                return 0.5
            
            # Calculate volatility as coefficient of variation
            mean_price = np.mean(price_values)
            std_price = np.std(price_values)
            volatility = std_price / mean_price if mean_price > 0 else 0.5
            
            return min(1.0, max(0.0, volatility))
            
        except Exception as e:
            logger.error(f"Failed to calculate market volatility: {e}")
            return 0.5

    def _perform_tensor_rebalancing(self):
        """Perform tensor weight rebalancing to maintain stability."""
        try:
            # Calculate target weights based on current market conditions
            target_weights = np.ones_like(self.tensor_weights)
            
            # Apply spectral analysis for rebalancing
            if len(self.tensor_weights) > 10:
                frequencies, power_spectrum = self.spectral_analysis.fourier_spectrum(
                    self.tensor_weights
                )
                # Use dominant frequency for rebalancing
                dominant_freq_idx = np.argmax(power_spectrum)
                dominant_freq = frequencies[dominant_freq_idx]
                
                # Adjust weights based on dominant frequency
                for i in range(len(target_weights)):
                    target_weights[i] *= (1.0 + 0.1 * np.sin(2 * np.pi * dominant_freq * i))
            
            # Smooth transition to target weights
            self.tensor_weights = 0.9 * self.tensor_weights + 0.1 * target_weights
            
            # Renormalize
            self.tensor_weights = self.tensor_weights / np.mean(self.tensor_weights)
            
            logger.info("Tensor weights rebalanced")
            
        except Exception as e:
            logger.error(f"Failed to perform tensor rebalancing: {e}")

    def get_live_handler_status(self) -> Dict[str, Any]:
        """Get status of live handlers."""
        return {
            "total_handlers": len(self.live_handlers),
            "active_handlers": sum(1 for h in self.live_handlers.values() if h["status"] == "active"),
            "average_similarity": np.mean([h["similarity"] for h in self.live_handlers.values()]),
            "total_executions": sum(h["execution_count"] for h in self.live_handlers.values()),
            "average_success_rate": np.mean([h["success_rate"] for h in self.live_handlers.values()]),
            "tensor_weight_variance": float(np.var(self.tensor_weights)),
            "rebalancing_threshold": self.rebalancing_threshold
        }

    def trigger_entry_exit(self, signal_packet, market_state, ccxt_executor):
        hash_vec = signal_packet["hash_vec"]
        asset = signal_packet.get("asset", "BTC/USDT")
        strategy_info = self.select_strategy(hash_vec, asset_hint=asset)
        strategy = strategy_info["strategy"]
        
        # Update API data cache
        self.api_data_cache.update(signal_packet.get("api_data", {}))
        
        # Update tensor weights from API data
        self.update_tensor_weights_from_api_data(self.api_data_cache)
        
        # Example: Use profit/drawdown logic
        if market_state["profit"] < 0.01 or market_state["trend"] == "bearish":
            asset = random.choice(
                ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
            )
            strategy = load_strategy(asset)
            self.dashboard_hook("asset_switch", {
                "asset": asset, 
                "reason": "Low profit or bearish trend"
            })
        
        # Execute via CCXT
        order = ccxt_executor.execute_trade(asset, strategy, signal_packet)
        
        # Update handler statistics
        if strategy_info["unified_hash"] in self.live_handlers:
            handler = self.live_handlers[strategy_info["unified_hash"]]
            handler["execution_count"] += 1
            # Update success rate based on order result
            if order.get("success", False):
                handler["success_rate"] = (handler["success_rate"] * 0.9 + 0.1)
            else:
                handler["success_rate"] = (handler["success_rate"] * 0.9)
        
        log_profit_tick({
            "order": order,
            "strategy": strategy,
            "asset": asset,
            "matrix_file": strategy_info["matrix_file"],
            "similarity": strategy_info["similarity"],
            "tensor_weights": self.tensor_weights.tolist()
        })
        
        self.dashboard_hook("trade_executed", {
            "order": order,
            "strategy": strategy,
            "asset": asset,
            "tensor_weights_updated": True
        })
        
        return {
            "order": order,
            "strategy": strategy,
            "asset": asset,
            "matrix_file": strategy_info["matrix_file"],
            "similarity": strategy_info["similarity"],
            "tensor_weights": self.tensor_weights.tolist()
        }