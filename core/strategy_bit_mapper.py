import logging
import random
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from core.matrix_mapper import load_matrix_from_file
from core.unified_math_system import generate_unified_hash
from core.strategy_loader import load_strategy
from core.fractal_core import fractal_quantize_vector

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

class StrategyBitMapper:
    """
    StrategyBitMapper: Handles bitwise strategy expansion, hash-to-matrix matching,
    fallback logic, and dashboard/profit logger integration for real-time trading.
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

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    def compute_cosine_similarity(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return float(np.dot(a, b))

    def expand_strategy_bits(self, strategy_id: int, target_bits: int = 8, mode: ExpansionMode = ExpansionMode.RANDOM) -> int:
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
        else:
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
        fallback_asset = asset_hint or random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
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

    def trigger_entry_exit(self, signal_packet, market_state, ccxt_executor):
        hash_vec = signal_packet["hash_vec"]
        asset = signal_packet.get("asset", "BTC/USDT")
        strategy_info = self.select_strategy(hash_vec, asset_hint=asset)
        strategy = strategy_info["strategy"]
        # Example: Use profit/drawdown logic
        if market_state["profit"] < 0.01 or market_state["trend"] == "bearish":
            asset = random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
            strategy = load_strategy(asset)
            self.dashboard_hook("asset_switch", {"asset": asset, "reason": "Low profit or bearish trend"})
        # Execute via CCXT
        order = ccxt_executor.execute_trade(asset, strategy, signal_packet)
        log_profit_tick({
            "order": order,
            "strategy": strategy,
            "asset": asset,
            "matrix_file": strategy_info["matrix_file"],
            "similarity": strategy_info["similarity"]
        })
        self.dashboard_hook("trade_executed", {
            "order": order,
            "strategy": strategy,
            "asset": asset
        })
        return {
            "order": order,
            "strategy": strategy,
            "asset": asset,
            "matrix_file": strategy_info["matrix_file"],
            "similarity": strategy_info["similarity"]
        }