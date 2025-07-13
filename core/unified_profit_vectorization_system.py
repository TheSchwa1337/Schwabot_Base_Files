#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Profit Vectorization System
==================================

Tracks and vectorizes profit metrics for Schwabot’s strategy core.
- Tick-based position entry timestamps
- Strategy matrix alignment score
- Hash-projected profit bands
- Entropy/drawdown/sigmoid scoring
- Export to hash pipeline for downstream consumers

Feeds into:
- profit_cycle_allocator.py
- strategy_bit_mapper.py
- fractal_core.py
- clean_unified_math.py
- backend_math.py
"""

import logging
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

# --- Utility Functions ---
def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    try:
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
    except Exception as e:
        logger.error(f"Sigmoid error: {e}")
        return 0.5

# --- Data Model ---
@dataclass
class ProfitVector:
    tick: int
    profit: float
    hash: str
    volatility: float
    drawdown: float
    vector_strength: float
    exit_type: str
    risk_profile: str
    timestamp: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

# --- Core System ---
class UnifiedProfitVectorizationSystem:
    """
    Core profit vectorization and hash export system for Schwabot.
    """
    def __init__(self):
        self.vectors: List[ProfitVector] = []
        self.callbacks: Dict[str, Callable] = {}
        self.max_history = 1000

    def generate_profit_vector(self, entry_tick: int, profit: float, strategy_hash: str, drawdown: float, entropy_delta: float, exit_type: str = "stack_hold", risk_profile: str = "low", meta: Optional[Dict[str, Any]] = None) -> ProfitVector:
        """
        Constructs a unified profit vector containing key trade metrics.
        """
        try:
            volatility = float(entropy_delta)
            vector_strength = sigmoid(profit - drawdown) * (1 - volatility)
            vector = ProfitVector(
                tick=entry_tick,
                profit=profit,
                hash=strategy_hash,
                volatility=volatility,
                drawdown=drawdown,
                vector_strength=vector_strength,
                exit_type=exit_type,
                risk_profile=risk_profile,
                meta=meta or {}
            )
            self._store_vector(vector)
            return vector
        except Exception as e:
            logger.error(f"Error generating profit vector: {e}")
            raise

    def _store_vector(self, vector: ProfitVector):
        self.vectors.append(vector)
        if len(self.vectors) > self.max_history:
            self.vectors.pop(0)

    def get_last_hash_profit_vectors(self, n: int = 5) -> List[ProfitVector]:
        """Return the last n profit vectors."""
        return self.vectors[-n:]

    def vectorize_new_trade_response(self, entry_tick: int, profit: float, strategy_hash: str, drawdown: float, entropy_delta: float, exit_type: str = "stack_hold", risk_profile: str = "low", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Vectorizes a new trade and returns the dictionary representation.
        """
        vector = self.generate_profit_vector(entry_tick, profit, strategy_hash, drawdown, entropy_delta, exit_type, risk_profile, meta)
        return asdict(vector)

    def synchronize_with_strategy_mapper(self, strategy_results: List[Dict[str, Any]]) -> None:
        """
        Integrate past strategy results and synchronize with the profit vector system.
        """
        for result in strategy_results:
            try:
                self.generate_profit_vector(
                    entry_tick=result.get("tick", 0),
                    profit=result.get("profit", 0.0),
                    strategy_hash=result.get("strategy_hash", ""),
                    drawdown=result.get("drawdown", 0.0),
                    entropy_delta=result.get("entropy_delta", 0.0),
                    exit_type=result.get("exit_type", "stack_hold"),
                    risk_profile=result.get("risk_profile", "low"),
                    meta=result
                )
            except Exception as e:
                logger.error(f"Error synchronizing strategy result: {e}")

    def export_to_hash_pipeline(self) -> List[Dict[str, Any]]:
        """
        Export all profit vectors to a hash pipeline for downstream consumers.
        """
        return [asdict(v) for v in self.vectors]

    def clear_history(self):
        self.vectors.clear()

    def register_callback(self, name: str, func: Callable):
        self.callbacks[name] = func

    def run_callback(self, name: str, *args, **kwargs):
        if name in self.callbacks:
            return self.callbacks[name](*args, **kwargs)
        else:
            logger.warning(f"Callback {name} not found.")
            return None

    # --- Real Integration Methods ---
    
    def integrate_with_clean_unified_math(self, math_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real integration with clean_unified_math.py."""
        try:
            from core.clean_unified_math import CleanUnifiedMathSystem
            math_system = CleanUnifiedMathSystem()
            
            # Integrate mathematical operations
            result = {
                'profit_vector': math_system.calculate_profit_vector(math_data),
                'volatility': math_system.calculate_volatility(math_data.get('prices', [])),
                'correlation': math_system.calculate_correlation(math_data.get('returns', [])),
                'optimization': math_system.optimize_profit(math_data.get('base_profit', 0.0))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error integrating with clean_unified_math: {e}")
            return {'error': str(e)}
    
    def integrate_with_backend_math(self, backend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real integration with backend_math.py."""
        try:
            from core.backend_math import BackendMathSystem
            backend_math = BackendMathSystem()
            
            # Integrate backend mathematical operations
            result = {
                'backend_profit': backend_math.calculate_backend_profit(backend_data),
                'advanced_metrics': backend_math.calculate_advanced_metrics(backend_data),
                'optimization_result': backend_math.optimize_strategy(backend_data)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error integrating with backend_math: {e}")
            return {'error': str(e)}

# --- Module-level instance ---
profit_vectorization_system = UnifiedProfitVectorizationSystem() 