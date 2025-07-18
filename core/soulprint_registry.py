"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soulprint Registry 🧬

Tracks and manages trade event 'soulprints' for:
• Phase/drift/tensor config analytics
• Profit vector tracking
• Backtesting and live trade outcome logging
• Cross-asset and strategy performance analysis

Features:
- SHA-256 hash for soulprint identity
- Vectorized math/statistics (NumPy/CuPy, GPU fallback)
- Hooks for CCX T-matrix, buy/sell wall, tensor math
- File persistence for backtesting/live replay
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import cupy as cp
    USING_CUDA = True
    xp = cp
    _backend = 'cupy (GPU)'
except ImportError:
    import numpy as np
    USING_CUDA = False
    xp = np
    _backend = 'numpy (CPU)'

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(f"⚡ SoulprintRegistry using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 SoulprintRegistry using CPU fallback: {_backend}")

def soulprint_hash(vector: Dict[str, Any], strategy_id: str, timestamp: float) -> str:
    """Generate a SHA-256 hash for a soulprint entry."""
    payload = json.dumps({"vector": vector, "strategy_id": strategy_id,
                         "timestamp": timestamp}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class SoulprintEntry:
    """Class for Schwabot trading functionality."""
    soulprint: str  # SHA-256 hash
    timestamp: float
    vector: Dict[str, float]  # e.g., {"phase": ..., "drift": ..., ...}
    strategy_id: str
    confidence: float
    canonical_hash: Optional[str] = None  # Reference to canonical trade registry
    is_executed: bool = False
    profit_result: Optional[float] = None
    replayable: bool = True


class SoulprintRegistry:
    """Class for Schwabot trading functionality."""
    """
    Registry for logging/querying Schwafit triggers, profit vectors, phase/drift optimization, and cross-asset analytics.
    Handles GPU/CPU fallback for vector operations.
    """
    
    def __init__(self, registry_file: Optional[str] = None) -> None:
        self.entries: List[SoulprintEntry] = []
        self.registry_file = registry_file
        if registry_file:
            self._load()

    def register_soulprint(self, vector: Dict[str, float], strategy_id: str, confidence: float, canonical_hash: Optional[str] = None, is_executed: bool = False, profit_result: Optional[float] = None, replayable: bool = True, timestamp: Optional[float] = None) -> str:
        """Register a new soulprint entry (trade event)."""
        ts = timestamp or time.time()
        sp_hash = soulprint_hash(vector, strategy_id, ts)
        entry = SoulprintEntry(
            soulprint=sp_hash,
            timestamp=ts,
            vector=vector,
            strategy_id=strategy_id,
            confidence=confidence,
            canonical_hash=canonical_hash,
            is_executed=is_executed,
            profit_result=profit_result,
            replayable=replayable,
        )
        self.entries.append(entry)
        logger.info(f"🧬 Registered soulprint: {sp_hash[:8]}... for strategy {strategy_id}")
        if self.registry_file:
            self._save()
        return sp_hash

    def log_backtest_signal(self, signal_data: Dict[str, Any]) -> None:
        """Log a signal from a backtest run, capturing key performance and context indicators."""
        vector = signal_data.get("signal_vector", {})
        strategy_id = signal_data.get("strategy_id", "unknown")
        confidence = signal_data.get("confidence", 0.0)
        profit_result = signal_data.get("projected_gain")
        is_executed = signal_data.get("executed", False)
        timestamp = signal_data.get("timestamp", time.time())
        self.register_soulprint(
            vector=vector,
            strategy_id=strategy_id,
            confidence=confidence,
            is_executed=is_executed,
            profit_result=profit_result,
            replayable=True,
            timestamp=timestamp,
        )

    def update_profit_outcome(self, soulprint: str, profit: float) -> None:
        """Update the profit result for a given soulprint (by hash)."""
        for entry in self.entries:
            if entry.soulprint == soulprint:
                entry.profit_result = profit
                logger.info(f"Updated profit for soulprint {soulprint[:8]}...: {profit:.4f}")
                if self.registry_file:
                    self._save()
                return

    def get_best_phase(self, asset: str, window: int = 1000) -> Optional[Dict[str, Any]]:
        """Return the phase/drift/tensor config with the highest profit in the last N entries for an asset."""
        filtered = [e for e in self.entries if e.vector.get("asset") == asset][-window:]
        if not filtered:
            return None
        profits = xp.array([e.profit_result if e.profit_result is not None else 0.0 for e in filtered])
        idx = int(xp.argmax(profits))
        best = filtered[idx]
        return {
            "vector": best.vector,
            "profit_result": best.profit_result,
            "strategy_id": best.strategy_id,
            "confidence": best.confidence,
            "timestamp": best.timestamp,
        }

    def get_profit_vector(self, asset: str, window: int = 1000) -> xp.ndarray:
        """Return a vector of profit results for an asset over the last N entries."""
        filtered = [e for e in self.entries if e.vector.get("asset") == asset][-window:]
        return xp.array([e.profit_result if e.profit_result is not None else 0.0 for e in filtered])

    def get_soulprint(self, soulprint_hash: str) -> Optional[SoulprintEntry]:
        """Get a soulprint entry by its hash."""
        for entry in self.entries:
            if entry.soulprint == soulprint_hash:
                return entry
        return None

    def all_soulprints(self) -> List[str]:
        """Return all registered soulprint hashes."""
        return [e.soulprint for e in self.entries]

    def _save(self) -> None:
        """Persist the registry to file (JSON)."""
        if not self.registry_file:
            return
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    def _load(self) -> None:
        """Load the registry from file (JSON)."""
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [SoulprintEntry(**e) for e in data]
        except Exception as e:
            logger.warning(f"Failed to load soulprint registry: {e}")


# Singleton instance for global use
soulprint_registry = SoulprintRegistry()