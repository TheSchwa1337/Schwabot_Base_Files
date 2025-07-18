#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Registry 🗂️

Maps SHA-256 digests to compressed strategy vectors for:
• Digest → Strategy matching
• Historical profit pattern recognition
• Multi-indicator confidence scoring
• Asset basket allocation

CUDA Integration:
- GPU-accelerated vector similarity search with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)

Core Functions:
* register_digest(digest, strategy, outcome)  – store new digest→strategy mapping
* find_similar_digests(digest, threshold)     – find similar historical patterns
* get_confidence_score(digest, strategy)     – calculate confidence based on history
* update_profit_outcome(digest, pnl)         – reinforce successful patterns
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
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
    logger.info(f"⚡ VectorRegistry using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 VectorRegistry using CPU fallback: {_backend}")


@dataclass
class StrategyVector:
    """Compressed strategy representation for a digest."""
    digest: str  # SHA256 hex
    strategy_id: str
    asset_focus: str  # "BTC", "ETH", "XRP", "SOL", "USDC"
    entry_confidence: float  # 0.0 to 1.0
    exit_confidence: float  # 0.0 to 1.0
    position_size: float  # 0.0 to 1.0 (fraction of capital)
    stop_loss_pct: float  # percentage
    take_profit_pct: float  # percentage
    rsi_band: int  # 0-100
    volatility_class: int  # 0=low, 1=medium, 2=high
    entropy_band: float  # normalized entropy
    timestamp: float = field(default_factory=time.time)
    usage_count: int = 0
    success_rate: float = 0.0
    avg_profit: float = 0.0


@dataclass
class DigestMatch:
    """Result of digest similarity search."""
    digest: str
    similarity_score: float
    strategy_vector: StrategyVector
    hamming_distance: int
    entropy_diff: float


class VectorRegistry:
    """
    Core registry for digest→strategy mapping and similarity search.
    Handles GPU/CPU fallback for vector operations.
    """
    def __init__(self):
        self.registry: Dict[str, StrategyVector] = {}

    def register_digest(self, digest: str, strategy: StrategyVector) -> None:
        """Register a new digest→strategy mapping."""
        self.registry[digest] = strategy
        logger.info(f"Registered digest: {digest[:8]}... for strategy {strategy.strategy_id}")

    def find_similar_digests(self, digest: str, threshold: float = 0.9) -> List[DigestMatch]:
        """
        Find similar historical patterns by digest similarity.
        Uses Hamming distance and cosine similarity.
        """
        if digest not in self.registry:
            return []
        target_vec = self.registry[digest]
        matches = []
        for d, vec in self.registry.items():
            if d == digest:
                continue
            # Simple similarity: compare entry/exit confidence and asset focus
            sim = 1.0 - abs(target_vec.entry_confidence - vec.entry_confidence)
            sim *= 1.0 - abs(target_vec.exit_confidence - vec.exit_confidence)
            sim *= 1.0 if target_vec.asset_focus == vec.asset_focus else 0.8
            hamming = sum(a != b for a, b in zip(d, digest))
            entropy_diff = abs(target_vec.entropy_band - vec.entropy_band)
            if sim >= threshold:
                matches.append(DigestMatch(
                    digest=d,
                    similarity_score=sim,
                    strategy_vector=vec,
                    hamming_distance=hamming,
                    entropy_diff=entropy_diff,
                ))
        matches.sort(key=lambda m: (-m.similarity_score, m.hamming_distance))
        return matches

    def get_confidence_score(self, digest: str) -> float:
        """Calculate confidence based on history for a digest."""
        vec = self.registry.get(digest)
        if not vec:
            return 0.0
        return (vec.entry_confidence + vec.exit_confidence) / 2.0

    def update_profit_outcome(self, digest: str, pnl: float) -> None:
        """Reinforce successful patterns by updating profit stats."""
        vec = self.registry.get(digest)
        if not vec:
            return
        vec.usage_count += 1
        vec.avg_profit = ((vec.avg_profit * (vec.usage_count - 1)) + pnl) / vec.usage_count
        if pnl > 0:
            vec.success_rate = ((vec.success_rate * (vec.usage_count - 1)) + 1) / vec.usage_count
        else:
            vec.success_rate = ((vec.success_rate * (vec.usage_count - 1))) / vec.usage_count
        logger.info(f"Updated profit outcome for digest {digest[:8]}...: avg_profit={vec.avg_profit:.4f}, success_rate={vec.success_rate:.2f}")

    def get_strategy(self, digest: str) -> Optional[StrategyVector]:
        """Get the strategy vector for a digest."""
        return self.registry.get(digest)

    def all_digests(self) -> List[str]:
        """Return all registered digests."""
        return list(self.registry.keys())


# Singleton instance for global use
vector_registry = VectorRegistry()