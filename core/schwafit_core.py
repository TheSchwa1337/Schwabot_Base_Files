#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwafit Core Module

Schwafit: Recursive, shape-matching, volatility-aware prediction system.
Implements delta, normalization, cosine/DTW, entropy, memory, and fit decision.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SchwafitCore:
    """
    Schwafit: Recursive, shape-matching, volatility-aware prediction system.
    
    Implements:
    - Second-order difference computation
    - Vector normalization (z-score, min-max)
    - Cosine similarity and DTW distance
    - Shannon entropy calculation
    - Pattern matching and fit scoring
    - Memory-based decision making
    """

    def __init__(
        self,
        window: int = 64,
        entropy_threshold: float = 2.5,
        fit_threshold: float = 0.85,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SchwafitCore.
        
        Args:
            window: Window size for pattern analysis
            entropy_threshold: Threshold for entropy-based decisions
            fit_threshold: Threshold for fit-based decisions
            config: Configuration dictionary
        """
        self.window = window
        self.entropy_threshold = entropy_threshold
        self.fit_threshold = fit_threshold
        self.config = config or {}
        
        # Memory storage for fit hashes, scores, profit, volatility
        self.memory: List[Dict[str, Any]] = []
        
        logger.info(f"✅ SchwafitCore initialized with window={window}, "
                   f"entropy_threshold={entropy_threshold}, fit_threshold={fit_threshold}")

    @staticmethod
    def delta2(series: List[float]) -> np.ndarray:
        """Compute second-order difference vector."""
        arr = np.array(series)
        return np.diff(arr, n=2)

    @staticmethod
    def normalize(vec: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalize vector using specified method.
        
        Args:
            vec: Input vector
            method: Normalization method ('zscore' or 'minmax')
            
        Returns:
            Normalized vector
        """
        if method == "zscore":
            mu = np.mean(vec)
            sigma = np.std(vec)
            return (vec - mu) / sigma if sigma > 0 else vec * 0
        elif method == "minmax":
            minv, maxv = np.min(vec), np.max(vec)
            return (vec - minv) / (maxv - minv) if maxv > minv else vec * 0
        else:
            return vec

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Dynamic Time Warping distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            DTW distance
        """
        n, m = len(a), len(b)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(a[i - 1] - b[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )
        
        return float(dtw_matrix[n, m])

    @staticmethod
    def entropy(vec: np.ndarray) -> float:
        """Compute Shannon entropy of normalized vector."""
        absvec = np.abs(vec)
        p = absvec / np.sum(absvec) if np.sum(absvec) > 0 else absvec
        p = p[p > 0]
        return float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0

    def fit_vector(
        self,
        price_series: List[float],
        pattern_library: List[np.ndarray],
        profit_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Main fit function. Returns dict with fit score, entropy, best matches, and decision.
        
        Args:
            price_series: Historical price data
            pattern_library: Library of pattern vectors
            profit_scores: Profit scores corresponding to patterns
            
        Returns:
            Dictionary containing fit analysis results
        """
        if len(price_series) < self.window + 2:
            logger.warning(f"Insufficient data: {len(price_series)} < {self.window + 2}")
            return self._empty_fit_result()
        
        # Compute second-order difference
        v = self.delta2(price_series[-(self.window + 2):])
        v_norm = self.normalize(v)
        ent = self.entropy(v_norm)
        v_hash = hashlib.sha256(v_norm.tobytes()).hexdigest()

        # Cosine similarity to all patterns
        sims = [self.cosine_similarity(v_norm, s) for s in pattern_library]
        top_indices = np.argsort(sims)[-3:][::-1]  # Top 3 matches
        top_scores = [sims[i] for i in top_indices]
        top_profits = [profit_scores[i] for i in top_indices]
        
        # Compute weighted fit score
        fit_score = float(np.average([s * p for s, p in zip(top_scores, top_profits)])
                         if top_scores else 0.0)

        # Decision logic
        decision = fit_score > self.fit_threshold and ent < self.entropy_threshold
        
        # Memory update
        memory_entry = {
            "hash": v_hash,
            "fit_score": fit_score,
            "entropy": ent,
            "top_scores": top_scores,
            "top_profits": top_profits,
            "decision": decision,
            "timestamp": len(self.memory),
        }
        self.memory.append(memory_entry)
        
        logger.info(f"Schwafit fit: hash={v_hash[:8]}, fit_score={fit_score:.3f}, "
                   f"entropy={ent:.3f}, decision={decision}")
        
        return {
            "fit_score": fit_score,
            "entropy": ent,
            "top_scores": top_scores,
            "top_profits": top_profits,
            "decision": decision,
            "hash": v_hash,
            "normalized_vector": v_norm,
        }

    def fit_vector_dtw(
        self,
        price_series: List[float],
        pattern_library: List[np.ndarray],
        profit_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Fit function using DTW distance instead of cosine similarity.
        
        Args:
            price_series: Historical price data
            pattern_library: Library of pattern vectors
            profit_scores: Profit scores corresponding to patterns
            
        Returns:
            Dictionary containing DTW-based fit analysis results
        """
        if len(price_series) < self.window + 2:
            return self._empty_fit_result()
        
        v = self.delta2(price_series[-(self.window + 2):])
        v_norm = self.normalize(v)
        ent = self.entropy(v_norm)
        v_hash = hashlib.sha256(v_norm.tobytes()).hexdigest()

        # DTW distance to all patterns (lower is better)
        distances = [self.dtw_distance(v_norm, s) for s in pattern_library]
        top_indices = np.argsort(distances)[:3]  # Top 3 matches (lowest distance)
        top_distances = [distances[i] for i in top_indices]
        top_profits = [profit_scores[i] for i in top_indices]
        
        # Convert distances to similarity scores (inverse relationship)
        max_dist = max(top_distances) if top_distances else 1.0
        top_scores = [1.0 - (d / max_dist) for d in top_distances]
        
        # Compute weighted fit score
        fit_score = float(np.average([s * p for s, p in zip(top_scores, top_profits)])
                         if top_scores else 0.0)

        decision = fit_score > self.fit_threshold and ent < self.entropy_threshold
        
        return {
            "fit_score": fit_score,
            "entropy": ent,
            "top_scores": top_scores,
            "top_profits": top_profits,
            "top_distances": top_distances,
            "decision": decision,
            "hash": v_hash,
            "method": "dtw",
        }

    def get_fit_memory(self) -> List[Dict[str, Any]]:
        """Return memory of all fits."""
        return self.memory

    def get_last_fit(self) -> Optional[Dict[str, Any]]:
        """Return the most recent fit result."""
        return self.memory[-1] if self.memory else None

    def clear_memory(self) -> None:
        """Clear the fit memory."""
        self.memory.clear()
        logger.info("✅ SchwafitCore memory cleared")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the fit memory."""
        if not self.memory:
            return {"count": 0, "avg_fit_score": 0.0, "avg_entropy": 0.0}
        
        fit_scores = [entry["fit_score"] for entry in self.memory]
        entropies = [entry["entropy"] for entry in self.memory]
        
        return {
            "count": len(self.memory),
            "avg_fit_score": float(np.mean(fit_scores)),
            "avg_entropy": float(np.mean(entropies)),
            "max_fit_score": float(np.max(fit_scores)),
            "min_entropy": float(np.min(entropies)),
        }

    def _empty_fit_result(self) -> Dict[str, Any]:
        """Return empty fit result when insufficient data."""
        return {
            "fit_score": 0.0,
            "entropy": 0.0,
            "top_scores": [],
            "top_profits": [],
            "decision": False,
            "hash": "",
            "error": "Insufficient data",
        }


# Factory function for creating SchwafitCore instances
def create_schwafit_core(
    window: int = 64,
    entropy_threshold: float = 2.5,
    fit_threshold: float = 0.85,
    config: Optional[Dict[str, Any]] = None,
) -> SchwafitCore:
    """
    Factory function to create a SchwafitCore instance.
    
    Args:
        window: Window size for pattern analysis
        entropy_threshold: Threshold for entropy-based decisions
        fit_threshold: Threshold for fit-based decisions
        config: Configuration dictionary
        
    Returns:
        Initialized SchwafitCore instance
    """
    return SchwafitCore(
        window=window,
        entropy_threshold=entropy_threshold,
        fit_threshold=fit_threshold,
        config=config,
    )
