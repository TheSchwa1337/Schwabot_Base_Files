#!/usr/bin/env python3
"""
riddle_gemm.py - Recursive Interlocking Dimensional Logic & GEMM Engine.

Serves as Schwabot's adaptive intelligence interface. It provides matrix-based
vector comparison logic, strategy scoring, and logic sequence modulation using
General Matrix-to-Matrix (GEMM) style operations.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple

from core.utils.math_utils import (
    calculate_hash_distance,
    calculate_weighted_confidence,
)

logger = logging.getLogger(__name__)


class RiddleGEMMEngine:
    """
    Manages and scores trading strategies against the current market state.
    """

    def __init__(self, vector_size: int, distance_threshold: float = 10.0):
        """
        Initialize the R.I.D.D.L.E. GEMM Engine.

        Args:
            vector_size: The dimensionality of the state and strategy vectors.
            distance_threshold: The max hash distance to consider strategies related.
        """
        self.vector_size = vector_size
        self.distance_threshold = distance_threshold

        # Stores strategy vectors, keyed by a unique name.
        self.strategy_vectors: Dict[str, np.ndarray] = {}
        # Stores a content hash of each strategy vector for quick comparison.
        self.strategy_hashes: Dict[str, str] = {}
        # Stores pre-calculated weight matrices for different scenarios.
        self.weight_matrices: Dict[str, np.ndarray] = {
            "default": np.identity(vector_size)
        }

        logger.info(f"RiddleGEMMEngine initialized with vector size {vector_size}.")

    def register_strategy(self, name: str, vector: List[float], content_hash: str) -> None:
        """
        Register a new strategy vector and its content hash.
        
        Args:
            name: The unique name for the strategy.
            vector: The list of floats representing the strategy's parameters.
            content_hash: A SHA-256 hash of the strategy's content.
        """
        if len(vector) != self.vector_size:
            logger.warning(
                f"Strategy '{name}' has incorrect vector size. "
                f"Expected {self.vector_size}, got {len(vector)}."
            )
            return
        
        self.strategy_vectors[name] = np.array(vector)
        self.strategy_hashes[name] = content_hash
        logger.debug(f"Registered strategy '{name}'.")

    def register_weight_matrix(self, name: str, matrix: List[List[float]]) -> None:
        """Register a new named weight matrix for state processing."""
        matrix_np = np.array(matrix)
        if matrix_np.shape != (self.vector_size, self.vector_size):
            logger.warning(
                f"Weight matrix '{name}' has incorrect shape. "
                f"Expected ({self.vector_size}, {self.vector_size}), got {matrix_np.shape}."
            )
            return
        self.weight_matrices[name] = matrix_np
        logger.info(f"Registered weight matrix '{name}'.")

    def score_strategies(
        self, current_state_vector: List[float], matrix_name: str = "default"
    ) -> Dict[str, float]:
        """
        Score all registered strategies against the current market state vector.

        Args:
            current_state_vector: The vector representing the current market state.
            matrix_name: The name of the weight matrix to use for transformation.

        Returns:
            A dictionary mapping strategy names to their confidence scores.
        """
        if len(current_state_vector) != self.vector_size:
            logger.error("Current state vector has incorrect size.")
            return {}

        state_vector_np = np.array(current_state_vector)
        
        # Apply GEMM-style transformation
        weight_matrix = self.weight_matrices.get(matrix_name, self.weight_matrices["default"])
        transformed_state = np.dot(weight_matrix, state_vector_np)

        scores = {}
        for name, strategy_vec in self.strategy_vectors.items():
            # Calculate confidence score using the weighted sigmoid utility
            confidence = calculate_weighted_confidence(strategy_vec, transformed_state)
            scores[name] = confidence
            
        return scores

    def find_best_strategy(
        self, current_state_vector: List[float], matrix_name: str = "default"
    ) -> Tuple[Optional[str], float]:
        """
        Find the single best strategy for the given market state.

        Returns:
            A tuple containing the name of the best strategy and its score,
            or (None, 0.0) if no strategies are registered.
        """
        scores = self.score_strategies(current_state_vector, matrix_name)
        if not scores:
            return None, 0.0

        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]
        
        logger.info(
            f"Best strategy found: '{best_strategy}' with score {best_score:.4f}"
        )
        
        # --- HOOKS INTO OTHER MODULES (Example) ---
        # if best_score > 0.8:
        #     # Hooks into profit_routing_engine.py or strategy_mapper.py
        #     self.trigger_strategy_activation(best_strategy)
        #
        # # Hooks into fault_bus.py to provide feedback
        # self.report_scoring_confidence(best_score)
        
        return best_strategy, best_score

    def find_related_strategies(self, strategy_name: str) -> List[Dict[str, Any]]:
        """
        Find strategies related to the given one based on hash distance.

        Args:
            strategy_name: The name of the strategy to find relatives for.

        Returns:
            A list of related strategies, including their name and distance.
        """
        if strategy_name not in self.strategy_hashes:
            logger.warning(f"Strategy '{strategy_name}' not found for relation search.")
            return []

        source_hash = self.strategy_hashes[strategy_name]
        related = []
        for name, target_hash in self.strategy_hashes.items():
            if name == strategy_name:
                continue
            
            # Calculate distance using the hash distance utility
            distance = calculate_hash_distance(source_hash, target_hash, method='hamming')
            
            if distance <= self.distance_threshold:
                related.append({"name": name, "distance": distance})
        
        # Sort by distance (closest first)
        related.sort(key=lambda x: x["distance"])
        return related 