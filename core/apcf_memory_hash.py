# -*- coding: utf-8 -*-
"""
APCF Memory Hashing Layer
=========================

This module is responsible for compressing APCF outcomes into deterministic,
repeatable hash blocks. This creates the "APCF Memory Pool," a searchable,
encrypted history of every decision and its result. This memory is crucial
for AI agent voting, fallback recall, and cross-asset pattern mirroring.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List

# from .adaptive_profit_cycle_function import APCFResult

logger = logging.getLogger(__name__)


class APCFMemoryHasher:
    """
    Creates and manages the APCF Memory Pool by hashing outcomes.
    """

    def __init__(self, memory_pool_path: str = "core/logs/apcf_memory_pool.json"):
        """
        Initializes the memory hasher.

        Args:
            memory_pool_path: Path to the file storing the memory pool.
        """
        self.memory_pool_path = memory_pool_path
        self.memory_pool = self._load_memory_pool()
        logger.info("APCF Memory Hasher initialized.")

    def _load_memory_pool(self) -> Dict[str, Dict[str, Any]]:
        """Loads the memory pool from a JSON file."""
        try:
            with open(self.memory_pool_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_memory_pool(self):
        """Saves the current memory pool to its file."""
        try:
            with open(self.memory_pool_path, "w") as f:
                json.dump(self.memory_pool, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save APCF memory pool: {e}")

    def create_memory_block(self, apcf_result: Any, outcome_roi: float) -> Dict[str, Any]:
        """
        Creates a new, hashed memory block from an APCF result and its outcome.

        Args:
            apcf_result: The APCFResult object.
            outcome_roi: The Return on Investment from the resulting action.

        Returns:
            The newly created memory block.
        """

        # The block's content is the essential data about the event
        block_content = {
            "apcf": apcf_result.apcf_value,
            "tick": apcf_result.timestamp,
            "outcome": f"profit_{outcome_roi:+.4f}%",
            "components": apcf_result.components,
            "signature": apcf_result.mathematical_signature,
        }

        # The block ID is a hash of its content, ensuring integrity
        block_id = hashlib.sha256(json.dumps(block_content, sort_keys=True).encode()).hexdigest()

        memory_block = {"block": block_id, **block_content}

        # Add to the pool and save
        self.memory_pool[block_id] = memory_block
        self._save_memory_pool()

        logger.info(f"Created new APCF memory block with ID: {block_id[:12]}...")
        return memory_block

    def find_similar_memories(self, apcf_components: Dict[str, float], top_n=5) -> List[Dict[str, Any]]:
        """
        Finds past memories that are similar to a new, potential event.
        This is used for predictive "voting" by AI agents.

        Args:
            apcf_components: The components of a new APCF calculation.
            top_n: The number of similar memories to return.

        Returns:
            A list of the most similar past memory blocks.
        """

        # This is a simplified similarity search. A real implementation might use
        # vector embeddings (e.g., from a language model) for more nuanced
        # search.
        scores = []
        for block_id, memory in self.memory_pool.items():
            similarity = self._calculate_component_similarity(apcf_components, memory["components"])
            scores.append((similarity, memory))

        # Sort by similarity score in descending order
        scores.sort(key=lambda x: x[0], reverse=True)

        return [memory for score, memory in scores[:top_n]]

    def _calculate_component_similarity(self, comp1: Dict, comp2: Dict) -> float:
        """Calculates a simple similarity score between two component dictionaries."""
        keys = set(comp1.keys()) & set(comp2.keys())
        if not keys:
            return 0.0

        # Calculate similarity based on normalized distance of component values
        distances = []
        for key in keys:
            val1 = comp1[key]
            val2 = comp2[key]
            # Avoid division by zero if one value is zero
            if val1 != 0 or val2 != 0:
                distance = abs(val1 - val2) / (abs(val1) + abs(val2) / 2 + 1e-9)
                distances.append(distance)

        if not distances:
            return 0.0

        # Average distance, inverted to become a similarity score
        avg_distance = sum(distances) / len(distances)
        return 1.0 - min(avg_distance, 1.0)


# Global instance
apcf_memory_hasher = APCFMemoryHasher()
