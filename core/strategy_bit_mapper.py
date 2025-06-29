# -*- coding: utf-8 -*-
"""
Strategy Bit Mapping Engine.

Expands 4-bit to 8-bit and 16-bit strategies with randomization
and flip-switch logic for increased strategy diversity.

Implements dualistic mirror functions for self-similarity detection
and profit vectorization mapping.
"""

import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


class StrategyBitMapper:
    """
    Strategy Bit Mapping Engine for expanding trading strategies.
    
    Supports 4-bit to 8-bit and 16-bit expansion with:
    - Flip-switch logic
    - Mirror functions
    - Randomization
    - Self-similarity detection
    """
    
    def __init__(
        self,
        enable_randomization: bool = True,
        enable_mirror_functions: bool = True,
        enable_self_similarity: bool = True,
        strategy_pool_size: int = 16,
        random_seed: Optional[int] = None,
    ):
        """Initialize the strategy bit mapper.
        
        Args:
            enable_randomization: Enable randomization in strategy expansion
            enable_mirror_functions: Enable mirror function logic
            enable_self_similarity: Enable self-similarity detection
            strategy_pool_size: Size of strategy pool for randomization
            random_seed: Random seed for reproducible results
        """
        self.enable_randomization = enable_randomization
        self.enable_mirror_functions = enable_mirror_functions
        self.enable_self_similarity = enable_self_similarity
        self.strategy_pool_size = strategy_pool_size
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Strategy pool for randomization
        self.strategy_pool = self._generate_strategy_pool()
        
        # Performance tracking
        self.mapping_stats = {
            "total_mappings": 0,
            "flip_mappings": 0,
            "mirror_mappings": 0,
            "random_mappings": 0,
            "self_similarity_detections": 0,
            "avg_processing_time": 0.0,
        }
        
        # Strategy history for self-similarity detection
        self.strategy_history: List[Dict[str, Union[int, float, str]]] = []
        self.max_history_size = 1000
        
        logger.info(
            f"StrategyBitMapper initialized: "
            f"randomization={enable_randomization}, "
            f"mirror_functions={enable_mirror_functions}, "
            f"self_similarity={enable_self_similarity}, "
            f"pool_size={strategy_pool_size}"
        )
    
    def _generate_strategy_pool(self) -> List[int]:
        """Generate strategy pool for randomization."""
        pool = []
        for i in range(self.strategy_pool_size):
            # Generate 4-bit strategies
            strategy = random.randint(0, 15)  # 0 to 15 (4 bits)
            pool.append(strategy)
        return pool
    
    def expand_strategy_bits(
        self, 
        base_bits: int, 
        target_depth: int = 8,
        mode: str = "flip",
        ferris_phase: Optional[float] = None
    ) -> List[int]:
        """
        Expand 4-bit strategy to 8-bit or 16-bit with specified mode.
        
        Args:
            base_bits: Base 4-bit strategy (0-15)
            target_depth: Target bit depth (8 or 16)
            mode: Expansion mode ('flip', 'mirror', 'random', 'ferris')
            ferris_phase: Ferris wheel phase for phase-dependent expansion
            
        Returns:
            List of expanded strategy bits
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not (0 <= base_bits <= 15):
                raise ValueError(f"base_bits must be 0-15, got {base_bits}")
            if target_depth not in [8, 16]:
                raise ValueError(f"target_depth must be 8 or 16, got {target_depth}")
            
            # Determine expansion mode
            if mode == "ferris" and ferris_phase is not None:
                expanded_strategies = self._ferris_expansion(base_bits, target_depth, ferris_phase)
            elif mode == "flip":
                expanded_strategies = self._flip_expansion(base_bits, target_depth)
            elif mode == "mirror":
                expanded_strategies = self._mirror_expansion(base_bits, target_depth)
            elif mode == "random":
                expanded_strategies = self._random_expansion(base_bits, target_depth)
            else:
                # Default to flip mode
                expanded_strategies = self._flip_expansion(base_bits, target_depth)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(mode, processing_time)
            
            # Store in history for self-similarity detection
            self._store_strategy_history(base_bits, expanded_strategies, mode, ferris_phase)
            
            logger.debug(
                f"Strategy expanded: {base_bits} -> {len(expanded_strategies)} strategies "
                f"(mode={mode}, depth={target_depth}, time={processing_time:.6f}s)"
            )
            
            return expanded_strategies
            
        except Exception as e:
            logger.error(f"Strategy expansion failed: {e}")
            # Return fallback strategy
            return [base_bits] * (target_depth // 4)
    
    def _flip_expansion(self, base_bits: int, target_depth: int) -> List[int]:
        """Flip-switch expansion with randomization."""
        strategies = []
        num_strategies = target_depth // 4
        
        for i in range(num_strategies):
            if self.enable_randomization:
                # Random flip mask from strategy pool
                flip_mask = random.choice(self.strategy_pool)
                # Apply flip with some probability
                if random.random() < 0.7:  # 70% chance of flip
                    strategy = base_bits ^ flip_mask
                else:
                    strategy = base_bits
            else:
                # Deterministic flip
                flip_mask = (i + 1) % 16
                strategy = base_bits ^ flip_mask
            
            strategies.append(strategy & 0xF)  # Ensure 4-bit
        
        self.mapping_stats["flip_mappings"] += 1
        return strategies
    
    def _mirror_expansion(self, base_bits: int, target_depth: int) -> List[int]:
        """Mirror function expansion."""
        strategies = []
        num_strategies = target_depth // 4
        
        # Create mirror of base strategy
        mirror_bits = (~base_bits) & 0xF
        
        for i in range(num_strategies):
            if i < num_strategies // 2:
                strategies.append(base_bits)
            else:
                strategies.append(mirror_bits)
        
        self.mapping_stats["mirror_mappings"] += 1
        return strategies
    
    def _random_expansion(self, base_bits: int, target_depth: int) -> List[int]:
        """Random expansion from strategy pool."""
        strategies = [base_bits]  # Always include base strategy
        
        num_additional = (target_depth // 4) - 1
        for _ in range(num_additional):
            strategy = random.choice(self.strategy_pool)
            strategies.append(strategy)
        
        self.mapping_stats["random_mappings"] += 1
        return strategies
    
    def _ferris_expansion(self, base_bits: int, target_depth: int, ferris_phase: float) -> List[int]:
        """Ferris wheel phase-dependent expansion."""
        strategies = []
        num_strategies = target_depth // 4
        
        # Use Ferris phase to modulate expansion
        phase_factor = np.cos(ferris_phase)
        phase_weight = (phase_factor + 1) / 2  # Normalize to [0, 1]
        
        for i in range(num_strategies):
            # Phase-dependent strategy selection
            if phase_weight > 0.7:  # High phase alignment
                strategy = base_bits
            elif phase_weight < 0.3:  # Low phase alignment
                strategy = (~base_bits) & 0xF  # Mirror
            else:  # Medium phase alignment
                # Random strategy with phase influence
                if random.random() < phase_weight:
                    strategy = base_bits
                else:
                    strategy = random.choice(self.strategy_pool)
            
            strategies.append(strategy)
        
        return strategies
    
    def detect_self_similarity(
        self, 
        current_strategies: List[int], 
        similarity_threshold: float = 0.8
    ) -> Dict[str, Union[bool, float, List[int]]]:
        """
        Detect self-similarity in strategy patterns.
        
        Args:
            current_strategies: Current strategy list
            similarity_threshold: Threshold for similarity detection
            
        Returns:
            Dictionary with similarity detection results
        """
        if not self.enable_self_similarity or len(self.strategy_history) < 2:
            return {
                "is_similar": False,
                "similarity_score": 0.0,
                "similar_strategies": [],
                "detection_time": time.time()
            }
        
        try:
            # Convert strategies to binary representation for comparison
            current_binary = self._strategies_to_binary(current_strategies)
            
            max_similarity = 0.0
            similar_strategies = []
            
            # Compare with historical strategies
            for history_entry in self.strategy_history[-100:]:  # Last 100 entries
                if "strategies" not in history_entry:
                    continue
                
                historical_binary = self._strategies_to_binary(history_entry["strategies"])
                similarity = self._compute_binary_similarity(current_binary, historical_binary)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    similar_strategies = history_entry["strategies"]
            
            is_similar = max_similarity > similarity_threshold
            
            if is_similar:
                self.mapping_stats["self_similarity_detections"] += 1
            
            return {
                "is_similar": is_similar,
                "similarity_score": max_similarity,
                "similar_strategies": similar_strategies,
                "detection_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Self-similarity detection failed: {e}")
            return {
                "is_similar": False,
                "similarity_score": 0.0,
                "similar_strategies": [],
                "detection_time": time.time()
            }
    
    def _strategies_to_binary(self, strategies: List[int]) -> np.ndarray:
        """Convert strategy list to binary representation."""
        binary_list = []
        for strategy in strategies:
            # Convert 4-bit strategy to binary
            binary = format(strategy & 0xF, '04b')
            binary_list.extend([int(b) for b in binary])
        return np.array(binary_list)
    
    def _compute_binary_similarity(self, binary1: np.ndarray, binary2: np.ndarray) -> float:
        """Compute similarity between binary representations."""
        # Pad shorter array with zeros
        max_len = max(len(binary1), len(binary2))
        padded1 = np.pad(binary1, (0, max_len - len(binary1)), 'constant')
        padded2 = np.pad(binary2, (0, max_len - len(binary2)), 'constant')
        
        # Compute Hamming similarity
        matches = np.sum(padded1 == padded2)
        total_bits = len(padded1)
        
        return matches / total_bits if total_bits > 0 else 0.0
    
    def _store_strategy_history(
        self, 
        base_bits: int, 
        expanded_strategies: List[int], 
        mode: str, 
        ferris_phase: Optional[float]
    ) -> None:
        """Store strategy in history for self-similarity detection."""
        history_entry = {
            "timestamp": time.time(),
            "base_bits": base_bits,
            "strategies": expanded_strategies,
            "mode": mode,
            "ferris_phase": ferris_phase,
            "strategy_hash": hash(tuple(expanded_strategies))
        }
        
        self.strategy_history.append(history_entry)
        
        # Maintain history size
        if len(self.strategy_history) > self.max_history_size:
            self.strategy_history.pop(0)
    
    def _update_stats(self, mode: str, processing_time: float) -> None:
        """Update mapping statistics."""
        self.mapping_stats["total_mappings"] += 1
        
        # Update average processing time
        total_time = self.mapping_stats["avg_processing_time"] * (
            self.mapping_stats["total_mappings"] - 1
        )
        self.mapping_stats["avg_processing_time"] = (
            (total_time + processing_time) / self.mapping_stats["total_mappings"]
        )
    
    def get_strategy_metrics(self, strategies: List[int]) -> Dict[str, Union[int, float]]:
        """
        Compute metrics for a strategy list.
        
        Args:
            strategies: List of strategy bits
            
        Returns:
            Dictionary of strategy metrics
        """
        try:
            metrics = {
                "strategy_count": len(strategies),
                "unique_strategies": len(set(strategies)),
                "diversity_ratio": len(set(strategies)) / len(strategies),
                "avg_strategy": float(np.mean(strategies)),
                "strategy_std": float(np.std(strategies)),
                "min_strategy": min(strategies),
                "max_strategy": max(strategies),
            }
            
            # Compute entropy of strategy distribution
            strategy_counts = {}
            for strategy in strategies:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            probabilities = [count / len(strategies) for count in strategy_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            metrics["entropy"] = float(entropy)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Strategy metrics computation failed: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Union[int, float]]:
        """Get mapping performance summary."""
        return self.mapping_stats.copy()
    
    def clear_history(self) -> None:
        """Clear strategy history."""
        self.strategy_history.clear()
        logger.info("Strategy history cleared")


def expand_strategy_bits(
    base_bits: int, 
    strategy_pool: List[int], 
    mode: str = "flip"
) -> List[int]:
    """
    Standalone function for strategy bit expansion.
    
    Args:
        base_bits: Base 4-bit strategy
        strategy_pool: Pool of strategies for randomization
        mode: Expansion mode ('flip', 'mirror', 'random')
        
    Returns:
        List of expanded strategies
    """
    mapper = StrategyBitMapper()
    return mapper.expand_strategy_bits(base_bits, target_depth=8, mode=mode)


# Global instance for easy access
strategy_bit_mapper = StrategyBitMapper()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test strategy expansion
    base_strategy = 0b1010  # 10 in decimal
    
    print(f"Base strategy: {base_strategy} (binary: {format(base_strategy, '04b')})")
    
    # Test different expansion modes
    modes = ["flip", "mirror", "random", "ferris"]
    
    for mode in modes:
        if mode == "ferris":
            ferris_phase = np.pi / 4  # 45 degrees
            expanded = strategy_bit_mapper.expand_strategy_bits(
                base_strategy, target_depth=8, mode=mode, ferris_phase=ferris_phase
            )
        else:
            expanded = strategy_bit_mapper.expand_strategy_bits(
                base_strategy, target_depth=8, mode=mode
            )
        
        print(f"\n{mode.upper()} expansion:")
        for i, strategy in enumerate(expanded):
            print(f"  Strategy {i+1}: {strategy} (binary: {format(strategy, '04b')})")
        
        # Compute metrics
        metrics = strategy_bit_mapper.get_strategy_metrics(expanded)
        print(f"  Metrics: {metrics}")
    
    # Test self-similarity detection
    print("\nSelf-similarity detection:")
    current_strategies = [10, 5, 15, 2]
    similarity_result = strategy_bit_mapper.detect_self_similarity(current_strategies)
    print(f"  Result: {similarity_result}")
    
    # Performance summary
    performance = strategy_bit_mapper.get_performance_summary()
    print(f"\nPerformance summary: {performance}") 