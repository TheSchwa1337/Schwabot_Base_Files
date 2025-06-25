from core.unified_math_system import unified_math
import numpy as np
import math
# #!/usr/bin/env python3
"""
riddle_gemm.py - Recursive Interlocking Dimensional Logic & GEMM Engine.

Serves as Schwabot's adaptive intelligence interface. It provides matrix-based
vector comparison logic, strategy scoring, and logic sequence modulation using
General Matrix-to-Matrix (GEMM) style operations.
"""

# from core.unified_math_system import unified_math  # F811: duplicate import
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import time

from utils.math_utils import (
    calculate_hash_distance,
    calculate_weighted_confidence,
)

# Import comprehensive typing system
# from type_defs import (  # F811: duplicate import
    MatrixControllerType, StateVector, HashSignature, ConfidenceScore,
    FourBitController, EightBitController, SixteenBitController, FortyTwoBitController,
    BitLevel, MatrixPhase, IdentityState, IdentityTrace, GhostLogicState,
    FallbackSystem, AIFeedback, AIConsensus, CrossBasketTrigger,
    create_matrix_controller, hash_state, save_identity_trace
)

logger = logging.getLogger(__name__)


class RiddleGEMMEngine:
    """
    Manages and scores trading strategies against the current market state.
    Enhanced with matrix controllers and identity tracking.
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

        # ✨ NEW: Matrix Controllers for different bit levels
        self.matrix_controllers: Dict[BitLevel, MatrixControllerType] = {}
        self._initialize_matrix_controllers()

        # ✨ NEW: Identity tracking system
        self.identity_trace = IdentityTrace()
        self.current_identity_state: Optional[IdentityState] = None

        # ✨ NEW: Ghost logic and fallback systems
        self.ghost_state = GhostLogicState()
        self.fallback_systems: Dict[str, FallbackSystem] = {}

        # ✨ NEW: AI consensus system
        self.ai_consensus = AIConsensus()

        # ✨ NEW: Cross-basket triggers
        self.cross_basket_triggers: List[CrossBasketTrigger] = []

        logger.info(f"RiddleGEMMEngine initialized with vector size {vector_size}.")

    def _initialize_matrix_controllers(self) -> None:
        """Initialize matrix controllers for all bit levels."""
        try:
            # Initialize 4-bit controller for basic operations
            self.matrix_controllers[BitLevel.FOUR_BIT] = create_matrix_controller(
                BitLevel.FOUR_BIT, MatrixPhase.INITIALIZATION
            )

            # Initialize 8-bit controller for intermediate operations
            self.matrix_controllers[BitLevel.EIGHT_BIT] = create_matrix_controller(
                BitLevel.EIGHT_BIT, MatrixPhase.ACCUMULATION
            )

            # Initialize 16-bit controller for advanced operations
            self.matrix_controllers[BitLevel.SIXTEEN_BIT] = create_matrix_controller(
                BitLevel.SIXTEEN_BIT, MatrixPhase.RESONANCE
            )

            # Initialize 42-bit controller for quantum operations
            self.matrix_controllers[BitLevel.FORTY_TWO_BIT] = create_matrix_controller(
                BitLevel.FORTY_TWO_BIT, MatrixPhase.FORTY_TWO_PHASE
            )

            logger.info(f"Initialized {len(self.matrix_controllers)} matrix controllers.")

        except Exception as e:
            logger.error(f"Failed to initialize matrix controllers: {e}")
            # Fallback: create basic controllers without advanced features
            self._create_fallback_controllers()

    def _create_fallback_controllers(self) -> None:
        """Create fallback matrix controllers if initialization fails."""
        logger.warning("Creating fallback matrix controllers...")

        # Simple fallback controllers
        for bit_level in [BitLevel.FOUR_BIT, BitLevel.EIGHT_BIT, BitLevel.SIXTEEN_BIT]:
            try:
                self.matrix_controllers[bit_level] = create_matrix_controller(
                    bit_level, MatrixPhase.INITIALIZATION
                )
            except Exception as e:
                logger.error(f"Failed to create fallback controller for {bit_level}: {e}")

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
        transformed_state = unified_math.unified_math.dot_product(weight_matrix, state_vector_np)

        # ✨ NEW: Update matrix controllers with transformed state
        self._update_matrix_controllers(transformed_state)

        scores = {}
        for name, strategy_vec in self.strategy_vectors.items():
            # Calculate confidence score using the weighted sigmoid utility
            confidence = calculate_weighted_confidence(strategy_vec, transformed_state)
            scores[name] = confidence

        return scores

    def _update_matrix_controllers(self, transformed_state: np.ndarray) -> None:
        """Update all matrix controllers with the transformed state."""
        try:
            # Update 4-bit controller (first 4 elements)
            if BitLevel.FOUR_BIT in self.matrix_controllers:
                four_bit_state = transformed_state[:4] if transformed_state.size >= 4 else np.zeros(4)
                self.matrix_controllers[BitLevel.FOUR_BIT].update_state(four_bit_state)

            # Update 8-bit controller (first 8 elements)
            if BitLevel.EIGHT_BIT in self.matrix_controllers:
                eight_bit_state = transformed_state[:8] if transformed_state.size >= 8 else np.zeros(8)
                self.matrix_controllers[BitLevel.EIGHT_BIT].update_state(eight_bit_state)

            # Update 16-bit controller (first 16 elements)
            if BitLevel.SIXTEEN_BIT in self.matrix_controllers:
                sixteen_bit_state = transformed_state[:16] if transformed_state.size >= 16 else np.zeros(16)
                self.matrix_controllers[BitLevel.SIXTEEN_BIT].update_state(sixteen_bit_state)

            # Update 42-bit controller (pad to 42 elements)
            if BitLevel.FORTY_TWO_BIT in self.matrix_controllers:
                forty_two_state = np.zeros(42)
                forty_two_state[:unified_math.min(transformed_state.size, 42)] = transformed_state[:42]
                self.matrix_controllers[BitLevel.FORTY_TWO_BIT].update_state(forty_two_state)

        except Exception as e:
            logger.error(f"Failed to update matrix controllers: {e}")

    def find_best_strategy(
        self, current_state_vector: List[float], matrix_name: str = "default"
    ) -> Tuple[Optional[str], float]:
        """
        Find the single best strategy for the given market state.

        Returns:
            A tuple containing the name of the best strategy and its score,
            or (None, 0.0) if no strategies are registered.
        """
        # ✨ NEW: Update identity tracking
        self._update_identity_state(current_state_vector)

        scores = self.score_strategies(current_state_vector, matrix_name)
        if not scores:
            return None, 0.0

        best_strategy = unified_math.max(scores, key=scores.get)
        best_score = scores[best_strategy]

        # ✨ NEW: Check for fallback triggers
        if self.ghost_state.should_trigger_fallback(best_score):
            logger.warning(f"Fallback triggered for strategy '{best_strategy}' with score {best_score:.4f}")
            return self._execute_fallback_strategy(current_state_vector)

        logger.info(
            f"Best strategy found: '{best_strategy}' with score {best_score:.4f}"
        )

        # ✨ NEW: Check cross-basket triggers
        self._check_cross_basket_triggers(best_strategy, best_score)

        return best_strategy, best_score

    def _update_identity_state(self, current_state_vector: List[float]) -> None:
        """Update identity tracking state."""
        try:
            strategy_state = {
                "vector_size": len(current_state_vector),
                "vector_hash": hash(tuple(current_state_vector)),
                "active_strategies": list(self.strategy_vectors.keys()),
                "matrix_controllers": {level.value: controller.phase.value
                                     for level, controller in self.matrix_controllers.items()}
            }

            # Create identity state
            self.current_identity_state = IdentityState(
                tick=int(time.time() * 1000),  # Use timestamp as tick
                strategy_state=strategy_state,
                ai_feedback=self.ai_consensus.final_recommendation if self.ai_consensus.final_recommendation else None
            )

            # Add to trace
            self.identity_trace.add_state(self.current_identity_state)

            # Save trace to log
            save_identity_trace(self.identity_trace, "riddle_gemm_identity")

        except Exception as e:
            logger.error(f"Failed to update identity state: {e}")

    def _execute_fallback_strategy(self, current_state_vector: List[float]) -> Tuple[Optional[str], float]:
        """Execute fallback strategy when confidence is low."""
        try:
            # Use the most conservative strategy as fallback
            if self.strategy_vectors:
                fallback_strategy = unified_math.min(self.strategy_vectors.keys())
                fallback_score = 0.3  # Conservative fallback score
                logger.info(f"Executing fallback strategy: '{fallback_strategy}'")
                return fallback_strategy, fallback_score
        except Exception as e:
            logger.error(f"Fallback strategy execution failed: {e}")

        return None, 0.0

    def _check_cross_basket_triggers(self, strategy_name: str, confidence: float) -> None:
        """Check and activate cross-basket triggers."""
        try:
            for trigger in self.cross_basket_triggers:
                if trigger.should_activate(MatrixPhase.RESONANCE, confidence):
                    logger.info(f"Cross-basket trigger activated: {trigger.source_basket} -> {trigger.target_basket}")
                    trigger.is_active = True
        except Exception as e:
            logger.error(f"Cross-basket trigger check failed: {e}")

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

    # ✨ NEW: Matrix controller access methods
    def get_matrix_controller(self, bit_level: BitLevel) -> Optional[MatrixControllerType]:
        """Get matrix controller for specific bit level."""
        return self.matrix_controllers.get(bit_level)

    def get_matrix_controller_state(self, bit_level: BitLevel) -> Optional[np.ndarray]:
        """Get current state of matrix controller."""
        controller = self.get_matrix_controller(bit_level)
        if controller:
            return controller.state_vector
        return None

    def add_cross_basket_trigger(self, trigger: CrossBasketTrigger) -> None:
        """Add cross-basket trigger."""
        self.cross_basket_triggers.append(trigger)
        logger.info(f"Added cross-basket trigger: {trigger.source_basket} -> {trigger.target_basket}")

    def add_ai_feedback(self, feedback: AIFeedback) -> None:
        """Add AI feedback to consensus system."""
        self.ai_consensus.add_feedback(feedback)
        logger.info(f"Added AI feedback from {feedback.model_name} with confidence {feedback.confidence_score:.4f}")

    def get_identity_trace_hash(self) -> str:
        """Get the current identity trace hash."""
        return self.identity_trace.get_current_hash()


# Create alias for backward compatibility
RiddleGEMM = RiddleGEMMEngine
