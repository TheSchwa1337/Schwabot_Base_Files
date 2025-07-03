"""Legacy Mathematical Connectivity: Historical Math Integration.

Implements valuable mathematical connectivity patterns and formulas from our
backup systems to enhance the current mathematical framework with proven
legacy algorithms and connectivity logic.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LegacyMathVector:
    """Legacy mathematical vector with connectivity properties."""

    components: np.ndarray
    magnitude: float
    phase: float
    connectivity_index: float
    mathematical_depth: int
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self: LegacyMathVector) -> None:
        """Calculate derived properties."""
        if self.magnitude == 0:
            self.magnitude = np.linalg.norm(self.components)
        if self.phase == 0:
            self.phase = math.atan2(
                self.components[1] if len(self.components) > 1 else 0,
                self.components[0] if len(self.components) > 0 else 1,
            )


@dataclass
class ConnectivityMatrix:
    """Mathematical connectivity matrix for legacy integration."""

    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    determinant: float
    trace: float
    rank: int
    condition_number: float

    @classmethod
    def from_vectors(
        cls: type[ConnectivityMatrix], vectors: List[LegacyMathVector]
    ) -> ConnectivityMatrix:
        """Create connectivity matrix from legacy vectors."""
        if not vectors:
            raise ValueError("Cannot create matrix from empty vectors")

        # Create matrix from vector components
        matrix_data = []
        for vector in vectors:
            if len(vector.components) > 0:
                matrix_data.append(vector.components[:4])  # Limit to 4x4 for stability

        # Pad to 4x4 matrix
        while len(matrix_data) < 4:
            matrix_data.append(np.zeros(4))

        matrix = np.array(matrix_data[:4])

        # Calculate mathematical properties
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            determinant = np.linalg.det(matrix)
            trace = np.trace(matrix)
            rank = np.linalg.matrix_rank(matrix)
            condition_number = np.linalg.cond(matrix)
        except np.linalg.LinAlgError:
            # Handle singular matrices
            eigenvalues = np.zeros(4)
            eigenvectors = np.eye(4)
            determinant = 0.0
            trace = np.trace(matrix)
            rank = 0
            condition_number = float("inf")

        return cls(
            matrix=matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            determinant=determinant,
            trace=trace,
            rank=rank,
            condition_number=condition_number,
        )


class LegacyMathematicalConnectivity:
    """Legacy mathematical connectivity integration system.

    Implements proven mathematical algorithms and connectivity patterns
    from our backup systems to enhance current mathematical capabilities.
    """

    def __init__(self: LegacyMathematicalConnectivity) -> None:
        """Initialize legacy mathematical connectivity system."""
        # Legacy mathematical constants from backup systems
        self.phi_golden = 1.618033988749895  # Golden ratio
        self.euler_constant = 2.718281828459045
        self.pi_constant = 3.141592653589793
        self.sqrt_2 = 1.4142135623730951
        self.sqrt_3 = 1.7320508075688772

        # Legacy connectivity parameters
        self.connectivity_threshold = 0.618  # Golden ratio threshold
        self.mathematical_depth_limit = 42
        self.resonance_frequency = 2.0
        self.decay_constant = 0.05

        # Historical mathematical vectors
        self.legacy_vectors: List[LegacyMathVector] = []
        self.connectivity_matrices: List[ConnectivityMatrix] = []

        # Performance tracking
        self.total_calculations = 0
        self.successful_connections = 0
        self.mathematical_stability_index = 1.0

    def create_legacy_vector(
        self: LegacyMathematicalConnectivity,
        input_value: float,
        mathematical_context: Dict[str, float],
        depth: int = 16,
    ) -> LegacyMathVector:
        """Create legacy mathematical vector with connectivity properties."""
        # Legacy mathematical transformations from backup systems
        phi_component = input_value * self.phi_golden
        euler_component = input_value * self.euler_constant
        pi_component = input_value * self.pi_constant

        # Fractal scaling based on mathematical depth
        depth_scalar = math.pow(self.phi_golden, depth / 16.0)

        # Create vector components using legacy algorithms
        components = np.array(
            [
                phi_component * depth_scalar,
                euler_component * math.sin(input_value * self.pi_constant),
                pi_component * math.cos(input_value * self.euler_constant),
                input_value * self.sqrt_2 * depth_scalar,
            ]
        )

        # Calculate connectivity index using legacy formula
        connectivity_index = self._calculate_legacy_connectivity_index(
            components, mathematical_context
        )

        # Calculate phase using legacy phase calculation
        phase = self._calculate_legacy_phase(components, input_value)

        vector = LegacyMathVector(
            components=components,
            magnitude=0,  # Will be calculated in __post_init__
            phase=phase,
            connectivity_index=connectivity_index,
            mathematical_depth=depth,
        )

        self.legacy_vectors.append(vector)

        # Keep vector history manageable
        if len(self.legacy_vectors) > 1000:
            self.legacy_vectors = self.legacy_vectors[-500:]

        return vector

    def _calculate_legacy_connectivity_index(
        self: LegacyMathematicalConnectivity,
        components: np.ndarray,
        context: Dict[str, float],
    ) -> float:
        """Calculate connectivity index using legacy mathematical formulas."""
        # Legacy connectivity formula from backup systems
        base_connectivity = np.sum(np.abs(components)) / len(components)

        # Context enhancement using golden ratio
        context_enhancement = 0.0
        for _key, value in context.items():
            if isinstance(value, (int, float)):
                context_enhancement += abs(value) * self.phi_golden

        # Normalize context enhancement
        context_factor = 1.0 + (context_enhancement / (1.0 + context_enhancement))

        # Apply legacy connectivity transformation
        connectivity_index = (
            base_connectivity * context_factor * self.phi_golden
        ) % 1.0

        return connectivity_index

    def _calculate_legacy_phase(
        self: LegacyMathematicalConnectivity,
        components: np.ndarray,
        input_value: float,
    ) -> float:
        """Calculate phase using legacy phase calculation algorithm."""
        # Legacy phase calculation from backup systems
        # Uses Euler's formula with golden ratio scaling
        real_part = np.sum(components[::2])  # Even components
        imaginary_part = np.sum(components[1::2])  # Odd components

        # Apply golden ratio phase shift
        phase_shift = input_value * self.phi_golden

        # Calculate complex phase
        complex_number = complex(
            real_part * math.cos(phase_shift), imaginary_part * math.sin(phase_shift)
        )

        return math.atan2(complex_number.imag, complex_number.real)

    def generate_connectivity_matrix(
        self: LegacyMathematicalConnectivity,
        recent_vectors: Optional[List[LegacyMathVector]] = None,
    ) -> ConnectivityMatrix:
        """Generate connectivity matrix from recent legacy vectors."""
        if recent_vectors is None:
            recent_vectors = (
                self.legacy_vectors[-4:]
                if len(self.legacy_vectors) >= 4
                else self.legacy_vectors
            )

        if not recent_vectors:
            # Create identity matrix as fallback
            identity_vector = LegacyMathVector(
                components=np.array([1.0, 0.0, 0.0, 0.0]),
                magnitude=1.0,
                phase=0.0,
                connectivity_index=self.phi_golden % 1.0,
                mathematical_depth=16,
            )
            recent_vectors = [identity_vector] * 4

        connectivity_matrix = ConnectivityMatrix.from_vectors(recent_vectors)
        self.connectivity_matrices.append(connectivity_matrix)

        # Keep matrix history manageable
        if len(self.connectivity_matrices) > 100:
            self.connectivity_matrices = self.connectivity_matrices[-50:]

        return connectivity_matrix

    def calculate_mathematical_resonance(
        self: LegacyMathematicalConnectivity,
        vector1: LegacyMathVector,
        vector2: LegacyMathVector,
    ) -> float:
        """Calculate mathematical resonance between two legacy vectors."""
        # Legacy resonance formula
        dot_product = np.dot(vector1.components, vector2.components)
        magnitude_product = vector1.magnitude * vector2.magnitude
        if magnitude_product == 0:
            return 0.0

        phase_difference = abs(vector1.phase - vector2.phase)
        phase_factor = math.cos(phase_difference)

        resonance = (dot_product / magnitude_product) * phase_factor
        return resonance

    def apply_legacy_transformation(
        self: LegacyMathematicalConnectivity,
        input_matrix: np.ndarray,
        transformation_type: str = "fibonacci",
    ) -> np.ndarray:
        """Apply legacy mathematical transformation to a matrix."""
        if transformation_type == "fibonacci":
            return self._apply_fibonacci_transformation(input_matrix)
        elif transformation_type == "golden_spiral":
            return self._apply_golden_spiral_transformation(input_matrix)
        elif transformation_type == "euler_rotation":
            return self._apply_euler_rotation_transformation(input_matrix)
        else:
            return input_matrix

    def _apply_fibonacci_transformation(
        self: LegacyMathematicalConnectivity, matrix: np.ndarray
    ) -> np.ndarray:
        """Apply Fibonacci sequence-based transformation."""
        fib_matrix = np.array([[1, 1], [1, 0]])
        transformed_matrix = matrix.dot(np.linalg.matrix_power(fib_matrix, 4))
        return transformed_matrix

    def _apply_golden_spiral_transformation(
        self: LegacyMathematicalConnectivity, matrix: np.ndarray
    ) -> np.ndarray:
        """Apply golden spiral transformation to a matrix."""
        angle = self.pi_constant / 2.0  # 90-degree turn for spiral
        rotation_matrix = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        scaling_factor = self.phi_golden
        # This is a simplified example; a true spiral is more complex.
        if matrix.shape == (2, 2):
            return matrix.dot(rotation_matrix) * scaling_factor
        return matrix

    def _apply_euler_rotation_transformation(
        self: LegacyMathematicalConnectivity, matrix: np.ndarray
    ) -> np.ndarray:
        """Apply Euler rotation-based transformation."""
        # Example rotation around Z-axis
        angle = self.pi_constant / 4.0  # 45 degrees
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        if matrix.shape[0] == 3:
            return matrix.dot(rotation_matrix)
        return matrix

    def calculate_stability_index(self: LegacyMathematicalConnectivity) -> float:
        """Calculate overall mathematical stability index."""
        if not self.connectivity_matrices:
            return 1.0

        # Weighted average of condition numbers
        condition_numbers = [
            m.condition_number
            for m in self.connectivity_matrices
            if m.condition_number is not None and np.isfinite(m.condition_number)
        ]
        if not condition_numbers:
            return 0.0

        avg_cond = np.mean(condition_numbers)
        # Stability is inversely related to condition number
        stability = 1 / (1 + math.log1p(avg_cond))

        # Update internal stability index with decay
        self.mathematical_stability_index = (
            self.mathematical_stability_index * (1 - self.decay_constant)
            + stability * self.decay_constant
        )
        return self.mathematical_stability_index

    def optimize_connectivity(
        self: LegacyMathematicalConnectivity, target_connectivity: float = 0.618
    ) -> Dict[str, Any]:
        """Iteratively adjust vector parameters to meet a target connectivity."""
        if not self.legacy_vectors:
            return {"status": "no_vectors", "iterations": 0}

        latest_vector = self.legacy_vectors[-1]
        current_connectivity = latest_vector.connectivity_index
        iterations = 0
        max_iterations = 100

        while (
            abs(current_connectivity - target_connectivity) > 0.01
            and iterations < max_iterations
        ):
            # Simple gradient descent-like adjustment
            error = target_connectivity - current_connectivity
            adjustment_factor = error * 0.1  # Learning rate

            # Adjust components and re-calculate
            new_components = latest_vector.components * (1 + adjustment_factor)
            new_connectivity = self._calculate_legacy_connectivity_index(
                new_components, {"adjustment": adjustment_factor}
            )

            latest_vector.components = new_components
            latest_vector.connectivity_index = new_connectivity
            current_connectivity = new_connectivity
            iterations += 1

        return {
            "status": "converged" if iterations < max_iterations else "max_iterations",
            "iterations": iterations,
            "final_connectivity": current_connectivity,
        }

    def get_legacy_analytics(self: LegacyMathematicalConnectivity) -> Dict[str, Any]:
        """Get analytics of the legacy mathematical connectivity system."""
        total_vectors = len(self.legacy_vectors)
        total_matrices = len(self.connectivity_matrices)

        avg_magnitude = (
            np.mean([v.magnitude for v in self.legacy_vectors])
            if total_vectors > 0
            else 0.0
        )
        avg_connectivity = (
            np.mean([v.connectivity_index for v in self.legacy_vectors])
            if total_vectors > 0
            else 0.0
        )
        avg_determinant = (
            np.mean([m.determinant for m in self.connectivity_matrices])
            if total_matrices > 0
            else 0.0
        )

        return {
            "total_legacy_vectors": total_vectors,
            "total_connectivity_matrices": total_matrices,
            "average_vector_magnitude": avg_magnitude,
            "average_connectivity_index": avg_connectivity,
            "average_matrix_determinant": avg_determinant,
            "mathematical_stability_index": self.calculate_stability_index(),
            "total_calculations": self.total_calculations,
            "successful_connections": self.successful_connections,
        }


def demo_legacy_mathematical_connectivity() -> Dict[str, Any]:
    """Demonstrate legacy mathematical connectivity functionality."""
    safe_print("--- Demonstrating Legacy Mathematical Connectivity ---")
    legacy_system = LegacyMathematicalConnectivity()

    # Create some legacy vectors
    for i in range(10):
        legacy_system.create_legacy_vector(
            input_value=i * 0.5,
            mathematical_context={"iteration": i, "mode": 1},
            depth=16 + i,
        )

    # Generate a connectivity matrix
    connectivity_matrix = legacy_system.generate_connectivity_matrix()
    safe_print("Generated Connectivity Matrix:")
    safe_print(str(connectivity_matrix.matrix))

    # Calculate resonance
    if len(legacy_system.legacy_vectors) >= 2:
        resonance = legacy_system.calculate_mathematical_resonance(
            legacy_system.legacy_vectors[0], legacy_system.legacy_vectors[1]
        )
        safe_print(f"Resonance between first two vectors: {resonance:.4f}")

    # Apply a transformation
    transformed_matrix = legacy_system.apply_legacy_transformation(
        connectivity_matrix.matrix, transformation_type="golden_spiral"
    )
    safe_print("Matrix after Golden Spiral Transformation:")
    safe_print(str(transformed_matrix))

    # Optimize connectivity
    optimization_results = legacy_system.optimize_connectivity()
    safe_print("Connectivity Optimization Results:")
    safe_print(str(optimization_results))

    # Get analytics
    analytics = legacy_system.get_legacy_analytics()
    safe_print("Legacy System Analytics:")
    safe_print(str(analytics))

    safe_print("--- Legacy Math Demo Complete ---")
    return analytics


if __name__ == "__main__":
    demo_legacy_mathematical_connectivity()
