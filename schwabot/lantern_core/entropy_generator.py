"""
Entropy Generator: Fractal Entropy Block Creation
================================================

Transforms SHA-256 price hashes into fractal entropy blocks through
recursive mathematical operations. This bridges the gap between
raw market data and semantic interpretation by creating structured
entropy that can be read as language patterns.

Hash → Entropy Block → Semantic Meaning
"""

from __future__ import annotations

import cmath
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FractalBlock:
    """A fractal entropy block generated from hash input"""

    source_hash: str
    fractal_dimensions: List[complex]
    entropy_layers: List[List[float]]
    recursion_depth: int
    convergence_patterns: List[float]
    harmonic_frequencies: List[float]
    phase_relationships: List[float]
    entropy_score: float
    stability_index: float
    temporal_signature: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "source_hash": self.source_hash,
            "fractal_dimensions": [
                {"real": z.real, "imag": z.imag} for z in self.fractal_dimensions
            ],
            "entropy_layers": self.entropy_layers,
            "recursion_depth": self.recursion_depth,
            "convergence_patterns": self.convergence_patterns,
            "harmonic_frequencies": self.harmonic_frequencies,
            "phase_relationships": self.phase_relationships,
            "entropy_score": self.entropy_score,
            "stability_index": self.stability_index,
            "temporal_signature": self.temporal_signature,
            "created_at": self.created_at,
        }


class EntropyGenerator:
    """
    Generates fractal entropy blocks from SHA-256 hashes

    Uses recursive mathematical transformations to convert hash strings
    into structured entropy patterns that can be semantically interpreted.
    """

    def __init__(self, max_recursion_depth: int = 50, layer_count: int = 7):
        self.max_recursion_depth = max_recursion_depth
        self.layer_count = layer_count

        # Fractal constants inspired by Ω-B-Γ Logic
        self.omega_constant = 0.006  # Entropy modulation
        self.beta_constant = 0.9944  # Stability coefficient
        self.gamma_constant = 0.91  # Quantum coherence threshold

        # Performance tracking
        self.blocks_generated = 0
        self.average_generation_time = 0.0

    def _hash_to_complex_seed(self, hash_value: str) -> complex:
        """Convert hash string to complex number seed"""
        # Use first 16 chars for real, next 16 for imaginary
        real_hex = hash_value[:16]
        imag_hex = hash_value[16:32]

        # Convert to floating point values
        real_part = int(real_hex, 16) / (16**16)  # Normalize to [0,1]
        imag_part = int(imag_hex, 16) / (16**16)

        return complex(real_part, imag_part)

    def _generate_fractal_sequence(self, seed: complex, depth: int) -> List[complex]:
        """Generate fractal sequence using recursive iteration"""
        sequence = [seed]
        z = seed

        for i in range(depth):
            # Apply fractal transformation: z = z^2 + c + entropy_modulation
            entropy_factor = self.omega_constant * (i + 1)
            stability_factor = self.beta_constant ** (i / depth)

            # Modified Mandelbrot-like iteration with entropy injection
            z_next = z**2 + seed + complex(entropy_factor, entropy_factor * 0.618)

            # Apply stability modulation
            z_next *= stability_factor

            # Prevent excessive divergence
            if abs(z_next) > 100:
                z_next = z_next / abs(z_next) * (100 * stability_factor)

            sequence.append(z_next)
            z = z_next

            # Check for convergence
            if i > 5 and abs(z - sequence[-2]) < 1e-10:
                break

        return sequence

    def _create_entropy_layers(self, fractal_sequence: List[complex]) -> List[List[float]]:
        """Create multiple entropy layers from fractal sequence"""
        layers = []

        for layer_idx in range(self.layer_count):
            layer = []

            # Generate layer-specific entropy values
            for i, z in enumerate(fractal_sequence):
                # Extract different aspects of complex number for each layer
                if layer_idx == 0:  # Magnitude layer
                    value = abs(z)
                elif layer_idx == 1:  # Phase layer
                    value = cmath.phase(z)
                elif layer_idx == 2:  # Real component layer
                    value = z.real
                elif layer_idx == 3:  # Imaginary component layer
                    value = z.imag
                elif layer_idx == 4:  # Harmonic layer
                    value = np.sin(abs(z) * np.pi)
                elif layer_idx == 5:  # Resonance layer
                    value = np.cos(cmath.phase(z) * 2)
                else:  # Combined layer
                    value = abs(z) * np.sin(cmath.phase(z))

                # Apply gamma modulation for quantum coherence
                value *= self.gamma_constant
                layer.append(value)

            layers.append(layer)

        return layers

    def _calculate_harmonic_frequencies(self, entropy_layers: List[List[float]]) -> List[float]:
        """Extract harmonic frequencies from entropy patterns"""
        frequencies = []

        # Analyze each layer for harmonic content
        for layer in entropy_layers:
            if len(layer) > 1:
                # Calculate dominant frequency using FFT-like analysis
                layer_array = np.array(layer)

                # Find peaks and valleys to identify frequency patterns
                diffs = np.diff(layer_array)
                sign_changes = np.where(np.diff(np.sign(diffs)))[0]

                if len(sign_changes) > 0:
                    # Estimate frequency from zero crossings
                    avg_period = (
                        len(layer) / len(sign_changes) if len(sign_changes) > 0 else len(layer)
                    )
                    frequency = 1.0 / avg_period if avg_period > 0 else 0.0
                else:
                    frequency = 0.0

                frequencies.append(frequency)
            else:
                frequencies.append(0.0)

        return frequencies

    def _calculate_phase_relationships(self, entropy_layers: List[List[float]]) -> List[float]:
        """Calculate phase relationships between entropy layers"""
        phases = []

        # Compare each layer with the first layer to find phase relationships
        if len(entropy_layers) > 1:
            reference_layer = entropy_layers[0]

            for layer in entropy_layers[1:]:
                if len(layer) >= len(reference_layer):
                    # Calculate cross-correlation to find phase relationship
                    correlation = np.correlate(
                        layer[: len(reference_layer)], reference_layer, mode="valid"
                    )
                    phase = np.angle(correlation[0]) if len(correlation) > 0 else 0.0
                else:
                    phase = 0.0

                phases.append(phase)

        return phases

    def _calculate_entropy_score(self, entropy_layers: List[List[float]]) -> float:
        """Calculate overall entropy score for the block"""
        total_entropy = 0.0
        total_values = 0

        for layer in entropy_layers:
            # Calculate Shannon-like entropy for each layer
            layer_array = np.array(layer)

            # Normalize to probabilities
            if np.sum(np.abs(layer_array)) > 0:
                probs = np.abs(layer_array) / np.sum(np.abs(layer_array))
                # Calculate entropy: -Σ p*log(p)
                # Add small value to avoid log(0)
                layer_entropy = -np.sum(probs * np.log(probs + 1e-10))
                total_entropy += layer_entropy
                total_values += 1

        return total_entropy / total_values if total_values > 0 else 0.0

    def _calculate_stability_index(
        self, fractal_sequence: List[complex], convergence_patterns: List[float]
    ) -> float:
        """Calculate stability index based on convergence behavior"""
        if len(convergence_patterns) == 0:
            return 0.0

        # Analyze convergence pattern for stability
        convergence_variance = np.var(convergence_patterns)
        final_convergence = convergence_patterns[-1] if convergence_patterns else 1.0

        # Calculate stability as inverse of variance with convergence weighting
        stability = 1.0 / (1.0 + convergence_variance) * (1.0 / (1.0 + final_convergence))

        return min(stability, 1.0)  # Cap at 1.0

    def _generate_temporal_signature(self, fractal_sequence: List[complex]) -> str:
        """Generate temporal signature from fractal sequence"""
        # Create signature based on sequence characteristics
        signature_parts = []

        # Add sequence length
        signature_parts.append(f"L{len(fractal_sequence)}")

        # Add magnitude pattern
        if fractal_sequence:
            avg_magnitude = np.mean([abs(z) for z in fractal_sequence])
            signature_parts.append(f"M{avg_magnitude:.4f}")

            # Add phase pattern
            avg_phase = np.mean([cmath.phase(z) for z in fractal_sequence])
            signature_parts.append(f"P{avg_phase:.4f}")

        # Add timestamp component
        timestamp_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        signature_parts.append(f"T{timestamp_hash}")

        return "_".join(signature_parts)

    def generate_fractal_block(
        self, hash_input: str, price_context: Optional[Dict[str, float]] = None
    ) -> FractalBlock:
        """
        Generate a complete fractal entropy block from hash input

        This is the core transformation: SHA-256 hash → Fractal Entropy Block
        """
        start_time = time.time()

        # Convert hash to complex seed
        seed = self._hash_to_complex_seed(hash_input)

        # Adjust recursion depth based on price context
        depth = self.max_recursion_depth
        if price_context:
            volume = price_context.get("volume", 1.0)
            if volume > 0:
                # Higher volume = deeper recursion
                depth = min(
                    self.max_recursion_depth,
                    int(self.max_recursion_depth * (1 + np.log10(volume) / 10)),
                )

        # Generate fractal sequence
        fractal_sequence = self._generate_fractal_sequence(seed, depth)

        # Create entropy layers
        entropy_layers = self._create_entropy_layers(fractal_sequence)

        # Calculate convergence patterns
        convergence_patterns = []
        for i in range(1, len(fractal_sequence)):
            convergence = abs(fractal_sequence[i] - fractal_sequence[i - 1])
            convergence_patterns.append(convergence)

        # Extract harmonic properties
        harmonic_frequencies = self._calculate_harmonic_frequencies(entropy_layers)
        phase_relationships = self._calculate_phase_relationships(entropy_layers)

        # Calculate metrics
        entropy_score = self._calculate_entropy_score(entropy_layers)
        stability_index = self._calculate_stability_index(fractal_sequence, convergence_patterns)
        temporal_signature = self._generate_temporal_signature(fractal_sequence)

        # Create fractal block
        fractal_block = FractalBlock(
            source_hash=hash_input,
            fractal_dimensions=fractal_sequence,
            entropy_layers=entropy_layers,
            recursion_depth=len(fractal_sequence),
            convergence_patterns=convergence_patterns,
            harmonic_frequencies=harmonic_frequencies,
            phase_relationships=phase_relationships,
            entropy_score=entropy_score,
            stability_index=stability_index,
            temporal_signature=temporal_signature,
        )

        # Update performance metrics
        generation_time = time.time() - start_time
        self.blocks_generated += 1
        self.average_generation_time = (
            self.average_generation_time * (self.blocks_generated - 1) + generation_time
        ) / self.blocks_generated

        return fractal_block

    def analyze_block_patterns(self, blocks: List[FractalBlock]) -> Dict[str, Any]:
        """Analyze patterns across multiple fractal blocks"""
        if not blocks:
            return {"error": "No blocks provided for analysis"}

        # Collect metrics
        entropy_scores = [block.entropy_score for block in blocks]
        stability_indices = [block.stability_index for block in blocks]
        recursion_depths = [block.recursion_depth for block in blocks]

        # Analyze harmonic patterns
        all_frequencies = []
        for block in blocks:
            all_frequencies.extend(block.harmonic_frequencies)

        return {
            "total_blocks_analyzed": len(blocks),
            "average_entropy_score": np.mean(entropy_scores),
            "entropy_variance": np.var(entropy_scores),
            "average_stability_index": np.mean(stability_indices),
            "stability_variance": np.var(stability_indices),
            "average_recursion_depth": np.mean(recursion_depths),
            "harmonic_frequency_range": (
                [min(all_frequencies), max(all_frequencies)] if all_frequencies else [0, 0]
            ),
            # Sample of signatures
            "temporal_signatures": [block.temporal_signature for block in blocks[:10]],
        }

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get entropy generation performance statistics"""
        return {
            "total_blocks_generated": self.blocks_generated,
            "average_generation_time": self.average_generation_time,
            "max_recursion_depth": self.max_recursion_depth,
            "layer_count": self.layer_count,
            "omega_constant": self.omega_constant,
            "beta_constant": self.beta_constant,
            "gamma_constant": self.gamma_constant,
        }
