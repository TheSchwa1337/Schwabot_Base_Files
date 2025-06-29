# -*- coding: utf-8 -*-
"""
Symbolic Collapse System - Tracking Symbol-to-Strategy Entropy Collapse
========================================================================

This module implements the Symbolic Collapse State (Ψ_c(t)), a critical mathematical
component that tracks the entropy collapse from potential glyph states into real trade outcomes.
It is modeled similarly to quantum decoherence, indicating how predictive a symbol is for a trade outcome.

Mathematical Definition:
Ψ_c(t) = Σ g∈G p(g) ⋅ (1 − Θ_b(t∣g))
Where:
- g∈G is a glyph from the set of all relevant glyphs.
- p(g) is the observed frequency (probability) of glyph g.
- Θ_b(t∣g) is the ASIC Vector Fidelity State conditioned on the presence of glyph g.

If Ψ_c(t) → 0, it indicates that the glyph state perfectly aligns with the trade outcome.
If Ψ_c(t) is high, it indicates that a symbol (or emoji) is no longer predictive,
and the system can flag it for Zygote decay.

This system helps in:
- Evaluating the predictive power of linguistic glyphs.
- Identifying and decaying non-contributing or misleading symbols.
- Guiding adaptive learning for glyph-to-strategy mapping.
"""

import numpy as np
from typing import Dict, List
import logging

from core.glyph.glyph_entropy_system import GlyphEntropySystem
from core.strategy_vector_fidelity import ASICVectorFidelitySystem

logger = logging.getLogger(__name__)


class SymbolicCollapseSystem:
    """
    Manages and calculates the Symbolic Collapse State (Ψ_c(t)).
    """

    def __init__(self, glyph_entropy_system: GlyphEntropySystem, asic_fidelity_system: ASICVectorFidelitySystem):
        self.glyph_entropy_system = glyph_entropy_system
        self.asic_fidelity_system = asic_fidelity_system
        logger.info("Symbolic Collapse System initialized.")

    def calculate_symbolic_collapse(self,
                                    current_bit_vector: List[float],
                                    current_profit_delta_vector: List[float],
                                    glyph_list: List[str]) -> float:
        """
        Calculates the Symbolic Collapse State (Ψ_c(t)).
        
        Args:
            current_bit_vector: The current multi-bit strategy vector B(t).
            current_profit_delta_vector: The current observed profit delta vector Δ(t).
            glyph_list: A list of recent glyphs to consider for their probabilities.
            
        Returns:
            The calculated symbolic collapse (Ψ_c(t)).
        """
        if not glyph_list:
            logger.warning("No glyphs provided for symbolic collapse calculation. Returning 0.0.")
            return 0.0

        total_collapse = 0.0
        
        # Temporarily add glyphs to the entropy system to get their frequencies
        original_glyph_history = list(self.glyph_entropy_system.glyph_history)
        self.glyph_entropy_system.add_glyph_occurrence_batch(glyph_list) # Assuming a batch add method exists or will be added

        glyph_frequencies = self.glyph_entropy_system.get_glyph_frequency()

        for glyph, p_g in glyph_frequencies.items():
            # For simplicity, we are assuming Θ_b(t|g) is the overall Θ_b(t) for now.
            # In a more complex implementation, this would involve a model that predicts
            # fidelity based on the presence of specific glyphs.
            theta_b_tg = self.asic_fidelity_system.calculate_fidelity(current_bit_vector, current_profit_delta_vector)
            
            total_collapse += p_g * (1 - theta_b_tg)
        
        # Restore original glyph history
        self.glyph_entropy_system.glyph_history = original_glyph_history

        logger.debug(f"Calculated Symbolic Collapse State (Ψ_c): {total_collapse:.4f}")
        return float(total_collapse)


# Extend GlyphEntropySystem to include batch addition
def add_glyph_occurrence_batch(self, glyphs: List[str]):
    for glyph in glyphs:
        self.add_glyph_occurrence(glyph)

GlyphEntropySystem.add_glyph_occurrence_batch = add_glyph_occurrence_batch

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    # Initialize dependent systems
    glyph_entropy_sys = GlyphEntropySystem()
    asic_fidelity_sys = ASICVectorFidelitySystem()
    
    collapse_system = SymbolicCollapseSystem(glyph_entropy_sys, asic_fidelity_sys)

    # Example 1: High Fidelity, low collapse
    bit_vec1 = [1.0, 0.5, -0.2]
    delta_vec1 = [0.9, 0.4, -0.1]
    glyphs1 = ["🚀", "📈", "💰", "🚀"]
    collapse1 = collapse_system.calculate_symbolic_collapse(bit_vec1, delta_vec1, glyphs1)
    print(f"Collapse 1 (Low): {collapse1:.4f}")

    # Example 2: Low Fidelity, high collapse
    bit_vec2 = [1.0, 0.5, -0.2]
    delta_vec2 = [-0.9, -0.4, 0.1]
    glyphs2 = ["🚀", "📈", "💰", "😡"]
    collapse2 = collapse_system.calculate_symbolic_collapse(bit_vec2, delta_vec2, glyphs2)
    print(f"Collapse 2 (High): {collapse2:.4f}")

    # Example 3: No glyphs
    bit_vec3 = [1.0, 1.0]
    delta_vec3 = [1.0, 1.0]
    glyphs3 = []
    collapse3 = collapse_system.calculate_symbolic_collapse(bit_vec3, delta_vec3, glyphs3)
    print(f"Collapse 3 (No glyphs): {collapse3:.4f}") 