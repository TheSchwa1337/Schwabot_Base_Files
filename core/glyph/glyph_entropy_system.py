# -*- coding: utf-8 -*-
"""
Glyph Entropy System - Quantifying Symbolic Information in Schwabot's Hashspace
=================================================================================

This module implements the Glyph Entropy State (Γ_g(t)), a critical mathematical
component for understanding and managing the symbolic information flow within
Schwabot's fractal-glyph-ASIC system. It measures how much novel symbolic
information persists in Schwabot's internal hashspace per glyph-state.

Mathematical Definition:
Γ_g(t) = - Σ p(gᵢ) ⋅ log₂(p(gᵢ))
Where:
- gᵢ is a glyph's presence in runtime signal stack
- p(gᵢ) is the observed frequency of glyph gᵢ

High Γ_g(t) indicates a diverse and novel symbolic state.
Low Γ_g(t) indicates repeated glyph patterns, signaling fractal redundancy.

This system helps in:
- Signal cleanliness and overfitting prevention.
- Identifying symbolic decay and guiding Zygote decay processes.
"""

import math
import time
from collections import Counter
from typing import Dict, List

import numpy as np


class GlyphEntropySystem:
    """
    Manages and calculates the Glyph Entropy State (Γ_g(t)).
    """

    def __init__(self):
        self.glyph_history: List[str] = []
        self.history_maxlen: int = 1000  # Max glyphs to keep in history

    def add_glyph_occurrence(self, glyph: str):
        """
        Adds a glyph occurrence to the history for entropy calculation.
        
        Args:
            glyph: The glyph string (e.g., emoji).
        """
        self.glyph_history.append(glyph)
        if len(self.glyph_history) > self.history_maxlen:
            self.glyph_history.pop(0)

    def calculate_glyph_entropy(self) -> float:
        """
        Calculates the Glyph Entropy State (Γ_g(t)).
        
        Returns:
            The calculated glyph entropy (Γ_g(t)).
        """
        if not self.glyph_history:
            return 0.0

        # Count frequencies of each glyph
        glyph_counts = Counter(self.glyph_history)
        total_glyphs = len(self.glyph_history)

        entropy = 0.0
        for count in glyph_counts.values():
            probability = count / total_glyphs
            if probability > 0:
                entropy -= probability * math.log2(probability)
        return entropy

    def get_glyph_frequency(self) -> Dict[str, float]:
        """
        Returns the observed frequency of each glyph in the history.
        """
        if not self.glyph_history:
            return {}
        glyph_counts = Counter(self.glyph_history)
        total_glyphs = len(self.glyph_history)
        return {glyph: count / total_glyphs for glyph, count in glyph_counts.items()}

    def reset_history(self):
        """
        Resets the glyph history.
        """
        self.glyph_history = []


# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    entropy_system = GlyphEntropySystem()
    print("Initial Entropy:", entropy_system.calculate_glyph_entropy())

    # Add some glyphs
    entropy_system.add_glyph_occurrence("🚀")
    entropy_system.add_glyph_occurrence("📈")
    entropy_system.add_glyph_occurrence("💰")
    print("Entropy after 3 unique glyphs:", entropy_system.calculate_glyph_entropy())

    entropy_system.add_glyph_occurrence("🚀")
    entropy_system.add_glyph_occurrence("📈")
    entropy_system.add_glyph_occurrence("🚀")
    print("Entropy after more glyphs (some repeated):", entropy_system.calculate_glyph_entropy())

    print("Glyph Frequencies:", entropy_system.get_glyph_frequency())

    entropy_system.reset_history()
    print("Entropy after reset:", entropy_system.calculate_glyph_entropy())

    for _ in range(50):
        entropy_system.add_glyph_occurrence("🔥")
    for _ in range(25):
        entropy_system.add_glyph_occurrence("🌊")
    for _ in range(10):
        entropy_system.add_glyph_occurrence("🌀")
    print("Entropy with more data:", entropy_system.calculate_glyph_entropy()) 