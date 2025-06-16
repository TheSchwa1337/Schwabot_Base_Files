"""
Anti-Pole Theory Implementation for Schwabot
===========================================

This module implements the mathematical foundations for Anti-Pole Theory
as applied to cryptocurrency trading and profit navigation.

Core Concepts:
- Anti-Pole Vectors: Counter-pole mathematical constructs for stability detection
- Inverse Drift-Shell Gradients: Cool-state detection for trade optimization
- ICAP (Inverse Cluster Activation Probability): Entry/exit signal generation
- ZBE Thermal Cooldown: System protection against overheating

Mathematical Foundation:
Δ̄Ψᵢ = ∇ₜ[1/(Hₙ+ε)] ⊗ (1-Λᵢ(t))
P̄(χ) = e^(-Δ̄Ψᵢ) · (1-Fₖ(t))
φₛ(t) = 1 ⟺ [Δ̄Ψᵢ > μc + σc] ∧ [P̄(χ) ≥ τᵢcₐₚ]
"""

__version__ = "4.0.0"
__author__ = "Schwabot Engineering Team"

from .vector import AntiPoleVector, AntiPoleConfig
from .zbe_controller import ZBEThermalCooldown, ThermalState
from .tesseract_bridge import TesseractVisualizer, GlyphPacket

__all__ = [
    'AntiPoleVector',
    'AntiPoleConfig', 
    'ZBEThermalCooldown',
    'ThermalState',
    'TesseractVisualizer',
    'GlyphPacket'
] 