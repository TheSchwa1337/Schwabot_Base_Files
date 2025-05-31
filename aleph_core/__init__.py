"""
Aleph Core Module Package
========================
Modular components for the Aleph Unitizer system.
"""

from .unitizer import AlephUnitizer
from .tesseract import TesseractPortal
from .pattern_matcher import PatternMatcher
from .entropy_analyzer import EntropyAnalyzer
from .batch_integration import BatchIntegrator
from .paradox_visualizer import ParadoxVisualizer
from .detonation_sequencer import DetonationSequencer

__version__ = "1.0.0"
__all__ = [
    "AlephUnitizer", 
    "TesseractPortal", 
    "PatternMatcher", 
    "EntropyAnalyzer", 
    "BatchIntegrator",
    "ParadoxVisualizer",
    "DetonationSequencer"
] 