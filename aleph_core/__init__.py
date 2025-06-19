"""
Aleph Core Module Package
========================
Modular components for the Aleph Unitizer system.
"""

from .unitizer import Unitizer as AlephUnitizer
from .tesseract import TesseractProcessor as TesseractPortal
from .pattern_matcher import PatternMatcher
from .entropy_analyzer import EntropyAnalyzer
from .batch_integration import BatchProcessor as BatchIntegrator
from .paradox_visualizer import ParadoxVisualizer
from .detonation_sequencer import DetonationSequencer
from .smart_money_analyzer import SmartMoneyAnalyzer

__version__ = "1.0.0"
__all__ = [
    "AlephUnitizer", 
    "TesseractPortal", 
    "PatternMatcher", 
    "EntropyAnalyzer", 
    "BatchIntegrator",
    "ParadoxVisualizer",
    "DetonationSequencer",
    "SmartMoneyAnalyzer"
] 