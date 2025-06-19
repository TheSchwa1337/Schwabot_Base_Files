"""
Core Module for Hash Recollection System
Exports all the key components for pattern recognition and trading.
"""

from .bit_operations import BitOperations, PhaseState
from .entropy_tracker import EntropyTracker, EntropyState
from .hash_recollection import HashRecollectionSystem, HashEntry
from .pattern_utils import PatternUtils, PatternMatch, ENTRY_KEYS, EXIT_KEYS
from .risk_engine import RiskEngine, PositionSignal, RiskMetrics
from .strange_loop_detector import StrangeLoopDetector, EchoPattern

__version__ = "0.045"
__all__ = [
    "HashRecollectionSystem",
    "HashEntry",
    "EntropyTracker",
    "EntropyState",
    "BitOperations",
    "PhaseState",
    "PatternUtils",
    "PatternMatch",
    "ENTRY_KEYS",
    "EXIT_KEYS",
    "StrangeLoopDetector",
    "EchoPattern",
    "RiskEngine",
    "PositionSignal",
    "RiskMetrics"
]
