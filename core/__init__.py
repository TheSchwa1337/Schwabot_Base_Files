"""
Core Module for Hash Recollection System
Exports all the key components for pattern recognition and trading.
"""

from .hash_recollection import HashRecollectionSystem, HashEntry
from .entropy_tracker import EntropyTracker, EntropyState
from .bit_operations import BitOperations, PhaseState
from .pattern_utils import PatternUtils, PatternMatch, ENTRY_KEYS, EXIT_KEYS
from .strange_loop_detector import StrangeLoopDetector, EchoPattern
from .risk_engine import RiskEngine, PositionSignal, RiskMetrics

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