"""
LanternCore: Advanced Semantic Hash Oracle System.

A revolutionary cryptographic semantic interpretation system that reads SHA-256 price hashes
as meaningful language patterns, creating a recursive profit oracle through entropy
fractal analysis.
"""

# Import base components first
from .entropy_generator import EntropyGenerator, FractalBlock
from .semantic_interpreter import LanguagePattern, SemanticInterpreter
from .truth_scorer import TruthScore, TruthScorer
from .nexus_thought_core import NexusThoughtCore, ZalgoLockState

# Import memory components
from .hash_memory import HashMemoryDatabase, SemanticCorrelation

# Import higher-level components that depend on others
from .lantern_eye import HashBlock, LanternEye, SemanticInterpretation
from .main_loop import LanternMainLoop, LanternProcessingResult

# Import enhanced components
from .channel_relay_navigator import (
    BitDepth,
    ChannelRelayNavigator,
    ChannelType,
    MathematicalState,
    NavigationVector,
    RelayState,
)
from .legacy_math_connectivity import (
    ConnectivityMatrix,
    LegacyMathematicalConnectivity,
    LegacyMathVector,
)
from .enhanced_main_loop import EnhancedLanternMainLoop

__all__ = [
    # Core Components
    "LanternEye",
    "EntropyGenerator",
    "SemanticInterpreter",
    "TruthScorer",
    "HashMemoryDatabase",
    "LanternMainLoop",
    "NexusThoughtCore",
    # Enhanced Components
    "ChannelRelayNavigator",
    "LegacyMathematicalConnectivity",
    "EnhancedLanternMainLoop",
    # Data Structures
    "HashBlock",
    "SemanticInterpretation",
    "FractalBlock",
    "LanguagePattern",
    "TruthScore",
    "SemanticCorrelation",
    "LanternProcessingResult",
    "ZalgoLockState",
    # Enhanced Data Structures
    "MathematicalState",
    "NavigationVector",
    "LegacyMathVector",
    "ConnectivityMatrix",
    # Enums
    "BitDepth",
    "ChannelType",
    "RelayState",
]

# Core instance for global access
_lantern_eye_instance = None


def get_lantern_eye() -> LanternEye:
    """Get global LanternEye instance"""
    global _lantern_eye_instance
    if _lantern_eye_instance is None:
        _lantern_eye_instance = LanternEye()
    return _lantern_eye_instance
