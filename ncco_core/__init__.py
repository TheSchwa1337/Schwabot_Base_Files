# update
# -*- coding: utf-8 -*-
"""
NCCO_CORE (Neuro-Collapsed Command Operator) - Core Intelligence Module

The NCCO_CORE is Schwabot's central recursive command intelligence system that reduces
all trade strategy noise into a singular, decisive action based on:
- Live tick entropy analysis
- Memory echo from prior trades  
- Strategy hash pattern matching
- Entropy drift + adversarial behavior detection

Purpose: Bridge the gap between chaos and clarity in recursive AI trading decisions.
"""

import platform
import os
from typing import Any

try:
    from .ncco_scorer import score_nccos
    from .ncco_generator import generate_nccos
    from .ncco import NCCO
    from .harmony_memory import HarmonyMemory
    from .control_panel import AdvancedControlPanel
    from .fill_conjunction import FillConjunctionEngine
except ImportError:
    # Fallback imports for development
    score_nccos = None
    generate_nccos = None
    NCCO = None
    HarmonyMemory = None
    AdvancedControlPanel = None
    FillConjunctionEngine = None


# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """Windows CLI compatibility for emoji and Unicode handling."""

    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment."""
        return platform.system() == "Windows" and (
            "cmd" in os.environ.get("COMSPEC", "").lower()
            or "powershell" in os.environ.get("PSModulePath", "").lower()
        )

    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """Print message safely with Windows CLI compatibility."""
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            emoji_mapping = {
                "🚨": "[ALERT]",
                "⚠️": "[WARNING]", 
                "✅": "[SUCCESS]",
                "❌": "[ERROR]",
                "🔄": "[PROCESSING]",
                "🎯": "[TARGET]",
            }
            for emoji, marker in emoji_mapping.items():
                message = message.replace(emoji, marker)
        return message

    @staticmethod
    def log_safe(logger: Any, level: str, message: str) -> None:
        """Log message safely with Windows CLI compatibility."""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            ascii_message = safe_message.encode(
                "ascii", errors="replace"
            ).decode("ascii")
            getattr(logger, level.lower())(ascii_message)


# =====================================
# NCCO_CORE MATHEMATICAL FUNCTIONS
# =====================================

def strategy_collapse_function(strategies, weights):
    """
    Ψ_collapse(t) = argmax(S_i(t) · C_i)
    
    Reduces all possible strategy outputs to the one most aligned 
    with confidence and entropy history.
    
    Args:
        strategies: List of strategy candidates
        weights: Confidence weights from entropy + profit feedback
        
    Returns:
        Collapsed strategy selection
    """
    if not strategies or not weights or len(strategies) != len(weights):
        return None
        
    max_weight = max(weights)
    max_index = weights.index(max_weight)
    return strategies[max_index]


def entropy_vector_score(tick_data):
    """
    E_tick = (1/n) * Σ(x_i - x̄)²
    
    Measures spread in tick behavior to detect adversarial market noise.
    
    Args:
        tick_data: Array of tick values
        
    Returns:
        Entropy score (standard deviation)
    """
    if not tick_data:
        return 0.0
        
    mean_val = sum(tick_data) / len(tick_data)
    variance = sum((x - mean_val) ** 2 for x in tick_data) / len(tick_data)
    return variance ** 0.5


def hash_vector_projection(hash_value, k=8):
    """
    H(hash) = (int(hash[-1]) mod k)
    
    Translate a hash value into a stable strategy matrix route.
    
    Args:
        hash_value: Hash string 
        k: Number of strategy vectors
        
    Returns:
        Strategy vector index
    """
    if not hash_value:
        return 0
    try:
        last_char = hash_value[-1]
        return int(last_char, 16) % k
    except (ValueError, IndexError):
        return 0


__all__ = [
    "NCCO",
    "generate_nccos", 
    "score_nccos",
    "FillConjunctionEngine",
    "AdvancedControlPanel", 
    "HarmonyMemory",
    "WindowsCliCompatibilityHandler",
    "strategy_collapse_function",
    "entropy_vector_score",
    "hash_vector_projection",
]