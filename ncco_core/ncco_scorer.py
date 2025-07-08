from core.clean_unified_math import clean_unified_math as unified_math
from typing import Any, List
import os
import platform

# -*- coding: utf-8 -*-
"""
NCCO Scorer Module for NCCO_CORE

This module implements the scoring system for Neuro-Collapsed Command Operators
(NCCOs) based on price deltas, bit modes, and unified mathematical operations.

Purpose: Score NCCO candidates to determine which strategy should be selected
during the collapse process.
"""


try:
    pass
except ImportError:
    # Fallback for development
    class UnifiedMathMock:
        @staticmethod
        def abs(value):
            return abs(value) if value is not None else 0
    unified_math = UnifiedMathMock()


# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """Windows CLI compatibility for emoji and Unicode handling."""

    @staticmethod
    def is_windows_cli():-> bool:
        """Detect if running in Windows CLI environment."""
        return platform.system() == "Windows" and (
            "cmd" in os.environ.get("COMSPEC", "").lower()
            or "powershell" in os.environ.get("PSModulePath", "").lower()
        )

    @staticmethod
    def safe_print():-> str:
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
    def log_safe():-> None:
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
# NCCO SCORING FUNCTIONS
# =====================================

def score_nccos():-> List[Any]:
    """
    Score a list of NCCO candidates based on price deltas and bit modes.
    
    Scoring formula: score = abs(price_delta) * bit_mode
    
    Args:
        nccos: List of NCCO objects with price_delta and bit_mode attributes
        
    Returns:
        List of scored NCCOs with updated score attributes
    """
    if not nccos:
        return []
        
    for ncco in nccos:
        # Basic scoring logic: score = abs(price_delta) * bit_mode
        price_delta = getattr(ncco, 'price_delta', 0)
        bit_mode = getattr(ncco, 'bit_mode', 1)
        
        # Calculate score using unified math
        ncco.score = unified_math.abs(price_delta) * bit_mode
        
        # Add confidence based on pattern matching
        if hasattr(ncco, 'pattern_confidence'):
            ncco.score *= ncco.pattern_confidence
            
        # Add entropy penalty for high volatility
        if hasattr(ncco, 'entropy_score'):
            entropy_penalty = max(0, 1 - ncco.entropy_score)
            ncco.score *= entropy_penalty
    
    return nccos


def rank_nccos_by_score():-> List[Any]:
    """
    Rank NCCOs by their calculated scores in descending order.
    
    Args:
        nccos: List of scored NCCO objects
        
    Returns:
        List of NCCOs ranked by score (highest first)
    """
    return sorted(nccos, key=lambda x: getattr(x, 'score', 0), reverse=True)


def filter_nccos_by_threshold():-> List[Any]:
    """
    Filter NCCOs to only include those above a minimum score threshold.
    
    Args:
        nccos: List of NCCO objects
        min_score: Minimum score threshold
        
    Returns:
        List of NCCOs with scores above threshold
    """
    return [ncco for ncco in nccos if getattr(ncco, 'score', 0) >= min_score]


def calculate_confidence_score():-> float:
    """
    Calculate a confidence score for a single NCCO based on multiple factors.
    
    Args:
        ncco: NCCO object to evaluate
        
    Returns:
        Confidence score between 0 and 1
    """
    base_score = getattr(ncco, 'score', 0)
    
    # Normalize to 0-1 range
    if base_score <= 0:
        return 0.0
        
    # Apply additional confidence factors
    confidence = min(1.0, base_score / 100.0)  # Normalize to 0-1
    
    # Boost confidence for patterns with good history
    if hasattr(ncco, 'pattern_success_rate'):
        confidence *= ncco.pattern_success_rate
        
    # Reduce confidence for high entropy
    if hasattr(ncco, 'entropy_score'):
        entropy_factor = max(0.1, 1 - ncco.entropy_score)
        confidence *= entropy_factor
        
    return min(1.0, confidence)
