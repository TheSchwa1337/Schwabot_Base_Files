#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two Gram Detector Module
=========================
Provides two gram detector functionality for the Schwabot trading system.

This module implements 2-gram pattern detection for signal analysis:
• Pattern recognition in signal sequences
• Burst intensity detection
• Similarity vector generation
• Shannon entropy calculation

Key Features:
- Hardware-optimized pattern detection (CPU/GPU)
- Context-aware signal processing
- Recursive pattern matching
- Integration with symbolic math framework
"""

import logging
import logging


import logging
import logging


import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Import dependencies
try:
    from core.math_cache import MathResultCache
    from core.math_config_manager import MathConfigManager
    from core.math_orchestrator import MathOrchestrator
    from core.symbolic_math_interface import SignalField, SymbolicContext, TimeIndex

    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Math infrastructure not available")


class Status(Enum):
    """System status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PROCESSING = "processing"


class Mode(Enum):
    """Operation mode enumeration."""
    NORMAL = "normal"
    DEBUG = "debug"
    TEST = "test"
    PRODUCTION = "production"


@dataclass
class Config:
    """Configuration data class."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    max_patterns: int = 1024
    pattern_threshold: float = 0.1
    burst_threshold: float = 0.5


@dataclass
class Result:
    """Result data class."""
    success: bool = False
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class TwoGramDetector:
    """
    Two Gram Detector Implementation
    Provides core two gram detector functionality for pattern recognition.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TwoGramDetector with configuration."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False
        
        # Pattern storage
        self.pattern_history = deque(maxlen=1000)
        self.gram_frequencies = defaultdict(int)
        self.burst_patterns = defaultdict(float)
        self.similarity_cache = {}
        
        # Initialize math infrastructure if available
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_config = MathConfigManager()
            self.math_cache = MathResultCache()
            self.math_orchestrator = MathOrchestrator()
        
        self._initialize_system()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'enabled': True,
            'timeout': 30.0,
            'retries': 3,
            'debug': False,
            'log_level': 'INFO',
            'max_patterns': 1024,
            'pattern_threshold': 0.1,
            'burst_threshold': 0.5,
            'similarity_threshold': 0.8,
            'entropy_window': 16,
        }
    
    def _initialize_system(self) -> None:
        """Initialize the system."""
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            self.initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False
    
    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.logger.info(f"✅ {self.__class__.__name__} activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating {self.__class__.__name__}: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info(f"✅ {self.__class__.__name__} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
            'pattern_count': len(self.gram_frequencies),
            'burst_count': len(self.burst_patterns),
        }
    
    def feed_signal(self, signal: Union[float, int, str], context: Optional[SymbolicContext] = None) -> Optional[Tuple]:
        """
        Feed a signal into the 2-gram detector.
        
        Args:
            signal: Input signal value
            context: Optional symbolic context
            
        Returns:
            Tuple of detected 2-gram pattern or None
        """
        if not self.active:
            self.logger.warning("TwoGramDetector not active")
            return None
        
        try:
            # Convert signal to hashable format
            signal_hash = self._hash_signal(signal)
            
            # Add to pattern history
            self.pattern_history.append(signal_hash)
            
            # Check if we have enough signals for 2-gram
            if len(self.pattern_history) >= 2:
                # Extract 2-gram pattern
                pattern = tuple(self.pattern_history[-2:])
                
                # Update frequency
                self.gram_frequencies[pattern] += 1
                
                # Check for burst pattern
                if self._is_burst_pattern(pattern, context):
                    self.burst_patterns[pattern] = time.time()
                
                return pattern
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error feeding signal: {e}")
            return None
    
    def _hash_signal(self, signal: Union[float, int, str]) -> str:
        """Convert signal to hashable string."""
        try:
            if isinstance(signal, (int, float)):
                # Quantize numeric signals
                return f"num_{int(signal * 1000)}"
            else:
                # String signals
                return str(signal)
        except Exception as e:
            self.logger.error(f"Error hashing signal: {e}")
            return "unknown"
    
    def _is_burst_pattern(self, pattern: Tuple, context: Optional[SymbolicContext] = None) -> bool:
        """Check if pattern represents a burst."""
        try:
            frequency = self.gram_frequencies[pattern]
            threshold = self.config['burst_threshold']
            
            # Context-aware burst detection
            if context and context.phantom_layer:
                threshold *= 0.8  # Lower threshold for phantom layer
            
            return frequency >= threshold
            
        except Exception as e:
            self.logger.error(f"Error checking burst pattern: {e}")
            return False
    
    def get_top_patterns(self, n: int = 10) -> List[Tuple[Tuple, int]]:
        """Get top N most frequent patterns."""
        try:
            sorted_patterns = sorted(
                self.gram_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_patterns[:n]
        except Exception as e:
            self.logger.error(f"Error getting top patterns: {e}")
            return []
    
    def get_burst_patterns(self) -> List[Tuple[Tuple, float]]:
        """Get current burst patterns with timestamps."""
        try:
            return list(self.burst_patterns.items())
        except Exception as e:
            self.logger.error(f"Error getting burst patterns: {e}")
            return []
    
    def calculate_shannon_entropy(self, window_size: Optional[int] = None) -> float:
        """
        Calculate Shannon entropy of recent patterns.
        
        Args:
            window_size: Size of window to analyze (default from config)
            
        Returns:
            Shannon entropy value
        """
        try:
            if window_size is None:
                window_size = self.config['entropy_window']
            
            # Get recent patterns
            recent_patterns = list(self.pattern_history)[-window_size:]
            
            if len(recent_patterns) == 0:
                return 0.0
            
            # Calculate pattern frequencies
            pattern_counts = defaultdict(int)
            for pattern in recent_patterns:
                pattern_counts[pattern] += 1
            
            # Calculate probabilities
            total_patterns = len(recent_patterns)
            probabilities = [count / total_patterns for count in pattern_counts.values()]
            
            # Calculate Shannon entropy
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            
            return float(entropy)
            
        except Exception as e:
            self.logger.error(f"Error calculating Shannon entropy: {e}")
            return 0.0
    
    def calculate_burst_score(self, pattern: Tuple) -> float:
        """
        Calculate burst intensity score for a pattern.
        
        Args:
            pattern: 2-gram pattern tuple
            
        Returns:
            Burst score (0.0 to 1.0)
        """
        try:
            frequency = self.gram_frequencies.get(pattern, 0)
            total_patterns = sum(self.gram_frequencies.values())
            
            if total_patterns == 0:
                return 0.0
            
            # Normalized frequency
            normalized_freq = frequency / total_patterns
            
            # Burst score based on frequency and recency
            burst_score = min(normalized_freq * 10, 1.0)  # Scale and clamp
            
            return float(burst_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating burst score: {e}")
            return 0.0
    
    def generate_similarity_vector(self, target_pattern: Tuple, top_k: int = 5) -> List[Tuple[Tuple, float]]:
        """
        Generate similarity vector for target pattern.
        
        Args:
            target_pattern: Target 2-gram pattern
            top_k: Number of similar patterns to return
            
        Returns:
            List of (pattern, similarity_score) tuples
        """
        try:
            similarities = []
            
            for pattern in self.gram_frequencies.keys():
                if pattern != target_pattern:
                    # Simple similarity based on pattern overlap
                    similarity = self._calculate_pattern_similarity(target_pattern, pattern)
                    similarities.append((pattern, similarity))
            
            # Sort by similarity and return top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error generating similarity vector: {e}")
            return []
    
    def _calculate_pattern_similarity(self, pattern1: Tuple, pattern2: Tuple) -> float:
        """Calculate similarity between two patterns."""
        try:
            if len(pattern1) != 2 or len(pattern2) != 2:
                return 0.0
            
            # Simple similarity: exact match = 1.0, partial match = 0.5, no match = 0.0
            matches = 0
            if pattern1[0] == pattern2[0]:
                matches += 1
            if pattern1[1] == pattern2[1]:
                matches += 1
            
            return matches / 2.0
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    def clear_patterns(self) -> bool:
        """Clear all stored patterns."""
        try:
            self.pattern_history.clear()
            self.gram_frequencies.clear()
            self.burst_patterns.clear()
            self.similarity_cache.clear()
            self.logger.info("Pattern storage cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing patterns: {e}")
            return False
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern statistics."""
        try:
            total_patterns = sum(self.gram_frequencies.values())
            unique_patterns = len(self.gram_frequencies)
            burst_count = len(self.burst_patterns)
            entropy = self.calculate_shannon_entropy()
            
            return {
                'total_patterns': total_patterns,
                'unique_patterns': unique_patterns,
                'burst_patterns': burst_count,
                'shannon_entropy': entropy,
                'pattern_history_size': len(self.pattern_history),
                'most_frequent_pattern': max(self.gram_frequencies.items(), key=lambda x: x[1]) if self.gram_frequencies else None,
            }
        except Exception as e:
            self.logger.error(f"Error getting pattern statistics: {e}")
            return {}


# Factory function
def create_two_gram_detector(config: Optional[Dict[str, Any]] = None) -> TwoGramDetector:
    """Create a two gram detector instance."""
    return TwoGramDetector(config)
