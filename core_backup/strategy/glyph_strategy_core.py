# -*- coding: utf-8 -*-
"""
Glyph-to-Strategy Proxy Core
----------------------------
Maps emojis, glyphs, or unicode characters to strategy bit-maps via SHA256.
Supports recursive strategy lookup, fractal memory encoding, and bitwise relay gear states.

Integrates with Schwabot's existing strategy infrastructure for both backtesting
and live execution modes.
"""

import hashlib
import random
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Import existing Schwabot components
try:
    from ..strategy_bit_mapper import StrategyBitMapper
    from ..strategy_logic import StrategyLogic, StrategyType, SignalType
    from ..unified_math_system import UnifiedMathSystem
except ImportError:
    # Fallback for standalone testing
    StrategyBitMapper = None
    StrategyLogic = None
    StrategyType = None
    SignalType = None
    UnifiedMathSystem = None

logger = logging.getLogger(__name__)

class GearState(Enum):
    """Gear state enumeration for strategy bit depth selection."""
    LOW_VOLUME = 4    # 4-bit strategies for low volume
    MED_VOLUME = 8    # 8-bit strategies for medium volume  
    HIGH_VOLUME = 16  # 16-bit strategies for high volume

@dataclass
class GlyphStrategyResult:
    """Result container for glyph strategy selection."""
    glyph: str
    gear_state: int
    strategy_id: int
    fractal_hash: str
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, any] = field(default_factory=dict)

class GlyphStrategyCore:
    """
    Core glyph-to-strategy mapping system.
    
    Maps emojis/glyphs to trading strategies via SHA256 hashing,
    with support for gear-driven bit depth selection and fractal memory.
    """
    
    def __init__(self, 
                 enable_fractal_memory: bool = True,
                 enable_gear_shifting: bool = True,
                 volume_thresholds: Tuple[float, float] = (1.5e6, 5e6),
                 random_seed: Optional[int] = None):
        """
        Initialize the glyph strategy core.
        
        Args:
            enable_fractal_memory: Enable persistent fractal hash memory
            enable_gear_shifting: Enable volume-based gear shifting
            volume_thresholds: (low_threshold, high_threshold) for gear selection
            random_seed: Random seed for reproducible results
        """
        self.enable_fractal_memory = enable_fractal_memory
        self.enable_gear_shifting = enable_gear_shifting
        self.volume_thresholds = volume_thresholds
        
        # Initialize fractal memory
        self.forever_fractal_hashes: List[str] = []
        self.fractal_memory_size = 10000
        
        # Strategy bit mapper integration
        self.bit_mapper = StrategyBitMapper() if StrategyBitMapper else None
        
        # Performance tracking
        self.stats = {
            "total_selections": 0,
            "gear_shifts": 0,
            "fractal_stores": 0,
            "avg_processing_time": 0.0
        }
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            
        logger.info(
            f"GlyphStrategyCore initialized: "
            f"fractal_memory={enable_fractal_memory}, "
            f"gear_shifting={enable_gear_shifting}"
        )
    
    def glyph_to_sha(self, glyph: str) -> str:
        """
        Convert glyph to SHA-256 hash.
        
        Args:
            glyph: Input glyph/emoji/unicode character
            
        Returns:
            SHA-256 hash string
        """
        return hashlib.sha256(glyph.encode('utf-8')).hexdigest()
    
    def sha_to_strategy_bits(self, sha: str, bit_depth: int = 4) -> int:
        """
        Convert SHA-256 hash to strategy bit pattern.
        
        Args:
            sha: SHA-256 hash string
            bit_depth: Target bit depth (4, 8, or 16)
            
        Returns:
            Strategy bit pattern as integer
        """
        # Extract first N hex characters based on bit depth
        hex_length = bit_depth // 4  # 4 bits per hex character
        hex_sub = sha[:hex_length]
        
        # Convert to binary and extract target bits
        binary = bin(int(hex_sub, 16))[2:].zfill(bit_depth)
        return int(binary[:bit_depth], 2)
    
    def glyph_strategy_lookup(self, glyph: str, gear_state: int = 4) -> int:
        """
        Translate glyph to strategy ID through SHA256 mapping.
        
        Args:
            glyph: Input glyph
            gear_state: Bit depth for strategy (4, 8, or 16)
            
        Returns:
            Strategy ID as integer
        """
        sha = self.glyph_to_sha(glyph)
        strategy_bits = self.sha_to_strategy_bits(sha, bit_depth=gear_state)
        return strategy_bits
    
    def gear_shift(self, current_volume: float) -> int:
        """
        Determine gear state based on volume signal.
        
        Args:
            current_volume: Current market volume
            
        Returns:
            Gear state (4, 8, or 16 bits)
        """
        if not self.enable_gear_shifting:
            return 4  # Default to 4-bit
            
        low_threshold, high_threshold = self.volume_thresholds
        
        if current_volume < low_threshold:
            gear_state = 4
        elif current_volume < high_threshold:
            gear_state = 8
        else:
            gear_state = 16
            
        self.stats["gear_shifts"] += 1
        return gear_state
    
    def store_fractal_hash(self, glyph: str, strategy_id: int, 
                          timestamp: Optional[str] = None) -> str:
        """
        Encode glyph + strategy into persistent fractal identity hash.
        
        Args:
            glyph: Input glyph
            strategy_id: Selected strategy ID
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            Fractal hash string
        """
        if not self.enable_fractal_memory:
            return ""
            
        ts = timestamp or datetime.utcnow().isoformat()
        core_string = f"{glyph}-{strategy_id}-{ts}"
        fractal_hash = hashlib.sha256(core_string.encode('utf-8')).hexdigest()
        
        # Store in fractal memory
        self.forever_fractal_hashes.append(fractal_hash)
        
        # Maintain memory size
        if len(self.forever_fractal_hashes) > self.fractal_memory_size:
            self.forever_fractal_hashes.pop(0)
            
        self.stats["fractal_stores"] += 1
        return fractal_hash
    
    def select_strategy(self, glyph: str, volume_signal: float = 0.0,
                       confidence_boost: float = 0.0) -> GlyphStrategyResult:
        """
        Combined strategy selection function for runtime use.
        
        Args:
            glyph: Input glyph/emoji
            volume_signal: Market volume signal for gear selection
            confidence_boost: Additional confidence boost (0.0 to 1.0)
            
        Returns:
            GlyphStrategyResult with complete strategy information
        """
        start_time = time.time()
        
        try:
            # Determine gear state
            gear_state = self.gear_shift(volume_signal)
            
            # Lookup strategy
            strategy_id = self.glyph_strategy_lookup(glyph, gear_state)
            
            # Store fractal hash
            fractal_hash = self.store_fractal_hash(glyph, strategy_id)
            
            # Calculate confidence
            base_confidence = 0.6  # Base confidence for glyph strategies
            confidence = min(1.0, base_confidence + confidence_boost)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_selections"] += 1
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (self.stats["total_selections"] - 1) + 
                 processing_time) / self.stats["total_selections"]
            )
            
            result = GlyphStrategyResult(
                glyph=glyph,
                gear_state=gear_state,
                strategy_id=strategy_id,
                fractal_hash=fractal_hash,
                confidence=confidence,
                metadata={
                    "processing_time": processing_time,
                    "volume_signal": volume_signal,
                    "gear_thresholds": self.volume_thresholds
                }
            )
            
            logger.debug(f"Strategy selected: {glyph} -> {strategy_id} "
                        f"(gear={gear_state}, confidence={confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            # Return fallback result
            return GlyphStrategyResult(
                glyph=glyph,
                gear_state=4,
                strategy_id=0,
                fractal_hash="",
                confidence=0.1,
                metadata={"error": str(e)}
            )
    
    def expand_strategy(self, base_strategy: int, target_depth: int = 8,
                       mode: str = "flip") -> List[int]:
        """
        Expand strategy using bit mapper if available.
        
        Args:
            base_strategy: Base strategy ID
            target_depth: Target bit depth
            mode: Expansion mode
            
        Returns:
            List of expanded strategies
        """
        if self.bit_mapper:
            return self.bit_mapper.expand_strategy_bits(
                base_strategy, target_depth, mode
            )
        else:
            # Fallback expansion
            return [base_strategy] * (target_depth // 4)
    
    def get_fractal_memory_stats(self) -> Dict[str, any]:
        """Get fractal memory statistics."""
        return {
            "total_hashes": len(self.forever_fractal_hashes),
            "memory_size": self.fractal_memory_size,
            "oldest_hash": self.forever_fractal_hashes[0] if self.forever_fractal_hashes else None,
            "newest_hash": self.forever_fractal_hashes[-1] if self.forever_fractal_hashes else None
        }
    
    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics."""
        return {
            **self.stats,
            "fractal_memory": self.get_fractal_memory_stats()
        }
    
    def reset_memory(self):
        """Reset fractal memory."""
        self.forever_fractal_hashes.clear()
        logger.info("Fractal memory reset")

# Convenience functions for direct use
def glyph_to_strategy(glyph: str, volume: float = 0.0) -> Dict[str, any]:
    """
    Convenience function for quick glyph-to-strategy conversion.
    
    Args:
        glyph: Input glyph
        volume: Market volume signal
        
    Returns:
        Strategy result dictionary
    """
    core = GlyphStrategyCore()
    result = core.select_strategy(glyph, volume)
    return {
        "glyph": result.glyph,
        "gear_state": result.gear_state,
        "strategy_id": result.strategy_id,
        "fractal_hash": result.fractal_hash,
        "confidence": result.confidence
    }

# Sample glyph mappings for testing
SAMPLE_GLYPH_MAPPINGS = {
    '🧠': "intelligence_strategy",
    '💀': "aggressive_strategy", 
    '🔥': "momentum_strategy",
    '⏳': "patience_strategy",
    '🌪️': "volatility_strategy",
    '⚡': "speed_strategy",
    '🛡️': "defensive_strategy",
    '🎯': "precision_strategy",
    '🔮': "prediction_strategy",
    '⚖️': "balance_strategy"
}

if __name__ == "__main__":
    # Test the glyph strategy core
    core = GlyphStrategyCore()
    
    test_glyphs = ['🧠', '💀', '🔥', '⏳', '🌪️']
    test_volumes = [1e6, 3e6, 6e6]  # Low, medium, high volume
    
    print("=== Glyph Strategy Core Test ===")
    
    for glyph in test_glyphs:
        for volume in test_volumes:
            result = core.select_strategy(glyph, volume)
            print(f"Glyph: {glyph}, Volume: {volume:.1e}, "
                  f"Strategy: {result.strategy_id}, Gear: {result.gear_state}")
    
    print(f"\nPerformance Stats: {core.get_performance_stats()}") 