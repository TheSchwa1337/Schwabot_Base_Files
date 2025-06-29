# -*- coding: utf-8 -*-
"""
Enhanced Lantern Core with Word Library and Glyph Integration
=============================================================

Provides advanced word categorization, entropy navigation, and bit gate processing
for the Schwabot trading system. Integrates with Ferris RDE and glyph containment
for recursive mathematical trading analysis.

Mathematical Integration:
- SHA-256 based word-to-hash mapping
- Bit gate processing for profit tier navigation  
- Entropy word generation for glyph routing
- 3.75-minute BTC price correlation with word patterns

MATHEMATICAL PRESERVATION: All core mathematical logic preserved.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np
import hashlib
import time
import random
import logging

logger = logging.getLogger(__name__)

# MATHEMATICAL PRESERVATION: Core bit gate type definitions
class BitGateType(Enum):
    """Bit gate types for lantern processing."""
    NULL_VECTOR = "NULL_VECTOR"
    LOW_TIER = "LOW_TIER"
    MID_TIER = "MID_TIER"
    PEAK_TIER = "PEAK_TIER"

class EntropyMode(Enum):
    """Entropy generation modes."""
    PROFIT_SYMBOLIC = "profit_symbolic"
    ENTROPY_RANDOM = "entropy_random"
    PATTERN_MATCH = "pattern_match"
    DUALISTIC_MAP = "dualistic_map"
    BTC_HASH_DERIVE = "btc_hash_derive"

# MATHEMATICAL PRESERVATION: Word library categories
PROFIT_WORDS = [
    "profit", "gain", "yield", "return", "growth", "increase", "rise",
    "bull", "moon", "rocket", "surge", "pump", "spike", "climb",
    "breakout", "momentum", "uptrend", "rally", "boom", "success",
    "wealth", "fortune", "treasure", "golden", "diamond", "victory"
]

NAVIGATION_WORDS = [
    "navigate", "steer", "guide", "direct", "route", "path", "journey",
    "compass", "beacon", "lighthouse", "map", "chart", "coordinate",
    "vector", "trajectory", "course", "heading", "waypoint", "anchor",
    "harbor", "dock", "port", "bridge", "passage", "channel"
]

# MATHEMATICAL PRESERVATION: Mathematical terms for glyph correlation
MATHEMATICAL_WORDS = [
    "matrix", "vector", "tensor", "algorithm", "equation", "formula",
    "calculate", "compute", "analyze", "measure", "quantify", "derive",
    "integrate", "differentiate", "optimize", "minimize", "maximize",
    "probability", "statistics", "variance", "correlation", "regression"
]

DUALISTIC_WORDS = [
    "dual", "binary", "toggle", "switch", "flip", "mirror", "reflect",
    "opposite", "inverse", "complement", "parallel", "balance", "harmony",
    "symmetry", "synchronize", "phase", "oscillate", "resonate", "align",
    "polar", "magnetic", "electric", "positive", "negative", "neutral"
]

# MATHEMATICAL PRESERVATION: Entropy and chaos terms
ENTROPY_WORDS = [
    "chaos", "random", "disorder", "turbulence", "volatility", "noise",
    "fluctuation", "variance", "deviation", "scatter", "dispersion",
    "unpredictable", "stochastic", "fractal", "complex", "dynamic",
    "emergence", "pattern", "structure", "order", "organization"
]

@dataclass
class BitGate:
    """Bit gate for processing states through tier navigation."""
    gate_type: BitGateType
    emoji: str
    processing_intensity: float = 1.0
    
    def process_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process state through bit gate with mathematical routing."""
        try:
            processed_state = data.copy()
            processed_state["bit_gate_type"] = self.gate_type.value
            processed_state["bit_gate_emoji"] = self.emoji
            processed_state["processing_timestamp"] = time.time()
            
            # Apply gate-specific processing
            if self.gate_type == BitGateType.NULL_VECTOR:
                processed_state = self._process_null_vector(processed_state)
            elif self.gate_type == BitGateType.LOW_TIER:
                processed_state = self._process_low_tier(processed_state)
            elif self.gate_type == BitGateType.MID_TIER:
                processed_state = self._process_mid_tier(processed_state)
            elif self.gate_type == BitGateType.PEAK_TIER:
                processed_state = self._process_peak_tier(processed_state)
                
            return processed_state
            
        except Exception as e:
            logger.error(f"Failed to process state through bit gate {self.gate_type.value}: {e}")
            return data
    
    def _process_null_vector(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """MATHEMATICAL PRESERVATION: Process null vector state."""
        data["null_vector_processed"] = True
        data["processing_intensity"] = 0.1
        data["state_energy"] = 0.0
        
        # Check for active flags
        for key in data.keys():
            if key.endswith("_active"):
                data[key] = False
                
        return data
    
    def _process_low_tier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """MATHEMATICAL PRESERVATION: Process low tier state."""
        data["low_tier_processed"] = True
        data["processing_intensity"] = 0.3
        data["state_energy"] = 0.25
        data["profit_potential"] = data.get("profit_potential", 0.0) * 1.1
        data["micro_profit_flag"] = True
        data["conservative_mode"] = True
        return data
    
    def _process_mid_tier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """MATHEMATICAL PRESERVATION: Process mid tier state."""
        data["mid_tier_processed"] = True
        data["processing_intensity"] = 0.6
        data["state_energy"] = 0.5
        data["profit_potential"] = data.get("profit_potential", 0.0) * 1.25
        data["momentum_analysis"] = True
        data["trend_tracking"] = True
        data["balanced_mode"] = True
        return data
    
    def _process_peak_tier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """MATHEMATICAL PRESERVATION: Process peak tier state."""
        data["peak_tier_processed"] = True
        data["processing_intensity"] = 1.0
        data["state_energy"] = 1.0
        data["profit_potential"] = data.get("profit_potential", 0.0) * 1.5
        data["max_profit_mode"] = True
        data["lantern_overlay"] = True
        data["aggressive_mode"] = True
        return data


class EnhancedLanternCore:
    """
    Enhanced Lantern Core with integrated word library and glyph processing.
    
    Integrates with Ferris RDE for 3.75-minute BTC price correlation
    and SHA-256 glyph routing for mathematical trading analysis.
    """
    
    def __init__(self):
        """Initialize Enhanced Lantern Core."""
        self.word_categories = {
            "profit_words": PROFIT_WORDS,
            "navigation_words": NAVIGATION_WORDS,
            "mathematical_words": MATHEMATICAL_WORDS,
            "dualistic_words": DUALISTIC_WORDS,
            "entropy_words": ENTROPY_WORDS
        }
        
        # MATHEMATICAL PRESERVATION: Bit gate mapping
        self.bit_gates = {
            "0": BitGate(BitGateType.NULL_VECTOR, "⚫"),
            "1": BitGate(BitGateType.LOW_TIER, "🟡"),
            "10": BitGate(BitGateType.MID_TIER, "🟠"),
            "11": BitGate(BitGateType.PEAK_TIER, "🔴")
        }
        
        self.bit_state_distribution = {"0": 0, "1": 0, "10": 0, "11": 0}
        self.word_usage_stats = {}
        self.entropy_cache = {}
        
        logger.info("✅ Enhanced Lantern Core with English Library initialized")
    
    def get_entropy_word(self, mode: EntropyMode = EntropyMode.ENTROPY_RANDOM) -> str:
        """Get entropy word based on mode for glyph routing."""
        try:
            if mode == EntropyMode.PROFIT_SYMBOLIC:
                return random.choice(PROFIT_WORDS)
            elif mode == EntropyMode.ENTROPY_RANDOM:
                return random.choice(ENTROPY_WORDS)
            elif mode == EntropyMode.PATTERN_MATCH:
                return random.choice(MATHEMATICAL_WORDS)
            elif mode == EntropyMode.DUALISTIC_MAP:
                return random.choice(DUALISTIC_WORDS)
            elif mode == EntropyMode.BTC_HASH_DERIVE:
                return random.choice(NAVIGATION_WORDS)
            else:
                return random.choice(ENTROPY_WORDS)
                
        except Exception as e:
            logger.error(f"Failed to get entropy word: {e}")
            return "entropy"
    
    def map_btc_price_to_word(self, btc_price: float) -> Dict[str, Any]:
        """Map BTC price to word entropy for 3.75-minute correlation."""
        try:
            # Generate hash from price
            price_str = f"{btc_price:.2f}"
            price_hash = hashlib.sha256(price_str.encode()).hexdigest()
            
            # Map hash to word category
            hash_int = int(price_hash[:4], 16)
            category_index = hash_int % len(self.word_categories)
            category_name = list(self.word_categories.keys())[category_index]
            
            # Select word from category
            words = self.word_categories[category_name]
            word_index = (hash_int // len(self.word_categories)) % len(words)
            selected_word = words[word_index]
            
            # Calculate word entropy
            word_entropy = self._calculate_word_entropy(selected_word)
            
            return {
                "btc_price": btc_price,
                "selected_word": selected_word,
                "category": category_name,
                "word_entropy": word_entropy,
                "price_hash": price_hash[:16],
                "mapping_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to map BTC price to word: {e}")
            return {"error": str(e)}
    
    def _calculate_word_entropy(self, word: str) -> float:
        """Calculate entropy value for word."""
        if word in self.entropy_cache:
            return self.entropy_cache[word]
            
        # Simple entropy calculation based on character distribution
        char_counts = {}
        for char in word.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
            
        total_chars = len(word)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * np.log2(probability)
                
        self.entropy_cache[word] = entropy
        return entropy
    
    def generate_word_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive word usage statistics."""
        try:
            total_words = sum(len(words) for words in self.word_categories.values())
            category_counts = {cat: len(words) for cat, words in self.word_categories.items()}
            
            # Calculate entropy for each category
            entropy_calculations = {}
            for category, words in self.word_categories.items():
                category_entropy = np.mean([self._calculate_word_entropy(word) for word in words])
                entropy_calculations[category] = category_entropy
            
            return {
                "total_words": total_words,
                "category_counts": category_counts,
                "entropy_calculations": entropy_calculations,
                "bit_state_distribution": self.bit_state_distribution.copy(),
                "generation_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate word statistics: {e}")
            return {"error": str(e)}
    
    def process_enhanced_state(self, data: Dict[str, Any], entropy_word: Optional[str] = None) -> Dict[str, Any]:
        """Process state with enhanced word mapping and bit gate routing."""
        try:
            if entropy_word is None:
                entropy_word = self.get_entropy_word()
            
            processed_state = data.copy()
            processed_state["entropy_word"] = entropy_word
            processed_state["text_bit_mapping"] = self._map_word_to_bits(entropy_word)
            processed_state["word_profit_symbolization"] = self._symbolize_word_profit(entropy_word)
            processed_state["text_entropy"] = self._calculate_word_entropy(entropy_word)
            
            # Route through appropriate bit gate
            bit_pattern = processed_state["text_bit_mapping"]
            if bit_pattern in self.bit_gates:
                bit_gate = self.bit_gates[bit_pattern]
                processed_state = bit_gate.process_state(processed_state)
                self.bit_state_distribution[bit_pattern] += 1
            
            logger.debug(f"Enhanced processing with word '{entropy_word}'")
            return processed_state
            
        except Exception as e:
            logger.error(f"Enhanced state processing failed: {e}")
            return data
    
    def _map_word_to_bits(self, word: str) -> str:
        """Map word to bit pattern for gate routing."""
        word_hash = hashlib.sha256(word.encode()).hexdigest()
        hash_int = int(word_hash[:2], 16)
        
        # Map to bit patterns: 0, 1, 10, 11
        if hash_int < 64:
            return "0"
        elif hash_int < 128:
            return "1"
        elif hash_int < 192:
            return "10"
        else:
            return "11"
    
    def _symbolize_word_profit(self, word: str) -> float:
        """Calculate profit symbolization value for word."""
        # Check if word is in profit categories
        profit_multiplier = 1.0
        
        if word in PROFIT_WORDS:
            profit_multiplier = 1.5
        elif word in MATHEMATICAL_WORDS:
            profit_multiplier = 1.3
        elif word in NAVIGATION_WORDS:
            profit_multiplier = 1.2
        elif word in DUALISTIC_WORDS:
            profit_multiplier = 1.1
        
        # Calculate base symbolization
        word_entropy = self._calculate_word_entropy(word)
        return word_entropy * profit_multiplier
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        try:
            total_processed = sum(self.bit_state_distribution.values())
            
            if total_processed > 0:
                # Calculate average energy from bit gate processing
                energy_weights = {"0": 0.0, "1": 0.25, "10": 0.5, "11": 1.0}
                weighted_energy = sum(
                    self.bit_state_distribution[pattern] * energy_weights[pattern]
                    for pattern in self.bit_state_distribution
                )
                average_energy = weighted_energy / total_processed
            else:
                average_energy = 0.0
            
            return {
                "total_processed": total_processed,
                "average_energy": average_energy,
                "bit_state_distribution": self.bit_state_distribution.copy(),
                "processing_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {"total_processed": 0, "average_energy": 0.0}


# Global instance for integration with Ferris RDE and Ghost Router
enhanced_lantern_core = EnhancedLanternCore()

# Export key functions for external access
def get_entropy_word(mode: EntropyMode = EntropyMode.ENTROPY_RANDOM) -> str:
    """Get entropy word for external use."""
    return enhanced_lantern_core.get_entropy_word(mode)

def map_btc_price_to_word(btc_price: float) -> Dict[str, Any]:
    """Map BTC price to word for external use."""
    return enhanced_lantern_core.map_btc_price_to_word(btc_price)

def process_enhanced_state(data: Dict[str, Any], entropy_word: Optional[str] = None) -> Dict[str, Any]:
    """Process enhanced state for external use."""
    return enhanced_lantern_core.process_enhanced_state(data, entropy_word)

# Export all key components
__all__ = [
    "EnhancedLanternCore",
    "BitGateType", 
    "EntropyMode",
    "enhanced_lantern_core",
    "get_entropy_word",
    "map_btc_price_to_word",
    "process_enhanced_state"
] 