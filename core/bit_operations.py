"""
Bit Operations Module
Handles 42-bit pattern encoding, phase extraction, and density analysis
for the Hash Recollection System.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhaseState:
    """Represents the multi-scale bit pattern phase state"""
    b4: int      # 4-bit micro phase
    b8: int      # 8-bit mid phase  
    b42: int     # 42-bit full pattern
    tier: int
    density: float
    timestamp: float
    variance_short: float = 0.0
    variance_mid: float = 0.0
    variance_long: float = 0.0


class BitOperations:
    """
    Manages bit pattern operations including encoding, phase extraction,
    density calculations, and variance tracking.
    """
    
    def __init__(self):
        """Initialize bit operations with density tracking buffers."""
        # Density tracking buffers for variance calculation
        self.density_history = {
            'short': deque(maxlen=5),    # 5-tick micro variance
            'mid': deque(maxlen=16),     # 16-tick standard variance
            'long': deque(maxlen=64)     # 64-tick macro variance
        }
        
        # Position cache for fast lookup
        self.position_cache = {}
        
        # Profit tier thresholds
        self.tier_thresholds = {
            0: (0.0, 0.25),    # Low activity
            1: (0.25, 0.42),   # Exit zone
            2: (0.42, 0.57),   # Neutral zone
            3: (0.57, 0.75),   # Entry zone
            4: (0.75, 1.0)     # High confidence zone
        }
        
        # Pattern analysis cache
        self.pattern_cache = {}
        
    def calculate_42bit_float(self, entropy: float) -> int:
        """
        Convert entropy float to 42-bit integer representation (Rule #3).
        
        Args:
            entropy: Float entropy value
            
        Returns:
            42-bit integer pattern
        """
        # Scale entropy to large integer preserving precision
        scaled = int(entropy * 1e12)
        
        # Mask to 42 bits
        mask_42 = (1 << 42) - 1
        bit_pattern = scaled & mask_42
        
        # Cache the pattern
        self.pattern_cache[entropy] = bit_pattern
        
        return bit_pattern
    
    def extract_phase_bits(self, bit_pattern: int) -> Tuple[int, int, int]:
        """
        Extract 4-bit, 8-bit, and 42-bit phase components (Rule #4).
        
        Args:
            bit_pattern: 42-bit integer pattern
            
        Returns:
            Tuple of (b4, b8, b42)
        """
        b42 = bit_pattern
        b8 = (bit_pattern >> 34) & 0xFF   # Extract 8 bits starting at position 34
        b4 = (bit_pattern >> 38) & 0xF    # Extract 4 bits starting at position 38
        
        return b4, b8, b42
    
    def calculate_bit_density(self, bit_pattern: int) -> float:
        """
        Calculate density of 1s in bit pattern (Rule #3).
        
        d = ||b||₁ / 42
        
        Args:
            bit_pattern: Integer bit pattern
            
        Returns:
            Density value between 0 and 1
        """
        # Count number of 1s (Hamming weight)
        ones_count = bin(bit_pattern).count('1')
        density = ones_count / 42.0
        
        return density
    
    def calculate_density_variance(self, densities: List[float]) -> float:
        """
        Calculate variance of density values (Rule #3).
        
        σ² = (1/n) * Σ(dᵢ - d̄)²
        
        Args:
            densities: List of density values
            
        Returns:
            Variance of densities
        """
        if len(densities) < 2:
            return 0.0
        
        mean = sum(densities) / len(densities)
        variance = sum((d - mean) ** 2 for d in densities) / len(densities)
        
        return variance
    
    def update_density_tracking(self, density: float):
        """
        Update all density tracking windows with new density value.
        
        Args:
            density: Current bit pattern density
        """
        for window in self.density_history.values():
            window.append(density)
    
    def get_multi_scale_variances(self) -> Dict[str, float]:
        """
        Calculate density variances across multiple time scales (Rule #3).
        
        Returns:
            Dictionary with short/mid/long variance values
        """
        variances = {}
        
        for scale, buffer in self.density_history.items():
            if len(buffer) >= 2:
                variances[scale] = self.calculate_density_variance(list(buffer))
            else:
                variances[scale] = 0.0
        
        return variances
    
    def analyze_bit_pattern(self, bit_pattern: int) -> Dict[str, float]:
        """
        Comprehensive bit pattern analysis including density and tier.
        
        Args:
            bit_pattern: 42-bit integer pattern
            
        Returns:
            Dictionary with pattern metrics
        """
        # Calculate basic density
        density = self.calculate_bit_density(bit_pattern)
        
        # Update density tracking
        self.update_density_tracking(density)
        
        # Get multi-scale variances
        variances = self.get_multi_scale_variances()
        
        # Extract phase components
        b4, b8, b42 = self.extract_phase_bits(bit_pattern)
        
        # Determine profit tier based on density
        tier = self.get_profit_tier(bit_pattern)
        
        # Calculate pattern strength using density and variance
        pattern_strength = self._calculate_pattern_strength(
            density, 
            variances.get('mid', 0.0)
        )
        
        # Segment density analysis (for different bit regions)
        long_density = self.calculate_bit_density(bit_pattern & 0x3FFF)  # Lower 14 bits
        mid_density = self.calculate_bit_density((bit_pattern >> 14) & 0x3FFF)  # Middle 14 bits
        short_density = self.calculate_bit_density((bit_pattern >> 28) & 0x3FFF)  # Upper 14 bits
        
        return {
            'pattern_strength': pattern_strength,
            'density': density,
            'long_density': long_density / 14.0 * 42.0,  # Normalize to 42-bit scale
            'mid_density': mid_density / 14.0 * 42.0,
            'short_density': short_density / 14.0 * 42.0,
            'tier': tier,
            'variance_short': variances.get('short', 0.0),
            'variance_mid': variances.get('mid', 0.0),
            'variance_long': variances.get('long', 0.0),
            'b4': b4,
            'b8': b8
        }
    
    def get_profit_tier(self, bit_pattern: int) -> int:
        """
        Determine profit tier based on bit pattern density.
        
        Args:
            bit_pattern: 42-bit pattern
            
        Returns:
            Tier level 0-4
        """
        density = self.calculate_bit_density(bit_pattern)
        
        for tier, (low, high) in self.tier_thresholds.items():
            if low <= density < high:
                return tier
        
        return 2  # Default neutral tier
    
    def _calculate_pattern_strength(self, density: float, variance: float) -> float:
        """
        Calculate pattern strength using density and variance.
        
        High density with low variance = strong pattern
        
        Args:
            density: Current bit density
            variance: Mid-scale variance
            
        Returns:
            Pattern strength score 0-1
        """
        # Base strength from density distance to ideal (0.57-0.75 range)
        ideal_density = 0.66  # Middle of entry zone
        density_score = 1.0 - abs(density - ideal_density) / 0.5
        density_score = max(0.0, min(1.0, density_score))
        
        # Penalize high variance (unstable patterns)
        variance_penalty = 1.0 - min(variance * 5.0, 1.0)
        
        # Combined strength
        pattern_strength = density_score * variance_penalty
        
        return pattern_strength
    
    def update_position_cache(self, value: int, density: float, tier: int, 
                            collapse_type: str = 'mid'):
        """
        Update position cache for fast pattern lookup.
        
        Args:
            value: Bit pattern value
            density: Pattern density
            tier: Profit tier
            collapse_type: Type of collapse ('short', 'mid', 'long')
        """
        self.position_cache[value] = {
            'density': density,
            'tier': tier,
            'collapse_type': collapse_type,
            'timestamp': np.datetime64('now')
        }
        
        # Limit cache size
        if len(self.position_cache) > 10000:
            # Remove oldest entries
            sorted_keys = sorted(
                self.position_cache.keys(), 
                key=lambda k: self.position_cache[k]['timestamp']
            )
            for key in sorted_keys[:1000]:
                del self.position_cache[key]
    
    def create_phase_state(self, bit_pattern: int, entropy_state) -> PhaseState:
        """
        Create a complete phase state from bit pattern and entropy.
        
        Args:
            bit_pattern: 42-bit pattern
            entropy_state: EntropyState object
            
        Returns:
            PhaseState object with all components
        """
        # Extract phase bits
        b4, b8, b42 = self.extract_phase_bits(bit_pattern)
        
        # Calculate density
        density = self.calculate_bit_density(bit_pattern)
        
        # Get tier
        tier = self.get_profit_tier(bit_pattern)
        
        # Get variances
        variances = self.get_multi_scale_variances()
        
        return PhaseState(
            b4=b4,
            b8=b8,
            b42=b42,
            tier=tier,
            density=density,
            timestamp=entropy_state.timestamp,
            variance_short=variances.get('short', 0.0),
            variance_mid=variances.get('mid', 0.0),
            variance_long=variances.get('long', 0.0)
        )
    
    def hamming_distance(self, pattern1: int, pattern2: int) -> int:
        """
        Calculate Hamming distance between two bit patterns.
        
        Args:
            pattern1: First bit pattern
            pattern2: Second bit pattern
            
        Returns:
            Number of differing bits
        """
        xor_result = pattern1 ^ pattern2
        return bin(xor_result).count('1')
    
    def pattern_similarity(self, pattern1: int, pattern2: int) -> float:
        """
        Calculate similarity score between two patterns.
        
        Args:
            pattern1: First bit pattern
            pattern2: Second bit pattern
            
        Returns:
            Similarity score 0-1
        """
        distance = self.hamming_distance(pattern1, pattern2)
        similarity = 1.0 - (distance / 42.0)
        return similarity 