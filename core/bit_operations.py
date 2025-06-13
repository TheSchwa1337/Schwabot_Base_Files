"""
Bit Operations Module
===================

Implements specialized bit-level operations for the Schwabot system:
- 42-bit float calculations
- 4/8-bit positional mapping
- Bit density calculations
- Profit tiering lattice operations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import struct

@dataclass
class BitPosition:
    """Container for bit position data"""
    value: int
    density: float
    tier: int
    collapse_type: str  # 'long', 'mid', or 'short'

class BitOperations:
    """Core implementation of bit-level operations"""
    
    def __init__(self):
        self.position_cache: Dict[int, BitPosition] = {}
        self.density_map = np.zeros((42,), dtype=np.float32)
        self.tier_history: List[Tuple[int, float]] = []
        
    def calculate_42bit_float(self, value: float) -> int:
        """
        Convert float to 42-bit representation
        
        Args:
            value: Input float value
            
        Returns:
            42-bit integer representation
        """
        float_bytes = struct.pack('d', value)
        float_int = int.from_bytes(float_bytes, byteorder='little')
        exponent = (float_int >> 52) & 0x7FF
        mantissa = float_int & 0xFFFFFFFFFFFFF
        return ((exponent & 0x3F) << 36) | (mantissa & 0xFFFFFFFFF)
    
    def calculate_bit_density(self, value: int, collapse_type: str = 'mid') -> float:
        """
        Calculate bit density for a value using 2/4/8-bit collapse blocks
        
        Args:
            value: Input integer value
            collapse_type: One of ['long', 'mid', 'short']
            
        Returns:
            Bit density score (0.0 to 1.0)
        """
        binary = bin(value)[2:].zfill(42)

        if collapse_type == 'long':
            block_size = 8
        elif collapse_type == 'mid':
            block_size = 4
        else:
            block_size = 2

        blocks = [binary[i:i + block_size] for i in range(0, len(binary), block_size)]
        return sum(1 for block in blocks if '1' in block) / len(blocks)
    
    def update_position_cache(self, value: int, density: float, tier: int, collapse_type: str):
        """
        Store current bit position with density and tier metadata
        
        Args:
            value: Encoded integer value
            density: Computed density
            tier: Profit tier
            collapse_type: Collapse category
        """
        self.position_cache[value] = BitPosition(value, density, tier, collapse_type)
        binary = bin(value)[2:].zfill(42)
        for i, bit in enumerate(binary):
            if bit == '1':
                self.density_map[i] += 1
        self.tier_history.append((tier, density))
        
    def get_profit_tier(self, value: int) -> int:
        """
        Compute profit tier [0–7] based on weighted bit densities
        
        Args:
            value: Bit-encoded integer
            
        Returns:
            Integer tier from 0 (low) to 7 (high)
        """
        long_density = self.calculate_bit_density(value, 'long')
        mid_density = self.calculate_bit_density(value, 'mid')
        short_density = self.calculate_bit_density(value, 'short')
        weighted_density = (long_density * 0.4 + mid_density * 0.4 + short_density * 0.2)
        return int(weighted_density * 7)
        
    def analyze_bit_pattern(self, value: int) -> Dict[str, float]:
        """
        Analyze a 42-bit pattern for long/mid/short-term signal strengths
        
        Args:
            value: Input 42-bit encoded integer
            
        Returns:
            Dictionary of density metrics and pattern strength
        """
        long_d = self.calculate_bit_density(value, 'long')
        mid_d = self.calculate_bit_density(value, 'mid')
        short_d = self.calculate_bit_density(value, 'short')
        tier = self.get_profit_tier(value)
        strength = long_d * 0.5 + mid_d * 0.3 + short_d * 0.2

        return {
            'long_density': long_d,
            'mid_density': mid_d,
            'short_density': short_d,
            'tier': tier,
            'pattern_strength': strength,
            'density_map': self.density_map.tolist()
        }
        
    def get_tier_history(self) -> List[Tuple[int, float]]:
        """Return chronological tier/density log"""
        return self.tier_history
        
    def debug_binary(self, value: int) -> str:
        """Return 42-bit binary string for inspection"""
        return bin(value)[2:].zfill(42)
        
    def clear_cache(self):
        """Clear bit cache, density map, and tier history"""
        self.position_cache.clear()
        self.density_map.fill(0)
        self.tier_history.clear() 