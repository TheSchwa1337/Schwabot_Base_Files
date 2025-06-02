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
        # Convert to 64-bit float first
        float_bytes = struct.pack('d', value)
        float_int = int.from_bytes(float_bytes, byteorder='little')
        
        # Extract 42 bits (6 bits for exponent, 36 bits for mantissa)
        exponent = (float_int >> 52) & 0x7FF
        mantissa = float_int & 0xFFFFFFFFFFFFF
        
        # Combine into 42-bit representation
        return ((exponent & 0x3F) << 36) | (mantissa & 0xFFFFFFFFF)
    
    def calculate_bit_density(self, value: int, collapse_type: str = 'mid') -> float:
        """
        Calculate bit density for a value
        
        Args:
            value: Input value
            collapse_type: Type of collapse ('long', 'mid', or 'short')
            
        Returns:
            Bit density value
        """
        # Convert to binary string
        binary = bin(value)[2:].zfill(42)
        
        # Calculate density based on collapse type
        if collapse_type == 'long':
            # Use 8-bit blocks
            blocks = [binary[i:i+8] for i in range(0, len(binary), 8)]
            density = sum(1 for block in blocks if '1' in block) / len(blocks)
        elif collapse_type == 'mid':
            # Use 4-bit blocks
            blocks = [binary[i:i+4] for i in range(0, len(binary), 4)]
            density = sum(1 for block in blocks if '1' in block) / len(blocks)
        else:  # short
            # Use 2-bit blocks
            blocks = [binary[i:i+2] for i in range(0, len(binary), 2)]
            density = sum(1 for block in blocks if '1' in block) / len(blocks)
            
        return density
    
    def update_position_cache(self, value: int, density: float, tier: int, collapse_type: str):
        """Update position cache with new value"""
        self.position_cache[value] = BitPosition(
            value=value,
            density=density,
            tier=tier,
            collapse_type=collapse_type
        )
        
        # Update density map
        binary = bin(value)[2:].zfill(42)
        for i, bit in enumerate(binary):
            if bit == '1':
                self.density_map[i] += 1
                
        # Update tier history
        self.tier_history.append((tier, density))
        
    def get_profit_tier(self, value: int) -> int:
        """
        Calculate profit tier based on bit patterns
        
        Args:
            value: Input value
            
        Returns:
            Profit tier (0-7)
        """
        # Calculate densities for all collapse types
        long_density = self.calculate_bit_density(value, 'long')
        mid_density = self.calculate_bit_density(value, 'mid')
        short_density = self.calculate_bit_density(value, 'short')
        
        # Weight the densities
        weighted_density = (
            long_density * 0.4 +  # Long-term patterns
            mid_density * 0.4 +   # Medium-term patterns
            short_density * 0.2   # Short-term patterns
        )
        
        # Map to profit tier (0-7)
        return int(weighted_density * 7)
        
    def analyze_bit_pattern(self, value: int) -> Dict[str, float]:
        """
        Analyze bit pattern for trading signals
        
        Args:
            value: Input value
            
        Returns:
            Dictionary of pattern metrics
        """
        # Calculate densities
        long_density = self.calculate_bit_density(value, 'long')
        mid_density = self.calculate_bit_density(value, 'mid')
        short_density = self.calculate_bit_density(value, 'short')
        
        # Calculate tier
        tier = self.get_profit_tier(value)
        
        # Calculate pattern strength
        pattern_strength = (
            long_density * 0.5 +   # Long-term stability
            mid_density * 0.3 +    # Medium-term trends
            short_density * 0.2    # Short-term signals
        )
        
        return {
            'long_density': long_density,
            'mid_density': mid_density,
            'short_density': short_density,
            'tier': tier,
            'pattern_strength': pattern_strength,
            'density_map': self.density_map.tolist()
        }
        
    def get_tier_history(self) -> List[Tuple[int, float]]:
        """Get history of tier changes"""
        return self.tier_history
        
    def clear_cache(self):
        """Clear position cache and reset density map"""
        self.position_cache.clear()
        self.density_map.fill(0)
        self.tier_history.clear() 