# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import numpy as np
import math
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""Bit Operations - Mathematical Bit Manipulation for Schwabot.

This module provides comprehensive bit operations for hash processing,
bit phase analysis, and binary operations used in Schwabot's trading logic.

Mathematical Foundation:
- Bit rotation: ROTL(x, n) = (x << n) | (x >> (32 - n))
- Bit counting: popcount(x) = Σ(x >> i) & 1
- Bit phase extraction: phase = (hash >> offset) & mask
- Hamming distance: d(x,y) = popcount(x ⊕ y)
"""

import logging
from typing import List, Tuple, Optional, Union
# from core.unified_math_system import unified_math  # F811: duplicate import

logger = logging.getLogger(__name__)

class BitOperations:
    """Mathematical bit operations for hash and signal processing."""

    def __init__(self):
        self.max_bits = 64
        self.bit_masks = {i: (1 << i) - 1 for i in range(1, 65)}
        logger.info("BitOperations initialized")

    def rotate_left(self, value: int, shift: int, bits: int = 32) -> int:
        """
        Rotate left operation: ROTL(x, n) = (x << n) | (x >> (bits - n))

        Parameters:
        -----------
        value : int
            Value to rotate
        shift : int
            Number of positions to rotate left
        bits : int
            Bit width (default 32)

        Returns:
        --------
        int
            Rotated value
        """
        try:
            mask = self.bit_masks.get(bits, (1 << bits) - 1)
            shift = shift % bits
            return ((value << shift) | (value >> (bits - shift))) & mask
        except Exception as e:
            logger.error(f"Error in rotate_left: {e}")
            return value

    def rotate_right(self, value: int, shift: int, bits: int = 32) -> int:
        """
        Rotate right operation: ROTR(x, n) = (x >> n) | (x << (bits - n))

        Parameters:
        -----------
        value : int
            Value to rotate
        shift : int
            Number of positions to rotate right
        bits : int
            Bit width (default 32)

        Returns:
        --------
        int
            Rotated value
        """
        try:
            mask = self.bit_masks.get(bits, (1 << bits) - 1)
            shift = shift % bits
            return ((value >> shift) | (value << (bits - shift))) & mask
        except Exception as e:
            logger.error(f"Error in rotate_right: {e}")
            return value

    def popcount(self, value: int) -> int:
        """
        Population count: count number of set bits.

        Parameters:
        -----------
        value : int
            Value to count bits in

        Returns:
        --------
        int
            Number of set bits
        """
        try:
            return bin(value).count('1')
        except Exception as e:
            logger.error(f"Error in popcount: {e}")
            return 0

    def hamming_distance(self, x: int, y: int) -> int:
        """
        Calculate Hamming distance between two integers.

        Parameters:
        -----------
        x : int
            First value
        y : int
            Second value

        Returns:
        --------
        int
            Hamming distance
        """
        try:
            return self.popcount(x ^ y)
        except Exception as e:
            logger.error(f"Error in hamming_distance: {e}")
            return 0

    def extract_bit_phase(self, hash_value: int, offset: int, length: int) -> int:
        """
        Extract bit phase from hash value.

        Parameters:
        -----------
        hash_value : int
            Hash value to extract from
        offset : int
            Starting bit position
        length : int
            Number of bits to extract

        Returns:
        --------
        int
            Extracted bit phase
        """
        try:
            mask = self.bit_masks.get(length, (1 << length) - 1)
            return (hash_value >> offset) & mask
        except Exception as e:
            logger.error(f"Error in extract_bit_phase: {e}")
            return 0

    def set_bit(self, value: int, position: int) -> int:
        """Set bit at specified position."""
        try:
            return value | (1 << position)
        except Exception as e:
            logger.error(f"Error in set_bit: {e}")
            return value

    def clear_bit(self, value: int, position: int) -> int:
        """Clear bit at specified position."""
        try:
            return value & ~(1 << position)
        except Exception as e:
            logger.error(f"Error in clear_bit: {e}")
            return value

    def toggle_bit(self, value: int, position: int) -> int:
        """Toggle bit at specified position."""
        try:
            return value ^ (1 << position)
        except Exception as e:
            logger.error(f"Error in toggle_bit: {e}")
            return value

    def test_bit(self, value: int, position: int) -> bool:
        """Test if bit is set at specified position."""
        try:
            return bool(value & (1 << position))
        except Exception as e:
            logger.error(f"Error in test_bit: {e}")
            return False

    def count_trailing_zeros(self, value: int) -> int:
        """Count trailing zero bits."""
        try:
            if value == 0:
                return 64
            return (value & -value).bit_length() - 1
        except Exception as e:
            logger.error(f"Error in count_trailing_zeros: {e}")
            return 0

    def count_leading_zeros(self, value: int) -> int:
        """Count leading zero bits."""
        try:
            if value == 0:
                return 64
            return 64 - value.bit_length()
        except Exception as e:
            logger.error(f"Error in count_leading_zeros: {e}")
            return 0

    def reverse_bits(self, value: int, bits: int = 32) -> int:
        """Reverse bit order."""
        try:
            result = 0
            for i in range(bits):
                if value & (1 << i):
                    result |= (1 << (bits - 1 - i))
            return result
        except Exception as e:
            logger.error(f"Error in reverse_bits: {e}")
            return value

    def bit_entropy(self, values: List[int]) -> float:
        """
        Calculate bit entropy across a sequence of values.

        Parameters:
        -----------
        values : List[int]
            List of integer values

        Returns:
        --------
        float
            Bit entropy score [0, 1]
        """
        try:
            if not values:
                return 0.0

            # Convert to binary strings and analyze bit patterns
            bit_sequences = []
            for val in values:
                bits = bin(val)[2:].zfill(32)
                bit_sequences.append(bits)

            # Calculate entropy for each bit position
            entropy_scores = []
            for pos in range(32):
                bit_column = [seq[pos] for seq in bit_sequences]
                ones = bit_column.count('1')
                zeros = bit_column.count('0')
                total = len(bit_column)

                if total == 0:
                    entropy_scores.append(0.0)
                    continue

                p1 = ones / total
                p0 = zeros / total

                # Shannon entropy
                entropy = 0.0
                if p1 > 0:
                    entropy -= p1 * np.log2(p1)
                if p0 > 0:
                    entropy -= p0 * np.log2(p0)

                # Normalize to [0, 1]
                entropy_scores.append(entropy)

            # Return average entropy
            return unified_math.unified_math.mean(entropy_scores)

        except Exception as e:
            logger.error(f"Error in bit_entropy: {e}")
            return 0.5

    def bit_correlation(self, x: int, y: int, bits: int = 32) -> float:
        """
        Calculate bit correlation between two values.

        Parameters:
        -----------
        x : int
            First value
        y : int
            Second value
        bits : int
            Number of bits to compare

        Returns:
        --------
        float
            Correlation score [-1, 1]
        """
        try:
            # Extract bits
            x_bits = [(x >> i) & 1 for i in range(bits)]
            y_bits = [(y >> i) & 1 for i in range(bits)]

            # Calculate correlation
            x_mean = unified_math.unified_math.mean(x_bits)
            y_mean = unified_math.unified_math.mean(y_bits)

            numerator = sum((x_bits[i] - x_mean) * (y_bits[i] - y_mean)
                          for i in range(bits))

            x_var = sum((x_bits[i] - x_mean) ** 2 for i in range(bits))
            y_var = sum((y_bits[i] - y_mean) ** 2 for i in range(bits))

            denominator = unified_math.unified_math.sqrt(x_var * y_var)

            if denominator == 0:
                return 0.0

            return numerator / denominator

        except Exception as e:
            logger.error(f"Error in bit_correlation: {e}")
            return 0.0

    def bit_phase_analysis(self, hash_sequence: List[int],
                          phase_lengths: List[int] = [4, 8, 16, 32]) -> Dict[str, float]:
        """
        Analyze bit phases across different lengths.

        Parameters:
        -----------
        hash_sequence : List[int]
            Sequence of hash values
        phase_lengths : List[int]
            Bit lengths to analyze

        Returns:
        --------
        Dict[str, float]
            Analysis results for each phase length
        """
        try:
            results = {}

            for length in phase_lengths:
                phases = []
                for hash_val in hash_sequence:
                    phase = self.extract_bit_phase(hash_val, 0, length)
                    phases.append(phase)

                # Calculate statistics
                if phases:
                    results[f"{length}bit_mean"] = unified_math.unified_math.mean(phases)
                    results[f"{length}bit_std"] = unified_math.unified_math.std(phases)
                    results[f"{length}bit_entropy"] = self.bit_entropy(phases)
                    results[f"{length}bit_range"] = unified_math.max(phases) - unified_math.min(phases)

            return results

        except Exception as e:
            logger.error(f"Error in bit_phase_analysis: {e}")
            return {}

def main() -> None:
    """Test function for BitOperations."""
    safe_print("🧮 Testing Bit Operations...")

    ops = BitOperations()

    # Test basic operations
    test_value = 0b1010101010101010
    safe_print(f"Original: {bin(test_value)}")
    safe_print(f"Rotate left 4: {bin(ops.rotate_left(test_value, 4))}")
    safe_print(f"Rotate right 4: {bin(ops.rotate_right(test_value, 4))}")
    safe_print(f"Popcount: {ops.popcount(test_value)}")

    # Test bit phase extraction
    hash_val = 0x1234567890ABCDEF
    phase = ops.extract_bit_phase(hash_val, 0, 8)
    safe_print(f"8-bit phase: {phase}")

    # Test bit entropy
    test_sequence = [0x12345678, 0x87654321, 0xDEADBEEF, 0xF00DBABE]
    entropy = ops.bit_entropy(test_sequence)
    safe_print(f"Bit entropy: {entropy:.3f}")

    # Test bit correlation
    corr = ops.bit_correlation(0x12345678, 0x87654321)
    safe_print(f"Bit correlation: {corr:.3f}")

    # Test phase analysis
    analysis = ops.bit_phase_analysis(test_sequence)
    safe_print(f"Phase analysis: {analysis}")

    return 0

if __name__ == "__main__":
    exit(main())
