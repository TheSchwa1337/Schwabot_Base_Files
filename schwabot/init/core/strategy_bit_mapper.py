"""
strategy_bit_mapper.py
----------------------
Converts a market tick blob into an n-bit binary pattern, then segments
it into fixed-size buckets for pattern matching against profit memory.
"""
from __future__ import annotations

import hashlib
from typing import List


def strategy_bit_mapper(tick_blob: str, bits: int = 512) -> str:
    """Hash tick_blob and return a binary string of length *bits*."""
    # Initial SHA-256 hash
    h = hashlib.sha256(tick_blob.encode()).hexdigest()
    bin_str = bin(int(h, 16))[2:].zfill(256)
    # Extend for bits > 256 by re-hashing
    prev = h
    while len(bin_str) < bits:
        prev = hashlib.sha256(prev.encode()).hexdigest()
        bin_str += bin(int(prev, 16))[2:].zfill(256)
    return bin_str[:bits]


def segment_pattern(bit_str: str, segment_size: int) -> List[str]:
    """Split bit_str into segments of *segment_size* bits."""
    return [bit_str[i:i+segment_size] for i in range(0, len(bit_str), segment_size)] 