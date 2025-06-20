#!/usr/bin/env python3
"""Hash tick synchronizer – SHA256-based tick matching and timing sync.

Implements the formulas:
    H_tick(t) = SHA256(p(t)·Δv·Δt)
    Ξ_sync = match(H_tick(t), H_map)
    Δτ = |tick(t₁) − tick(t₂)|
    σ_sync(t) = e^(−Δτ² / σ²) · 𝟙_{Ξ_sync}

This module provides hash-based synchronization between market ticks and
internal ghost state transitions for temporal alignment.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict

__all__: list[str] = ["compute_tick_hash", "sync_probability", "hash_match_check"]

# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------


def compute_tick_hash(
    price: float,
    delta_volume: float,
    delta_time: float,
) -> str:  # noqa: D401
    """Return H_tick(t) = SHA256(p(t)·Δv·Δt) as hex string.

    Parameters
    ----------
    price
        Current market price p(t).
    delta_volume
        Volume change Δv since last tick.
    delta_time
        Time delta Δt since last tick (seconds).
    """
    # Compute product and encode as bytes
    product = price * delta_volume * delta_time
    data = f"{product:.10f}".encode('utf-8')
    
    # SHA256 hash
    hash_obj = hashlib.sha256(data)
    return hash_obj.hexdigest()


# ---------------------------------------------------------------------------
# Synchronization logic
# ---------------------------------------------------------------------------


def hash_match_check(
    current_hash: str,
    hash_map: Dict[str, float],
    *,
    tolerance: int = 2,
) -> bool:  # noqa: D401
    """Return Ξ_sync = match(H_tick(t), H_map) boolean indicator.

    Parameters
    ----------
    current_hash
        Current tick hash to check.
    hash_map
        Dictionary mapping known hashes to their values.
    tolerance
        Maximum Hamming distance for fuzzy matching.
    """
    if current_hash in hash_map:
        return True
    
    # Fuzzy match via Hamming distance
    for known_hash in hash_map:
        if len(known_hash) == len(current_hash):
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(current_hash, known_hash))
            if hamming_dist <= tolerance:
                return True
    
    return False


def sync_probability(
    tick_t1: float,
    tick_t2: float,
    sigma: float,
    xi_sync: bool,
) -> float:  # noqa: D401
    """Return σ_sync(t) = e^(−Δτ² / σ²) · 𝟙_{Ξ_sync}.

    Parameters
    ----------
    tick_t1, tick_t2
        Timestamps of two ticks for Δτ calculation.
    sigma
        Gaussian spread parameter.
    xi_sync
        Boolean indicator from hash_match_check.
    """
    if not xi_sync:
        return 0.0
    
    if sigma <= 0:
        return 1.0 if tick_t1 == tick_t2 else 0.0
    
    delta_tau = abs(tick_t1 - tick_t2)
    gaussian_weight = math.exp(-(delta_tau ** 2) / (sigma ** 2))
    
    return gaussian_weight 