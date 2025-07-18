"""
hash_drift_sync.py
------------------
Utility helpers for generating deterministic market hashes and validating
similarity / drift between hashes.  These hashes are used as part of the
execution-layer security gate: a trade will only execute if the current market
hash matches a known profitable pattern within a configurable tolerance.
"""

from __future__ import annotations

import hashlib

PREFIX_MATCH_DEFAULT = 12  # how many leading hex chars must match for approval


def market_hash(tick_blob: str) -> str:
    """Return a SHA-256 hexadecimal digest for *tick_blob* (e.g. price data)."""
    return hashlib.sha256(tick_blob.encode()).hexdigest()


def hash_similarity(hash_a: str, hash_b: str, prefix: int = PREFIX_MATCH_DEFAULT) -> bool:
    """True if the first *prefix* hex chars of *hash_a* and *hash_b* match."""
    return hash_a[:prefix] == hash_b[:prefix]
