#!/usr/bin/env python3
"""Slot-State Mapper 🚦

Converts millisecond-level BTC ticks into:
    • dynamic 2-bit slot codes (00/01/10/11)
    • entropy-smoothed 16-slot frames
    • recursive SHA-256 digests with momentum coupling

Implements the quantitative upgrades outlined in the design notes:
1. Slot-weighted σ based on intra-slot variance
2. VWΔ (drift-aware volume-weighted delta) classification
3. Local entropy compression to eliminate redundant states
4. Digest momentum coupling  (Digestₜ = SHA256(bitsₜ ⊕ Digestₜ₋₁))

This module is intentionally self-contained so it can be unit-tested in
isolation and later plugged into `clock_tick_router` or any live feed.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Tick:
    ts: float  # seconds (epoch)
    price: float
    volume: float


@dataclass
class SlotResult:
    start_ts: float
    end_ts: float
    bit_code: int  # 0..3
    sigma: float
    vwd_delta: float
    entropy_before: float
    entropy_after: float
    raw_state_sequence: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SlotAccumulator – collects raw ticks and outputs 2-bit slot codes
# ---------------------------------------------------------------------------

class SlotAccumulator:
    """Aggregate ticks into a fixed-length slot and produce a 2-bit code."""

    def __init__(self, slot_seconds: int = 225, entropy_window: int = 4):
        self.slot_seconds = slot_seconds
        self.entropy_window = entropy_window  # how many previous states to eval entropy

        self._ticks: List[Tick] = []
        self._current_slot_start: float | None = None
        self._state_history: Deque[int] = deque(maxlen=entropy_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_tick(self, price: float, volume: float, ts: float | None = None) -> Tuple[bool, SlotResult | None]:
        """Feed a new tick.  Returns (slot_finished, slot_result)."""
        ts = ts or time.time()
        if self._current_slot_start is None:
            self._current_slot_start = ts

        self._ticks.append(Tick(ts, price, volume))

        # slot finished?
        if ts - self._current_slot_start >= self.slot_seconds:
            result = self._close_slot()
            return True, result
        return False, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_slot(self) -> SlotResult:
        ticks = self._ticks
        start_ts = self._current_slot_start or ticks[0].ts
        end_ts = ticks[-1].ts

        prices = np.array([t.price for t in ticks], dtype=float)
        vols = np.array([t.volume for t in ticks], dtype=float)

        # 1. drift-aware VWAP and returns
        vwap = np.sum(prices * vols) / max(np.sum(vols), 1e-9)
        returns = np.diff(prices) / prices[:-1]

        sigma = float(np.std(returns)) if len(returns) > 1 else 0.0

        # Volume weighted delta (VWΔ)
        vw_delta = float(np.sum((prices - vwap) * vols) / max(np.sum(vols), 1e-9))

        # Classification boundaries
        down_th = -sigma
        up_th = sigma

        if vw_delta < down_th:
            bit_code = 0  # 00 large down
        elif down_th <= vw_delta < 0:
            bit_code = 1  # 01 small down / drift
        elif 0 <= vw_delta < up_th:
            bit_code = 2  # 10 small up / drift
        else:
            bit_code = 3  # 11 large up or volume spike

        # Entropy BEFORE compression
        entropy_before = _local_entropy(list(self._state_history) + [bit_code])

        # Local entropy compression: if entropy is below threshold collapse to 0
        ENTROPY_THRESHOLD = 0.2  # tunable
        compressed_code = 0 if entropy_before < ENTROPY_THRESHOLD else bit_code

        # update history with compressed code
        self._state_history.append(compressed_code)
        entropy_after = _local_entropy(list(self._state_history))

        # reset slot
        self._ticks = []
        self._current_slot_start = None

        return SlotResult(
            start_ts=start_ts,
            end_ts=end_ts,
            bit_code=compressed_code,
            sigma=sigma,
            vwd_delta=vw_delta,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            raw_state_sequence=list(self._state_history),
        )


# ---------------------------------------------------------------------------
# Digest builder with momentum coupling
# ---------------------------------------------------------------------------

class DigestMomentumBuilder:
    """Builds recursive SHA-256 digests from slot bit-streams."""

    def __init__(self):
        self.prev_digest: bytes = b"\x00" * 32  # 256-bit zero vector
        self.bits_buffer: List[int] = []  # collects 16 slots

    def add_slot(self, slot_code: int) -> Tuple[bool, bytes]:
        """Add a 2-bit code. Returns (digest_ready, digest)."""
        # store as 2 bits
        self.bits_buffer.append(slot_code & 0x3)
        if len(self.bits_buffer) < 16:
            return False, b""

        # build 32-byte bit-vector (=256 bits)
        bitstring = "".join(f"{code:02b}" for code in self.bits_buffer)
        preimage_bytes = int(bitstring, 2).to_bytes(32, "big")

        # momentum coupling
        xored = bytes(a ^ b for a, b in zip(preimage_bytes, self.prev_digest))
        digest = hashlib.sha256(xored).digest()

        # roll state
        self.prev_digest = digest
        self.bits_buffer.clear()

        return True, digest


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _local_entropy(seq: List[int]) -> float:
    if not seq:
        return 0.0
    counts = np.bincount(seq, minlength=4)  # 4 possible states
    probs = counts / counts.sum()
    non_zero = probs[probs > 0]
    return float(-np.sum(non_zero * np.log2(non_zero)))


# ---------------------------------------------------------------------------
# Quick self-test (run as module)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    acc = SlotAccumulator(slot_seconds=1)  # fast test slot
    dig = DigestMomentumBuilder()

    start = time.time()
    for _ in range(40):  # 40 seconds
        price = 50000 + random.uniform(-100, 100)
        vol = random.uniform(0.1, 5)
        done, slot = acc.add_tick(price, vol, ts=start + _)
        if done and slot:
            ready, digest = dig.add_slot(slot.bit_code)
            print(f"slot {slot.bit_code} σ={slot.sigma:.4f} VWΔ={slot.vwd_delta:.2f} H_before={slot.entropy_before:.3f}")
            if ready:
                print("digest:", digest.hex()[:16], "…") 