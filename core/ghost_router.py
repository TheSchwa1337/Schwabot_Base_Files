#!/usr/bin/env python3
"""Ghost router – conditional trade core.

This module wires together the **seven primary conditionals** outlined in the
high-level design note (hash-drift, pool stability, lantern vector match, AI
consensus, re-entry tolerance, profit-lock sync, narrative glyph overlay).

The implementation is *deliberately lightweight* – each conditional is a pure
function that returns a boolean.  :class:`GhostRouter` evaluates them in the
canonical order and emits one of three routes:

* ``"ghost_trade"``  – enter a stealth trade (BTC long or USDC exit).
* ``"hold_usdc"``    – defensive hold triggered by news overlay.
* ``"noop"``         – no action / wait.

Only NumPy + std-lib are required, keeping the stub dependency-free beyond
what Schwabot already ships.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Final, Tuple, Literal

import numpy as np

__all__: list[str] = ["GhostRouter", "ghost_router"]

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _hamming_dist(a: str, b: str) -> int:  # noqa: D401
    """Return Hamming distance of two equal-length hex strings."""
    if len(a) != len(b):  # pad shorter one for robustness
        raise ValueError("hash strings must have equal length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.shape != v2.shape:
        raise ValueError("vectors must share shape for cosine similarity")
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return 0.0 if norm == 0 else dot / norm

# -----------------------------------------------------------------------------
# Conditionals – each returns bool
# -----------------------------------------------------------------------------

_HASH_EPS: Final = 8  # ≤ 8 differing hex chars → similar hash
_POOL_STAB_EPS: Final = 0.1
_VECTOR_COS_THRESHOLD: Final = 0.97
_AI_TRUST_THRESHOLD: Final = 0.9
_DECAY_LAMBDA: Final = 0.001  # smaller ⇒ longer forgiveness window
_DECAY_THRESHOLD: Final = 0.5
_PROFIT_LOCK_EPS: Final = 0.0  # > projected_exit + ε
_NEWS_OVERLAY_THRESHOLD: Final = 0.6


def _hash_drift_detect(curr_hash: str, mem_hash: str) -> bool:
    return _hamming_dist(curr_hash, mem_hash) <= _HASH_EPS


def _pool_stability_check(vol_series: np.ndarray) -> bool:
    if vol_series.size == 0:
        return False
    return float(np.std(vol_series) / np.mean(vol_series)) < _POOL_STAB_EPS


def _lantern_match(vec: np.ndarray, reference: np.ndarray) -> bool:
    return _cosine_similarity(vec, reference) >= _VECTOR_COS_THRESHOLD


def _ai_consensus(hashes: Tuple[str, str, str], weights: Tuple[float, float, float]) -> bool:
    h1, h2, h3 = hashes
    if not (h1 == h2 == h3):
        return False
    trust = sum(weights)
    return trust >= _AI_TRUST_THRESHOLD


def _reentry_tolerance(opportunity_ts: float, now_ts: float) -> bool:
    decay = math.exp(-_DECAY_LAMBDA * (now_ts - opportunity_ts))
    return decay >= _DECAY_THRESHOLD


def _profit_lock_sync(curr_profit: float, projected_exit: float) -> bool:
    return curr_profit > projected_exit + _PROFIT_LOCK_EPS


def _news_overlay_route(score: float) -> bool:
    """Return True if bearish news requires USDC hold."""
    return score > _NEWS_OVERLAY_THRESHOLD

# -----------------------------------------------------------------------------
# Dataclass container for all inputs (optional convenience)
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class RouterInput:
    tick_hash: str
    mem_hash: str
    pool_volumes: np.ndarray  # USDC pool volume window
    btc_dip: bool  # Quick boolean indicator
    lantern_vec: np.ndarray
    lantern_ref: np.ndarray
    ai_hashes: Tuple[str, str, str]
    ai_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    opportunity_ts: float = 0.0
    now_ts: float = time.time()
    price_now: float = 0.0
    price_pred: float = 0.0
    curr_profit: float = 0.0
    projected_exit: float = 0.0
    news_score: float = 0.0  # aggregated sentiment score


# -----------------------------------------------------------------------------
# Core router
# -----------------------------------------------------------------------------


class GhostRouter:
    """Evaluate conditional chain; return routing decision string."""

    def route(self, data: RouterInput) -> str:  # noqa: D401
        # 1. Hash drift detection
        if not _hash_drift_detect(data.tick_hash, data.mem_hash):
            return "noop"

        # 2. Pool stability + BTC dip
        if not (_pool_stability_check(data.pool_volumes) and data.btc_dip):
            return "noop"

        # 3. Lantern vector match
        if not _lantern_match(data.lantern_vec, data.lantern_ref):
            return "noop"

        # 4. AI consensus chain
        if not _ai_consensus(data.ai_hashes, data.ai_weights):
            return "noop"

        # 5. Dead-signal re-entry tolerance
        if not _reentry_tolerance(data.opportunity_ts, data.now_ts):
            return "noop"

        # 6. Profit lock – if we are already beyond target, exit route
        if _profit_lock_sync(data.curr_profit, data.projected_exit):
            return "hold_usdc"

        # 7. Narrative glyph overlay – may override to defensive hold
        if _news_overlay_route(data.news_score):
            return "hold_usdc"

        # All green
        return "ghost_trade"


# -----------------------------------------------------------------------------
# Functional wrapper
# -----------------------------------------------------------------------------


def ghost_router(data: RouterInput) -> str:  # noqa: D401
    """Convenience wrapper around :class:`GhostRouter.route`."""
    return GhostRouter().route(data)


@dataclass(slots=True)
class ExecPacket:
    """Executable order packet ⟨V_final , route , O_t , τ_t⟩ (formula 16)."""

    volume: float
    route: Literal["vault_mode", "long_mode", "short_mode", "mid_mode"]
    price_offset: float
    hash_tag: str


# ------------------------------------------------------------------
# High-level helper – implement equations (1) … (8)
# ------------------------------------------------------------------


def compute_ghost_route(
    *,
    H_t: int,
    H_prev: int,
    E_t: float,
    D_t: float,
    rho_t: float,
    P_res: float,
    S_t: float,
    base_vol: float,
    psi: float = 0.5,
    theta_high: float = 0.8,
    theta_low: float = 0.2,
    beta1: float = 1.0,
    beta2: float = 0.5,
    beta3: float = 0.3,
    kappa: float = 0.01,
    epsilon: float = 1e-9,
    Q_max: float = 1e6,
    timestamp: float | None = None,
) -> ExecPacket:  # noqa: D401
    """Compute ghost routing decision and order size.

    Returns an :class:`ExecPacket` with volume, route string, price offset 0.0
    (placeholder) and hash-tag τₜ.
    """
    import hashlib
    import math

    delta_H = H_t - H_prev
    # (1) Φ_t
    phi_t = 1.0 / (1.0 + math.exp(-(beta1 * E_t - beta2 * abs(delta_H) + beta3 * D_t)))

    # (3) execution velocity
    v_exec = math.sqrt(P_res / (rho_t * phi_t + epsilon))

    # (4) volume throttle
    V_adj = base_vol * math.exp(-kappa * S_t)

    # (5)(6) route weights
    w_btc = psi * (1.0 - phi_t) * v_exec
    w_usdc = (1.0 - psi) * phi_t * v_exec

    # (7) route decision
    if phi_t > theta_high:
        route = "vault_mode"
    elif phi_t < theta_low and delta_H < 0:
        route = "long_mode"
    elif phi_t < theta_low and delta_H > 0:
        route = "short_mode"
    else:
        route = "mid_mode"

    # (8) final executable size
    Q_exec = max(0.0, min(V_adj * (w_btc + w_usdc), Q_max))

    # (9) hash-tag
    if timestamp is None:
        import time as _time
        timestamp = _time.time()
    tag_data = f"{H_t}{route}{timestamp}".encode()
    tau_t = hashlib.sha256(tag_data).hexdigest()

    return ExecPacket(volume=Q_exec, route=route, price_offset=0.0, hash_tag=tau_t) 