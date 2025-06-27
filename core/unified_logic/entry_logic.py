# -*- coding: utf-8 -*-
"""Phantom entry logic for price-pressure based signals, phase- and tier-aware."""

def entry_score(
    dp_norm: float, sigma_vol: float, w_btc: float = 1.2, w_usdc: float = 0.8, phase: str = "mid"
) -> float:
    """
    Calculate entry score for trading signals, phase- and tier-aware.

    Args:
        dp_norm: Normalized price change
        sigma_vol: Volatility measure
        w_btc: BTC weight coefficient (default 1.2)
        w_usdc: USDC weight coefficient (default 0.8)
        phase: 'low', 'mid', or 'high' (affects weighting)

    Returns:
        Entry score (positive → long, negative → short)
    """
    phase_weights = {"low": 0.8, "mid": 1.0, "high": 1.2}
    return (w_btc * phase_weights[phase] * dp_norm) - (w_usdc * phase_weights[phase] * sigma_vol) 