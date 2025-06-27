# -*- coding: utf - 8 -*-\n"""Phantom entry logic for price - pressure based signals."""
""""""
""""""
""""""
""""""
# -*- coding: utf - 8 -*-\n"""Phantom entry logic for price - pressure based signals."""

""""""
""""""
""""""
""""""
# -*- coding: utf - 8 -*-\n"""Phantom entry logic for price - pressure based signals."""
# -*- coding: utf - 8 -*-\n"""Phantom entry logic for price - pressure based signals."""


from __future__ import annotations
def entry_score():

    dp_norm: float, sigma_vol: float, w_btc: float = 1.2, w_usdc: float = 0.8
    -> float:


"""Calculate entry score based on price pressure."""
""""""
""""""

Compute entry signal: \\u03a6_entry = w_btc.deltap_norm - w_usdc.sigma_vol

Args:
dp_norm: Normalized price change
sigma_vol: Volatility measure
w_btc: BTC weight coefficient (default 1.2)
        w_usdc: USDC weight coefficient (default 0.8)

Returns:
Entry score (positive -> long, negative -> short)
    """"""
""""""
""""""
# return (w_btc * dp_norm) - (w_usdc * sigma_vol)



""""""
""""""
""""""
""""""
