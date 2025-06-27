# -*- coding: utf-8 -*-
"""Unified Logic Module for BTC tick processing, ghost conditionals, and phase-aware math."""

from .btc_tick_backlog import BTCTick, save_btc_tick
from .float_valuation import float_valuation
from .ghost_conditionals import ghost_conditional
from .entry_logic import entry_score
from .unicode_emoji_asic import label_state
from .phase_math import phase_adjust

__all__ = [
    "BTCTick",
    "save_btc_tick",
    "float_valuation",
    "ghost_conditional",
    "entry_score",
    "label_state",
    "phase_adjust"
] 