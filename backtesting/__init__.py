#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtesting Module for Schwabot Trading System
==============================================

This module provides backtesting functionality for the Schwabot trading system.

Main Components:
- SimpleBacktester: Basic backtesting engine
- HistoricalDataManager: Historical data management
- InternalBacktester: Internal backtesting utilities
"""

from .historical_data_manager import HistoricalDataManager
from .simple_backtester import SimpleBacktester

__all__ = [
    "SimpleBacktester",
    "HistoricalDataManager",
] 