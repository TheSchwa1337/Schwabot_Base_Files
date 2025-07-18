"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Cache Module
=================

Provides caching functionality for mathematical results.
This module is a simplified interface to the mathematical framework integrator.
"""


# Re-export for backward compatibility
from .math.mathematical_framework_integrator import MathResultCache, MathConfig

__all__ = ["MathResultCache", "MathConfig"]
