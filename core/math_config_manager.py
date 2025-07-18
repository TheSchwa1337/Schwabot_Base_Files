"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Config Manager Module
==========================

Provides configuration management for mathematical operations.
This module is a simplified interface to the mathematical framework integrator.
"""


# Re-export for backward compatibility
from .math.mathematical_framework_integrator import MathConfigManager, MathConfig

__all__ = ["MathConfigManager", "MathConfig"]
