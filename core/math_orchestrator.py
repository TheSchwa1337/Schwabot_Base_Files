"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Orchestrator Module
========================

Provides orchestration of mathematical operations.
This module is a simplified interface to the mathematical framework integrator.
"""

from core.math.mathematical_framework_integrator import MathOrchestrator, MathConfig

# Re-export for backward compatibility
__all__ = ['MathOrchestrator', 'MathConfig']
