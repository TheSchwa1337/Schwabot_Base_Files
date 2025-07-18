#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathLib Package for Schwabot Trading System
==========================================

Provides mathematical libraries and utilities.
"""

__version__ = "1.0.0"
__author__ = "Schwabot Team"

# Import key modules
from . import mathlib_v4
from . import matrix_fault_resolver
from . import memkey_sync
from . import persistent_homology
from . import quantum_strategy

__all__ = [
    'mathlib_v4',
    'matrix_fault_resolver',
    'memkey_sync',
    'persistent_homology',
    'quantum_strategy'
]
