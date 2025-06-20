#!/usr/bin/env python3
"""
Schwabot Core Package
====================

Core system components for the Schwabot trading system.
Provides fault handling, constants, and essential utilities.
"""

# Core imports
from .constants import (
    PSI_INFINITY,
    FIBONACCI_SCALING,
    INVERSE_PSI,
    KELLY_SAFETY_FACTOR,
    SHARPE_TARGET,
    WindowsCliCompatibilityHandler
)

from .fault_bus import (
    FaultBus,
    FaultBusEvent,
    FaultType,
    FaultResolver,
    ThermalFaultResolver,
    ProfitFaultResolver,
    BitmapFaultResolver,
    GPUFaultResolver,
    RecursiveLoopResolver,
    FallbackFaultResolver
)

from .error_handler import ErrorHandler
from .filters import DataFilter
from .type_defs import *
from .import_resolver import ImportResolver

# Version information
__version__ = "1.0.0"
__author__ = "Schwabot Development Team"

# Package exports
__all__ = [
    # Constants
    'PSI_INFINITY',
    'FIBONACCI_SCALING', 
    'INVERSE_PSI',
    'KELLY_SAFETY_FACTOR',
    'SHARPE_TARGET',
    'WindowsCliCompatibilityHandler',
    
    # Fault handling
    'FaultBus',
    'FaultBusEvent',
    'FaultType',
    'FaultResolver',
    'ThermalFaultResolver',
    'ProfitFaultResolver',
    'BitmapFaultResolver',
    'GPUFaultResolver',
    'RecursiveLoopResolver',
    'FallbackFaultResolver',
    
    # Utilities
    'ErrorHandler',
    'DataFilter',
    'ImportResolver',
]
