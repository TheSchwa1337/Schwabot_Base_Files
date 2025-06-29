# -*- coding: utf-8 -*-
"""
Internal State Management System
===============================

Provides comprehensive state management, fileization, and visualizer integration
for the trading system. Ensures continuous functionality, prevents JSON hang-ups,
and maintains proper lint compliance.

Main Components:
- StateContinuityManager: Core state management and validation
- FileizationManager: Safe file I/O with JSON hang-up prevention
- VisualizerIntegration: Connects states to visualizers and panels
- EnhancedStateManager: Advanced state management with logging, memory, and backlogs
- SystemIntegration: Connects all internal systems with proper initialization
"""

from .enhanced_state_manager import BacklogEntry, BTCPriceHash, EnhancedStateManager, LogLevel, SystemMemory, SystemMode
from .fileization_manager import FileizationManager
from .state_continuity_manager import StateContinuityManager, StateSnapshot, StateType
from .system_integration import SystemIntegration
from .visualizer_integration import VisualizerIntegration

__all__ = [
    "StateContinuityManager",
    "StateType",
    "StateSnapshot",
    "FileizationManager",
    "VisualizerIntegration",
    "EnhancedStateManager",
    "SystemMode",
    "LogLevel",
    "SystemMemory",
    "BacklogEntry",
    "BTCPriceHash",
    "SystemIntegration",
]

# Version information
__version__ = "2.0.0"
__author__ = "Schwabot Development Team"
__description__ = "Enhanced Internal State Management System for Trading Bot"
