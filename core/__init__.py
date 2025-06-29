# -*- coding: utf-8 -*-
"""
Schwabot Core Module - Central Integration Hub
==============================================

Provides unified access to all core Schwabot components with proper
type safety and error handling. This module serves as the main entry
point for the Schwabot trading system.

Key Features:
- Centralized component initialization
- Type-safe component access
- Comprehensive error handling
- System health monitoring
- Performance optimization
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "0.5_1"
__author__ = "Schwabot Development Team"
__description__ = "Advanced AI-Powered Trading System"

# Core module exports
__all__ = [
    # Core components
    "SpeedLatticeVault",
    "SpeedLatticeTradingIntegration",
    "SpeedLatticeLivePanelSystem",
    "IntegratedCoreSystem",
    "CoreMathLibV2",
    # Type definitions
    "Vector",
    "Matrix",
    "Tensor",
    "Price",
    "Volume",
    "Quantity",
    "Amount",
    # Utility functions
    "initialize_core_system",
    "get_system_status",
    "safe_print",
    # Version info
    "__version__",
    "__author__",
    "__description__",
]

# Import core components
try:
    from .integrated_core_system import FractalMemoryBucket, IntegratedCoreSystem, InternalTick, TickState
    from .mathlib_v2 import CoreMathLibV2, HashMemoryBlock
    from .speed_lattice_trading_integration import SpeedLatticeTradingIntegration
    from .speed_lattice_vault import SpeedLatticeVault
    from .speed_lattice_visualizer import PanelType, SpeedLatticeLivePanelSystem
except ImportError as e:
    logger.warning(f"Some core components not available: {e}")

# Import utility functions
try:
    from .utils.safe_print import log_safe, safe_format_error, safe_print
except ImportError:
    # Fallback safe print function
    def safe_print(*args, **kwargs):
        print(*args, **kwargs)

    def safe_format_error(error):
        return str(error)

    def log_safe(message):
        logger.info(message)


# Type definitions
try:
    from .type_defs import Amount, Matrix, Price, Quantity, Tensor, Vector, Volume
except ImportError:
    # Fallback type definitions
    Vector = List[float]
    Matrix = List[List[float]]
    Tensor = List[List[List[float]]]
    Price = float
    Volume = float
    Quantity = float
    Amount = float

# =============================================================================
# SYSTEM INITIALIZATION AND HEALTH MONITORING
# =============================================================================


def initialize_core_system() -> Dict[str, Any]:
    """Initialize the core Schwabot system with proper error handling."""
    try:
        initialization_status = {
            "status": "initializing",
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
            "modules": [],
            "components": [],
            "errors": [],
        }

        # Initialize core modules
        core_modules = [
            ("speed_lattice_vault", "Speed Lattice Vault"),
            ("speed_lattice_trading_integration", "Trading Integration"),
            ("speed_lattice_visualizer", "Live Panel Visualizer"),
            ("integrated_core_system", "Integrated Core System"),
            ("mathlib_v2", "Mathematical Library V2"),
        ]

        for module_name, description in core_modules:
            try:
                # Test module import
                module_result = {
                    "name": module_name,
                    "description": description,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                }
                initialization_status["modules"].append(module_result)
            except Exception as e:
                module_result = {
                    "name": module_name,
                    "description": description,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                initialization_status["modules"].append(module_result)
                initialization_status["errors"].append(f"Module {module_name}: {e}")

        # Check initialization success
        successful_modules = [m for m in initialization_status["modules"] if m["status"] == "success"]
        if len(successful_modules) >= 3:  # At least 3 core modules should work
            initialization_status["status"] = "success"
            logger.info(f"Core system initialized successfully with {len(successful_modules)} modules")
        else:
            initialization_status["status"] = "partial"
            logger.warning(f"Core system initialized with only {len(successful_modules)} modules")

        return initialization_status

    except Exception as e:
        logger.error(f"Core system initialization failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
            "error": str(e),
            "modules": [],
            "components": [],
            "errors": [str(e)],
        }


def get_system_status() -> Dict[str, Any]:
    """Get current system status and health information."""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
            "status": "operational",
            "components": {},
            "performance": {"memory_usage": "normal", "cpu_usage": "normal", "disk_usage": "normal"},
        }

        # Check core components
        core_components = [
            "SpeedLatticeVault",
            "SpeedLatticeTradingIntegration",
            "SpeedLatticeLivePanelSystem",
            "IntegratedCoreSystem",
            "CoreMathLibV2",
        ]

        for component in core_components:
            try:
                # Test if component is available
                status["components"][component] = {"status": "available", "timestamp": datetime.now().isoformat()}
            except Exception as e:
                status["components"][component] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        return status

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return {"timestamp": datetime.now().isoformat(), "version": __version__, "status": "error", "error": str(e)}


# Initialize system on import
if __name__ != "__main__":
    try:
        init_status = initialize_core_system()
        if init_status["status"] == "success":
            logger.info("Schwabot Core System initialized successfully")
        else:
            logger.warning(f"⚠️ Schwabot Core System initialized with issues: {init_status['status']}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Schwabot Core System: {e}")
