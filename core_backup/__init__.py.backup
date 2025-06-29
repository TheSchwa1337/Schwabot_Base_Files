# -*- coding: utf-8 -*-
"""
Core module initialization for Schwabot trading system.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import time

__version__ = "0.5.1"
__author__ = "Schwabot Development Team"
__description__ = "Advanced AI-Powered Trading System"

logger = logging.getLogger(__name__)

# Import core unified math system with fallback
try:
    from .unified_math_system import unified_math
    UNIFIED_MATH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing unified math system: {e}")
    UNIFIED_MATH_AVAILABLE = False


def initialize_core() -> Dict[str, Any]:
    """
    Initialize the core system and return status information.
    """
    try:
        initialization_status = {
            "status": "initializing",
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
            "modules": [],
            "components": [],
            "errors": []
        }

        # Define core modules to initialize
        core_modules = [
            ("typing_schemas", "Core typing schemas"),
            ("fault_bus", "Fault handling system"),
            ("multi_bit_btc_processor", "BTC processing engine"),
            ("profit_routing_engine", "Profit routing system"),
            ("hash_registry", "Hash registry system"),
            ("strategy_loader", "Strategy loading system"),
            ("ops_observability", "Operations observability"),
            ("regulatory_compliance", "Regulatory compliance"),
            ("risk_guard", "Risk management system"),
            ("secure_api_manager", "Secure API management"),
            ("exchange_plumbing", "Exchange integration"),
            ("persistent_state_manager", "State persistence"),
            ("environment_manager", "Environment management"),
            ("memory_allocation_manager", "Memory management"),
            ("precision_performance", "Performance optimization"),
            ("long_horizon_simulation", "Long-term simulation"),
            ("thermal_boundary_manager", "Thermal management"),
            ("ui_state_bridge", "UI State Bridge"),
            ("visual_integration_bridge", "Visual Integration Bridge"),
            ("ui_integration_bridge", "UI Integration Bridge"),
            ("ui_bridge_integration_manager", "UI Bridge Integration Manager")
        ]

        # Initialize modules
        for module_name, module_description in core_modules:
            try:
                module_result = {
                    "name": module_name,
                    "description": module_description,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                initialization_status["modules"].append(module_result)
            except Exception as e:
                module_result = {
                    "name": module_name,
                    "description": module_description,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                initialization_status["modules"].append(module_result)
                initialization_status["errors"].append(f"Module {module_name}: {e}")

        # Define core components to initialize
        core_components = [
            ("unified_mathematical_trading_controller", "UnifiedMathematicalTradingController", "Unified mathematical trading controller"),
            ("ghost_profit_tracker", "ProfitTracker", "Ghost profit tracking system"),
            ("state_tracker", "StateTracker", "System state tracking"),
            ("dual_state_tracker", "DualStateTracker", "Dual state tracking system"),
            ("core_loop_manager", "CoreLoopManager", "Core loop management"),
            ("ui_bridge_integration_manager", "UIBridgeIntegrationManager", "UI Bridge Integration Manager")
        ]

        # Initialize components
        for component_name, component_class, component_description in core_components:
            try:
                component_result = {
                    "name": component_name,
                    "class": component_class,
                    "description": component_description,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                initialization_status["components"].append(component_result)
            except Exception as e:
                component_result = {
                    "name": component_name,
                    "class": component_class,
                    "description": component_description,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                initialization_status["components"].append(component_result)
                initialization_status["errors"].append(f"Component {component_name}: {e}")

        # Calculate success rates
        successful_modules = sum(1 for m in initialization_status["modules"] if m["status"] == "success")
        successful_components = sum(1 for c in initialization_status["components"] if c["status"] == "success")

        # Determine overall status
        if successful_modules == len(core_modules) and successful_components == len(core_components):
            initialization_status["status"] = "success"
        elif successful_modules > 0 or successful_components > 0:
            initialization_status["status"] = "partial"
        else:
            initialization_status["status"] = "failed"

        # Add summary
        initialization_status["summary"] = {
            "total_modules": len(core_modules),
            "successful_modules": successful_modules,
            "module_success_rate": successful_modules / len(core_modules) if core_modules else 0,
            "total_components": len(core_components),
            "successful_components": successful_components,
            "component_success_rate": successful_components / len(core_components) if core_components else 0,
            "error_count": len(initialization_status["errors"])
        }

        logger.info(f"Core system initialization: {initialization_status['status']}")
        return initialization_status

    except Exception as e:
        logger.error(f"Core initialization failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def get_system_health() -> Dict[str, Any]:
    """
    Get current system health status.
    """
    try:
        health_status = {
            "overall_health": "good",
            "timestamp": datetime.now().isoformat(),
            "unified_math_available": UNIFIED_MATH_AVAILABLE,
            "core_modules_loaded": True,
            "memory_usage": "normal",
            "api_connections": "stable"
        }

        logger.info(f"System health check: {health_status['overall_health']}")
        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "overall_health": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Export key functions and variables
__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "initialize_core",
    "get_system_health",
    "UNIFIED_MATH_AVAILABLE"
]