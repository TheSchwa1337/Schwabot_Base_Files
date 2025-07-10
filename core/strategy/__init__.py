#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Module Package Initializer

This module contains the strategic intelligence components for Schwabot trading.
Provides access to advanced trading strategies including:
- Multi-phase strategy weight tensor
- Loss anticipation curve
- Enhanced math operations
- Flip switch logic lattice
"""

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__author__ = "Schwabot Development Team"

# Import strategy modules with error handling
try:
    from .multi_phase_strategy_weight_tensor import MultiPhaseStrategyWeightTensor
    MULTI_PHASE_TENSOR_AVAILABLE = True
except ImportError:
    MultiPhaseStrategyWeightTensor = None
    MULTI_PHASE_TENSOR_AVAILABLE = False
    logger.warning("Multi-phase strategy weight tensor not available")

try:
    from .loss_anticipation_curve import LossAnticipationCurve
    LOSS_ANTICIPATION_AVAILABLE = True
except ImportError:
    LossAnticipationCurve = None
    LOSS_ANTICIPATION_AVAILABLE = False
    logger.warning("Loss anticipation curve not available")

try:
    from .enhanced_math_ops import EnhancedMathOps
    ENHANCED_MATH_AVAILABLE = True
except ImportError:
    EnhancedMathOps = None
    ENHANCED_MATH_AVAILABLE = False
    logger.warning("Enhanced math operations not available")

try:
    from .flip_switch_logic_lattice import FlipSwitchLogicLattice
    FLIP_SWITCH_AVAILABLE = True
except ImportError:
    FlipSwitchLogicLattice = None
    FLIP_SWITCH_AVAILABLE = False
    logger.warning("Flip switch logic lattice not available")

# Export list
__all__ = [
    "MultiPhaseStrategyWeightTensor",
    "MULTI_PHASE_TENSOR_AVAILABLE",
    "LossAnticipationCurve",
    "LOSS_ANTICIPATION_AVAILABLE",
    "EnhancedMathOps",
    "ENHANCED_MATH_AVAILABLE",
    "FlipSwitchLogicLattice",
    "FLIP_SWITCH_AVAILABLE",
    "create_trading_strategy_system",
    "get_strategy_status",
]


def create_trading_strategy_system(
    config: Optional[Dict[str, Any]] = None,
    enable_multi_phase: bool = True,
    enable_loss_anticipation: bool = True,
    enable_enhanced_math: bool = True,
    enable_flip_switch: bool = True,
) -> Dict[str, Any]:
    """
    Factory function to create an integrated trading strategy system.

    Args:
        config: Configuration dictionary
        enable_multi_phase: Enable multi-phase strategy weight tensor
        enable_loss_anticipation: Enable loss anticipation curve
        enable_enhanced_math: Enable enhanced math operations
        enable_flip_switch: Enable flip switch logic lattice

    Returns:
        Dictionary containing initialized strategy components
    """
    system = {}

    if enable_multi_phase and MULTI_PHASE_TENSOR_AVAILABLE:
        try:
            system["multi_phase_tensor"] = MultiPhaseStrategyWeightTensor(config)
            logger.info("✅ Multi-phase strategy weight tensor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize multi-phase tensor: {e}")

    if enable_loss_anticipation and LOSS_ANTICIPATION_AVAILABLE:
        try:
            system["loss_anticipation"] = LossAnticipationCurve(config)
            logger.info("✅ Loss anticipation curve initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize loss anticipation: {e}")

    if enable_enhanced_math and ENHANCED_MATH_AVAILABLE:
        try:
            system["enhanced_math"] = EnhancedMathOps(config)
            logger.info("✅ Enhanced math operations initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced math: {e}")

    if enable_flip_switch and FLIP_SWITCH_AVAILABLE:
        try:
            system["flip_switch"] = FlipSwitchLogicLattice(config)
            logger.info("✅ Flip switch logic lattice initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize flip switch: {e}")

    logger.info(f"✅ Trading strategy system created with {len(system)} components")
    return system


def get_strategy_status() -> Dict[str, bool]:
    """Get the status of all strategy components."""
    return {
        "multi_phase_tensor": MULTI_PHASE_TENSOR_AVAILABLE,
        "loss_anticipation": LOSS_ANTICIPATION_AVAILABLE,
        "enhanced_math": ENHANCED_MATH_AVAILABLE,
        "flip_switch": FLIP_SWITCH_AVAILABLE,
    }
