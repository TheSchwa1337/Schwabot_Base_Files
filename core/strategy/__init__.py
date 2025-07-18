#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Module - Core Trading Strategy Components
================================================

Provides core trading strategy components for the Schwabot system.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "Schwabot Development Team"

# Import strategy modules with error handling
try:
    from .multi_phase_strategy_weight_tensor import (
        MultiPhaseStrategyWeightTensor,
        create_multi_phase_strategy_weight_tensor,
    )

    MULTI_PHASE_TENSOR_AVAILABLE = True
except ImportError:
    MultiPhaseStrategyWeightTensor = None
    create_multi_phase_strategy_weight_tensor = None
    MULTI_PHASE_TENSOR_AVAILABLE = False
    logger.warning("Multi-phase strategy weight tensor not available")

try:
    from .loss_anticipation_curve import (
        LossAnticipationCurve,
        create_loss_anticipation_curve,
    )

    LOSS_ANTICIPATION_AVAILABLE = True
except ImportError:
    LossAnticipationCurve = None
    create_loss_anticipation_curve = None
    LOSS_ANTICIPATION_AVAILABLE = False
    logger.warning("Loss anticipation curve not available")

try:
    from .enhanced_math_ops import EnhancedMathOps, create_enhanced_math_ops

    ENHANCED_MATH_AVAILABLE = True
except ImportError:
    EnhancedMathOps = None
    create_enhanced_math_ops = None
    ENHANCED_MATH_AVAILABLE = False
    logger.warning("Enhanced math operations not available")

try:
    from .volume_weighted_hash_oscillator import (
        VolumeWeightedHashOscillator,
        create_volume_weighted_hash_oscillator,
    )

    VOLUME_HASH_OSCILLATOR_AVAILABLE = True
except ImportError:
    VolumeWeightedHashOscillator = None
    create_volume_weighted_hash_oscillator = None
    VOLUME_HASH_OSCILLATOR_AVAILABLE = False
    logger.warning("Volume-weighted hash oscillator not available")

try:
    from .zygot_zalgo_entropy_dual_key_gate import (
        ZygotZalgoEntropyDualKeyGate,
        create_zygot_zalgo_entropy_dual_key_gate,
    )

    ZYGOT_ZALGO_GATE_AVAILABLE = True
except ImportError:
    ZygotZalgoEntropyDualKeyGate = None
    create_zygot_zalgo_entropy_dual_key_gate = None
    ZYGOT_ZALGO_GATE_AVAILABLE = False
    logger.warning("Zygot-Zalgo entropy dual key gate not available")

try:
    from .flip_switch_logic_lattice import (
        FlipSwitchLogicLattice,
        create_flip_switch_logic_lattice,
    )

    FLIP_SWITCH_AVAILABLE = True
except ImportError:
    FlipSwitchLogicLattice = None
    create_flip_switch_logic_lattice = None
    FLIP_SWITCH_AVAILABLE = False
    logger.warning("Flip switch logic lattice not available")

try:
    from .strategy_executor import StrategyExecutor

    STRATEGY_EXECUTOR_AVAILABLE = True
except ImportError:
    StrategyExecutor = None
    STRATEGY_EXECUTOR_AVAILABLE = False
    logger.warning("Strategy executor not available")

try:
    from .strategy_loader import StrategyLoader

    STRATEGY_LOADER_AVAILABLE = True
except ImportError:
    StrategyLoader = None
    STRATEGY_LOADER_AVAILABLE = False
    logger.warning("Strategy loader not available")

# Export list
__all__ = [
    # Core strategy classes
    "MultiPhaseStrategyWeightTensor",
    "LossAnticipationCurve",
    "EnhancedMathOps",
    "VolumeWeightedHashOscillator",
    "ZygotZalgoEntropyDualKeyGate",
    "FlipSwitchLogicLattice",
    "StrategyExecutor",
    "StrategyLoader",
    # Factory functions
    "create_multi_phase_strategy_weight_tensor",
    "create_loss_anticipation_curve",
    "create_enhanced_math_ops",
    "create_volume_weighted_hash_oscillator",
    "create_zygot_zalgo_entropy_dual_key_gate",
    "create_flip_switch_logic_lattice",
    # Availability flags
    "MULTI_PHASE_TENSOR_AVAILABLE",
    "LOSS_ANTICIPATION_AVAILABLE",
    "ENHANCED_MATH_AVAILABLE",
    "VOLUME_HASH_OSCILLATOR_AVAILABLE",
    "ZYGOT_ZALGO_GATE_AVAILABLE",
    "FLIP_SWITCH_AVAILABLE",
    "STRATEGY_EXECUTOR_AVAILABLE",
    "STRATEGY_LOADER_AVAILABLE",
    # System functions
    "create_trading_strategy_system",
    "get_strategy_status",
]


def create_trading_strategy_system(
    config: Optional[Dict[str, Any]] = None,
    enable_multi_phase: bool = True,
    enable_loss_anticipation: bool = True,
    enable_enhanced_math: bool = True,
    enable_volume_hash: bool = True,
    enable_zygot_zalgo: bool = True,
    enable_flip_switch: bool = True,
    enable_executor: bool = True,
    enable_loader: bool = True,
) -> Dict[str, Any]:
    """
    Factory function to create an integrated trading strategy system.

    Args:
    config: Configuration dictionary
    enable_multi_phase: Enable multi-phase strategy weight tensor
    enable_loss_anticipation: Enable loss anticipation curve
    enable_enhanced_math: Enable enhanced math operations
    enable_volume_hash: Enable volume-weighted hash oscillator
    enable_zygot_zalgo: Enable Zygot-Zalgo entropy dual key gate
    enable_flip_switch: Enable flip switch logic lattice
    enable_executor: Enable strategy executor
    enable_loader: Enable strategy loader

    Returns:
    Dictionary containing initialized strategy components
    """
    system = {}

    if enable_multi_phase and MULTI_PHASE_TENSOR_AVAILABLE:
        try:
            system["multi_phase_tensor"] = create_multi_phase_strategy_weight_tensor(
                config
            )
            logger.info("✅ Multi-phase strategy weight tensor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize multi-phase tensor: {e}")

    if enable_loss_anticipation and LOSS_ANTICIPATION_AVAILABLE:
        try:
            system["loss_anticipation"] = create_loss_anticipation_curve(config)
            logger.info("✅ Loss anticipation curve initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize loss anticipation: {e}")

    if enable_enhanced_math and ENHANCED_MATH_AVAILABLE:
        try:
            system["enhanced_math"] = create_enhanced_math_ops(config)
            logger.info("✅ Enhanced math operations initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced math: {e}")

    if enable_volume_hash and VOLUME_HASH_OSCILLATOR_AVAILABLE:
        try:
            system["volume_hash_oscillator"] = create_volume_weighted_hash_oscillator(
                config
            )
            logger.info("✅ Volume-weighted hash oscillator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize volume hash oscillator: {e}")

    if enable_zygot_zalgo and ZYGOT_ZALGO_GATE_AVAILABLE:
        try:
            system["zygot_zalgo_gate"] = create_zygot_zalgo_entropy_dual_key_gate(
                config
            )
            logger.info("✅ Zygot-Zalgo entropy dual key gate initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Zygot-Zalgo gate: {e}")

    if enable_flip_switch and FLIP_SWITCH_AVAILABLE:
        try:
            system["flip_switch"] = create_flip_switch_logic_lattice(config)
            logger.info("✅ Flip switch logic lattice initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize flip switch: {e}")

    if enable_executor and STRATEGY_EXECUTOR_AVAILABLE:
        try:
            system["executor"] = StrategyExecutor(config)
            logger.info("✅ Strategy executor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize strategy executor: {e}")

    if enable_loader and STRATEGY_LOADER_AVAILABLE:
        try:
            system["loader"] = StrategyLoader(config)
            logger.info("✅ Strategy loader initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize strategy loader: {e}")

    return system


def get_strategy_status() -> Dict[str, Any]:
    """
    Get the status of all strategy components.

    Returns:
    Dictionary containing availability status of all components
    """
    return {
        "multi_phase_tensor": MULTI_PHASE_TENSOR_AVAILABLE,
        "loss_anticipation": LOSS_ANTICIPATION_AVAILABLE,
        "enhanced_math": ENHANCED_MATH_AVAILABLE,
        "volume_hash_oscillator": VOLUME_HASH_OSCILLATOR_AVAILABLE,
        "zygot_zalgo_gate": ZYGOT_ZALGO_GATE_AVAILABLE,
        "flip_switch": FLIP_SWITCH_AVAILABLE,
        "strategy_executor": STRATEGY_EXECUTOR_AVAILABLE,
        "strategy_loader": STRATEGY_LOADER_AVAILABLE,
    } 