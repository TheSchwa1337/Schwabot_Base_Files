#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core module for Schwabot trading system.

This module provides clean, error-free implementations of the core
mathematical and trading components for algorithmic trading.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Math Infrastructure
try:
    from .math_cache import MathResultCache
    from .math_config_manager import MathConfigManager
    from .math_orchestrator import MathOrchestrator

    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Math infrastructure not available")

# Core Utilities
try:
    from .core_utilities import CoreUtilities

    CORE_UTILITIES_AVAILABLE = True
except ImportError:
    CORE_UTILITIES_AVAILABLE = False
    logger.warning("Core utilities not available")

# Bio Cellular Integration
try:
    from .bio_cellular_integration import (
        BioCellularIntegration,
        create_bio_cellular_integration,
    )

    BIO_CELLULAR_AVAILABLE = True
except ImportError:
    BIO_CELLULAR_AVAILABLE = False
    logger.warning("Bio cellular integration not available")

# Profit Optimization Engine
try:
    from .profit_optimization_engine import ProfitOptimizationEngine

    PROFIT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PROFIT_OPTIMIZATION_AVAILABLE = False
    logger.warning("Profit optimization engine not available")

# Quantum Mathematical Bridge
try:
    from .quantum_mathematical_bridge import QuantumMathematicalBridge

    QUANTUM_BRIDGE_AVAILABLE = True
except ImportError:
    QUANTUM_BRIDGE_AVAILABLE = False
    logger.warning("Quantum mathematical bridge not available")

# Entropy Math
try:
    from .entropy_math import EntropyMath

    ENTROPY_MATH_AVAILABLE = True
except ImportError:
    ENTROPY_MATH_AVAILABLE = False
    logger.warning("Entropy math not available")

# Tensor Score Utils
try:
    from .tensor_score_utils import TensorScoreUtils

    TENSOR_SCORE_AVAILABLE = True
except ImportError:
    TENSOR_SCORE_AVAILABLE = False
    logger.warning("Tensor score utils not available")

# DLT Waveform Engine
try:
    from .dlt_waveform_engine import DLTWaveformEngine

    DLT_WAVEFORM_AVAILABLE = True
except ImportError:
    DLT_WAVEFORM_AVAILABLE = False
    logger.warning("DLT waveform engine not available")

# Advanced Tensor Algebra
try:
    from .advanced_tensor_algebra import AdvancedTensorAlgebra

    ADVANCED_TENSOR_AVAILABLE = True
except ImportError:
    ADVANCED_TENSOR_AVAILABLE = False
    logger.warning("Advanced tensor algebra not available")

# Unified Profit Vectorization System
try:
    from .unified_profit_vectorization_system import (
        UnifiedProfitVectorizationSystem,
    )

    UNIFIED_PROFIT_AVAILABLE = True
except ImportError:
    UNIFIED_PROFIT_AVAILABLE = False
    logger.warning("Unified profit vectorization system not available")

# Strategy Logic
try:
    from .strategy_logic import StrategyLogic

    STRATEGY_LOGIC_AVAILABLE = True
except ImportError:
    STRATEGY_LOGIC_AVAILABLE = False
    logger.warning("Strategy logic not available")

# Unified Mathematical Core
try:
    from .unified_mathematical_core import UnifiedMathematicalCore

    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False
    logger.warning("Unified mathematical core not available")

# Vectorized Profit Orchestrator
try:
    from .vectorized_profit_orchestrator import VectorizedProfitOrchestrator

    VECTORIZED_PROFIT_AVAILABLE = True
except ImportError:
    VECTORIZED_PROFIT_AVAILABLE = False
    logger.warning("Vectorized profit orchestrator not available")

# Consolidated Utilities
try:
    from .consolidated_math_utils import (
        ConsolidatedMathUtils,
        create_consolidated_math_utils,
    )
    from .consolidated_system_utils import (
        ConsolidatedSystemUtils,
        create_consolidated_system_utils,
    )

    CONSOLIDATED_UTILS_AVAILABLE = True
except ImportError:
    CONSOLIDATED_UTILS_AVAILABLE = False
    logger.warning("Consolidated utilities not available")

__all__ = [
    "MathConfigManager",
    "MathResultCache",
    "MathOrchestrator",
    "MATH_INFRASTRUCTURE_AVAILABLE",
    "CoreUtilities",
    "CORE_UTILITIES_AVAILABLE",
    "BioCellularIntegration",
    "create_bio_cellular_integration",
    "BIO_CELLULAR_AVAILABLE",
    "ProfitOptimizationEngine",
    "PROFIT_OPTIMIZATION_AVAILABLE",
    "QuantumMathematicalBridge",
    "QUANTUM_BRIDGE_AVAILABLE",
    "EntropyMath",
    "ENTROPY_MATH_AVAILABLE",
    "TensorScoreUtils",
    "TENSOR_SCORE_AVAILABLE",
    "DLTWaveformEngine",
    "DLT_WAVEFORM_AVAILABLE",
    "AdvancedTensorAlgebra",
    "ADVANCED_TENSOR_AVAILABLE",
    "UnifiedProfitVectorizationSystem",
    "UNIFIED_PROFIT_AVAILABLE",
    "StrategyLogic",
    "STRATEGY_LOGIC_AVAILABLE",
    "UnifiedMathematicalCore",
    "UNIFIED_MATH_AVAILABLE",
    "VectorizedProfitOrchestrator",
    "VECTORIZED_PROFIT_AVAILABLE",
    "ConsolidatedMathUtils",
    "ConsolidatedSystemUtils",
    "create_consolidated_math_utils",
    "create_consolidated_system_utils",
    "CONSOLIDATED_UTILS_AVAILABLE",
]


def get_system_status() -> Dict[str, Any]:
    """Get the status of all core systems."""
    return {
        "math_infrastructure": MATH_INFRASTRUCTURE_AVAILABLE,
        "core_utilities": CORE_UTILITIES_AVAILABLE,
        "bio_cellular": BIO_CELLULAR_AVAILABLE,
        "profit_optimization": PROFIT_OPTIMIZATION_AVAILABLE,
        "quantum_bridge": QUANTUM_BRIDGE_AVAILABLE,
        "entropy_math": ENTROPY_MATH_AVAILABLE,
        "tensor_score": TENSOR_SCORE_AVAILABLE,
        "dlt_waveform": DLT_WAVEFORM_AVAILABLE,
        "advanced_tensor": ADVANCED_TENSOR_AVAILABLE,
        "unified_profit": UNIFIED_PROFIT_AVAILABLE,
        "strategy_logic": STRATEGY_LOGIC_AVAILABLE,
        "unified_math": UNIFIED_MATH_AVAILABLE,
        "vectorized_profit": VECTORIZED_PROFIT_AVAILABLE,
        "consolidated_utils": CONSOLIDATED_UTILS_AVAILABLE,
    }


def create_core_system(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a complete core system with all available components."""
    system = {}

    if MATH_INFRASTRUCTURE_AVAILABLE:
        system["math_config"] = MathConfigManager()
        system["math_cache"] = MathResultCache()
        system["math_orchestrator"] = MathOrchestrator()

    if CORE_UTILITIES_AVAILABLE:
        system["core_utilities"] = CoreUtilities()

    if BIO_CELLULAR_AVAILABLE:
        system["bio_cellular"] = create_bio_cellular_integration(config)

    if PROFIT_OPTIMIZATION_AVAILABLE:
        system["profit_optimization"] = ProfitOptimizationEngine(config)

    if QUANTUM_BRIDGE_AVAILABLE:
        system["quantum_bridge"] = QuantumMathematicalBridge()

    if ENTROPY_MATH_AVAILABLE:
        system["entropy_math"] = EntropyMath(config)

    if TENSOR_SCORE_AVAILABLE:
        system["tensor_score"] = TensorScoreUtils(config)

    if DLT_WAVEFORM_AVAILABLE:
        system["dlt_waveform"] = DLTWaveformEngine(config)

    if ADVANCED_TENSOR_AVAILABLE:
        system["advanced_tensor"] = AdvancedTensorAlgebra(config)

    if UNIFIED_PROFIT_AVAILABLE:
        system["unified_profit"] = UnifiedProfitVectorizationSystem(config)

    if STRATEGY_LOGIC_AVAILABLE:
        system["strategy_logic"] = StrategyLogic(config)

    if UNIFIED_MATH_AVAILABLE:
        system["unified_math"] = UnifiedMathematicalCore(config)

    if VECTORIZED_PROFIT_AVAILABLE:
        system["vectorized_profit"] = VectorizedProfitOrchestrator(config)

    if CONSOLIDATED_UTILS_AVAILABLE:
        system["consolidated_math"] = create_consolidated_math_utils()
        system["consolidated_system"] = create_consolidated_system_utils()

    logger.info(f"✅ Core system created with {len(system)} components")
    return system


def initialize_core_system(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the complete core system."""
    try:
        system = create_core_system(config)

        for name, component in system.items():
            if hasattr(component, "activate"):
                if component.activate():
                    logger.info(f"✅ {name} activated")
                else:
                    logger.warning(f"⚠️ {name} failed to activate")

        logger.info("✅ Core system initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Error initializing core system: {e}")
        return False
