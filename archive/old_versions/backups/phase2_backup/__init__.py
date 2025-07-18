#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Backup - Clean Mathematical Foundation
=============================================

This module provides clean, production-ready mathematical implementations
for the Schwabot trading system.

Key Components:
- CleanMathFoundation: Core mathematical operations
- CleanProfitVectorization: Profit calculation and vectorization
- CleanTradingPipeline: Trading pipeline with mathematical integration
- OrbitalBRAINSystem: Advanced brain system for decision making
- AlgorithmicPortfolioBalancer: Portfolio balancing algorithms
- BTC/USDC Trading Integration: Specialized BTC/USDC trading
- GPU System State Profiler: GPU performance profiling
- GPU DNA Auto-Detection: GPU capability detection
- GPU Shader Integration: GPU-accelerated computations
"""

# Standard library imports
import logging

# Clean Math Foundation
try:
    from .clean_math_foundation import (
        BitPhase,
        CleanMathFoundation,
        MathOperation,
        ThermalState,
        create_math_foundation,
        quick_calculation,
    )
    CLEAN_MATH_AVAILABLE = True
except ImportError:
    CLEAN_MATH_AVAILABLE = False

# Clean Profit Vectorization
try:
    from .clean_profit_vectorization import (
        CleanProfitVectorization,
        ProfitVector,
        VectorizationMode,
        create_profit_vectorizer,
    )
    CLEAN_PROFIT_AVAILABLE = True
except ImportError:
    CLEAN_PROFIT_AVAILABLE = False

# Orbital Brain System
try:
    from .orbital_shell_brain_system import OrbitalBRAINSystem, OrbitalShell
    ORBITAL_BRAIN_AVAILABLE = True
except ImportError:
    ORBITAL_BRAIN_AVAILABLE = False

# Clean Trading Pipeline
try:
    from .clean_trading_pipeline import (
        CleanTradingPipeline,
        MarketData,
        StrategyBranch,
        TradingDecision,
        create_trading_pipeline,
        run_trading_simulation,
    )
    CLEAN_PIPELINE_AVAILABLE = True
except ImportError:
    CLEAN_PIPELINE_AVAILABLE = False

# Algorithmic Portfolio Balancer
try:
    from .algorithmic_portfolio_balancer import (
        AlgorithmicPortfolioBalancer,
        AssetAllocation,
        RebalancingStrategy,
        create_portfolio_balancer,
    )
    PORTFOLIO_BALANCER_AVAILABLE = True
except ImportError:
    PORTFOLIO_BALANCER_AVAILABLE = False

# BTC/USDC Trading Integration
try:
    from .btc_usdc_trading_integration import BTCUSDCTradingConfig, BTCUSDCTradingIntegration, create_btc_usdc_integration
    BTC_USDC_INTEGRATION_AVAILABLE = True
except ImportError:
    BTC_USDC_INTEGRATION_AVAILABLE = False

# GPU System State Profiler Integration
try:
    from .system_state_profiler import (
        CPUProfile,
        CPUTier,
        GPUProfile,
        GPUTier,
        SystemProfile,
        SystemStateProfiler,
        SystemTier,
        create_system_profiler,
        get_gpu_shader_config,
        get_system_profile,
    )
    SYSTEM_PROFILER_AVAILABLE = True
except ImportError:
    SYSTEM_PROFILER_AVAILABLE = False

# GPU DNA Auto-Detection
try:
    from .gpu_dna_autodetect import (
        GPUDNAAutoDetect,
        ShaderConfig,
        create_gpu_dna_detector,
        detect_gpu_dna,
        get_cosine_similarity_config,
        run_gpu_fit_test,
    )
    GPU_DNA_AVAILABLE = True
except ImportError:
    GPU_DNA_AVAILABLE = False

# GPU Shader Integration
try:
    from .gpu_shader_integration import (
        GPUShaderIntegration,
        ShaderProgramConfig,
        compute_strategy_similarities_gpu,
        create_gpu_shader_integration,
    )
    GPU_SHADER_INTEGRATION_AVAILABLE = True
except ImportError:
    GPU_SHADER_INTEGRATION_AVAILABLE = False

# Add new import
try:
    from config.schwabot_adaptive_config_manager import SchwabotAdaptiveConfigManager, create_adaptive_config_manager
    ADAPTIVE_CONFIG_AVAILABLE = True
except ImportError:
    ADAPTIVE_CONFIG_AVAILABLE = False

# Core exports - only clean implementations
__all__ = [
    # Clean implementations (recommended)
    "CleanMathFoundation",
    "MathOperation",
    "ThermalState",
    "BitPhase",
    "create_math_foundation",
    "quick_calculation",
    "CleanProfitVectorization",
    "VectorizationMode",
    "ProfitVector",
    "create_profit_vectorizer",
    "CleanTradingPipeline",
    "MarketData",
    "TradingDecision",
    "StrategyBranch",
    "create_trading_pipeline",
    "run_trading_simulation",
    # New orbital brain components
    "OrbitalBRAINSystem",
    "OrbitalShell",
    # Portfolio balancing components
    "AlgorithmicPortfolioBalancer",
    "RebalancingStrategy",
    "AssetAllocation",
    "create_portfolio_balancer",
    # BTC/USDC integration components
    "BTCUSDCTradingIntegration",
    "BTCUSDCTradingConfig",
    "create_btc_usdc_integration",
    # GPU System State Profiler components
    "SystemStateProfiler",
    "SystemProfile",
    "CPUProfile",
    "GPUProfile",
    "SystemTier",
    "CPUTier",
    "GPUTier",
    "create_system_profiler",
    "get_system_profile",
    "get_gpu_shader_config",
    # GPU DNA Auto-Detection components
    "GPUDNAAutoDetect",
    "ShaderConfig",
    "create_gpu_dna_detector",
    "detect_gpu_dna",
    "get_cosine_similarity_config",
    "run_gpu_fit_test",
    # GPU Shader Integration components
    "GPUShaderIntegration",
    "ShaderProgramConfig",
    "create_gpu_shader_integration",
    "compute_strategy_similarities_gpu",
    # Availability flags
    "CLEAN_MATH_AVAILABLE",
    "CLEAN_PROFIT_AVAILABLE",
    "CLEAN_PIPELINE_AVAILABLE",
    "ORBITAL_BRAIN_AVAILABLE",
    "PORTFOLIO_BALANCER_AVAILABLE",
    "BTC_USDC_INTEGRATION_AVAILABLE",
    "SYSTEM_PROFILER_AVAILABLE",
    "GPU_DNA_AVAILABLE",
    "GPU_SHADER_INTEGRATION_AVAILABLE",
    "ADAPTIVE_CONFIG_AVAILABLE",
    # Utility functions
]

# Add to __all__ if available
if ADAPTIVE_CONFIG_AVAILABLE:
    __all__ += [
        'SchwabotAdaptiveConfigManager',
        'create_adaptive_config_manager'
    ]

def get_system_status():
    """Get status of all available systems."""
    return {
        'clean_math': CLEAN_MATH_AVAILABLE,
        'clean_profit': CLEAN_PROFIT_AVAILABLE,
        'clean_pipeline': CLEAN_PIPELINE_AVAILABLE,
        'orbital_brain': ORBITAL_BRAIN_AVAILABLE,
        'portfolio_balancer': PORTFOLIO_BALANCER_AVAILABLE,
        'btc_usdc_integration': BTC_USDC_INTEGRATION_AVAILABLE,
        'system_profiler': SYSTEM_PROFILER_AVAILABLE,
        'gpu_dna': GPU_DNA_AVAILABLE,
        'gpu_shader': GPU_SHADER_INTEGRATION_AVAILABLE,
        'adaptive_config': ADAPTIVE_CONFIG_AVAILABLE,
    }

def create_clean_trading_system(initial_capital=100000.0, enable_gpu_acceleration=True):
    """Create a complete clean trading system."""
    try:
        if CLEAN_MATH_AVAILABLE and CLEAN_PIPELINE_AVAILABLE:
            # Create math foundation
            math_foundation = create_math_foundation()
            
            # Create trading pipeline
            pipeline = create_trading_pipeline(initial_capital=initial_capital)
            
            # Create profit vectorizer
            if CLEAN_PROFIT_AVAILABLE:
                profit_vectorizer = create_profit_vectorizer()
            else:
                profit_vectorizer = None
            
            return {
                'math_foundation': math_foundation,
                'trading_pipeline': pipeline,
                'profit_vectorizer': profit_vectorizer,
                'gpu_acceleration': enable_gpu_acceleration
            }
        else:
            raise ImportError("Required clean components not available")
    except Exception as e:
        logging.error(f"Failed to create clean trading system: {e}")
        return None

def setup_cross_platform_compatibility():
    """Setup cross-platform compatibility."""
    try:
        # Setup system profiler if available
        if SYSTEM_PROFILER_AVAILABLE:
            profiler = create_system_profiler()
            return profiler.get_system_profile()
        else:
            return {'platform': 'unknown', 'tier': 'basic'}
    except Exception as e:
        logging.error(f"Failed to setup cross-platform compatibility: {e}")
        return {'platform': 'unknown', 'tier': 'basic'}

def initialize_adaptive_configuration(initial_capital=100000.0):
    """Initialize adaptive configuration system."""
    try:
        if ADAPTIVE_CONFIG_AVAILABLE:
            config_manager = create_adaptive_config_manager(initial_capital=initial_capital)
            return config_manager
        else:
            logging.warning("Adaptive configuration not available")
            return None
    except Exception as e:
        logging.error(f"Failed to initialize adaptive configuration: {e}")
        return None
