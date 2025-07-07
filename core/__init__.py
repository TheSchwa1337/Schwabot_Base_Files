"""
Core module for Schwabot trading system.

This module provides clean, error-free implementations of the core
mathematical and trading components.
"""

# -*- coding: utf-8 -*-

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

try:
    from .orbital_shell_brain_system import OrbitalBRAINSystem, OrbitalShell
    ORBITAL_BRAIN_AVAILABLE = True
except ImportError:
    ORBITAL_BRAIN_AVAILABLE = False

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

try:
    from .algorithmic_portfolio_balancer import (
        AlgorithmicPortfolioBalancer,
        RebalancingStrategy,
        AssetAllocation,
        create_portfolio_balancer,
    )
    PORTFOLIO_BALANCER_AVAILABLE = True
except ImportError:
    PORTFOLIO_BALANCER_AVAILABLE = False

try:
    from .btc_usdc_trading_integration import (
        BTCUSDCTradingIntegration,
        BTCUSDCTradingConfig,
        create_btc_usdc_integration,
    )
    BTC_USDC_INTEGRATION_AVAILABLE = True
except ImportError:
    BTC_USDC_INTEGRATION_AVAILABLE = False

# GPU System State Profiler Integration
try:
    from .system_state_profiler import (
        SystemStateProfiler,
        SystemProfile,
        CPUProfile,
        GPUProfile,
        SystemTier,
        CPUTier,
        GPUTier,
        create_system_profiler,
        get_system_profile,
        get_gpu_shader_config
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
        run_gpu_fit_test
    )
    GPU_DNA_AVAILABLE = True
except ImportError:
    GPU_DNA_AVAILABLE = False

# GPU Shader Integration
try:
    from .gpu_shader_integration import (
        GPUShaderIntegration,
        ShaderProgramConfig,
        create_gpu_shader_integration,
        compute_strategy_similarities_gpu
    )
    GPU_SHADER_INTEGRATION_AVAILABLE = True
except ImportError:
    GPU_SHADER_INTEGRATION_AVAILABLE = False

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
    # Utility functions
    "get_system_status",
    "create_clean_trading_system",
    "initialize_gpu_system",
]


def get_system_status():
    """Get the status of all system components."""
    return {
        "clean_implementations": {
            "math_foundation": CLEAN_MATH_AVAILABLE,
            "profit_vectorization": CLEAN_PROFIT_AVAILABLE,
            "trading_pipeline": CLEAN_PIPELINE_AVAILABLE,
            "orbital_brain_system": ORBITAL_BRAIN_AVAILABLE,
            "portfolio_balancer": PORTFOLIO_BALANCER_AVAILABLE,
            "btc_usdc_integration": BTC_USDC_INTEGRATION_AVAILABLE,
        },
        "gpu_system": {
            "system_profiler": SYSTEM_PROFILER_AVAILABLE,
            "gpu_dna_detection": GPU_DNA_AVAILABLE,
            "shader_integration": GPU_SHADER_INTEGRATION_AVAILABLE,
        },
        "system_operational": (
            CLEAN_MATH_AVAILABLE and 
            CLEAN_PROFIT_AVAILABLE and 
            CLEAN_PIPELINE_AVAILABLE and
            PORTFOLIO_BALANCER_AVAILABLE and
            BTC_USDC_INTEGRATION_AVAILABLE
        ),
        "gpu_acceleration_available": (
            SYSTEM_PROFILER_AVAILABLE and
            GPU_DNA_AVAILABLE and
            GPU_SHADER_INTEGRATION_AVAILABLE
        ),
    }


def initialize_gpu_system():
    """
    Initialize the GPU acceleration system for Schwabot.
    
    Returns:
        Dictionary with GPU system components and status
    """
    gpu_system = {
        "system_profile": None,
        "gpu_dna_profile": None,
        "shader_integration": None,
        "initialization_status": {
            "profiler": False,
            "dna_detection": False,
            "shader_integration": False
        }
    }
    
    try:
        # Initialize system profiler
        if SYSTEM_PROFILER_AVAILABLE:
            gpu_system["system_profile"] = get_system_profile()
            gpu_system["initialization_status"]["profiler"] = True
        
        # Initialize GPU DNA detection
        if GPU_DNA_AVAILABLE:
            gpu_system["gpu_dna_profile"] = detect_gpu_dna()
            gpu_system["initialization_status"]["dna_detection"] = True
        
        # Initialize shader integration
        if GPU_SHADER_INTEGRATION_AVAILABLE:
            gpu_system["shader_integration"] = create_gpu_shader_integration()
            gpu_system["initialization_status"]["shader_integration"] = True
        
        return gpu_system
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"GPU system initialization failed: {e}")
        return gpu_system


def create_clean_trading_system(initial_capital=100000.0, enable_gpu_acceleration=True):
    """
    Create a complete clean trading system with all components.

    Args:
        initial_capital: Initial capital for the trading system
        enable_gpu_acceleration: Whether to enable GPU acceleration features

    Returns:
        Dictionary with all initialized components
    """
    if not (CLEAN_MATH_AVAILABLE and CLEAN_PROFIT_AVAILABLE and CLEAN_PIPELINE_AVAILABLE):
        raise ImportError("Clean implementations not available")

    # Base configuration
    config = {
        "portfolio_config": {
            "rebalancing_strategy": "phantom_adaptive",
            "rebalance_threshold": 0.05,
            "max_rebalance_frequency": 3600,
        },
        "btc_usdc_config": {
            "symbol": "BTC/USDC",
            "base_order_size": 0.001,
            "max_order_size": 0.01,
            "enable_portfolio_balancing": True,
        },
        "exchange_config": {
            "exchange": "binance",
            "sandbox": True,
        }
    }

    system = {
        "math_foundation": create_math_foundation(),
        "profit_vectorizer": create_profit_vectorizer(),
        "trading_pipeline": create_trading_pipeline(initial_capital=initial_capital),
    }

    if ORBITAL_BRAIN_AVAILABLE:
        system["orbital_brain"] = OrbitalBRAINSystem()

    if PORTFOLIO_BALANCER_AVAILABLE:
        system["portfolio_balancer"] = create_portfolio_balancer(config)

    if BTC_USDC_INTEGRATION_AVAILABLE:
        system["btc_usdc_integration"] = create_btc_usdc_integration(config)

    # Initialize GPU acceleration if requested and available
    if enable_gpu_acceleration:
        gpu_system = initialize_gpu_system()
        system["gpu_system"] = gpu_system
        
        # Log GPU system status
        import logging
        logger = logging.getLogger(__name__)
        if gpu_system["initialization_status"]["shader_integration"]:
            logger.info("🚀 GPU-accelerated trading system initialized")
            if gpu_system["system_profile"]:
                profile = gpu_system["system_profile"]
                logger.info(f"🎮 GPU: {profile.gpu.renderer} ({profile.gpu.gpu_tier.value})")
                logger.info(f"📊 Matrix Size: {profile.gpu.max_matrix_size}x{profile.gpu.max_matrix_size}")
        else:
            logger.info("🔄 CPU-only trading system initialized (GPU acceleration unavailable)")

    return system
