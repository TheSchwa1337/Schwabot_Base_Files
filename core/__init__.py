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

# Core System Components - Only what's actually implemented
try:
    from .schwabot_core_system import SchwabotCoreSystem, create_system_instance, get_system_instance
    CORE_SYSTEM_AVAILABLE = True
except ImportError:
    CORE_SYSTEM_AVAILABLE = False

# Mathematical Infrastructure
try:
    from .advanced_tensor_algebra import AdvancedTensorAlgebra
    from .bitmap_hash_folding import BitmapHashFolding
    from .enhanced_mathematical_core import EnhancedMathematicalCore
    from .entropy_decay_system import EntropyDecaySystem
    from .entropy_drift_engine import EntropyDriftEngine
    from .entropy_math import EntropyMath
    from .math_cache import MathResultCache
    from .math_config_manager import MathConfigManager
    from .math_orchestrator import MathOrchestrator
    from .mathematical_framework_integrator import MathematicalFrameworkIntegrator
    from .orbital_energy_quantizer import OrbitalEnergyQuantizer
    from .quantum_mathematical_bridge import QuantumState
    from .symbolic_math_interface import SymbolicMathInterface
    from .symbolic_registry import SymbolicRegistry
    from .tcell_survival_engine import TCellSurvivalEngine
    from .tensor_score_utils import TensorScoreResult
    from .two_gram_detector import TwoGramDetector
    from .vault_orbital_bridge import VaultOrbitalBridge
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False

# Trading Components
try:
    from .btc_usdc_trading_engine import BTCTradingEngine
    from .ccxt_trading_executor import CCXTTradingExecutor
    from .enhanced_ccxt_trading_engine import EnhancedCCXTTradingEngine
    from .fill_handler import FillHandler
    from .profit_feedback_engine import ProfitFeedbackEngine
    from .profit_optimization_engine import ProfitOptimizationEngine
    from .real_multi_exchange_trader import RealMultiExchangeTrader
    from .risk_manager import RiskManager
    from .secure_exchange_manager import SecureExchangeManager
    from .unified_btc_trading_pipeline import UnifiedBTCTradingPipeline
    from .unified_pipeline_manager import UnifiedPipelineManager
    TRADING_COMPONENTS_AVAILABLE = True
except ImportError:
    TRADING_COMPONENTS_AVAILABLE = False

# Strategy Components
try:
    from .quad_bit_strategy_array import QuadBitStrategyArray
    from .registry_strategy import RegistryStrategy
    from .strategy.strategy_executor import StrategyExecutor
    from .strategy.strategy_loader import StrategyLoader
    from .strategy_bit_mapper import StrategyBitMapper
    STRATEGY_COMPONENTS_AVAILABLE = True
except ImportError:
    STRATEGY_COMPONENTS_AVAILABLE = False

# Market Data and Execution
try:
    from .order_book_analyzer import WallType as OrderBookAnalyzer
    from .order_book_manager import OrderBookManager
    from .unified_market_data_pipeline import DataSource
    from .unified_trade_router import UnifiedTradeRouter
    MARKET_DATA_AVAILABLE = True
except ImportError:
    MARKET_DATA_AVAILABLE = False

# Portfolio and Tracking
try:
    from .portfolio_tracker import PositionType as PortfolioTracker
    from .registry_backtester import RegistryBacktester
    PORTFOLIO_COMPONENTS_AVAILABLE = True
except ImportError:
    PORTFOLIO_COMPONENTS_AVAILABLE = False

# Registry and Storage
try:
    from .phantom_detector import PhantomDetector
    from .phantom_logger import PhantomLogger
    from .phantom_registry import PhantomRegistry
    from .soulprint_registry import SoulprintRegistry
    from .vector_registry import VectorRegistry
    REGISTRY_COMPONENTS_AVAILABLE = True
except ImportError:
    REGISTRY_COMPONENTS_AVAILABLE = False

# AI and Processing
try:
    from .visual_decision_engine import VisualDecisionEngine
    AI_COMPONENTS_AVAILABLE = True
except ImportError:
    AI_COMPONENTS_AVAILABLE = False

# Integration and Bridges
try:
    from .crwf_crlf_integration import CRWFCrlfIntegration
    from .schwafit_core import SchwafitCore
    from .unified_component_bridge import BridgeMode
    INTEGRATION_COMPONENTS_AVAILABLE = True
except ImportError:
    INTEGRATION_COMPONENTS_AVAILABLE = False

# Advanced Systems
try:
    from .fractal_core import FractalCore
    from .fractal_memory_tracker import FractalMemoryTracker
    from .ghost_core import GhostCore
    ADVANCED_SYSTEMS_AVAILABLE = True
except ImportError:
    ADVANCED_SYSTEMS_AVAILABLE = False

# GPU and Hardware
try:
    from .gpu_dna_autodetect import GPUDNAAutodetect
    from .gpu_shader_integration import GPUShaderIntegration
    GPU_COMPONENTS_AVAILABLE = True
except ImportError:
    GPU_COMPONENTS_AVAILABLE = False

# Define what's actually available for import
__all__ = [
    # Core System
    "SchwabotCoreSystem",
    "get_system_instance", 
    "create_system_instance",
    "CORE_SYSTEM_AVAILABLE",
    
    # Math Infrastructure
    "MathConfigManager",
    "MathResultCache", 
    "MathOrchestrator",
    "EnhancedMathematicalCore",
    "MathematicalFrameworkIntegrator",
    "TCellSurvivalEngine",
    "EntropyMath",
    "TensorScoreResult",
    "AdvancedTensorAlgebra",
    "SymbolicRegistry",
    "BitmapHashFolding",
    "OrbitalEnergyQuantizer",
    "EntropyDriftEngine",
    "VaultOrbitalBridge",
    "EntropyDecaySystem",
    "TwoGramDetector",
    "SymbolicMathInterface",
    "QuantumState",
    "MATH_INFRASTRUCTURE_AVAILABLE",
    
    # Trading Components
    "BTCTradingEngine",
    "RiskManager",
    "SecureExchangeManager",
    "UnifiedPipelineManager",
    "UnifiedBTCTradingPipeline",
    "ProfitOptimizationEngine",
    "RealMultiExchangeTrader",
    "ProfitFeedbackEngine",
    "CCXTTradingExecutor",
    "FillHandler",
    "EnhancedCCXTTradingEngine",
    "TRADING_COMPONENTS_AVAILABLE",
    
    # Strategy Components
    "StrategyLoader",
    "StrategyExecutor",
    "RegistryStrategy",
    "QuadBitStrategyArray",
    "StrategyBitMapper",
    "STRATEGY_COMPONENTS_AVAILABLE",
    
    # Market Data
    "DataSource",
    "UnifiedTradeRouter",
    "OrderBookManager",
    "OrderBookAnalyzer",
    "MARKET_DATA_AVAILABLE",
    
    # Portfolio
    "PortfolioTracker",
    "RegistryBacktester",
    "PORTFOLIO_COMPONENTS_AVAILABLE",
    
    # Registry
    "SoulprintRegistry",
    "VectorRegistry",
    "PhantomRegistry",
    "PhantomLogger",
    "PhantomDetector",
    "REGISTRY_COMPONENTS_AVAILABLE",
    
    # AI
    "VisualDecisionEngine",
    "AI_COMPONENTS_AVAILABLE",
    
    # Integration
    "BridgeMode",
    "CRWFCrlfIntegration",
    "SchwafitCore",
    "INTEGRATION_COMPONENTS_AVAILABLE",
    
    # Advanced Systems
    "GhostCore",
    "FractalCore",
    "FractalMemoryTracker",
    "ADVANCED_SYSTEMS_AVAILABLE",
    
    # GPU
    "GPUShaderIntegration",
    "GPUDNAAutodetect",
    "GPU_COMPONENTS_AVAILABLE",
]


def get_system_status() -> Dict[str, Any]:
    """Get the status of all core systems."""
    return {
        "core_system": CORE_SYSTEM_AVAILABLE,
        "math_infrastructure": MATH_INFRASTRUCTURE_AVAILABLE,
        "trading_components": TRADING_COMPONENTS_AVAILABLE,
        "strategy_components": STRATEGY_COMPONENTS_AVAILABLE,
        "market_data": MARKET_DATA_AVAILABLE,
        "portfolio_components": PORTFOLIO_COMPONENTS_AVAILABLE,
        "registry_components": REGISTRY_COMPONENTS_AVAILABLE,
        "ai_components": AI_COMPONENTS_AVAILABLE,
        "integration_components": INTEGRATION_COMPONENTS_AVAILABLE,
        "advanced_systems": ADVANCED_SYSTEMS_AVAILABLE,
        "gpu_components": GPU_COMPONENTS_AVAILABLE,
    }


def create_core_system(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a complete core system with all available components."""
    system = {}
    
    if CORE_SYSTEM_AVAILABLE:
        system["core_system"] = SchwabotCoreSystem(config)
    
    if MATH_INFRASTRUCTURE_AVAILABLE:
        system["math_config"] = MathConfigManager()
        system["math_cache"] = MathResultCache()
        system["math_orchestrator"] = MathOrchestrator()
        system["enhanced_math_core"] = EnhancedMathematicalCore()
        system["math_framework"] = MathematicalFrameworkIntegrator()
        system["tcell_survival"] = TCellSurvivalEngine()
        system["entropy_math"] = EntropyMath()
        system["tensor_score"] = TensorScoreResult()
        system["advanced_tensor"] = AdvancedTensorAlgebra()
        system["symbolic_registry"] = SymbolicRegistry()
        system["bitmap_hash"] = BitmapHashFolding()
        system["orbital_energy"] = OrbitalEnergyQuantizer()
        system["entropy_drift"] = EntropyDriftEngine()
        system["vault_orbital"] = VaultOrbitalBridge()
        system["entropy_decay"] = EntropyDecaySystem()
        system["two_gram"] = TwoGramDetector()
        system["symbolic_math"] = SymbolicMathInterface()
        system["quantum_state"] = QuantumState()
    
    if TRADING_COMPONENTS_AVAILABLE:
        system["btc_trading"] = BTCTradingEngine(config)
        system["risk_manager"] = RiskManager()
        system["secure_exchange"] = SecureExchangeManager()
        system["pipeline_manager"] = UnifiedPipelineManager()
        system["btc_pipeline"] = UnifiedBTCTradingPipeline()
        system["profit_optimization"] = ProfitOptimizationEngine()
        system["multi_exchange"] = RealMultiExchangeTrader()
        system["profit_feedback"] = ProfitFeedbackEngine()
        system["ccxt_executor"] = CCXTTradingExecutor()
        system["fill_handler"] = FillHandler()
        system["enhanced_ccxt"] = EnhancedCCXTTradingEngine()
    
    if STRATEGY_COMPONENTS_AVAILABLE:
        system["strategy_loader"] = StrategyLoader()
        system["strategy_executor"] = StrategyExecutor()
        system["registry_strategy"] = RegistryStrategy()
        system["quad_bit_strategy"] = QuadBitStrategyArray()
        system["strategy_bit_mapper"] = StrategyBitMapper()
    
    if MARKET_DATA_AVAILABLE:
        system["market_data"] = DataSource()
        system["trade_router"] = UnifiedTradeRouter()
        system["order_book_manager"] = OrderBookManager()
        system["order_book_analyzer"] = OrderBookAnalyzer()
    
    if PORTFOLIO_COMPONENTS_AVAILABLE:
        system["portfolio_tracker"] = PortfolioTracker()
        system["registry_backtester"] = RegistryBacktester()
    
    if REGISTRY_COMPONENTS_AVAILABLE:
        system["soulprint_registry"] = SoulprintRegistry()
        system["vector_registry"] = VectorRegistry()
        system["phantom_registry"] = PhantomRegistry()
        system["phantom_logger"] = PhantomLogger()
        system["phantom_detector"] = PhantomDetector()
    
    if AI_COMPONENTS_AVAILABLE:
        system["visual_decision"] = VisualDecisionEngine()
    
    if INTEGRATION_COMPONENTS_AVAILABLE:
        system["component_bridge"] = BridgeMode()
        system["crwf_crlf"] = CRWFCrlfIntegration()
        system["schwafit_core"] = SchwafitCore()
    
    if ADVANCED_SYSTEMS_AVAILABLE:
        system["ghost_core"] = GhostCore()
        system["fractal_core"] = FractalCore()
        system["fractal_memory"] = FractalMemoryTracker()
    
    if GPU_COMPONENTS_AVAILABLE:
        system["gpu_shader"] = GPUShaderIntegration()
        system["gpu_dna"] = GPUDNAAutodetect()
    
    return system


def initialize_core_system(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the core system with all available components."""
    try:
        system = create_core_system(config)
        logger.info("✅ Core system initialized successfully")
        logger.info(f"📊 System status: {get_system_status()}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize core system: {e}")
        return False
