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
from .import_resolver import ImportResolver

# Type definitions - specific imports instead of wildcard
from .type_defs import (
    QuantumState,
    EnergyLevel,
    Entropy,
    WaveFunction,
    EnergyOperator,
    RecursionDepth,
    RecursionStack,
    Tensor,
    Complex,
    Vector,
    Matrix,
    QuantumHash,
    TimeSlot,
    StrategyId,
    PriceState,
    DriftCoefficient
)

from .ghost_pipeline import (
    GhostPipeline,
    ghost_validator_pipeline,
)  # new stealth helpers

from .ghost_trigger import ghost_trigger  # new stealth helper

from .ghost_swap_vector import (
    ghost_swap_vector,
)

from .ghost_router import GhostRouter, ghost_router

from .ghost_memory import GhostMemory, store_ghost_hash, last_profitable_hash

from .vector_state_mapper import map_glyph_to_state

from .ghost_profit_tracker import ProfitTracker, register_profit, profit_summary
from .drift_compensator import compute_drift_vector

from .glyph_phase_anchor import phase_anchor_index, glyph_active_for_tick
from .glyph_hysteresis import HysteresisField

from .pool_volume_translator import translate_news_to_pool_vector

from .memory_drift_corrector import drift_score, relink_required

from .ghost_conditionals import ghost_route_activation
from .phantom_entry_logic import phantom_entry_probability
from .phantom_exit_logic import phantom_exit_score
from .ghost_news_glyph_map import news_to_glyph_weight
from .ghost_news_vectorizer import vectorize_news
from .ghost_strategy_matrix import build_strategy_matrix
from .lantern_trigger import lantern_trigger
from .lantern_vector_memory import LanternMemory
from .lantern_hash_echo import lantern_hash_echo

from .exec_packet import ExecPacket
from .compute_ghost_route import compute_ghost_route
from .ghost_phase_integrator import GhostPhasePacket, compute_ghost_phase_packet
from .profit_feedback_loop import profit_feedback_delta

from .phantom_memory import PhantomMemory, GhostEvent, compute_memory_recall
from .entropy_flattener import entropy_flatten, compute_second_derivative, adaptive_smooth
from .news_sentiment_interpreter import interpret_news_sentiment, weight_sentiment_events
from .glyph_vector_executor import GlyphInstruction, execute_glyph_vectors

from .hash_tick_synchronizer import compute_tick_hash, sync_probability, hash_match_check
from .glyph_math_core import glyph_determinant, glyph_matrix, glyph_psi, glyph_tensor
from .news_quant_field import quantize_news, news_gradient, news_psi, news_spectral_field
from .btc_vector_aggregator import btc_vector, btc_eta, btc_xi, btc_spectral_aggregate
from .usdc_position_manager import usdc_position, usdc_trading, usdc_sigma, usdc_optimal_time

from .zpe_core_matrix import zpe_psi, zpe_phi, zpe_xi, zpe_g

from .ghost_strategy_integrator import (
    GhostStrategyIntegrator,
    StrategyTriggerPipeline,
    FerrisWheelActivator,
    CoreVectorProcessor,
    CoreVectorData,
    StrategyExecutionPacket,
)

from .strategy_mapper import StrategyMapper, map_strategy
from .profit_cycle_allocator import ProfitCycleAllocator, allocate_profit_cycle

from .recursive_strategy_router import RecursiveStrategyRouter, route_strategy
from .lantern_trigger_validator import LanternTriggerValidator, validate_lantern_trigger
from .ghost_memory_router import GhostMemoryRouter
from .phantom_profit_tracker import ProfitTracker as PhantomProfitTracker, register_profit as register_phantom_profit, profit_summary as phantom_profit_summary

from .phantom_price_vector_synchronizer import (
    PhantomPriceSynchronizer,
    compute_phantom_velocity,
    synchronize_price_vectors,
)
from .conditional_glyph_feedback_loop import (
    ConditionalGlyphFeedback,
    compute_news_flow_gradient,
    apply_feedback_loop,
)
from .zbe_position_tracker import (
    ZBEPositionTracker,
    compute_zalgo_evolution,
    track_position_state,
)
from .btc_usdc_router_relay import (
    BTCUSDCRouterRelay,
    compute_ghost_triggers,
    route_btc_usdc_flow,
)
from .entry_exit_vector_analyzer import (
    EntryExitVectorAnalyzer,
    compute_routing_elasticity,
    analyze_entry_exit_vectors,
)
from .profit_echo_velocity_driver import (
    ProfitEchoVelocityDriver,
    compute_volatility_burst_memory,
    drive_profit_echo,
)

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
    
    # Type definitions
    'QuantumState',
    'EnergyLevel',
    'Entropy',
    'WaveFunction',
    'EnergyOperator',
    'RecursionDepth',
    'RecursionStack',
    'Tensor',
    'Complex',
    'Vector',
    'Matrix',
    'QuantumHash',
    'TimeSlot',
    'StrategyId',
    'PriceState',
    'DriftCoefficient',
    'GhostPipeline',
    'ghost_validator_pipeline',
    'ghost_trigger',
    'ghost_swap_vector',
    'GhostRouter',
    'ghost_router',
    'GhostMemory',
    'store_ghost_hash',
    'last_profitable_hash',
    'map_glyph_to_state',
    'ProfitTracker',
    'register_profit',
    'profit_summary',
    'compute_drift_vector',
    'phase_anchor_index',
    'glyph_active_for_tick',
    'HysteresisField',
    'translate_news_to_pool_vector',
    'drift_score',
    'relink_required',
    'ghost_route_activation',
    'phantom_entry_probability',
    'phantom_exit_score',
    'news_to_glyph_weight',
    'vectorize_news',
    'build_strategy_matrix',
    'lantern_trigger',
    'LanternMemory',
    'lantern_hash_echo',
    'ExecPacket',
    'compute_ghost_route',
    'GhostPhasePacket',
    'compute_ghost_phase_packet',
    'profit_feedback_delta',
    'PhantomMemory',
    'GhostEvent',
    'compute_memory_recall',
    'entropy_flatten',
    'compute_second_derivative',
    'adaptive_smooth',
    'interpret_news_sentiment',
    'weight_sentiment_events',
    'GlyphInstruction',
    'execute_glyph_vectors',
    'compute_tick_hash',
    'sync_probability',
    'hash_match_check',
    'glyph_determinant',
    'glyph_matrix',
    'glyph_psi',
    'glyph_tensor',
    'quantize_news',
    'news_gradient',
    'news_psi',
    'news_spectral_field',
    'btc_vector',
    'btc_eta',
    'btc_xi',
    'btc_spectral_aggregate',
    'usdc_position',
    'usdc_trading',
    'usdc_sigma',
    'usdc_optimal_time',
    'zpe_psi',
    'zpe_phi',
    'zpe_xi',
    'zpe_g',
    'GhostStrategyIntegrator',
    'StrategyTriggerPipeline',
    'FerrisWheelActivator',
    'CoreVectorProcessor',
    'CoreVectorData',
    'StrategyExecutionPacket',
    'StrategyMapper',
    'map_strategy',
    'ProfitCycleAllocator',
    'allocate_profit_cycle',
    'RecursiveStrategyRouter',
    'route_strategy',
    'LanternTriggerValidator',
    'validate_lantern_trigger',
    'GhostMemoryRouter',
    'PhantomProfitTracker',
    'register_phantom_profit',
    'phantom_profit_summary',
    'PhantomPriceSynchronizer',
    'compute_phantom_velocity',
    'synchronize_price_vectors',
    'ConditionalGlyphFeedback',
    'compute_news_flow_gradient',
    'apply_feedback_loop',
    'ZBEPositionTracker',
    'compute_zalgo_evolution',
    'track_position_state',
    'BTCUSDCRouterRelay',
    'compute_ghost_triggers',
    'route_btc_usdc_flow',
    'EntryExitVectorAnalyzer',
    'compute_routing_elasticity',
    'analyze_entry_exit_vectors',
    'ProfitEchoVelocityDriver',
    'compute_volatility_burst_memory',
    'drive_profit_echo',
]
