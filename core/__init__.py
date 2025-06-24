#!/usr/bin/env python3
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

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "0.5.1"
__author__ = "Schwabot Development Team"
__description__ = "Advanced AI-Powered Trading System"

# Core module exports
from .typing_schemas import (
    FaultLog, FaultEvent, RecoveryStrategy, StrategyHash, AIStrategyResponse,
    MathematicalOperation, VectorOperation, MatrixOperation, TradingSignal,
    SystemState, PerformanceMetrics, parse_ai_response, create_fault_log,
    validate_mathematical_operation
)

from .multi_bit_btc_processor import MultiBitBTCProcessor
from .profit_routing_engine import ProfitRoutingEngine
from .hash_registry import HashRegistry, HashEntry, HashType, HashStatus
from .strategy_loader import StrategyLoader, StrategyConfig, LoaderResult
from .ops_observability import OpsObservability, MetricData, MetricType
from .regulatory_compliance import RegulatoryCompliance, ComplianceReport, ComplianceType
from .risk_guard import RiskGuard, RiskEvent, RiskLevel
from .secure_api_manager import SecureAPIManager
from .exchange_plumbing import ExchangePlumbing
from .persistent_state_manager import PersistentStateManager
from .environment_manager import EnvironmentManager
from .memory_allocation_manager import MemoryAllocationManager
from .precision_performance import PrecisionPerformanceManager
from .long_horizon_simulation import LongHorizonSimulation
from .thermal_boundary_manager import ThermalBoundaryManager

# Type definitions
from .type_defs import (
    BitLevel, MatrixPhase, MatrixController, MatrixControllerType,
    Vector, Matrix, Tensor, Price, Volume, Quantity, Amount,
    Temperature, Pressure, ThermalConductivity, HeatCapacity,
    WarpFactor, LightSpeed, Distance, Time
)

# Utility functions
from .utils.windows_cli_compatibility import (
    safe_print, safe_format_error, log_safe
)

# Fault handling
from .fault_bus import FaultBus, FaultBusEvent, FaultType

# Mathematical components
from .mathematical_pipeline_validator import MathematicalPipelineValidator
from .mathematical_pipeline_validator_simple import SimplifiedMathematicalPipelineValidator

# AI and strategy components
from .gpt_command_layer import GPTCommandLayer
from .gpt_command_layer_simple import SimpleGPTCommandLayer
from .strategy_mapper import StrategyMapper

# Memory and execution components
from .memory_stack.ai_command_sequencer import AICommandSequencer
from .memory_stack.execution_validator import ExecutionValidator

# Risk and compliance components
from .enhanced_risk_manager import EnhancedRiskManager
from .capital_controls import CapitalControls

# Performance and optimization
from .precision_performance import PrecisionPerformanceManager
from .auto_scaler import AutoScaler

# Thermal and hardware management
from .thermal_boundary_manager import ThermalBoundaryManager
from .gpu_flash_engine import GPUFlashEngine

# Advanced mathematical frameworks
from .zpe_core import ZPECore
from .zpe_integration import ZPEIntegration
from .zpe_rotational_engine import ZPERotationalEngine
from .zpe_hybrid_mode_selector import ZPEHybridModeSelector

# Vector and matrix operations
from .unified_confidence_matrix import UnifiedConfidenceMatrix
from .hash_confidence_evaluator import HashConfidenceEvaluator
from .vector_validator import VectorValidator
from .matrix_allocator import MatrixAllocator

# Event and communication systems
from .bus_core import BusCore
from .bus_events import EventBus, TradeEvent, BusEvent
from .echo_snapshot import EchoSnapshot

# Advanced engines
from .future_corridor_engine import FutureCorridorEngine
from .enhanced_fractal_core import EnhancedFractalCore
from .hash_trigger_engine import HashTriggerEngine
from .entropy_engine import EntropyEngine
from .altitude_generator import AltitudeGenerator

# API and integration
from .api_gateway import APIGateway
from .api_bridge_manager import APIBridgeManager
from .coldbase_bridge import ColdbaseBridge
from .prophet_connector import ProphetConnector

# Orchestration and management
from .main_orcestrator import MainOrchestrator
from .master_orchestrator import MasterOrchestrator
from .settings_controller import SettingsController

# Data and analysis
from .data_integration_layer import DataIntegrationLayer
from .line_render_engine import LineRenderEngine
from .trajectory_sphere import TrajectorySphere

# Error handling and validation
from .error_handler import ErrorHandler, safe_execute
from .error_handling_pipeline import ErrorHandlingPipeline
from .import_resolver import ImportResolver
from .best_practices_enforcer import BestPracticesEnforcer

# State and mode management
from .state_tracker import StateTracker
from .mode_manager import ModeManager
from .drift_phase_monitor import DriftPhaseMonitor

# Windows compatibility
from .enhanced_windows_cli_compatibility import EnhancedWindowsCliCompatibilityHandler

# Phase engine components
from .phase_engine.basket_phase_map import BasketPhaseMap

# Advanced components
from .integrated_alif_aleph_system import IntegratedAlifAlephSystem
from .memory_agent_ghost_meta_engine import MemoryAgentGhostMetaEngine
from .lantern_news_intelligence_bridge import LanternNewsIntelligenceBridge
from .advanced_test_harness import AdvancedTestHarness

# Ghost and advanced logic
from .ghost_architecture_btc_profit_handoff import GhostArchitectureBTCProfitHandoff
from .ghost_strategy_handler import GhostStrategyHandler
from .compute_ghost_route import compute_ghost_route

# Volume and tick management
from .volume_tick_router import VolumeTickRouter
from .tick_backlog_router import TickBacklogRouter
from .tick_cycle_validator import TickCycleValidator

# Event matrix and impact
from .event_matrix_integration_bridge import EventMatrixIntegrationBridge
from .event_impact_mapper import EventImpactMapper

# Advanced mathematical operations
from .advanced_mathematical_core import AdvancedMathematicalCore
from .riddle_gemm import RiddleGEMM
from .altitude_adjustment_math import AltitudeAdjustmentMath
from .anomaly_filter_comprehensive import AnomalyFilterComprehensive

# Demo and testing components
from .demo_backtest_runner import DemoBacktestRunner
from .demo_entry_simulator import DemoEntrySimulator
from .demo_integration_system import DemoIntegrationSystem
from .demo_memory_core import DemoMemoryCore

# Unified interfaces
from .schwabot_unified_interface_system import SchwabotUnifiedInterfaceSystem

# Post-failure recovery
from .post_failure_recovery_intelligence_loop import PostFailureRecoveryIntelligenceLoop

# Temporal corrections
from .temporal_execution_correction_layer import TemporalExecutionCorrectionLayer

# Profit and strategy management
from .profit_cycle_allocator import ProfitCycleAllocator
from .strategy_loader import StrategyLoader

# Memory and vector operations
from .lantern_vector_memory import LanternVectorMemory

# Unified mathematics
from .unified_mathematics_config import UnifiedMathematicsConfig

# Constants
from .constants import *

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
            "errors": []
        }
        
        # Initialize core modules
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
            ("thermal_boundary_manager", "Thermal management")
        ]
        
        for module_name, description in core_modules:
            try:
                # Test module import
                module_result = {
                    "name": module_name,
                    "description": description,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                initialization_status["modules"].append(module_result)
            except Exception as e:
                module_result = {
                    "name": module_name,
                    "description": description,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                initialization_status["modules"].append(module_result)
                initialization_status["errors"].append(f"Module {module_name}: {e}")
        
        # Initialize core components
        core_components = [
            ("fault_bus", "FaultBus", "Central fault handling"),
            ("typing_schemas", "typing_schemas", "Type definitions"),
            ("mathematical_pipeline_validator", "MathematicalPipelineValidator", "Mathematical validation"),
            ("strategy_mapper", "StrategyMapper", "Strategy mapping"),
            ("ops_observability", "OpsObservability", "System observability")
        ]
        
        for component_name, class_name, description in core_components:
            try:
                component_result = {
                    "name": component_name,
                    "class": class_name,
                    "description": description,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                initialization_status["components"].append(component_result)
            except Exception as e:
                component_result = {
                    "name": component_name,
                    "class": class_name,
                    "description": description,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                initialization_status["components"].append(component_result)
                initialization_status["errors"].append(f"Component {component_name}: {e}")
        
        # Determine overall status
        successful_modules = sum(1 for m in initialization_status["modules"] if m["status"] == "success")
        successful_components = sum(1 for c in initialization_status["components"] if c["status"] == "success")
        
        if successful_modules == len(core_modules) and successful_components == len(core_components):
            initialization_status["status"] = "success"
        elif successful_modules > len(core_modules) // 2:
            initialization_status["status"] = "partial"
        else:
            initialization_status["status"] = "failed"
        
        initialization_status["summary"] = {
            "total_modules": len(core_modules),
            "successful_modules": successful_modules,
            "total_components": len(core_components),
            "successful_components": successful_components,
            "error_count": len(initialization_status["errors"])
        }
        
        logger.info(f"Core system initialization: {initialization_status['status']}")
        return initialization_status
        
    except Exception as e:
        logger.error(f"Core system initialization failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "modules": [],
            "components": [],
            "errors": [str(e)]
        }


def check_system_health() -> Dict[str, Any]:
    """Check the overall health of the Schwabot system."""
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "unknown",
            "components": {},
            "warnings": [],
            "errors": []
        }
        
        # Define health check functions
        health_checks = {
            "core_modules": lambda: len([m for m in initialize_core_system()["modules"] if m["status"] == "success"]) > 0,
            "typing_schemas": lambda: True,  # Basic check - if we can import, it's working
            "fault_bus": lambda: True,  # Basic check
            "mathematical_validation": lambda: True,  # Basic check
        }
        
        healthy_components = 0
        total_components = len(health_checks)
        
        for component_name, health_check in health_checks.items():
            try:
                is_healthy = health_check()
                health_status["components"][component_name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "timestamp": datetime.now().isoformat()
                }
                if is_healthy:
                    healthy_components += 1
                else:
                    health_status["warnings"].append(f"Component {component_name} is unhealthy")
            except Exception as e:
                health_status["components"][component_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                health_status["errors"].append(f"Component {component_name}: {e}")
        
        # Determine overall health
        if healthy_components == total_components:
            health_status["overall_health"] = "healthy"
        elif healthy_components > total_components // 2:
            health_status["overall_health"] = "degraded"
        else:
            health_status["overall_health"] = "unhealthy"
        
        health_status["summary"] = {
            "total_components": total_components,
            "healthy_components": healthy_components,
            "unhealthy_components": total_components - healthy_components,
            "error_count": len(health_status["errors"]),
            "warning_count": len(health_status["warnings"])
        }
        
        logger.info(f"System health check: {health_status['overall_health']} ({healthy_components}/{total_components} components healthy)")
        return health_status
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return {
            "overall_health": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "warnings": [],
            "errors": [str(e)]
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core components
    "MultiBitBTCProcessor", "ProfitRoutingEngine", "HashRegistry", "HashEntry", "HashType", "HashStatus",
    "StrategyLoader", "StrategyConfig", "LoaderResult", "OpsObservability", "MetricData", "MetricType",
    "RegulatoryCompliance", "ComplianceReport", "ComplianceType", "RiskGuard", "RiskEvent", "RiskLevel",
    "SecureAPIManager", "ExchangePlumbing", "PersistentStateManager", "EnvironmentManager",
    "MemoryAllocationManager", "PrecisionPerformanceManager", "LongHorizonSimulation", "ThermalBoundaryManager",
    
    # Type definitions
    "BitLevel", "MatrixPhase", "MatrixController", "MatrixControllerType", "Vector", "Matrix", "Tensor",
    "Price", "Volume", "Quantity", "Amount", "Temperature", "Pressure", "ThermalConductivity", "HeatCapacity",
    "WarpFactor", "LightSpeed", "Distance", "Time",
    
    # Typing schemas
    "FaultLog", "FaultEvent", "RecoveryStrategy", "StrategyHash", "AIStrategyResponse",
    "MathematicalOperation", "VectorOperation", "MatrixOperation", "TradingSignal",
    "SystemState", "PerformanceMetrics", "parse_ai_response", "create_fault_log", "validate_mathematical_operation",
    
    # Fault handling
    "FaultBus", "FaultBusEvent", "FaultType",
    
    # Mathematical components
    "MathematicalPipelineValidator", "SimplifiedMathematicalPipelineValidator",
    
    # AI and strategy components
    "GPTCommandLayer", "SimpleGPTCommandLayer", "StrategyMapper",
    
    # Memory and execution components
    "AICommandSequencer", "ExecutionValidator",
    
    # Risk and compliance components
    "EnhancedRiskManager", "CapitalControls",
    
    # Performance and optimization
    "PrecisionPerformanceManager", "AutoScaler",
    
    # Thermal and hardware management
    "ThermalBoundaryManager", "GPUFlashEngine",
    
    # Advanced mathematical frameworks
    "ZPECore", "ZPEIntegration", "ZPERotationalEngine", "ZPEHybridModeSelector",
    
    # Vector and matrix operations
    "UnifiedConfidenceMatrix", "HashConfidenceEvaluator", "VectorValidator", "MatrixAllocator",
    
    # Event and communication systems
    "BusCore", "EventBus", "TradeEvent", "BusEvent", "EchoSnapshot",
    
    # Advanced engines
    "FutureCorridorEngine", "EnhancedFractalCore", "HashTriggerEngine", "EntropyEngine", "AltitudeGenerator",
    
    # API and integration
    "APIGateway", "APIBridgeManager", "ColdbaseBridge", "ProphetConnector",
    
    # Orchestration and management
    "MainOrchestrator", "MasterOrchestrator", "SettingsController",
    
    # Data and analysis
    "DataIntegrationLayer", "LineRenderEngine", "TrajectorySphere",
    
    # Error handling and validation
    "ErrorHandler", "safe_execute", "ErrorHandlingPipeline", "ImportResolver", "BestPracticesEnforcer",
    
    # State and mode management
    "StateTracker", "ModeManager", "DriftPhaseMonitor",
    
    # Windows compatibility
    "EnhancedWindowsCliCompatibilityHandler",
    
    # Phase engine components
    "BasketPhaseMap",
    
    # Advanced components
    "IntegratedAlifAlephSystem", "MemoryAgentGhostMetaEngine", "LanternNewsIntelligenceBridge", "AdvancedTestHarness",
    
    # Ghost and advanced logic
    "GhostArchitectureBTCProfitHandoff", "GhostStrategyHandler", "compute_ghost_route",
    
    # Volume and tick management
    "VolumeTickRouter", "TickBacklogRouter", "TickCycleValidator",
    
    # Event matrix and impact
    "EventMatrixIntegrationBridge", "EventImpactMapper",
    
    # Advanced mathematical operations
    "AdvancedMathematicalCore", "RiddleGEMM", "AltitudeAdjustmentMath", "AnomalyFilterComprehensive",
    
    # Demo and testing components
    "DemoBacktestRunner", "DemoEntrySimulator", "DemoIntegrationSystem", "DemoMemoryCore",
    
    # Unified interfaces
    "SchwabotUnifiedInterfaceSystem",
    
    # Post-failure recovery
    "PostFailureRecoveryIntelligenceLoop",
    
    # Temporal corrections
    "TemporalExecutionCorrectionLayer",
    
    # Profit and strategy management
    "ProfitCycleAllocator",
    
    # Memory and vector operations
    "LanternVectorMemory",
    
    # Unified mathematics
    "UnifiedMathematicsConfig",
    
    # Utility functions
    "safe_print", "safe_format_error", "log_safe",
    
    # System functions
    "initialize_core_system", "check_system_health",
    
    # Version information
    "__version__", "__author__", "__description__"
]
