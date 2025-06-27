# -*- coding: utf-8 -*-
"""Strategy Mapper - Hybrid Implementation with Legacy & Ghost Phase Integration"""

This module provides both legacy UROS v1.0 / ZPE mathematical framework and modern
Ghost Phase Strategy Loader integration via a dual-path system.

Legacy Path: Preserves original UROS v1.0, ZPE, Prophet Connector math
Modern Path: Uses centralized GhostPhaseStrategyLoader decision engine

The system can be toggled via configuration flags or runtime switches.
""""""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Modern Ghost Phase Integration
from core.ghost_phase_strategy_loader import GhostPhaseStrategyLoader, GhostPhaseDecision

# Safe print utilities
try:
    from utils.safe_print import safe_print as safe_safe_print, info, warn, error, success, debug
except ImportError:
    # Fallback implementations
    def safe_safe_print(message):
        print(message)

    def info(message):
        print(f"[INFO] {message}")

    def warn(message):
        print(f"[WARN] {message}")

    def error(message):
        print(f"[ERROR] {message}")

    def success(message):
        print(f"[SUCCESS] {message}")

    def debug(message):
        print(f"[DEBUG] {message}")

# Legacy UROS v1.0 modules
try:
    from core.memory_stack.ai_command_sequencer import ()
        AICommandSequencer, sequence_ai_command, update_command_sequence_result
    
    from core.memory_stack.memory_key_allocator import ()
        MemoryKeyAllocator, allocate_memory_key, KeyType
    
    from core.memory_stack.execution_validator import ()
        ExecutionValidator, simulate_execution_cost, validate_execution
    
    from core.prophet_connector import ()
        ProphetConnector, compute_alpha_score, analyze_curve_alignment
    
    from core.gpt_command_layer import ()
        AIAgentType, CommandDomain, CommandPriority, AICommand, CommandResponse
    
    UROS_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"UROS v1.0 modules not available: {e}")
    UROS_MODULES_AVAILABLE = False

# Legacy ZPE Mathematical Framework
try:
    from core.zpe_core import ZPECore
    ZPE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ZPE modules not available: {e}")
    ZPE_MODULES_AVAILABLE = False

# Legacy unified math system
try:
    from core.unified_math_system import unified_math
except ImportError:
    # Fallback math implementation
    class Placeholder: pass
        @staticmethod
        def min(a, b):
            return min(a, b)

        @staticmethod
        def max(a, b):
            return max(a, b)

    unified_math = UnifiedMath()

# Legacy CLI compatibility
try:
    from core.utils.windows_cli_compatibility import safe_format_error, log_safe
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False

    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"

    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)

logger = logging.getLogger(__name__)


@dataclass
class Placeholder: pass
    """Unified result structure supporting both legacy and modern paths."""
    success: bool
    strategy_id: str

    # Modern Ghost Phase fields
    ghost_decision: Optional[GhostPhaseDecision] = None
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy UROS/ZPE fields (preserved for backward compatibility)
    mapped_strategy: Optional[Dict[str, Any]] = None
    alpha_score: float = 0.0
    memory_key: Optional[str] = None
    execution_cost: float = 0.0
    validation_score: float = 0.0
    zpe_work: float = 0.0
    zpe_alignment: Optional[Dict[str, Any]] = None
    zpe_spin_score: float = 0.0
    zpe_should_spin: bool = False


class Placeholder: pass
    """Hybrid strategy mapper with legacy UROS / ZPE and modern Ghost Phase support."""

    def __init__()
        self,
        overlay_json: str = "memory_stack/aleph_overlays.json",
        *,
        enable_ghost_phase: bool = True,
        enable_legacy: bool = True,
        default_to_legacy: bool = False,
     -> None:
        """Initialize hybrid strategy mapper."""

        Args:
            overlay_json: Path to overlay configuration for Ghost Phase
            enable_ghost_phase: Enable modern Ghost Phase logic
            enable_legacy: Enable legacy UROS / ZPE logic
            default_to_legacy: Default to legacy path when both enabled
        """"""
        self.enable_ghost_phase = enable_ghost_phase
        self.enable_legacy = enable_legacy
        self.default_to_legacy = default_to_legacy

        # Modern Ghost Phase Strategy Loader
        if self.enable_ghost_phase:
            try:
                self.ghost_loader = GhostPhaseStrategyLoader(overlay_json)
            except Exception as e:
                logging.warning()
                    f"Ghost Phase loader failed to initialize: {e}"
                self.enable_ghost_phase = False

        # Legacy UROS v1.0 components
        if self.enable_legacy:
            self.sequencer = AICommandSequencer() if UROS_MODULES_AVAILABLE else None
            self.memory_allocator = MemoryKeyAllocator() if UROS_MODULES_AVAILABLE else None
            self.execution_validator = ExecutionValidator() if UROS_MODULES_AVAILABLE else None
            self.prophet_connector = ProphetConnector() if UROS_MODULES_AVAILABLE else None

        # ZPE Integration
        self.zpe_core = ZPECore() if ZPE_MODULES_AVAILABLE else None

        # Performance tracking (shared between paths)
        self.total_mappings = 0
        self.successful_mappings = 0
        self.average_alpha_score = 0.0
        self.zpe_spin_count = 0
        self.ghost_decisions = 0
        self.legacy_mappings = 0

        paths_enabled = []
        if self.enable_ghost_phase:
            paths_enabled.append("Ghost Phase")
        if self.enable_legacy:
            paths_enabled.append("Legacy UROS/ZPE")

        safe_safe_print()
            f"Hybrid Strategy Mapper initialized with: {"}
                ', '.join(paths_enabled")"

    # ------------------------------------------------------------------
    # PUBLIC API - Dual Path Strategy Mapping
    # ------------------------------------------------------------------

    def map_strategy()
        self,
        prices: Sequence[float],
        live_vector: Sequence[float],
        raw_signals: Sequence[float],
        execution_packet: Optional[Dict[str, Any]] = None,
        use_legacy: Optional[bool] = None,
     -> StrategyMappingResult:
        """Map strategy using Ghost Phase or legacy logic."""

        Args:
            prices: Historical price data (for Ghost Phase)
            live_vector: Current market state vector (for Ghost Phase)
            raw_signals: Strategy confidence signals
            execution_packet: Legacy execution context
            use_legacy: Force legacy path (None=auto-detect)

        Returns:
            StrategyMappingResult with strategy ID and diagnostics
        """"""
        try:
            self.total_mappings += 1

            # Determine which path to use
            should_use_legacy = self._determine_path()
                use_legacy, execution_packet

            if should_use_legacy and self.enable_legacy:
                return self._modern_legacy_wrapper()
                    prices, raw_signals, execution_packet
                
            elif self.enable_ghost_phase:
                return self._ghost_phase_path()
                    prices, live_vector, raw_signals, execution_packet
                
            else:
                return self._fallback_strategy()
                    execution_packet, "No valid path available"

        except Exception as e:
            error_msg = safe_format_error(e, "map_strategy")
            logging.error(f"Strategy mapping failed: {error_msg}")
            return self._fallback_strategy(execution_packet, error_msg)

    async def map_strategy_enhanced()
        self,
        execution_packet: Dict[str, Any],
        agent_type: Optional[object] = None,
        prophet_curve_id: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None,
        use_legacy: Optional[bool] = None,
     -> StrategyMappingResult:
        """Enhanced strategy mapping with async support and market data integration."""

        Args:
            execution_packet: Execution context packet
            agent_type: AI agent type for legacy path
            prophet_curve_id: Prophet curve identifier
            market_data: Market data for Ghost Phase
            use_legacy: Force legacy path

        Returns:
            Enhanced StrategyMappingResult
        """"""
        try:
            self.total_mappings += 1

            # Extract data for Ghost Phase
            prices = self._extract_prices_from_packet()
                execution_packet, market_data
            live_vector = self._extract_live_vector_from_packet()
                execution_packet, market_data
            raw_signals = self._extract_signals_from_packet(execution_packet)

            # Determine path
            should_use_legacy = self._determine_path()
                use_legacy, execution_packet

            if should_use_legacy and self.enable_legacy:
                return await self._legacy_enhanced_path()
                    execution_packet, agent_type, prophet_curve_id, market_data
                
            elif self.enable_ghost_phase:
                return self._ghost_phase_path()
                    prices, live_vector, raw_signals, execution_packet
                
            else:
                return self._fallback_strategy()
                    execution_packet, "No valid path available"

        except Exception as e:
            error_msg = safe_format_error(e, "map_strategy_enhanced")
            logging.error(f"Enhanced strategy mapping failed: {error_msg}")
            return self._fallback_strategy(execution_packet, error_msg)

    # ------------------------------------------------------------------
    # GHOST PHASE PATH
    # ------------------------------------------------------------------

    def _ghost_phase_path()
        self,
        prices: Sequence[float],
        live_vector: Sequence[float],
        raw_signals: Sequence[float],
        execution_packet: Optional[Dict[str, Any]] = None,
     -> StrategyMappingResult:
        """Execute Ghost Phase strategy mapping."""
        try:
            # Get Ghost Phase decision
            ghost_decision = self.ghost_loader.get_decision()
                prices, live_vector, raw_signals
            

            self.ghost_decisions += 1

            # Create result
            result = StrategyMappingResult()
                success=True,
                strategy_id=ghost_decision.strategy_id,
                ghost_decision=ghost_decision,
                recommendations=ghost_decision.recommendations,
                metadata={}
                    "path": "ghost_phase",
                    "confidence": ghost_decision.confidence,
                    "phase": ghost_decision.phase,
                    "execution_packet": execution_packet,
                ,
            

            self.successful_mappings += 1
            return result

        except Exception as e:
            error_msg = safe_format_error(e, "ghost_phase_path")
            logging.error(f"Ghost Phase path failed: {error_msg}")
            return self._fallback_strategy(execution_packet, error_msg)

    # ------------------------------------------------------------------
    # LEGACY PATH
    # ------------------------------------------------------------------

    def _modern_legacy_wrapper()
        self,
        prices: Sequence[float],
        raw_signals: Sequence[float],
        execution_packet: Optional[Dict[str, Any]] = None,
     -> StrategyMappingResult:
        """Modern wrapper around legacy UROS/ZPE logic."""
        try:
            # Create execution packet if not provided
            if execution_packet is None:
                execution_packet = {}
                    "prices": list(prices),
                    "signals": list(raw_signals),
                    "timestamp": datetime.now().isoformat(),
                

            # Execute legacy mapping
            legacy_result = self._legacy_map_strategy_core(execution_packet)

            # Convert to modern result format
            result = StrategyMappingResult()
                success=legacy_result[0],
                strategy_id=legacy_result[1].get("strategy_id", "legacy_default"),
                mapped_strategy=legacy_result[1].get("mapped_strategy"),
                alpha_score=legacy_result[1].get("alpha_score", 0.0),
                memory_key=legacy_result[1].get("memory_key"),
                execution_cost=legacy_result[1].get("execution_cost", 0.0),
                validation_score=legacy_result[1].get("validation_score", 0.0),
                zpe_work=legacy_result[1].get("zpe_work", 0.0),
                zpe_alignment=legacy_result[1].get("zpe_alignment"),
                zpe_spin_score=legacy_result[1].get("zpe_spin_score", 0.0),
                zpe_should_spin=legacy_result[1].get("zpe_should_spin", False),
                metadata={}
                    "path": "legacy_uros_zpe",
                    "execution_packet": execution_packet,
                ,
            

            self.legacy_mappings += 1
            if result.success:
                self.successful_mappings += 1
                self._update_average_alpha(result.alpha_score)

            return result

        except Exception as e:
            error_msg = safe_format_error(e, "modern_legacy_wrapper")
            logging.error(f"Legacy wrapper failed: {error_msg}")
            return self._fallback_strategy(execution_packet, error_msg)

    async def _legacy_enhanced_path()
        self,
        execution_packet: Dict[str, Any],
        agent_type: Optional[object] = None,
        prophet_curve_id: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None,
     -> StrategyMappingResult:
        """Enhanced legacy path with async support."""
        try:
            # Execute legacy enhanced mapping
            legacy_result = await self._legacy_enhanced_core()
                execution_packet, agent_type, prophet_curve_id, market_data
            

            # Convert to modern result format
            result = StrategyMappingResult()
                success=legacy_result[0],
                strategy_id=legacy_result[1].get("strategy_id", "legacy_enhanced"),
                mapped_strategy=legacy_result[1].get("mapped_strategy"),
                alpha_score=legacy_result[1].get("alpha_score", 0.0),
                memory_key=legacy_result[1].get("memory_key"),
                execution_cost=legacy_result[1].get("execution_cost", 0.0),
                validation_score=legacy_result[1].get("validation_score", 0.0),
                zpe_work=legacy_result[1].get("zpe_work", 0.0),
                zpe_alignment=legacy_result[1].get("zpe_alignment"),
                zpe_spin_score=legacy_result[1].get("zpe_spin_score", 0.0),
                zpe_should_spin=legacy_result[1].get("zpe_should_spin", False),
                metadata={}
                    "path": "legacy_enhanced",
                    "agent_type": str(agent_type) if agent_type else None,
                    "prophet_curve_id": prophet_curve_id,
                    "market_data": market_data,
                ,
            

            self.legacy_mappings += 1
            if result.success:
                self.successful_mappings += 1
                self._update_average_alpha(result.alpha_score)

            return result

        except Exception as e:
            error_msg = safe_format_error(e, "legacy_enhanced_path")
            logging.error(f"Legacy enhanced path failed: {error_msg}")
            return self._fallback_strategy(execution_packet, error_msg)

    def _legacy_map_strategy_core()
        self, execution_packet: Dict[str, Any]
     -> Tuple[bool, Dict[str, Any]]:
        """Core legacy strategy mapping logic."""
        try:
            # Basic mapping
            mapped_packet = self._map_strategy_core(execution_packet)

            # Calculate complexity
            complexity = self._calculate_complexity(execution_packet)

            # Generate strategy ID
            strategy_id = f"legacy_{int(time.time())}_{complexity:.2f}"

            result = {}
                "success": True,
                "strategy_id": strategy_id,
                "mapped_strategy": mapped_packet,
                "alpha_score": 0.7,  # Default alpha score
                "execution_cost": complexity,
                "validation_score": 0.8,
            

            return True, result

        except Exception as e:
            error_msg = safe_format_error(e, "legacy_map_strategy_core")
            logging.error(f"Legacy core mapping failed: {error_msg}")
            return False, {"error": error_msg}

    async def _legacy_enhanced_core()
        self,
        execution_packet: Dict[str, Any],
        agent_type: Optional[object] = None,
        prophet_curve_id: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None,
     -> Tuple[bool, Dict[str, Any]]:
        """Enhanced legacy core with async operations."""
        try:
            # Create AI command for sequencing
            ai_command = self._create_ai_command(execution_packet, agent_type)

            # Sequence the command
            if self.sequencer:
                sequence_result = await self.sequencer.sequence_command(ai_command)
            else:
                sequence_result = {"success": True, "sequence_id": "default"}

            # Allocate memory key
            memory_key = None
            if self.memory_allocator:
                memory_key = self.memory_allocator.allocate_key()
                    KeyType.STRATEGY

            # Validate execution
            validation_score = 0.8
            if self.execution_validator:
                validation_result = self.execution_validator.validate_execution()
                    execution_packet, complexity=self._calculate_complexity(execution_packet)
                validation_score = validation_result.get("score", 0.8)

            # Prophet analysis
            alpha_score = 0.7
            if self.prophet_connector and prophet_curve_id:
                alpha_score = self.prophet_connector.compute_alpha_score()
                    prophet_curve_id, execution_packet
                

            # ZPE analysis
            zpe_work = 0.0
            zpe_alignment = None
            zpe_spin_score = 0.0
            zpe_should_spin = False

            if self.zpe_core:
                zpe_result = self.zpe_core.analyze_execution_packet()
                    execution_packet
                zpe_work = zpe_result.get("work", 0.0)
                zpe_alignment = zpe_result.get("alignment")
                zpe_spin_score = zpe_result.get("spin_score", 0.0)
                zpe_should_spin = zpe_result.get("should_spin", False)

                if zpe_should_spin:
                    self.zpe_spin_count += 1

            # Create result
            strategy_id = f"enhanced_{int(time.time())}_{alpha_score:.2f}"
            mapped_packet = self._map_strategy_core(execution_packet)

            result = {}
                "success": True,
                "strategy_id": strategy_id,
                "mapped_strategy": mapped_packet,
                "alpha_score": alpha_score,
                "memory_key": memory_key,
                "execution_cost": self._calculate_complexity(execution_packet),
                "validation_score": validation_score,
                "zpe_work": zpe_work,
                "zpe_alignment": zpe_alignment,
                "zpe_spin_score": zpe_spin_score,
                "zpe_should_spin": zpe_should_spin,
                "sequence_result": sequence_result,
            

            return True, result

        except Exception as e:
            error_msg = safe_format_error(e, "legacy_enhanced_core")
            logging.error(f"Legacy enhanced core failed: {error_msg}")
            return False, {"error": error_msg}

    def _create_ai_command()
        self, execution_packet: Dict[str, Any], agent_type: Optional[object] = None
     -> Any:
        """Create AI command object for legacy sequencing."""
        # Create a simple command object for legacy compatibility
        class Placeholder: pass
            def __init__(self):
                self.command_id = f"strategy_map_{int(time.time())}"
                self.hash_signature = self._generate_hash(execution_packet)
                self.payload = execution_packet
                self.domain = type('Domain', (), {'value': 'STRATEGY'})()

        return SimpleCommand()

    def _map_strategy_core()
        self, execution_packet: Dict[str, Any]
     -> Dict[str, Any]:
        """Core strategy mapping logic (original implementation)."""
        # This is the original mapping logic
        mapped_packet = execution_packet.copy()

        # Add mapping metadata
        mapped_packet['mapped_at'] = datetime.now().isoformat()
        mapped_packet['mapper_version'] = 'uros_v1.0_zpe_hybrid'

        return mapped_packet

    def _calculate_complexity(self, execution_packet: Dict[str, Any]) -> float:
        """Calculate complexity score for execution packet."""
        try:
            # Simple complexity calculation based on packet size and content
            base_complexity = 1.0

            # Add complexity for different strategy types
            strategy_type = execution_packet.get('strategy_type', 'unknown')
            if strategy_type in ['high_frequency', 'arbitrage']:
                base_complexity += 0.5
            elif strategy_type in ['momentum', 'mean_reversion']:
                base_complexity += 0.3

            # Add complexity for packet size
            packet_size = len(str(execution_packet))
            size_complexity = unified_math.min()
                0.5, packet_size / 10000  # Cap at 0.5

            return base_complexity + size_complexity

        except Exception:
            return 1.0

    def _generate_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash signature for data."""
        import hashlib
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()[]
            :16  # Truncate for readability

    def _update_average_alpha(self, new_alpha: float) -> None:
        """Update average alpha score."""
        if self.total_mappings > 0:
            self.average_alpha_score = ()
                (self.average_alpha_score * (self.total_mappings - 1) +)
                 new_alpha / self.total_mappings
            

    # ------------------------------------------------------------------
    # HELPER METHODS
    # ------------------------------------------------------------------

    def _determine_path()
        self,
        use_legacy: Optional[bool],
        execution_packet: Optional[Dict[str, Any]]
     -> bool:
        """Determine which path to use based on configuration and context."""
        # Explicit override
        if use_legacy is not None:
            return use_legacy

        # Auto-detection based on packet contents
        if execution_packet:
            # If packet has legacy UROS/ZPE fields, use legacy
            legacy_indicators = []
                'agent_type', 'prophet_curve_id', 'zpe_work',
                'alpha_score', 'memory_key', 'execution_cost'

            if any(key in execution_packet for key in legacy_indicators):
                return True

        # Default based on configuration
        return self.default_to_legacy

    def _extract_prices_from_packet()
        self,
        execution_packet: Dict[str, Any],
        market_data: Optional[Dict[str, Any]]
     -> Sequence[float]:
        """Extract price sequence for Ghost Phase from legacy packet."""
        if 'prices' in execution_packet:
            return execution_packet['prices']
        if market_data and 'prices' in market_data:
            return market_data['prices']
        # Generate dummy prices for compatibility
        import random
        return [50000 + random.random() * 1000 for _ in range(20)]

    def _extract_live_vector_from_packet()
        self,
        execution_packet: Dict[str, Any],
        market_data: Optional[Dict[str, Any]]
     -> Sequence[float]:
        """Extract live vector for Ghost Phase from legacy packet."""
        if 'live_vector' in execution_packet:
            return execution_packet['live_vector']
        if market_data and 'live_vector' in market_data:
            return market_data['live_vector']
        # Generate from available data
        return [0.6, 0.4, 0.7, 0.3, 0.8, 0.2]

    def _extract_signals_from_packet()
        self, execution_packet: Dict[str, Any]
     -> Sequence[float]:
        """Extract signals for Ghost Phase from legacy packet."""
        if 'signals' in execution_packet:
            return execution_packet['signals']
        if 'raw_signals' in execution_packet:
            return execution_packet['raw_signals']
        # Generate from available data
        return [0.7, 0.3, 0.6, 0.8, 0.4]

    def _fallback_strategy()
        self,
        execution_packet: Optional[Dict[str, Any]],
        error: Optional[str] = None
     -> StrategyMappingResult:
        """Safe fallback when both paths fail."""
        safe_safe_print("Using fallback strategy due to system failure")

        return StrategyMappingResult()
            success=False,
            strategy_id="emergency_hold",
            recommendations=[]
                "System fallback - hold position",
                "Check system configuration"
            ,
            metadata={}
                "path": "fallback",
                "error": error,
                "execution_packet": execution_packet,
                "total_mappings": self.total_mappings,
            ,
        

    # ------------------------------------------------------------------
    # PERFORMANCE & COMPATIBILITY
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        success_rate = ()
            self.successful_mappings / self.total_mappings
            if self.total_mappings > 0 else 0.0
        

        return {}
            "total_mappings": self.total_mappings,
            "successful_mappings": self.successful_mappings,
            "success_rate": success_rate,
            "ghost_decisions": self.ghost_decisions,
            "legacy_mappings": self.legacy_mappings,
            "average_alpha_score": self.average_alpha_score,
            "zpe_spin_count": self.zpe_spin_count,
            "zpe_spin_rate": self.zpe_spin_count /
            unified_math.max()
                self.total_mappings,
                1,
            "paths_enabled": {}
                "ghost_phase": self.enable_ghost_phase,
                "legacy": self.enable_legacy,
            ,
            "modules_available": {}
                "uros": UROS_MODULES_AVAILABLE,
                "zpe": ZPE_MODULES_AVAILABLE,
                "cli_handler": CLI_HANDLER_AVAILABLE,
            ,
        


# ------------------------------------------------------------------
# LEGACY COMPATIBILITY FUNCTIONS
# ------------------------------------------------------------------

def map_strategy(execution_packet: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy strategy mapping function for backward compatibility."""
    mapper = StrategyMapper(enable_ghost_phase=False, enable_legacy=True)
    result = mapper._legacy_map_strategy_core(execution_packet)
    return result[1].get("mapped_strategy", execution_packet)


async def map_strategy_enhanced()
    execution_packet: Dict[str, Any],
    agent_type: Optional[object] = None,
    prophet_curve_id: Optional[str] = None,
    market_data: Optional[Dict[str, Any]] = None
 -> StrategyMappingResult:
    """Legacy enhanced strategy mapping function."""
    mapper = StrategyMapper()
        enable_ghost_phase=True,
        enable_legacy=True,
        default_to_legacy=True
    return await mapper.map_strategy_enhanced()
        execution_packet, agent_type, prophet_curve_id, market_data, use_legacy=True
    


