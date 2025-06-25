from __future__ import annotations
import math

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
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
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""Strategy Mapper - UROS v1.0 Integration with ZPE Mathematical Framework.

This module maps strategies using the new UROS v1.0 components:
- AI Command Sequencer for command tracking
- Memory Key Allocator for memory management
- Execution Validator for cost simulation
- Prophet Connector for alpha score calculation
- ZPE Mathematical Framework for rotational profit alignment
"""


import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from datetime import datetime

# Import new UROS v1.0 modules
try:
    from core.memory_stack.ai_command_sequencer import (
        AICommandSequencer, sequence_ai_command, update_command_sequence_result
    )
#     from core.memory_stack.memory_key_allocator import (  # F811: duplicate import
        MemoryKeyAllocator, allocate_memory_key, KeyType
    )
#     from core.memory_stack.execution_validator import (  # F811: duplicate import
        ExecutionValidator, simulate_execution_cost, validate_execution
    )
#     from core.prophet_connector import (  # F811: duplicate import
        ProphetConnector, compute_alpha_score, analyze_curve_alignment
    )
#     from core.gpt_command_layer import (  # F811: duplicate import
        AIAgentType, CommandDomain, CommandPriority, AICommand, CommandResponse
    )
    UROS_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"UROS v1.0 modules not available: {e}")
    UROS_MODULES_AVAILABLE = False

# Import ZPE Mathematical Framework
try:
    from core.zpe_core import ZPECore
    ZPE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ZPE modules not available: {e}")
    ZPE_MODULES_AVAILABLE = False

# Import centralized CLI handler
try:
#     from core.utils.windows_cli_compatibility import (  # F811: duplicate import
        safe_print, safe_format_error, log_safe
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)

logger = logging.getLogger(__name__)


@dataclass
class StrategyMappingResult:
    """Result of strategy mapping operation with ZPE integration."""
    success: bool
    mapped_strategy: Dict[str, Any]
    alpha_score: float = 0.0
    memory_key: Optional[str] = None
    execution_cost: float = 0.0
    validation_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # ZPE Integration Fields
    zpe_work: float = 0.0
    zpe_alignment: Optional[Dict[str, Any]] = None
    zpe_spin_score: float = 0.0
    zpe_should_spin: bool = False


@dataclass
class StrategyMapper:
    """Enhanced strategy mapper with UROS v1.0 and ZPE integration."""

    def __init__(self) -> None:
        """Initialize the strategy mapper."""
        self.sequencer = AICommandSequencer() if UROS_MODULES_AVAILABLE else None
        self.memory_allocator = MemoryKeyAllocator() if UROS_MODULES_AVAILABLE else None
        self.execution_validator = ExecutionValidator() if UROS_MODULES_AVAILABLE else None
        self.prophet_connector = ProphetConnector() if UROS_MODULES_AVAILABLE else None

        # ZPE Integration
        self.zpe_core = ZPECore() if ZPE_MODULES_AVAILABLE else None

        # Performance tracking
        self.total_mappings = 0
        self.successful_mappings = 0
        self.average_alpha_score = 0.0
        self.zpe_spin_count = 0

        safe_safe_print("🗺️ Strategy Mapper initialized with UROS v1.0 and ZPE integration")

    async def map_strategy_enhanced(
        self,
        execution_packet: Dict[str, Any],
        agent_type: AIAgentType = AIAgentType.SCHWABOT,
        prophet_curve_id: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> StrategyMappingResult:
        """
        Enhanced strategy mapping with full UROS v1.0 and ZPE integration.

        Args:
            execution_packet: Strategy execution packet
            agent_type: AI agent type for command tracking
            prophet_curve_id: Optional Prophet curve ID for alpha calculation
            market_data: Optional market data for analysis

        Returns:
            StrategyMappingResult with full mapping data and ZPE calculations
        """
        try:
            start_time = time.time()

            # Create AI command for tracking
            command = self._create_ai_command(execution_packet, agent_type)

            # Sequence the command
            sequence = None
            if self.sequencer:
                sequence = await self.sequencer.sequence_command(
                    command,
                    tick=execution_packet.get('tick', 0),
                    prophet_curve_id=prophet_curve_id,
                    market_data=market_data
                )

            # Allocate memory key
            memory_key = None
            if self.memory_allocator:
                memory_key_obj = self.memory_allocator.allocate_memory_key(
                    agent_type=agent_type.value,
                    domain=command.domain.value,
                    hash_signature=command.hash_signature,
                    tick=execution_packet.get('tick', 0),
                    key_type=KeyType.AUTO_GENERATED,
                    alpha_score=0.0,  # Will be updated after execution
                    metadata={'execution_packet': execution_packet}
                )
                memory_key = memory_key_obj.key_id

            # Simulate execution cost
            execution_cost = None
            if self.execution_validator:
                execution_cost = self.execution_validator.simulate_execution_cost(
                    command_id=command.command_id,
                    payload=command.payload,
                    market_data=market_data,
                    complexity_score=self._calculate_complexity(execution_packet)
                )

            # Execute strategy mapping (original logic)
            mapped_strategy = self._map_strategy_core(execution_packet)

            # ZPE Integration - Apply ZPE mathematical framework
            zpe_work = 0.0
            zpe_alignment = None
            zpe_spin_score = 0.0
            zpe_should_spin = False

            if self.zpe_core and market_data:
                try:
                    # Extract strategy vectors for multi-asset alignment
                    strategy_vectors = self._extract_strategy_vectors(execution_packet)
                    weights = self._extract_strategy_weights(execution_packet)

                    # Apply ZPE multi-vector alignment
                    zpe_alignment = self.zpe_core.calculate_multi_vector_alignment(strategy_vectors, weights)

                    # Calculate ZPE work
                    trend_strength = market_data.get('trend_strength', 0.0)
                    entry_exit_range = market_data.get('entry_exit_range', 0.0)
                    zpe_work = self.zpe_core.calculate_zpe_work(trend_strength, entry_exit_range)

                    # Spin the ZPE profit wheel
                    zpe_result = self.zpe_core.spin_profit_wheel(market_data)
                    zpe_spin_score = zpe_result.get('spin_score', 0.0)
                    zpe_should_spin = zpe_result.get('should_spin', False)

                    # Update mapped strategy with ZPE data
                    mapped_strategy['zpe_work'] = zpe_work
                    mapped_strategy['zpe_alignment'] = zpe_alignment
                    mapped_strategy['zpe_spin_score'] = zpe_spin_score
                    mapped_strategy['zpe_should_spin'] = zpe_should_spin

                    if zpe_should_spin:
                        self.zpe_spin_count += 1
                        safe_safe_print(f"🔄 ZPE Spin Decision: SPIN (score: {zpe_spin_score:.6f})")
                    else:
                        safe_safe_print(f"⏸️ ZPE Spin Decision: HOLD (score: {zpe_spin_score:.6f})")

                except Exception as e:
                    safe_safe_print(f"⚠️ ZPE integration failed: {safe_format_error(e, 'zpe_integration')}")

            # Calculate alpha score if Prophet curve available
            alpha_score = 0.0
            if prophet_curve_id and self.prophet_connector:
                try:
                    expected_profit = execution_packet.get('expected_profit', 0.0)
                    actual_profit = execution_packet.get('actual_profit', 0.0)
                    execution_time = time.time() - start_time

                    alpha_result = compute_alpha_score(
                        p_actual=actual_profit,
                        p_expected=expected_profit,
                        delta_t=execution_time,
                        curve_id=prophet_curve_id
                    )
                    alpha_score = alpha_result.alpha_value

                    # Update memory key with alpha score
                    if memory_key and self.memory_allocator:
                        memory_key_obj = self.memory_allocator.get_memory_key(memory_key)
                        if memory_key_obj:
                            memory_key_obj.alpha_score = alpha_score
                            memory_key_obj.profit_delta = actual_profit

                except Exception as e:
                    safe_safe_print(f"⚠️ Alpha calculation failed: {safe_format_error(e, 'alpha_calculation')}")

            # Validate execution
            validation_score = 0.0
            recommendations = []
            if execution_cost and self.execution_validator:
                try:
                    # Create drift validation (simplified)
                    expected_time = datetime.now()
                    actual_time = datetime.now()

                    drift_validation = self.execution_validator.validate_drift(
                        command_id=command.command_id,
                        expected_time=expected_time,
                        actual_time=actual_time,
                        alpha_score=alpha_score,
                        confidence_score=0.8
                    )

                    # Perform full execution validation
                    execution_validation = self.execution_validator.validate_execution(
                        command_id=command.command_id,
                        execution_cost=execution_cost,
                        drift_validation=drift_validation,
                        zpe_data={
                            'zpe_work': zpe_work,
                            'zpe_spin_score': zpe_spin_score,
                            'zpe_should_spin': zpe_should_spin
                        }
                    )

                    validation_score = execution_validation.overall_score
                    recommendations = execution_validation.recommendations

                except Exception as e:
                    safe_safe_print(f"⚠️ Execution validation failed: {safe_format_error(e, 'execution_validation')}")

            # Update command sequence result
            if sequence and self.sequencer:
                try:
                    response = CommandResponse(
                        command_id=command.command_id,
                        success=True,
                        result=mapped_strategy,
                        execution_time=time.time() - start_time,
                        timestamp=datetime.now()
                    )

                    await self.sequencer.update_command_result(
                        sequence_id=sequence.sequence_id,
                        response=response,
                        profit_delta=execution_packet.get('actual_profit', 0.0),
                        prophet_curve_id=prophet_curve_id,
                        market_data=market_data
                    )

                except Exception as e:
                    safe_safe_print(f"⚠️ Command sequence update failed: {safe_format_error(e, 'sequence_update')}")

            # Update performance metrics
            self.total_mappings += 1
            self.successful_mappings += 1
            self._update_average_alpha(alpha_score)

            safe_safe_print(f"🗺️ Strategy mapped successfully - Alpha: {alpha_score:.4f}, Validation: {validation_score:.3f}, ZPE Work: {zpe_work:.6f}")

            return StrategyMappingResult(
                success=True,
                mapped_strategy=mapped_strategy,
                alpha_score=alpha_score,
                memory_key=memory_key,
                execution_cost=execution_cost.total_cost if execution_cost else 0.0,
                validation_score=validation_score,
                recommendations=recommendations,
                zpe_work=zpe_work,
                zpe_alignment=zpe_alignment,
                zpe_spin_score=zpe_spin_score,
                zpe_should_spin=zpe_should_spin,
                metadata={
                    'sequence_id': sequence.sequence_id if sequence else None,
                    'execution_time': time.time() - start_time,
                    'agent_type': agent_type.value,
                    'zpe_integration': ZPE_MODULES_AVAILABLE
                }
            )

        except Exception as e:
            error_msg = safe_format_error(e, "strategy_mapping")
            safe_safe_print(f"❌ Strategy mapping failed: {error_msg}")

            return StrategyMappingResult(
                success=False,
                mapped_strategy=execution_packet,
                metadata={'error': error_msg}
            )

    def _extract_strategy_vectors(self, execution_packet: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract strategy vectors for ZPE multi-vector alignment."""
        vectors = {}

        # Extract asset-specific vectors from execution packet
        for asset in ['BTC', 'ETH', 'XRP', 'USDC']:
            asset_data = execution_packet.get(asset.lower(), {})
            vectors[asset] = {
                'magnitude': asset_data.get('volume', 0.0),
                'resonance': asset_data.get('confidence', 0.0)
            }

        # If no asset-specific data, create default vectors
        if not any(v['magnitude'] > 0 for v in vectors.values()):
            vectors = {
                'BTC': {'magnitude': 0.5, 'resonance': 0.5},
                'ETH': {'magnitude': 0.3, 'resonance': 0.3},
                'XRP': {'magnitude': 0.2, 'resonance': 0.2},
                'USDC': {'magnitude': 0.1, 'resonance': 0.1}
            }

        return vectors

    def _extract_strategy_weights(self, execution_packet: Dict[str, Any]) -> Dict[str, float]:
        """Extract strategy weights for ZPE multi-vector alignment."""
        weights = execution_packet.get('asset_weights', {})

        # If no weights provided, use equal distribution
        if not weights:
            weights = {
                'BTC': 0.4,
                'ETH': 0.3,
                'XRP': 0.2,
                'USDC': 0.1
            }

        return weights

    def _create_ai_command(self, execution_packet: Dict[str, Any], agent_type: AIAgentType) -> AICommand:
        """Create AI command from execution packet."""
        return AICommand(
            command_id=f"strategy_map_{int(time.time())}",
            agent_type=agent_type,
            domain=CommandDomain.STRATEGY,
            priority=CommandPriority.MEDIUM,
            hash_signature=self._generate_hash(execution_packet),
            timestamp=datetime.now(),
            payload=execution_packet,
            context={'mapping_type': 'strategy_execution'}
        )

    def _map_strategy_core(self, execution_packet: Dict[str, Any]) -> Dict[str, Any]:
        """Core strategy mapping logic (original implementation)."""
        # This is the original mapping logic
        # In a real implementation, this would contain sophisticated strategy mapping
        mapped_packet = execution_packet.copy()

        # Add mapping metadata
        mapped_packet['mapped_at'] = datetime.now().isoformat()
        mapped_packet['mapper_version'] = 'uros_v1.0_zpe'

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
            size_complexity = unified_math.min(0.5, packet_size / 10000)  # Cap at 0.5

            return base_complexity + size_complexity

        except Exception:
            return 1.0

    def _generate_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash signature for data."""
        import hashlib
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _update_average_alpha(self, new_alpha: float) -> None:
        """Update average alpha score."""
        if self.total_mappings > 0:
            self.average_alpha_score = (
                (self.average_alpha_score * (self.total_mappings - 1) + new_alpha) / self.total_mappings
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics including ZPE statistics."""
        return {
            'total_mappings': self.total_mappings,
            'successful_mappings': self.successful_mappings,
            'success_rate': self.successful_mappings / unified_math.max(self.total_mappings, 1),
            'average_alpha_score': self.average_alpha_score,
            'uros_modules_available': UROS_MODULES_AVAILABLE,
            'zpe_modules_available': ZPE_MODULES_AVAILABLE,
            'zpe_spin_count': self.zpe_spin_count,
            'zpe_spin_rate': self.zpe_spin_count / unified_math.max(self.total_mappings, 1)
        }


# Legacy function for backward compatibility
def map_strategy(execution_packet: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy strategy mapping function."""
    mapper = StrategyMapper()
    return mapper._map_strategy_core(execution_packet)


# Enhanced function for external use
async def map_strategy_enhanced(
    execution_packet: Dict[str, Any],
    agent_type: AIAgentType = AIAgentType.SCHWABOT,
    prophet_curve_id: Optional[str] = None,
    market_data: Optional[Dict[str, Any]] = None
) -> StrategyMappingResult:
    """Enhanced strategy mapping with ZPE integration."""
    mapper = StrategyMapper()
    return await mapper.map_strategy_enhanced(
        execution_packet, agent_type, prophet_curve_id, market_data
    )
