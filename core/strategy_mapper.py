"""Strategy Mapper - Hybrid Implementation with Legacy & Ghost Phase Integration

This module provides both legacy UROS v1.0/ZPE mathematical framework and modern
Ghost Phase Strategy Loader integration via a dual-path system.

Legacy Path: Preserves original UROS v1.0, ZPE, Prophet Connector math
Modern Path: Uses centralized GhostPhaseStrategyLoader decision engine

The system can be toggled via configuration flags or runtime switches.
"""

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
from core.memory_stack.ai_command_sequencer import (
        AICommandSequencer, sequence_ai_command, update_command_sequence_result
    )
    from core.memory_stack.memory_key_allocator import (
        MemoryKeyAllocator, allocate_memory_key, KeyType
    )
    from core.memory_stack.execution_validator import (
        ExecutionValidator, simulate_execution_cost, validate_execution
    )
    from core.prophet_connector import (
        ProphetConnector, compute_alpha_score, analyze_curve_alignment
    )
    from core.gpt_command_layer import (
        AIAgentType, CommandDomain, CommandPriority, AICommand, CommandResponse
    )
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
    class UnifiedMath:
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
class StrategyMappingResult:
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


class StrategyMapper:
    """Hybrid strategy mapper with legacy UROS/ZPE and modern Ghost Phase support."""

    def __init__(
        self, 
        overlay_json: str = "memory_stack/aleph_overlays.json",
        *,
        enable_ghost_phase: bool = True,
        enable_legacy: bool = True,
        default_to_legacy: bool = False,
    ) -> None:
        """Initialize hybrid strategy mapper.
        
        Args:
            overlay_json: Path to overlay configuration for Ghost Phase
            enable_ghost_phase: Enable modern Ghost Phase logic
            enable_legacy: Enable legacy UROS/ZPE logic  
            default_to_legacy: Default to legacy path when both enabled
        """
        self.enable_ghost_phase = enable_ghost_phase
        self.enable_legacy = enable_legacy
        self.default_to_legacy = default_to_legacy
        
        # Modern Ghost Phase Strategy Loader
        if self.enable_ghost_phase:
            try:
                self.ghost_loader = GhostPhaseStrategyLoader(overlay_json)
            except Exception as e:
                logging.warning(f"Ghost Phase loader failed to initialize: {e}")
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
            
        safe_safe_print(f"🗺️ Hybrid Strategy Mapper initialized with: {', '.join(paths_enabled)}")

    # ------------------------------------------------------------------
    # PUBLIC API - Dual Path Strategy Mapping
    # ------------------------------------------------------------------

    def map_strategy(
        self,
        prices: Sequence[float],
        live_vector: Sequence[float],
        raw_signals: Sequence[float],
        execution_packet: Optional[Dict[str, Any]] = None,
        use_legacy: Optional[bool] = None,
    ) -> StrategyMappingResult:
        """Map strategy using Ghost Phase or legacy logic.
        
        Args:
            prices: Historical price data (for Ghost Phase)
            live_vector: Current market state vector (for Ghost Phase)
            raw_signals: Strategy confidence signals
            execution_packet: Legacy execution context
            use_legacy: Force legacy path (None = auto-detect)
            
        Returns:
            StrategyMappingResult with strategy ID and diagnostics
        """
        try:
            self.total_mappings += 1
            
            # Determine which path to use
            should_use_legacy = self._determine_path(use_legacy, execution_packet)
            
            if should_use_legacy and self.enable_legacy:
                return self._modern_legacy_wrapper(
                    prices, raw_signals, execution_packet
                )
            elif self.enable_ghost_phase:
                return self._ghost_phase_path(
                    prices, live_vector, raw_signals, execution_packet
                )
            else:
                # Fallback to basic strategy
                return self._fallback_strategy(execution_packet)
                
        except Exception as e:
            logger.error(f"Strategy mapping failed: {e}")
            return self._fallback_strategy(execution_packet, str(e))

async def map_strategy_enhanced(
        self,
execution_packet: Dict[str, Any],
        agent_type: Optional[object] = None,
        prophet_curve_id: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None,
        use_legacy: bool = True,
    ) -> StrategyMappingResult:
        """Legacy async interface with full UROS v1.0 and ZPE integration.
        
        Preserves original async signature for backward compatibility.
        """
        if use_legacy and self.enable_legacy:
            return await self._legacy_map_strategy_enhanced(
                execution_packet, agent_type, prophet_curve_id, market_data
            )
        else:
            # Convert to modern path
            prices = self._extract_prices_from_packet(execution_packet, market_data)
            live_vector = self._extract_live_vector_from_packet(execution_packet, market_data)
            raw_signals = self._extract_signals_from_packet(execution_packet)
            
            return self.map_strategy(prices, live_vector, raw_signals, execution_packet, False)

    # ------------------------------------------------------------------
    # MODERN GHOST PHASE PATH
    # ------------------------------------------------------------------

    def _ghost_phase_path(
        self,
        prices: Sequence[float],
        live_vector: Sequence[float],
        raw_signals: Sequence[float],
        execution_packet: Optional[Dict[str, Any]],
    ) -> StrategyMappingResult:
        """Modern Ghost Phase decision path."""
        self.ghost_decisions += 1
        
        # Use Ghost Phase Strategy Loader for unified decision
        ghost_decision = self.ghost_loader.decide(prices, live_vector, raw_signals)
        
        # Generate recommendations based on decision
        recommendations = self._generate_recommendations(ghost_decision)
        
        # Create metadata
        metadata = {
            "path": "ghost_phase",
            "total_mappings": self.total_mappings,
            "phase_state": ghost_decision.phase_report.phase_state.name,
            "consensus_reached": ghost_decision.consensus,
            "overlay_similarity": ghost_decision.overlay_match.similarity,
            "drift_weight": ghost_decision.drift_report.drift_weight,
        }
        
        if execution_packet:
            metadata["execution_context"] = execution_packet
            
        self.successful_mappings += 1
        
        safe_safe_print(f"🗺️ Ghost Phase Strategy: {ghost_decision.strategy_id}")
        
        return StrategyMappingResult(
            success=True,
            strategy_id=ghost_decision.strategy_id,
            ghost_decision=ghost_decision,
            recommendations=recommendations,
            metadata=metadata,
        )

    def _generate_recommendations(self, decision: GhostPhaseDecision) -> List[str]:
        """Generate strategy recommendations based on ghost decision."""
        recommendations = []
        
        # Phase-based recommendations
        phase = decision.phase_report.phase_state.name.lower()
        if phase == "high":
            recommendations.append("Consider reducing position size in high-risk phase")
        elif phase == "low":
            recommendations.append("Opportunity for increased position size in stable phase")
            
        # Consensus-based recommendations
        if not decision.consensus:
            recommendations.append("Signals lack consensus - consider waiting")
        else:
            recommendations.append("Strong signal consensus detected")
            
        # Overlay similarity recommendations
        similarity = decision.overlay_match.similarity
        if similarity > 0.8:
            recommendations.append("High overlay match - strong strategy confidence")
        elif similarity < 0.3:
            recommendations.append("Low overlay match - exercise caution")
            
        return recommendations

    # ------------------------------------------------------------------
    # LEGACY UROS/ZPE PATH (PRESERVED MATHEMATICAL LOGIC)
    # ------------------------------------------------------------------

    def _modern_legacy_wrapper(
        self,
        prices: Sequence[float],
        raw_signals: Sequence[float],
        execution_packet: Optional[Dict[str, Any]],
    ) -> StrategyMappingResult:
        """Modern wrapper for legacy path with simplified interface."""
        self.legacy_mappings += 1
        
        # Convert modern inputs to legacy format
        if not execution_packet:
            execution_packet = {
                "strategy_type": "momentum",
                "prices": list(prices),
                "signals": list(raw_signals),
                "timestamp": time.time(),
            }
        
        # Use legacy logic
        strategy_id, diagnostics = self._legacy_map_strategy_core(execution_packet)
        
        self.successful_mappings += 1
        
        safe_safe_print(f"🗺️ Legacy Strategy: {strategy_id}")
        
        return StrategyMappingResult(
            success=True,
            strategy_id=strategy_id,
            mapped_strategy=diagnostics.get("mapped_strategy"),
            alpha_score=diagnostics.get("alpha_score", 0.0),
            execution_cost=diagnostics.get("execution_cost", 0.0),
            validation_score=diagnostics.get("validation_score", 0.0),
            zpe_work=diagnostics.get("zpe_work", 0.0),
            zpe_alignment=diagnostics.get("zpe_alignment"),
            zpe_spin_score=diagnostics.get("zpe_spin_score", 0.0),
            zpe_should_spin=diagnostics.get("zpe_should_spin", False),
            recommendations=["Legacy UROS/ZPE route engaged"],
            metadata={
                "path": "legacy",
                **diagnostics
            },
        )

    async def _legacy_map_strategy_enhanced(
        self,
        execution_packet: Dict[str, Any],
        agent_type: Optional[object] = None,
prophet_curve_id: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None,
) -> StrategyMappingResult:
"""
        LEGACY MATH LOGIC (DO NOT DELETE)
        
        Original UROS v1.0 integration with ZPE Mathematical Framework.
        Used in 16-bit entropy smoothing & early DLT phase injectors.
        
        Preserves:
        - AI Command Sequencer for command tracking
        - Memory Key Allocator for memory management  
        - Execution Validator for cost simulation
        - Prophet Connector for alpha score calculation
        - ZPE Mathematical Framework for rotational profit alignment
        """
        if not self.enable_legacy:
            raise ValueError("Legacy path disabled")
            
        self.legacy_mappings += 1
        
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
                    agent_type=getattr(agent_type, 'value', 'SCHWABOT'),
                    domain=getattr(command.domain, 'value', 'STRATEGY'),
hash_signature=command.hash_signature,
tick=execution_packet.get('tick', 0),
                    key_type=getattr(KeyType, 'AUTO_GENERATED', 'AUTO_GENERATED'),
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
self.successful_mappings += 1
self._update_average_alpha(alpha_score)

            safe_safe_print(f"🗺️ Legacy strategy mapped - Alpha: {alpha_score:.4f}, Validation: {validation_score:.3f}, ZPE Work: {zpe_work:.6f}")

            # Extract strategy ID from mapped strategy
            strategy_id = mapped_strategy.get('strategy_id', 'legacy_strategy')

            return StrategyMappingResult(
                success=True,
                strategy_id=strategy_id,
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
                    'path': 'legacy_enhanced',
'sequence_id': sequence.sequence_id if sequence else None,
'execution_time': time.time() - start_time,
                    'agent_type': getattr(agent_type, 'value', 'SCHWABOT'),
                    'zpe_integration': ZPE_MODULES_AVAILABLE,
                    'uros_integration': UROS_MODULES_AVAILABLE,
}
            )

        except Exception as e:
            error_msg = safe_format_error(e, "legacy_strategy_mapping")
            safe_safe_print(f"❌ Legacy strategy mapping failed: {error_msg}")

            return StrategyMappingResult(
                success=False,
                strategy_id="fallback_hold",
mapped_strategy=execution_packet,
                metadata={'error': error_msg, 'path': 'legacy_enhanced'}
            )

    def _legacy_map_strategy_core(
        self,
        execution_packet: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        LEGACY MATH LOGIC (DO NOT DELETE)
        
        Core strategy mapping logic with original drift/entropy/path math.
        Used in 16-bit entropy smoothing & early DLT phase injectors.
        
        Returns strategy_id and diagnostic metadata.
        """
        try:
            # Legacy drift calculation
            prices = execution_packet.get('prices', [])
            if prices:
                drift = self._calculate_custom_drift(prices)
            else:
                drift = 0.0
            
            # Legacy entropy calculation
            signals = execution_packet.get('signals', [])
            if signals:
                entropy = self._calculate_entropy_legacy(signals)
            else:
                entropy = 0.0
            
            # Legacy strategy determination
            strategy_id = self._determine_legacy_strategy(drift, entropy, execution_packet)
            
            # Core strategy mapping (preserved original logic)
            mapped_packet = execution_packet.copy()
            mapped_packet['mapped_at'] = datetime.now().isoformat()
            mapped_packet['mapper_version'] = 'uros_v1.0_zpe_hybrid'
            mapped_packet['strategy_id'] = strategy_id
            mapped_packet['legacy_drift'] = drift
            mapped_packet['legacy_entropy'] = entropy
            
            return strategy_id, {
                "mapped_strategy": mapped_packet,
                "legacy_drift": drift,
                "entropy_score": entropy,
                "strategy_logic": "legacy_core",
                "alpha_score": 0.0,
                "execution_cost": 0.0,
                "validation_score": 0.8,  # Default legacy validation
            }
            
        except Exception as e:
            safe_safe_print(f"⚠️ Legacy core mapping failed: {e}")
            return "fallback_hold", {
                "mapped_strategy": execution_packet,
                "error": str(e),
                "strategy_logic": "legacy_fallback",
            }

    # ------------------------------------------------------------------
    # LEGACY MATHEMATICAL IMPLEMENTATIONS (PRESERVED)
    # ------------------------------------------------------------------

    def _calculate_custom_drift(self, prices: Sequence[float]) -> float:
        """Legacy drift calculation with custom math."""
        try:
            if len(prices) < 2:
                return 0.0
                
            # Simple momentum-based drift
            recent = prices[-5:] if len(prices) >= 5 else prices
            if len(recent) < 2:
                return 0.0
                
            drift = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0.0
            return float(drift)
            
        except Exception:
            return 0.0

    def _calculate_entropy_legacy(self, signals: Sequence[float]) -> float:
        """Legacy entropy calculation."""
        try:
            if not signals:
                return 0.0
                
            # Simple variance-based entropy
            mean_signal = sum(signals) / len(signals)
            variance = sum((s - mean_signal) ** 2 for s in signals) / len(signals)
            entropy = math.sqrt(variance)
            
            return float(entropy)
            
        except Exception:
            return 0.0

    def _determine_legacy_strategy(
        self, 
        drift: float, 
        entropy: float, 
        execution_packet: Dict[str, Any]
    ) -> str:
        """Legacy strategy determination logic."""
        strategy_type = execution_packet.get('strategy_type', 'momentum')
        
        # Legacy decision tree
        if abs(drift) > 0.05:  # High drift
            if entropy > 0.3:  # High uncertainty
                return f"{strategy_type}_conservative"
            else:  # Low uncertainty
                return f"{strategy_type}_aggressive"
        else:  # Low drift
            if entropy > 0.3:  # High uncertainty
                return f"{strategy_type}_hold"
            else:  # Low uncertainty  
                return f"{strategy_type}_moderate"

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

    def _create_ai_command(self, execution_packet: Dict[str, Any], agent_type: Optional[object]) -> object:
        """Create AI command from execution packet."""
        # Create a simple command object for legacy compatibility
        class SimpleCommand:
            def __init__(self):
                self.command_id = f"strategy_map_{int(time.time())}"
                self.hash_signature = self._generate_hash(execution_packet)
                self.payload = execution_packet
                self.domain = type('Domain', (), {'value': 'STRATEGY'})()
                
        return SimpleCommand()

def _map_strategy_core(self, execution_packet: Dict[str, Any]) -> Dict[str, Any]:
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
            size_complexity = unified_math.min(0.5, packet_size / 10000)  # Cap at 0.5

            return base_complexity + size_complexity

        except Exception:
            return 1.0

def _generate_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash signature for data."""
import hashlib
data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]  # Truncate for readability

def _update_average_alpha(self, new_alpha: float) -> None:
        """Update average alpha score."""
        if self.total_mappings > 0:
self.average_alpha_score = (
                (self.average_alpha_score * (self.total_mappings - 1) + new_alpha) / self.total_mappings
            )

    # ------------------------------------------------------------------
    # HELPER METHODS
    # ------------------------------------------------------------------

    def _determine_path(
        self, 
        use_legacy: Optional[bool], 
        execution_packet: Optional[Dict[str, Any]]
    ) -> bool:
        """Determine which path to use based on configuration and context."""
        # Explicit override
        if use_legacy is not None:
            return use_legacy
            
        # Auto-detection based on packet contents
        if execution_packet:
            # If packet has legacy UROS/ZPE fields, use legacy
            legacy_indicators = [
                'agent_type', 'prophet_curve_id', 'zpe_work', 
                'alpha_score', 'memory_key', 'execution_cost'
            ]
            if any(key in execution_packet for key in legacy_indicators):
                return True
                
        # Default based on configuration
        return self.default_to_legacy

    def _extract_prices_from_packet(
        self, 
        execution_packet: Dict[str, Any], 
        market_data: Optional[Dict[str, Any]]
    ) -> Sequence[float]:
        """Extract price sequence for Ghost Phase from legacy packet."""
        if 'prices' in execution_packet:
            return execution_packet['prices']
        if market_data and 'prices' in market_data:
            return market_data['prices']
        # Generate dummy prices for compatibility
        import random
        return [50000 + random.random() * 1000 for _ in range(20)]

    def _extract_live_vector_from_packet(
        self, 
        execution_packet: Dict[str, Any], 
        market_data: Optional[Dict[str, Any]]
    ) -> Sequence[float]:
        """Extract live vector for Ghost Phase from legacy packet."""
        if 'live_vector' in execution_packet:
            return execution_packet['live_vector']
        if market_data and 'live_vector' in market_data:
            return market_data['live_vector']
        # Generate from available data
        return [0.6, 0.4, 0.7, 0.3, 0.8, 0.2]

    def _extract_signals_from_packet(self, execution_packet: Dict[str, Any]) -> Sequence[float]:
        """Extract signals for Ghost Phase from legacy packet."""
        if 'signals' in execution_packet:
            return execution_packet['signals']
        if 'raw_signals' in execution_packet:
            return execution_packet['raw_signals']
        # Generate from available data
        return [0.7, 0.3, 0.6, 0.8, 0.4]

    def _fallback_strategy(
        self, 
        execution_packet: Optional[Dict[str, Any]], 
        error: Optional[str] = None
    ) -> StrategyMappingResult:
        """Safe fallback when both paths fail."""
        safe_safe_print("⚠️ Using fallback strategy due to system failure")
        
        return StrategyMappingResult(
            success=False,
            strategy_id="emergency_hold",
            recommendations=["System fallback - hold position", "Check system configuration"],
            metadata={
                "path": "fallback",
                "error": error,
                "execution_packet": execution_packet,
                "total_mappings": self.total_mappings,
            },
        )

    # ------------------------------------------------------------------
    # PERFORMANCE & COMPATIBILITY
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        success_rate = (
            self.successful_mappings / self.total_mappings 
            if self.total_mappings > 0 else 0.0
        )
        
        return {
            "total_mappings": self.total_mappings,
            "successful_mappings": self.successful_mappings,
            "success_rate": success_rate,
            "ghost_decisions": self.ghost_decisions,
            "legacy_mappings": self.legacy_mappings,
            "average_alpha_score": self.average_alpha_score,
            "zpe_spin_count": self.zpe_spin_count,
            "zpe_spin_rate": self.zpe_spin_count / unified_math.max(self.total_mappings, 1),
            "paths_enabled": {
                "ghost_phase": self.enable_ghost_phase,
                "legacy": self.enable_legacy,
            },
            "modules_available": {
                "uros": UROS_MODULES_AVAILABLE,
                "zpe": ZPE_MODULES_AVAILABLE,
                "cli_handler": CLI_HANDLER_AVAILABLE,
            },
        }


# ------------------------------------------------------------------
# LEGACY COMPATIBILITY FUNCTIONS
# ------------------------------------------------------------------

def map_strategy(execution_packet: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy strategy mapping function for backward compatibility."""
    mapper = StrategyMapper(enable_ghost_phase=False, enable_legacy=True)
    result = mapper._legacy_map_strategy_core(execution_packet)
    return result[1].get("mapped_strategy", execution_packet)


async def map_strategy_enhanced(
    execution_packet: Dict[str, Any],
    agent_type: Optional[object] = None,
prophet_curve_id: Optional[str] = None,
market_data: Optional[Dict[str, Any]] = None
) -> StrategyMappingResult:
    """Legacy enhanced strategy mapping function."""
    mapper = StrategyMapper(enable_ghost_phase=True, enable_legacy=True, default_to_legacy=True)
    return await mapper.map_strategy_enhanced(
        execution_packet, agent_type, prophet_curve_id, market_data, use_legacy=True
    )


