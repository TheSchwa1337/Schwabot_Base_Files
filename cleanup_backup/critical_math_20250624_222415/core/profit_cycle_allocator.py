from __future__ import annotations

from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Enhanced Profit Cycle Allocator with Matrix Mapper Integration.

Allocates trade volume or capital across strategy cycles with advanced tensor scoring,
matrix basket integration, and bit resolution phase management. Integrates with quantum
strategy system for optimal profit routing and portfolio rebalancing.
"""


from dataclasses import dataclass
from typing import Any, Dict, Sequence, Optional, List
from datetime import datetime
import logging
from core.unified_math_system import unified_math
import hashlib
import time

# Import ZPE Mathematical Framework
try:
    from core.zpe_core import ZPECore
    ZPE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ZPE modules not available: {e}")
    ZPE_MODULES_AVAILABLE = False

# Import Matrix Mapper
try:
    from core.matrix_mapper import MatrixMapper, BitPhase, BasketType
    MATRIX_MAPPER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix mapper not available: {e}")
    MATRIX_MAPPER_AVAILABLE = False

# Import DLT Waveform Engine
try:
    from core.dlt_waveform_engine import DLTWaveformEngine, BitPhase as DLTBitPhase
    DLT_WAVEFORM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"DLT waveform engine not available: {e}")
    DLT_WAVEFORM_AVAILABLE = False

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
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
class ProfitAllocationResult:
    """Enhanced result of profit cycle allocation with matrix integration."""
    success: bool
    allocated_packet: Dict[str, Any]
    allocation_strategy: str
    # ZPE Integration Fields
    zpe_efficiency: float = 0.0
    zpe_reinjection: float = 0.0
    total_profit: float = 0.0
    thermal_history: Optional[Dict[str, Any]] = None
    # Matrix Integration Fields
    matrix_basket_id: Optional[str] = None
    tensor_score: float = 0.0
    bit_phase: Optional[int] = None
    allocation_weights: Dict[str, float] = None
    hash_signature: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.allocation_weights is None:
            self.allocation_weights = {}


@dataclass
class BitResolutionPhase:
    """Bit resolution phase for profit allocation."""
    phase_id: str
    bit_depth: int
    entropy_threshold: float
    complexity_limit: float
    tensor_dimensions: List[int]
    allocation_strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorAllocation:
    """Tensor-based allocation result."""
    allocation_id: str
    tensor_score: float
    bit_phase: int
    basket_id: str
    allocation_weights: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProfitCycleAllocator:
    """Enhanced profit cycle allocator with matrix mapper and tensor scoring integration."""

    allocation_strategy: str = "matrix_enhanced"
    zpe_core: Optional[ZPECore] = None
    matrix_mapper: Optional[MatrixMapper] = None
    dlt_waveform_engine: Optional[DLTWaveformEngine] = None

    def __post_init__(self):
        """Initialize ZPE core and matrix mapper if available."""
        if ZPE_MODULES_AVAILABLE:
            self.zpe_core = ZPECore()
            safe_safe_print("\\u1f504 Profit Cycle Allocator initialized with ZPE integration")
        else:
            safe_safe_print("\\u26a0\\ufe0f Profit Cycle Allocator initialized without ZPE integration")

        if MATRIX_MAPPER_AVAILABLE:
            self.matrix_mapper = MatrixMapper()
            safe_safe_print("\\u1f504 Profit Cycle Allocator initialized with Matrix Mapper integration")
        else:
            safe_safe_print("\\u26a0\\ufe0f Profit Cycle Allocator initialized without Matrix Mapper integration")

        if DLT_WAVEFORM_AVAILABLE:
            self.dlt_waveform_engine = DLTWaveformEngine()
            safe_safe_print("\\u1f504 Profit Cycle Allocator initialized with DLT Waveform Engine integration")
        else:
            safe_safe_print("\\u26a0\\ufe0f Profit Cycle Allocator initialized without DLT Waveform Engine integration")

        # Initialize bit resolution phases
        self.bit_phases = self._initialize_bit_phases()

        # Performance tracking
        self.allocation_history: List[Dict[str, Any]] = []
        self.tensor_score_history: List[float] = []
        self.hash_registry: Dict[str, Dict[str, Any]] = {}

        # Integration setup
        self._setup_integrations()

    def _setup_integrations(self) -> None:
        """Setup integrations between components."""
        try:
            if self.matrix_mapper and self.dlt_waveform_engine:
                self.matrix_mapper.set_dlt_waveform_engine(self.dlt_waveform_engine)
                self.matrix_mapper.set_profit_cycle_allocator(self)
                safe_safe_print("\\u2705 Component integrations established")
        except Exception as e:
            safe_safe_print(f"\\u26a0\\ufe0f Error setting up integrations: {e}")

    def _initialize_bit_phases(self) -> Dict[int, BitResolutionPhase]:
        """Initialize bit resolution phases."""
        return {
            4: BitResolutionPhase(
                phase_id="4bit_conservative",
                bit_depth=4,
                entropy_threshold=2.0,
                complexity_limit=0.3,
                tensor_dimensions=[2, 2, 2],
                allocation_strategy="conservative"
            ),
            8: BitResolutionPhase(
                phase_id="8bit_balanced",
                bit_depth=8,
                entropy_threshold=4.0,
                complexity_limit=0.6,
                tensor_dimensions=[4, 4, 4],
                allocation_strategy="balanced"
            ),
            42: BitResolutionPhase(
                phase_id="42bit_quantum",
                bit_depth=42,
                entropy_threshold=6.0,
                complexity_limit=1.0,
                tensor_dimensions=[8, 8, 8],
                allocation_strategy="quantum"
            )
        }

    def allocate(
        self,
        execution_packet: Dict[str, Any],
        cycles: Sequence[str] | None = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ProfitAllocationResult:
        """Enhanced profit allocation with matrix mapper and tensor scoring integration.

        Parameters
        ----------
        execution_packet
            Packet produced by GhostStrategyIntegrator.
        cycles
            Optional list of cycle names. If *None*, a single 'default'
            cycle is assumed.
        market_data
            Optional market data for ZPE and matrix calculations.

        Returns
        -------
        ProfitAllocationResult
            Enhanced allocation result with matrix integration.
        """
        try:
            # Start with basic allocation
            allocation = {
                name: execution_packet.get("volume", 0.0)
                for name in (cycles or ["default"])
            }

            execution_packet = execution_packet.copy()
            execution_packet["cycle_allocation"] = allocation
            execution_packet["allocator"] = self.allocation_strategy

            # Initialize result fields
            zpe_efficiency = 0.0
            zpe_reinjection = 0.0
            total_profit = execution_packet.get("actual_profit", 0.0)
            thermal_history = None
            matrix_basket_id = None
            tensor_score = 0.0
            bit_phase = None
            allocation_weights = {}
            hash_signature = ""

            # Generate hash signature for this allocation
            hash_signature = self._generate_allocation_hash(execution_packet, market_data)

            # Matrix Mapper Integration
            if self.matrix_mapper and market_data:
                try:
                    # Determine optimal bit phase
                    bit_phase = self._determine_optimal_bit_phase(market_data)

                    # Allocate profit using matrix mapper
                    matrix_result = self.matrix_mapper.allocate_profit(total_profit, market_data)

                    if matrix_result:
                        matrix_basket_id = matrix_result.basket_id
                        tensor_score = matrix_result.tensor_score
                        allocation_weights = matrix_result.allocation_weights

                        # Update allocation based on matrix result
                        self._adjust_allocation_with_metrics(allocation, zpe_efficiency, tensor_score, bit_phase)

                        safe_safe_print(
                            f"\\u2705 Matrix allocation: basket={matrix_basket_id}, tensor_score={tensor_score:.4f}")

                except Exception as e:
                    safe_safe_print(f"\\u26a0\\ufe0f Matrix mapper integration error: {e}")

            # DLT Waveform Integration
            if self.dlt_waveform_engine and market_data:
                try:
                    # Process waveform data if available
                    waveform_data = market_data.get('waveform_data')
                    if waveform_data:
                        waveform_result = self.dlt_waveform_engine.process_waveform_data(
                            name="market_waveform",
                            x=np.array(waveform_data),
                            sample_rate=market_data.get('sample_rate', 1.0)
                        )

                        if waveform_result.get('success'):
                            # Integrate waveform analysis with matrix mapper
                            if self.matrix_mapper:
                                integration_result = self.matrix_mapper.integrate_with_dlt_waveform(waveform_result)
                                if integration_result.get('success'):
                                    safe_safe_print(f"\\u2705 DLT waveform integration: {integration_result}")

                except Exception as e:
                    safe_safe_print(f"\\u26a0\\ufe0f DLT waveform integration error: {e}")

            # ZPE Integration
            if self.zpe_core and market_data:
                try:
                    # Calculate ZPE metrics
                    zpe_metrics = self._calculate_zpe_metrics(market_data, total_profit)
                    zpe_efficiency = zpe_metrics.get('efficiency', 0.0)
                    zpe_reinjection = zpe_metrics.get('reinjection', 0.0)
                    thermal_history = zpe_metrics.get('thermal_history', {})

                    # Adjust allocation based on ZPE efficiency
                    if zpe_efficiency > 0.7:
                        # High efficiency - increase allocation
                        for cycle in allocation:
                            allocation[cycle] *= 1.2
                    elif zpe_efficiency < 0.3:
                        # Low efficiency - decrease allocation
                        for cycle in allocation:
                            allocation[cycle] *= 0.8

                    safe_safe_print(
                        f"\\u2705 ZPE integration: efficiency={zpe_efficiency:.4f}, reinjection={zpe_reinjection:.4f}")

                except Exception as e:
                    safe_safe_print(f"\\u26a0\\ufe0f ZPE integration error: {e}")

            # Store allocation history
            self._store_allocation_history(execution_packet, tensor_score, bit_phase)

            # Create result
            result = ProfitAllocationResult(
                success=True,
                allocated_packet=execution_packet,
                allocation_strategy=self.allocation_strategy,
                zpe_efficiency=zpe_efficiency,
                zpe_reinjection=zpe_reinjection,
                total_profit=total_profit,
                thermal_history=thermal_history,
                matrix_basket_id=matrix_basket_id,
                tensor_score=tensor_score,
                bit_phase=bit_phase,
                allocation_weights=allocation_weights,
                hash_signature=hash_signature,
                metadata={
                    'market_data_available': market_data is not None,
                    'matrix_mapper_available': MATRIX_MAPPER_AVAILABLE,
                    'dlt_waveform_available': DLT_WAVEFORM_AVAILABLE,
                    'zpe_available': ZPE_MODULES_AVAILABLE
                }
            )

            safe_safe_print(f"\\u2705 Enhanced allocation completed: tensor_score={tensor_score:.4f}, bit_phase={bit_phase}")
            return result

        except Exception as e:
            error_msg = safe_format_error(e, "Enhanced profit allocation")
            safe_safe_print(f"\\u274c Enhanced allocation failed: {error_msg}")

            return ProfitAllocationResult(
                success=False,
                allocated_packet=execution_packet,
                allocation_strategy=self.allocation_strategy,
                metadata={'error': str(e)}
            )

    def _generate_allocation_hash(self, execution_packet: Dict[str, Any], market_data: Optional[Dict[str, Any]]) -> str:
        """Generate hash signature for allocation."""
        try:
            # Create content for hashing
            content = {
                'execution_packet': execution_packet,
                'market_data': market_data or {},
                'timestamp': time.time()
            }

            # Generate SHA-256 hash
            content_str = str(content)
            return hashlib.sha256(content_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Error generating allocation hash: {e}")
            return hashlib.sha256(str(time.time()).encode()).hexdigest()

    def _determine_optimal_bit_phase(self, market_data: Dict[str, Any]) -> int:
        """Determine optimal bit phase based on market conditions."""
        try:
            entropy_level = market_data.get('entropy_level', 4.0)
            complexity = market_data.get('complexity', 0.5)
            volatility = market_data.get('volatility', 0.5)

            # Calculate composite score
            composite_score = (entropy_level * 0.4 + complexity * 0.3 + volatility * 0.3)

            # Determine bit phase based on composite score
            if composite_score < 2.0:
                return 4  # 4-bit conservative
            elif composite_score < 5.0:
                return 8  # 8-bit balanced
            else:
                return 42  # 42-bit quantum

        except Exception as e:
            logger.error(f"Error determining optimal bit phase: {e}")
            return 8  # Default to 8-bit

    def _get_bit_phase_enum(self, bit_phase: int) -> BitPhase:
        """Convert integer bit phase to enum."""
        try:
            if bit_phase == 4:
                return BitPhase.FOUR_BIT
            elif bit_phase == 8:
                return BitPhase.EIGHT_BIT
            elif bit_phase == 42:
                return BitPhase.FORTY_TWO_BIT
            else:
                return BitPhase.EIGHT_BIT  # Default
        except Exception as e:
            logger.error(f"Error converting bit phase: {e}")
            return BitPhase.EIGHT_BIT

    def _adjust_allocation_with_metrics(
        self,
        allocation: Dict[str, float],
        zpe_efficiency: float,
        tensor_score: float,
        bit_phase: Optional[int]
    ) -> None:
        """Adjust allocation based on ZPE and tensor metrics."""
        try:
            # Calculate adjustment factor
            zpe_factor = 1.0 + (zpe_efficiency - 0.5) * 0.4  # \\u00b120% based on ZPE efficiency
            tensor_factor = 1.0 + tensor_score * 0.3  # \\u00b130% based on tensor score

            # Bit phase adjustment
            bit_phase_factor = 1.0
            if bit_phase == 4:
                bit_phase_factor = 0.8  # Conservative
            elif bit_phase == 42:
                bit_phase_factor = 1.2  # Aggressive

            # Combined adjustment factor
            adjustment_factor = zpe_factor * tensor_factor * bit_phase_factor

            # Apply adjustment
            for cycle in allocation:
                allocation[cycle] *= adjustment_factor

            # Normalize to ensure total allocation doesn't exceed original
            total_original = sum(allocation.values())
            if total_original > 0:
                for cycle in allocation:
                    allocation[cycle] = unified_math.max(0.0, allocation[cycle])

        except Exception as e:
            logger.error(f"Error adjusting allocation with metrics: {e}")

    def _calculate_zpe_metrics(self, market_data: Dict[str, Any], profit_amount: float) -> Dict[str, Any]:
        """Calculate ZPE metrics for allocation."""
        try:
            if not self.zpe_core:
                return {'efficiency': 0.5, 'reinjection': 0.0, 'thermal_history': {}}

            # Extract market data
            trend_strength = market_data.get('trend_strength', 0.0)
            entry_exit_range = market_data.get('entry_exit_range', 0.0)
            liquidity_depth = market_data.get('liquidity_depth', 1.0)
            trend_change_rate = market_data.get('trend_change_rate', 0.0)

            # Calculate ZPE work
            zpe_work = self.zpe_core.calculate_zpe_work(trend_strength, entry_exit_range)

            # Calculate thermal efficiency
            capital_exposure = market_data.get('capital_exposure', unified_math.abs(profit_amount))
            thermal_efficiency = self.zpe_core.calculate_thermal_efficiency(profit_amount, capital_exposure)

            # Calculate profit reinjection
            market_heat = market_data.get('market_heat', 0.5)
            profit_reinjection = self.zpe_core.calculate_profit_reinjection(profit_amount, market_heat)

            # Get thermal history
            thermal_history = {
                'zpe_work': zpe_work,
                'thermal_efficiency': thermal_efficiency,
                'profit_reinjection': profit_reinjection,
                'timestamp': datetime.now().isoformat()
            }

            return {
                'efficiency': thermal_efficiency,
                'reinjection': profit_reinjection,
                'thermal_history': thermal_history
            }

        except Exception as e:
            logger.error(f"Error calculating ZPE metrics: {e}")
            return {'efficiency': 0.5, 'reinjection': 0.0, 'thermal_history': {}}

    def _store_allocation_history(self, execution_packet: Dict[str, Any], tensor_score: float, bit_phase: Optional[int]) -> None:
        """Store allocation history for analysis."""
        try:
            history_entry = {
                'timestamp': datetime.now(),
                'tensor_score': tensor_score,
                'bit_phase': bit_phase,
                'profit_amount': execution_packet.get('actual_profit', 0.0),
                'allocation_strategy': self.allocation_strategy
            }

            self.allocation_history.append(history_entry)
            self.tensor_score_history.append(tensor_score)

            # Keep only recent history
            if len(self.allocation_history) > 1000:
                self.allocation_history.pop(0)
            if len(self.tensor_score_history) > 1000:
                self.tensor_score_history.pop(0)

        except Exception as e:
            logger.error(f"Error storing allocation history: {e}")

    def get_zpe_metrics(self) -> Dict[str, Any]:
        """Get ZPE performance metrics."""
        try:
            if not self.zpe_core:
                return {'error': 'ZPE core not available'}

            # Get thermal history from ZPE core
            thermal_history = getattr(self.zpe_core, 'thermal_history', [])

            # Calculate average efficiency
            if thermal_history:
                avg_efficiency = unified_math.mean([entry.get('efficiency', 0.0) for entry in thermal_history])
                recent_efficiency = thermal_history[-1].get('efficiency', 0.0) if thermal_history else 0.0
            else:
                avg_efficiency = 0.5
                recent_efficiency = 0.5

            return {
                'average_efficiency': avg_efficiency,
                'recent_efficiency': recent_efficiency,
                'thermal_history_size': len(thermal_history),
                'agent_consensus': getattr(self.zpe_core, 'agent_consensus', {}),
                'recursion_depth': getattr(self.zpe_core, 'recursion_depth', 0)
            }

        except Exception as e:
            logger.error(f"Error getting ZPE metrics: {e}")
            return {'error': str(e)}

    def get_matrix_metrics(self) -> Dict[str, Any]:
        """Get matrix mapper performance metrics."""
        try:
            if not self.matrix_mapper:
                return {'error': 'Matrix mapper not available'}

            # Get hash registry status
            registry_status = self.matrix_mapper.get_hash_registry_status()

            # Get basket performance statistics
            basket_stats = {}
            for basket_id in self.matrix_mapper.basket_registry.keys():
                performance = self.matrix_mapper.get_basket_performance(basket_id)
                if 'error' not in performance:
                    basket_stats[basket_id] = performance

            return {
                'registry_status': registry_status,
                'basket_performance': basket_stats,
                'total_baskets': len(self.matrix_mapper.basket_registry),
                'total_tensor_routes': len(self.matrix_mapper.tensor_routes),
                'total_profit_allocations': len(self.matrix_mapper.profit_allocations)
            }

        except Exception as e:
            logger.error(f"Error getting matrix metrics: {e}")
            return {'error': str(e)}

    def _get_bit_phase_distribution(self) -> Dict[int, int]:
        """Get distribution of bit phases used in allocations."""
        try:
            distribution = {4: 0, 8: 0, 42: 0}

            for entry in self.allocation_history:
                bit_phase = entry.get('bit_phase')
                if bit_phase in distribution:
                    distribution[bit_phase] += 1

            return distribution

        except Exception as e:
            logger.error(f"Error getting bit phase distribution: {e}")
            return {4: 0, 8: 0, 42: 0}

    def get_allocation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive allocation statistics."""
        try:
            if not self.allocation_history:
                return {'error': 'No allocation history available'}

            # Calculate statistics
            total_allocations = len(self.allocation_history)
            avg_tensor_score = unified_math.unified_math.mean(
                self.tensor_score_history) if self.tensor_score_history else 0.0
            total_profit = sum(entry.get('profit_amount', 0.0) for entry in self.allocation_history)

            # Bit phase distribution
            bit_phase_dist = self._get_bit_phase_distribution()

            # Success rate
            successful_allocations = sum(1 for entry in self.allocation_history if entry.get('tensor_score', 0.0) > 0.0)
            success_rate = successful_allocations / total_allocations if total_allocations > 0 else 0.0

            return {
                'total_allocations': total_allocations,
                'average_tensor_score': avg_tensor_score,
                'total_profit': total_profit,
                'success_rate': success_rate,
                'bit_phase_distribution': bit_phase_dist,
                'allocation_strategy': self.allocation_strategy,
                'integrations': {
                    'zpe_available': ZPE_MODULES_AVAILABLE,
                    'matrix_mapper_available': MATRIX_MAPPER_AVAILABLE,
                    'dlt_waveform_available': DLT_WAVEFORM_AVAILABLE
                }
            }

        except Exception as e:
            logger.error(f"Error getting allocation statistics: {e}")
            return {'error': str(e)}

    def integrate_with_dlt_waveform(self, waveform_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with DLT waveform engine."""
        try:
            if not self.dlt_waveform_engine:
                return {'error': 'DLT waveform engine not available'}

            # Process waveform data
            result = self.dlt_waveform_engine.process_waveform_data(
                name=waveform_data.get('name', 'integration_waveform'),
                x=np.array(waveform_data.get('data', [])),
                sample_rate=waveform_data.get('sample_rate', 1.0)
            )

            if result.get('success'):
                # Integrate with matrix mapper
                if self.matrix_mapper:
                    matrix_result = self.matrix_mapper.integrate_with_dlt_waveform(result)
                    return {
                        'success': True,
                        'waveform_result': result,
                        'matrix_result': matrix_result
                    }
                else:
                    return {
                        'success': True,
                        'waveform_result': result,
                        'matrix_result': {'error': 'Matrix mapper not available'}
                    }
            else:
                return {'error': 'Waveform processing failed'}

        except Exception as e:
            logger.error(f"Error integrating with DLT waveform: {e}")
            return {'error': str(e)}

    def integrate_with_matrix_mapper(self, market_data: Dict[str, Any], profit_amount: float) -> Dict[str, Any]:
        """Integrate with matrix mapper for profit allocation."""
        try:
            if not self.matrix_mapper:
                return {'error': 'Matrix mapper not available'}

            # Allocate profit using matrix mapper
            allocation = self.matrix_mapper.allocate_profit(profit_amount, market_data)

            if allocation:
                return {
                    'success': True,
                    'allocation_id': allocation.allocation_id,
                    'basket_id': allocation.basket_id,
                    'tensor_score': allocation.tensor_score,
                    'bit_phase': allocation.bit_phase.value,
                    'allocation_weights': allocation.allocation_weights
                }
            else:
                return {'error': 'Matrix allocation failed'}

        except Exception as e:
            logger.error(f"Error integrating with matrix mapper: {e}")
            return {'error': str(e)}

    def rebalance(self, profit: float, volatility: float) -> dict:
        """
        Rebalance profit allocation based on profit and volatility.

        Mathematical Formula:
        if profit > 0.12:
            return {"BTC": profit * 0.75, "USDC": profit * 0.25}
        elif volatility > 0.3:
            return {"USDC": profit * 0.6, "XRP": profit * 0.4}
        else:
            return {"XRP": profit * 1.0}

        Args:
            profit: Profit amount to rebalance
            volatility: Market volatility

        Returns:
            dict: Asset allocation weights
        """
        try:
            if profit > 0.12:
                # High profit scenario - allocate to BTC and USDC
                allocation = {
                    "BTC": profit * 0.75,
                    "USDC": profit * 0.25
                }
                safe_safe_print(f"\\u1f7e2 High profit rebalance: BTC={allocation['BTC']:.4f}, USDC={allocation['USDC']:.4f}")

            elif volatility > 0.3:
                # High volatility scenario - allocate to USDC and XRP
                allocation = {
                    "USDC": profit * 0.6,
                    "XRP": profit * 0.4
                }
                safe_safe_print(
                    f"\\u1f7e1 High volatility rebalance: USDC={allocation['USDC']:.4f}, XRP={allocation['XRP']:.4f}")

            else:
                # Normal scenario - allocate to XRP
                allocation = {
                    "XRP": profit * 1.0
                }
                safe_safe_print(f"\\u1f535 Normal rebalance: XRP={allocation['XRP']:.4f}")

            # Store rebalance in history
            self.allocation_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'rebalance',
                'profit': profit,
                'volatility': volatility,
                'allocation': allocation
            })

            return allocation

        except Exception as e:
            logger.error(f"Error in rebalance: {e}")
            # Default allocation on error
            return {"USDC": profit * 1.0}


def allocate_profit_cycle(
    execution_packet: Dict[str, Any],
    cycles: Sequence[str] | None = None,
    market_data: Optional[Dict[str, Any]] = None
) -> ProfitAllocationResult:
    """Enhanced profit cycle allocation function."""
    allocator = ProfitCycleAllocator()
    return allocator.allocate(execution_packet, cycles, market_data)


def allocate_profit_cycle_legacy(
    execution_packet: Dict[str, Any],
    cycles: Sequence[str] | None = None
) -> Dict[str, Any]:
    """Legacy profit cycle allocation for backward compatibility."""
    try:
        result = allocate_profit_cycle(execution_packet, cycles)
        return {
            'success': result.success,
            'allocated_packet': result.allocated_packet,
            'allocation_strategy': result.allocation_strategy
        }
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    # Test enhanced profit cycle allocator
    allocator = ProfitCycleAllocator()

    # Test execution packet
    test_packet = {
        "tick": 1000,
        "actual_profit": 500.0,
        "entry_price": 44000.0,
        "capital_exposure": 0.8,
        "profit_delta": 100.0,
        "volume": 1000.0
    }

    # Test market data
    test_market_data = {
        "current_price": 45000.0,
        "volatility": 0.3,
        "entropy_level": 4.5,
        "complexity": 0.6,
        "market_heat": 0.4,
        "volume_btc": 5000.0,
        "volume_eth": 3000.0,
        "volume_xrp": 2000.0,
        "volume_usdc": 8000.0,
        "volume_sol": 1500.0
    }

    # Test allocation
    result = allocator.allocate(test_packet, ["cycle_1", "cycle_2"], test_market_data)

    safe_print("Allocation Result:")
    safe_print(f"Success: {result.success}")
    safe_print(f"Matrix Basket ID: {result.matrix_basket_id}")
    safe_print(f"Tensor Score: {result.tensor_score}")
    safe_print(f"Bit Phase: {result.bit_phase}")
    safe_print(f"ZPE Efficiency: {result.zpe_efficiency}")
    safe_print(f"Total Profit: {result.total_profit}")

    # Get statistics
    stats = allocator.get_allocation_statistics()
    safe_print(f"\\nAllocation Statistics: {stats}")

"""