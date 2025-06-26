# -*- coding: utf-8 -*-
"""
Enhanced Profit Cycle Allocator with Matrix Mapper Integration.

Allocates trade volume or capital across strategy cycles with advanced tensor scoring,
matrix basket integration, and bit resolution phase management. Integrates with quantum
strategy system for optimal profit routing and portfolio rebalancing.
"""

from __future__ import annotations

import time
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, Sequence, Optional, List
from dataclasses import dataclass, field
import numpy as np
import math

# Import safe print for Windows compatibility
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, safe_format_error, log_safe, info, warn, error, success, debug
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    
    # Fallback functions
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)
    
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

# Import unified math system
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
        
        @staticmethod
        def abs(x):
            return abs(x)
    
    unified_math = UnifiedMath()

# Import ZPE Mathematical Framework
try:
    from core.zpe_core import ZPECore
    ZPE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ZPE modules not available: {e}")
    ZPE_MODULES_AVAILABLE = False
    ZPECore = None

# Import Matrix Mapper
try:
    from core.matrix_mapper import MatrixMapper, BitPhase, BasketType
    MATRIX_MAPPER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix mapper not available: {e}")
    MATRIX_MAPPER_AVAILABLE = False
    MatrixMapper = None
    BitPhase = None
    BasketType = None

# Import DLT Waveform Engine
try:
    from core.dlt_waveform_engine import DLTWaveformEngine, BitPhase as DLTBitPhase
    DLT_WAVEFORM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"DLT waveform engine not available: {e}")
    DLT_WAVEFORM_AVAILABLE = False
    DLTWaveformEngine = None
    DLTBitPhase = None

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


@dataclass
class ProfitCycleAllocator:
    """Enhanced profit cycle allocator with matrix mapper and tensor scoring integration."""
    
    allocation_strategy: str = "matrix_enhanced"
    zpe_core: Optional[ZPECore] = None
    matrix_mapper: Optional[MatrixMapper] = None
    dlt_waveform_engine: Optional[DLTWaveformEngine] = None
    bit_phases: Dict[int, BitResolutionPhase] = field(default_factory=dict)
    allocation_history: List[Dict[str, Any]] = field(default_factory=list)
    tensor_score_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize bit phases and validate components."""
        self._initialize_bit_phases()
        self._validate_components()
    
    def _initialize_bit_phases(self):
        """Initialize bit resolution phases for different allocation strategies."""
        self.bit_phases = {
            8: BitResolutionPhase(
                phase_id="8bit_balanced",
                bit_depth=8,
                entropy_threshold=4.0,
                complexity_limit=0.6,
                tensor_dimensions=[4, 4, 4],
                allocation_strategy="balanced"
            ),
            16: BitResolutionPhase(
                phase_id="16bit_enhanced",
                bit_depth=16,
                entropy_threshold=5.0,
                complexity_limit=0.8,
                tensor_dimensions=[6, 6, 6],
                allocation_strategy="enhanced"
            ),
            32: BitResolutionPhase(
                phase_id="32bit_advanced",
                bit_depth=32,
                entropy_threshold=5.5,
                complexity_limit=0.9,
                tensor_dimensions=[8, 8, 8],
                allocation_strategy="advanced"
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
    
    def _validate_components(self):
        """Validate that required components are available."""
        if not ZPE_MODULES_AVAILABLE and self.zpe_core is None:
            warn("ZPE core not available - ZPE features will be disabled")
        
        if not MATRIX_MAPPER_AVAILABLE and self.matrix_mapper is None:
            warn("Matrix mapper not available - matrix features will be disabled")
        
        if not DLT_WAVEFORM_AVAILABLE and self.dlt_waveform_engine is None:
            warn("DLT waveform engine not available - waveform features will be disabled")

    def allocate(
        self,
        execution_packet: Dict[str, Any],
        cycles: Optional[Sequence[str]] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ProfitAllocationResult:
        """
        Enhanced profit allocation with matrix mapper and tensor scoring integration.

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

                    safe_print(f"✅ Matrix allocation: basket={matrix_basket_id}, tensor_score={tensor_score:.4f}")

                except Exception as e:
                    safe_print(f"⚠️ Matrix mapper integration error: {e}")

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
                                    safe_print(f"✅ DLT waveform integration: {integration_result}")

                except Exception as e:
                    safe_print(f"⚠️ DLT waveform integration error: {e}")

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

                    safe_print(f"✅ ZPE integration: efficiency={zpe_efficiency:.4f}, reinjection={zpe_reinjection:.4f}")

                except Exception as e:
                    safe_print(f"⚠️ ZPE integration error: {e}")

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

            safe_print(f"✅ Enhanced allocation completed: tensor_score={tensor_score:.4f}, bit_phase={bit_phase}")
            return result

        except Exception as e:
            error_msg = safe_format_error(e, "Enhanced profit allocation")
            safe_print(f"❌ Enhanced allocation failed: {error_msg}")

            return ProfitAllocationResult(
                success=False,
                allocated_packet=execution_packet,
                allocation_strategy=self.allocation_strategy,
                metadata={'error': str(e)}
            )

    def _generate_allocation_hash(
        self, 
        execution_packet: Dict[str, Any], 
        market_data: Optional[Dict[str, Any]]
    ) -> str:
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
            # Analyze market volatility
            volatility = market_data.get('volatility', 0.5)
            
            # Analyze market volume
            volume = market_data.get('volume', 1000)
            
            # Analyze price momentum
            momentum = market_data.get('momentum', 0.0)
            
            # Calculate complexity score
            complexity = (
                volatility * 0.4
                + unified_math.min(volume / 10000, 1.0) * 0.3
                + unified_math.abs(momentum) * 0.3
            )
            
            # Select bit phase based on complexity
            if complexity < 0.3:
                return 8  # Low complexity - 8-bit
            elif complexity < 0.6:
                return 16  # Medium complexity - 16-bit
            elif complexity < 0.8:
                return 32  # High complexity - 32-bit
            else:
                return 42  # Very high complexity - 42-bit (quantum)
                
        except Exception as e:
            logger.error(f"Error determining optimal bit phase: {e}")
            return 16  # Default to 16-bit

    def _calculate_zpe_metrics(
        self, 
        market_data: Dict[str, Any], 
        total_profit: float
    ) -> Dict[str, Any]:
        """Calculate ZPE (Zero Point Energy) metrics for allocation optimization."""
        try:
            if not self.zpe_core:
                return {'efficiency': 0.5, 'reinjection': 0.0, 'thermal_history': {}}
            
            # Calculate ZPE efficiency based on market conditions
            volatility = market_data.get('volatility', 0.5)
            volume = market_data.get('volume', 1000)
            
            # ZPE efficiency calculation
            efficiency = 0.5 + (0.3 * (1.0 - volatility)) + (0.2 * unified_math.min(volume / 10000, 1.0))
            efficiency = unified_math.max(0.0, unified_math.min(1.0, efficiency))
            
            # ZPE reinjection calculation
            reinjection = 0.1 * volatility * (total_profit / 1000)  # Reinjection based on volatility and profit
            reinjection = unified_math.max(0.0, reinjection)
            
            # Thermal history
            thermal_history = {
                'timestamp': time.time(),
                'efficiency': efficiency,
                'reinjection': reinjection,
                'volatility': volatility,
                'volume': volume
            }
            
            return {
                'efficiency': efficiency,
                'reinjection': reinjection,
                'thermal_history': thermal_history
            }
            
        except Exception as e:
            logger.error(f"Error calculating ZPE metrics: {e}")
            return {'efficiency': 0.5, 'reinjection': 0.0, 'thermal_history': {}}

    def _adjust_allocation_with_metrics(
        self, 
        allocation: Dict[str, float], 
        zpe_efficiency: float, 
        tensor_score: float, 
        bit_phase: Optional[int]
    ):
        """Adjust allocation based on ZPE efficiency and tensor score."""
        try:
            # Calculate adjustment factor
            zpe_factor = 1.0 + (zpe_efficiency - 0.5) * 0.4  # ±20% based on ZPE efficiency
            tensor_factor = 1.0 + (tensor_score - 0.5) * 0.3  # ±15% based on tensor score
            
            # Apply adjustments
            adjustment_factor = zpe_factor * tensor_factor
            
            for cycle in allocation:
                allocation[cycle] *= adjustment_factor
                
        except Exception as e:
            logger.error(f"Error adjusting allocation with metrics: {e}")

    def _store_allocation_history(
        self, 
        execution_packet: Dict[str, Any], 
        tensor_score: float, 
        bit_phase: Optional[int]
    ):
        """Store allocation history for analysis and optimization."""
        try:
            history_entry = {
                'timestamp': time.time(),
                'tensor_score': tensor_score,
                'bit_phase': bit_phase,
                'allocation_strategy': self.allocation_strategy,
                'execution_packet_id': execution_packet.get('packet_id', 'unknown')
            }
            
            self.allocation_history.append(history_entry)
            self.tensor_score_history.append(tensor_score)
            
            # Keep history size manageable
            if len(self.allocation_history) > 1000:
                self.allocation_history = self.allocation_history[-500:]
            if len(self.tensor_score_history) > 1000:
                self.tensor_score_history = self.tensor_score_history[-500:]
                
        except Exception as e:
            logger.error(f"Error storing allocation history: {e}")

    def get_allocation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive allocation statistics."""
        try:
            if not self.tensor_score_history:
                return {'error': 'No allocation history available'}
            
            avg_tensor_score = sum(self.tensor_score_history) / len(self.tensor_score_history)
            max_tensor_score = max(self.tensor_score_history)
            min_tensor_score = min(self.tensor_score_history)
            
            return {
                'total_allocations': len(self.allocation_history),
                'average_tensor_score': avg_tensor_score,
                'max_tensor_score': max_tensor_score,
                'min_tensor_score': min_tensor_score,
                'allocation_strategy': self.allocation_strategy,
                'components_available': {
                    'zpe': ZPE_MODULES_AVAILABLE,
                    'matrix_mapper': MATRIX_MAPPER_AVAILABLE,
                    'dlt_waveform': DLT_WAVEFORM_AVAILABLE
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting allocation statistics: {e}")
            return {'error': str(e)}


# Legacy compatibility function
def allocate_profit_cycles(
    execution_packet: Dict[str, Any],
    cycles: Optional[Sequence[str]] = None,
    market_data: Optional[Dict[str, Any]] = None
) -> ProfitAllocationResult:
    """Legacy function for backward compatibility."""
    allocator = ProfitCycleAllocator()
    return allocator.allocate(execution_packet, cycles, market_data)
