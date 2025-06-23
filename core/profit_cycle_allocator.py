#!/usr/bin/env python3
"""Profit Cycle Allocator with ZPE Mathematical Framework Integration.

Allocates trade volume or capital across strategy cycles with ZPE thermal efficiency
and profit reinjection calculations. The ZPE framework provides rotational profit
optimization beyond traditional statistical allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Optional
from datetime import datetime
import logging

# Import ZPE Mathematical Framework
try:
    from core.zpe_core import ZPECore
    ZPE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ZPE modules not available: {e}")
    ZPE_MODULES_AVAILABLE = False

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
    """Result of profit cycle allocation with ZPE integration."""
    success: bool
    allocated_packet: Dict[str, Any]
    allocation_strategy: str
    # ZPE Integration Fields
    zpe_efficiency: float = 0.0
    zpe_reinjection: float = 0.0
    total_profit: float = 0.0
    thermal_history: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass(slots=True)
class ProfitCycleAllocator:
    """Enhanced profit cycle allocator with ZPE mathematical framework integration."""

    allocation_strategy: str = "zpe_enhanced"
    zpe_core: Optional[ZPECore] = None
    
    def __post_init__(self):
        """Initialize ZPE core if available."""
        if ZPE_MODULES_AVAILABLE:
            self.zpe_core = ZPECore()
            safe_print("🔄 Profit Cycle Allocator initialized with ZPE integration")
        else:
            safe_print("⚠️ Profit Cycle Allocator initialized without ZPE integration")

    def allocate(
        self,
        execution_packet: Dict[str, Any],
        cycles: Sequence[str] | None = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ProfitAllocationResult:
        """Allocate profit cycles with ZPE mathematical framework integration.

        Parameters
        ----------
        execution_packet
            Packet produced by GhostStrategyIntegrator.
        cycles
            Optional list of cycle names. If *None*, a single 'default'
            cycle is assumed.
        market_data
            Optional market data for ZPE calculations.
            
        Returns
        -------
        ProfitAllocationResult
            Enhanced allocation result with ZPE calculations.
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
            
            # ZPE Integration
            zpe_efficiency = 0.0
            zpe_reinjection = 0.0
            total_profit = execution_packet.get("actual_profit", 0.0)
            thermal_history = None
            
            if self.zpe_core and market_data:
                try:
                    # Calculate thermal efficiency
                    profit_generated = execution_packet.get("actual_profit", 0.0)
                    capital_exposure = execution_packet.get("capital_exposure", 1.0)
                    zpe_efficiency = self.zpe_core.calculate_thermal_efficiency(
                        profit_generated, capital_exposure
                    )
                    
                    # Calculate profit reinjection
                    profit_delta = execution_packet.get("profit_delta", 0.0)
                    market_heat = market_data.get("market_heat", 0.5)
                    zpe_reinjection = self.zpe_core.calculate_profit_reinjection(
                        profit_delta, market_heat
                    )
                    
                    # Update total profit with reinjection
                    total_profit = profit_generated + zpe_reinjection
                    
                    # Get thermal history
                    thermal_history = {
                        'efficiency_history': self.zpe_core.thermal_history[-10:] if self.zpe_core.thermal_history else [],
                        'current_efficiency': zpe_efficiency,
                        'reinjection_rate': zpe_reinjection / max(profit_delta, 1.0)
                    }
                    
                    # Update allocation with ZPE data
                    execution_packet["zpe_efficiency"] = zpe_efficiency
                    execution_packet["zpe_reinjection"] = zpe_reinjection
                    execution_packet["total_profit"] = total_profit
                    execution_packet["thermal_history"] = thermal_history
                    
                    # Adjust allocation based on ZPE efficiency
                    if zpe_efficiency > 0.5:  # High efficiency
                        # Increase allocation for high-performing cycles
                        for cycle_name in allocation:
                            allocation[cycle_name] *= (1.0 + zpe_efficiency * 0.2)
                    elif zpe_efficiency < 0.2:  # Low efficiency
                        # Reduce allocation for low-performing cycles
                        for cycle_name in allocation:
                            allocation[cycle_name] *= (1.0 - (0.2 - zpe_efficiency) * 0.5)
                    
                    execution_packet["cycle_allocation"] = allocation
                    
                    safe_print(f"🔄 ZPE Allocation - Efficiency: {zpe_efficiency:.6f}, Reinjection: {zpe_reinjection:.6f}, Total: {total_profit:.6f}")
                    
                except Exception as e:
                    safe_print(f"⚠️ ZPE allocation failed: {safe_format_error(e, 'zpe_allocation')}")
            
            return ProfitAllocationResult(
                success=True,
                allocated_packet=execution_packet,
                allocation_strategy=self.allocation_strategy,
                zpe_efficiency=zpe_efficiency,
                zpe_reinjection=zpe_reinjection,
                total_profit=total_profit,
                thermal_history=thermal_history,
                metadata={
                    'zpe_integration': ZPE_MODULES_AVAILABLE,
                    'allocation_timestamp': datetime.now().isoformat(),
                    'cycles_allocated': len(allocation)
                }
            )
            
        except Exception as e:
            error_msg = safe_format_error(e, "profit_allocation")
            safe_print(f"❌ Profit allocation failed: {error_msg}")
            
            return ProfitAllocationResult(
                success=False,
                allocated_packet=execution_packet,
                allocation_strategy=self.allocation_strategy,
                metadata={'error': error_msg}
            )

    def get_zpe_metrics(self) -> Dict[str, Any]:
        """Get ZPE performance metrics."""
        if not self.zpe_core:
            return {'zpe_available': False}
        
        try:
            thermal_history = self.zpe_core.thermal_history
            recent_efficiencies = [entry['efficiency'] for entry in thermal_history[-10:]]
            
            return {
                'zpe_available': True,
                'thermal_history_length': len(thermal_history),
                'average_efficiency': sum(recent_efficiencies) / len(recent_efficiencies) if recent_efficiencies else 0.0,
                'max_efficiency': max(recent_efficiencies) if recent_efficiencies else 0.0,
                'min_efficiency': min(recent_efficiencies) if recent_efficiencies else 0.0,
                'recursion_depth': self.zpe_core.recursion_depth
            }
        except Exception as e:
            return {
                'zpe_available': True,
                'error': str(e)
            }


# Functional helper with ZPE integration
def allocate_profit_cycle(
    execution_packet: Dict[str, Any], 
    cycles: Sequence[str] | None = None,
    market_data: Optional[Dict[str, Any]] = None
) -> ProfitAllocationResult:
    """Enhanced stateless wrapper around ProfitCycleAllocator.allocate with ZPE integration."""
    allocator = ProfitCycleAllocator()
    return allocator.allocate(execution_packet, cycles, market_data)


# Legacy function for backward compatibility
def allocate_profit_cycle_legacy(
    execution_packet: Dict[str, Any], 
    cycles: Sequence[str] | None = None
) -> Dict[str, Any]:
    """Legacy function for backward compatibility without ZPE integration."""
    result = allocate_profit_cycle(execution_packet, cycles)
    return result.allocated_packet if result.success else execution_packet
