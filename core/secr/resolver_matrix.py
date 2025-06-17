"""
SECR Resolver Matrix
===================

Hierarchical resolver system that generates targeted patches for
different failure types with inheritance from parent groups.
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import time

from .failure_logger import FailureKey, FailureGroup, FailureSubGroup
from .allocator import AllocationDecision, ComputeLane

logger = logging.getLogger(__name__)

@dataclass
class PatchConfig:
    """Configuration patch to apply to the system"""
    strategy_mod: Optional[Dict[str, Any]] = None
    engine_mod: Optional[Dict[str, Any]] = None
    risk_mod: Optional[Dict[str, Any]] = None
    timing_mod: Optional[Dict[str, Any]] = None
    persistence_ticks: int = 16  # How long to keep the patch active
    priority: int = 1  # Higher priority patches override lower ones
    metadata: Optional[Dict[str, Any]] = None
    
    def merge_with(self, other: 'PatchConfig') -> 'PatchConfig':
        """Merge with another patch config, prioritizing higher priority"""
        if other.priority > self.priority:
            return other.merge_with(self)
        
        # Merge dictionaries, with self taking precedence
        merged_strategy = {**(other.strategy_mod or {}), **(self.strategy_mod or {})}
        merged_engine = {**(other.engine_mod or {}), **(self.engine_mod or {})}
        merged_risk = {**(other.risk_mod or {}), **(self.risk_mod or {})}
        merged_timing = {**(other.timing_mod or {}), **(self.timing_mod or {})}
        merged_metadata = {**(other.metadata or {}), **(self.metadata or {})}
        
        return PatchConfig(
            strategy_mod=merged_strategy if merged_strategy else None,
            engine_mod=merged_engine if merged_engine else None,
            risk_mod=merged_risk if merged_risk else None,
            timing_mod=merged_timing if merged_timing else None,
            persistence_ticks=max(self.persistence_ticks, other.persistence_ticks),
            priority=self.priority,
            metadata=merged_metadata if merged_metadata else None
        )

class BaseResolver(ABC):
    """Base resolver interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def resolve(self, failure_key: FailureKey, allocation: AllocationDecision) -> PatchConfig:
        """Generate a patch configuration for the failure"""
        pass
    
    def get_fast_patch(self, failure_key: FailureKey) -> PatchConfig:
        """Generate a fast patch for immediate application"""
        return PatchConfig(
            strategy_mod={'emergency_mode': True},
            persistence_ticks=4,
            priority=10,
            metadata={'patch_type': 'fast', 'reason': 'emergency_fallback'}
        )
    
    def get_long_patch(self, failure_key: FailureKey) -> PatchConfig:
        """Generate a long-term patch for system improvement"""
        return PatchConfig(
            strategy_mod={'learning_mode': True},
            persistence_ticks=64,
            priority=1,
            metadata={'patch_type': 'long', 'reason': 'system_improvement'}
        )

class PerformanceResolver(BaseResolver):
    """Resolver for PERF group failures"""
    
    def resolve(self, failure_key: FailureKey, allocation: AllocationDecision) -> PatchConfig:
        """Resolve performance-related failures"""
        severity = failure_key.severity
        subgroup = failure_key.subgroup
        
        if subgroup == FailureSubGroup.GPU_LAG:
            return self._resolve_gpu_lag(failure_key, allocation, severity)
        elif subgroup == FailureSubGroup.CPU_STALL:
            return self._resolve_cpu_stall(failure_key, allocation, severity)
        elif subgroup == FailureSubGroup.RAM_PRESSURE:
            return self._resolve_ram_pressure(failure_key, allocation, severity)
        else:
            return self._default_perf_resolution(failure_key, allocation, severity)
    
    def _resolve_gpu_lag(self, failure_key: FailureKey, allocation: AllocationDecision, severity: float) -> PatchConfig:
        """Resolve GPU lag issues"""
        base_config = {
            'gpu_batch_reduction': min(0.5, severity),
            'gpu_timeout_extension': int(severity * 200),  # Additional ms
            'enable_cpu_fallback': True
        }
        
        # High severity - aggressive intervention
        if severity > 0.8:
            strategy_mod = {
                **base_config,
                'force_cpu_mode': True,
                'gpu_cooldown_ticks': 32
            }
            engine_mod = {
                'gpu_queue_limit': max(1, int(8 * (1 - severity))),
                'gpu_memory_limit': 0.7
            }
            persistence = 32
            
        # Medium severity - balanced approach
        elif severity > 0.5:
            strategy_mod = {
                **base_config,
                'hybrid_processing': True,
                'gpu_priority_reduction': 0.3
            }
            engine_mod = {
                'gpu_queue_limit': max(2, int(16 * (1 - severity))),
                'gpu_memory_limit': 0.8
            }
            persistence = 16
            
        # Low severity - gentle adjustment
        else:
            strategy_mod = {
                **base_config,
                'gpu_batch_size_multiplier': 0.8
            }
            engine_mod = {
                'gpu_queue_limit': max(4, int(32 * (1 - severity)))
            }
            persistence = 8
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            engine_mod=engine_mod,
            persistence_ticks=persistence,
            priority=int(severity * 10),
            metadata={
                'resolver': 'gpu_lag',
                'severity': severity,
                'allocation_lane': allocation.lane.value
            }
        )
    
    def _resolve_cpu_stall(self, failure_key: FailureKey, allocation: AllocationDecision, severity: float) -> PatchConfig:
        """Resolve CPU stall issues"""
        base_config = {
            'cpu_thread_reduction': min(0.4, severity),
            'cpu_priority_adjustment': -int(severity * 5),
            'enable_gpu_offload': True
        }
        
        if severity > 0.6:
            strategy_mod = {
                **base_config,
                'force_gpu_mode': allocation.lane == ComputeLane.GPU,
                'cpu_cooldown_ticks': 16
            }
            engine_mod = {
                'cpu_core_limit': max(1, int(4 * (1 - severity))),
                'thread_pool_size': max(2, int(8 * (1 - severity)))
            }
            persistence = 24
        else:
            strategy_mod = {
                **base_config,
                'cpu_batch_size_multiplier': 0.7
            }
            engine_mod = {
                'thread_pool_size': max(4, int(16 * (1 - severity)))
            }
            persistence = 12
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            engine_mod=engine_mod,
            persistence_ticks=persistence,
            priority=int(severity * 8),
            metadata={'resolver': 'cpu_stall', 'severity': severity}
        )
    
    def _resolve_ram_pressure(self, failure_key: FailureKey, allocation: AllocationDecision, severity: float) -> PatchConfig:
        """Resolve RAM pressure issues"""
        strategy_mod = {
            'memory_optimization': True,
            'batch_size_reduction': severity * 0.6,
            'garbage_collection_frequency': min(10, int(severity * 20))
        }
        
        engine_mod = {
            'max_memory_usage': max(0.3, 0.9 - severity),
            'buffer_size_limit': max(1024, int(8192 * (1 - severity))),
            'enable_streaming': True
        }
        
        risk_mod = {
            'position_size_reduction': severity * 0.3,
            'max_concurrent_orders': max(1, int(10 * (1 - severity)))
        }
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            engine_mod=engine_mod,
            risk_mod=risk_mod,
            persistence_ticks=int(severity * 40),
            priority=int(severity * 9),
            metadata={'resolver': 'ram_pressure', 'severity': severity}
        )
    
    def _default_perf_resolution(self, failure_key: FailureKey, allocation: AllocationDecision, severity: float) -> PatchConfig:
        """Default performance resolution"""
        return PatchConfig(
            strategy_mod={
                'performance_mode': 'conservative',
                'batch_size_multiplier': 0.8
            },
            engine_mod={
                'resource_monitoring': True,
                'adaptive_throttling': True
            },
            persistence_ticks=16,
            priority=5,
            metadata={'resolver': 'default_perf', 'severity': severity}
        )

class OrderResolver(BaseResolver):
    """Resolver for ORDER group failures"""
    
    def resolve(self, failure_key: FailureKey, allocation: AllocationDecision) -> PatchConfig:
        """Resolve order execution failures"""
        subgroup = failure_key.subgroup
        severity = failure_key.severity
        
        if subgroup == FailureSubGroup.BATCH_MISS:
            return self._resolve_batch_miss(failure_key, severity)
        elif subgroup == FailureSubGroup.SLIP_DRIFT:
            return self._resolve_slip_drift(failure_key, severity)
        elif subgroup == FailureSubGroup.PARTIAL_FILL:
            return self._resolve_partial_fill(failure_key, severity)
        else:
            return self._default_order_resolution(failure_key, severity)
    
    def _resolve_batch_miss(self, failure_key: FailureKey, severity: float) -> PatchConfig:
        """Resolve batch miss issues"""
        strategy_mod = {
            'order_timeout_padding': int(severity * 500),  # Additional ms
            'batch_size_reduction': severity * 0.5,
            'execution_priority': 'timing',
            'enable_aggressive_retries': True
        }
        
        timing_mod = {
            'tick_buffer_ms': int(severity * 100),
            'order_confirmation_timeout': max(1000, int(3000 * (1 + severity))),
            'batch_dispatch_delay': max(10, int(50 * (1 - severity)))
        }
        
        risk_mod = {
            'icap_adjustment': min(0.1, severity * 0.2),  # Increase ICAP threshold
            'max_order_attempts': max(2, int(5 * (1 + severity)))
        }
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            timing_mod=timing_mod,
            risk_mod=risk_mod,
            persistence_ticks=int(severity * 32),
            priority=9,
            metadata={'resolver': 'batch_miss', 'severity': severity}
        )
    
    def _resolve_slip_drift(self, failure_key: FailureKey, severity: float) -> PatchConfig:
        """Resolve price slippage issues"""
        slip_pct = failure_key.ctx.get('price_slip_pct', 0)
        
        strategy_mod = {
            'slippage_tolerance': max(0.05, slip_pct * 1.5),
            'price_improvement_hunting': True,
            'order_size_fragmentation': severity > 0.6
        }
        
        risk_mod = {
            'icap_adjustment': min(0.15, slip_pct * 2),
            'spread_buffer_multiplier': 1 + severity,
            'max_slippage_threshold': max(0.1, slip_pct * 1.2)
        }
        
        timing_mod = {
            'price_staleness_limit': max(100, int(500 * (1 - severity))),
            'order_book_refresh_rate': max(50, int(200 * (1 - severity)))
        }
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            risk_mod=risk_mod,
            timing_mod=timing_mod,
            persistence_ticks=24,
            priority=7,
            metadata={'resolver': 'slip_drift', 'severity': severity, 'slip_pct': slip_pct}
        )
    
    def _resolve_partial_fill(self, failure_key: FailureKey, severity: float) -> PatchConfig:
        """Resolve partial fill issues"""
        fill_ratio = failure_key.ctx.get('fill_ratio', 1.0)
        
        strategy_mod = {
            'order_management': 'aggressive',
            'partial_fill_threshold': max(0.8, fill_ratio - 0.1),
            'order_splitting': True,
            'market_impact_consideration': True
        }
        
        risk_mod = {
            'position_sizing_adjustment': severity * 0.2,
            'liquidity_requirement': 1 + severity
        }
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            risk_mod=risk_mod,
            persistence_ticks=16,
            priority=6,
            metadata={'resolver': 'partial_fill', 'severity': severity, 'fill_ratio': fill_ratio}
        )
    
    def _default_order_resolution(self, failure_key: FailureKey, severity: float) -> PatchConfig:
        """Default order resolution"""
        return PatchConfig(
            strategy_mod={
                'order_execution': 'conservative',
                'risk_management': 'enhanced'
            },
            risk_mod={
                'icap_adjustment': severity * 0.1
            },
            persistence_ticks=12,
            priority=4,
            metadata={'resolver': 'default_order', 'severity': severity}
        )

class EntropyResolver(BaseResolver):
    """Resolver for ENTROPY group failures"""
    
    def resolve(self, failure_key: FailureKey, allocation: AllocationDecision) -> PatchConfig:
        """Resolve entropy/mathematical failures"""
        subgroup = failure_key.subgroup
        severity = failure_key.severity
        
        if subgroup == FailureSubGroup.ENTROPY_SPIKE:
            return self._resolve_entropy_spike(failure_key, severity)
        elif subgroup == FailureSubGroup.ICAP_COLLAPSE:
            return self._resolve_icap_collapse(failure_key, severity)
        elif subgroup == FailureSubGroup.PHASE_INVERT:
            return self._resolve_phase_invert(failure_key, severity)
        else:
            return self._default_entropy_resolution(failure_key, severity)
    
    def _resolve_entropy_spike(self, failure_key: FailureKey, severity: float) -> PatchConfig:
        """Resolve entropy spike issues"""
        delta_psi = failure_key.ctx.get('delta_psi', 0)
        
        strategy_mod = {
            'entropy_gate_threshold': min(1.0, severity * 2),
            'phase_lock_duration': int(severity * 16),
            'adaptive_window_sizing': True,
            'entropy_smoothing_factor': max(0.1, 1 - severity)
        }
        
        risk_mod = {
            'corridor_width_multiplier': 1 + severity,
            'profit_corridor_adjustment': severity * 0.3,
            'position_freeze_threshold': severity * 0.8
        }
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            risk_mod=risk_mod,
            persistence_ticks=int(severity * 48),
            priority=8,
            metadata={'resolver': 'entropy_spike', 'severity': severity, 'delta_psi': delta_psi}
        )
    
    def _resolve_icap_collapse(self, failure_key: FailureKey, severity: float) -> PatchConfig:
        """Resolve ICAP collapse issues"""
        icap_value = failure_key.ctx.get('icap_value', 0)
        
        strategy_mod = {
            'icap_recovery_mode': True,
            'entry_suspension': severity > 0.8,
            'probability_recalibration': True
        }
        
        risk_mod = {
            'icap_floor': max(0.1, icap_value * 2),
            'corridor_expansion': severity * 0.5,
            'exit_only_mode': severity > 0.9
        }
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            risk_mod=risk_mod,
            persistence_ticks=int(severity * 64),
            priority=10,
            metadata={'resolver': 'icap_collapse', 'severity': severity, 'icap_value': icap_value}
        )
    
    def _resolve_phase_invert(self, failure_key: FailureKey, severity: float) -> PatchConfig:
        """Resolve phase inversion issues"""
        strategy_mod = {
            'phase_detection': 'enhanced',
            'inversion_compensation': True,
            'temporal_buffer_extension': int(severity * 8)
        }
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            persistence_ticks=int(severity * 32),
            priority=7,
            metadata={'resolver': 'phase_invert', 'severity': severity}
        )
    
    def _default_entropy_resolution(self, failure_key: FailureKey, severity: float) -> PatchConfig:
        """Default entropy resolution"""
        return PatchConfig(
            strategy_mod={
                'mathematical_stability': 'enhanced',
                'entropy_monitoring': True
            },
            persistence_ticks=16,
            priority=5,
            metadata={'resolver': 'default_entropy', 'severity': severity}
        )

class ThermalResolver(BaseResolver):
    """Resolver for THERMAL group failures"""
    
    def resolve(self, failure_key: FailureKey, allocation: AllocationDecision) -> PatchConfig:
        """Resolve thermal issues"""
        severity = failure_key.severity
        cpu_temp = failure_key.ctx.get('cpu_temp', 0)
        gpu_temp = failure_key.ctx.get('gpu_temp', 0)
        
        strategy_mod = {
            'thermal_throttling': True,
            'processing_reduction': severity * 0.5,
            'cooling_pause_duration': int(severity * 30)
        }
        
        engine_mod = {
            'cpu_throttle_factor': max(0.3, 1 - severity),
            'gpu_throttle_factor': max(0.2, 1 - severity),
            'thermal_monitoring_interval': max(1, int(10 * (1 - severity)))
        }
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            engine_mod=engine_mod,
            persistence_ticks=int(severity * 60),
            priority=9,
            metadata={
                'resolver': 'thermal', 
                'severity': severity, 
                'cpu_temp': cpu_temp, 
                'gpu_temp': gpu_temp
            }
        )

class NetworkResolver(BaseResolver):
    """Resolver for NET group failures"""
    
    def resolve(self, failure_key: FailureKey, allocation: AllocationDecision) -> PatchConfig:
        """Resolve network issues"""
        severity = failure_key.severity
        api_latency = failure_key.ctx.get('api_response_ms', 0)
        
        strategy_mod = {
            'network_retry_strategy': 'exponential_backoff',
            'api_timeout_multiplier': 1 + severity,
            'connection_pooling': True
        }
        
        timing_mod = {
            'api_timeout_base': max(5000, int(api_latency * 2)),
            'retry_delay_ms': max(100, int(severity * 1000)),
            'connection_timeout': max(3000, int(5000 * (1 + severity)))
        }
        
        return PatchConfig(
            strategy_mod=strategy_mod,
            timing_mod=timing_mod,
            persistence_ticks=int(severity * 32),
            priority=6,
            metadata={'resolver': 'network', 'severity': severity, 'api_latency': api_latency}
        )

class ResolverRegistry:
    """Registry for managing resolvers with inheritance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resolvers: Dict[str, BaseResolver] = {}
        self.resolver_inheritance: Dict[str, str] = {}
        
        # Initialize default resolvers
        self._init_default_resolvers()
        
    def _init_default_resolvers(self):
        """Initialize default resolver instances"""
        self.resolvers['PERF'] = PerformanceResolver(self.config.get('perf', {}))
        self.resolvers['ORDER'] = OrderResolver(self.config.get('order', {}))
        self.resolvers['ENTROPY'] = EntropyResolver(self.config.get('entropy', {}))
        self.resolvers['THERMAL'] = ThermalResolver(self.config.get('thermal', {}))
        self.resolvers['NET'] = NetworkResolver(self.config.get('net', {}))
        
        # Set up inheritance (subgroup -> parent group)
        subgroup_mappings = {
            f'PERF/{sub.value}': 'PERF' for sub in [
                FailureSubGroup.GPU_LAG, FailureSubGroup.CPU_STALL, FailureSubGroup.RAM_PRESSURE
            ]
        }
        subgroup_mappings.update({
            f'ORDER/{sub.value}': 'ORDER' for sub in [
                FailureSubGroup.BATCH_MISS, FailureSubGroup.SLIP_DRIFT, FailureSubGroup.PARTIAL_FILL
            ]
        })
        subgroup_mappings.update({
            f'ENTROPY/{sub.value}': 'ENTROPY' for sub in [
                FailureSubGroup.ENTROPY_SPIKE, FailureSubGroup.ICAP_COLLAPSE, FailureSubGroup.PHASE_INVERT
            ]
        })
        subgroup_mappings.update({
            f'THERMAL/{sub.value}': 'THERMAL' for sub in [
                FailureSubGroup.THERMAL_HALT, FailureSubGroup.FAN_STALL
            ]
        })
        subgroup_mappings.update({
            f'NET/{sub.value}': 'NET' for sub in [
                FailureSubGroup.API_TIMEOUT, FailureSubGroup.SOCKET_DROP
            ]
        })
        
        self.resolver_inheritance.update(subgroup_mappings)
    
    def register_resolver(self, key: str, resolver: BaseResolver, parent: Optional[str] = None):
        """Register a custom resolver"""
        self.resolvers[key] = resolver
        if parent:
            self.resolver_inheritance[key] = parent
    
    def get_resolver(self, failure_key: FailureKey) -> BaseResolver:
        """Get appropriate resolver with inheritance fallback"""
        # Try specific subgroup resolver first
        subgroup_key = f"{failure_key.group.value}/{failure_key.subgroup.value}"
        if subgroup_key in self.resolvers:
            return self.resolvers[subgroup_key]
        
        # Fall back to parent group resolver
        group_key = failure_key.group.value
        if group_key in self.resolvers:
            return self.resolvers[group_key]
        
        # Ultimate fallback to PERF resolver
        return self.resolvers.get('PERF', self.resolvers[list(self.resolvers.keys())[0]])

class ResolverMatrix:
    """Main resolver matrix coordinator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = ResolverRegistry(config)
        self.active_patches: Dict[str, Tuple[PatchConfig, int, float]] = {}  # hash -> (patch, ticks_remaining, applied_at)
        self.patch_history: List[Tuple[str, PatchConfig, float, float]] = []  # hash, patch, applied_at, resolved_at
        
    def resolve_failure(self, failure_key: FailureKey, allocation: AllocationDecision) -> PatchConfig:
        """
        Resolve a failure and generate patch configuration
        
        Args:
            failure_key: The failure to resolve
            allocation: Resource allocation decision
            
        Returns:
            PatchConfig to apply to the system
        """
        # Get appropriate resolver
        resolver = self.registry.get_resolver(failure_key)
        
        # Generate patch
        patch = resolver.resolve(failure_key, allocation)
        
        # Store active patch
        self.active_patches[failure_key.hash] = (patch, patch.persistence_ticks, time.time())
        
        logger.info(f"Generated patch for {failure_key.hash}: "
                   f"priority={patch.priority}, persistence={patch.persistence_ticks}")
        
        return patch
    
    def get_merged_patches(self) -> PatchConfig:
        """Get all active patches merged by priority"""
        if not self.active_patches:
            return PatchConfig()
        
        # Sort patches by priority (highest first)
        sorted_patches = sorted(
            [(patch, ticks, applied_at) for patch, ticks, applied_at in self.active_patches.values()],
            key=lambda x: x[0].priority,
            reverse=True
        )
        
        # Start with highest priority patch
        merged = sorted_patches[0][0]
        
        # Merge with lower priority patches
        for patch, ticks, applied_at in sorted_patches[1:]:
            merged = merged.merge_with(patch)
        
        return merged
    
    def tick_update(self) -> None:
        """Update patch persistence counters"""
        expired_patches = []
        current_time = time.time()
        
        for hash_key, (patch, ticks_remaining, applied_at) in self.active_patches.items():
            new_ticks = ticks_remaining - 1
            
            if new_ticks <= 0:
                expired_patches.append(hash_key)
                # Move to history
                self.patch_history.append((hash_key, patch, applied_at, current_time))
            else:
                self.active_patches[hash_key] = (patch, new_ticks, applied_at)
        
        # Remove expired patches
        for hash_key in expired_patches:
            del self.active_patches[hash_key]
            logger.debug(f"Patch {hash_key} expired")
        
        # Trim history if too long
        if len(self.patch_history) > 1000:
            self.patch_history = self.patch_history[-1000:]
    
    def force_expire_patch(self, failure_hash: str) -> bool:
        """Force expire a specific patch"""
        if failure_hash in self.active_patches:
            patch, ticks, applied_at = self.active_patches[failure_hash]
            self.patch_history.append((failure_hash, patch, applied_at, time.time()))
            del self.active_patches[failure_hash]
            logger.info(f"Force expired patch {failure_hash}")
            return True
        return False
    
    def get_patch_stats(self) -> Dict[str, Any]:
        """Get patch statistics"""
        active_count = len(self.active_patches)
        total_applied = len(self.patch_history) + active_count
        
        # Group stats by resolver type
        resolver_counts = {}
        for hash_key, (patch, _, _) in self.active_patches.items():
            resolver = patch.metadata.get('resolver', 'unknown') if patch.metadata else 'unknown'
            resolver_counts[resolver] = resolver_counts.get(resolver, 0) + 1
        
        # Average persistence
        if self.active_patches:
            avg_persistence = sum(patch.persistence_ticks for patch, _, _ in self.active_patches.values()) / active_count
        else:
            avg_persistence = 0
        
        return {
            'active_patches': active_count,
            'total_applied': total_applied,
            'resolver_counts': resolver_counts,
            'avg_persistence': avg_persistence,
            'patch_history_size': len(self.patch_history)
        } 