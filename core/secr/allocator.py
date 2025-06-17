"""
SECR Resource Allocator
======================

Dynamic resource allocation system that routes computation between
CPU, GPU, and hybrid modes based on real-time pressure analysis.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
import psutil
import time

from .failure_logger import FailureKey, FailureGroup, FailureSubGroup, PressureIndex

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class ComputeLane(Enum):
    """Available compute lanes"""
    CPU = "CPU"
    GPU = "GPU"
    HYBRID = "HYBRID"
    SPLIT = "SPLIT"

@dataclass
class AllocationDecision:
    """Resource allocation decision"""
    lane: ComputeLane
    reason: str
    confidence: float
    batch_size_modifier: float = 1.0
    priority_boost: bool = False
    fallback_lane: Optional[ComputeLane] = None

@dataclass
class ResourceMetrics:
    """Current resource utilization metrics"""
    cpu_usage: float
    gpu_usage: float
    ram_usage: float
    gpu_memory_usage: float
    cpu_temp: float
    gpu_temp: float
    
    @classmethod
    def capture_current(cls) -> 'ResourceMetrics':
        """Capture current system resource metrics"""
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            ram_usage = psutil.virtual_memory().percent
            
            # Get CPU temperature if available
            cpu_temp = 0.0
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temp = max(temp.current for temp in temps['coretemp'])
            except (AttributeError, KeyError):
                pass
            
            # GPU metrics
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            gpu_temp = 0.0
            
            if GPU_AVAILABLE:
                try:
                    # Get GPU utilization if available
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = util.gpu
                    gpu_memory_usage = util.memory
                    
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temp = temp
                    
                except Exception:
                    # Fallback: use CuPy memory info if available
                    try:
                        meminfo = cp.cuda.runtime.memGetInfo()
                        used = meminfo[1] - meminfo[0]
                        gpu_memory_usage = (used / meminfo[1]) * 100
                    except Exception:
                        pass
            
            return cls(
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                ram_usage=ram_usage,
                gpu_memory_usage=gpu_memory_usage,
                cpu_temp=cpu_temp,
                gpu_temp=gpu_temp
            )
        except Exception as e:
            logger.warning(f"Error capturing resource metrics: {e}")
            return cls(0, 0, 0, 0, 0, 0)

class AllocationStrategy:
    """Base allocation strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Thresholds from config
        self.cpu_threshold = config.get('cpu_threshold', 80.0)
        self.gpu_threshold = config.get('gpu_threshold', 85.0)
        self.temp_threshold_cpu = config.get('temp_threshold_cpu', 85.0)
        self.temp_threshold_gpu = config.get('temp_threshold_gpu', 83.0)
        self.pressure_threshold = config.get('pressure_threshold', 0.7)
        
    def should_use_gpu(self, metrics: ResourceMetrics, pressure: PressureIndex) -> Tuple[bool, str]:
        """Determine if GPU should be used"""
        if not GPU_AVAILABLE:
            return False, "GPU not available"
            
        # GPU overheated
        if metrics.gpu_temp > self.temp_threshold_gpu:
            return False, f"GPU too hot: {metrics.gpu_temp}°C"
            
        # GPU overloaded
        if metrics.gpu_usage > self.gpu_threshold:
            return False, f"GPU overloaded: {metrics.gpu_usage}%"
            
        # GPU memory pressure
        if metrics.gpu_memory_usage > 90:
            return False, f"GPU memory pressure: {metrics.gpu_memory_usage}%"
            
        # High GPU pressure index
        if pressure.gpu_pressure > self.pressure_threshold:
            return False, f"GPU pressure too high: {pressure.gpu_pressure:.3f}"
            
        return True, "GPU available"
    
    def should_use_cpu(self, metrics: ResourceMetrics, pressure: PressureIndex) -> Tuple[bool, str]:
        """Determine if CPU should be used"""
        # CPU overheated
        if metrics.cpu_temp > self.temp_threshold_cpu:
            return False, f"CPU too hot: {metrics.cpu_temp}°C"
            
        # CPU overloaded
        if metrics.cpu_usage > self.cpu_threshold:
            return False, f"CPU overloaded: {metrics.cpu_usage}%"
            
        # High CPU pressure index
        if pressure.cpu_pressure > self.pressure_threshold:
            return False, f"CPU pressure too high: {pressure.cpu_pressure:.3f}"
            
        return True, "CPU available"

class PerformanceStrategy(AllocationStrategy):
    """Performance-focused allocation strategy"""
    
    def allocate(self, failure_key: FailureKey, metrics: ResourceMetrics) -> AllocationDecision:
        """Allocate resources for PERF group failures"""
        pressure = PressureIndex(**failure_key.ctx.get('pressures', {}))
        subgroup = failure_key.subgroup
        
        if subgroup == FailureSubGroup.GPU_LAG:
            return self._handle_gpu_lag(failure_key, metrics, pressure)
        elif subgroup == FailureSubGroup.CPU_STALL:
            return self._handle_cpu_stall(failure_key, metrics, pressure)
        elif subgroup == FailureSubGroup.RAM_PRESSURE:
            return self._handle_ram_pressure(failure_key, metrics, pressure)
        else:
            return self._default_allocation(metrics, pressure)
    
    def _handle_gpu_lag(self, failure_key: FailureKey, metrics: ResourceMetrics, pressure: PressureIndex) -> AllocationDecision:
        """Handle GPU lag by switching to CPU if possible"""
        severity = failure_key.severity
        
        # High severity GPU lag - force CPU
        if severity > 0.8:
            can_use_cpu, cpu_reason = self.should_use_cpu(metrics, pressure)
            if can_use_cpu:
                return AllocationDecision(
                    lane=ComputeLane.CPU,
                    reason=f"High GPU lag (severity={severity:.3f}), switching to CPU",
                    confidence=0.9,
                    batch_size_modifier=0.8,  # Reduce batch size for CPU
                    fallback_lane=ComputeLane.HYBRID
                )
            else:
                return AllocationDecision(
                    lane=ComputeLane.HYBRID,
                    reason=f"GPU lag + CPU unavailable: {cpu_reason}",
                    confidence=0.6,
                    batch_size_modifier=0.5
                )
        
        # Medium severity - try hybrid
        elif severity > 0.5:
            return AllocationDecision(
                lane=ComputeLane.HYBRID,
                reason=f"Medium GPU lag, using hybrid mode",
                confidence=0.7,
                batch_size_modifier=0.7
            )
        
        # Low severity - reduce GPU batch size
        else:
            return AllocationDecision(
                lane=ComputeLane.GPU,
                reason=f"Low GPU lag, reducing batch size",
                confidence=0.8,
                batch_size_modifier=0.6
            )
    
    def _handle_cpu_stall(self, failure_key: FailureKey, metrics: ResourceMetrics, pressure: PressureIndex) -> AllocationDecision:
        """Handle CPU stall by switching to GPU if possible"""
        severity = failure_key.severity
        
        can_use_gpu, gpu_reason = self.should_use_gpu(metrics, pressure)
        
        # High severity CPU stall - try GPU
        if severity > 0.6 and can_use_gpu:
            return AllocationDecision(
                lane=ComputeLane.GPU,
                reason=f"CPU stall (severity={severity:.3f}), switching to GPU",
                confidence=0.9,
                priority_boost=True,
                fallback_lane=ComputeLane.SPLIT
            )
        
        # GPU not available or medium severity - split workload
        elif severity > 0.4:
            return AllocationDecision(
                lane=ComputeLane.SPLIT,
                reason=f"CPU stall, splitting workload. GPU status: {gpu_reason}",
                confidence=0.7,
                batch_size_modifier=0.5
            )
        
        # Low severity - continue with CPU but reduce batch
        else:
            return AllocationDecision(
                lane=ComputeLane.CPU,
                reason=f"Low CPU stall, reducing batch size",
                confidence=0.6,
                batch_size_modifier=0.7
            )
    
    def _handle_ram_pressure(self, failure_key: FailureKey, metrics: ResourceMetrics, pressure: PressureIndex) -> AllocationDecision:
        """Handle RAM pressure by reducing memory usage"""
        severity = failure_key.severity
        
        if severity > 0.8:
            return AllocationDecision(
                lane=ComputeLane.SPLIT,
                reason=f"High RAM pressure (severity={severity:.3f}), splitting workload",
                confidence=0.8,
                batch_size_modifier=0.3,
                priority_boost=True
            )
        else:
            return AllocationDecision(
                lane=ComputeLane.CPU,  # CPU typically uses less RAM than GPU operations
                reason=f"RAM pressure, reducing batch size",
                confidence=0.7,
                batch_size_modifier=0.5
            )
    
    def _default_allocation(self, metrics: ResourceMetrics, pressure: PressureIndex) -> AllocationDecision:
        """Default allocation when no specific strategy applies"""
        can_use_gpu, gpu_reason = self.should_use_gpu(metrics, pressure)
        can_use_cpu, cpu_reason = self.should_use_cpu(metrics, pressure)
        
        if can_use_gpu and metrics.gpu_usage < metrics.cpu_usage:
            return AllocationDecision(
                lane=ComputeLane.GPU,
                reason="GPU preferred for performance",
                confidence=0.8
            )
        elif can_use_cpu:
            return AllocationDecision(
                lane=ComputeLane.CPU,
                reason="CPU fallback",
                confidence=0.7
            )
        else:
            return AllocationDecision(
                lane=ComputeLane.HYBRID,
                reason=f"Both resources constrained - CPU: {cpu_reason}, GPU: {gpu_reason}",
                confidence=0.5,
                batch_size_modifier=0.4
            )

class OrderStrategy(AllocationStrategy):
    """Order execution focused allocation strategy"""
    
    def allocate(self, failure_key: FailureKey, metrics: ResourceMetrics) -> AllocationDecision:
        """Allocate resources for ORDER group failures"""
        subgroup = failure_key.subgroup
        
        if subgroup == FailureSubGroup.BATCH_MISS:
            return AllocationDecision(
                lane=ComputeLane.CPU,  # CPU for deterministic timing
                reason="Batch miss - use CPU for predictable timing",
                confidence=0.9,
                batch_size_modifier=0.5,
                priority_boost=True
            )
        elif subgroup == FailureSubGroup.SLIP_DRIFT:
            return AllocationDecision(
                lane=ComputeLane.HYBRID,
                reason="Price slip - use hybrid for fast response",
                confidence=0.8,
                batch_size_modifier=0.7
            )
        else:
            return AllocationDecision(
                lane=ComputeLane.CPU,
                reason="Order issue - CPU preferred for reliability",
                confidence=0.7
            )

class ResourceAllocator:
    """Main resource allocation controller"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize strategies
        self.strategies = {
            FailureGroup.PERF: PerformanceStrategy(config.get('perf', {})),
            FailureGroup.ORDER: OrderStrategy(config.get('order', {})),
            # Add default strategy for other groups
        }
        
        # Default strategy for unhandled groups
        self.default_strategy = PerformanceStrategy(config.get('default', {}))
        
        # Allocation history for learning
        self.allocation_history: List[Tuple[FailureKey, AllocationDecision, float]] = []
        self.max_history = config.get('max_history', 1000)
        
    def choose_lane(self, failure_key: FailureKey) -> AllocationDecision:
        """
        Choose the optimal compute lane for handling a failure
        
        Args:
            failure_key: The failure event to handle
            
        Returns:
            AllocationDecision with lane and reasoning
        """
        # Capture current resource state
        metrics = ResourceMetrics.capture_current()
        
        # Get appropriate strategy
        strategy = self.strategies.get(failure_key.group, self.default_strategy)
        
        # Make allocation decision
        decision = strategy.allocate(failure_key, metrics)
        
        # Log decision
        logger.info(f"Allocation decision for {failure_key.hash}: "
                   f"{decision.lane.value} - {decision.reason}")
        
        # Store in history for learning
        self._record_decision(failure_key, decision)
        
        return decision
    
    def _record_decision(self, failure_key: FailureKey, decision: AllocationDecision) -> None:
        """Record allocation decision for learning"""
        timestamp = time.time()
        self.allocation_history.append((failure_key, decision, timestamp))
        
        # Trim history if too long
        if len(self.allocation_history) > self.max_history:
            self.allocation_history = self.allocation_history[-self.max_history:]
    
    def update_decision_outcome(self, failure_hash: str, outcome_score: float) -> None:
        """Update allocation decision with outcome for learning"""
        for i, (key, decision, timestamp) in enumerate(self.allocation_history):
            if key.hash == failure_hash:
                # Store outcome score in decision object
                decision.confidence = outcome_score
                self.allocation_history[i] = (key, decision, timestamp)
                break
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics"""
        if not self.allocation_history:
            return {}
        
        # Count decisions by lane
        lane_counts = {}
        confidence_by_lane = {}
        
        for key, decision, timestamp in self.allocation_history:
            lane = decision.lane.value
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
            
            if lane not in confidence_by_lane:
                confidence_by_lane[lane] = []
            confidence_by_lane[lane].append(decision.confidence)
        
        # Calculate average confidence by lane
        avg_confidence = {}
        for lane, confidences in confidence_by_lane.items():
            avg_confidence[lane] = sum(confidences) / len(confidences)
        
        # Recent activity (last hour)
        recent_cutoff = time.time() - 3600
        recent_decisions = [
            (key, decision, ts) for key, decision, ts in self.allocation_history
            if ts > recent_cutoff
        ]
        
        return {
            'total_decisions': len(self.allocation_history),
            'recent_decisions': len(recent_decisions),
            'lane_counts': lane_counts,
            'avg_confidence_by_lane': avg_confidence,
            'most_used_lane': max(lane_counts, key=lane_counts.get) if lane_counts else None
        }
    
    def suggest_config_updates(self) -> Dict[str, Any]:
        """Suggest configuration updates based on allocation history"""
        suggestions = {}
        
        if len(self.allocation_history) < 50:
            return suggestions
        
        # Analyze GPU vs CPU effectiveness
        gpu_decisions = [d for k, d, t in self.allocation_history if d.lane == ComputeLane.GPU]
        cpu_decisions = [d for k, d, t in self.allocation_history if d.lane == ComputeLane.CPU]
        
        if gpu_decisions and cpu_decisions:
            gpu_avg_confidence = sum(d.confidence for d in gpu_decisions) / len(gpu_decisions)
            cpu_avg_confidence = sum(d.confidence for d in cpu_decisions) / len(cpu_decisions)
            
            # If GPU consistently performs worse, suggest raising GPU threshold
            if gpu_avg_confidence < cpu_avg_confidence - 0.1:
                suggestions['gpu_threshold'] = min(self.strategies[FailureGroup.PERF].gpu_threshold + 5, 95)
                suggestions['reason'] = 'GPU underperforming, raising threshold'
            
            # If CPU consistently performs worse, suggest raising CPU threshold  
            elif cpu_avg_confidence < gpu_avg_confidence - 0.1:
                suggestions['cpu_threshold'] = min(self.strategies[FailureGroup.PERF].cpu_threshold + 5, 95)
                suggestions['reason'] = 'CPU underperforming, raising threshold'
        
        return suggestions 