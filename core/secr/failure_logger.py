"""
SECR Failure Logger
==================

Captures, classifies, and stores system failures with full context.
Implements hierarchical failure taxonomy with parent/sub-group logic.
"""

import json
import time
import hashlib
import psutil
import numpy as np
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class FailureGroup(Enum):
    """Parent failure groups"""
    PERF = "PERF"           # Performance bottlenecks
    ORDER = "ORDER"         # Trading execution issues
    ENTROPY = "ENTROPY"     # Mathematical/phase issues
    THERMAL = "THERMAL"     # Temperature/cooling issues
    NET = "NET"            # Network/API issues

class FailureSubGroup(Enum):
    """Sub-group classifications"""
    # PERF subgroups
    GPU_LAG = "GPU_LAG"
    CPU_STALL = "CPU_STALL"
    RAM_PRESSURE = "RAM_PRESSURE"
    
    # ORDER subgroups
    BATCH_MISS = "BATCH_MISS"
    PARTIAL_FILL = "PARTIAL_FILL"
    SLIP_DRIFT = "SLIP_DRIFT"
    
    # ENTROPY subgroups
    ENTROPY_SPIKE = "ENTROPY_SPIKE"
    PHASE_INVERT = "PHASE_INVERT"
    ICAP_COLLAPSE = "ICAP_COLLAPSE"
    
    # THERMAL subgroups
    THERMAL_HALT = "THERMAL_HALT"
    FAN_STALL = "FAN_STALL"
    
    # NET subgroups
    API_TIMEOUT = "API_TIMEOUT"
    SOCKET_DROP = "SOCKET_DROP"

@dataclass
class PressureIndex:
    """System pressure metrics"""
    gpu_pressure: float = 0.0
    cpu_pressure: float = 0.0
    ram_pressure: float = 0.0
    net_pressure: float = 0.0
    thermal_pressure: float = 0.0
    
    def max_pressure(self) -> float:
        """Return the highest pressure component"""
        return max(
            self.gpu_pressure,
            self.cpu_pressure, 
            self.ram_pressure,
            self.net_pressure,
            self.thermal_pressure
        )
    
    def dominant_resource(self) -> str:
        """Return the resource with highest pressure"""
        pressures = {
            'gpu': self.gpu_pressure,
            'cpu': self.cpu_pressure,
            'ram': self.ram_pressure,
            'net': self.net_pressure,
            'thermal': self.thermal_pressure
        }
        return max(pressures, key=pressures.get)

@dataclass
class FailureKey:
    """Individual failure event record"""
    hash: str
    group: FailureGroup
    subgroup: FailureSubGroup
    pair: str
    timestamp: str
    severity: float
    ctx: Dict[str, Any]
    patch: Optional[Dict[str, Any]] = None
    resolved_ts: Optional[str] = None
    outcome: Optional[float] = None
    score: Optional[float] = None
    closed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['group'] = self.group.value
        result['subgroup'] = self.subgroup.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FailureKey':
        """Create from dictionary"""
        data['group'] = FailureGroup(data['group'])
        data['subgroup'] = FailureSubGroup(data['subgroup'])
        return cls(**data)

class FailureClassifier:
    """Classifies failures into appropriate groups and subgroups"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gpu_threshold = config.get('gpu_lag_ms', 180)
        self.cpu_threshold = config.get('cpu_stall_ms', 120)
        self.ram_threshold = config.get('ram_pressure_pct', 85)
        self.entropy_sigma = config.get('entropy_spike_sigma', 2.0)
        
    def classify(self, context: Dict[str, Any]) -> tuple[FailureGroup, FailureSubGroup]:
        """
        Classify failure based on context metrics
        
        Args:
            context: Dictionary containing failure context metrics
            
        Returns:
            Tuple of (FailureGroup, FailureSubGroup)
        """
        # GPU lag detection
        if context.get('gpu_latency_ms', 0) > self.gpu_threshold:
            return FailureGroup.PERF, FailureSubGroup.GPU_LAG
            
        # CPU stall detection
        if context.get('cpu_load_1s', 0) > 90:
            return FailureGroup.PERF, FailureSubGroup.CPU_STALL
            
        # RAM pressure detection
        if context.get('ram_usage_pct', 0) > self.ram_threshold:
            return FailureGroup.PERF, FailureSubGroup.RAM_PRESSURE
            
        # Batch miss detection
        if context.get('order_ack_delay_ms', 0) > context.get('tick_duration_ms', 1000):
            return FailureGroup.ORDER, FailureSubGroup.BATCH_MISS
            
        # Slip drift detection
        if context.get('price_slip_pct', 0) > 0.1:
            return FailureGroup.ORDER, FailureSubGroup.SLIP_DRIFT
            
        # Entropy spike detection
        delta_psi = context.get('delta_psi', 0)
        psi_mean = context.get('psi_mean', 0)
        psi_std = context.get('psi_std', 1)
        if delta_psi > (psi_mean + self.entropy_sigma * psi_std):
            return FailureGroup.ENTROPY, FailureSubGroup.ENTROPY_SPIKE
            
        # ICAP collapse detection
        if context.get('icap_value', 1.0) < 0.1:
            return FailureGroup.ENTROPY, FailureSubGroup.ICAP_COLLAPSE
            
        # Thermal halt detection
        if context.get('cpu_temp', 0) > 85 or context.get('gpu_temp', 0) > 83:
            return FailureGroup.THERMAL, FailureSubGroup.THERMAL_HALT
            
        # API timeout detection
        if context.get('api_response_ms', 0) > 5000:
            return FailureGroup.NET, FailureSubGroup.API_TIMEOUT
            
        # Default fallback
        return FailureGroup.PERF, FailureSubGroup.CPU_STALL

class PressureCalculator:
    """Calculates system pressure indices"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'beta': 0.3,   # GPU kernel latency weight
            'gamma': 0.2,  # CPU context switch weight  
            'delta': 0.4   # GC events weight
        }
        
    def calculate_gpu_pressure(self, context: Dict[str, Any]) -> float:
        """Calculate GPU pressure index"""
        if not GPU_AVAILABLE:
            return 0.0
            
        try:
            queue_usage = context.get('cuda_queue_usage', 0) / context.get('cuda_queue_max', 1)
            kernel_latency = context.get('kernel_latency_ms', 0) / context.get('gpu_timeout_ms', 1000)
            
            pressure = queue_usage + self.weights['beta'] * kernel_latency
            return min(pressure, 1.0)
        except Exception:
            return 0.0
    
    def calculate_cpu_pressure(self, context: Dict[str, Any]) -> float:
        """Calculate CPU pressure index"""
        try:
            load_1s = context.get('cpu_load_1s', 0) / 100.0
            ctx_switches = context.get('ctx_switches_per_sec', 0) / context.get('ctx_switch_baseline', 1000)
            
            pressure = load_1s + self.weights['gamma'] * ctx_switches
            return min(pressure, 1.0)
        except Exception:
            return psutil.cpu_percent(interval=None) / 100.0
    
    def calculate_ram_pressure(self, context: Dict[str, Any]) -> float:
        """Calculate RAM pressure index"""
        try:
            rss_usage = context.get('rss_mb', 0) / context.get('total_ram_mb', 1)
            gc_events = context.get('gc_events_per_min', 0) / context.get('gc_baseline', 10)
            
            pressure = rss_usage + self.weights['delta'] * gc_events
            return min(pressure, 1.0)
        except Exception:
            return psutil.virtual_memory().percent / 100.0
    
    def calculate_net_pressure(self, context: Dict[str, Any]) -> float:
        """Calculate network pressure index"""
        try:
            api_latency = context.get('api_response_ms', 0) / 5000.0  # 5s timeout baseline
            reconnect_rate = context.get('reconnects_per_min', 0) / 2.0  # 2/min baseline
            
            pressure = api_latency + reconnect_rate
            return min(pressure, 1.0)
        except Exception:
            return 0.0
    
    def calculate_thermal_pressure(self, context: Dict[str, Any]) -> float:
        """Calculate thermal pressure index"""
        try:
            cpu_temp = context.get('cpu_temp', 20) / 90.0  # 90C baseline
            gpu_temp = context.get('gpu_temp', 20) / 85.0  # 85C baseline
            
            pressure = max(cpu_temp, gpu_temp)
            return min(pressure, 1.0)
        except Exception:
            return 0.0
    
    def calculate_all_pressures(self, context: Dict[str, Any]) -> PressureIndex:
        """Calculate all pressure indices"""
        return PressureIndex(
            gpu_pressure=self.calculate_gpu_pressure(context),
            cpu_pressure=self.calculate_cpu_pressure(context),
            ram_pressure=self.calculate_ram_pressure(context),
            net_pressure=self.calculate_net_pressure(context),
            thermal_pressure=self.calculate_thermal_pressure(context)
        )

class FailureLogger:
    """Main failure logging system"""
    
    def __init__(self, 
                 storage_path: Union[str, Path] = "data/phantom_corridors.json",
                 max_keys: int = 10000,
                 config: Optional[Dict[str, Any]] = None):
        self.storage_path = Path(storage_path)
        self.max_keys = max_keys
        self.config = config or {}
        
        # Initialize components
        self.classifier = FailureClassifier(self.config)
        self.pressure_calc = PressureCalculator(self.config.get('pressure_weights'))
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing keys
        self.keys = self._load_keys()
        
    def _load_keys(self) -> List[FailureKey]:
        """Load existing failure keys from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    return [FailureKey.from_dict(item) for item in data]
            return []
        except Exception as e:
            logger.error(f"Error loading failure keys: {e}")
            return []
    
    def _save_keys(self) -> None:
        """Save failure keys to storage"""
        try:
            # Keep only the most recent keys
            if len(self.keys) > self.max_keys:
                self.keys = self.keys[-self.max_keys:]
            
            data = [key.to_dict() for key in self.keys]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving failure keys: {e}")
    
    def _generate_hash(self, timestamp: str, pair: str, reason: str) -> str:
        """Generate unique hash for failure event"""
        data = f"{timestamp}{pair}{reason}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def capture_failure(self, 
                       pair: str,
                       context: Dict[str, Any],
                       custom_reason: Optional[str] = None) -> FailureKey:
        """
        Capture a failure event with full context
        
        Args:
            pair: Trading pair (e.g. "BTC/USDC")
            context: Full system context at time of failure
            custom_reason: Optional manual reason override
            
        Returns:
            FailureKey object
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Classify the failure
        if custom_reason:
            # Parse custom reason into group/subgroup
            group, subgroup = self._parse_custom_reason(custom_reason)
        else:
            group, subgroup = self.classifier.classify(context)
        
        # Calculate pressure indices
        pressures = self.pressure_calc.calculate_all_pressures(context)
        
        # Generate hash
        hash_key = self._generate_hash(timestamp, pair, subgroup.value)
        
        # Create enhanced context with pressures
        enhanced_ctx = {
            **context,
            'pressures': asdict(pressures),
            'dominant_resource': pressures.dominant_resource()
        }
        
        # Create failure key
        failure_key = FailureKey(
            hash=hash_key,
            group=group,
            subgroup=subgroup,
            pair=pair,
            timestamp=timestamp,
            severity=pressures.max_pressure(),
            ctx=enhanced_ctx
        )
        
        # Store and save
        self.keys.append(failure_key)
        self._save_keys()
        
        logger.info(f"Captured failure: {group.value}/{subgroup.value} "
                   f"severity={failure_key.severity:.3f} pair={pair}")
        
        return failure_key
    
    def _parse_custom_reason(self, reason: str) -> tuple[FailureGroup, FailureSubGroup]:
        """Parse custom reason string into group/subgroup"""
        try:
            if '/' in reason:
                group_str, subgroup_str = reason.split('/', 1)
                group = FailureGroup(group_str.upper())
                subgroup = FailureSubGroup(subgroup_str.upper())
            else:
                subgroup = FailureSubGroup(reason.upper())
                # Infer group from subgroup
                group = self._infer_group_from_subgroup(subgroup)
            return group, subgroup
        except ValueError:
            logger.warning(f"Invalid custom reason: {reason}, using default classification")
            return FailureGroup.PERF, FailureSubGroup.CPU_STALL
    
    def _infer_group_from_subgroup(self, subgroup: FailureSubGroup) -> FailureGroup:
        """Infer parent group from subgroup"""
        mapping = {
            FailureSubGroup.GPU_LAG: FailureGroup.PERF,
            FailureSubGroup.CPU_STALL: FailureGroup.PERF,
            FailureSubGroup.RAM_PRESSURE: FailureGroup.PERF,
            FailureSubGroup.BATCH_MISS: FailureGroup.ORDER,
            FailureSubGroup.PARTIAL_FILL: FailureGroup.ORDER,
            FailureSubGroup.SLIP_DRIFT: FailureGroup.ORDER,
            FailureSubGroup.ENTROPY_SPIKE: FailureGroup.ENTROPY,
            FailureSubGroup.PHASE_INVERT: FailureGroup.ENTROPY,
            FailureSubGroup.ICAP_COLLAPSE: FailureGroup.ENTROPY,
            FailureSubGroup.THERMAL_HALT: FailureGroup.THERMAL,
            FailureSubGroup.FAN_STALL: FailureGroup.THERMAL,
            FailureSubGroup.API_TIMEOUT: FailureGroup.NET,
            FailureSubGroup.SOCKET_DROP: FailureGroup.NET,
        }
        return mapping.get(subgroup, FailureGroup.PERF)
    
    def get_open_keys(self) -> List[FailureKey]:
        """Get all unresolved failure keys"""
        return [key for key in self.keys if not key.closed]
    
    def get_keys_by_group(self, group: FailureGroup) -> List[FailureKey]:
        """Get failure keys by group"""
        return [key for key in self.keys if key.group == group]
    
    def get_keys_by_pair(self, pair: str) -> List[FailureKey]:
        """Get failure keys by trading pair"""
        return [key for key in self.keys if key.pair == pair]
    
    def get_recent_keys(self, hours: int = 24) -> List[FailureKey]:
        """Get failure keys from the last N hours"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        return [key for key in self.keys 
                if datetime.fromisoformat(key.timestamp.replace('Z', '+00:00')).timestamp() > cutoff_time]
    
    def update_key(self, hash_key: str, **updates) -> bool:
        """Update a failure key with new information"""
        for key in self.keys:
            if key.hash == hash_key:
                for field, value in updates.items():
                    if hasattr(key, field):
                        setattr(key, field, value)
                self._save_keys()
                return True
        return False
    
    def close_key(self, hash_key: str, outcome: float, score: float) -> bool:
        """Close a failure key with resolution outcome"""
        resolved_ts = datetime.now(timezone.utc).isoformat()
        return self.update_key(
            hash_key,
            resolved_ts=resolved_ts,
            outcome=outcome,
            score=score,
            closed=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get failure statistics"""
        total_keys = len(self.keys)
        open_keys = len(self.get_open_keys())
        
        # Group statistics
        group_counts = {}
        subgroup_counts = {}
        for key in self.keys:
            group_counts[key.group.value] = group_counts.get(key.group.value, 0) + 1
            subgroup_counts[key.subgroup.value] = subgroup_counts.get(key.subgroup.value, 0) + 1
        
        # Recent activity
        recent_24h = len(self.get_recent_keys(24))
        recent_1h = len(self.get_recent_keys(1))
        
        # Average severity
        avg_severity = np.mean([key.severity for key in self.keys]) if self.keys else 0.0
        
        return {
            'total_keys': total_keys,
            'open_keys': open_keys,
            'closed_keys': total_keys - open_keys,
            'recent_24h': recent_24h,
            'recent_1h': recent_1h,
            'avg_severity': float(avg_severity),
            'group_counts': group_counts,
            'subgroup_counts': subgroup_counts
        } 