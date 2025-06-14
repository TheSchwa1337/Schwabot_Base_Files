"""
Diogenic Logic Trading (DLT) Waveform Engine
Implements recursive pattern recognition and phase validation for trading decisions
Enhanced with profit-fault correlation and JuMBO-style anomaly detection
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta
from .quantum_visualizer import PanicDriftVisualizer, plot_entropy_waveform
from .pattern_metrics import PatternMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import mathlib
import json
from statistics import stdev
import hashlib
import time
import psutil
import threading
import logging
import os
from pathlib import Path

# Enhanced imports for profit-fault correlation
try:
    from core.fault_bus import FaultBus, FaultType, FaultBusEvent
    from profit_cycle_navigator import ProfitCycleNavigator, ProfitVector, ProfitCycleState
    ENHANCED_MODE = True
except ImportError:
    # Fallback mode if enhanced modules not available
    ENHANCED_MODE = False
    logging.warning("Enhanced profit-fault correlation modules not available. Running in basic mode.")

class PhaseDomain(Enum):
    SHORT = "short"    # Seconds to Hours
    MID = "mid"        # Hours to Days  
    LONG = "long"      # Days to Months

@dataclass
class PhaseTrust:
    """Trust metrics for each phase domain"""
    successful_echoes: int
    entropy_consistency: float
    last_validation: datetime
    trust_threshold: float = 0.8
    memory_coherence: float = 0.0  # Added for tensor state integration
    thermal_state: float = 0.0     # Added for resource management
    profit_correlation: float = 0.0  # NEW: Correlation with profit outcomes
    fault_sensitivity: float = 0.0   # NEW: Sensitivity to fault events

@dataclass 
class BitmapTrigger:
    """Represents a trigger point in the 16-bit trading map"""
    phase: PhaseDomain
    time_window: timedelta
    diogenic_score: float
    frequency: float
    last_trigger: datetime
    success_count: int
    tensor_signature: np.ndarray  # Added for tensor state tracking
    resource_usage: float = 0.0   # Added for resource management
    profit_correlation: float = 0.0  # NEW: Historical profit correlation
    anomaly_strength: float = 0.0    # NEW: JuMBO-style anomaly detection

class BitmapCascadeManager:
    """
    Enhanced bitmap cascade manager with profit-fault correlation
    Manages multiple bitmap tiers for signal amplification and memory-driven propagation.
    """
    def __init__(self):
        self.bitmaps = {
            4: np.zeros(4, dtype=bool),
            8: np.zeros(8, dtype=bool),
            16: np.zeros(16, dtype=bool),
            42: np.zeros(42, dtype=bool),
            81: np.zeros(81, dtype=bool),
        }
        self.memory_log = []  # List of dicts: {hash, bitmap_size, outcome, timestamp, ...}
        self.profit_correlations = {}  # Track profit correlation per bitmap tier
        self.anomaly_clusters = {}     # Track anomaly clusters per tier
        
    def update_bitmap(self, tier: int, idx: int, signal: bool, profit_context: float = None):
        """Enhanced bitmap update with profit correlation tracking"""
        self.bitmaps[tier][idx % tier] = signal
        
        # Track profit correlation if provided
        if profit_context is not None:
            if tier not in self.profit_correlations:
                self.profit_correlations[tier] = []
            self.profit_correlations[tier].append({
                'index': idx,
                'signal': signal,
                'profit': profit_context,
                'timestamp': datetime.now()
            })
        
        # Propagate up if needed (example: 16 triggers 42/81)
        if signal and tier == 16:
            self.bitmaps[42][idx % 42] = True
            self.bitmaps[81][(idx * 3) % 81] = True

    def readout(self):
        """Enhanced readout with profit correlation data"""
        basic_readout = {k: np.where(v)[0].tolist() for k, v in self.bitmaps.items() if np.any(v)}
        
        # Add profit correlation summaries
        correlation_summary = {}
        for tier, correlations in self.profit_correlations.items():
            if correlations:
                profits = [c['profit'] for c in correlations[-20:]]  # Last 20 entries
                correlation_summary[f'tier_{tier}_avg_profit'] = np.mean(profits)
                correlation_summary[f'tier_{tier}_profit_std'] = np.std(profits)
        
        return {
            'bitmap_state': basic_readout,
            'profit_correlations': correlation_summary
        }

    def detect_profit_anomaly(self, tier: int, current_profit: float) -> Tuple[bool, float]:
        """Detect JuMBO-style profit anomalies for specific tier"""
        if tier not in self.profit_correlations or len(self.profit_correlations[tier]) < 10:
            return False, 0.0
        
        recent_profits = [c['profit'] for c in self.profit_correlations[tier][-20:]]
        mean_profit = np.mean(recent_profits)
        std_profit = np.std(recent_profits)
        
        if std_profit == 0:
            return False, 0.0
        
        z_score = abs(current_profit - mean_profit) / std_profit
        
        # Check for anomaly cluster (multiple recent anomalies)
        anomaly_count = sum(1 for p in recent_profits[-5:] if abs(p - mean_profit) / std_profit > 2.0)
        
        if z_score > 2.5 and anomaly_count >= 2:
            anomaly_strength = min(z_score / 5.0, 1.0)
            return True, anomaly_strength
        
        return False, 0.0

class WaveformAuditLogger:
    """Enhanced audit logger with profit-fault correlation tracking"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.audit_file = self.log_dir / "waveform_audit.log"
        self.profit_file = self.log_dir / "profit_correlation.log"
        
    def log_waveform_event(self, event_type: str, entropy: float, coherence: float, 
                          profit_context: Optional[float] = None, metadata: Optional[Dict] = None):
        """Log waveform processing events with profit correlation"""
        log_entry = {
            "event": event_type,
            "entropy": entropy,
            "coherence": coherence,
            "profit_context": profit_context,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def log_profit_correlation(self, sha_hash: str, profit_delta: float, fault_context: Dict):
        """Log profit-fault correlations for analysis"""
        correlation_entry = {
            "sha_hash": sha_hash,
            "profit_delta": profit_delta,
            "fault_context": fault_context,
            "timestamp": time.time()
        }
        
        with open(self.profit_file, "a") as f:
            f.write(json.dumps(correlation_entry) + "\n")

class DLTWaveformEngine:
    """
    Enhanced core engine for Diogenic Logic Trading pattern recognition
    Now includes profit-fault correlation and recursive loop prevention
    """
    
    def __init__(self, max_cpu_percent: float = 80.0, max_memory_percent: float = 70.0):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Resource management
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.resource_lock = threading.Lock()
        
        # Trading parameters
        self.max_position_size = 1.0  
        self.current_symbol = None    
        self.trade_vector = np.zeros(10000, dtype=np.float32)
        
        # Initialize enhanced fault bus and profit navigator
        self.fault_bus = FaultBus()
        self.profit_navigator = ProfitCycleNavigator(self.fault_bus)
        
        # Waveform integrity tracking
        self.blacklist_hashes = set()
        self.file_integrity_cache = {}
        
        # 16-bit trading map (4-bit, 8-bit, 16-bit allocations)
        self.state_maps = {
            4: np.zeros(4, dtype=bool),
            8: np.zeros(8, dtype=bool),
            16: np.zeros(16, dtype=bool),
            42: np.zeros(42, dtype=bool),
            81: np.zeros(81, dtype=np.int8),  # Ternary: -1, 0, 1 or 0, 1, 2
        }
        
        # Enhanced phase trust tracking with profit correlation
        self.phase_trust: Dict[PhaseDomain, PhaseTrust] = {
            PhaseDomain.SHORT: PhaseTrust(0, 0.0, datetime.now()),
            PhaseDomain.MID: PhaseTrust(0, 0.0, datetime.now()),
            PhaseDomain.LONG: PhaseTrust(0, 0.0, datetime.now())
        }
        
        # Trigger memory with enhanced correlation tracking
        self.triggers: List[BitmapTrigger] = []
        
        # Phase validation thresholds with dynamic adjustment
        self.phase_thresholds = {
            PhaseDomain.LONG: 3,    # 3+ successful echoes in 90d
            PhaseDomain.MID: 5,     # 5+ echoes with entropy consistency
            PhaseDomain.SHORT: 10   # 10+ phase-aligned echoes
        }
        
        self.data = None
        self.processed_data = None
        
        self.hooks = {}
        
        # Enhanced thresholds with thermal state consideration
        self.entropy_thresholds = {'SHORT': 4.0, 'MID': 3.5, 'LONG': 3.0}
        self.coherence_thresholds = {'SHORT': 0.6, 'MID': 0.5, 'LONG': 0.4}
        
        # Unified tensor state with profit correlation
        self.tensor_map = np.zeros(256)
        self.tensor_history: List[np.ndarray] = []
        self.max_tensor_history = 1000
        
        # Enhanced components
        self.bitmap_cascade = BitmapCascadeManager()
        self.audit_logger = WaveformAuditLogger()
        
        # Resource monitoring
        self.last_resource_check = datetime.now()
        self.resource_check_interval = timedelta(seconds=5)
        
        # JuMBO-style pattern detection
        self.pattern_hash_history = {}
        self.loop_detection_window = 50
        
    def validate_waveform_file(self, path: str) -> str:
        """Validate waveform file integrity and detect tampering"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Waveform file not found: {path}")
        
        # Compute SHA256 hash
        with open(path, "rb") as f:
            file_content = f.read()
            sha_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check against blacklist
        if sha_hash in self.blacklist_hashes:
            raise ValueError(f"Tampered waveform file detected! SHA256: {sha_hash}")
        
        # Cache for future reference
        self.file_integrity_cache[path] = {
            'sha_hash': sha_hash,
            'last_checked': datetime.now(),
            'file_size': len(file_content)
        }
        
        self.logger.info(f"Waveform file validated: {path} (SHA: {sha_hash[:16]})")
        return sha_hash
        
    def check_resources(self) -> bool:
        """Enhanced resource checking with fault correlation"""
        with self.resource_lock:
            current_time = datetime.now()
            if current_time - self.last_resource_check < self.resource_check_interval:
                return True
                
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            self.last_resource_check = current_time
            
            # Create fault events for resource issues
            if cpu_percent > self.max_cpu_percent:
                fault_event = FaultBusEvent(
                    tick=int(time.time()),
                    module="resource_monitor",
                    type=FaultType.THERMAL_HIGH,
                    severity=cpu_percent / 100.0,
                    metadata={"cpu_percent": cpu_percent}
                )
                self.fault_bus.push(fault_event)
                
            if memory_percent > self.max_memory_percent:
                fault_event = FaultBusEvent(
                    tick=int(time.time()),
                    module="resource_monitor",
                    type=FaultType.GPU_OVERLOAD,
                    severity=memory_percent / 100.0,
                    metadata={"memory_percent": memory_percent}
                )
                self.fault_bus.push(fault_event)
            
            if cpu_percent > self.max_cpu_percent or memory_percent > self.max_memory_percent:
                self.logger.warning(f"Resource limits exceeded - CPU: {cpu_percent}%, Memory: {memory_percent}%")
                return False
                
            return True
        
    def update_phase_trust(self, phase: PhaseDomain, success: bool, entropy: float, profit_delta: float = None):
        """Enhanced phase trust update with profit correlation"""
        trust = self.phase_trust[phase]
        
        if success:
            trust.successful_echoes += 1
            trust.entropy_consistency = (trust.entropy_consistency * 0.9 + entropy * 0.1)
            
            # Update memory coherence based on tensor state
            if self.tensor_history:
                recent_tensors = self.tensor_history[-3:]
                trust.memory_coherence = np.mean([np.std(t) for t in recent_tensors])
                
            # Update profit correlation if provided
            if profit_delta is not None:
                trust.profit_correlation = (trust.profit_correlation * 0.9 + profit_delta * 0.1)
        
        # Update thermal state based on resource usage
        trust.thermal_state = psutil.cpu_percent() / 100.0
        
        # Update fault sensitivity based on recent fault events
        recent_faults = [e for e in self.fault_bus.memory_log[-10:]]
        if recent_faults:
            avg_severity = np.mean([e.severity for e in recent_faults])
            trust.fault_sensitivity = (trust.fault_sensitivity * 0.8 + avg_severity * 0.2)
        
        trust.last_validation = datetime.now()
        
    def is_phase_trusted(self, phase: PhaseDomain) -> bool:
        """Enhanced phase trust checking with profit correlation"""
        if not self.check_resources():
            return False
            
        trust = self.phase_trust[phase]
        
        # Enhanced trust criteria including profit correlation
        base_trust = (trust.successful_echoes >= self.phase_thresholds[phase] and 
                     trust.entropy_consistency > 0.8 and
                     trust.thermal_state < 0.9)
        
        # Additional profit correlation check
        profit_trust = trust.profit_correlation > -0.1  # Not consistently losing money
        
        return base_trust and profit_trust
        
    def detect_recursive_loop(self, entropy: float, coherence: float, current_profit: float) -> bool:
        """Detect recursive loops in profit patterns using SHA-based detection"""
        # Create pattern signature
        pattern_key = f"{entropy:.4f}_{coherence:.4f}_{current_profit:.4f}"
        pattern_hash = hashlib.sha256(pattern_key.encode()).hexdigest()[:16]
        
        # Check for recursive patterns
        if pattern_hash in self.pattern_hash_history:
            self.pattern_hash_history[pattern_hash]['count'] += 1
            self.pattern_hash_history[pattern_hash]['last_seen'] = datetime.now()
            
            # If pattern repeats too often, it's likely a false loop
            if self.pattern_hash_history[pattern_hash]['count'] > 5:
                # Create recursive loop fault event
                loop_event = FaultBusEvent(
                    tick=int(time.time()),
                    module="pattern_detector",
                    type=FaultType.RECURSIVE_LOOP,
                    severity=min(self.pattern_hash_history[pattern_hash]['count'] / 10.0, 1.0),
                    metadata={
                        'pattern_hash': pattern_hash,
                        'repeat_count': self.pattern_hash_history[pattern_hash]['count'],
                        'entropy': entropy,
                        'coherence': coherence,
                        'profit': current_profit
                    },
                    sha_signature=pattern_hash
                )
                self.fault_bus.push(loop_event)
                return True
        else:
            self.pattern_hash_history[pattern_hash] = {
                'count': 1,
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'entropy': entropy,
                'coherence': coherence,
                'profit': current_profit
            }
        
        # Clean old patterns
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.pattern_hash_history = {
            k: v for k, v in self.pattern_hash_history.items()
            if v['last_seen'] > cutoff_time
        }
        
        return False

    def load_data(self, filename: str):
        """Enhanced data loading with integrity validation"""
        try:
            # Validate file integrity
            file_hash = self.validate_waveform_file(filename)
            
            with open(filename, "r") as f:
                lines = f.readlines()
                if len(lines) == 1 and lines[0].strip().startswith("["):
                    self.data = json.loads(lines[0])
                else:
                    self.data = [float(line.strip()) for line in lines if line.strip()]
                    
            self.logger.info(f"Loaded {len(self.data)} waveform entries from {filename}")
            
            # Log audit event
            self.audit_logger.log_waveform_event(
                "load_data", 
                entropy=0.0, 
                coherence=0.0,
                metadata={'file_hash': file_hash, 'data_points': len(self.data)}
            )
            
            self.trigger_hooks("on_waveform_loaded", data=self.data)
            
        except FileNotFoundError:
            self.logger.error(f"Waveform file not found: {filename}")
            self.data = None
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.data = None

    def process_waveform(self):
        """Enhanced waveform processing with profit correlation and loop detection"""
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data first.")
            
        try:
            self.processed_data = [self.normalize(x) for x in self.data]
            self.logger.info("Waveform normalized")
            
            # Calculate entropy and coherence
            entropy = self.calculate_entropy(self.processed_data)
            coherence = self.calculate_coherence(self.processed_data)
            
            self.logger.info(f"Entropy: {entropy:.4f}, Coherence: {coherence:.4f}")
            
            # Get current profit context from navigator
            current_time = datetime.now()
            profit_vector = self.profit_navigator.update_market_state(
                current_price=100.0,  # Placeholder - should come from market data
                current_volume=1000.0,
                timestamp=current_time
            )
            
            # Detect recursive loops
            is_loop = self.detect_recursive_loop(entropy, coherence, profit_vector.magnitude)
            
            if is_loop:
                self.logger.warning("Recursive loop detected - breaking cycle")
                return
            
            # Check for profit anomalies in bitmap cascade
            for tier in [16, 42, 81]:
                is_anomaly, anomaly_strength = self.bitmap_cascade.detect_profit_anomaly(
                    tier, profit_vector.magnitude
                )
                if is_anomaly:
                    self.logger.info(f"Profit anomaly detected in tier {tier}: strength {anomaly_strength:.3f}")
                    
                    # Create anomaly fault event
                    anomaly_event = FaultBusEvent(
                        tick=int(time.time()),
                        module="bitmap_cascade",
                        type=FaultType.PROFIT_ANOMALY,
                        severity=anomaly_strength,
                        metadata={
                            'tier': tier,
                            'profit_magnitude': profit_vector.magnitude,
                            'anomaly_strength': anomaly_strength
                        },
                        profit_context=profit_vector.magnitude
                    )
                    self.fault_bus.push(anomaly_event)
            
            # Update bitmap cascade with profit context
            for idx, bit in enumerate(self.state_maps[16]):
                if bit:
                    self.bitmap_cascade.update_bitmap(
                        16, idx, True, profit_context=profit_vector.magnitude
                    )
            
            # Log audit event with profit correlation
            self.audit_logger.log_waveform_event(
                "process_waveform",
                entropy=entropy,
                coherence=coherence,
                profit_context=profit_vector.magnitude,
                metadata={
                    'profit_confidence': profit_vector.confidence,
                    'profit_direction': profit_vector.direction,
                    'anomaly_strength': profit_vector.anomaly_strength
                }
            )
            
            # Trigger hooks with enhanced context
            self.trigger_hooks("on_entropy_vector_generated", 
                             entropy=self.processed_data,
                             profit_vector=profit_vector)
            self.trigger_hooks("on_bitmap_cascade_updated", 
                             cascade_state=self.bitmap_cascade.readout())
            
        except Exception as e:
            self.logger.error(f"Waveform processing failed: {e}")
            return

    def calculate_entropy(self, data: List[float]) -> float:
        """Calculate Shannon entropy of the data"""
        if not data or len(data) < 2:
            return 0.0
            
        # Normalize data to [0,1] range
        data_norm = (np.array(data) - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        
        # Create histogram
        hist, _ = np.histogram(data_norm, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        return entropy
    
    def calculate_coherence(self, data: List[float]) -> float:
        """Calculate pattern coherence using autocorrelation"""
        if not data or len(data) < 2:
            return 0.0
            
        # Calculate autocorrelation
        data_norm = (np.array(data) - np.mean(data)) / (np.std(data) + 1e-8)
        autocorr = np.correlate(data_norm, data_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize and calculate coherence
        autocorr = autocorr / (autocorr[0] + 1e-8)
        coherence = np.mean(np.abs(autocorr[:min(20, len(autocorr))]))
        
        return coherence

    def normalize(self, x, min_val=0.0, max_val=1.0):
        raw_min, raw_max = -1.0, 1.0
        return min_val + ((x - raw_min) / (raw_max - raw_min)) * (max_val - min_val)

    def register_hook(self, hook_name: str, hook_function: Callable):
        """Register a callback for a specific event."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(hook_function)

    def trigger_hooks(self, event: str, **kwargs):
        """Safely trigger registered callbacks."""
        for hook in self.hooks.get(event, []):
            try:
                hook(**kwargs)
            except Exception as e:
                self.logger.error(f"Hook '{event}' failed: {e}")

    def get_profit_correlations(self) -> Dict:
        """Get current profit correlations from fault bus and bitmap cascade"""
        return {
            'fault_correlations': self.fault_bus.export_correlation_matrix(),
            'bitmap_correlations': self.bitmap_cascade.readout(),
            'phase_trust': {
                phase.value: {
                    'profit_correlation': trust.profit_correlation,
                    'fault_sensitivity': trust.fault_sensitivity,
                    'entropy_consistency': trust.entropy_consistency
                }
                for phase, trust in self.phase_trust.items()
            }
        }

    def export_comprehensive_log(self, file_path: str = None) -> str:
        """Export comprehensive log including fault bus, profit correlations, and navigation state"""
        comprehensive_data = {
            'timestamp': datetime.now().isoformat(),
            'fault_bus_log': json.loads(self.fault_bus.export_memory_log()),
            'correlation_matrix': json.loads(self.fault_bus.export_correlation_matrix()),
            'navigation_log': json.loads(self.profit_navigator.export_navigation_log()),
            'profit_correlations': self.get_profit_correlations(),
            'pattern_hash_history': {
                k: {
                    'count': v['count'],
                    'first_seen': v['first_seen'].isoformat(),
                    'last_seen': v['last_seen'].isoformat(),
                    'entropy': v['entropy'],
                    'coherence': v['coherence'],
                    'profit': v['profit']
                }
                for k, v in self.pattern_hash_history.items()
            }
        }
        
        output = json.dumps(comprehensive_data, indent=2)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(output)
        return output

# Example usage with enhanced functionality
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create enhanced engine
    engine = DLTWaveformEngine()
    
    # Register enhanced hooks
    engine.register_hook("on_waveform_loaded", 
                        lambda data, **kwargs: print(f"Loaded {len(data)} waveform entries"))
    engine.register_hook("on_entropy_vector_generated", 
                        lambda entropy, profit_vector, **kwargs: 
                        print(f"Entropy generated with profit magnitude: {profit_vector.magnitude:.4f}"))
    
    # Simulate processing with profit correlation
    test_data = [0.1, 0.5, 0.9, 0.3, 0.6, 0.2] * 10  # Repeated pattern to test loop detection
    engine.data = test_data
    
    try:
        engine.process_waveform()
        
        # Dispatch fault bus events
        import asyncio
        asyncio.run(engine.fault_bus.dispatch())
        
        # Export comprehensive log
        print("\n=== Comprehensive Analysis ===")
        log_output = engine.export_comprehensive_log()
        print(log_output[:1000] + "..." if len(log_output) > 1000 else log_output)
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        print(f"Error: {e}") 