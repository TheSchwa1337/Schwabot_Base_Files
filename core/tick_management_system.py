"""
Schwabot Tick Management System
==============================
Core tick management with ALIF/ALEPH synchronization, drift correction, and ghost data recovery.
Handles stack log decay, pipe integrity, and temporal desync buffering.
"""

import time
import json
import hashlib
import threading
import queue
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import deque, defaultdict

# Import ALEPH and NCCO cores
try:
    from aleph_core import DetonationSequencer, EntropyAnalyzer, PatternMatcher
    from ncco_core import NCCO
    CORES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core modules not available: {e}")
    CORES_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TickContext:
    """Complete context for a single tick cycle"""
    tick_id: int
    timestamp: datetime
    delta_t: float
    entropy: float = 0.0
    drift_score: float = 0.0
    echo_strength: float = 0.0
    alif_state: str = "unknown"
    aleph_state: str = "unknown"
    stack_ready: bool = False
    validated: bool = False
    ghost_tick: bool = False
    compression_mode: bool = False

@dataclass
class RuntimeCounters:
    """Comprehensive runtime counters for system monitoring"""
    tick_count: int = 0
    entry_attempts: int = 0
    entry_suppressed: int = 0
    compression_count: int = 0
    visual_broadcasts: int = 0
    echo_confirmations: int = 0
    echo_failures: int = 0
    strategy_confirms: int = 0
    fallbacks_triggered: int = 0
    hash_reloads: int = 0
    path_conflicts: int = 0
    drift_ticks: int = 0
    visual_panel_refreshes: int = 0
    ghost_ticks_recovered: int = 0
    stack_timeouts: int = 0
    pipe_integrity_failures: int = 0
    
    def print_summary(self):
        """Print formatted counter summary"""
        print("[📊 Runtime State Summary]")
        for k, v in self.__dict__.items():
            status = "⚠️" if k in ["entry_suppressed", "fallbacks_triggered", "drift_ticks"] and v > 0 else "✅"
            print(f" {status} {k.replace('_', ' ').title()}: {v}")

@dataclass
class CompressionMode:
    """Compression mode state tracking"""
    mode: str = "LO_SYNC"  # LO_SYNC, Δ_DRIFT, ECHO_GLIDE, COMPRESS_HOLD, OVERLOAD_FALLBACK
    alif_compressed: bool = False
    aleph_compressed: bool = False
    delta_tolerance: float = 0.15
    echo_threshold: float = 0.5
    last_mode_change: datetime = field(default_factory=datetime.now)
    
    def update_mode(self, t_alif: float, t_aleph: float, echo_strength: float):
        """Update compression mode based on timing and echo"""
        delta = abs(t_alif - t_aleph)
        
        if delta > self.delta_tolerance and echo_strength < self.echo_threshold:
            self.mode = "Δ_DRIFT"
            self.alif_compressed = True
            self.aleph_compressed = False
        elif delta < 0.02:
            self.mode = "LO_SYNC"
            self.alif_compressed = False
            self.aleph_compressed = False
        elif echo_strength < 0.3:
            self.mode = "ECHO_GLIDE"
            self.alif_compressed = False
            self.aleph_compressed = True
        elif delta > 0.5:
            self.mode = "OVERLOAD_FALLBACK"
            self.alif_compressed = True
            self.aleph_compressed = True
        else:
            self.mode = "COMPRESS_HOLD"
            
        self.last_mode_change = datetime.now()

class GhostTickReservoir:
    """Buffer for ticks that arrive too early or during system instability"""
    
    def __init__(self, max_size: int = 100):
        self.reservoir = {}
        self.max_size = max_size
        self.recovery_count = 0
        
    def store_ghost_tick(self, tick_context: TickContext):
        """Store a ghost tick for later recovery"""
        tick_context.ghost_tick = True
        self.reservoir[tick_context.tick_id] = tick_context
        
        # Prevent overflow
        if len(self.reservoir) > self.max_size:
            oldest_tick = min(self.reservoir.keys())
            del self.reservoir[oldest_tick]
    
    def recover_tick(self, tick_id: int) -> Optional[TickContext]:
        """Recover a ghost tick if available"""
        if tick_id in self.reservoir:
            tick = self.reservoir.pop(tick_id)
            self.recovery_count += 1
            return tick
        return None
    
    def get_pending_count(self) -> int:
        """Get number of pending ghost ticks"""
        return len(self.reservoir)

class StackLogIntegrityChecker:
    """Ensures stack log integrity and prevents corruption"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_dir = Path(log_directory)
        self.log_dir.mkdir(exist_ok=True)
        self.quarantine_dir = self.log_dir / "quarantine"
        self.quarantine_dir.mkdir(exist_ok=True)
        
    def validate_log_entry(self, entry: Dict) -> bool:
        """Validate log entry structure"""
        required_fields = ["tick_id", "timestamp", "entropy"]
        return all(field in entry for field in required_fields)
    
    def secure_write(self, filename: str, data: Dict) -> bool:
        """Write data with integrity checking"""
        try:
            if not self.validate_log_entry(data):
                self.quarantine_entry(data, "invalid_structure")
                return False
            
            filepath = self.log_dir / filename
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Verify write integrity
            with open(filepath, 'r') as f:
                verified_data = json.load(f)
                
            return True
        except Exception as e:
            logger.error(f"Stack log write failed: {e}")
            self.quarantine_entry(data, f"write_error_{e}")
            return False
    
    def quarantine_entry(self, data: Any, reason: str):
        """Quarantine corrupted or invalid data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        quarantine_file = self.quarantine_dir / f"quarantine_{timestamp}_{reason}.json"
        
        try:
            with open(quarantine_file, 'w') as f:
                json.dump({
                    "reason": reason,
                    "timestamp": timestamp,
                    "data": str(data)  # Convert to string to handle any type
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to quarantine data: {e}")

class ALIFCore:
    """Adaptive Logic Integration Framework - Entry decision logic"""
    
    def __init__(self, counters: RuntimeCounters):
        self.entropy_state = 0.0
        self.latency = 0.0
        self.visual_sync = False
        self.compression_active = False
        self.counters = counters
        self.last_entry_time = 0.0
        
    def on_tick(self, tick: TickContext) -> Dict[str, Any]:
        """Process tick and determine entry logic"""
        self.counters.entry_attempts += 1
        
        # Check entropy and make entry decision
        self.entropy_state = self._calculate_entropy(tick)
        tick.entropy = self.entropy_state
        
        if self.entropy_state > 0.85:
            self.compression_active = True
            self.counters.entry_suppressed += 1
            self.counters.compression_count += 1
            tick.alif_state = "compressed"
            return {"action": "compress", "reason": "high_entropy"}
        
        # Normal operation
        self.compression_active = False
        tick.alif_state = "active"
        self.last_entry_time = time.time()
        
        # Broadcast glyph if conditions are met
        if self.entropy_state < 0.7:
            self.counters.visual_broadcasts += 1
            return {"action": "broadcast_glyph", "entropy": self.entropy_state}
        
        return {"action": "hold", "entropy": self.entropy_state}
    
    def _calculate_entropy(self, tick: TickContext) -> float:
        """Calculate current entropy state"""
        # Use real entropy calculation here
        base_entropy = 0.3 + np.sin(tick.tick_id * 0.1) * 0.3
        
        # Add drift-based entropy
        if tick.drift_score > 0.1:
            base_entropy += tick.drift_score * 0.5
            
        return np.clip(base_entropy, 0.0, 1.0)

class ALEPHCore:
    """Autonomous Logic Execution Path Hierarchy - Memory validation"""
    
    def __init__(self, counters: RuntimeCounters):
        self.hold_state = False
        self.echo_threshold = 0.5
        self.memory_bank = deque(maxlen=1000)
        self.counters = counters
        self.last_validation_time = 0.0
        
    def on_tick(self, tick: TickContext) -> Dict[str, Any]:
        """Process tick and validate strategy memory"""
        echo_strength = self._check_echo_strength(tick)
        tick.echo_strength = echo_strength
        
        if echo_strength < self.echo_threshold:
            self.hold_state = True
            self.counters.echo_failures += 1
            tick.aleph_state = "hold"
            return {"action": "hold", "reason": "weak_echo", "echo": echo_strength}
        
        # Echo is strong enough
        self.hold_state = False
        self.counters.echo_confirmations += 1
        self.counters.strategy_confirms += 1
        tick.aleph_state = "confirmed"
        self.last_validation_time = time.time()
        
        # Store in memory bank
        self.memory_bank.append({
            "tick_id": tick.tick_id,
            "echo_strength": echo_strength,
            "timestamp": tick.timestamp
        })
        
        return {"action": "confirm", "echo": echo_strength}
    
    def _check_echo_strength(self, tick: TickContext) -> float:
        """Calculate echo strength based on memory correlation"""
        # Simulate echo strength calculation
        base_echo = 0.6 + np.cos(tick.tick_id * 0.15) * 0.2
        
        # Factor in memory consistency
        if len(self.memory_bank) > 10:
            recent_echoes = [entry["echo_strength"] for entry in list(self.memory_bank)[-10:]]
            consistency = 1.0 - np.std(recent_echoes)
            base_echo *= consistency
            
        return np.clip(base_echo, 0.0, 1.0)

class TickManager:
    """Master tick management system with drift correction and error recovery"""
    
    def __init__(self, tick_interval: float = 1.0):
        self.tick_interval = tick_interval
        self.last_tick_time = time.time()
        self.tick_count = 0
        self.callbacks = []
        
        # Initialize components
        self.counters = RuntimeCounters()
        self.compression_mode = CompressionMode()
        self.ghost_reservoir = GhostTickReservoir()
        self.stack_checker = StackLogIntegrityChecker()
        
        # Initialize cores if available
        if CORES_AVAILABLE:
            self.alif_core = ALIFCore(self.counters)
            self.aleph_core = ALEPHCore(self.counters)
            self.detonation_sequencer = DetonationSequencer()
            self.entropy_analyzer = EntropyAnalyzer()
        else:
            self.alif_core = ALIFCore(self.counters)
            self.aleph_core = ALEPHCore(self.counters)
            logger.warning("Running in fallback mode - core modules unavailable")
        
        # Drift and timing correction
        self.tick_times = deque(maxlen=20)
        self.drift_correction_factor = 1.0
        self.max_drift_tolerance = 2.0  # seconds
        
        # Error tracking
        self.error_history = deque(maxlen=100)
        self.last_health_check = time.time()
        
    def register_callback(self, callback_fn: Callable):
        """Register callback for tick events"""
        self.callbacks.append(callback_fn)
    
    def run_tick_cycle(self) -> TickContext:
        """Execute a complete tick cycle with error handling"""
        now = time.time()
        
        # Check if it's time for a new tick
        if now - self.last_tick_time < self.tick_interval:
            return None
        
        # Calculate timing metrics
        delta_t = now - self.last_tick_time
        self.tick_times.append(delta_t)
        
        # Detect timing drift
        if self._is_timing_unstable():
            self.counters.drift_ticks += 1
            self._apply_drift_correction()
        
        # Create tick context
        self.tick_count += 1
        self.counters.tick_count += 1
        
        tick_context = TickContext(
            tick_id=self.tick_count,
            timestamp=datetime.now(),
            delta_t=delta_t,
            drift_score=self._calculate_drift_score()
        )
        
        # Check stack readiness
        if not self._wait_for_stack_ready(tick_context.tick_id):
            # Store as ghost tick if stack not ready
            self.ghost_reservoir.store_ghost_tick(tick_context)
            self.counters.stack_timeouts += 1
            return None
        
        tick_context.stack_ready = True
        self.last_tick_time = now
        
        # Process through ALIF and ALEPH
        try:
            alif_result = self.alif_core.on_tick(tick_context)
            aleph_result = self.aleph_core.on_tick(tick_context)
            
            # Update compression mode
            self.compression_mode.update_mode(
                self.alif_core.last_entry_time,
                self.aleph_core.last_validation_time,
                tick_context.echo_strength
            )
            
            # Check for path conflicts
            if self._detect_path_conflict(alif_result, aleph_result):
                self.counters.path_conflicts += 1
                tick_context.validated = False
            else:
                tick_context.validated = True
            
            # Execute callbacks
            for callback in self.callbacks:
                try:
                    callback(tick_context, alif_result, aleph_result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # Log tick data with integrity checking
            self._log_tick_data(tick_context, alif_result, aleph_result)
            
        except Exception as e:
            logger.error(f"Tick processing error: {e}")
            self.error_history.append({
                "tick_id": tick_context.tick_id,
                "error": str(e),
                "timestamp": datetime.now()
            })
            self.counters.fallbacks_triggered += 1
        
        # Periodic health check
        if now - self.last_health_check > 60:  # Every minute
            self._perform_health_check()
            self.last_health_check = now
        
        return tick_context
    
    def _wait_for_stack_ready(self, tick_id: int) -> bool:
        """Wait for stack to be ready with timeout"""
        timeout = 2.0
        start = time.time()
        
        while time.time() - start < timeout:
            if self._is_stack_ready(tick_id):
                return True
            time.sleep(0.05)
        
        return False
    
    def _is_stack_ready(self, tick_id: int) -> bool:
        """Check if stack is ready for processing"""
        # Simulate stack readiness check
        # In real implementation, check file locks, memory state, etc.
        return True
    
    def _is_timing_unstable(self) -> bool:
        """Detect timing instability"""
        if len(self.tick_times) < 5:
            return False
        
        diffs = list(self.tick_times)
        max_diff = max(diffs)
        min_diff = min(diffs)
        
        return (max_diff - min_diff) > self.max_drift_tolerance
    
    def _apply_drift_correction(self):
        """Apply timing drift correction"""
        if len(self.tick_times) > 0:
            avg_delta = np.mean(list(self.tick_times))
            self.drift_correction_factor = self.tick_interval / avg_delta
            logger.info(f"Applied drift correction factor: {self.drift_correction_factor:.3f}")
    
    def _calculate_drift_score(self) -> float:
        """Calculate current drift score"""
        if len(self.tick_times) < 2:
            return 0.0
        
        recent_times = list(self.tick_times)[-5:]
        std_dev = np.std(recent_times)
        return min(std_dev / self.tick_interval, 1.0)
    
    def _detect_path_conflict(self, alif_result: Dict, aleph_result: Dict) -> bool:
        """Detect conflicts between ALIF and ALEPH decisions"""
        alif_action = alif_result.get("action", "unknown")
        aleph_action = aleph_result.get("action", "unknown")
        
        # Define conflicting action pairs
        conflicts = [
            ("broadcast_glyph", "hold"),
            ("compress", "confirm")
        ]
        
        return (alif_action, aleph_action) in conflicts or (aleph_action, alif_action) in conflicts
    
    def _log_tick_data(self, tick: TickContext, alif_result: Dict, aleph_result: Dict):
        """Log tick data with integrity checking"""
        log_entry = {
            "tick_id": tick.tick_id,
            "timestamp": tick.timestamp.isoformat(),
            "entropy": tick.entropy,
            "echo_strength": tick.echo_strength,
            "drift_score": tick.drift_score,
            "alif_action": alif_result.get("action", "unknown"),
            "aleph_action": aleph_result.get("action", "unknown"),
            "compression_mode": self.compression_mode.mode,
            "validated": tick.validated
        }
        
        filename = f"tick_{tick.tick_id:06d}.json"
        self.stack_checker.secure_write(filename, log_entry)
    
    def _perform_health_check(self):
        """Perform system health check and cleanup"""
        # Check error rate
        recent_errors = [e for e in self.error_history 
                        if datetime.now() - e["timestamp"] < timedelta(minutes=5)]
        
        if len(recent_errors) > 10:
            logger.warning(f"High error rate detected: {len(recent_errors)} errors in 5 minutes")
        
        # Recovery ghost ticks if any
        recovered = 0
        for tick_id in list(self.ghost_reservoir.reservoir.keys()):
            if self._is_stack_ready(tick_id):
                recovered_tick = self.ghost_reservoir.recover_tick(tick_id)
                if recovered_tick:
                    self.counters.ghost_ticks_recovered += 1
                    recovered += 1
        
        if recovered > 0:
            logger.info(f"Recovered {recovered} ghost ticks during health check")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "tick_count": self.tick_count,
            "compression_mode": self.compression_mode.mode,
            "alif_compressed": self.compression_mode.alif_compressed,
            "aleph_compressed": self.compression_mode.aleph_compressed,
            "ghost_ticks_pending": self.ghost_reservoir.get_pending_count(),
            "recent_error_count": len(self.error_history),
            "drift_correction_factor": self.drift_correction_factor,
            "counters": self.counters.__dict__
        }
    
    def export_diagnostics(self, filepath: str = "system_diagnostics.json"):
        """Export comprehensive diagnostics"""
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "error_history": [
                {
                    "tick_id": e["tick_id"],
                    "error": e["error"],
                    "timestamp": e["timestamp"].isoformat()
                } for e in self.error_history
            ],
            "timing_history": list(self.tick_times),
            "compression_history": {
                "mode": self.compression_mode.mode,
                "last_change": self.compression_mode.last_mode_change.isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        
        logger.info(f"Diagnostics exported to {filepath}")

def create_tick_manager(tick_interval: float = 1.0) -> TickManager:
    """Factory function to create configured tick manager"""
    return TickManager(tick_interval=tick_interval)

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create tick manager
    tick_manager = create_tick_manager(tick_interval=1.0)
    
    # Register a simple callback
    def example_callback(tick_context, alif_result, aleph_result):
        print(f"[Tick {tick_context.tick_id}] "
              f"ALIF: {alif_result['action']} | "
              f"ALEPH: {aleph_result['action']} | "
              f"Entropy: {tick_context.entropy:.3f}")
    
    tick_manager.register_callback(example_callback)
    
    # Run simulation
    print("🚀 Starting Schwabot Tick Management System...")
    print("📊 Running 10 tick cycles for demonstration...")
    
    for i in range(10):
        tick = tick_manager.run_tick_cycle()
        if tick:
            print(f"✅ Tick {tick.tick_id} processed successfully")
        time.sleep(1.1)  # Slightly over tick interval to test timing
    
    # Print final status
    print("\n📋 Final System Status:")
    tick_manager.counters.print_summary()
    
    print(f"\n🔧 Compression Mode: {tick_manager.compression_mode.mode}")
    print(f"👻 Ghost Ticks Pending: {tick_manager.ghost_reservoir.get_pending_count()}")
    
    # Export diagnostics
    tick_manager.export_diagnostics("test_diagnostics.json")
    print("📄 Diagnostics exported to test_diagnostics.json") 