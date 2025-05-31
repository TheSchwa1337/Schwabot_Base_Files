"""
MEMKEY CPU-SYNC Math Logic
Implements memkey-triggered hash cycles and microtimed CPU-sequenced pattern recognition
"""

from hashlib import sha256
from datetime import datetime
from time import perf_counter
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

# Constants
PHI = 3.61803398875  # Golden Ratio for drift offset
MICROSECOND = 1e-6   # Microsecond scaling factor
TIMING_THRESHOLD = 10.0  # CPU timing threshold in microseconds

class MemkeyResult(NamedTuple):
    """Result of memkey calculation"""
    timestamp: str
    value: float
    hash_full: str
    hash_prefix: Optional[str]
    cpu_time: float
    fault_triggered: bool
    fault_type: str
    memkey: Optional[str]

@dataclass
class MemkeyStats:
    """Statistics for memkey calculations"""
    total_ticks: int = 0
    fault_count: int = 0
    prefix_faults: int = 0
    suffix_faults: int = 0
    timing_faults: int = 0
    memkey_count: int = 0
    avg_cpu_time: float = 0.0
    max_cpu_time: float = 0.0

class MemkeySync:
    """
    Implements MEMKEY CPU-SYNC math logic for Schwabot
    """
    
    def __init__(self):
        """Initialize MEMKEY sync system"""
        self.stats = MemkeyStats()
        self.memkey_history: Dict[str, List[int]] = {}
        self.fault_log: List[MemkeyResult] = []
    
    def calculate_tick_value(self, tick_index: int) -> float:
        """
        Calculate value for tick i using phi-modulated drift
        
        Args:
            tick_index: Current tick index
            
        Returns:
            Modulated drift value
        """
        return 100 + (tick_index * PHI) % 400
    
    def calculate_hash(self, timestamp: str, value: float) -> str:
        """
        Calculate SHA-256 hash of timestamp and value
        
        Args:
            timestamp: ISO format timestamp
            value: Tick value
            
        Returns:
            SHA-256 hash as hex string
        """
        input_str = f"{timestamp}{value}"
        return sha256(input_str.encode()).hexdigest()
    
    def check_faults(self, 
                    hash_value: str, 
                    cpu_time: float) -> Tuple[bool, str]:
        """
        Check for fault conditions in hash and timing
        
        Args:
            hash_value: Full hash value
            cpu_time: CPU time in microseconds
            
        Returns:
            Tuple of (fault_triggered, fault_type)
        """
        faults = []
        
        if hash_value.startswith("00"):
            faults.append("prefix")
        if hash_value.endswith("ff"):
            faults.append("suffix")
        if cpu_time > TIMING_THRESHOLD:
            faults.append("timing")
        
        fault_triggered = len(faults) > 0
        fault_type = "+".join(faults) if faults else "none"
        
        return fault_triggered, fault_type
    
    def process_tick(self, tick_index: int) -> MemkeyResult:
        """
        Process a single tick through the MEMKEY system
        
        Args:
            tick_index: Current tick index
            
        Returns:
            MemkeyResult with all calculations
        """
        # Get timestamp and calculate value
        timestamp = datetime.now().isoformat()
        value = self.calculate_tick_value(tick_index)
        
        # Measure CPU time for hash calculation
        start_time = perf_counter()
        hash_value = self.calculate_hash(timestamp, value)
        end_time = perf_counter()
        cpu_time = (end_time - start_time) * MICROSECOND
        
        # Check for faults
        fault_triggered, fault_type = self.check_faults(hash_value, cpu_time)
        
        # Extract memkey if hash starts with "00"
        hash_prefix = hash_value[:8] if hash_value.startswith("00") else None
        memkey = hash_prefix if hash_value.startswith("00") else None
        
        # Update statistics
        self.stats.total_ticks += 1
        if fault_triggered:
            self.stats.fault_count += 1
            if "prefix" in fault_type:
                self.stats.prefix_faults += 1
            if "suffix" in fault_type:
                self.stats.suffix_faults += 1
            if "timing" in fault_type:
                self.stats.timing_faults += 1
        if memkey:
            self.stats.memkey_count += 1
            if memkey not in self.memkey_history:
                self.memkey_history[memkey] = []
            self.memkey_history[memkey].append(tick_index)
        
        # Update timing stats
        self.stats.avg_cpu_time = (self.stats.avg_cpu_time * (self.stats.total_ticks - 1) + 
                                 cpu_time) / self.stats.total_ticks
        self.stats.max_cpu_time = max(self.stats.max_cpu_time, cpu_time)
        
        # Create result
        result = MemkeyResult(
            timestamp=timestamp,
            value=value,
            hash_full=hash_value,
            hash_prefix=hash_prefix,
            cpu_time=cpu_time,
            fault_triggered=fault_triggered,
            fault_type=fault_type,
            memkey=memkey
        )
        
        # Log fault if triggered
        if fault_triggered:
            self.fault_log.append(result)
        
        return result
    
    def get_stats(self) -> Dict:
        """
        Get current statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_ticks': self.stats.total_ticks,
            'fault_count': self.stats.fault_count,
            'prefix_faults': self.stats.prefix_faults,
            'suffix_faults': self.stats.suffix_faults,
            'timing_faults': self.stats.timing_faults,
            'memkey_count': self.stats.memkey_count,
            'avg_cpu_time': self.stats.avg_cpu_time,
            'max_cpu_time': self.stats.max_cpu_time
        }
    
    def get_fault_log(self) -> pd.DataFrame:
        """
        Get fault log as DataFrame
        
        Returns:
            DataFrame of fault entries
        """
        return pd.DataFrame(self.fault_log)
    
    def get_memkey_history(self) -> Dict[str, List[int]]:
        """
        Get history of memkey occurrences
        
        Returns:
            Dictionary mapping memkeys to tick indices
        """
        return self.memkey_history
    
    def check_memkey_reuse(self, memkey: str) -> bool:
        """
        Check if memkey has been seen before
        
        Args:
            memkey: Memkey to check
            
        Returns:
            True if memkey has been seen before
        """
        return memkey in self.memkey_history and len(self.memkey_history[memkey]) > 1 