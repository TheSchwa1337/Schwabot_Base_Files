"""
Line Render Engine for Schwabot v0.3
Processes each tick into a matrix-viewable row with safety checks
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import yaml
from pathlib import Path
import psutil
from hashlib import sha256
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

# Constants
COLLISION_THRESHOLDS = {
    'safe': 120,
    'warn': 180,
    'fail': float('inf')
}

DRIFT_THRESHOLDS = {
    'safe': 1.0,
    'warn': 2.5,
    'fail': float('inf')
}

TIMING_THRESHOLDS = {
    'safe': 20,  # μs
    'warn': 25,  # μs
    'fail': float('inf')
}

TEMP_THRESHOLDS = {
    'safe': 65,  # °C
    'elevated': 75,  # °C
    'zpe_risk': float('inf')
}

# New constants for scalability
MAX_NODES = 1000  # Maximum number of nodes before label toggling
BATCH_SIZE = 100  # Number of nodes to process in each batch
MAX_MEMORY_USAGE = 0.8  # Maximum memory usage threshold (80%)
LABEL_TOGGLE_THRESHOLD = 500  # Number of nodes before labels are toggled

@dataclass
class LineState:
    """State of a rendered line"""
    tick: int
    timestamp: str
    value: float
    hash: str
    cpu_time: float
    profit: float
    temp_zone: str
    collision_score: float
    drift: float
    volatility: float
    status: str
    matrix_response: str
    override: bool = False
    show_label: bool = True  # New field for label visibility

class LineRenderEngine:
    """
    Implements line-by-line rendering of tick data with safety checks
    """
    
    def __init__(self, log_path: str = "rendered_tick_memkey.log"):
        """Initialize line render engine"""
        self.log_path = Path(log_path)
        self.line_history: List[LineState] = []
        self.matrix_state = "hold"
        self.load_matrix_paths()
        
        # Initialize thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory monitoring
        self._last_memory_check = datetime.now()
        self._memory_check_interval = 60  # seconds
    
    def load_matrix_paths(self):
        """Load matrix response paths from YAML"""
        try:
            with open("matrix_response_paths.yaml", 'r') as f:
                self.matrix_paths = yaml.safe_load(f)
        except FileNotFoundError:
            self.matrix_paths = {
                "safe": "hold",
                "warn": "delay_entry",
                "fail": "matrix_realign",
                "ZPE-risk": "cooldown_abort"
            }
    
    def calculate_hash(self, timestamp: str, value: float) -> str:
        """
        Calculate SHA-256 hash
        
        Args:
            timestamp: ISO format timestamp
            value: Tick value
            
        Returns:
            SHA-256 hash as hex string
        """
        input_str = f"{timestamp}{value}"
        return sha256(input_str.encode()).hexdigest()
    
    def calculate_collision_score(self, hash_value: str) -> float:
        """
        Calculate collision entropy score
        
        Args:
            hash_value: SHA-256 hash
            
        Returns:
            Collision score (0-225)
        """
        prefix = hash_value[:8]
        suffix = hash_value[-8:]
        
        # Count prefix collisions
        prefix_coll = sum(1 for i in range(7) if prefix[i] == prefix[i+1])
        
        # Count suffix collisions
        suffix_coll = sum(1 for i in range(7) if suffix[i] == suffix[i+1])
        
        # Calculate weighted score
        score = (prefix_coll * 1.25) + suffix_coll
        
        # Normalize to 0-225 range
        return min((score / 8.0) * 225, 225)
    
    def get_temp_zone(self) -> str:
        """
        Get current CPU temperature zone
        
        Returns:
            Temperature zone string
        """
        try:
            temp = psutil.sensors_temperatures()['coretemp'][0].current
            if temp < TEMP_THRESHOLDS['safe']:
                return "safe"
            elif temp < TEMP_THRESHOLDS['elevated']:
                return "elevated"
            else:
                return "ZPE-risk"
        except:
            return "safe"  # Default to safe if can't read temp
    
    def get_line_status(self, 
                       collision_score: float,
                       drift: float,
                       cpu_time: float,
                       temp_zone: str) -> str:
        """
        Determine line status based on metrics
        
        Args:
            collision_score: BCHS collision score
            drift: Profit drift
            cpu_time: CPU time in microseconds
            temp_zone: Temperature zone
            
        Returns:
            Status string
        """
        if temp_zone == "ZPE-risk":
            return "ZPE-risk"
        
        if (collision_score >= COLLISION_THRESHOLDS['fail'] or
            drift >= DRIFT_THRESHOLDS['fail'] or
            cpu_time >= TIMING_THRESHOLDS['fail']):
            return "fail"
        
        if (collision_score >= COLLISION_THRESHOLDS['warn'] or
            drift >= DRIFT_THRESHOLDS['warn'] or
            cpu_time >= TIMING_THRESHOLDS['warn']):
            return "warn"
        
        return "safe"
    
    def get_matrix_response(self, status: str) -> str:
        """
        Get matrix response for status
        
        Args:
            status: Line status
            
        Returns:
            Matrix response string
        """
        return self.matrix_paths.get(status, "hold")
    
    def check_memory_usage(self) -> bool:
        """
        Check if memory usage is within safe limits
        
        Returns:
            bool: True if memory usage is safe, False otherwise
        """
        current_time = datetime.now()
        if (current_time - self._last_memory_check).seconds < self._memory_check_interval:
            return True
            
        try:
            memory = psutil.virtual_memory()
            if memory.percent > MAX_MEMORY_USAGE * 100:
                self.logger.warning(f"High memory usage detected: {memory.percent}%")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking memory usage: {e}")
            return True  # Continue if we can't check memory
    
    def process_batch(self, lines: List[LineState]) -> List[LineState]:
        """
        Process a batch of lines with error handling
        
        Args:
            lines: List of LineState objects to process
            
        Returns:
            List of processed LineState objects
        """
        processed_lines = []
        for line in lines:
            try:
                # Check memory usage
                if not self.check_memory_usage():
                    self.logger.warning("Memory usage too high, pausing processing")
                    break
                    
                # Process line with error handling
                processed_line = self._process_single_line(line)
                if processed_line:
                    processed_lines.append(processed_line)
                    
            except Exception as e:
                self.logger.error(f"Error processing line {line.tick}: {e}")
                continue
                
        return processed_lines
    
    def _process_single_line(self, line: LineState) -> Optional[LineState]:
        """
        Process a single line with error handling
        
        Args:
            line: LineState object to process
            
        Returns:
            Processed LineState object or None if processing failed
        """
        try:
            # Calculate hash and metrics
            hash_value = self.calculate_hash(line.timestamp, line.value)
            collision_score = self.calculate_collision_score(hash_value)
            temp_zone = self.get_temp_zone()
            
            # Update line state
            line.hash = hash_value
            line.collision_score = collision_score
            line.temp_zone = temp_zone
            line.status = self.get_line_status(
                collision_score, line.drift, line.cpu_time, temp_zone
            )
            line.matrix_response = self.get_matrix_response(line.status)
            
            # Toggle label visibility based on node count
            line.show_label = len(self.line_history) < LABEL_TOGGLE_THRESHOLD
            
            return line
            
        except Exception as e:
            self.logger.error(f"Error processing line {line.tick}: {e}")
            return None
    
    def render_line(self,
                   tick: int,
                   value: float,
                   profit: float,
                   drift: float,
                   volatility: float) -> Optional[LineState]:
        """
        Render a single line from tick data with error handling
        
        Args:
            tick: Tick index
            value: Trade value
            profit: Current profit
            drift: Profit drift
            volatility: Volatility measure
            
        Returns:
            LineState object or None if rendering failed
        """
        try:
            # Check if we're approaching node limit
            if len(self.line_history) >= MAX_NODES:
                self.logger.warning(f"Maximum node limit reached ({MAX_NODES})")
                return None
                
            # Get timestamp and calculate hash
            timestamp = datetime.now().isoformat()
            start_time = datetime.now()
            
            # Create initial line state
            line = LineState(
                tick=tick,
                timestamp=timestamp,
                value=value,
                hash="",  # Will be calculated in _process_single_line
                cpu_time=(datetime.now() - start_time).microseconds,
                profit=profit,
                temp_zone="safe",  # Will be updated in _process_single_line
                collision_score=0.0,  # Will be calculated in _process_single_line
                drift=drift,
                volatility=volatility,
                status="safe",  # Will be updated in _process_single_line
                matrix_response="hold",  # Will be updated in _process_single_line
                show_label=len(self.line_history) < LABEL_TOGGLE_THRESHOLD
            )
            
            # Process line with error handling
            processed_line = self._process_single_line(line)
            if processed_line:
                with self._lock:
                    self.line_history.append(processed_line)
                    self.log_line(processed_line)
                    
            return processed_line
            
        except Exception as e:
            self.logger.error(f"Error rendering line for tick {tick}: {e}")
            return None
    
    def render_batch(self, lines: List[Tuple[int, float, float, float, float]]) -> List[LineState]:
        """
        Render a batch of lines with parallel processing
        
        Args:
            lines: List of (tick, value, profit, drift, volatility) tuples
            
        Returns:
            List of processed LineState objects
        """
        try:
            # Check if batch would exceed node limit
            if len(self.line_history) + len(lines) > MAX_NODES:
                self.logger.warning(f"Batch would exceed maximum node limit ({MAX_NODES})")
                lines = lines[:MAX_NODES - len(self.line_history)]
            
            # Process lines in parallel
            futures = []
            for tick, value, profit, drift, volatility in lines:
                future = self._executor.submit(
                    self.render_line, tick, value, profit, drift, volatility
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {e}")
                    continue
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch rendering: {e}")
            return []
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self._executor.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def log_line(self, line: LineState):
        """
        Log line to file
        
        Args:
            line: LineState object
        """
        log_entry = (
            f"tick={line.tick} | {line.hash[:8]}... | "
            f"BCHS={line.collision_score:.1f} | "
            f"τ={line.cpu_time:.1f}μs | "
            f"DRIFT={line.drift:+.1f}% | "
            f"ZPE={line.temp_zone} | "
            f"→ {line.matrix_response}"
        )
        
        with open(self.log_path, 'a') as f:
            f.write(log_entry + '\n')
    
    def get_history(self) -> List[LineState]:
        """
        Get line history
        
        Returns:
            List of LineState objects
        """
        return self.line_history 