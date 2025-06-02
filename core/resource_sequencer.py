"""
Resource Sequencer
================

Manages system resources and trading sequences across different timeframes.
Implements adaptive position sizing, retry logic, and profit swing allocation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import psutil
import GPUtil
import time
import logging
from enum import Enum

class ResourceState(Enum):
    OPTIMAL = "optimal"      # System running at ideal capacity
    WARM = "warm"           # Elevated but acceptable load
    HOT = "hot"             # High load, need to reduce activity
    CRITICAL = "critical"   # Critical load, minimal trading

@dataclass
class ResourceMetrics:
    """Current system resource metrics"""
    cpu_percent: float
    gpu_percent: float
    memory_percent: float
    thermal_state: float
    timestamp: datetime
    resource_state: ResourceState

@dataclass
class SequenceMetrics:
    """Metrics for a trading sequence"""
    sequence_id: str
    start_time: datetime
    profit_target: float
    max_drawdown: float
    position_size: float
    retry_count: int
    last_retry: Optional[datetime]
    success_rate: float
    thermal_cost: float

class ResourceSequencer:
    """Manages system resources and trading sequences"""
    
    def __init__(
        self,
        max_cpu_percent: float = 80.0,
        max_gpu_percent: float = 85.0,
        max_memory_percent: float = 75.0,
        thermal_threshold: float = 0.8,
        retry_max_attempts: int = 3,
        retry_base_delay: float = 1.0
    ):
        self.max_cpu_percent = max_cpu_percent
        self.max_gpu_percent = max_gpu_percent
        self.max_memory_percent = max_memory_percent
        self.thermal_threshold = thermal_threshold
        self.retry_max_attempts = retry_max_attempts
        self.retry_base_delay = retry_base_delay
        
        # Initialize resource tracking
        self.resource_history: List[ResourceMetrics] = []
        self.active_sequences: Dict[str, SequenceMetrics] = {}
        self.sequence_history: List[SequenceMetrics] = []
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize resource state
        self.current_state = ResourceState.OPTIMAL
        self.last_state_change = datetime.now()
        
        # Initialize position sizing parameters
        self.base_position_size = 1.0
        self.min_position_size = 0.1
        self.position_size_step = 0.1
        
        # Initialize thermal management
        self.thermal_history: List[float] = []
        self.thermal_window = 100  # Track last 100 thermal readings

    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Get GPU metrics if available
            try:
                gpus = GPUtil.getGPUs()
                gpu_percent = gpus[0].load * 100 if gpus else 0.0
            except:
                gpu_percent = 0.0
            
            # Calculate thermal state (weighted average of CPU and GPU)
            thermal_state = (cpu_percent * 0.7 + gpu_percent * 0.3) / 100.0
            
            # Determine resource state
            if thermal_state > self.thermal_threshold:
                resource_state = ResourceState.CRITICAL
            elif thermal_state > self.thermal_threshold * 0.8:
                resource_state = ResourceState.HOT
            elif thermal_state > self.thermal_threshold * 0.6:
                resource_state = ResourceState.WARM
            else:
                resource_state = ResourceState.OPTIMAL
            
            metrics = ResourceMetrics(
                cpu_percent=cpu_percent,
                gpu_percent=gpu_percent,
                memory_percent=memory_percent,
                thermal_state=thermal_state,
                timestamp=datetime.now(),
                resource_state=resource_state
            )
            
            # Update history
            self.resource_history.append(metrics)
            if len(self.resource_history) > 1000:  # Keep last 1000 readings
                self.resource_history.pop(0)
            
            # Update thermal history
            self.thermal_history.append(thermal_state)
            if len(self.thermal_history) > self.thermal_window:
                self.thermal_history.pop(0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting resource metrics: {e}")
            return ResourceMetrics(
                cpu_percent=0.0,
                gpu_percent=0.0,
                memory_percent=0.0,
                thermal_state=0.0,
                timestamp=datetime.now(),
                resource_state=ResourceState.CRITICAL
            )

    def calculate_position_size(self, base_size: float) -> float:
        """Calculate adjusted position size based on system load"""
        metrics = self.get_resource_metrics()
        
        # Calculate size reduction factor based on resource state
        if metrics.resource_state == ResourceState.CRITICAL:
            reduction = 0.1  # 90% reduction
        elif metrics.resource_state == ResourceState.HOT:
            reduction = 0.3  # 70% reduction
        elif metrics.resource_state == ResourceState.WARM:
            reduction = 0.5  # 50% reduction
        else:
            reduction = 0.0  # No reduction
        
        # Apply reduction
        adjusted_size = base_size * (1 - reduction)
        
        # Ensure minimum size
        return max(adjusted_size, self.min_position_size)

    def should_retry_sequence(self, sequence_id: str) -> Tuple[bool, float]:
        """Determine if a sequence should be retried and calculate delay"""
        if sequence_id not in self.active_sequences:
            return False, 0.0
        
        sequence = self.active_sequences[sequence_id]
        
        # Check if max retries reached
        if sequence.retry_count >= self.retry_max_attempts:
            return False, 0.0
        
        # Calculate exponential backoff delay
        delay = self.retry_base_delay * (2 ** sequence.retry_count)
        
        # Add jitter to prevent thundering herd
        jitter = np.random.uniform(0, 0.1 * delay)
        total_delay = delay + jitter
        
        return True, total_delay

    def start_sequence(self, sequence_id: str, profit_target: float, max_drawdown: float) -> bool:
        """Start a new trading sequence"""
        metrics = self.get_resource_metrics()
        
        # Check if system can handle new sequence
        if metrics.resource_state == ResourceState.CRITICAL:
            self.logger.warning("Cannot start sequence: system in critical state")
            return False
        
        # Calculate initial position size
        position_size = self.calculate_position_size(self.base_position_size)
        
        # Create sequence metrics
        sequence = SequenceMetrics(
            sequence_id=sequence_id,
            start_time=datetime.now(),
            profit_target=profit_target,
            max_drawdown=max_drawdown,
            position_size=position_size,
            retry_count=0,
            last_retry=None,
            success_rate=0.0,
            thermal_cost=metrics.thermal_state
        )
        
        self.active_sequences[sequence_id] = sequence
        return True

    def update_sequence(self, sequence_id: str, success: bool, profit: float) -> None:
        """Update sequence metrics after execution"""
        if sequence_id not in self.active_sequences:
            return
        
        sequence = self.active_sequences[sequence_id]
        
        # Update success rate
        total_trades = sequence.retry_count + 1
        sequence.success_rate = ((sequence.success_rate * (total_trades - 1)) + 
                               (1.0 if success else 0.0)) / total_trades
        
        # Update thermal cost
        metrics = self.get_resource_metrics()
        sequence.thermal_cost = (sequence.thermal_cost + metrics.thermal_state) / 2
        
        # Store in history if sequence complete
        if success or sequence.retry_count >= self.retry_max_attempts:
            self.sequence_history.append(sequence)
            del self.active_sequences[sequence_id]

    def get_optimal_sequence_params(self) -> Dict[str, float]:
        """Get optimal parameters for new sequences based on history"""
        if not self.sequence_history:
            return {
                'position_size': self.base_position_size,
                'profit_target': 0.02,  # 2% default
                'max_drawdown': 0.01    # 1% default
            }
        
        # Calculate averages from successful sequences
        successful = [s for s in self.sequence_history if s.success_rate > 0.5]
        if not successful:
            return {
                'position_size': self.base_position_size * 0.5,
                'profit_target': 0.01,
                'max_drawdown': 0.005
            }
        
        return {
            'position_size': np.mean([s.position_size for s in successful]),
            'profit_target': np.mean([s.profit_target for s in successful]),
            'max_drawdown': np.mean([s.max_drawdown for s in successful])
        }

    def get_resource_report(self) -> Dict:
        """Get comprehensive resource usage report"""
        metrics = self.get_resource_metrics()
        
        return {
            'current_state': metrics.resource_state.value,
            'cpu_usage': metrics.cpu_percent,
            'gpu_usage': metrics.gpu_percent,
            'memory_usage': metrics.memory_percent,
            'thermal_state': metrics.thermal_state,
            'active_sequences': len(self.active_sequences),
            'sequence_success_rate': np.mean([s.success_rate for s in self.sequence_history]) 
                if self.sequence_history else 0.0,
            'average_thermal_cost': np.mean(self.thermal_history) if self.thermal_history else 0.0
        }

# Example usage:
"""
from core.resource_sequencer import ResourceSequencer

# Initialize sequencer
sequencer = ResourceSequencer()

# Start a trading sequence
sequence_id = "trade_001"
if sequencer.start_sequence(sequence_id, profit_target=0.02, max_drawdown=0.01):
    # Get position size
    position_size = sequencer.calculate_position_size(1.0)
    
    # Execute trade...
    
    # Update sequence
    sequencer.update_sequence(sequence_id, success=True, profit=0.015)
    
    # Get resource report
    report = sequencer.get_resource_report()
    print(f"Resource state: {report['current_state']}")
""" 