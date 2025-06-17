"""
Mock Thermal Zone Manager
========================

Provides the same interface as thermal_zone_manager.py but with mock functionality
for testing and development when GPUtil is not available.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import threading
import random

logger = logging.getLogger(__name__)

class ThermalState(Enum):
    """Thermal state enumeration"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ThermalMetrics:
    """Mock thermal metrics"""
    cpu_temperature: float = 45.0
    gpu_temperature: float = 50.0
    cooling_efficiency: float = 0.85
    thermal_throttling: bool = False
    timestamp: float = 0.0

class ThermalZoneManager:
    """Mock thermal zone manager for testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mock thermal manager"""
        self.config = config or {}
        
        # Mock thermal state
        self.current_metrics = ThermalMetrics()
        self.thermal_state = ThermalState.NORMAL
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Thresholds
        self.normal_threshold = 70.0
        self.elevated_threshold = 80.0
        self.high_threshold = 90.0
        self.critical_threshold = 95.0
        
        logger.info("Mock ThermalZoneManager initialized")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current thermal state"""
        # Generate realistic mock data
        base_cpu_temp = 45.0 + random.uniform(-5, 15)  # 40-60°C range
        base_gpu_temp = 50.0 + random.uniform(-5, 20)  # 45-70°C range
        
        self.current_metrics = ThermalMetrics(
            cpu_temperature=base_cpu_temp,
            gpu_temperature=base_gpu_temp,
            cooling_efficiency=0.8 + random.uniform(-0.1, 0.15),
            thermal_throttling=base_gpu_temp > 85.0,
            timestamp=time.time()
        )
        
        # Determine thermal state
        max_temp = max(base_cpu_temp, base_gpu_temp)
        if max_temp >= self.critical_threshold:
            self.thermal_state = ThermalState.CRITICAL
        elif max_temp >= self.high_threshold:
            self.thermal_state = ThermalState.HIGH
        elif max_temp >= self.elevated_threshold:
            self.thermal_state = ThermalState.ELEVATED
        else:
            self.thermal_state = ThermalState.NORMAL
        
        return {
            'cpu_temperature': self.current_metrics.cpu_temperature,
            'gpu_temperature': self.current_metrics.gpu_temperature,
            'cooling_efficiency': self.current_metrics.cooling_efficiency,
            'thermal_throttling': self.current_metrics.thermal_throttling,
            'thermal_state': self.thermal_state.value,
            'timestamp': self.current_metrics.timestamp
        }
    
    def apply_correction(self, action: str, magnitude: float) -> bool:
        """Apply thermal correction (mock implementation)"""
        logger.info(f"Mock thermal correction: {action} with magnitude {magnitude:.2f}")
        
        # Simulate correction effects
        if action == "reduce_thermal_load":
            # Simulate temperature reduction
            self.current_metrics.cpu_temperature *= (1.0 - magnitude * 0.1)
            self.current_metrics.gpu_temperature *= (1.0 - magnitude * 0.1)
            
        elif action == "increase_cooling":
            # Simulate improved cooling efficiency
            self.current_metrics.cooling_efficiency = min(1.0, 
                self.current_metrics.cooling_efficiency + magnitude * 0.1)
            
        elif action == "optimize_cooling_efficiency":
            # Simulate cooling optimization
            self.current_metrics.cooling_efficiency = min(1.0,
                self.current_metrics.cooling_efficiency + magnitude * 0.05)
        
        return True
    
    def start_monitoring(self) -> None:
        """Start thermal monitoring (mock)"""
        self.is_monitoring = True
        logger.info("Mock thermal monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop thermal monitoring (mock)"""
        self.is_monitoring = False
        logger.info("Mock thermal monitoring stopped")
    
    def get_thermal_history(self, window_minutes: int = 10) -> List[Dict[str, Any]]:
        """Get thermal history (mock)"""
        # Generate mock history
        history = []
        current_time = time.time()
        
        for i in range(window_minutes):
            timestamp = current_time - (window_minutes - i) * 60
            temp_variation = random.uniform(-2, 2)
            
            history.append({
                'timestamp': timestamp,
                'cpu_temperature': 45.0 + temp_variation,
                'gpu_temperature': 50.0 + temp_variation,
                'cooling_efficiency': 0.85 + random.uniform(-0.05, 0.05),
                'thermal_state': ThermalState.NORMAL.value
            })
        
        return history

# Create alias for compatibility
MockThermalZoneManager = ThermalZoneManager 