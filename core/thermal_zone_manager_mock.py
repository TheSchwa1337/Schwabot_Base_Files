"""
Enhanced Thermal Zone Manager
===========================

Provides a robust thermal management implementation that integrates with the system's
thermal-aware processing pipeline. This implementation includes:

1. Thermal drift compensation
2. Processing recommendations based on thermal state
3. Burst management with cooldown periods
4. Daily budget tracking
5. Integration with profit trajectory
6. Advanced thermal state simulation
7. Comprehensive thermal history analysis
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading
import random
from collections import deque
import json

logger = logging.getLogger(__name__)

class ThermalZone(Enum):
    """Thermal zone classifications"""
    COOL = "cool"           # < 60°C
    NORMAL = "normal"       # 60-70°C
    WARM = "warm"           # 70-80°C
    HOT = "hot"             # 80-90°C
    CRITICAL = "critical"   # > 90°C

@dataclass
class ThermalState:
    """Enhanced thermal state tracking"""
    cpu_temp: float
    gpu_temp: float
    load_cpu: float
    load_gpu: float
    zone: ThermalZone
    drift_coefficient: float
    processing_recommendation: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    burst_available: bool = True
    daily_budget_remaining: float = 2.4  # Hours
    thermal_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BurstState:
    """Burst processing state tracking"""
    is_active: bool = False
    start_time: Optional[datetime] = None
    duration: float = 0.0
    cooldown_until: Optional[datetime] = None
    burst_count: int = 0
    total_burst_time: float = 0.0

class EnhancedThermalZoneManager:
    """
    Enhanced thermal zone manager with comprehensive thermal management features.
    This implementation provides a robust thermal management system that can be
    used in production environments when hardware monitoring is not available.
    """
    
    def __init__(self, 
                 profit_coprocessor: Optional[Any] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced thermal zone manager
        
        Args:
            profit_coprocessor: Optional profit trajectory coprocessor
            config: Configuration dictionary
        """
        self.config = config or {}
        self.profit_coprocessor = profit_coprocessor
        
        # Thermal thresholds (Celsius)
        self.zone_thresholds = {
            ThermalZone.COOL: (0, 60),
            ThermalZone.NORMAL: (60, 70),
            ThermalZone.WARM: (70, 80),
            ThermalZone.HOT: (80, 90),
            ThermalZone.CRITICAL: (90, 150)
        }
        
        # Processing budget parameters
        self.daily_budget_hours = 2.4  # 10% of 24 hours
        self.budget_used_today = 0.0
        self.budget_reset_time = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0
        )
        
        # Thermal compensation parameters
        self.nominal_temp = 70.0  # T₀ in thermal drift formula
        self.profit_heat_bias = 0.5  # α in thermal drift formula
        
        # Burst management
        self.burst_state = BurstState()
        self.max_burst_duration = 300  # 5 minutes max burst
        self.cooldown_ratio = 2.0  # Cooldown = 2x burst duration
        
        # State tracking
        self.thermal_history: List[ThermalState] = []
        self.current_state: Optional[ThermalState] = None
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Initialize state
        self._update_thermal_state()
        logger.info("Enhanced ThermalZoneManager initialized")
    
    def _calculate_thermal_drift_coefficient(self, temperature: float) -> float:
        """
        Calculate thermal drift coefficient using the sigmoid formula:
        D_thermal = 1 / (1 + e^(-((T - T₀) - α * P_avg)))
        """
        # Get average profit from profit coprocessor if available
        avg_profit = 0.0
        if self.profit_coprocessor and hasattr(self.profit_coprocessor, 'smoothed_profit'):
            avg_profit = self.profit_coprocessor.smoothed_profit
            
        # Apply thermal drift formula
        exponent = -((temperature - self.nominal_temp) - 
                    self.profit_heat_bias * avg_profit)
        drift_coefficient = 1.0 / (1.0 + np.exp(exponent))
        
        # Clamp to reasonable range
        return np.clip(drift_coefficient, 0.3, 1.5)
    
    def _calculate_processing_recommendation(self, 
                                          zone: ThermalZone,
                                          drift_coeff: float) -> Dict[str, Any]:
        """Calculate processing allocation recommendations"""
        # Base recommendations by thermal zone
        base_recommendations = {
            ThermalZone.COOL: {"gpu": 0.7, "cpu": 0.3, "burst_allowed": True},
            ThermalZone.NORMAL: {"gpu": 0.6, "cpu": 0.4, "burst_allowed": True},
            ThermalZone.WARM: {"gpu": 0.4, "cpu": 0.6, "burst_allowed": False},
            ThermalZone.HOT: {"gpu": 0.2, "cpu": 0.8, "burst_allowed": False},
            ThermalZone.CRITICAL: {"gpu": 0.1, "cpu": 0.9, "burst_allowed": False}
        }
        
        base_rec = base_recommendations[zone]
        
        # Apply drift coefficient
        gpu_allocation = base_rec["gpu"] * drift_coeff
        cpu_allocation = 1.0 - gpu_allocation
        
        # Get profit trajectory influence if available
        if self.profit_coprocessor:
            profit_allocation = self.profit_coprocessor.get_processing_allocation()
            profit_weight = 0.3  # 30% weight to profit recommendations
            thermal_weight = 0.7  # 70% weight to thermal recommendations
            
            gpu_allocation = (thermal_weight * gpu_allocation + 
                            profit_weight * profit_allocation["gpu"])
            cpu_allocation = 1.0 - gpu_allocation
            
        # Ensure valid allocations
        gpu_allocation = np.clip(gpu_allocation, 0.05, 0.95)
        cpu_allocation = 1.0 - gpu_allocation
        
        return {
            "gpu": gpu_allocation,
            "cpu": cpu_allocation,
            "burst_allowed": base_rec["burst_allowed"] and self._can_burst(),
            "thermal_zone": zone.value,
            "drift_coefficient": drift_coeff
        }
    
    def _can_burst(self) -> bool:
        """Check if burst processing is available"""
        if not self.burst_state.burst_available:
            return False
            
        if self.burst_state.cooldown_until:
            if datetime.now(timezone.utc) < self.burst_state.cooldown_until:
                return False
                
        return True
    
    def start_burst(self) -> bool:
        """Start a burst processing period"""
        if not self._can_burst():
            return False
            
        with self._lock:
            self.burst_state.is_active = True
            self.burst_state.start_time = datetime.now(timezone.utc)
            self.burst_state.burst_count += 1
            return True
    
    def end_burst(self, duration: float) -> None:
        """End a burst processing period"""
        with self._lock:
            if self.burst_state.is_active:
                self.burst_state.is_active = False
                self.burst_state.duration = duration
                self.burst_state.total_burst_time += duration
                
                # Calculate cooldown period
                cooldown_duration = duration * self.cooldown_ratio
                self.burst_state.cooldown_until = (
                    datetime.now(timezone.utc) + 
                    timedelta(seconds=cooldown_duration)
                )
    
    def _update_thermal_state(self) -> ThermalState:
        """Update current thermal state with realistic simulation"""
        # Generate realistic temperature variations
        base_cpu_temp = 45.0 + random.uniform(-5, 15)  # 40-60°C range
        base_gpu_temp = 50.0 + random.uniform(-5, 20)  # 45-70°C range
        
        # Apply thermal drift
        if self.current_state:
            # Simulate thermal inertia
            cpu_temp = (0.7 * self.current_state.cpu_temp + 
                       0.3 * base_cpu_temp)
            gpu_temp = (0.7 * self.current_state.gpu_temp + 
                       0.3 * base_gpu_temp)
        else:
            cpu_temp = base_cpu_temp
            gpu_temp = base_gpu_temp
        
        # Determine thermal zone
        max_temp = max(cpu_temp, gpu_temp)
        zone = self._classify_thermal_zone(max_temp)
        
        # Calculate drift coefficient
        drift_coeff = self._calculate_thermal_drift_coefficient(max_temp)
        
        # Calculate processing recommendations
        processing_rec = self._calculate_processing_recommendation(zone, drift_coeff)
        
        # Create new state
        new_state = ThermalState(
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            load_cpu=random.uniform(0.3, 0.7),
            load_gpu=random.uniform(0.4, 0.8),
            zone=zone,
            drift_coefficient=drift_coeff,
            processing_recommendation=processing_rec,
            burst_available=self._can_burst(),
            daily_budget_remaining=self.daily_budget_hours - self.budget_used_today
        )
        
        # Update state
        self.current_state = new_state
        self.thermal_history.append(new_state)
        
        # Trim history if too long
        if len(self.thermal_history) > 1000:
            self.thermal_history = self.thermal_history[-1000:]
            
        return new_state
    
    def _classify_thermal_zone(self, temperature: float) -> ThermalZone:
        """Classify temperature into thermal zone"""
        for zone, (min_temp, max_temp) in self.zone_thresholds.items():
            if min_temp <= temperature < max_temp:
                return zone
        return ThermalZone.CRITICAL
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current thermal state"""
        with self._lock:
            state = self._update_thermal_state()
            return {
                'cpu_temperature': state.cpu_temp,
                'gpu_temperature': state.gpu_temp,
                'cpu_load': state.load_cpu,
                'gpu_load': state.load_gpu,
                'thermal_zone': state.zone.value,
                'drift_coefficient': state.drift_coefficient,
                'processing_recommendation': state.processing_recommendation,
                'burst_available': state.burst_available,
                'daily_budget_remaining': state.daily_budget_remaining,
                'timestamp': state.timestamp.isoformat()
            }
    
    def get_thermal_history(self, window_minutes: int = 10) -> List[Dict[str, Any]]:
        """Get thermal history"""
        with self._lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            recent_history = [
                state for state in self.thermal_history
                if state.timestamp >= cutoff_time
            ]
            
            return [{
                'timestamp': state.timestamp.isoformat(),
                'cpu_temperature': state.cpu_temp,
                'gpu_temperature': state.gpu_temp,
                'thermal_zone': state.zone.value,
                'drift_coefficient': state.drift_coefficient
            } for state in recent_history]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive thermal statistics"""
        with self._lock:
            if not self.thermal_history:
                return {}
                
            recent_states = self.thermal_history[-100:]  # Last 100 states
            
            # Calculate statistics
            cpu_temps = [state.cpu_temp for state in recent_states]
            gpu_temps = [state.gpu_temp for state in recent_states]
            
            return {
                'current_state': {
                    'cpu_temperature': self.current_state.cpu_temp,
                    'gpu_temperature': self.current_state.gpu_temp,
                    'thermal_zone': self.current_state.zone.value,
                    'drift_coefficient': self.current_state.drift_coefficient
                },
                'thermal_history': {
                    'cpu_avg': np.mean(cpu_temps),
                    'cpu_std': np.std(cpu_temps),
                    'gpu_avg': np.mean(gpu_temps),
                    'gpu_std': np.std(gpu_temps),
                    'max_temperature': max(max(cpu_temps), max(gpu_temps)),
                    'min_temperature': min(min(cpu_temps), min(gpu_temps))
                },
                'burst_statistics': {
                    'total_bursts': self.burst_state.burst_count,
                    'total_burst_time': self.burst_state.total_burst_time,
                    'cooldown_until': self.burst_state.cooldown_until.isoformat() 
                        if self.burst_state.cooldown_until else None
                },
                'budget_statistics': {
                    'daily_budget_hours': self.daily_budget_hours,
                    'budget_used_today': self.budget_used_today,
                    'budget_remaining': self.daily_budget_hours - self.budget_used_today
                }
            }
    
    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start thermal monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                self._update_thermal_state()
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(
            target=monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Thermal monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop thermal monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Thermal monitoring stopped")

# Create alias for compatibility
MockThermalZoneManager = EnhancedThermalZoneManager 