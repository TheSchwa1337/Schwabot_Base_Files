"""
ZBE Thermal Cooldown Controller
===============================

Implements thermal protection and cooldown logic for the Anti-Pole system.
Prevents overheating during high-frequency trading and maintains system stability.
"""

import numpy as np
import time
import psutil
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ThermalState(Enum):
    """System thermal states"""
    COLD = "COLD"           # < 30% thermal load
    COOL = "COOL"           # 30-50% thermal load  
    WARM = "WARM"           # 50-70% thermal load
    HOT = "HOT"             # 70-85% thermal load
    CRITICAL = "CRITICAL"   # > 85% thermal load
    EMERGENCY = "EMERGENCY" # > 95% thermal load

@dataclass
class ThermalMetrics:
    """Thermal system metrics"""
    timestamp: datetime
    cpu_temp: float
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    thermal_load: float
    state: ThermalState
    cooldown_active: bool
    time_to_cool: Optional[timedelta] = None

class ZBEThermalCooldown:
    """
    ZBE (Zero-Boil-Entropy) Thermal Cooldown System
    
    Monitors system resources and implements adaptive cooldown
    to prevent thermal overload during intensive computations.
    """
    
    def __init__(self, window_size: int = 32, cooldown_threshold: float = 0.75):
        self.window_size = window_size
        self.cooldown_threshold = cooldown_threshold
        
        # Circular buffers for metrics
        self.cpu_temp_buffer = np.zeros(window_size)
        self.cpu_usage_buffer = np.zeros(window_size)
        self.memory_buffer = np.zeros(window_size)
        self.thermal_load_buffer = np.zeros(window_size)
        self.buffer_index = 0
        
        # Cooldown state
        self.cooldown_active = False
        self.cooldown_start_time = None
        self.emergency_stops = 0
        
        # Thermal coefficients (tunable)
        self.temp_weight = 0.4
        self.cpu_weight = 0.3
        self.memory_weight = 0.2
        self.io_weight = 0.1
        
        # Performance tracking
        self.thermal_events = []
        self.last_metrics = None
        
        logger.info(f"ZBEThermalCooldown initialized: window={window_size}, threshold={cooldown_threshold}")

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk I/O (if available)
            try:
                disk_io = psutil.disk_io_counters()
                disk_usage = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)  # MB
            except:
                disk_usage = 0.0
            
            # Network I/O (if available)
            try:
                net_io = psutil.net_io_counters()
                network_usage = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB
            except:
                network_usage = 0.0
            
            # CPU temperature (if available - Linux/some systems)
            cpu_temp = self._get_cpu_temperature()
            
            return {
                'cpu_temp': cpu_temp,
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_io': disk_usage,
                'network_io': network_usage
            }
            
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {
                'cpu_temp': 45.0,    # Default safe values
                'cpu_usage': 25.0,
                'memory_usage': 30.0,
                'disk_io': 0.0,
                'network_io': 0.0
            }

    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (platform dependent)"""
        try:
            # Try different methods for different platforms
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for name, entries in temps.items():
                        if entries:
                            return entries[0].current
            
            # Fallback to estimated temperature based on CPU usage
            if hasattr(self, 'cpu_usage_buffer') and self.buffer_index > 0:
                recent_cpu = np.mean(self.cpu_usage_buffer[-5:])
                estimated_temp = 30.0 + (recent_cpu * 0.5)  # Rough estimation
                return min(estimated_temp, 90.0)
            
            return 45.0  # Safe default
            
        except:
            return 45.0

    def calculate_thermal_load(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite thermal load from system metrics
        Returns value between 0.0 and 1.0
        """
        # Normalize each metric to 0-1 scale
        temp_normalized = min(metrics['cpu_temp'] / 80.0, 1.0)  # 80°C = 100%
        cpu_normalized = metrics['cpu_usage'] / 100.0
        memory_normalized = metrics['memory_usage'] / 100.0
        
        # I/O normalization (adaptive based on historical data)
        if self.buffer_index > 10:
            max_disk = np.max(self.thermal_load_buffer[:self.buffer_index]) * 1000  # MB scale
            max_network = np.max(self.thermal_load_buffer[:self.buffer_index]) * 1000
            disk_normalized = min(metrics['disk_io'] / max(max_disk, 100), 1.0)
            network_normalized = min(metrics['network_io'] / max(max_network, 100), 1.0)
        else:
            disk_normalized = min(metrics['disk_io'] / 1000.0, 1.0)  # 1GB/s max
            network_normalized = min(metrics['network_io'] / 1000.0, 1.0)
        
        io_combined = (disk_normalized + network_normalized) / 2.0
        
        # Weighted thermal load calculation
        thermal_load = (
            temp_normalized * self.temp_weight +
            cpu_normalized * self.cpu_weight +
            memory_normalized * self.memory_weight +
            io_combined * self.io_weight
        )
        
        return min(thermal_load, 1.0)

    def determine_thermal_state(self, thermal_load: float) -> ThermalState:
        """Determine thermal state from load"""
        if thermal_load >= 0.95:
            return ThermalState.EMERGENCY
        elif thermal_load >= 0.85:
            return ThermalState.CRITICAL
        elif thermal_load >= 0.70:
            return ThermalState.HOT
        elif thermal_load >= 0.50:
            return ThermalState.WARM
        elif thermal_load >= 0.30:
            return ThermalState.COOL
        else:
            return ThermalState.COLD

    def should_activate_cooldown(self, thermal_load: float, state: ThermalState) -> bool:
        """Determine if cooldown should be activated"""
        # Emergency conditions
        if state in [ThermalState.EMERGENCY, ThermalState.CRITICAL]:
            return True
        
        # Threshold-based activation
        if thermal_load >= self.cooldown_threshold:
            return True
        
        # Trending analysis - activate if trending up rapidly
        if self.buffer_index >= 5:
            recent_loads = self.thermal_load_buffer[max(0, self.buffer_index-5):self.buffer_index]
            if len(recent_loads) >= 3:
                trend = np.polyfit(range(len(recent_loads)), recent_loads, 1)[0]
                if trend > 0.05:  # Rapid increase
                    return True
        
        return False

    def calculate_cooldown_duration(self, thermal_load: float, state: ThermalState) -> timedelta:
        """Calculate required cooldown duration"""
        base_cooldown = timedelta(seconds=30)
        
        if state == ThermalState.EMERGENCY:
            return timedelta(minutes=10)
        elif state == ThermalState.CRITICAL:
            return timedelta(minutes=5)
        elif state == ThermalState.HOT:
            return timedelta(minutes=2)
        else:
            # Proportional to thermal load
            multiplier = max(1.0, thermal_load * 3.0)
            return timedelta(seconds=int(base_cooldown.total_seconds() * multiplier))

    def update(self) -> ThermalMetrics:
        """
        Update thermal monitoring and return current metrics
        """
        timestamp = datetime.now()
        
        # Get system metrics
        metrics = self.get_system_metrics()
        
        # Calculate thermal load
        thermal_load = self.calculate_thermal_load(metrics)
        
        # Update buffers
        idx = self.buffer_index % self.window_size
        self.cpu_temp_buffer[idx] = metrics['cpu_temp']
        self.cpu_usage_buffer[idx] = metrics['cpu_usage']
        self.memory_buffer[idx] = metrics['memory_usage']
        self.thermal_load_buffer[idx] = thermal_load
        self.buffer_index += 1
        
        # Determine thermal state
        thermal_state = self.determine_thermal_state(thermal_load)
        
        # Check cooldown conditions
        should_cooldown = self.should_activate_cooldown(thermal_load, thermal_state)
        
        # Handle cooldown state transitions
        if should_cooldown and not self.cooldown_active:
            # Start cooldown
            self.cooldown_active = True
            self.cooldown_start_time = timestamp
            cooldown_duration = self.calculate_cooldown_duration(thermal_load, thermal_state)
            
            self.thermal_events.append({
                'timestamp': timestamp,
                'event': 'COOLDOWN_START',
                'thermal_load': thermal_load,
                'state': thermal_state.value,
                'duration': cooldown_duration
            })
            
            logger.warning(f"🌡️ THERMAL COOLDOWN ACTIVATED: {thermal_state.value} "
                          f"(Load: {thermal_load:.3f}, Duration: {cooldown_duration})")
            
            if thermal_state == ThermalState.EMERGENCY:
                self.emergency_stops += 1
                logger.critical(f"🚨 EMERGENCY THERMAL STOP #{self.emergency_stops}")
        
        elif self.cooldown_active:
            # Check if cooldown should end
            if self.cooldown_start_time:
                required_duration = self.calculate_cooldown_duration(thermal_load, thermal_state)
                elapsed = timestamp - self.cooldown_start_time
                
                if elapsed >= required_duration and thermal_load < (self.cooldown_threshold * 0.8):
                    # End cooldown
                    self.cooldown_active = False
                    self.cooldown_start_time = None
                    
                    self.thermal_events.append({
                        'timestamp': timestamp,
                        'event': 'COOLDOWN_END',
                        'thermal_load': thermal_load,
                        'state': thermal_state.value,
                        'duration': elapsed
                    })
                    
                    logger.info(f"❄️ THERMAL COOLDOWN ENDED: {thermal_state.value} "
                               f"(Load: {thermal_load:.3f}, Duration: {elapsed})")
        
        # Calculate time to cool
        time_to_cool = None
        if self.cooldown_active and self.cooldown_start_time:
            required_duration = self.calculate_cooldown_duration(thermal_load, thermal_state)
            elapsed = timestamp - self.cooldown_start_time
            time_to_cool = max(timedelta(0), required_duration - elapsed)
        
        # Create metrics object
        thermal_metrics = ThermalMetrics(
            timestamp=timestamp,
            cpu_temp=metrics['cpu_temp'],
            cpu_usage=metrics['cpu_usage'],
            memory_usage=metrics['memory_usage'],
            disk_io=metrics['disk_io'],
            network_io=metrics['network_io'],
            thermal_load=thermal_load,
            state=thermal_state,
            cooldown_active=self.cooldown_active,
            time_to_cool=time_to_cool
        )
        
        self.last_metrics = thermal_metrics
        return thermal_metrics

    def get_thermal_statistics(self) -> Dict:
        """Get thermal system statistics"""
        if self.buffer_index == 0:
            return {}
        
        filled_size = min(self.buffer_index, self.window_size)
        
        return {
            'avg_thermal_load': np.mean(self.thermal_load_buffer[:filled_size]),
            'max_thermal_load': np.max(self.thermal_load_buffer[:filled_size]),
            'avg_cpu_temp': np.mean(self.cpu_temp_buffer[:filled_size]),
            'max_cpu_temp': np.max(self.cpu_temp_buffer[:filled_size]),
            'avg_cpu_usage': np.mean(self.cpu_usage_buffer[:filled_size]),
            'avg_memory_usage': np.mean(self.memory_buffer[:filled_size]),
            'cooldown_active': self.cooldown_active,
            'emergency_stops': self.emergency_stops,
            'thermal_events_count': len(self.thermal_events),
            'buffer_fill': filled_size / self.window_size
        }

    def reset_emergency_counter(self):
        """Reset emergency stop counter (admin function)"""
        old_count = self.emergency_stops
        self.emergency_stops = 0
        logger.info(f"Emergency stop counter reset: {old_count} -> 0")

    def is_safe_to_trade(self) -> bool:
        """Check if system is thermally safe for trading operations"""
        if not self.last_metrics:
            return True  # No data yet, assume safe
        
        return (
            not self.cooldown_active and 
            self.last_metrics.state not in [ThermalState.CRITICAL, ThermalState.EMERGENCY] and
            self.last_metrics.thermal_load < self.cooldown_threshold
        ) 