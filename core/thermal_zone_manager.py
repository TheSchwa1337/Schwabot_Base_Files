"""
Thermal Zone Manager
===================

Manages thermal states and zone-based drift for optimal system performance.
Integrates with ProfitTrajectoryCoprocessor to make intelligent thermal-aware
processing decisions while maintaining profitability.
"""

import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import logging
import threading
import time
from enum import Enum
from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor, ProfitZoneState

# Setup logging
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
    """Container for thermal state information"""
    cpu_temp: float
    gpu_temp: float
    zone: ThermalZone
    load_cpu: float
    load_gpu: float
    memory_usage: float
    timestamp: datetime
    drift_coefficient: float
    processing_recommendation: Dict[str, float]

class ThermalZoneManager:
    """
    Manages thermal zones and provides thermal-aware processing recommendations.
    
    This manager implements the thermal drift compensation logic that modulates
    system behavior based on temperature, profit trajectory, and processing load.
    """
    
    def __init__(self, profit_coprocessor: Optional[ProfitTrajectoryCoprocessor] = None):
        """
        Initialize thermal zone manager
        
        Args:
            profit_coprocessor: Optional profit trajectory coprocessor for integration
        """
        self.profit_coprocessor = profit_coprocessor
        self.thermal_history: List[ThermalState] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
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
        self.budget_reset_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        
        # Thermal compensation parameters
        self.nominal_temp = 70.0  # T₀ in thermal drift formula
        self.profit_heat_bias = 0.5  # α in thermal drift formula
        
        # Current state
        self.current_state: Optional[ThermalState] = None
        
        # Spike management
        self.burst_history: List[Tuple[datetime, float]] = []  # (timestamp, duration)
        self.max_burst_duration = 300  # 5 minutes max burst
        self.cooldown_ratio = 2.0  # Cooldown = 2x burst duration
        
    def start_monitoring(self, interval: float = 10.0) -> None:
        """
        Start thermal monitoring in background thread
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            logger.warning("Thermal monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started thermal monitoring (interval: {interval}s)")
        
    def stop_monitoring(self) -> None:
        """Stop thermal monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped thermal monitoring")
        
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop (runs in separate thread)"""
        while self.monitoring_active:
            try:
                self.update_thermal_state()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in thermal monitoring loop: {e}")
                time.sleep(interval)
                
    def update_thermal_state(self) -> ThermalState:
        """Update current thermal state and return it"""
        with self._lock:
            # Get temperature readings
            cpu_temp = self._get_cpu_temperature()
            gpu_temp = self._get_gpu_temperature()
            
            # Get load information
            cpu_load = psutil.cpu_percent(interval=1)
            gpu_load = self._get_gpu_load()
            memory_usage = psutil.virtual_memory().percent
            
            # Determine thermal zone
            max_temp = max(cpu_temp, gpu_temp)
            zone = self._classify_thermal_zone(max_temp)
            
            # Calculate drift coefficient
            drift_coeff = self._calculate_thermal_drift_coefficient(max_temp)
            
            # Get processing recommendations
            processing_rec = self._calculate_processing_recommendation(zone, drift_coeff)
            
            # Create thermal state
            state = ThermalState(
                cpu_temp=cpu_temp,
                gpu_temp=gpu_temp,
                zone=zone,
                load_cpu=cpu_load,
                load_gpu=gpu_load,
                memory_usage=memory_usage,
                timestamp=datetime.now(timezone.utc),
                drift_coefficient=drift_coeff,
                processing_recommendation=processing_rec
            )
            
            # Update current state and history
            self.current_state = state
            self.thermal_history.append(state)
            
            # Keep only last 1000 readings
            if len(self.thermal_history) > 1000:
                self.thermal_history = self.thermal_history[-1000:]
                
            return state
            
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (with fallback for different systems)"""
        try:
            # Try psutil first (Linux)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Look for common CPU temperature sensors
                    for name, entries in temps.items():
                        if any(sensor in name.lower() for sensor in ['cpu', 'core', 'processor']):
                            if entries:
                                return float(entries[0].current)
                                
            # Fallback: estimate from CPU load (rough approximation)
            cpu_load = psutil.cpu_percent(interval=0.1)
            base_temp = 45.0  # Base temperature
            load_temp = cpu_load * 0.5  # Rough scaling
            return base_temp + load_temp
            
        except Exception as e:
            logger.debug(f"Could not get CPU temperature: {e}")
            return 65.0  # Default safe temperature
            
    def _get_gpu_temperature(self) -> float:
        """Get GPU temperature"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return float(gpus[0].temperature)
            else:
                # No GPU detected, return CPU-based estimate
                return self._get_cpu_temperature() - 5.0
        except Exception as e:
            logger.debug(f"Could not get GPU temperature: {e}")
            return 60.0  # Default safe temperature
            
    def _get_gpu_load(self) -> float:
        """Get GPU load percentage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return float(gpus[0].load * 100)
            else:
                return 0.0
        except Exception as e:
            logger.debug(f"Could not get GPU load: {e}")
            return 0.0
            
    def _classify_thermal_zone(self, temperature: float) -> ThermalZone:
        """Classify temperature into thermal zone"""
        for zone, (min_temp, max_temp) in self.zone_thresholds.items():
            if min_temp <= temperature < max_temp:
                return zone
        return ThermalZone.CRITICAL  # Default for extreme temperatures
        
    def _calculate_thermal_drift_coefficient(self, temperature: float) -> float:
        """
        Calculate thermal drift coefficient using the sigmoid formula:
        D_thermal = 1 / (1 + e^(-((T - T₀) - α * P_avg)))
        """
        # Get average profit from profit coprocessor if available
        avg_profit = 0.0
        if self.profit_coprocessor and self.profit_coprocessor.last_vector:
            avg_profit = self.profit_coprocessor.smoothed_profit
            
        # Apply thermal drift formula
        exponent = -((temperature - self.nominal_temp) - self.profit_heat_bias * avg_profit)
        drift_coefficient = 1.0 / (1.0 + np.exp(exponent))
        
        # Clamp to reasonable range
        return np.clip(drift_coefficient, 0.3, 1.5)
        
    def _calculate_processing_recommendation(self, zone: ThermalZone, 
                                          drift_coeff: float) -> Dict[str, float]:
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
        """Check if burst processing is allowed based on budget and cooldown"""
        now = datetime.now(timezone.utc)
        
        # Reset daily budget if new day
        if now.date() > self.budget_reset_time.date():
            self.budget_used_today = 0.0
            self.budget_reset_time = now.replace(hour=0, minute=0, second=0)
            
        # Check if daily budget available
        remaining_budget = self.daily_budget_hours - self.budget_used_today
        if remaining_budget <= 0:
            return False
            
        # Check cooldown from recent bursts
        cutoff_time = now - timedelta(minutes=30)  # Check last 30 minutes
        recent_bursts = [burst for burst in self.burst_history if burst[0] > cutoff_time]
        
        if recent_bursts:
            total_burst_time = sum(burst[1] for burst in recent_bursts)
            required_cooldown = total_burst_time * self.cooldown_ratio
            
            # Check if enough cooldown time has passed
            last_burst_end = max(burst[0] for burst in recent_bursts)
            time_since_last = (now - last_burst_end).total_seconds()
            
            if time_since_last < required_cooldown:
                return False
                
        return True
        
    def start_burst(self) -> bool:
        """
        Start a processing burst if conditions allow
        
        Returns:
            True if burst was started successfully
        """
        if not self.current_state:
            return False
            
        # Check thermal conditions
        if self.current_state.zone in {ThermalZone.HOT, ThermalZone.CRITICAL}:
            logger.warning("Burst denied: thermal zone too hot")
            return False
            
        # Check if burst is allowed
        if not self._can_burst():
            logger.warning("Burst denied: budget or cooldown restrictions")
            return False
            
        # Record burst start
        burst_start = datetime.now(timezone.utc)
        logger.info(f"Starting processing burst at {burst_start}")
        
        return True
        
    def end_burst(self, duration_seconds: float) -> None:
        """
        End a processing burst and update budget
        
        Args:
            duration_seconds: Duration of the completed burst
        """
        now = datetime.now(timezone.utc)
        duration_hours = duration_seconds / 3600.0
        
        # Update budget
        self.budget_used_today += duration_hours
        
        # Record in burst history
        self.burst_history.append((now, duration_seconds))
        
        # Keep only recent burst history
        cutoff_time = now - timedelta(hours=12)
        self.burst_history = [b for b in self.burst_history if b[0] > cutoff_time]
        
        logger.info(f"Burst ended. Duration: {duration_seconds:.1f}s, "
                   f"Budget used today: {self.budget_used_today:.2f}h")
        
    def should_reduce_gpu_load(self) -> bool:
        """Check if GPU load should be reduced due to thermal conditions"""
        if not self.current_state:
            return False
            
        return self.current_state.zone in {ThermalZone.HOT, ThermalZone.CRITICAL}
        
    def get_current_state(self) -> Optional[ThermalState]:
        """Get current thermal state"""
        return self.current_state
        
    def get_thermal_history(self, limit: Optional[int] = None) -> List[ThermalState]:
        """Get thermal state history"""
        with self._lock:
            if limit:
                return self.thermal_history[-limit:]
            return self.thermal_history.copy()
            
    def get_statistics(self) -> Dict[str, Union[float, int, str]]:
        """Get thermal zone manager statistics"""
        if not self.current_state:
            return {"status": "no_data"}
            
        with self._lock:
            # Calculate temperature statistics
            recent_temps = [state.cpu_temp for state in self.thermal_history[-100:]]
            avg_temp = np.mean(recent_temps) if recent_temps else 0.0
            max_temp = np.max(recent_temps) if recent_temps else 0.0
            
            # Calculate zone distribution
            zone_counts = {}
            for state in self.thermal_history[-100:]:
                zone = state.zone.value
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
                
            return {
                "current_cpu_temp": self.current_state.cpu_temp,
                "current_gpu_temp": self.current_state.gpu_temp,
                "current_zone": self.current_state.zone.value,
                "current_cpu_load": self.current_state.load_cpu,
                "current_gpu_load": self.current_state.load_gpu,
                "avg_temp_recent": avg_temp,
                "max_temp_recent": max_temp,
                "drift_coefficient": self.current_state.drift_coefficient,
                "budget_used_today_hours": self.budget_used_today,
                "budget_remaining_hours": max(0, self.daily_budget_hours - self.budget_used_today),
                "burst_allowed": self._can_burst(),
                "zone_distribution": zone_counts,
                "processing_recommendation": self.current_state.processing_recommendation
            }
            
    def reset_daily_budget(self) -> None:
        """Reset the daily processing budget"""
        self.budget_used_today = 0.0
        self.budget_reset_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        logger.info("Daily processing budget reset")

# Example usage and testing
if __name__ == "__main__":
    from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor
    
    # Create profit coprocessor
    profit_coprocessor = ProfitTrajectoryCoprocessor()
    
    # Add some sample profit data
    for i in range(20):
        profit = 1000 + i * 10 + np.random.normal(0, 5)
        profit_coprocessor.update(profit)
    
    # Create thermal zone manager
    thermal_manager = ThermalZoneManager(profit_coprocessor)
    
    # Update thermal state
    state = thermal_manager.update_thermal_state()
    
    print("Thermal Zone Manager Test:")
    print(f"  CPU Temperature: {state.cpu_temp:.1f}°C")
    print(f"  GPU Temperature: {state.gpu_temp:.1f}°C")
    print(f"  Thermal Zone: {state.zone.value}")
    print(f"  CPU Load: {state.load_cpu:.1f}%")
    print(f"  GPU Load: {state.load_gpu:.1f}%")
    print(f"  Drift Coefficient: {state.drift_coefficient:.3f}")
    
    print("\nProcessing Recommendations:")
    for key, value in state.processing_recommendation.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1%}" if value < 1 else f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test burst functionality
    print(f"\nBurst Testing:")
    print(f"  Can burst: {thermal_manager._can_burst()}")
    
    if thermal_manager.start_burst():
        print("  Burst started successfully")
        time.sleep(2)  # Simulate 2-second burst
        thermal_manager.end_burst(2.0)
        print("  Burst ended")
    
    # Get statistics
    stats = thermal_manager.get_statistics()
    print(f"\nThermal Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}") 