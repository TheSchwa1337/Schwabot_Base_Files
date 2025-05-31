"""
System Metrics and Drift Band Management
======================================

Handles system resource monitoring, drift band calculations, and ZPE zone management
for optimizing resource allocation across different bit modes.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import psutil
import GPUtil
import logging
from pathlib import Path

# Constants for drift band thresholds
DRIFT_THRESHOLDS = {
    'SAFE': 0.2,    # 20% drift
    'WARM': 0.4,    # 40% drift
    'UNSAFE': 0.6   # 60% drift
}

# Resource allocation profiles for different bit modes
RESOURCE_PROFILES = {
    4: {  # 4-bit mode
        'gpu_utilization': 0.3,  # 30% GPU
        'cpu_utilization': 0.4,  # 40% CPU
        'memory_limit': 0.5,     # 50% memory
        'batch_size': 1000
    },
    8: {  # 8-bit mode
        'gpu_utilization': 0.5,  # 50% GPU
        'cpu_utilization': 0.6,  # 60% CPU
        'memory_limit': 0.7,     # 70% memory
        'batch_size': 500
    },
    42: {  # 42-bit mode
        'gpu_utilization': 0.7,  # 70% GPU
        'cpu_utilization': 0.8,  # 80% CPU
        'memory_limit': 0.8,     # 80% memory
        'batch_size': 100
    }
}

@dataclass
class SystemMetrics:
    """Current system resource metrics"""
    timestamp: str
    gpu_utilization: float
    gpu_temperature: float
    cpu_utilization: float
    cpu_temperature: float
    memory_usage: float
    zpe_zone: str
    drift_band: float
    bit_mode: int
    strategy: str

class SystemMonitor:
    """Monitors system resources and manages drift bands"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics_history: List[SystemMetrics] = []
        
        # Setup logging
        logging.basicConfig(
            filename=self.log_dir / "system_metrics.log",
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SystemMonitor')
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            # Get GPU metrics
            gpus = GPUtil.getGPUs()
            gpu = gpus[0] if gpus else None
            gpu_util = gpu.load * 100 if gpu else 0
            gpu_temp = gpu.temperature if gpu else 0
            
            # Get CPU metrics
            cpu_util = psutil.cpu_percent()
            cpu_temp = self._get_cpu_temperature()
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            return {
                'gpu_utilization': gpu_util,
                'gpu_temperature': gpu_temp,
                'cpu_utilization': cpu_util,
                'cpu_temperature': cpu_temp,
                'memory_usage': memory_usage
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {
                'gpu_utilization': 0,
                'gpu_temperature': 0,
                'cpu_utilization': 0,
                'cpu_temperature': 0,
                'memory_usage': 0
            }
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (platform dependent)"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return max(temp.current for temp in temps['coretemp'])
            return 0
        except:
            return 0
    
    def calculate_drift_band(self, 
                           centroid_distance: float, 
                           ideal_center: float = 7.5) -> float:
        """Calculate drift band from centroid distance"""
        return abs(centroid_distance - ideal_center) / ideal_center
    
    def determine_zpe_zone(self, drift_band: float) -> str:
        """Determine ZPE zone based on drift band"""
        if drift_band >= DRIFT_THRESHOLDS['UNSAFE']:
            return 'UNSAFE'
        elif drift_band >= DRIFT_THRESHOLDS['WARM']:
            return 'WARM'
        return 'SAFE'
    
    def get_resource_allocation(self, 
                              bit_mode: int, 
                              zpe_zone: str) -> Dict:
        """Get resource allocation profile based on bit mode and ZPE zone"""
        base_profile = RESOURCE_PROFILES.get(bit_mode, RESOURCE_PROFILES[4])
        
        # Adjust based on ZPE zone
        if zpe_zone == 'UNSAFE':
            # Reduce resource usage in unsafe conditions
            return {
                'gpu_utilization': base_profile['gpu_utilization'] * 0.5,
                'cpu_utilization': base_profile['cpu_utilization'] * 0.5,
                'memory_limit': base_profile['memory_limit'] * 0.5,
                'batch_size': base_profile['batch_size'] // 2
            }
        elif zpe_zone == 'WARM':
            # Slightly reduce resource usage
            return {
                'gpu_utilization': base_profile['gpu_utilization'] * 0.8,
                'cpu_utilization': base_profile['cpu_utilization'] * 0.8,
                'memory_limit': base_profile['memory_limit'] * 0.8,
                'batch_size': int(base_profile['batch_size'] * 0.8)
            }
        return base_profile
    
    def update_metrics(self, 
                      bit_mode: int, 
                      strategy: str, 
                      centroid_distance: float) -> SystemMetrics:
        """Update system metrics with current state"""
        # Get system metrics
        metrics = self.get_system_metrics()
        
        # Calculate drift band and ZPE zone
        drift_band = self.calculate_drift_band(centroid_distance)
        zpe_zone = self.determine_zpe_zone(drift_band)
        
        # Create metrics object
        current_metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            gpu_utilization=metrics['gpu_utilization'],
            gpu_temperature=metrics['gpu_temperature'],
            cpu_utilization=metrics['cpu_utilization'],
            cpu_temperature=metrics['cpu_temperature'],
            memory_usage=metrics['memory_usage'],
            zpe_zone=zpe_zone,
            drift_band=drift_band,
            bit_mode=bit_mode,
            strategy=strategy
        )
        
        # Add to history
        self.metrics_history.append(current_metrics)
        
        # Log metrics
        self.logger.info(
            f"Metrics updated - Mode: {bit_mode}, Strategy: {strategy}, "
            f"ZPE: {zpe_zone}, Drift: {drift_band:.2f}"
        )
        
        return current_metrics
    
    def get_optimization_suggestions(self, 
                                   current_metrics: SystemMetrics) -> Dict:
        """Get suggestions for resource optimization"""
        suggestions = {
            'resource_adjustments': {},
            'strategy_changes': [],
            'warnings': []
        }
        
        # Check resource usage against limits
        profile = self.get_resource_allocation(
            current_metrics.bit_mode, 
            current_metrics.zpe_zone
        )
        
        # GPU suggestions
        if current_metrics.gpu_utilization > profile['gpu_utilization'] * 100:
            suggestions['resource_adjustments']['gpu'] = 'reduce'
            suggestions['warnings'].append('High GPU utilization')
        
        # CPU suggestions
        if current_metrics.cpu_utilization > profile['cpu_utilization'] * 100:
            suggestions['resource_adjustments']['cpu'] = 'reduce'
            suggestions['warnings'].append('High CPU utilization')
        
        # Memory suggestions
        if current_metrics.memory_usage > profile['memory_limit'] * 100:
            suggestions['resource_adjustments']['memory'] = 'reduce'
            suggestions['warnings'].append('High memory usage')
        
        # Strategy suggestions based on drift band
        if current_metrics.drift_band > DRIFT_THRESHOLDS['UNSAFE']:
            suggestions['strategy_changes'].append('switch_to_stable')
        elif current_metrics.drift_band > DRIFT_THRESHOLDS['WARM']:
            suggestions['strategy_changes'].append('reduce_aggression')
        
        return suggestions 