"""
GPU Flash Engine for Schwabot v0.3
Implements safe GPU activation and memory swap logic with ZPE field validation
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import yaml
from pathlib import Path
import psutil
import GPUtil
from hashlib import sha256

# Constants
ZPE_ZONES = {
    'ZPE_SAFE': {
        'hash_range': (0.000, 0.200),
        'z_score_band': (-1.5, 1.5),
        'flash_permission': True,
        'memory_swap': 'full'
    },
    'ZPE_WARM': {
        'hash_range': (0.200, 0.400),
        'z_score_band': (-2.0, 2.0),
        'flash_permission': True,
        'memory_swap': 'partial'
    },
    'ZPE_UNSAFE': {
        'hash_range': (0.400, 1.000),
        'z_score_band': (2.0, float('inf')),
        'flash_permission': False,
        'memory_swap': 'cpu_only'
    }
}

GPU_SAFETY = {
    'max_utilization': 0.50,  # 50% max GPU load
    'max_temperature': 65,    # 65°C max temp
    'z_score_threshold': 2.0, # Max allowed z-score
    'hash_threshold': 0.5     # Max allowed hash value
}

@dataclass
class FlashState:
    """State of GPU flash activation"""
    timestamp: str
    hash_value: float
    z_score: float
    gpu_utilization: float
    gpu_temperature: float
    zpe_zone: str
    flash_permitted: bool
    memory_swap_mode: str
    cpu_time_us: float
    echo_sync: bool = False

class GPUFlashEngine:
    """
    Implements GPU flash activation and memory swap logic
    """
    
    def __init__(self, config_path: str = "permissions.yaml"):
        """Initialize GPU flash engine"""
        self.config_path = config_path
        self.flash_history: List[FlashState] = []
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {
                "gpu_flash": GPU_SAFETY,
                "zpe_zone_rules": ZPE_ZONES
            }
    
    def get_gpu_metrics(self) -> Tuple[float, float]:
        """
        Get current GPU utilization and temperature
        
        Returns:
            Tuple of (utilization, temperature)
        """
        try:
            gpu = GPUtil.getGPUs()[0]  # Get first GPU
            return gpu.load, gpu.temperature
        except:
            return 0.0, 0.0  # Default to safe values if can't read GPU
    
    def classify_zpe_zone(self, hash_value: float, z_score: float) -> str:
        """
        Classify current state into ZPE zone
        
        Args:
            hash_value: Normalized hash value
            z_score: Current z-score
            
        Returns:
            ZPE zone name
        """
        for zone, rules in ZPE_ZONES.items():
            h_min, h_max = rules['hash_range']
            z_min, z_max = rules['z_score_band']
            
            if (h_min <= hash_value <= h_max and 
                z_min <= z_score <= z_max):
                return zone
        
        return 'ZPE_UNSAFE'
    
    def check_flash_permission(self,
                             hash_value: float,
                             z_score: float,
                             gpu_util: float,
                             gpu_temp: float) -> bool:
        """
        Check if flash activation is permitted
        
        Args:
            hash_value: Normalized hash value
            z_score: Current z-score
            gpu_util: GPU utilization
            gpu_temp: GPU temperature
            
        Returns:
            True if flash is permitted
        """
        # Check GPU safety limits
        if (gpu_util >= GPU_SAFETY['max_utilization'] or
            gpu_temp >= GPU_SAFETY['max_temperature'] or
            z_score >= GPU_SAFETY['z_score_threshold'] or
            hash_value >= GPU_SAFETY['hash_threshold']):
            return False
        
        # Check ZPE zone
        zone = self.classify_zpe_zone(hash_value, z_score)
        return ZPE_ZONES[zone]['flash_permission']
    
    def get_memory_swap_mode(self, zone: str) -> str:
        """
        Get memory swap mode for ZPE zone
        
        Args:
            zone: ZPE zone name
            
        Returns:
            Memory swap mode
        """
        return ZPE_ZONES[zone]['memory_swap']
    
    def process_tick(self,
                    hash_value: float,
                    z_score: float,
                    cpu_time_us: float) -> FlashState:
        """
        Process a single tick through the flash engine
        
        Args:
            hash_value: Normalized hash value
            z_score: Current z-score
            cpu_time_us: CPU time in microseconds
            
        Returns:
            FlashState object
        """
        # Get GPU metrics
        gpu_util, gpu_temp = self.get_gpu_metrics()
        
        # Classify ZPE zone
        zone = self.classify_zpe_zone(hash_value, z_score)
        
        # Check flash permission
        flash_permitted = self.check_flash_permission(
            hash_value, z_score, gpu_util, gpu_temp)
        
        # Get memory swap mode
        memory_swap = self.get_memory_swap_mode(zone)
        
        # Create flash state
        state = FlashState(
            timestamp=datetime.now().isoformat(),
            hash_value=hash_value,
            z_score=z_score,
            gpu_utilization=gpu_util,
            gpu_temperature=gpu_temp,
            zpe_zone=zone,
            flash_permitted=flash_permitted,
            memory_swap_mode=memory_swap,
            cpu_time_us=cpu_time_us
        )
        
        # Add to history
        self.flash_history.append(state)
        
        return state
    
    def get_flash_history(self) -> List[FlashState]:
        """
        Get flash history
        
        Returns:
            List of FlashState objects
        """
        return self.flash_history 