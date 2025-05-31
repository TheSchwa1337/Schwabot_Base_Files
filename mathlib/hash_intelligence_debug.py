"""
Hash Intelligence Debug System for Schwabot v0.3
Implements safe memory swapping and CPU strain monitoring
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
CPU_STRAIN_LEVELS = {
    'SAFE': {
        'max_usage': 0.60,    # 60% max CPU usage
        'max_temp': 65,       # 65°C max temp
        'max_memory': 0.75,   # 75% max memory usage
        'hash_rate': 'normal' # Normal hash processing
    },
    'WARNING': {
        'max_usage': 0.75,    # 75% max CPU usage
        'max_temp': 70,       # 70°C max temp
        'max_memory': 0.85,   # 85% max memory usage
        'hash_rate': 'reduced' # Reduced hash processing
    },
    'CRITICAL': {
        'max_usage': 0.90,    # 90% max CPU usage
        'max_temp': 75,       # 75°C max temp
        'max_memory': 0.95,   # 95% max memory usage
        'hash_rate': 'minimal' # Minimal hash processing
    }
}

MEMORY_SWAP_CONFIG = {
    'SAFE': {
        'swap_threshold': 0.60,  # Start swapping at 60% memory
        'swap_size': 'full',     # Full memory swap allowed
        'echo_sync': True        # Enable echo synchronization
    },
    'WARNING': {
        'swap_threshold': 0.75,  # Start swapping at 75% memory
        'swap_size': 'partial',  # Partial memory swap
        'echo_sync': True        # Enable echo synchronization
    },
    'CRITICAL': {
        'swap_threshold': 0.85,  # Start swapping at 85% memory
        'swap_size': 'minimal',  # Minimal memory swap
        'echo_sync': False       # Disable echo synchronization
    }
}

@dataclass
class HashDebugState:
    """State of hash processing and system resources"""
    timestamp: str
    hash_value: str
    cpu_usage: float
    cpu_temp: float
    memory_usage: float
    strain_level: str
    swap_mode: str
    hash_rate: str
    echo_sync: bool
    debug_code: str = ""

class HashIntelligenceDebug:
    """
    Implements hash processing intelligence and system resource monitoring
    """
    
    def __init__(self, config_path: str = "hash_debug_config.yaml"):
        """Initialize hash intelligence debug system"""
        self.config_path = config_path
        self.debug_history: List[HashDebugState] = []
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {
                "cpu_strain": CPU_STRAIN_LEVELS,
                "memory_swap": MEMORY_SWAP_CONFIG
            }
    
    def get_system_metrics(self) -> Tuple[float, float, float]:
        """
        Get current system metrics
        
        Returns:
            Tuple of (cpu_usage, cpu_temp, memory_usage)
        """
        # Get CPU usage
        cpu_usage = psutil.cpu_percent() / 100.0
        
        # Get CPU temperature
        try:
            cpu_temp = psutil.sensors_temperatures()['coretemp'][0].current
        except:
            cpu_temp = 0.0
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100.0
        
        return cpu_usage, cpu_temp, memory_usage
    
    def determine_strain_level(self,
                             cpu_usage: float,
                             cpu_temp: float,
                             memory_usage: float) -> str:
        """
        Determine current system strain level
        
        Args:
            cpu_usage: CPU usage (0-1)
            cpu_temp: CPU temperature (°C)
            memory_usage: Memory usage (0-1)
            
        Returns:
            Strain level string
        """
        if (cpu_usage >= CPU_STRAIN_LEVELS['CRITICAL']['max_usage'] or
            cpu_temp >= CPU_STRAIN_LEVELS['CRITICAL']['max_temp'] or
            memory_usage >= CPU_STRAIN_LEVELS['CRITICAL']['max_memory']):
            return 'CRITICAL'
        
        if (cpu_usage >= CPU_STRAIN_LEVELS['WARNING']['max_usage'] or
            cpu_temp >= CPU_STRAIN_LEVELS['WARNING']['max_temp'] or
            memory_usage >= CPU_STRAIN_LEVELS['WARNING']['max_memory']):
            return 'WARNING'
        
        return 'SAFE'
    
    def get_memory_swap_config(self, strain_level: str) -> Dict:
        """
        Get memory swap configuration for strain level
        
        Args:
            strain_level: Current strain level
            
        Returns:
            Memory swap configuration
        """
        return MEMORY_SWAP_CONFIG[strain_level]
    
    def generate_debug_code(self,
                          strain_level: str,
                          hash_value: str,
                          cpu_usage: float,
                          memory_usage: float) -> str:
        """
        Generate debug code for current state
        
        Args:
            strain_level: Current strain level
            hash_value: Current hash value
            cpu_usage: CPU usage
            memory_usage: Memory usage
            
        Returns:
            Debug code string
        """
        # Create debug code based on state
        code = f"{strain_level[0]}"  # First letter of strain level
        
        # Add hash prefix
        code += hash_value[:4]
        
        # Add resource indicators
        if cpu_usage > 0.75:
            code += "C"  # CPU warning
        if memory_usage > 0.75:
            code += "M"  # Memory warning
        
        return code
    
    def process_hash(self, hash_value: str) -> HashDebugState:
        """
        Process a hash value with system monitoring
        
        Args:
            hash_value: Hash value to process
            
        Returns:
            HashDebugState object
        """
        # Get system metrics
        cpu_usage, cpu_temp, memory_usage = self.get_system_metrics()
        
        # Determine strain level
        strain_level = self.determine_strain_level(
            cpu_usage, cpu_temp, memory_usage)
        
        # Get memory swap configuration
        swap_config = self.get_memory_swap_config(strain_level)
        
        # Generate debug code
        debug_code = self.generate_debug_code(
            strain_level, hash_value, cpu_usage, memory_usage)
        
        # Create debug state
        state = HashDebugState(
            timestamp=datetime.now().isoformat(),
            hash_value=hash_value,
            cpu_usage=cpu_usage,
            cpu_temp=cpu_temp,
            memory_usage=memory_usage,
            strain_level=strain_level,
            swap_mode=swap_config['swap_size'],
            hash_rate=CPU_STRAIN_LEVELS[strain_level]['hash_rate'],
            echo_sync=swap_config['echo_sync'],
            debug_code=debug_code
        )
        
        # Add to history
        self.debug_history.append(state)
        
        return state
    
    def get_debug_history(self) -> List[HashDebugState]:
        """
        Get debug history
        
        Returns:
            List of HashDebugState objects
        """
        return self.debug_history 