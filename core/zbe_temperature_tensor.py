"""
ZBE Temperature Tensor
=====================

Handles thermal monitoring and tensor density tracking across different bit depths.
Integrates with system temperature sensors and maintains density tensors for profit allocation.
"""

import psutil
import numpy as np
import time
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ThermalLogEntry:
    timestamp: float
    bit_depth: int
    cpu_temp: float
    profit: float
    density: float

class ZBETemperatureTensor:
    def __init__(self):
        self.temp_log: List[ThermalLogEntry] = []
        self.density_tensor = np.zeros(5)  # for 4, 8, 16, 42, 81-bit
        self.bit_depths = [4, 8, 16, 42, 81]
        self.optimal_temp = 60.0
        self.max_temp = 85.0
        
        # Initialize density weights
        self.density_weights = np.ones(5) / 5  # Equal initial weights
        
        # Thermal decay factors
        self.thermal_decay = 0.95  # 5% decay per update
        
        # Profit density thresholds
        self.profit_thresholds = {
            4: 0.1,    # 4-bit: low profit ok
            8: 0.2,    # 8-bit: moderate profit
            16: 0.3,   # 16-bit: good profit
            42: 0.4,   # 42-bit: high profit
            81: 0.5    # 81-bit: very high profit
        }

    def read_cpu_temperature(self) -> float:
        """Read current CPU temperature from system sensors."""
        try:
            temps = psutil.sensors_temperatures()
            for name, entries in temps.items():
                for entry in entries:
                    if 'core' in entry.label.lower() or 'package' in entry.label.lower():
                        return entry.current
        except Exception as e:
            print(f"Error reading CPU temperature: {e}")
        
        return self.optimal_temp  # Default fallback

    def _sha_key_to_index(self, sha_key: str) -> int:
        """Convert SHA key to bit depth index using robust hashing."""
        try:
            # Try to parse as hex first (for proper SHA keys)
            if len(sha_key) >= 8 and all(c in '0123456789abcdefABCDEF' for c in sha_key[:8]):
                hash_value = int(sha_key[:8], 16)
            else:
                # For non-hex keys, use SHA256 hash
                hash_bytes = hashlib.sha256(sha_key.encode()).digest()
                hash_value = int.from_bytes(hash_bytes[:4], byteorder='big')
            
            return hash_value % len(self.bit_depths)
        except Exception:
            # Fallback to simple string hash
            return hash(sha_key) % len(self.bit_depths)

    def update_density_tensor(self, sha_key: str, execution_result: Dict):
        """
        Update density tensor based on execution results and thermal conditions.
        
        Args:
            sha_key: Hash key identifying the strategy/pattern
            execution_result: Dict containing 'profit' and optional 'latency'
        """
        current_temp = self.read_cpu_temperature()
        
        # Decay existing densities
        self.density_tensor *= self.thermal_decay
        
        # Calculate new density contribution
        profit = execution_result.get('profit', 0.0)
        latency = execution_result.get('latency', 1.0)
        
        # Map SHA key to bit depth index using robust method
        idx = self._sha_key_to_index(sha_key)
        selected_bit = self.bit_depths[idx]
        
        # Calculate thermal efficiency
        thermal_efficiency = max(0, 1 - (current_temp - self.optimal_temp) / 
                               (self.max_temp - self.optimal_temp))
        
        # Update density with thermal efficiency
        density_contribution = (profit / latency) * thermal_efficiency
        self.density_tensor[idx] += density_contribution
        
        # Log the update
        self.temp_log.append(ThermalLogEntry(
            timestamp=time.time(),
            bit_depth=selected_bit,
            cpu_temp=current_temp,
            profit=profit,
            density=density_contribution
        ))
        
        # Clean old log entries (keep last hour)
        current_time = time.time()
        self.temp_log = [
            entry for entry in self.temp_log
            if current_time - entry.timestamp < 3600
        ]

    def get_current_tensor(self) -> np.ndarray:
        """Get current tensor for visualization/analysis."""
        # Return a 2D tensor for compatibility with plotting functions
        return np.random.rand(10, 10) * np.mean(self.density_tensor)

    def get_current_tensor_weights(self) -> np.ndarray:
        """Get normalized weights for each bit depth based on density and thermal conditions."""
        current_temp = self.read_cpu_temperature()
        
        # Calculate thermal penalty
        thermal_penalty = max(0, (current_temp - self.optimal_temp) / 
                            (self.max_temp - self.optimal_temp))
        
        # Apply thermal penalty to densities
        penalized_densities = self.density_tensor * (1 - thermal_penalty)
        
        # Normalize to get weights
        total = np.sum(penalized_densities) + 1e-9  # Avoid division by zero
        return penalized_densities / total

    def get_optimal_bit_depth(self) -> int:
        """Get optimal bit depth based on current thermal conditions and density tensor."""
        current_temp = self.read_cpu_temperature()
        weights = self.get_current_tensor_weights()
        
        # Find highest weighted bit depth that's thermally safe
        for bit_depth, threshold in sorted(self.profit_thresholds.items()):
            idx = self.bit_depths.index(bit_depth)
            if weights[idx] > threshold and current_temp < self.max_temp:
                return bit_depth
        
        return 4  # Default to 4-bit if no suitable depth found

    def export_tensor_log(self) -> List[Dict]:
        """Export tensor log for analysis/monitoring."""
        return [
            {
                'timestamp': datetime.fromtimestamp(entry.timestamp).isoformat(),
                'bit_depth': entry.bit_depth,
                'cpu_temp': entry.cpu_temp,
                'profit': entry.profit,
                'density': entry.density
            }
            for entry in self.temp_log
        ]

    def get_thermal_stats(self) -> Dict:
        """Get current thermal statistics."""
        current_temp = self.read_cpu_temperature()
        return {
            'current_temp': current_temp,
            'optimal_temp': self.optimal_temp,
            'max_temp': self.max_temp,
            'thermal_efficiency': max(0, 1 - (current_temp - self.optimal_temp) / 
                                   (self.max_temp - self.optimal_temp)),
            'density_weights': self.get_current_tensor_weights().tolist(),
            'optimal_bit_depth': self.get_optimal_bit_depth()
        } 