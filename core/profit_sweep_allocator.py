"""
Profit Sweep Allocator
=====================

Coordinates thermal-aware profit allocation and sweep detection across different bit depths.
Integrates with ZBE temperature tensor, profit tensor store, and sweep signal log.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .zbe_temperature_tensor import ZBETemperatureTensor
from .profit_tensor import ProfitTensorStore
from .bitmap_engine import BitmapEngine

@dataclass
class SweepSignal:
    timestamp: float
    price: float
    volume: float
    direction: str  # 'buy' or 'sell'
    sha_key: str
    thermal_tag: Optional[float] = None
    profit_potential: Optional[float] = None

class ProfitSweepAllocator:
    def __init__(self):
        self.zbe_tensor = ZBETemperatureTensor()
        self.profit_store = ProfitTensorStore()
        self.bitmap_engine = BitmapEngine()
        self.sweep_log: List[SweepSignal] = []
        self.wall_pressure_map: Dict[str, Dict] = {}
        
        # Thermal thresholds for bit depth transitions
        self.thermal_thresholds = {
            4: 70.0,   # 4-bit: high temp allowed
            8: 65.0,   # 8-bit: moderate temp
            16: 60.0,  # 16-bit: careful temp
            42: 55.0,  # 42-bit: low temp
            81: 50.0   # 81-bit: very low temp
        }
        
        # Bit depth normalization factor
        self.bit_depth_factor = {
            4: 1,
            8: 2,
            16: 4,
            42: 8,
            81: 16
        }

        self.depth_memory: Dict[int, List[str]] = {d: [] for d in self.bit_depth_factor}

    def _normalize_tensor(self, tensor: np.ndarray, bit_depth: int) -> np.ndarray:
        max_val = 2 ** bit_depth - 1
        return np.clip(tensor / max_val, 0, 1)

    def check_tensor_drift(self, tensor: np.ndarray, threshold: float = 0.05) -> bool:
        if tensor.size == 0:
            return False
        avg = np.mean(tensor)
        drift = np.abs(tensor - avg)
        return np.any(drift > threshold)

    def _update_plot(self):
        sha_key = self.choose_next_sha_key() or "default"
        bit_depth = self.get_optimal_bit_depth(sha_key)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(range(...), range(...))  # depending on tensor shape
        
        try:
            # Normalize tensor values based on bit depth
            normalized_tensor = self._normalize_tensor(self.zbe_tensor.get_current_tensor(), bit_depth)
            
            # Plot surface
            ax.plot_surface(x[:,:,0], y[:,:,0], normalized_tensor[:,:,0], cmap='viridis', alpha=0.8)
        except ValueError as e:
            print(f"[WARN] Surface plot error: {e}")
        
        # Set stable z-limits
        ax.set_zlim(0, 1)

    def map_tensor_to_symbolic(self, tensor: np.ndarray) -> str:
        flat = tensor.flatten()
        return ''.join(chr(int(val * 95) + 32) for val in flat[:64])  # 64-char symbol

    def map_wall_pressure(self, sweep: SweepSignal) -> Dict:
        """Map pressure points in buy/sell walls based on sweep signals."""
        if sweep.sha_key not in self.wall_pressure_map:
            self.wall_pressure_map[sweep.sha_key] = {
                'buy': [],
                'sell': [],
                'last_update': datetime.now().timestamp()
            }
        
        wall_map = self.wall_pressure_map[sweep.sha_key]
        wall_map[sweep.direction].append({
            'price': sweep.price,
            'volume': sweep.volume,
            'timestamp': sweep.timestamp,
            'thermal': sweep.thermal_tag
        })
        
        # Clean old entries (older than 1 hour)
        current_time = datetime.now().timestamp()
        wall_map[sweep.direction] = [
            entry for entry in wall_map[sweep.direction]
            if current_time - entry['timestamp'] < 3600
        ]
        
        return wall_map

    def assign_profit_zones(self, sha_key: str) -> List[Dict]:
        """Assign profit zones based on wall pressure and thermal conditions."""
        if sha_key not in self.wall_pressure_map:
            return []
            
        wall_map = self.wall_pressure_map[sha_key]
        profit_zones = []
        
        # Analyze buy wall pressure
        for entry in wall_map['buy']:
            if entry['thermal'] and entry['thermal'] < self.thermal_thresholds[4]:
                profit_zones.append({
                    'type': 'buy',
                    'price': entry['price'],
                    'volume': entry['volume'],
                    'thermal': entry['thermal'],
                    'bit_depth': self._select_bit_depth(entry['thermal'])
                })
        
        # Analyze sell wall pressure
        for entry in wall_map['sell']:
            if entry['thermal'] and entry['thermal'] < self.thermal_thresholds[4]:
                profit_zones.append({
                    'type': 'sell',
                    'price': entry['price'],
                    'volume': entry['volume'],
                    'thermal': entry['thermal'],
                    'bit_depth': self._select_bit_depth(entry['thermal'])
                })
        
        return profit_zones

    def calculate_profit_per_thermal_unit(self, sha_key: str) -> float:
        """Calculate profit efficiency per thermal unit."""
        profit_zones = self.assign_profit_zones(sha_key)
        if not profit_zones:
            return 0.0
            
        total_profit = sum(zone['volume'] for zone in profit_zones)
        total_thermal = sum(zone['thermal'] for zone in profit_zones)
        
        if total_thermal == 0:
            return 0.0
            
        return total_profit / total_thermal

    def choose_next_sha_key(self) -> Optional[str]:
        """Choose next SHA key based on profit/thermal efficiency."""
        current_temp = self.zbe_tensor.read_cpu_temperature()
        available_keys = list(self.wall_pressure_map.keys())
        
        if not available_keys:
            return None
            
        # Calculate efficiency scores
        efficiency_scores = []
        for sha_key in available_keys:
            profit_per_thermal = self.calculate_profit_per_thermal_unit(sha_key)
            thermal_weight = 1.0 - (current_temp / 100.0)  # Higher temp = lower weight
            efficiency_scores.append(profit_per_thermal * thermal_weight)
        
        # Select best performing key
        best_idx = np.argmax(efficiency_scores)
        return available_keys[best_idx]

    def process_sweep(self, sweep: SweepSignal):
        """Process a new sweep signal and update internal state."""
        # Get current CPU temperature
        sweep.thermal_tag = self.zbe_tensor.read_cpu_temperature()
        
        # Map wall pressure
        self.map_wall_pressure(sweep)
        
        # Update ZBE tensor
        self.zbe_tensor.update_density_tensor(
            sweep.sha_key,
            {'profit': sweep.volume}  # Use volume as proxy for profit
        )
        
        # Store in sweep log
        self.sweep_log.append(sweep)
        
        # Clean old sweep log entries
        current_time = datetime.now().timestamp()
        self.sweep_log = [
            s for s in self.sweep_log
            if current_time - s.timestamp < 3600  # Keep last hour
        ]

    def get_optimal_bit_depth(self, sha_key: str) -> int:
        """Get optimal bit depth for current conditions."""
        current_temp = self.zbe_tensor.read_cpu_temperature()
        profit_zones = self.assign_profit_zones(sha_key)
        
        if not profit_zones:
            return 4  # Default to 4-bit if no profit zones
            
        # Calculate average thermal load
        avg_thermal = np.mean([zone['thermal'] for zone in profit_zones])
        
        # Select bit depth based on thermal conditions
        return self._select_bit_depth(avg_thermal)

    def _select_bit_depth(self, thermal: float) -> int:
        """Select appropriate bit depth based on thermal conditions."""
        for bit_depth, threshold in sorted(self.thermal_thresholds.items()):
            if thermal <= threshold:
                return bit_depth
        return 4  # Default to 4-bit if too hot

    def export_state(self) -> Dict:
        """Export current state for monitoring/debugging."""
        return {
            'sweep_count': len(self.sweep_log),
            'wall_pressure_count': len(self.wall_pressure_map),
            'current_temp': self.zbe_tensor.read_cpu_temperature(),
            'tensor_weights': self.zbe_tensor.get_current_tensor_weights().tolist()
        }

    def check_drift_alert(self, threshold: float = 5.0) -> bool:
        temps = [s.thermal_tag for s in self.sweep_log if s.thermal_tag]
        if len(temps) < 2:
            return False
        drift = np.max(temps) - np.min(temps)
        return drift > threshold

    def reinforce_depth(self, sha_key: str, bit_depth: int):
        if sha_key not in self.depth_memory[bit_depth]:
            self.depth_memory[bit_depth].append(sha_key)

    def tag_heat_cluster(self, sha_key: str):
        zones = self.assign_profit_zones(sha_key)
        return {
            'sha_key': sha_key,
            'zone_count': len(zones),
            'avg_thermal': np.mean([z['thermal'] for z in zones]) if zones else None
        }

    def score_sweep(self, sweep: SweepSignal) -> float:
        # Example scoring logic
        return sweep.volume * np.exp(-sweep.entropy / 10)

    def rebind_zone(self, sha_key: str, new_volume: float):
        for zone in self.assign_profit_zones(sha_key):
            if abs(zone['volume'] - new_volume) > 0.2 * zone['volume']:
                zone['volume'] = (zone['volume'] + new_volume) / 2 