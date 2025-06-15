"""
Profit Sweep Allocator
=====================

Coordinates thermal-aware profit allocation and sweep detection across different bit depths.
Integrates with ZBE temperature tensor, profit tensor store, and sweep signal log.
Enhanced with fractal command dispatcher integration for TFF/TPF/TEF systems.
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
from .fractal_command_dispatcher import (
    FractalCommandDispatcher, FractalSystemType, CommandType, FractalCommand
)
from .recursive_profit import RecursiveMarketState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SweepSignal:
    timestamp: float
    price: float
    volume: float
    direction: str  # 'buy' or 'sell'
    sha_key: str
    thermal_tag: Optional[float] = None
    profit_potential: Optional[float] = None
    # Enhanced with fractal predictions
    tff_prediction: Optional[float] = None
    tpf_resolution: Optional[float] = None
    tef_echo_strength: Optional[float] = None

class ProfitSweepAllocator:
    def __init__(self):
        self.zbe_tensor = ZBETemperatureTensor()
        self.profit_store = ProfitTensorStore()
        self.bitmap_engine = BitmapEngine()
        self.sweep_log: List[SweepSignal] = []
        self.wall_pressure_map: Dict[str, Dict] = {}
        
        # Initialize fractal command dispatcher
        self.fractal_dispatcher = FractalCommandDispatcher()
        logger.info("Profit Sweep Allocator initialized with fractal integration")
        
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

    def _create_market_state(self, sweep: SweepSignal) -> RecursiveMarketState:
        """Create market state from sweep signal for fractal analysis"""
        return RecursiveMarketState(
            timestamp=datetime.fromtimestamp(sweep.timestamp),
            price=sweep.price,
            volume=sweep.volume,
            tff_stability_index=0.8,  # Default, will be updated by fractal calculations
            paradox_stability_score=0.7,  # Default
            memory_coherence_level=0.6,  # Default
            historical_echo_strength=0.5  # Default
        )

    def _enhance_sweep_with_fractals(self, sweep: SweepSignal) -> SweepSignal:
        """Enhance sweep signal with fractal predictions"""
        try:
            # Create market state for fractal analysis
            market_state = self._create_market_state(sweep)
            
            # TFF Prediction Command
            tff_cmd = self.fractal_dispatcher.create_tff_command(
                CommandType.PREDICT,
                market_state=market_state,
                horizon=10
            )
            self.fractal_dispatcher.dispatch_command(tff_cmd)
            
            # TPF Resolution Command (check for paradoxes)
            base_profit = sweep.volume * 0.01  # Simple profit estimate
            tff_profit = base_profit * 1.1  # Assume TFF expansion
            tpf_cmd = self.fractal_dispatcher.create_tpf_command(
                CommandType.RESOLVE,
                base_profit=base_profit,
                tff_profit=tff_profit,
                market_state=market_state
            )
            self.fractal_dispatcher.dispatch_command(tpf_cmd)
            
            # TEF Echo Amplification Command
            tef_cmd = self.fractal_dispatcher.create_tef_command(
                CommandType.AMPLIFY,
                market_state=market_state,
                lookback_periods=50
            )
            self.fractal_dispatcher.dispatch_command(tef_cmd)
            
            # Process all commands
            processed_commands = self.fractal_dispatcher.process_commands()
            
            # Extract results
            for cmd in processed_commands:
                if cmd.status == "completed" and cmd.result:
                    if cmd.system_type == FractalSystemType.TFF and cmd.command_type == CommandType.PREDICT:
                        sweep.tff_prediction = cmd.result.get('predicted_movement', 0.0)
                    elif cmd.system_type == FractalSystemType.TPF and cmd.command_type == CommandType.RESOLVE:
                        sweep.tpf_resolution = cmd.result.get('resolved_profit', 0.0)
                    elif cmd.system_type == FractalSystemType.TEF and cmd.command_type == CommandType.AMPLIFY:
                        sweep.tef_echo_strength = cmd.result.get('amplified_signal', 0.0)
            
            logger.info(f"Enhanced sweep {sweep.sha_key} with fractal predictions: "
                       f"TFF={sweep.tff_prediction}, TPF={sweep.tpf_resolution}, TEF={sweep.tef_echo_strength}")
            
        except Exception as e:
            logger.error(f"Error enhancing sweep with fractals: {e}")
            # Set default values if fractal enhancement fails
            sweep.tff_prediction = 0.0
            sweep.tpf_resolution = 0.0
            sweep.tef_echo_strength = 0.0
        
        return sweep

    def _calculate_fractal_profit_potential(self, sweep: SweepSignal) -> float:
        """Calculate profit potential using all three fractal systems"""
        if not all([sweep.tff_prediction, sweep.tpf_resolution, sweep.tef_echo_strength]):
            return 0.0
        
        # Weighted combination of fractal predictions
        tff_weight = 0.4  # Forever Fractals - structural foundation
        tpf_weight = 0.3  # Paradox Fractals - stability correction
        tef_weight = 0.3  # Echo Fractals - historical validation
        
        fractal_profit = (
            sweep.tff_prediction * tff_weight +
            sweep.tpf_resolution * tpf_weight +
            sweep.tef_echo_strength * tef_weight
        )
        
        return fractal_profit

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
        x, y = np.meshgrid(range(10), range(10))  # Fixed dimensions
        
        try:
            # Normalize tensor values based on bit depth
            normalized_tensor = self._normalize_tensor(self.zbe_tensor.get_current_tensor(), bit_depth)
            
            # Ensure tensor has correct shape for plotting
            if normalized_tensor.shape[0] >= 10 and normalized_tensor.shape[1] >= 10:
                plot_tensor = normalized_tensor[:10, :10]
                ax.plot_surface(x, y, plot_tensor, cmap='viridis', alpha=0.8)
            else:
                # Create dummy surface if tensor is wrong shape
                dummy_surface = np.random.rand(10, 10) * 0.5
                ax.plot_surface(x, y, dummy_surface, cmap='viridis', alpha=0.8)
                
        except ValueError as e:
            logger.warning(f"Surface plot error: {e}")
        
        # Set stable z-limits
        ax.set_zlim(0, 1)

    def map_tensor_to_symbolic(self, tensor: np.ndarray) -> str:
        flat = tensor.flatten()
        return ''.join(chr(int(val * 95) + 32) for val in flat[:64])  # 64-char symbol

    def map_wall_pressure(self, sweep: SweepSignal) -> Dict:
        """Map pressure points in buy/sell walls based on sweep signals with fractal enhancement."""
        if sweep.sha_key not in self.wall_pressure_map:
            self.wall_pressure_map[sweep.sha_key] = {
                'buy': [],
                'sell': [],
                'last_update': datetime.now().timestamp()
            }
        
        wall_map = self.wall_pressure_map[sweep.sha_key]
        
        # Enhanced wall entry with fractal data
        wall_entry = {
            'price': sweep.price,
            'volume': sweep.volume,
            'timestamp': sweep.timestamp,
            'thermal': sweep.thermal_tag,
            'fractal_profit': self._calculate_fractal_profit_potential(sweep),
            'tff_prediction': sweep.tff_prediction,
            'tpf_resolution': sweep.tpf_resolution,
            'tef_echo': sweep.tef_echo_strength
        }
        
        wall_map[sweep.direction].append(wall_entry)
        
        # Clean old entries (older than 1 hour)
        current_time = datetime.now().timestamp()
        wall_map[sweep.direction] = [
            entry for entry in wall_map[sweep.direction]
            if current_time - entry['timestamp'] < 3600
        ]
        
        return wall_map

    def assign_profit_zones(self, sha_key: str) -> List[Dict]:
        """Assign profit zones based on wall pressure, thermal conditions, and fractal predictions."""
        if sha_key not in self.wall_pressure_map:
            return []
            
        wall_map = self.wall_pressure_map[sha_key]
        profit_zones = []
        
        # Analyze buy wall pressure with fractal enhancement
        for entry in wall_map['buy']:
            if entry['thermal'] and entry['thermal'] < self.thermal_thresholds[4]:
                fractal_score = entry.get('fractal_profit', 0.0)
                
                profit_zones.append({
                    'type': 'buy',
                    'price': entry['price'],
                    'volume': entry['volume'],
                    'thermal': entry['thermal'],
                    'bit_depth': self._select_bit_depth(entry['thermal']),
                    'fractal_score': fractal_score,
                    'tff_prediction': entry.get('tff_prediction', 0.0),
                    'tpf_resolution': entry.get('tpf_resolution', 0.0),
                    'tef_echo': entry.get('tef_echo', 0.0),
                    'priority': self._calculate_zone_priority(entry, fractal_score)
                })
        
        # Analyze sell wall pressure with fractal enhancement
        for entry in wall_map['sell']:
            if entry['thermal'] and entry['thermal'] < self.thermal_thresholds[4]:
                fractal_score = entry.get('fractal_profit', 0.0)
                
                profit_zones.append({
                    'type': 'sell',
                    'price': entry['price'],
                    'volume': entry['volume'],
                    'thermal': entry['thermal'],
                    'bit_depth': self._select_bit_depth(entry['thermal']),
                    'fractal_score': fractal_score,
                    'tff_prediction': entry.get('tff_prediction', 0.0),
                    'tpf_resolution': entry.get('tpf_resolution', 0.0),
                    'tef_echo': entry.get('tef_echo', 0.0),
                    'priority': self._calculate_zone_priority(entry, fractal_score)
                })
        
        # Sort by priority (higher priority first)
        profit_zones.sort(key=lambda z: z['priority'], reverse=True)
        
        return profit_zones

    def _calculate_zone_priority(self, entry: Dict, fractal_score: float) -> float:
        """Calculate priority score for profit zone using fractal data"""
        base_priority = entry['volume'] / (entry['thermal'] + 1.0)  # Volume/thermal ratio
        fractal_boost = fractal_score * 2.0  # Fractal predictions boost priority
        
        # TFF stability bonus
        tff_bonus = entry.get('tff_prediction', 0.0) * 0.5
        
        # TPF paradox resolution bonus
        tpf_bonus = abs(entry.get('tpf_resolution', 0.0)) * 0.3
        
        # TEF echo strength bonus
        tef_bonus = entry.get('tef_echo', 0.0) * 0.2
        
        total_priority = base_priority + fractal_boost + tff_bonus + tpf_bonus + tef_bonus
        
        return max(0.0, total_priority)

    def calculate_profit_per_thermal_unit(self, sha_key: str) -> float:
        """Calculate profit efficiency per thermal unit with fractal enhancement."""
        profit_zones = self.assign_profit_zones(sha_key)
        if not profit_zones:
            return 0.0
            
        # Include fractal scores in profit calculation
        total_profit = sum(zone['volume'] + zone.get('fractal_score', 0.0) for zone in profit_zones)
        total_thermal = sum(zone['thermal'] for zone in profit_zones)
        
        if total_thermal == 0:
            return 0.0
            
        return total_profit / total_thermal

    def choose_next_sha_key(self) -> Optional[str]:
        """Choose next SHA key based on profit/thermal efficiency and fractal predictions."""
        current_temp = self.zbe_tensor.read_cpu_temperature()
        available_keys = list(self.wall_pressure_map.keys())
        
        if not available_keys:
            return None
            
        # Calculate efficiency scores with fractal enhancement
        efficiency_scores = []
        for sha_key in available_keys:
            profit_per_thermal = self.calculate_profit_per_thermal_unit(sha_key)
            thermal_weight = 1.0 - (current_temp / 100.0)  # Higher temp = lower weight
            
            # Add fractal prediction bonus
            profit_zones = self.assign_profit_zones(sha_key)
            fractal_bonus = sum(zone.get('fractal_score', 0.0) for zone in profit_zones) / max(len(profit_zones), 1)
            
            total_efficiency = (profit_per_thermal * thermal_weight) + (fractal_bonus * 0.3)
            efficiency_scores.append(total_efficiency)
        
        # Select best performing key
        best_idx = np.argmax(efficiency_scores)
        selected_key = available_keys[best_idx]
        
        logger.info(f"Selected SHA key {selected_key} with efficiency score {efficiency_scores[best_idx]:.4f}")
        return selected_key

    def process_sweep(self, sweep: SweepSignal):
        """Process a new sweep signal with fractal enhancement and update internal state."""
        # Get current CPU temperature
        sweep.thermal_tag = self.zbe_tensor.read_cpu_temperature()
        
        # Enhance sweep with fractal predictions
        sweep = self._enhance_sweep_with_fractals(sweep)
        
        # Calculate fractal-enhanced profit potential
        sweep.profit_potential = self._calculate_fractal_profit_potential(sweep)
        
        # Map wall pressure with fractal data
        self.map_wall_pressure(sweep)
        
        # Update ZBE tensor with enhanced profit data
        self.zbe_tensor.update_density_tensor(
            sweep.sha_key,
            {
                'profit': sweep.volume,
                'fractal_profit': sweep.profit_potential,
                'tff_prediction': sweep.tff_prediction or 0.0,
                'tpf_resolution': sweep.tpf_resolution or 0.0,
                'tef_echo': sweep.tef_echo_strength or 0.0
            }
        )
        
        # Store in sweep log
        self.sweep_log.append(sweep)
        
        # Clean old sweep log entries
        current_time = datetime.now().timestamp()
        self.sweep_log = [
            s for s in self.sweep_log
            if current_time - s.timestamp < 3600  # Keep last hour
        ]
        
        logger.info(f"Processed sweep {sweep.sha_key} with fractal profit potential: {sweep.profit_potential:.4f}")

    def get_optimal_bit_depth(self, sha_key: str) -> int:
        """Get optimal bit depth for current conditions with fractal considerations."""
        current_temp = self.zbe_tensor.read_cpu_temperature()
        profit_zones = self.assign_profit_zones(sha_key)
        
        if not profit_zones:
            return 4  # Default to 4-bit if no profit zones
            
        # Calculate average thermal load and fractal scores
        avg_thermal = np.mean([zone['thermal'] for zone in profit_zones])
        avg_fractal_score = np.mean([zone.get('fractal_score', 0.0) for zone in profit_zones])
        
        # Adjust thermal threshold based on fractal predictions
        # Higher fractal scores allow for slightly higher thermal tolerance
        thermal_adjustment = avg_fractal_score * 2.0  # Up to 2 degrees adjustment
        adjusted_thermal = avg_thermal - thermal_adjustment
        
        # Select bit depth based on adjusted thermal conditions
        return self._select_bit_depth(max(adjusted_thermal, 0.0))

    def _select_bit_depth(self, thermal: float) -> int:
        """Select appropriate bit depth based on thermal conditions."""
        for bit_depth, threshold in sorted(self.thermal_thresholds.items()):
            if thermal <= threshold:
                return bit_depth
        return 4  # Default to 4-bit if too hot

    def export_state(self) -> Dict:
        """Export current state for monitoring/debugging with fractal data."""
        fractal_status = self.fractal_dispatcher.get_system_status()
        
        return {
            'sweep_count': len(self.sweep_log),
            'wall_pressure_count': len(self.wall_pressure_map),
            'current_temp': self.zbe_tensor.read_cpu_temperature(),
            'tensor_weights': self.zbe_tensor.get_current_tensor_weights().tolist(),
            'fractal_systems': fractal_status['systems'],
            'fractal_queue_size': fractal_status['queue_size'],
            'total_fractal_commands': fractal_status['total_commands_processed'],
            'avg_fractal_profit': np.mean([s.profit_potential for s in self.sweep_log if s.profit_potential]) if self.sweep_log else 0.0
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
            'avg_thermal': np.mean([z['thermal'] for z in zones]) if zones else None,
            'avg_fractal_score': np.mean([z.get('fractal_score', 0.0) for z in zones]) if zones else 0.0
        }

    def score_sweep(self, sweep: SweepSignal) -> float:
        # Enhanced scoring with fractal data
        base_score = sweep.volume * np.exp(-getattr(sweep, 'entropy', 0.0) / 10)
        fractal_bonus = (sweep.profit_potential or 0.0) * 2.0
        return base_score + fractal_bonus

    def rebind_zone(self, sha_key: str, new_volume: float):
        for zone in self.assign_profit_zones(sha_key):
            if abs(zone['volume'] - new_volume) > 0.2 * zone['volume']:
                zone['volume'] = (zone['volume'] + new_volume) / 2 