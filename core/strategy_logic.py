"""
Strategy Logic
=============

Coordinates all components for thermal-aware profit allocation.
Routes signals through orchestrator and manages strategy execution.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import hashlib

from .profit_sweep_allocator import ProfitSweepAllocator, SweepSignal
from .zbe_temperature_tensor import ZBETemperatureTensor
from .profit_tensor import ProfitTensorStore
from .fault_bus import FaultBus, FaultBusEvent
from .bitmap_engine import BitmapEngine
from .memory_timing_orchestrator import MemoryTimingOrchestrator

@dataclass
class StrategyState:
    sha_key: str
    bit_depth: int
    profit_history: List[float]
    thermal_history: List[float]
    last_update: float
    is_active: bool = True

class StrategyLogic:
    def __init__(self):
        self.sweep_allocator = ProfitSweepAllocator()
        self.zbe_tensor = ZBETemperatureTensor()
        self.profit_store = ProfitTensorStore()
        self.fault_bus = FaultBus()
        self.bitmap_engine = BitmapEngine()
        self.orchestrator = MemoryTimingOrchestrator()
        
        # Strategy state tracking
        self.strategy_states: Dict[str, StrategyState] = {}
        
        # Register fault handlers
        self._register_fault_handlers()
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _register_fault_handlers(self):
        """Register handlers for various fault types."""
        def handle_thermal_fault(event: FaultBusEvent):
            if event.severity > 0.8:
                self._deactivate_hot_strategies()
        
        def handle_profit_fault(event: FaultBusEvent):
            if event.severity < 0.2:
                self._optimize_profit_strategies()
        
        self.fault_bus.register_handler("thermal_high", handle_thermal_fault)
        self.fault_bus.register_handler("profit_low", handle_profit_fault)

    def process_tick(self, tick_data: Dict) -> Optional[Dict]:
        """
        Process a single market tick through the strategy pipeline.
        
        Args:
            tick_data: Dictionary containing tick information
                - price: float
                - volume: float
                - timestamp: float
                - direction: str ('buy' or 'sell')
        
        Returns:
            Optional[Dict]: Strategy decision if any
        """
        try:
            # Create sweep signal
            sweep = SweepSignal(
                timestamp=tick_data['timestamp'],
                price=tick_data['price'],
                volume=tick_data['volume'],
                direction=tick_data['direction'],
                sha_key=self._generate_sha_key(tick_data)
            )
            
            # Process sweep through allocator
            self.sweep_allocator.process_sweep(sweep)
            
            # Get current thermal conditions
            current_temp = self.zbe_tensor.read_cpu_temperature()
            
            # Check thermal conditions
            if current_temp > self.zbe_tensor.max_temp:
                self.fault_bus.push(FaultBusEvent(
                    tick=int(tick_data['timestamp']),
                    module="strategy_logic",
                    type="thermal_critical",
                    severity=1.0,
                    metadata={"temperature": current_temp}
                ))
                return None
            
            # Get optimal bit depth
            optimal_depth = self.sweep_allocator.get_optimal_bit_depth(sweep.sha_key)
            
            # Update strategy state
            self._update_strategy_state(sweep.sha_key, optimal_depth)
            
            # Calculate profit potential
            profit_zones = self.sweep_allocator.assign_profit_zones(sweep.sha_key)
            if not profit_zones:
                return None
            
            # Get best profit zone
            best_zone = max(profit_zones, key=lambda x: x['volume'])
            
            # Calculate execution parameters
            execution_rate = self.zbe_tensor.get_current_tensor_weights()[optimal_depth - 1]
            
            return {
                'action': best_zone['type'],
                'price': best_zone['price'],
                'volume': best_zone['volume'] * execution_rate,
                'bit_depth': optimal_depth,
                'confidence': execution_rate
            }
            
        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")
            return None

    def _generate_sha_key(self, tick_data: Dict) -> str:
        """Generate SHA key from tick data."""
        data_str = f"{tick_data['price']}:{tick_data['volume']}:{tick_data['direction']}"
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _update_strategy_state(self, sha_key: str, bit_depth: int):
        """Update strategy state with new information."""
        if sha_key not in self.strategy_states:
            self.strategy_states[sha_key] = StrategyState(
                sha_key=sha_key,
                bit_depth=bit_depth,
                profit_history=[],
                thermal_history=[],
                last_update=time.time()
            )
        
        state = self.strategy_states[sha_key]
        state.bit_depth = bit_depth
        state.last_update = time.time()
        
        # Update profit history
        profit = self.sweep_allocator.calculate_profit_per_thermal_unit(sha_key)
        state.profit_history.append(profit)
        state.profit_history = state.profit_history[-1000:]  # Keep last 1000 entries
        
        # Update thermal history
        thermal = self.zbe_tensor.read_cpu_temperature()
        state.thermal_history.append(thermal)
        state.thermal_history = state.thermal_history[-1000:]  # Keep last 1000 entries

    def _deactivate_hot_strategies(self):
        """Deactivate strategies that are causing high thermal load."""
        current_temp = self.zbe_tensor.read_cpu_temperature()
        for sha_key, state in self.strategy_states.items():
            if state.is_active and np.mean(state.thermal_history[-10:]) > self.zbe_tensor.optimal_temp:
                state.is_active = False
                self.logger.warning(f"Deactivated strategy {sha_key} due to high thermal load")

    def _optimize_profit_strategies(self):
        """Optimize strategies for better profit performance."""
        for sha_key, state in self.strategy_states.items():
            if state.is_active:
                avg_profit = np.mean(state.profit_history[-100:])
                if avg_profit < 0.1:  # Low profit threshold
                    state.bit_depth = max(4, state.bit_depth - 1)  # Reduce bit depth
                    self.logger.info(f"Reduced bit depth for strategy {sha_key} to {state.bit_depth}")

    def get_strategy_stats(self) -> Dict:
        """Get statistics about current strategy states."""
        return {
            'active_strategies': sum(1 for s in self.strategy_states.values() if s.is_active),
            'total_strategies': len(self.strategy_states),
            'avg_profit': np.mean([np.mean(s.profit_history) for s in self.strategy_states.values()]),
            'avg_thermal': np.mean([np.mean(s.thermal_history) for s in self.strategy_states.values()]),
            'bit_depth_distribution': {
                depth: sum(1 for s in self.strategy_states.values() if s.bit_depth == depth)
                for depth in [4, 8, 16, 42, 81]
            }
        }

    def tensors_in_sync(self, tensor_a: np.ndarray, tensor_b: np.ndarray, tolerance: float = 0.01) -> bool:
        diff = np.abs(tensor_a - tensor_b)
        return np.mean(diff) < tolerance

    def visualize_sync_drift(self, tensors: Dict[str, np.ndarray], tolerance: float = 0.01):
        for key in tensors.keys():
            plt.scatter(tensors[key], label=key)
        # Add drift-colored points or sync masks here
        plt.legend()
        plt.show()

    def auto_correct_drift(self, tensor_a: np.ndarray, tensor_b: np.ndarray, tolerance: float = 0.01):
        if not self.tensors_in_sync(tensor_a, tensor_b, tolerance):
            # Implement re-alignment logic here
            print("Drift detected, attempting to correct...")
            # Example: Resample or interpolate tensors

class TensorSyncMonitor:
    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance

    def check_sync(self, tensors: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], bool]:
        results = {}
        types = list(tensors.keys())
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                a, b = types[i], types[j]
                synced = np.mean(np.abs(tensors[a] - tensors[b])) < self.tolerance
                results[(a, b)] = synced
        return results

def hash_tensor(tensor: np.ndarray) -> str:
    return hashlib.sha256(tensor.tobytes()).hexdigest()

def detect_symbolic_change(tensors: Dict[str, np.ndarray]) -> Dict[str, bool]:
    current_hashes = {key: hash_tensor(tensor) for key, tensor in tensors.items()}
    previous_hashes = {key: current_hashes[key] if key in previous_hashes else None for key in current_hashes.keys()}
    changes = {key: current_hashes[key] != previous_hashes[key] for key in current_hashes.keys()}
    return changes

# Example usage:
if __name__ == "__main__":
    strategy = StrategyLogic()
    
    # Process a sample tick
    tick_data = {
        'price': 50000.0,
        'volume': 1.5,
        'timestamp': time.time(),
        'direction': 'buy'
    }
    
    decision = strategy.process_tick(tick_data)
    if decision:
        print(f"Action: {decision['action']}")
        print(f"Price: {decision['price']}")
        print(f"Volume: {decision['volume']}")
        print(f"Bit Depth: {decision['bit_depth']}")
        print(f"Confidence: {decision['confidence']}")
    
    # Print strategy stats
    print("\nStrategy Stats:")
    stats = strategy.get_strategy_stats()
    print(json.dumps(stats, indent=2))

    # Generate a new memory key
    key = strategy.orchestrator.generate_memory_key(tick_data, bit_depth=16)

    # Access and update a memory key
    accessed_key = strategy.orchestrator.access_memory_key(strategy._generate_sha_key(tick_data))

    # Update success score
    strategy.orchestrator.update_success_score(strategy._generate_sha_key(tick_data), 0.8)

    # Get memory statistics
    stats = strategy.orchestrator.get_memory_stats()

    # Clean up old keys
    strategy.orchestrator.cleanup_old_keys(max_age_hours=24) 