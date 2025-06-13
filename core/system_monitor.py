"""
System Monitor
============

Implements real-time monitoring of Schwabot's system state.
Integrates with all components and provides comprehensive monitoring.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import threading
import time
import json
from collections import deque
import random
import hashlib
import string
import svgwrite

from core.unified_observability_tensor import UnifiedObservabilityTensor
from core.tensor_visualization_controller import TensorVisualizationController
from core.basket_swapper import BasketSwapper
from core.phase_engine.basket_phase_map import BasketPhaseMap
from core.basket_tensor_feedback import BasketTensorFeedback
from core.basket_entropy_allocator import BasketEntropyAllocator
from core.basket_log_controller import BasketLogController
from core.drift_exit_detector import DriftExitDetector
from core.basket_swap_overlay_router import BasketSwapOverlayRouter
from core.edge_vector_field import EdgeVectorField
from core.thermal_map_allocator import ThermalMapAllocator

class SystemMonitor:
    """Monitors and visualizes Schwabot's system state"""
    
    def __init__(
        self,
        update_interval: float = 0.1,
        history_size: int = 1000
    ):
        self.update_interval = update_interval
        self.history_size = history_size
        
        # Initialize core components
        self.uot = UnifiedObservabilityTensor(history_size=history_size)
        self.visualizer = TensorVisualizationController(
            update_interval=update_interval,
            history_size=history_size
        )
        
        # Initialize component references
        self.components = {}
        self.component_lock = threading.Lock()
        
        # Initialize monitoring state
        self.running = False
        self.monitor_thread = None
        self.monitor_lock = threading.Lock()
        
        # Initialize performance tracking
        self.performance_history = deque(maxlen=history_size)
        self.last_update_time = datetime.now().timestamp()

    def register_component(
        self,
        component_name: str,
        component_instance: object
    ):
        """Register a component for monitoring"""
        with self.component_lock:
            self.components[component_name] = component_instance

    def start_monitoring(self):
        """Start system monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start visualization
        self.visualizer.start_visualization()

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.visualizer.stop_visualization()

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                with self.monitor_lock:
                    # Update component states
                    self._update_component_states()
                    
                    # Update visualization
                    tensors = self.uot.get_visualization_tensors()
                    self.visualizer.update_tensors(
                        tensors['thermal'],
                        tensors['memory'],
                        tensors['profit']
                    )
                    
                    # Store performance metrics
                    self._store_performance_metrics()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                time.sleep(1.0)  # Prevent tight loop on error

    def _update_component_states(self):
        """Update states of all registered components"""
        # Update thermal state
        if 'thermal_map_allocator' in self.components:
            allocator = self.components['thermal_map_allocator']
            stats = allocator.get_memory_stats()
            self.uot.update_component(
                "thermal_entropy",
                value=stats['avg_thermal_state'],
                source_module="ThermalMapAllocator",
                thermal_cost=stats['avg_thermal_state'],
                memory_weight=stats['memory_utilization']
            )
        
        # Update basket state
        if 'basket_swapper' in self.components:
            swapper = self.components['basket_swapper']
            metrics = swapper.get_basket_metrics()
            self.uot.update_component(
                "basket_phase",
                value=metrics,
                source_module="BasketSwapper",
                confidence=metrics.get('trust_score', 0.0),
                phase_depth=metrics.get('phase_depth', 0.0)
            )
        
        # Update edge vectors
        if 'edge_vector_field' in self.components:
            field = self.components['edge_vector_field']
            stats = field.get_pattern_stats()
            self.uot.update_component(
                "edge_vectors",
                value=stats,
                source_module="EdgeVectorField",
                confidence=stats.get('success_rate', 0.0)
            )
        
        # Update drift metrics
        if 'drift_exit_detector' in self.components:
            detector = self.components['drift_exit_detector']
            stats = detector.get_drift_stats('BTC/USD')  # Example pair
            self.uot.update_component(
                "drift_metrics",
                value=stats,
                source_module="DriftExitDetector",
                confidence=stats.get('avg_drift_confidence', 0.0)
            )
        
        # Update bit depth activation
        if 'basket_swap_overlay_router' in self.components:
            router = self.components['basket_swap_overlay_router']
            stats = router.get_tier_stats()
            self.uot.update_component(
                "bit_depth_activation",
                value=stats,
                source_module="BasketSwapOverlayRouter",
                confidence=stats.get('success_rate', 0.0)
            )

    def _store_performance_metrics(self):
        """Store current performance metrics"""
        metrics = self.uot.get_performance_metrics()
        metrics['timestamp'] = datetime.now().timestamp()
        self.performance_history.append(metrics)
        self.last_update_time = metrics['timestamp']

    def get_performance_history(self, window_size: int = 100) -> List[Dict]:
        """Get recent performance history"""
        return list(self.performance_history)[-window_size:]

    def get_system_snapshot(self) -> Dict:
        """Get complete system state snapshot"""
        return {
            'timestamp': datetime.fromtimestamp(self.last_update_time).isoformat(),
            'components': {
                name: type(instance).__name__
                for name, instance in self.components.items()
            },
            'performance': self.uot.get_performance_metrics(),
            'visualization': self.uot.get_visualization_tensors()
        }

    def _calculate_thermal_state(self, key: str, size: int) -> float:
        base_thermal = min(1.0, size / self.max_memory)
        if key in self.thermal_map:
            existing_thermal = self.thermal_map[key]
            return (base_thermal + existing_thermal) / 2
        return base_thermal

    def _calculate_coherence_score(self, key: str) -> float:
        if key not in self.memory_regions:
            return 1.0
        regions = self.memory_regions[key]
        if not regions:
            return 1.0
        total_size = sum(region.size for region in regions)
        max_gap = 0
        sorted_regions = sorted(regions, key=lambda r: r.start_address)
        for i in range(len(sorted_regions) - 1):
            gap = sorted_regions[i + 1].start_address - (
                sorted_regions[i].start_address + sorted_regions[i].size
            )
            max_gap = max(max_gap, gap)
        return 1.0 / (1.0 + max_gap / total_size)

    def calculate_advanced_coherence(self, region: MemoryRegion) -> float:
        """Calculate advanced coherence score with thermal awareness"""
        base_coherence = self._calculate_coherence_score(region.key)
        thermal_factor = 1.0 - (region.thermal_state / self.max_thermal_threshold)
        access_factor = region.access_frequency
        return base_coherence * thermal_factor * access_factor

    def update_thermal_thresholds(self, current_temp: float, profit_gradient: float):
        """Dynamically adjust thermal thresholds based on performance"""
        for bit_depth in self.thermal_thresholds:
            base_threshold = self.thermal_thresholds[bit_depth]
            profit_factor = 1.0 + (profit_gradient * 0.1)  # Scale by profit
            temp_factor = 1.0 - (current_temp / 100.0)  # Scale by current temp
            self.thermal_thresholds[bit_depth] = base_threshold * profit_factor * temp_factor

    def optimize_bit_depth(self, sha_key: str) -> int:
        """Optimize bit depth based on profit and thermal efficiency"""
        current_temp = self.zbe_tensor.read_cpu_temperature()
        profit_zones = self.assign_profit_zones(sha_key)
        
        if not profit_zones:
            return 4
        
        efficiency_scores = []
        for bit_depth in self.thermal_thresholds:
            thermal_cost = current_temp / self.thermal_thresholds[bit_depth]
            profit_potential = self.calculate_profit_potential(bit_depth, profit_zones)
            efficiency_scores.append(profit_potential / thermal_cost)
        
        return self.bit_depths[np.argmax(efficiency_scores)]

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()[:8]  # short hash

def generate_matrix(rows=12, cols=24, seed=None):
    if seed is not None:
        random.seed(seed)

    charset = string.digits + string.ascii_letters
    matrix = []

    for _ in range(rows):
        row = ''.join(random.choice(charset) for _ in range(cols))
        matrix.append(row)

    return matrix

def invert_syntax(matrix):
    # Example: reverse every even row, flip case every 3rd
    inverted = []
    for i, row in enumerate(matrix):
        if i % 2 == 0:
            row = row[::-1]
        if i % 3 == 0:
            row = ''.join(c.upper() if c.islower() else c.lower() for c in row)
        inverted.append(row)
    return inverted

def co_match_hashes(matrix):
    hash_map = {}
    for i, row in enumerate(matrix):
        h = hash_string(row)
        if h not in hash_map:
            hash_map[h] = []
        hash_map[h].append(i)
    return {h: idxs for h, idxs in hash_map.items() if len(idxs) > 1}

def print_matrix(matrix):
    for row in matrix:
        print(row)

def generate_diagonal_hashes(matrix):
    diagonals = {}
    n = len(matrix)
    m = len(matrix[0])

    # Generate hashes along main diagonal
    for i in range(n):
        diag_hash = hash_string(''.join(matrix[i][i] for i in range(n)))
        if diag_hash not in diagonals:
            diagonals[diag_hash] = []
        diagonals[diag_hash].append(i)

    # Generate hashes along anti-diagonal
    for i in range(n):
        diag_hash = hash_string(''.join(matrix[i][n - 1 - i] for i in range(n)))
        if diag_hash not in diagonals:
            diagonals[diag_hash] = []
        diagonals[diag_hash].append(i)

    return diagonals

def print_diagonal_hashes(diagonals):
    for h, idxs in diagonals.items():
        print(f"Diagonal Hash: {h} -> Rows: {idxs}")

def generate_visual_grid(matrix, filename='matrix.svg'):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    cell_size = 20
    margin = 10

    for i, row in enumerate(matrix):
        for j, char in enumerate(row):
            x = j * cell_size + margin
            y = i * cell_size + margin
            dwg.add(dwg.text(char, insert=(x, y), font_size=cell_size))

    dwg.save()

def integrate_with_profit_mapping(matrix, profit_map_script):
    # Example: Run a script to map profits across the matrix
    print(f"Integrating with {profit_map_script}...")
    # This is a placeholder for actual integration logic
    pass

# Run full pipeline
matrix = generate_matrix()
inverted = invert_syntax(matrix)
matches = co_match_hashes(inverted)
diagonals = generate_diagonal_hashes(inverted)

print("Original:")
print_matrix(matrix)
print("\nInverted:")
print_matrix(inverted)
print("\nCo-matching hashes:")
for h, idxs in matches.items():
    print(f"Hash: {h} -> Rows: {idxs}")
print("\nDiagonal Hashes:")
print_diagonal_hashes(diagonals)

# Generate visual grid
generate_visual_grid(matrix)

# Integrate with profit mapping script
integrate_with_profit_mapping(matrix, 'profit_map_script.py')

# Example usage
if __name__ == "__main__":
    monitor = SystemMonitor()
    
    # Register components
    monitor.register_component('thermal_map_allocator', ThermalMapAllocator())
    monitor.register_component('basket_swapper', BasketSwapper())
    monitor.register_component('edge_vector_field', EdgeVectorField())
    monitor.register_component('drift_exit_detector', DriftExitDetector())
    monitor.register_component('basket_swap_overlay_router', BasketSwapOverlayRouter())
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()

thermal_thresholds = {
    4: 70.0,   # 4-bit: high temp allowed
    8: 65.0,   # 8-bit: moderate temp
    16: 60.0,  # 16-bit: careful temp
    42: 55.0,  # 42-bit: low temp
    81: 50.0   # 81-bit: very low temp
} 