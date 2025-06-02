"""
Unified Observability Tensor
==========================

Implements Schwabot's central nervous system for real-time monitoring and visualization.
Integrates all system components into a unified tensor space for recursive feedback.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading
from collections import deque

@dataclass
class TensorComponent:
    """Represents a component's contribution to the unified tensor"""
    value: Any
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    source_module: str = ""
    confidence: float = 1.0
    thermal_cost: float = 0.0
    memory_weight: float = 0.0
    phase_depth: float = 0.0

class UnifiedObservabilityTensor:
    """Central nervous system for Schwabot's real-time monitoring"""
    
    def __init__(
        self,
        history_size: int = 1000,
        update_interval: float = 0.1
    ):
        self.history_size = history_size
        self.update_interval = update_interval
        
        # Initialize core tensor components
        self._components: Dict[str, TensorComponent] = {
            # Thermal and Memory Components
            "thermal_entropy": TensorComponent(
                value=0.0,
                source_module="ZBETemperatureTensor"
            ),
            "memory_allocation": TensorComponent(
                value={},
                source_module="ThermalMapAllocator"
            ),
            "fault_state": TensorComponent(
                value={},
                source_module="FaultBus"
            ),
            
            # Trading Intelligence Components
            "basket_phase": TensorComponent(
                value={},
                source_module="BasketPhaseMap"
            ),
            "edge_vectors": TensorComponent(
                value={},
                source_module="EdgeVectorField"
            ),
            "drift_metrics": TensorComponent(
                value={},
                source_module="DriftExitDetector"
            ),
            
            # Bit-Depth and Strategy Components
            "bit_depth_activation": TensorComponent(
                value={},
                source_module="BasketSwapOverlayRouter"
            ),
            "active_strategy": TensorComponent(
                value="",
                source_module="StrategyLogic"
            ),
            "profit_metrics": TensorComponent(
                value={},
                source_module="ProfitSweepAllocator"
            )
        }
        
        # Initialize state tracking
        self.last_update_time = datetime.now().timestamp()
        self.state_history = deque(maxlen=history_size)
        self.tensor_lock = threading.Lock()
        
        # Initialize visualization tensors
        self.thermal_tensor = np.zeros((10, 10, 10))  # 3D thermal state
        self.memory_tensor = np.zeros((10, 10, 10))   # 3D memory state
        self.profit_tensor = np.zeros((10, 10, 10))   # 3D profit state
        
        # Initialize performance metrics
        self.performance_metrics = {
            'total_updates': 0,
            'thermal_violations': 0,
            'memory_violations': 0,
            'fault_events': 0,
            'avg_update_latency': 0.0,
            'avg_thermal_cost': 0.0,
            'avg_memory_weight': 0.0
        }

    def update_component(
        self,
        component_name: str,
        value: Any,
        source_module: str,
        confidence: float = 1.0,
        thermal_cost: float = 0.0,
        memory_weight: float = 0.0,
        phase_depth: float = 0.0
    ):
        """Update a component's state in the unified tensor"""
        with self.tensor_lock:
            if component_name in self._components:
                # Update component
                self._components[component_name].value = value
                self._components[component_name].timestamp = datetime.now().timestamp()
                self._components[component_name].source_module = source_module
                self._components[component_name].confidence = confidence
                self._components[component_name].thermal_cost = thermal_cost
                self._components[component_name].memory_weight = memory_weight
                self._components[component_name].phase_depth = phase_depth
                
                # Update performance metrics
                self.performance_metrics['total_updates'] += 1
                self.performance_metrics['avg_thermal_cost'] = (
                    0.9 * self.performance_metrics['avg_thermal_cost'] +
                    0.1 * thermal_cost
                )
                self.performance_metrics['avg_memory_weight'] = (
                    0.9 * self.performance_metrics['avg_memory_weight'] +
                    0.1 * memory_weight
                )
                
                # Update visualization tensors
                self._update_visualization_tensors(component_name, value)
                
                # Store state in history
                self._store_state()
            else:
                print(f"Warning: Unknown component {component_name}")

    def get_component_state(self, component_name: str) -> Optional[TensorComponent]:
        """Get current state of a component"""
        return self._components.get(component_name)

    def get_system_snapshot(self) -> Dict[str, Any]:
        """Get complete system state snapshot"""
        with self.tensor_lock:
            snapshot = {
                'timestamp': datetime.fromtimestamp(self.last_update_time).isoformat(),
                'components': {
                    name: {
                        'value': comp.value,
                        'timestamp': datetime.fromtimestamp(comp.timestamp).isoformat(),
                        'source': comp.source_module,
                        'confidence': comp.confidence,
                        'thermal_cost': comp.thermal_cost,
                        'memory_weight': comp.memory_weight,
                        'phase_depth': comp.phase_depth
                    }
                    for name, comp in self._components.items()
                },
                'performance': self.performance_metrics.copy(),
                'visualization': {
                    'thermal_tensor': self.thermal_tensor.tolist(),
                    'memory_tensor': self.memory_tensor.tolist(),
                    'profit_tensor': self.profit_tensor.tolist()
                }
            }
            return snapshot

    def get_state_history(self, window_size: int = 100) -> List[Dict]:
        """Get recent state history"""
        return list(self.state_history)[-window_size:]

    def _update_visualization_tensors(self, component_name: str, value: Any):
        """Update visualization tensors based on component updates"""
        if component_name == "thermal_entropy":
            # Update thermal tensor
            thermal_value = float(value)
            x = int(thermal_value * 9)
            y = int(self._components["memory_allocation"].memory_weight * 9)
            z = int(self._components["profit_metrics"].value.get('profit_gradient', 0.0) * 9)
            self.thermal_tensor[x, y, z] = thermal_value
            
        elif component_name == "memory_allocation":
            # Update memory tensor
            memory_value = float(value.get('utilization', 0.0))
            x = int(memory_value * 9)
            y = int(self._components["thermal_entropy"].value * 9)
            z = int(self._components["bit_depth_activation"].value.get('current_depth', 0) / 81 * 9)
            self.memory_tensor[x, y, z] = memory_value
            
        elif component_name == "profit_metrics":
            # Update profit tensor
            profit_value = float(value.get('profit_gradient', 0.0))
            x = int(profit_value * 9)
            y = int(self._components["basket_phase"].phase_depth * 9)
            z = int(self._components["edge_vectors"].value.get('confidence', 0.0) * 9)
            self.profit_tensor[x, y, z] = profit_value

    def _store_state(self):
        """Store current state in history"""
        self.state_history.append(self.get_system_snapshot())
        self.last_update_time = datetime.now().timestamp()

    def get_visualization_tensors(self) -> Dict[str, np.ndarray]:
        """Get current visualization tensors"""
        return {
            'thermal': self.thermal_tensor.copy(),
            'memory': self.memory_tensor.copy(),
            'profit': self.profit_tensor.copy()
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

# Example usage
if __name__ == "__main__":
    uot = UnifiedObservabilityTensor()
    
    # Simulate component updates
    uot.update_component(
        "thermal_entropy",
        value=0.75,
        source_module="ZBETemperatureTensor",
        thermal_cost=0.3
    )
    
    uot.update_component(
        "memory_allocation",
        value={'utilization': 0.6, 'coherence': 0.8},
        source_module="ThermalMapAllocator",
        memory_weight=0.5
    )
    
    uot.update_component(
        "profit_metrics",
        value={'profit_gradient': 0.002, 'thermal_cost': 0.2},
        source_module="ProfitSweepAllocator",
        confidence=0.9
    )
    
    # Get system snapshot
    snapshot = uot.get_system_snapshot()
    print("\nSystem Snapshot:")
    print(json.dumps(snapshot, indent=2))
    
    # Get visualization tensors
    tensors = uot.get_visualization_tensors()
    print("\nVisualization Tensors:")
    for name, tensor in tensors.items():
        print(f"{name}: shape={tensor.shape}, mean={tensor.mean():.4f}") 