# -*- coding: utf-8 -*-\n# Import safe print for Windows compatibility
try:
from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import math
except ImportError:
    pass
    pass
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
    pass
    pass
def safe_print(message):


    pass
    pass
    print(message)
def info(message):


    pass
    pass
    print(f"[INFO] {message}")
def warn(message):


    pass
    pass
    print(f"[WARN] {message}")
def error(message):


    pass
    pass
    print(f"[ERROR] {message}")
def success(message):


    pass
    pass
    print(f"[SUCCESS] {message}")
def debug(message):


    pass
    pass
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
"""
Schwabot Matrix Allocator
=========================

Flow director that controls how matrices (SFS/SFSS/SFSSS) receive and execute vector logic,
routing based on settings, adjusted vector weights, and tick map overlays.

This component:
- Routes trade logic to appropriate matrix cores
- Manages 10K tick memory and 16-bit map overlays
- Distributes vector logic based on settings controller
- Handles matrix waveform modes (4-bit, 8-bit, 42-phase)
- Integrates with fault controller and reinforcement learning
"""

import json
# from core.unified_math_system import unified_math  # F811: duplicate import
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

from .settings_controller import get_settings_controller
from .vector_validator import get_vector_validator


@dataclass
class MatrixAllocation:


    """Represents a matrix allocation decision"""
matrix_id: str
vector_id: str
allocation_confidence: float
routing_path: str
bit_level: int
phase_count: int
thermal_state: float
entropy_level: float
priority_weight: float
execution_mode: str  # "immediate", "queued", "monitored", "avoided"
timestamp: datetime


@dataclass
class TickMapState:


    """Represents the current state of the 10K tick map"""
tick_id: int
bit_level: int
phase_position: int
thermal_load: float
entropy_level: float
active_matrices: List[str]
memory_usage: float
last_update: datetime


class MatrixAllocator:


    """Flow director for matrix-based vector execution"""

def __init__(self):


    pass
    pass
        self.settings_controller = get_settings_controller()
        self.vector_validator = get_vector_validator()

        # Matrix registry
self.registered_matrices = {
"SFS8-A5": {"bit_level": 8, "phase_count": 42, "status": "active"},
"SFS16-B3": {"bit_level": 16, "phase_count": 42, "status": "active"},
"SFS42-C7": {"bit_level": 42, "phase_count": 42, "status": "active"},
"SFSS-D1": {"bit_level": 16, "phase_count": 64, "status": "active"},
"SFSSS-E9": {"bit_level": 32, "phase_count": 128, "status": "active"}
}

        # Tick map management (10K tick memory)
        self.tick_map = {}
self.current_tick_id = 0
self.tick_map_size = 10000

        # Allocation history
self.allocation_history: List[MatrixAllocation] = []

        # Matrix performance tracking
self.matrix_performance = {}

        # Initialize tick map
self._initialize_tick_map()

def _initialize_tick_map(self):


    pass
    pass
        """Initialize the 10K tick map"""
        for i in range(self.tick_map_size):
            self.tick_map[i] = TickMapState(]
                tick_id=i,
bit_level=16,
phase_position=0,
thermal_load=0.0,
entropy_level=0.5,
active_matrices=[],
memory_usage=0.0,
last_update=datetime.now()


def allocate_vector(self, vector_data: Dict[str, Any]) -> MatrixAllocation:


    pass
    pass
        """Allocate a vector to the appropriate matrix"""
        # Validate vector first
validation_result = self.vector_validator.validate_vector(vector_data)

        # Get current tick state
current_tick = self._get_current_tick_state()

        # Determine best matrix for allocation
best_matrix = self._select_best_matrix(vector_data, validation_result, current_tick)

        # Create allocation decision
allocation = MatrixAllocation(
            matrix_id=best_matrix["matrix_id"],
vector_id=vector_data.get("vector_id", "unknown"),
            allocation_confidence=validation_result.confidence_score,
routing_path=best_matrix["routing_path"],
bit_level=best_matrix["bit_level"],
phase_count=best_matrix["phase_count"],
thermal_state=current_tick.thermal_load,
entropy_level=current_tick.entropy_level,
priority_weight=validation_result.adjusted_weight,
execution_mode=self._determine_execution_mode(validation_result, best_matrix),
            timestamp=datetime.now()


        # Update tick map
self._update_tick_map(allocation)

        # Add to history
self.allocation_history.append(allocation)

        # Update matrix performance
self._update_matrix_performance(allocation)

        return allocation

def _get_current_tick_state(self) -> TickMapState:


    pass
    pass
        """Get current tick map state"""
        return self.tick_map[self.current_tick_id]

def _select_best_matrix(self, vector_data: Dict[str, Any],]


                          validation_result: Any, current_tick: TickMapState) -> Dict[str, Any]:
"""Select the best matrix for vector allocation"""
best_matrix = None
best_score = -1.0

        for matrix_id, matrix_info in self.registered_matrices.items():
            if matrix_info["status"] != "active":
                continue

            # Calculate matrix score
score = self._calculate_matrix_score(
                matrix_id, matrix_info, vector_data, validation_result, current_tick


            if score > best_score:
best_score = score
best_matrix = {
"matrix_id": matrix_id,
"bit_level": matrix_info["bit_level"],
"phase_count": matrix_info["phase_count"],
"routing_path": f"{matrix_id}_path_{self.current_tick_id % 1000}",
"score": score
}

        # Fallback to default matrix if no suitable matrix found
        if best_matrix is None:
best_matrix = {
"matrix_id": "SFS8-A5",
"bit_level": 8,
"phase_count": 42,
"routing_path": "default_path",
"score": 0.5
}

        return best_matrix

def _calculate_matrix_score(self, matrix_id: str, matrix_info: Dict[str, Any],]


                              vector_data: Dict[str, Any], validation_result: Any,
current_tick: TickMapState) -> float:
"""Calculate allocation score for a matrix"""
score = 0.0

        # Base score from validation result
score += validation_result.confidence_score * 0.4

        # Matrix weight from settings controller
matrix_weight = self.settings_controller.get_matrix_weight(matrix_id)
        score += matrix_weight * 0.3

        # Thermal compatibility
thermal_compatibility = 1.0 - unified_math.abs(current_tick.thermal_load - 0.5)
        score += thermal_compatibility * 0.1

        # Entropy compatibility
entropy_compatibility = 1.0 - unified_math.abs(current_tick.entropy_level - 0.5)
        score += entropy_compatibility * 0.1

        # Bit level compatibility
target_bit_level = vector_data.get("target_bit_level", 16)
        bit_compatibility = 1.0 - unified_math.abs(matrix_info["bit_level"] - target_bit_level) / 64.0
        score += bit_compatibility * 0.1

        # Matrix performance history
matrix_perf = self.matrix_performance.get(matrix_id, {})
        success_rate = matrix_perf.get("success_rate", 0.5)
        score += success_rate * 0.2

        return score

def _determine_execution_mode(self, validation_result: Any, best_matrix: Dict[str, Any]) -> str:


    pass
    pass
        """Determine execution mode based on validation result and matrix state"""
        if validation_result.recommended_action == "avoid":
            return "avoided"
        elif validation_result.recommended_action == "monitor":
            return "monitored"
        elif validation_result.confidence_score > 0.8:
            return "immediate"
        else:
            return "queued"

def _update_tick_map(self, allocation: MatrixAllocation):


    pass
    pass
        """Update tick map with new allocation"""
current_tick = self.tick_map[self.current_tick_id]

        # Update thermal load
current_tick.thermal_load = unified_math.min(1.0, current_tick.thermal_load + 0.1)

        # Update entropy level
current_tick.entropy_level = allocation.entropy_level

        # Update active matrices
        if allocation.matrix_id not in current_tick.active_matrices:
current_tick.active_matrices.append(allocation.matrix_id)

        # Update memory usage
current_tick.memory_usage = unified_math.min(1.0, current_tick.memory_usage + 0.05)

        # Update bit level and phase position
current_tick.bit_level = allocation.bit_level
current_tick.phase_position = (current_tick.phase_position + 1) % allocation.phase_count

        # Update timestamp
current_tick.last_update = datetime.now()

        # Advance tick counter
self.current_tick_id = (self.current_tick_id + 1) % self.tick_map_size

def _update_matrix_performance(self, allocation: MatrixAllocation):


    pass
    pass
        """Update matrix performance tracking"""
matrix_id = allocation.matrix_id

        if matrix_id not in self.matrix_performance:
self.matrix_performance[matrix_id] = {]
"total_allocations": 0,
"successful_allocations": 0,
"success_rate": 0.5,
"avg_confidence": 0.5,
"avg_priority_weight": 0.5,
"last_allocation": None
}

perf = self.matrix_performance[matrix_id]
perf["total_allocations"] += 1
perf["last_allocation"] = allocation.timestamp

        # Update success rate (assuming immediate execution is successful)
        if allocation.execution_mode == "immediate":
perf["successful_allocations"] += 1

perf["success_rate"] = perf["successful_allocations"] / perf["total_allocations"]

        # Update average confidence
current_avg_conf = perf["avg_confidence"]
perf["avg_confidence"] = (current_avg_conf * (perf["total_allocations"] - 1) +)
                                allocation.allocation_confidence) / perf["total_allocations"]

        # Update average priority weight
current_avg_weight = perf["avg_priority_weight"]
perf["avg_priority_weight"] = (current_avg_weight * (perf["total_allocations"] - 1) +)
                                     allocation.priority_weight) / perf["total_allocations"]

def get_matrix_status(self, matrix_id: str) -> Dict[str, Any]:


    pass
    pass
        """Get current status of a matrix"""
        if matrix_id not in self.registered_matrices:
            return {"status": "not_found"}

matrix_info = self.registered_matrices[matrix_id]
performance = self.matrix_performance.get(matrix_id, {})

        return {
"matrix_id": matrix_id,
"status": matrix_info["status"],
"bit_level": matrix_info["bit_level"],
"phase_count": matrix_info["phase_count"],
"performance": performance,
"current_tick": self.current_tick_id,
"thermal_load": self.tick_map[self.current_tick_id].thermal_load,
"entropy_level": self.tick_map[self.current_tick_id].entropy_level
}

def get_all_matrices_status(self) -> Dict[str, Dict[str, Any]]:


    pass
    pass
        """Get status of all matrices"""
status = {}
        for matrix_id in self.registered_matrices:
status[matrix_id] = self.get_matrix_status(matrix_id)
        return status

def get_tick_map_summary(self) -> Dict[str, Any]:


    pass
    pass
        """Get summary of tick map state"""
current_tick = self.tick_map[self.current_tick_id]

        return {
"current_tick_id": self.current_tick_id,
"tick_map_size": self.tick_map_size,
"thermal_load": current_tick.thermal_load,
"entropy_level": current_tick.entropy_level,
"bit_level": current_tick.bit_level,
"phase_position": current_tick.phase_position,
"active_matrices": current_tick.active_matrices,
"memory_usage": current_tick.memory_usage,
"last_update": current_tick.last_update.isoformat()
        }

def get_allocation_summary(self) -> Dict[str, Any]:


    pass
    pass
        """Get summary of allocation performance"""
total_allocations = len(self.allocation_history)

        if total_allocations == 0:
            return {"total_allocations": 0}

        # Calculate statistics
immediate_count = sum(1 for a in self.allocation_history if a.execution_mode == "immediate")
        queued_count = sum(1 for a in self.allocation_history if a.execution_mode == "queued")
        monitored_count = sum(1 for a in self.allocation_history if a.execution_mode == "monitored")
        avoided_count = sum(1 for a in self.allocation_history if a.execution_mode == "avoided")

avg_confidence = sum(a.allocation_confidence for a in self.allocation_history) / total_allocations
        avg_priority_weight = sum(a.priority_weight for a in self.allocation_history) / total_allocations

        return {
"total_allocations": total_allocations,
"execution_modes": {
"immediate": immediate_count,
"queued": queued_count,
"monitored": monitored_count,
"avoided": avoided_count
},
"average_confidence": avg_confidence,
"average_priority_weight": avg_priority_weight,
"matrix_performance": self.matrix_performance,
"last_allocation": self.allocation_history[-1].timestamp.isoformat() if self.allocation_history else None
        }

def set_matrix_status(self, matrix_id: str, status: str):


    pass
    pass
        """Set status of a matrix (active/inactive/maintenance)"""
        if matrix_id in self.registered_matrices:
self.registered_matrices[matrix_id]["status"] = status

def reset_tick_map(self):


    pass
    pass
        """Reset tick map to initial state"""
self._initialize_tick_map()
        self.current_tick_id = 0

def save_allocation_data(self, filepath: str = "allocation_data.json"):


    pass
    pass
        """Save allocation data to file"""
data = {
"allocation_history": [asdict(a) for a in self.allocation_history],
            "matrix_performance": self.matrix_performance,
"tick_map_summary": self.get_tick_map_summary(),
            "registered_matrices": self.registered_matrices,
"timestamp": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

def load_allocation_data(self, filepath: str = "allocation_data.json"):


    pass
    pass
        """Load allocation data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Load allocation history
self.allocation_history = [MatrixAllocation(**a) for a in data.get("allocation_history", [])]

            # Load performance data
self.matrix_performance = data.get("matrix_performance", {})

            # Load registered matrices
self.registered_matrices = data.get("registered_matrices", self.registered_matrices)

        except FileNotFoundError:
safe_print(f"Allocation data file {filepath} not found. Starting with empty data.")
        except Exception as e:
safe_print(f"Error loading allocation data: {e}")

def get_optimal_routing_path(self, vector_data: Dict[str, Any]) -> str:


    pass
    pass
        """Get optimal routing path for a vector"""
        # This would implement complex routing logic based on:
        # - Vector characteristics
        # - Matrix availability
        # - Current system state
        # - Historical performance

matrix_id = vector_data.get("matrix_id", "SFS8-A5")
        tick_id = self.current_tick_id

        return f"{matrix_id}_optimal_path_{tick_id % 1000}"

def should_override_fault_controller(self, matrix_id: str) -> bool:


    pass
    pass
        """Check if fault controller should be overridden for this matrix"""
        return self.settings_controller.should_override_fault_controller(matrix_id)


# Global matrix allocator instance
matrix_allocator = MatrixAllocator()


def get_matrix_allocator() -> MatrixAllocator:


    pass
    pass
    """Get the global matrix allocator instance"""
    return matrix_allocator


if __name__ == "__main__":
    pass
    pass
    # Test the matrix allocator
allocator = MatrixAllocator()

safe_print("=== Schwabot Matrix Allocator Test ===")

    # Test vector data
test_vector_data = {
'vector_id': 'test_vec_001',
'matrix_id': 'SFS8-A5',
'tick_id': 12345,
'entry_price': 50000.0,
'exit_price': 50100.0,
'entry_time': datetime.now().isoformat(),
        'exit_time': datetime.now().isoformat(),
        'success': True,
'profit_loss': 100.0,
'confidence': 0.8,
'volume_data': {'current': 1000000, 'average': 800000},
'ghost_signal_strength': 0.7,
'entropy_level': 0.3,
'target_bit_level': 16
}

    # Allocate vector
allocation = allocator.allocate_vector(test_vector_data)

safe_print(f"Vector ID: {test_vector_data['vector_id']}")
    safe_print(f"Allocated to Matrix: {allocation.matrix_id}")
    safe_print(f"Allocation Confidence: {allocation.allocation_confidence:.3f}")
    safe_print(f"Execution Mode: {allocation.execution_mode}")
    safe_print(f"Routing Path: {allocation.routing_path}")
    safe_print(f"Bit Level: {allocation.bit_level}")
    safe_print(f"Phase Count: {allocation.phase_count}")

    # Get summaries
tick_summary = allocator.get_tick_map_summary()
    allocation_summary = allocator.get_allocation_summary()

safe_print("\nTick Map Summary:")
    safe_print(f"Current Tick: {tick_summary['current_tick_id']}")
    safe_print(f"Thermal Load: {tick_summary['thermal_load']:.3f}")
    safe_print(f"Entropy Level: {tick_summary['entropy_level']:.3f}")
    safe_print(f"Active Matrices: {tick_summary['active_matrices']}")

safe_print("\nAllocation Summary:")
    safe_print(f"Total Allocations: {allocation_summary['total_allocations']}")
    safe_print(f"Execution Modes: {allocation_summary['execution_modes']}")
    safe_print(f"Average Confidence: {allocation_summary['average_confidence']:.3f}")

safe_print("Matrix allocator test completed!")
