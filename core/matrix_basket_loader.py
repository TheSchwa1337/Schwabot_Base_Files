# Import safe print for Windows compatibility
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import math
try:
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""
Matrix Basket Loader - Schwabot UROS v1.0
========================================

Matrix basket loading system integrated with the 32-entry hash registry scaffold.
Provides dynamic basket loading, recursive trigger functionality, and seamless integration.

Mathematical Foundation:
- Basket ID Resolution: basket_id = matrix_basket_id from hash registry
- Bit Depth Mapping: 4-bit, 8-bit, 42-bit → BitPhase enum
- Tensor Route Assignment: route_0 to route_4 → tensor operations
- Priority-Based Loading: 0.1 to 3.2 priority system
- Enabled/Disabled State Management
"""

import json
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import os
import sys
import threading
import queue

# Add core directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from math.tensor_algebra import UnifiedTensorAlgebra, BitPhaseResult
from matrix_mapper import MatrixMapper, BitPhase, MatrixBasket, BasketType
from tensor_matcher import TensorMatcher
from bit_resolution_engine import BitResolutionEngine
from hash_registry_manager import HashRegistryManager, HashRegistryEntry

logger = logging.getLogger(__name__)

class BasketLoadTrigger(Enum):
    """Basket load trigger types."""
HASH_RESOLUTION = "hash_resolution"
PRIORITY_UPDATE = "priority_update"
ROUTE_CHANGE = "route_change"
ENABLE_TOGGLE = "enable_toggle"
MANUAL_LOAD = "manual_load"
AUTO_REFRESH = "auto_refresh"

@dataclass
class BasketLoadRequest:
    """Basket load request with trigger information."""
basket_id: str
hash_id: str
bit_depth: int
tensor_route: str
priority: float
trigger: BasketLoadTrigger
timestamp: datetime
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BasketLoadResult:
    """Result of basket loading operation."""
success: bool
basket_id: str
hash_id: str
basket: Optional[MatrixBasket] = None
error_message: Optional[str] = None
load_time: float = 0.0
metadata: Dict[str, Any] = field(default_factory=dict)

class MatrixBasketLoader:
    """
Matrix Basket Loader for Schwabot UROS v1.0.

Integrates with the 32-entry hash registry scaffold to provide:
- Dynamic basket loading based on hash registry entries
- Recursive trigger functionality for basket updates
- Priority-based basket selection
- Enabled/disabled state management
- Seamless integration with matrix mapper
"""

    def __init__(self, hash_registry_manager: HashRegistryManager = None):
        """Initialize matrix basket loader."""
        # Core components
self.hash_registry_manager = hash_registry_manager or HashRegistryManager()
        self.matrix_mapper = MatrixMapper()
        self.tensor_algebra = UnifiedTensorAlgebra()
        self.tensor_matcher = TensorMatcher()
        self.bit_resolution_engine = BitResolutionEngine()

        # Basket management
self.loaded_baskets: Dict[str, MatrixBasket] = {}
self.basket_load_history: List[BasketLoadResult] = []
self.active_baskets: Dict[str, bool] = {}

        # Trigger system
self.load_triggers: Dict[BasketLoadTrigger, List[Callable]] = {
trigger: [] for trigger in BasketLoadTrigger
}
self.trigger_queue = queue.Queue()
        self.trigger_thread = None
self.trigger_running = False

        # Performance tracking
self.load_stats: Dict[str, int] = {}
self.load_times: Dict[str, List[float]] = {}

        # Integration setup
self._setup_integrations()
        self._start_trigger_system()
        logger.info("Matrix Basket Loader initialized")

    def _setup_integrations(self) -> None:
        """Setup integrations with other components."""
        try:
            # Integrate hash registry with matrix mapper
self.hash_registry_manager.integrate_with_matrix_mapper(self.matrix_mapper)

            # Setup tensor matcher integrations
self.tensor_matcher.set_bit_phase_engine(self.bit_resolution_engine)
            self.tensor_matcher.set_matrix_mapper(self.matrix_mapper)

logger.info("Matrix basket loader integrations setup complete")

        except Exception as e:
logger.error(f"Error setting up integrations: {e}")

    def _start_trigger_system(self) -> None:
        """Start the trigger processing system."""
        try:
self.trigger_running = True
self.trigger_thread = threading.Thread(target=self._process_triggers, daemon=True)
            self.trigger_thread.start()
            logger.info("Trigger system started")

        except Exception as e:
logger.error(f"Error starting trigger system: {e}")

    def _process_triggers(self) -> None:
        """Process trigger queue in background thread."""
        while self.trigger_running:
            try:
                # Get trigger from queue with timeout
trigger_data = self.trigger_queue.get(timeout=1.0)

                if trigger_data:
self._execute_trigger(trigger_data)

            except queue.Empty:
                continue
            except Exception as e:
logger.error(f"Error processing trigger: {e}")

    def _execute_trigger(self, trigger_data: Dict[str, Any]) -> None:
        """Execute a trigger action."""
        try:
trigger_type = trigger_data.get('trigger_type')
            basket_id = trigger_data.get('basket_id')
            hash_id = trigger_data.get('hash_id')

            if trigger_type == BasketLoadTrigger.HASH_RESOLUTION:
self._handle_hash_resolution_trigger(basket_id, hash_id)
            elif trigger_type == BasketLoadTrigger.PRIORITY_UPDATE:
self._handle_priority_update_trigger(basket_id, hash_id)
            elif trigger_type == BasketLoadTrigger.ROUTE_CHANGE:
self._handle_route_change_trigger(basket_id, hash_id)
            elif trigger_type == BasketLoadTrigger.ENABLE_TOGGLE:
self._handle_enable_toggle_trigger(basket_id, hash_id)
            elif trigger_type == BasketLoadTrigger.AUTO_REFRESH:
self._handle_auto_refresh_trigger()

        except Exception as e:
logger.error(f"Error executing trigger: {e}")

    def load_basket_from_registry(self, hash_id: str, trigger: BasketLoadTrigger = BasketLoadTrigger.MANUAL_LOAD) -> BasketLoadResult:
        """Load basket from hash registry entry."""
        try:
start_time = time.time()

            # Get hash registry entry
entry = self.hash_registry_manager.get_hash_entry(hash_id)
            if not entry:
                return BasketLoadResult(
                    success=False,
basket_id=f"basket_{hash_id}",
hash_id=hash_id,
error_message=f"Hash registry entry not found: {hash_id}"


            if not entry.enabled:
                return BasketLoadResult(
                    success=False,
basket_id=f"basket_{entry.matrix_basket_id}",
hash_id=hash_id,
error_message=f"Hash registry entry disabled: {hash_id}"


            # Generate basket ID
basket_id = f"basket_{entry.matrix_basket_id}"

            # Check if basket already loaded
            if basket_id in self.loaded_baskets:
                return BasketLoadResult(
                    success=True,
basket_id=basket_id,
hash_id=hash_id,
basket=self.loaded_baskets[basket_id],
load_time=time.time() - start_time


            # Create basket load request
request = BasketLoadRequest(
                basket_id=basket_id,
hash_id=hash_id,
bit_depth=entry.bit_depth,
tensor_route=entry.tensor_route,
priority=entry.priority,
trigger=trigger,
timestamp=datetime.now()


            # Load basket
basket = self._create_basket_from_entry(entry)
            if not basket:
                return BasketLoadResult(
                    success=False,
basket_id=basket_id,
hash_id=hash_id,
error_message="Failed to create basket from registry entry"


            # Store basket
self.loaded_baskets[basket_id] = basket
self.active_baskets[basket_id] = True

            # Update statistics
self.load_stats[hash_id] = self.load_stats.get(hash_id, 0) + 1
            if hash_id not in self.load_times:
self.load_times[hash_id] = []
self.load_times[hash_id].append(time.time() - start_time)

            # Create result
result = BasketLoadResult(
                success=True,
basket_id=basket_id,
hash_id=hash_id,
basket=basket,
load_time=time.time() - start_time


            # Add to history
self.basket_load_history.append(result)

            # Trigger callbacks
self._trigger_callbacks(trigger, result)

logger.info(f"Loaded basket {basket_id} from hash registry entry {hash_id}")
            return result

        except Exception as e:
logger.error(f"Error loading basket from registry: {e}")
            return BasketLoadResult(
                success=False,
basket_id=f"basket_{hash_id}",
hash_id=hash_id,
error_message=str(e)


    def _create_basket_from_entry(self, entry: HashRegistryEntry) -> Optional[MatrixBasket]:
        """Create matrix basket from hash registry entry."""
        try:
            # Convert bit depth to BitPhase
bit_phase = BitPhase(entry.bit_depth)

            # Determine tensor dimensions based on bit depth
            if entry.bit_depth == 4:
tensor_dimensions = [2, 2, 2]
            elif entry.bit_depth == 8:
tensor_dimensions = [4, 4, 4]
            else:  # 42-bit
tensor_dimensions = [8, 8, 8]

            # Generate asset weights based on tensor route
asset_weights = self._generate_asset_weights_from_route(entry.tensor_route)

            # Generate sequence vector
sequence_vector = self._generate_sequence_vector(tensor_dimensions, entry.hash_id)

            # Calculate modulation factor based on priority
modulation_factor = entry.priority / 3.2  # Normalize to 0-1 range

            # Calculate resonance score
resonance_score = self._calculate_resonance_score(asset_weights, sequence_vector, entry.priority)

            # Generate hash signature
hash_signature = self._generate_basket_hash_signature(entry)

            # Create basket
basket = MatrixBasket(
                basket_id=f"basket_{entry.matrix_basket_id}",
basket_type=BasketType.STANDARD,
bit_phase=bit_phase,
tensor_dimensions=tensor_dimensions,
asset_weights=asset_weights,
sequence_vector=sequence_vector,
modulation_factor=modulation_factor,
resonance_score=resonance_score,
hash_signature=hash_signature,
timestamp=datetime.now(),
                performance_metrics={
'creation_tick': int(time.time()),
                    'creation_price': 50000.0,  # Default BTC price
'total_trades': 0,
'total_profit': 0.0,
'hash_id': entry.hash_id,
'tensor_route': entry.tensor_route,
'priority': entry.priority
}


            return basket

        except Exception as e:
logger.error(f"Error creating basket from entry: {e}")
            return None

    def _generate_asset_weights_from_route(self, tensor_route: str) -> Dict[str, float]:
        """Generate asset weights based on tensor route."""
        try:
            # Base asset weights
base_weights = {
'BTC': 0.4,
'ETH': 0.25,
'USDC': 0.2,
'XRP': 0.1,
'SOL': 0.05
}

            # Adjust weights based on route
route_adjustments = {
'route_0': {'BTC': 0.1, 'ETH': 0.1},  # BTC/ETH focused
'route_1': {'USDC': 0.1, 'XRP': 0.1},  # USDC/XRP focused
'route_2': {'ETH': 0.1, 'SOL': 0.1},   # ETH/SOL focused
'route_3': {'BTC': 0.1, 'USDC': 0.1},  # BTC/USDC focused
'route_4': {'XRP': 0.1, 'SOL': 0.1}    # XRP/SOL focused
}

            if tensor_route in route_adjustments:
                for asset, adjustment in route_adjustments[tensor_route].items():
                    base_weights[asset] += adjustment

                    # Reduce other weights proportionally
total_adjustment = adjustment
other_assets = [a for a in base_weights.keys() if a != asset]
                    reduction_per_asset = total_adjustment / len(other_assets)

                    for other_asset in other_assets:
base_weights[other_asset] = unified_math.max(0.01, base_weights[other_asset] - reduction_per_asset)

            # Normalize weights
total_weight = sum(base_weights.values())
            normalized_weights = {asset: weight / total_weight for asset, weight in base_weights.items()}

            return normalized_weights

        except Exception as e:
logger.error(f"Error generating asset weights: {e}")
            return {'BTC': 1.0}

    def _generate_sequence_vector(self, tensor_dimensions: List[int], hash_id: str) -> List[float]:
        """Generate sequence vector for basket."""
        try:
            # Use hash_id to generate deterministic sequence
hash_bytes = hashlib.sha256(hash_id.encode()).digest()

            # Calculate total elements needed
total_elements = 1
            for dim in tensor_dimensions:
total_elements *= dim

            # Generate sequence vector
sequence_vector = []
            for i in range(unified_math.min(total_elements, 64)):  # Limit to 64 elements
                byte_value = hash_bytes[i % len(hash_bytes)]
                normalized_value = byte_value / 255.0
sequence_vector.append(normalized_value)

            # Pad if needed
            while len(sequence_vector) < total_elements:
                sequence_vector.append(0.5)  # Default value

            return sequence_vector[:total_elements]

        except Exception as e:
logger.error(f"Error generating sequence vector: {e}")
            return [0.5] * 8  # Default fallback

    def _calculate_resonance_score(self, asset_weights: Dict[str, float], sequence_vector: List[float], priority: float) -> float:
        """Calculate resonance score for basket."""
        try:
            # Base resonance from asset diversity
asset_diversity = len(asset_weights) / 5.0  # Normalize by max expected assets

            # Sequence vector coherence
sequence_coherence = sum(sequence_vector) / len(sequence_vector)

            # Priority influence
priority_influence = priority / 3.2  # Normalize priority

            # Calculate resonance score
resonance_score = (asset_diversity * 0.3 +
                             sequence_coherence * 0.4 +
priority_influence * 0.3)

            return unified_math.min(1.0, unified_math.max(0.0, resonance_score))

        except Exception as e:
logger.error(f"Error calculating resonance score: {e}")
            return 0.5

    def _generate_basket_hash_signature(self, entry: HashRegistryEntry) -> str:
        """Generate hash signature for basket."""
        try:
content = f"{entry.hash_id}_{entry.bit_depth}_{entry.tensor_route}_{entry.matrix_basket_id}_{entry.priority}"
            return hashlib.sha256(content.encode()).hexdigest()

        except Exception as e:
logger.error(f"Error generating basket hash signature: {e}")
            return hashlib.sha256(str(time.time()).encode()).hexdigest()

    def load_baskets_by_bit_depth(self, bit_depth: int) -> List[BasketLoadResult]:
        """Load all baskets with specified bit depth."""
        try:
results = []
entries = self.hash_registry_manager.get_entries_by_bit_depth(bit_depth)

            for entry in entries:
                if entry.enabled:
result = self.load_basket_from_registry(entry.hash_id, BasketLoadTrigger.MANUAL_LOAD)
                    results.append(result)

logger.info(f"Loaded {len(results)} baskets with bit depth {bit_depth}")
            return results

        except Exception as e:
logger.error(f"Error loading baskets by bit depth: {e}")
            return []

    def load_baskets_by_route(self, tensor_route: str) -> List[BasketLoadResult]:
        """Load all baskets with specified tensor route."""
        try:
results = []
entries = self.hash_registry_manager.get_entries_by_route(tensor_route)

            for entry in entries:
                if entry.enabled:
result = self.load_basket_from_registry(entry.hash_id, BasketLoadTrigger.MANUAL_LOAD)
                    results.append(result)

logger.info(f"Loaded {len(results)} baskets with route {tensor_route}")
            return results

        except Exception as e:
logger.error(f"Error loading baskets by route: {e}")
            return []

    def load_baskets_by_priority_range(self, min_priority: float, max_priority: float) -> List[BasketLoadResult]:
        """Load baskets within priority range."""
        try:
results = []
entries = self.hash_registry_manager.get_entries_by_priority_range(min_priority, max_priority)

            for entry in entries:
                if entry.enabled:
result = self.load_basket_from_registry(entry.hash_id, BasketLoadTrigger.MANUAL_LOAD)
                    results.append(result)

logger.info(f"Loaded {len(results)} baskets with priority range {min_priority}-{max_priority}")
            return results

        except Exception as e:
logger.error(f"Error loading baskets by priority range: {e}")
            return []

    def load_all_enabled_baskets(self) -> List[BasketLoadResult]:
        """Load all enabled baskets from registry."""
        try:
results = []
entries = self.hash_registry_manager.get_enabled_entries()

            for entry in entries:
result = self.load_basket_from_registry(entry.hash_id, BasketLoadTrigger.MANUAL_LOAD)
                results.append(result)

logger.info(f"Loaded {len(results)} enabled baskets")
            return results

        except Exception as e:
logger.error(f"Error loading all enabled baskets: {e}")
            return []

    def unload_basket(self, basket_id: str) -> bool:
        """Unload a basket from memory."""
        try:
            if basket_id in self.loaded_baskets:
                del self.loaded_baskets[basket_id]
self.active_baskets[basket_id] = False
logger.info(f"Unloaded basket: {basket_id}")
                return True
            return False

        except Exception as e:
logger.error(f"Error unloading basket {basket_id}: {e}")
            return False

    def get_loaded_basket(self, basket_id: str) -> Optional[MatrixBasket]:
        """Get loaded basket by ID."""
        return self.loaded_baskets.get(basket_id)

    def get_active_baskets(self) -> Dict[str, MatrixBasket]:
        """Get all active baskets."""
        return {basket_id: basket for basket_id, basket in self.loaded_baskets.items()
                if self.active_baskets.get(basket_id, False)}

    def add_trigger_callback(self, trigger: BasketLoadTrigger, callback: Callable) -> None:
        """Add callback for trigger events."""
        try:
            if trigger not in self.load_triggers:
self.load_triggers[trigger] = []
self.load_triggers[trigger].append(callback)
            logger.info(f"Added callback for trigger: {trigger}")

        except Exception as e:
logger.error(f"Error adding trigger callback: {e}")

    def _trigger_callbacks(self, trigger: BasketLoadTrigger, result: BasketLoadResult) -> None:
        """Trigger callbacks for a specific trigger type."""
        try:
            if trigger in self.load_triggers:
                for callback in self.load_triggers[trigger]:
                    try:
callback(result)
                    except Exception as e:
logger.error(f"Error in trigger callback: {e}")

        except Exception as e:
logger.error(f"Error triggering callbacks: {e}")

    def _handle_hash_resolution_trigger(self, basket_id: str, hash_id: str) -> None:
        """Handle hash resolution trigger."""
        try:
            # Reload basket with updated hash resolution
self.load_basket_from_registry(hash_id, BasketLoadTrigger.HASH_RESOLUTION)

        except Exception as e:
logger.error(f"Error handling hash resolution trigger: {e}")

    def _handle_priority_update_trigger(self, basket_id: str, hash_id: str) -> None:
        """Handle priority update trigger."""
        try:
            # Reload basket with updated priority
self.load_basket_from_registry(hash_id, BasketLoadTrigger.PRIORITY_UPDATE)

        except Exception as e:
logger.error(f"Error handling priority update trigger: {e}")

    def _handle_route_change_trigger(self, basket_id: str, hash_id: str) -> None:
        """Handle route change trigger."""
        try:
            # Reload basket with updated route
self.load_basket_from_registry(hash_id, BasketLoadTrigger.ROUTE_CHANGE)

        except Exception as e:
logger.error(f"Error handling route change trigger: {e}")

    def _handle_enable_toggle_trigger(self, basket_id: str, hash_id: str) -> None:
        """Handle enable toggle trigger."""
        try:
entry = self.hash_registry_manager.get_hash_entry(hash_id)
            if entry and entry.enabled:
                # Load basket if enabled
self.load_basket_from_registry(hash_id, BasketLoadTrigger.ENABLE_TOGGLE)
            else:
                # Unload basket if disabled
self.unload_basket(basket_id)

        except Exception as e:
logger.error(f"Error handling enable toggle trigger: {e}")

    def _handle_auto_refresh_trigger(self) -> None:
        """Handle auto refresh trigger."""
        try:
            # Reload all active baskets
active_basket_ids = list(self.active_baskets.keys())
            for basket_id in active_basket_ids:
                if self.active_baskets.get(basket_id, False):
                    # Find corresponding hash_id
                    for hash_id, entry in self.hash_registry_manager.hash_entries.items():
                        if f"basket_{entry.matrix_basket_id}" == basket_id:
self.load_basket_from_registry(hash_id, BasketLoadTrigger.AUTO_REFRESH)
                            break

        except Exception as e:
logger.error(f"Error handling auto refresh trigger: {e}")

    def get_loader_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loader statistics."""
        try:
total_loaded = len(self.loaded_baskets)
            active_baskets = len(self.get_active_baskets())

            # Bit depth distribution
bit_depth_dist = {}
            for basket in self.loaded_baskets.values():
                bit_depth = basket.bit_phase.value
bit_depth_dist[bit_depth] = bit_depth_dist.get(bit_depth, 0) + 1

            # Route distribution
route_dist = {}
            for basket in self.loaded_baskets.values():
                route = basket.performance_metrics.get('tensor_route', 'unknown')
                route_dist[route] = route_dist.get(route, 0) + 1

            # Load time statistics
all_load_times = []
            for times in self.load_times.values():
                all_load_times.extend(times)

avg_load_time = sum(all_load_times) / len(all_load_times) if all_load_times else 0

            return {
"total_loaded_baskets": total_loaded,
"active_baskets": active_baskets,
"bit_depth_distribution": bit_depth_dist,
"route_distribution": route_dist,
"load_statistics": {
"total_loads": sum(self.load_stats.values()),
                    "average_load_time": avg_load_time,
"most_loaded_hash": unified_math.max(self.load_stats.items(), key=lambda x: x[1])[0] if self.load_stats else None
                },
"trigger_statistics": {
"total_triggers": len(self.basket_load_history),
                    "successful_loads": sum(1 for result in self.basket_load_history if result.success),
                    "failed_loads": sum(1 for result in self.basket_load_history if not result.success)
                }
}

        except Exception as e:
logger.error(f"Error getting loader statistics: {e}")
            return {"error": str(e)}

    def export_loader_summary(self, output_path: str = "matrix_basket_loader_summary.json") -> None:
        """Export loader summary to JSON file."""
        try:
summary = {
"loader_info": {
"total_loaded_baskets": len(self.loaded_baskets),
                    "active_baskets": len(self.get_active_baskets()),
                    "total_load_history": len(self.basket_load_history)
                },
"loaded_baskets": {
basket_id: {
"hash_id": basket.performance_metrics.get('hash_id'),
                        "bit_depth": basket.bit_phase.value,
"tensor_route": basket.performance_metrics.get('tensor_route'),
                        "priority": basket.performance_metrics.get('priority'),
                        "resonance_score": basket.resonance_score,
"modulation_factor": basket.modulation_factor,
"asset_count": len(basket.asset_weights),
                        "active": self.active_baskets.get(basket_id, False)
                    }
                    for basket_id, basket in self.loaded_baskets.items()
                },
"statistics": self.get_loader_statistics()
            }

            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)

logger.info(f"Loader summary exported to {output_path}")

        except Exception as e:
logger.error(f"Error exporting loader summary: {e}")


def main():
    """Main function for matrix basket loader testing."""
safe_print("📦 Matrix Basket Loader - Schwabot UROS v1.0")
    safe_print("=" * 50)

    # Initialize loader
loader = MatrixBasketLoader()

    # Test loading by bit depth
safe_print("\n🔍 Testing bit depth loading...")
    results_4bit = loader.load_baskets_by_bit_depth(4)
    results_8bit = loader.load_baskets_by_bit_depth(8)
    results_42bit = loader.load_baskets_by_bit_depth(42)

safe_print(f"4-bit baskets loaded: {len(results_4bit)}")
    safe_print(f"8-bit baskets loaded: {len(results_8bit)}")
    safe_print(f"42-bit baskets loaded: {len(results_42bit)}")

    # Test loading by route
safe_print("\n🛣️ Testing route loading...")
    results_route_0 = loader.load_baskets_by_route("route_0")
    safe_print(f"Route 0 baskets loaded: {len(results_route_0)}")

    # Test priority range loading
safe_print("\n⚖️ Testing priority range loading...")
    results_high_priority = loader.load_baskets_by_priority_range(2.0, 3.2)
    safe_print(f"High priority baskets loaded: {len(results_high_priority)}")

    # Test individual basket loading
safe_print("\n🎯 Testing individual basket loading...")
    result = loader.load_basket_from_registry("hash_10")
    if result.success:
safe_print(f"Successfully loaded basket: {result.basket_id}")
        safe_print(f"Bit depth: {result.basket.bit_phase.value}")
        safe_print(f"Resonance score: {result.basket.resonance_score:.4f}")

    # Get statistics
stats = loader.get_loader_statistics()
    safe_print(f"\n📊 Loader statistics: {stats}")

    # Export summary
loader.export_loader_summary()

safe_print("\n✅ Matrix Basket Loader test completed")


if __name__ == "__main__":
main()
