# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import math
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
Hash Registry Core - Pure Mathematical Functions
===============================================

Pure mathematical functions for hash registry operations.
No external dependencies - only standard library imports.

Mathematical Functions:
- Hash ID generation (hash_00 to hash_31)
- Bit depth calculation (4, 8, 42-bit logic)
- Tensor route assignment (route_0 to route_4)
- Priority calculation (0.1 to 3.2 range)
- Hash resolution algorithms
- Basket mapping logic
"""

import hashlib
# from core.unified_math_system import unified_math  # F811: duplicate import
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class HashRegistryStructure(Enum):
    """Hash registry structure types."""
SIMPLIFIED = "simplified"  # 5-field structure
EXTENDED = "extended"      # Full strategy structure
DYNAMIC = "dynamic"        # Auto-generated structure

@dataclass
class HashRegistryEntry:
    """Hash registry entry with mathematical structure."""
hash_id: str
bit_depth: int
tensor_route: str
matrix_basket_id: int
priority: float
enabled: bool
metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
self.metadata = {}

class HashRegistryCore:
    """
Pure mathematical functions for hash registry operations.
No external dependencies - only mathematical calculations.
"""

    # Mathematical constants
TOTAL_ENTRIES = 32
BIT_DEPTHS = [4, 8, 42]
TENSOR_ROUTES = ["route_0", "route_1", "route_2", "route_3", "route_4"]
PRIORITY_MIN = 0.1
PRIORITY_MAX = 3.2

@staticmethod
    def generate_hash_id(index: int) -> str:
        """Generate hash ID: hash_XX where XX = 00-31."""
        if not 0 <= index < HashRegistryCore.TOTAL_ENTRIES:
            raise ValueError(f"Index must be 0-{HashRegistryCore.TOTAL_ENTRIES-1}")
        return f"hash_{index:02d}"

@staticmethod
    def calculate_bit_depth(index: int) -> int:
        """Calculate bit depth based on index: 4, 8, or 42-bit logic."""
        # Mathematical pattern: index % 3 determines bit depth
remainder = index % 3
        if remainder == 0:
            return 4
        elif remainder == 1:
            return 8
        else:
            return 42

@staticmethod
    def calculate_tensor_route(index: int) -> str:
        """Calculate tensor route: route_0 through route_4."""
route_index = index % len(HashRegistryCore.TENSOR_ROUTES)
        return HashRegistryCore.TENSOR_ROUTES[route_index]

@staticmethod
    def calculate_matrix_basket_id(index: int) -> int:
        """Calculate matrix basket ID: 0-31."""
        return index % HashRegistryCore.TOTAL_ENTRIES

@staticmethod
    def calculate_priority(index: int) -> float:
        """Calculate priority: 0.1 to 3.2 linear progression."""
        # Mathematical formula: min + (index * step)
        step = (HashRegistryCore.PRIORITY_MAX - HashRegistryCore.PRIORITY_MIN) / (HashRegistryCore.TOTAL_ENTRIES - 1)
        priority = HashRegistryCore.PRIORITY_MIN + (index * step)
        return round(priority, 1)  # Round to 1 decimal place

@staticmethod
    def determine_bit_depth_from_hash(hash_value: str) -> int:
        """Determine bit depth from hash value using first byte analysis."""
        try:
            # Use first byte to determine bit depth
first_byte = int(hash_value[0:2], 16)

            # Mathematical thresholds:
            # 0-84: 4-bit (85 values)
            # 85-169: 8-bit (85 values)
            # 170-255: 42-bit (86 values)
            if first_byte < 85:
                return 4
            elif first_byte < 170:
                return 8
            else:
                return 42

        except (ValueError, IndexError):
            return 8  # Default to 8-bit on error

@staticmethod
    def generate_basket_hash_signature(hash_id: str, bit_depth: int, tensor_route: str,
                                     matrix_basket_id: int, priority: float) -> str:
"""Generate hash signature for basket using SHA-256."""
content = f"{hash_id}_{bit_depth}_{tensor_route}_{matrix_basket_id}_{priority}"
        return hashlib.sha256(content.encode()).hexdigest()

@staticmethod
    def resolve_hash_to_basket(hash_value: str, bit_depth: Optional[int] = None) -> Optional[str]:
        """Resolve hash value to matrix basket ID."""
        try:
            # If bit_depth not specified, determine from hash
            if bit_depth is None:
bit_depth = HashRegistryCore.determine_bit_depth_from_hash(hash_value)

            # Find matching entry based on bit depth
            # For now, use a simple mapping based on bit depth
            if bit_depth == 4:
basket_id = 0
            elif bit_depth == 8:
basket_id = 1
            else:  # 42-bit
basket_id = 2

            return f"basket_{basket_id}"

        except Exception:
            return None

@staticmethod
    def generate_registry_entry(index: int) -> HashRegistryEntry:
        """Generate complete registry entry for given index."""
hash_id = HashRegistryCore.generate_hash_id(index)
        bit_depth = HashRegistryCore.calculate_bit_depth(index)
        tensor_route = HashRegistryCore.calculate_tensor_route(index)
        matrix_basket_id = HashRegistryCore.calculate_matrix_basket_id(index)
        priority = HashRegistryCore.calculate_priority(index)

        return HashRegistryEntry(
            hash_id=hash_id,
bit_depth=bit_depth,
tensor_route=tensor_route,
matrix_basket_id=matrix_basket_id,
priority=priority,
enabled=True


@staticmethod
    def generate_complete_registry() -> Dict[str, HashRegistryEntry]:
        """Generate complete 32-entry registry."""
registry = {}

        for i in range(HashRegistryCore.TOTAL_ENTRIES):
            entry = HashRegistryCore.generate_registry_entry(i)
            registry[entry.hash_id] = entry

        return registry

@staticmethod
    def get_entries_by_bit_depth(registry: Dict[str, HashRegistryEntry], bit_depth: int) -> List[HashRegistryEntry]:
        """Get all entries with specified bit depth."""
        return [entry for entry in registry.values() if entry.bit_depth == bit_depth]

@staticmethod
    def get_entries_by_route(registry: Dict[str, HashRegistryEntry], tensor_route: str) -> List[HashRegistryEntry]:
        """Get all entries with specified tensor route."""
        return [entry for entry in registry.values() if entry.tensor_route == tensor_route]

@staticmethod
    def get_entries_by_priority_range(registry: Dict[str, HashRegistryEntry],
                                    min_priority: float, max_priority: float) -> List[HashRegistryEntry]:
"""Get entries within priority range."""
        return [
entry for entry in registry.values()
            if min_priority <= entry.priority <= max_priority
]

@staticmethod
    def get_enabled_entries(registry: Dict[str, HashRegistryEntry]) -> List[HashRegistryEntry]:
        """Get all enabled entries."""
        return [entry for entry in registry.values() if entry.enabled]

@staticmethod
    def get_disabled_entries(registry: Dict[str, HashRegistryEntry]) -> List[HashRegistryEntry]:
        """Get all disabled entries."""
        return [entry for entry in registry.values() if not entry.enabled]

@staticmethod
    def get_best_matching_hash(registry: Dict[str, HashRegistryEntry],
                             bit_depth: int, tensor_route: Optional[str] = None,
min_priority: float = 0.0) -> Optional[HashRegistryEntry]:
"""Get best matching hash entry based on criteria."""
candidates = []

        for entry in registry.values():
            if not entry.enabled:
                continue

            if entry.bit_depth != bit_depth:
                continue

            if tensor_route and entry.tensor_route != tensor_route:
                continue

            if entry.priority < min_priority:
                continue

candidates.append(entry)

        if not candidates:
            return None

        # Return highest priority candidate
        return unified_math.max(candidates, key=lambda x: x.priority)

@staticmethod
    def calculate_registry_statistics(registry: Dict[str, HashRegistryEntry]) -> Dict:
        """Calculate comprehensive registry statistics."""
total_entries = len(registry)
        enabled_entries = len(HashRegistryCore.get_enabled_entries(registry))
        disabled_entries = len(HashRegistryCore.get_disabled_entries(registry))

        # Bit depth distribution
bit_depth_dist = {}
        for entry in registry.values():
            bit_depth = entry.bit_depth
bit_depth_dist[bit_depth] = bit_depth_dist.get(bit_depth, 0) + 1

        # Route distribution
route_dist = {}
        for entry in registry.values():
            route = entry.tensor_route
route_dist[route] = route_dist.get(route, 0) + 1

        # Priority statistics
priorities = [entry.priority for entry in registry.values()]
        avg_priority = sum(priorities) / len(priorities) if priorities else 0

        return {
"total_entries": total_entries,
"enabled_entries": enabled_entries,
"disabled_entries": disabled_entries,
"bit_depth_distribution": bit_depth_dist,
"route_distribution": route_dist,
"priority_statistics": {
"average": avg_priority,
"min": unified_math.min(priorities) if priorities else 0,
                "max": unified_math.max(priorities) if priorities else 0
            }
}


def main():
    """Test the pure mathematical functions."""
safe_print("🔢 Hash Registry Core - Mathematical Functions Test")
    safe_print("=" * 50)

    # Test hash ID generation
safe_print("Hash ID Generation:")
    for i in range(5):
        hash_id = HashRegistryCore.generate_hash_id(i)
        safe_print(f"  Index {i} -> {hash_id}")

    # Test bit depth calculation
safe_print("\nBit Depth Calculation:")
    for i in range(9):
        bit_depth = HashRegistryCore.calculate_bit_depth(i)
        safe_print(f"  Index {i} -> {bit_depth}-bit")

    # Test tensor route calculation
safe_print("\nTensor Route Calculation:")
    for i in range(10):
        route = HashRegistryCore.calculate_tensor_route(i)
        safe_print(f"  Index {i} -> {route}")

    # Test priority calculation
safe_print("\nPriority Calculation:")
    for i in range(5):
        priority = HashRegistryCore.calculate_priority(i)
        safe_print(f"  Index {i} -> {priority}")

    # Test complete registry generation
safe_print("\nComplete Registry Generation:")
    registry = HashRegistryCore.generate_complete_registry()
    safe_print(f"  Generated {len(registry)} entries")

    # Test statistics
stats = HashRegistryCore.calculate_registry_statistics(registry)
    safe_print("\nRegistry Statistics:")
    safe_print(f"  Total entries: {stats['total_entries']}")
    safe_print(f"  Enabled entries: {stats['enabled_entries']}")
    safe_print(f"  Bit depth distribution: {stats['bit_depth_distribution']}")

safe_print("\n✅ Hash Registry Core test completed")


if __name__ == "__main__":
main()
