from bit_resolution_engine import BitResolutionEngine
from tensor_matcher import TensorMatcher
from matrix_mapper import MatrixMapper, BitPhase
from math.tensor_algebra import UnifiedTensorAlgebra, BitPhaseResult
from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Hash Registry Manager - Schwabot UROS v1.0
=========================================

Comprehensive hash registry management system for the 32-entry scaffold structure.
Provides dynamic generation, matrix basket loading, and recursive trigger functionality.

Mathematical Foundation:
- Hash ID Structure: hash_XX where XX = 00-31
- Bit Depth Range: 4-bit, 8-bit, 42-bit logic
- Tensor Routes: route_0 through route_4
- Matrix Basket IDs: 0-31
- Priority System: 0.1 to 3.2 with enabled/disabled states
"""

import json
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import os
import sys

# Add core directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


logger = logging.getLogger(__name__)


class HashRegistryStructure(Enum):
    """Hash registry structure types."""
    SIMPLIFIED = "simplified"  # 5-field structure
    EXTENDED = "extended"      # Full strategy structure
    DYNAMIC = "dynamic"        # Auto-generated structure


@dataclass
class HashRegistryEntry:
    """Hash registry entry with simplified structure."""
    hash_id: str
    bit_depth: int
    tensor_route: str
    matrix_basket_id: int
    priority: float
    enabled: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HashRegistryConfig:
    """Hash registry configuration."""
    total_entries: int = 32
    bit_depths: List[int] = field(default_factory=lambda: [4, 8, 42])
    tensor_routes: List[str] = field(default_factory=lambda: ["route_0", "route_1", "route_2", "route_3", "route_4"])
    priority_range: Tuple[float, float] = (0.1, 3.2)
    auto_generate: bool = True
    dynamic_updates: bool = True


class HashRegistryManager:
    """
    Hash Registry Manager for Schwabot UROS v1.0.

    Manages the 32-entry hash registry scaffold with:
    - 4-bit to 42-bit range logic
    - Hash ID naming structure (hash_00 to hash_31)
    - Basket IDs (0-31)
    - Route logic (route_0 to route_4)
    - Bit prioritization (0.1 to 3.2)
    - Enabled/disabled switch
    """

    def __init__(self, registry_path: str = "core/hash_registry.json"):
        """Initialize hash registry manager."""
        self.registry_path = registry_path
        self.config = HashRegistryConfig()

        # Core components
        self.tensor_algebra = UnifiedTensorAlgebra()
        self.matrix_mapper = MatrixMapper()
        self.tensor_matcher = TensorMatcher()
        self.bit_resolution_engine = BitResolutionEngine()

        # Registry data
        self.registry_data: Dict[str, Dict[str, Any]] = {}
        self.hash_entries: Dict[str, HashRegistryEntry] = {}
        self.basket_mappings: Dict[int, str] = {}
        self.route_mappings: Dict[str, List[str]] = {}

        # Performance tracking
        self.usage_stats: Dict[str, int] = {}
        self.performance_metrics: Dict[str, float] = {}

        # Load or generate registry
        self._load_or_generate_registry()
        logger.info("Hash Registry Manager initialized")

    def _load_or_generate_registry(self) -> None:
        """Load existing registry or generate new one."""
        try:
            if os.path.exists(self.registry_path):
                self._load_registry()
                logger.info(f"Loaded existing hash registry from {self.registry_path}")
            else:
                self._generate_registry()
                self._save_registry()
                logger.info(f"Generated new hash registry with {self.config.total_entries} entries")

        except Exception as e:
            logger.error(f"Error loading/generating registry: {e}")
            self._generate_fallback_registry()

    def _load_registry(self) -> None:
        """Load hash registry from JSON file."""
        try:
            with open(self.registry_path, 'r') as f:
                self.registry_data = json.load(f)

            # Parse entries
            for hash_id, entry_data in self.registry_data.items():
                entry = HashRegistryEntry(
                    hash_id=hash_id,
                    bit_depth=entry_data.get('bit_depth', 8),
                    tensor_route=entry_data.get('tensor_route', 'route_0'),
                    matrix_basket_id=entry_data.get('matrix_basket_id', 0),
                    priority=entry_data.get('priority', 1.0),
                    enabled=entry_data.get('enabled', True),
                    metadata=entry_data.get('metadata', {})
                )
                self.hash_entries[hash_id] = entry

                # Build mappings
                self.basket_mappings[entry.matrix_basket_id] = hash_id
                if entry.tensor_route not in self.route_mappings:
                    self.route_mappings[entry.tensor_route] = []
                self.route_mappings[entry.tensor_route].append(hash_id)

        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            raise

    def _generate_registry(self) -> None:
        """Generate new hash registry with 32 entries."""
        try:
            self.registry_data = {}
            self.hash_entries = {}
            self.basket_mappings = {}
            self.route_mappings = {}

            # Generate 32 entries
            for i in range(self.config.total_entries):
                hash_id = f"hash_{i:02d}"

                # Determine bit depth (4, 8, or 42)
                if i % 3 == 0:
                    bit_depth = 4
                elif i % 3 == 1:
                    bit_depth = 8
                else:
                    bit_depth = 42

                # Determine tensor route (route_0 to route_4)
                tensor_route = f"route_{i % 5}"

                # Matrix basket ID (0-31)
                matrix_basket_id = i

                # Priority (0.1 to 3.2)
                priority = 0.1 + (i * 0.1)

                # All enabled by default
                enabled = True

                # Create entry
                entry_data = {
                    "bit_depth": bit_depth,
                    "tensor_route": tensor_route,
                    "matrix_basket_id": matrix_basket_id,
                    "priority": priority,
                    "enabled": enabled
                }

                self.registry_data[hash_id] = entry_data

                # Create HashRegistryEntry
                entry = HashRegistryEntry(
                    hash_id=hash_id,
                    bit_depth=bit_depth,
                    tensor_route=tensor_route,
                    matrix_basket_id=matrix_basket_id,
                    priority=priority,
                    enabled=enabled
                )

                self.hash_entries[hash_id] = entry
                self.basket_mappings[matrix_basket_id] = hash_id

                if tensor_route not in self.route_mappings:
                    self.route_mappings[tensor_route] = []
                self.route_mappings[tensor_route].append(hash_id)

        except Exception as e:
            logger.error(f"Error generating registry: {e}")
            raise

    def _generate_fallback_registry(self) -> None:
        """Generate fallback registry if loading fails."""
        logger.warning("Generating fallback hash registry")
        self.config.total_entries = 8  # Reduced for fallback
        self._generate_registry()

    def _save_registry(self) -> None:
        """Save hash registry to JSON file."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry_data, f, indent=2)
            logger.info(f"Hash registry saved to {self.registry_path}")

        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def get_hash_entry(self, hash_id: str) -> Optional[HashRegistryEntry]:
        """Get hash registry entry by ID."""
        return self.hash_entries.get(hash_id)

    def get_entries_by_bit_depth(self, bit_depth: int) -> List[HashRegistryEntry]:
        """Get all entries with specified bit depth."""
        return [entry for entry in self.hash_entries.values() if entry.bit_depth == bit_depth]

    def get_entries_by_route(self, tensor_route: str) -> List[HashRegistryEntry]:
        """Get all entries with specified tensor route."""
        return [entry for entry in self.hash_entries.values() if entry.tensor_route == tensor_route]

    def get_entries_by_priority_range(self, min_priority: float, max_priority: float) -> List[HashRegistryEntry]:
        """Get entries within priority range."""
        return [
            entry for entry in self.hash_entries.values()
            if min_priority <= entry.priority <= max_priority
        ]

    def get_enabled_entries(self) -> List[HashRegistryEntry]:
        """Get all enabled entries."""
        return [entry for entry in self.hash_entries.values() if entry.enabled]

    def get_disabled_entries(self) -> List[HashRegistryEntry]:
        """Get all disabled entries."""
        return [entry for entry in self.hash_entries.values() if not entry.enabled]

    def enable_entry(self, hash_id: str) -> bool:
        """Enable a hash registry entry."""
        try:
            if hash_id in self.hash_entries:
                self.hash_entries[hash_id].enabled = True
                self.registry_data[hash_id]["enabled"] = True
                self._save_registry()
                logger.info(f"Enabled hash entry: {hash_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error enabling entry {hash_id}: {e}")
            return False

    def disable_entry(self, hash_id: str) -> bool:
        """Disable a hash registry entry."""
        try:
            if hash_id in self.hash_entries:
                self.hash_entries[hash_id].enabled = False
                self.registry_data[hash_id]["enabled"] = False
                self._save_registry()
                logger.info(f"Disabled hash entry: {hash_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error disabling entry {hash_id}: {e}")
            return False

    def update_priority(self, hash_id: str, new_priority: float) -> bool:
        """Update priority of a hash registry entry."""
        try:
            if hash_id in self.hash_entries:
                self.hash_entries[hash_id].priority = new_priority
                self.registry_data[hash_id]["priority"] = new_priority
                self._save_registry()
                logger.info(f"Updated priority for {hash_id}: {new_priority}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error updating priority for {hash_id}: {e}")
            return False

    def get_best_matching_hash(self, bit_depth: int, tensor_route: str = None, min_priority: float = 0.0) -> Optional[HashRegistryEntry]:
        """Get best matching hash entry based on criteria."""
        try:
            candidates = []

            for entry in self.hash_entries.values():
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

        except Exception as e:
            logger.error(f"Error finding best matching hash: {e}")
            return None

    def resolve_hash_to_basket(self, hash_value: str, bit_depth: int = None) -> Optional[str]:
        """Resolve hash value to matrix basket using registry."""
        try:
            # If bit_depth not specified, determine from hash
            if bit_depth is None:
                bit_depth = self._determine_bit_depth_from_hash(hash_value)

            # Find best matching entry
            entry = self.get_best_matching_hash(bit_depth)
            if not entry:
                return None

            # Generate basket ID
            basket_id = f"basket_{entry.matrix_basket_id}"

            # Update usage stats
            self.usage_stats[entry.hash_id] = self.usage_stats.get(entry.hash_id, 0) + 1

            logger.debug(f"Resolved hash to basket {basket_id} via {entry.hash_id}")
            return basket_id

        except Exception as e:
            logger.error(f"Error resolving hash to basket: {e}")
            return None

    def _determine_bit_depth_from_hash(self, hash_value: str) -> int:
        """Determine bit depth from hash value."""
        try:
            # Use first byte to determine bit depth
            first_byte = int(hash_value[0:2], 16)

            if first_byte < 85:  # 0-84
                return 4
            elif first_byte < 170:  # 85-169
                return 8
            else:  # 170-255
                return 42

        except Exception as e:
            logger.warning(f"Error determining bit depth from hash: {e}")
            return 8  # Default to 8-bit

    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        try:
            total_entries = len(self.hash_entries)
            enabled_entries = len(self.get_enabled_entries())
            disabled_entries = len(self.get_disabled_entries())

            # Bit depth distribution
            bit_depth_dist = {}
            for entry in self.hash_entries.values():
                bit_depth = entry.bit_depth
                bit_depth_dist[bit_depth] = bit_depth_dist.get(bit_depth, 0) + 1

            # Route distribution
            route_dist = {}
            for entry in self.hash_entries.values():
                route = entry.tensor_route
                route_dist[route] = route_dist.get(route, 0) + 1

            # Priority statistics
            priorities = [entry.priority for entry in self.hash_entries.values()]
            avg_priority = sum(priorities) / len(priorities) if priorities else 0

            # Usage statistics
            total_usage = sum(self.usage_stats.values())
            most_used = unified_math.max(self.usage_stats.items(
            ), key=lambda x: x[1]) if self.usage_stats else (None, 0)

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
                },
                "usage_statistics": {
                    "total_usage": total_usage,
                    "most_used_entry": most_used[0],
                    "most_used_count": most_used[1]
                }
            }

        except Exception as e:
            logger.error(f"Error getting registry statistics: {e}")
            return {"error": str(e)}

    def integrate_with_matrix_mapper(self, matrix_mapper: MatrixMapper) -> None:
        """Integrate hash registry with matrix mapper."""
        try:
            self.matrix_mapper = matrix_mapper

            # Register all baskets in matrix mapper
            for entry in self.hash_entries.values():
                basket_id = f"basket_{entry.matrix_basket_id}"

                # Create basket if not exists
                if basket_id not in matrix_mapper.basket_registry:
                    # Generate hash signature for basket
                    hash_signature = self._generate_basket_hash_signature(entry)

                    # Create basket mapping
                    bit_phase = BitPhase(entry.bit_depth)
                    hash_mapping = matrix_mapper.HashBasketMapping(
                        hash_id=entry.hash_id,
                        basket_id=basket_id,
                        bit_phase=bit_phase,
                        hash_value=hash_signature,
                        basket_type=matrix_mapper.BasketType.STANDARD,
                        tensor_score=entry.priority,
                        resonance_score=entry.priority,
                        timestamp=datetime.now()
                    )

                    matrix_mapper.hash_registry[hash_signature] = hash_mapping
                    logger.debug(f"Integrated basket {basket_id} with hash registry")

            logger.info("Hash registry integrated with matrix mapper")

        except Exception as e:
            logger.error(f"Error integrating with matrix mapper: {e}")

    def _generate_basket_hash_signature(self, entry: HashRegistryEntry) -> str:
        """Generate hash signature for basket."""
        try:
            content = f"{entry.hash_id}_{entry.bit_depth}_{entry.tensor_route}_{entry.matrix_basket_id}_{entry.priority}"
            return hashlib.sha256(content.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Error generating basket hash signature: {e}")
            return hashlib.sha256(str(time.time()).encode()).hexdigest()

    def export_registry_summary(self, output_path: str = "hash_registry_summary.json") -> None:
        """Export registry summary to JSON file."""
        try:
            summary = {
                "registry_info": {
                    "total_entries": len(self.hash_entries),
                    "enabled_entries": len(self.get_enabled_entries()),
                    "bit_depths": list(set(entry.bit_depth for entry in self.hash_entries.values())),
                    "tensor_routes": list(set(entry.tensor_route for entry in self.hash_entries.values())),
                    "priority_range": {
                        "min": unified_math.min(entry.priority for entry in self.hash_entries.values()),
                        "max": unified_math.max(entry.priority for entry in self.hash_entries.values())
                    }
                },
                "entries": {
                    hash_id: {
                        "bit_depth": entry.bit_depth,
                        "tensor_route": entry.tensor_route,
                        "matrix_basket_id": entry.matrix_basket_id,
                        "priority": entry.priority,
                        "enabled": entry.enabled,
                        "usage_count": self.usage_stats.get(hash_id, 0)
                    }
                    for hash_id, entry in self.hash_entries.items()
                },
                "statistics": self.get_registry_statistics()
            }

            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Registry summary exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting registry summary: {e}")


def main():
    """Main function for hash registry manager testing."""
    safe_print("\\u1f5c4\\ufe0f Hash Registry Manager - Schwabot UROS v1.0")
    safe_print("=" * 50)

    # Initialize manager
    manager = HashRegistryManager()

    # Test basic functionality
    safe_print(f"Total entries: {len(manager.hash_entries)}")
    safe_print(f"Enabled entries: {len(manager.get_enabled_entries())}")

    # Test bit depth queries
    entries_4bit = manager.get_entries_by_bit_depth(4)
    entries_8bit = manager.get_entries_by_bit_depth(8)
    entries_42bit = manager.get_entries_by_bit_depth(42)

    safe_print(f"4-bit entries: {len(entries_4bit)}")
    safe_print(f"8-bit entries: {len(entries_8bit)}")
    safe_print(f"42-bit entries: {len(entries_42bit)}")

    # Test route queries
    entries_route_0 = manager.get_entries_by_route("route_0")
    safe_print(f"Route 0 entries: {len(entries_route_0)}")

    # Test hash resolution
    test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
    basket_id = manager.resolve_hash_to_basket(test_hash)
    safe_print(f"Resolved basket ID: {basket_id}")

    # Get statistics
    stats = manager.get_registry_statistics()
    safe_print(f"Registry statistics: {stats}")

    # Export summary
    manager.export_registry_summary()

    safe_print("\\u2705 Hash Registry Manager test completed")


if __name__ == "__main__":
    main()
