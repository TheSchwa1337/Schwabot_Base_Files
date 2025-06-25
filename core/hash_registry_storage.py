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
Hash Registry Storage - File I/O and Persistence
===============================================

Storage layer for hash registry operations.
Handles JSON file I/O and persistence with minimal dependencies.

Functions:
- Load registry from JSON file
- Save registry to JSON file
- Export registry summary
- Registry backup and restore
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Conditional import to handle both script and module execution
try:
    from .hash_registry_core import HashRegistryEntry, HashRegistryCore
except ImportError:
    # When run as script, use direct import
#     from hash_registry_core import HashRegistryEntry, HashRegistryCore  # F811: duplicate import

logger = logging.getLogger(__name__)

class HashRegistryStorage:
    """
Storage layer for hash registry operations.
Handles file I/O and persistence with minimal dependencies.
"""

    def __init__(self, registry_path: str = "core/hash_registry.json"):
        """Initialize storage with registry file path."""
self.registry_path = registry_path
self.backup_path = f"{registry_path}.backup"

    def load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load hash registry from JSON file."""
        try:
            if not os.path.exists(self.registry_path):
                logger.warning(f"Registry file not found: {self.registry_path}")
                return {}

            with open(self.registry_path, 'r') as f:
                registry_data = json.load(f)

logger.info(f"Loaded registry from {self.registry_path}")
            return registry_data

        except Exception as e:
logger.error(f"Error loading registry: {e}")
            return {}

    def save_registry(self, registry_data: Dict[str, Dict[str, Any]]) -> bool:
        """Save hash registry to JSON file."""
        try:
            # Create directory if it doesn't exist
dir_path = os.path.dirname(self.registry_path)
            if dir_path:  # Only create directory if path is not empty
os.makedirs(dir_path, exist_ok=True)

            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)

logger.info(f"Registry saved to {self.registry_path}")
            return True

        except Exception as e:
logger.error(f"Error saving registry: {e}")
            return False

    def parse_registry_entries(self, registry_data: Dict[str, Dict[str, Any]]) -> Dict[str, HashRegistryEntry]:
        """Parse registry data into HashRegistryEntry objects."""
entries = {}

        for hash_id, entry_data in registry_data.items():
            try:
entry = HashRegistryEntry(
                    hash_id=hash_id,
bit_depth=entry_data.get('bit_depth', 8),
                    tensor_route=entry_data.get('tensor_route', 'route_0'),
                    matrix_basket_id=entry_data.get('matrix_basket_id', 0),
                    priority=entry_data.get('priority', 1.0),
                    enabled=entry_data.get('enabled', True),
                    metadata=entry_data.get('metadata', {})

entries[hash_id] = entry

            except Exception as e:
logger.warning(f"Error parsing entry {hash_id}: {e}")
                continue

        return entries

    def serialize_registry_entries(self, entries: Dict[str, HashRegistryEntry]) -> Dict[str, Dict[str, Any]]:
        """Serialize HashRegistryEntry objects to dictionary format."""
registry_data = {}

        for hash_id, entry in entries.items():
            registry_data[hash_id] = {
"bit_depth": entry.bit_depth,
"tensor_route": entry.tensor_route,
"matrix_basket_id": entry.matrix_basket_id,
"priority": entry.priority,
"enabled": entry.enabled,
"metadata": entry.metadata
}

        return registry_data

    def create_backup(self) -> bool:
        """Create backup of current registry."""
        try:
            if os.path.exists(self.registry_path):
                import shutil
shutil.copy2(self.registry_path, self.backup_path)
                logger.info(f"Registry backup created: {self.backup_path}")
                return True
            return False

        except Exception as e:
logger.error(f"Error creating backup: {e}")
            return False

    def restore_backup(self) -> bool:
        """Restore registry from backup."""
        try:
            if os.path.exists(self.backup_path):
                import shutil
shutil.copy2(self.backup_path, self.registry_path)
                logger.info(f"Registry restored from backup: {self.backup_path}")
                return True
            return False

        except Exception as e:
logger.error(f"Error restoring backup: {e}")
            return False

    def export_registry_summary(self, entries: Dict[str, HashRegistryEntry],
                              output_path: str = "hash_registry_summary.json") -> bool:
"""Export registry summary to JSON file."""
        try:
            # Calculate statistics
stats = HashRegistryCore.calculate_registry_statistics(entries)

            # Create summary
summary = {
"export_info": {
"timestamp": datetime.now().isoformat(),
                    "total_entries": len(entries),
                    "export_version": "1.0"
},
"registry_info": {
"total_entries": len(entries),
                    "enabled_entries": len(HashRegistryCore.get_enabled_entries(entries)),
                    "bit_depths": list(set(entry.bit_depth for entry in entries.values())),
                    "tensor_routes": list(set(entry.tensor_route for entry in entries.values())),
                    "priority_range": {
"min": unified_math.min(entry.priority for entry in entries.values()),
                        "max": unified_math.max(entry.priority for entry in entries.values())
                    }
},
"entries": {
hash_id: {
"bit_depth": entry.bit_depth,
"tensor_route": entry.tensor_route,
"matrix_basket_id": entry.matrix_basket_id,
"priority": entry.priority,
"enabled": entry.enabled
}
                    for hash_id, entry in entries.items()
                },
"statistics": stats
}

            # Create directory if it doesn't exist
dir_path = os.path.dirname(output_path)
            if dir_path:  # Only create directory if path is not empty
os.makedirs(dir_path, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)

logger.info(f"Registry summary exported to {output_path}")
            return True

        except Exception as e:
logger.error(f"Error exporting registry summary: {e}")
            return False

    def load_or_generate_registry(self) -> Dict[str, HashRegistryEntry]:
        """Load existing registry or generate new one."""
        try:
            # Try to load existing registry
registry_data = self.load_registry()

            if registry_data:
                # Parse existing entries
entries = self.parse_registry_entries(registry_data)
                logger.info(f"Loaded existing registry with {len(entries)} entries")
                return entries
            else:
                # Generate new registry
entries = HashRegistryCore.generate_complete_registry()

                # Save new registry
registry_data = self.serialize_registry_entries(entries)
                self.save_registry(registry_data)

logger.info(f"Generated new registry with {len(entries)} entries")
                return entries

        except Exception as e:
logger.error(f"Error loading/generating registry: {e}")
            # Generate fallback registry
entries = HashRegistryCore.generate_complete_registry()
            logger.warning(f"Generated fallback registry with {len(entries)} entries")
            return entries

    def update_entry(self, hash_id: str, updates: Dict[str, Any]) -> bool:
        """Update a specific registry entry."""
        try:
            # Load current registry
registry_data = self.load_registry()

            if hash_id not in registry_data:
logger.warning(f"Entry {hash_id} not found in registry")
                return False

            # Apply updates
            for key, value in updates.items():
                if key in ['bit_depth', 'tensor_route', 'matrix_basket_id', 'priority', 'enabled', 'metadata']:
registry_data[hash_id][key] = value

            # Save updated registry
            return self.save_registry(registry_data)

        except Exception as e:
logger.error(f"Error updating entry {hash_id}: {e}")
            return False

    def delete_entry(self, hash_id: str) -> bool:
        """Delete a specific registry entry."""
        try:
            # Load current registry
registry_data = self.load_registry()

            if hash_id not in registry_data:
logger.warning(f"Entry {hash_id} not found in registry")
                return False

            # Remove entry
            del registry_data[hash_id]

            # Save updated registry
            return self.save_registry(registry_data)

        except Exception as e:
logger.error(f"Error deleting entry {hash_id}: {e}")
            return False

    def get_registry_info(self) -> Dict[str, Any]:
        """Get basic registry information."""
        try:
            if os.path.exists(self.registry_path):
                stat = os.stat(self.registry_path)
                return {
"exists": True,
"size_bytes": stat.st_size,
"modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": self.registry_path
}
            else:
                return {
"exists": False,
"path": self.registry_path
}

        except Exception as e:
logger.error(f"Error getting registry info: {e}")
            return {"error": str(e)}


def main():
    """Test the storage layer."""
safe_print("💾 Hash Registry Storage - File I/O Test")
    safe_print("=" * 50)

    # Initialize storage
storage = HashRegistryStorage("test_registry.json")

    # Test registry generation
safe_print("Generating test registry...")
    entries = HashRegistryCore.generate_complete_registry()
    safe_print(f"Generated {len(entries)} entries")

    # Test serialization
safe_print("\nSerializing registry...")
    registry_data = storage.serialize_registry_entries(entries)
    safe_print(f"Serialized {len(registry_data)} entries")

    # Test saving
safe_print("\nSaving registry...")
    success = storage.save_registry(registry_data)
    safe_print(f"Save successful: {success}")

    # Test loading
safe_print("\nLoading registry...")
    loaded_data = storage.load_registry()
    safe_print(f"Loaded {len(loaded_data)} entries")

    # Test parsing
safe_print("\nParsing entries...")
    parsed_entries = storage.parse_registry_entries(loaded_data)
    safe_print(f"Parsed {len(parsed_entries)} entries")

    # Test export
safe_print("\nExporting summary...")
    export_success = storage.export_registry_summary(parsed_entries, "test_summary.json")
    safe_print(f"Export successful: {export_success}")

    # Test registry info
safe_print("\nRegistry info:")
    info = storage.get_registry_info()
    safe_print(f"  {info}")

    # Cleanup
    try:
os.remove("test_registry.json")
        os.remove("test_summary.json")
        safe_print("\n✅ Test files cleaned up")
    except:
        pass

safe_print("\n✅ Hash Registry Storage test completed")


if __name__ == "__main__":
main()
