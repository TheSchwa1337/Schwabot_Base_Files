"""Module for Schwabot trading system."""

#!/usr/bin/env python3
"""
Registry Coordinator - Unified Registry Management
================================================

Coordinates the relationship between the canonical trade registry and specialized registries.
Ensures proper hash tracking, avoids redundancy, and maintains data integrity across all registries.

Features:
- Canonical registry as single source of truth
- Specialized registry linkage management
- Hash consistency validation
- Performance analytics aggregation
- Registry health monitoring
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from core.trade_registry import canonical_trade_registry, TradeEntry

logger = logging.getLogger(__name__)

@dataclass
class RegistryLink:
"""Class for Schwabot trading functionality."""
"""Link between canonical trade and specialized registry."""
canonical_hash: str
specialized_registry: str
specialized_hash: str
link_timestamp: float
metadata: Dict[str, Any]

class RegistryCoordinator:
"""Class for Schwabot trading functionality."""
"""Coordinates all registries to maintain data integrity and avoid redundancy."""


def __init__(self) -> None:
"""Initialize the registry coordinator."""
self.canonical_registry = canonical_trade_registry
self.specialized_registries: Dict[str, Any] = {}
self.registry_links: List[RegistryLink] = []
self.health_metrics: Dict[str, Any] = {}

logger.info("🔗 Registry Coordinator initialized")

def register_specialized_registry(self, registry_name: str, registry_instance: Any) -> bool:
"""Register a specialized registry with the coordinator."""
try:
self.specialized_registries[registry_name] = registry_instance
logger.info(f"📋 Registered specialized registry: {registry_name}")
return True
except Exception as e:
logger.error(f"Error registering specialized registry {registry_name}: {e}")
return False

def add_trade_with_linkages(self, trade_data: Dict[str, Any], specialized_data: Dict[str, Dict[str, Any]] = None) -> str:
"""
Add a trade to the canonical registry and link to specialized registries.

Args:
trade_data: Trade data for canonical registry
specialized_data: Dict of {registry_name: specialized_data}
"""
try:
# Add to canonical registry first
canonical_hash = self.canonical_registry.add_trade(trade_data)

# Link to specialized registries if provided
if specialized_data:
for registry_name, data in specialized_data.items():
if registry_name in self.specialized_registries:
specialized_hash = self._add_to_specialized_registry(registry_name, data, canonical_hash)
if specialized_hash:
self._create_registry_link(canonical_hash, registry_name, specialized_hash)

logger.info(f"✅ Trade added with linkages: {canonical_hash[:8]}...")
return canonical_hash

except Exception as e:
logger.error(f"Error adding trade with linkages: {e}")
raise

def _add_to_specialized_registry(self, registry_name: str, data: Dict[str, Any], canonical_hash: str) -> Optional[str]:
"""Add data to a specialized registry and return its hash."""
try:
registry = self.specialized_registries[registry_name]

# Add canonical hash reference to specialized data
data['canonical_hash'] = canonical_hash

# Call appropriate method based on registry type
if hasattr(registry, 'add_trade'):
specialized_hash = registry.add_trade(data)
elif hasattr(registry, 'register_soulprint'):
specialized_hash = registry.register_soulprint("data")
elif hasattr(registry, 'add_profitable_trade'):
specialized_hash = registry.add_profitable_trade(data)
elif hasattr(registry, 'register_digest'):
specialized_hash = registry.register_digest(data)
else:
logger.warning(f"Unknown registry type for {registry_name}")
return None

return specialized_hash

except Exception as e:
logger.error(f"Error adding to specialized registry {registry_name}: {e}")
return None

def _create_registry_link(self, canonical_hash: str, registry_name: str, specialized_hash: str) -> None:
"""Create a link between canonical and specialized registry."""
link = RegistryLink(
canonical_hash=canonical_hash,
specialized_registry=registry_name,
specialized_hash=specialized_hash,
link_timestamp=time.time(),
metadata={}
)

self.registry_links.append(link)

# Update canonical registry linkage
self.canonical_registry.link_specialized_registry(canonical_hash, registry_name, specialized_hash)

def get_trade_with_all_linkages(self, canonical_hash: str) -> Dict[str, Any]:
"""Get a trade with all its specialized registry linkages."""
trade_entry = self.canonical_registry.get_trade(canonical_hash)
if not trade_entry:
return {}

result = {
'canonical_trade': trade_entry,
'specialized_data': {}
}

# Get data from all linked specialized registries
for registry_name, specialized_hash in trade_entry.specialized_hashes.items():
if registry_name in self.specialized_registries:
registry = self.specialized_registries[registry_name]

# Try to get data from specialized registry
specialized_data = self._get_from_specialized_registry(registry, specialized_hash)
if specialized_data:
result['specialized_data'][registry_name] = specialized_data

return result

def _get_from_specialized_registry(self, registry: Any, specialized_hash: str) -> Optional[Dict[str, Any]]:
"""Get data from a specialized registry by hash."""
try:
if hasattr(registry, 'get_trade'):
return registry.get_trade(specialized_hash)
elif hasattr(registry, 'get_soulprint'):
return registry.get_soulprint("specialized_hash")
elif hasattr(registry, 'get_bucket'):
return registry.get_bucket(specialized_hash)
else:
return None
except Exception as e:
logger.error(f"Error getting from specialized registry: {e}")
return None

def get_performance_analytics(self) -> Dict[str, Any]:
"""Get comprehensive performance analytics across all registries."""
try:
# Get canonical registry performance
canonical_performance = self.canonical_registry.get_performance_summary()

# Aggregate specialized registry performance
specialized_performance = {}
for registry_name, registry in self.specialized_registries.items():
if hasattr(registry, 'get_performance_summary'):
specialized_performance[registry_name] = registry.get_performance_summary()
elif hasattr(registry, 'get_stats'):
specialized_performance[registry_name] = registry.get_stats()

# Calculate registry health metrics
health_metrics = self._calculate_registry_health()

return {
'canonical_registry': canonical_performance,
'specialized_registries': specialized_performance,
'registry_health': health_metrics,
'total_linkages': len(self.registry_links),
'active_registries': list(self.specialized_registries.keys())
}

except Exception as e:
logger.error(f"Error getting performance analytics: {e}")
return {'error': str(e)}

def _calculate_registry_health(self) -> Dict[str, Any]:
"""Calculate health metrics for all registries."""
health_metrics = {
'canonical_registry': {
'status': 'healthy',
'trade_count': len(self.canonical_registry.trades),
'linked_trades': len([t for t in self.canonical_registry.trades.values() if t.linked_registries])
},
'specialized_registries': {}
}

for registry_name, registry in self.specialized_registries.items():
try:
if hasattr(registry, 'trades'):
trade_count = len(registry.trades)
elif hasattr(registry, 'entries'):
trade_count = len(registry.entries)
elif hasattr(registry, 'buckets'):
trade_count = len(registry.buckets)
else:
trade_count = 0

health_metrics['specialized_registries'][registry_name] = {
'status': 'healthy',
'entry_count': trade_count
}
except Exception as e:
health_metrics['specialized_registries'][registry_name] = {
'status': 'error',
'error': str(e)
}

return health_metrics

def validate_registry_consistency(self) -> Dict[str, Any]:
"""Validate consistency between canonical and specialized registries."""
validation_results = {
'canonical_registry': {'status': 'valid', 'issues': []},
'specialized_registries': {},
'linkage_consistency': {'status': 'valid', 'issues': []}
}

# Validate canonical registry
canonical_trades = self.canonical_registry.trades
for trade_hash, trade_entry in canonical_trades.items():
# Check for missing specialized registry data
for registry_name, specialized_hash in trade_entry.specialized_hashes.items():
if registry_name not in self.specialized_registries:
validation_results['canonical_registry']['issues'].append(
f"Missing specialized registry: {registry_name}"
)
validation_results['canonical_registry']['status'] = 'warning'

# Validate specialized registries
for registry_name, registry in self.specialized_registries.items():
validation_results['specialized_registries'][registry_name] = {
'status': 'valid',
'issues': []
}

# Check for orphaned entries (no canonical reference)
try:
if hasattr(registry, 'trades'):
for specialized_hash, entry in registry.trades.items():
if not hasattr(entry, 'canonical_hash') or not entry.canonical_hash:
validation_results['specialized_registries'][registry_name]['issues'].append(
f"Orphaned entry: {specialized_hash[:8]}..."
)
validation_results['specialized_registries'][registry_name]['status'] = 'warning'
except Exception as e:
validation_results['specialized_registries'][registry_name]['issues'].append(
f"Validation error: {e}"
)
validation_results['specialized_registries'][registry_name]['status'] = 'error'

return validation_results

def cleanup_orphaned_entries(self) -> Dict[str, int]:
"""Clean up orphaned entries in specialized registries."""
cleanup_results = {}

for registry_name, registry in self.specialized_registries.items():
try:
cleaned_count = 0

if hasattr(registry, 'trades'):
# Find orphaned entries
orphaned_hashes = []
for specialized_hash, entry in registry.trades.items():
if not hasattr(entry, 'canonical_hash') or not entry.canonical_hash:
orphaned_hashes.append(specialized_hash)

# Remove orphaned entries
for hash_to_remove in orphaned_hashes:
del registry.trades[hash_to_remove]
cleaned_count += 1

cleanup_results[registry_name] = cleaned_count
if cleaned_count > 0:
logger.info(f"🧹 Cleaned {cleaned_count} orphaned entries from {registry_name}")

except Exception as e:
logger.error(f"Error cleaning up {registry_name}: {e}")
cleanup_results[registry_name] = -1

return cleanup_results

def get_registry_statistics(self) -> Dict[str, Any]:
"""Get comprehensive statistics about all registries."""
stats = {
'canonical_registry': {
'total_trades': len(self.canonical_registry.trades),
'total_profit': self.canonical_registry.total_profit,
'successful_trades': self.canonical_registry.successful_trades,
'linked_trades': len([t for t in self.canonical_registry.trades.values() if t.linked_registries])
},
'specialized_registries': {},
'linkages': {
'total_linkages': len(self.registry_links),
'registry_coverage': {}
}
}

# Calculate specialized registry statistics
for registry_name, registry in self.specialized_registries.items():
try:
if hasattr(registry, 'trades'):
entry_count = len(registry.trades)
elif hasattr(registry, 'entries'):
entry_count = len(registry.entries)
elif hasattr(registry, 'buckets'):
entry_count = len(registry.buckets)
else:
entry_count = 0

stats['specialized_registries'][registry_name] = {
'entry_count': entry_count
}

# Calculate coverage (percentage of canonical trades linked to this registry)
linked_count = len([t for t in self.canonical_registry.trades.values()
if registry_name in t.linked_registries])
coverage = (linked_count / len(self.canonical_registry.trades) * 100) if self.canonical_registry.trades else 0
stats['linkages']['registry_coverage'][registry_name] = coverage

except Exception as e:
stats['specialized_registries'][registry_name] = {'error': str(e)}

return stats

# Global instance for easy access
registry_coordinator = RegistryCoordinator()