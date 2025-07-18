#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Key Allocator for Schwabot Trading System
===============================================

Manages memory key allocation and deallocation for the trading system.
"""

import logging
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryKey:
"""Memory key structure."""

key_id: str
key_type: str
allocated_at: float
size: int
metadata: Dict[str, Any]


class MemoryKeyAllocator:
"""Allocates and manages memory keys."""

def __init__(self):
"""Initialize the memory key allocator."""
self.allocated_keys: Dict[str, MemoryKey] = {}
self.key_counter = 0
self.logger = logging.getLogger(__name__)

def allocate_key(
self,
key_type: str = "symbolic",
size: int = 1024,
metadata: Optional[Dict[str, Any]] = None,
) -> str:
"""Allocate a new memory key."""
self.key_counter += 1
key_id = f"{key_type}_{self.key_counter}_{int(time.time())}"

memory_key = MemoryKey(
key_id=key_id,
key_type=key_type,
allocated_at=time.time(),
size=size,
metadata=metadata or {},
)

self.allocated_keys[key_id] = memory_key
self.logger.info(f"Allocated memory key: {key_id}")

return key_id

def deallocate_key(self, key_id: str) -> bool:
"""Deallocate a memory key."""
if key_id in self.allocated_keys:
del self.allocated_keys[key_id]
self.logger.info(f"Deallocated memory key: {key_id}")
return True
return False

def get_key_info(self, key_id: str) -> Optional[MemoryKey]:
"""Get information about a memory key."""
return self.allocated_keys.get(key_id)

def list_keys(self, key_type: Optional[str] = None) -> List[MemoryKey]:
"""List all allocated keys, optionally filtered by type."""
if key_type:
return [
key for key in self.allocated_keys.values() if key.key_type == key_type
]
return list(self.allocated_keys.values())

def cleanup_expired_keys(self, max_age_seconds: float = 3600) -> int:
"""Clean up expired memory keys."""
current_time = time.time()
expired_keys = [
key_id
for key_id, key in self.allocated_keys.items()
if current_time - key.allocated_at > max_age_seconds
]

for key_id in expired_keys:
self.deallocate_key(key_id)

return len(expired_keys)


# Global instance
memory_key_allocator = MemoryKeyAllocator()


def get_memory_key_allocator() -> MemoryKeyAllocator:
"""Get the global memory key allocator instance."""
return memory_key_allocator
