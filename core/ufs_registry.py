"""
UFS Registry
Manages Universal File System (UFS) registrations and lookups.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class UFSMetadata:
    """Metadata for a UFS entry"""
    path: str
    size: int
    last_modified: float
    is_active: bool
    metadata: Dict[str, Any]

class UFSRegistry:
    """Registry for Universal File System entries"""
    
    def __init__(self):
        self.ufs: Dict[str, UFSMetadata] = {}
        self.last_update = time.time()
        
    def register(self, path: str, size: int, 
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new UFS entry.
        
        Args:
            path: Path to the UFS entry
            size: Size of the entry in bytes
            metadata: Optional metadata dictionary
        """
        self.ufs[path] = UFSMetadata(
            path=path,
            size=size,
            last_modified=time.time(),
            is_active=True,
            metadata=metadata or {}
        )
        self.last_update = time.time()
        
    def unregister(self, path: str) -> None:
        """
        Unregister a UFS entry.
        
        Args:
            path: Path to the UFS entry
        """
        if path in self.ufs:
            del self.ufs[path]
            self.last_update = time.time()
            
    def get_metadata(self, path: str) -> Optional[UFSMetadata]:
        """
        Get metadata for a UFS entry.
        
        Args:
            path: Path to the UFS entry
            
        Returns:
            Optional[UFSMetadata]: Metadata if entry exists, None otherwise
        """
        return self.ufs.get(path)
        
    def is_registered(self, path: str) -> bool:
        """
        Check if a path is registered.
        
        Args:
            path: Path to check
            
        Returns:
            bool: True if path is registered
        """
        return path in self.ufs
        
    def clear(self) -> None:
        """Clear all registrations"""
        self.ufs.clear()
        self.last_update = time.time()

# Global instance
ufs_registry = UFSRegistry() 