# -*- coding: utf-8 -*-
"""
Hash Relay System (Mathematical Relay Core)
===========================================

Centralized relay for mathematical state, price, or vector data.
Supports hashing, relay, and event subscription for interconnected core modules.

Features:
- Accepts data from any core system (state, price, vectors, etc.)
- Hashes and relays data to downstream consumers
- Supports multiple hash algorithms (SHA-256, custom, etc.)
- Relay strategies: broadcast, selective, deduplication
- Event subscription for processed hashes or relay events
- Integration-ready for dualistic_thought_engines, lantern_core, api_bridge, etc.
"""

import hashlib
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import time

class HashRelaySystem:
    """
    Centralized hash/mathematical relay system for the Schwabot core.
    """
    def __init__(self):
        self.subscribers: List[Callable[[str, Dict[str, Any]], None]] = []
        self.relay_history: Set[str] = set()
        self.lock = threading.Lock()

    def submit(self, data: Dict[str, Any], algorithm: str = "sha256", relay_strategy: str = "broadcast") -> str:
        """
        Submit data for hashing and relay.
        Args:
            data: The data to hash and relay (dict, must be JSON-serializable)
            algorithm: Hash algorithm (default: sha256)
            relay_strategy: Relay strategy (default: broadcast)
        Returns:
            The resulting hash string
        """
        # Serialize data for hashing
        data_bytes = str(sorted(data.items())).encode()
        if algorithm == "sha256":
            hash_str = hashlib.sha256(data_bytes).hexdigest()
        elif algorithm == "md5":
            hash_str = hashlib.md5(data_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        with self.lock:
            if hash_str in self.relay_history:
                # Already relayed, skip if deduplication is desired
                return hash_str
            self.relay_history.add(hash_str)

        # Relay to subscribers
        self._relay(hash_str, data, relay_strategy)
        return hash_str

    def _relay(self, hash_str: str, data: Dict[str, Any], relay_strategy: str):
        """Relay the hash/data to all subscribers (or selective, in future)."""
        for callback in self.subscribers:
            try:
                callback(hash_str, data)
            except Exception as e:
                print(f"[HashRelaySystem] Relay error: {e}")

    def subscribe(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Subscribe to relay events. Callback receives (hash_str, data).
        """
        with self.lock:
            self.subscribers.append(callback)

    def get_history(self, limit: int = 100) -> List[str]:
        """
        Get recent relay history (hashes).
        """
        with self.lock:
            return list(self.relay_history)[-limit:]

    def clear_history(self):
        """
        Clear relay history (for testing or reset).
        """
        with self.lock:
            self.relay_history.clear()

# Global instance for easy access
hash_relay_system = HashRelaySystem()

# Example usage (can be removed in production)
if __name__ == "__main__":
    def print_relay(hash_str, data):
        print(f"[Relay] Hash: {hash_str} | Data: {data}")

    hash_relay_system.subscribe(print_relay)
    test_data = {"price": 62000, "symbol": "BTC/USDC", "timestamp": time.time()}
    hash_relay_system.submit(test_data) 