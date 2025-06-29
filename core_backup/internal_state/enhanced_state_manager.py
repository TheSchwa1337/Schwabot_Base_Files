# -*- coding: utf-8 -*-
""""""
Enhanced State Manager
=====================

Comprehensive state management system that integrates with internal logging,
    system states, memories, and backlogs. Specializes in BTC price hashing for
demo states and ensures proper initialization, organization, and connection
to all internal systems.

Features:
- Internal logging integration with structured logging
- System state management with memory and backlog tracking
- BTC price hashing for demo state generation
- Testing, demo, and live mode support
- Memory management with automatic cleanup
- Backlog processing with priority queuing
""""""

import hashlib
import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System operation modes."""

    TESTING = "testing"
    DEMO = "demo"
    LIVE = "live"


class LogLevel(Enum):
    """Logging levels for internal system."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMemory:
    """System memory state for tracking and persistence."""

    memory_id: str
    data: Dict[str, Any]
    timestamp: datetime
    ttl: float = 3600.0  # Time to live in seconds
    access_count: int = 0
    last_accessed: datetime = None

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)

    def access(self) -> None:
        """Mark memory as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class BacklogEntry:
    """Backlog entry for processing queue."""

    entry_id: str
    priority: int  # Higher number = higher priority
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    target: str
    processed: bool = False
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.entry_id is None:
            self.entry_id = f"{self.source}_{self.target}_{int(self.timestamp.timestamp())}"


@dataclass
class BTCPriceHash:
    """BTC price hash for demo state generation."""

    price: float
    volume: float
    timestamp: datetime
    hash_value: str
    phase: int
    agent: str = "BTC"

    @classmethod
    def from_price_data(cls, price: float, volume: float, phase: int = 32) -> "BTCPriceHash":
        """Create BTC price hash from price data."""
        timestamp = datetime.now()
        data_str = f"{price:.8f}_{volume:.8f}_{timestamp.isoformat()}_{phase}"
        hash_value = hashlib.sha256(data_str.encode()).hexdigest()
        return cls(price=price, volume=volume, timestamp=timestamp, hash_value=hash_value, phase=phase)


class EnhancedStateManager:
    """"""
    Enhanced state manager with logging, memory, and backlog integration.
    """"""

    def __init__(self, mode: SystemMode = SystemMode.DEMO, log_level: LogLevel = LogLevel.INFO):
        self.mode = mode
        self.log_level = log_level
        self.start_time = datetime.now()

        # Initialize core components
        self.memories: Dict[str, SystemMemory] = {}
        self.backlog_queue = queue.PriorityQueue()
        self.backlog_processed: List[BacklogEntry] = []
        self.btc_price_history: List[BTCPriceHash] = []

        # Threading and locks
        self.memory_lock = threading.RLock()
        self.backlog_lock = threading.RLock()
        self.btc_lock = threading.RLock()

        # Background workers
        self.memory_cleanup_thread = threading.Thread(target=self._memory_cleanup_loop, daemon=True)
        self.backlog_processor_thread = threading.Thread(target=self._backlog_processor_loop, daemon=True)
        self.btc_hash_generator_thread = threading.Thread(target=self._btc_hash_generator_loop, daemon=True)

        # Start background workers
        self.memory_cleanup_thread.start()
        self.backlog_processor_thread.start()
        self.btc_hash_generator_thread.start()

        # Initialize logging
        self._setup_logging()

        logger.info(f"EnhancedStateManager initialized in {mode.value} mode")

    def _setup_logging(self) -> None:
        """Setup internal logging system."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Create logs directory if it doesn't exist'
        os.makedirs("logs", exist_ok=True)

        # File handler for system logs
        file_handler = logging.FileHandler(f"logs/enhanced_state_manager_{self.mode.value}.log")
        file_handler.setLevel(getattr(logging, self.log_level.value.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level.value.upper()))
        console_handler.setFormatter(logging.Formatter(log_format))

        # Configure logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(getattr(logging, self.log_level.value.upper()))

    def store_memory(self, memory_id: str, data: Dict[str, Any], ttl: float = 3600.0) -> None:
        """Store data in system memory."""
        with self.memory_lock:
            memory = SystemMemory(memory_id=memory_id, data=data, timestamp=datetime.now(), ttl=ttl)
            self.memories[memory_id] = memory
            logger.info(f"Stored memory: {memory_id} (TTL: {ttl}s)")

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from system memory."""
        with self.memory_lock:
            memory = self.memories.get(memory_id)
            if memory:
                if memory.is_expired():
                    del self.memories[memory_id]
                    logger.warning(f"Memory expired: {memory_id}")
                    return None
                memory.access()
                logger.debug(f"Accessed memory: {memory_id} (access count: {memory.access_count})")
                return memory.data
            return None

    def add_backlog_entry(self, priority: int, data: Dict[str, Any], source: str, target: str) -> str:
        """Add entry to processing backlog."""
        with self.backlog_lock:
            entry = BacklogEntry(priority=priority, data=data, timestamp=datetime.now(), source=source, target=target)
            # Priority queue uses negative priority (higher priority = lower number)
            self.backlog_queue.put((-priority, entry))
            logger.info(f"Added backlog entry: {entry.entry_id} (priority: {priority})")
            return entry.entry_id

    def get_backlog_status(self) -> Dict[str, Any]:
        """Get backlog processing status."""
        with self.backlog_lock:
            return {}
                "queue_size": self.backlog_queue.qsize(),
                    "processed_count": len(self.backlog_processed),
                        "total_entries": self.backlog_queue.qsize() + len(self.backlog_processed),
                        "timestamp": datetime.now().isoformat(),
}
    def generate_btc_price_hash(self, price: float, volume: float, phase: int = 32) -> BTCPriceHash:
        """Generate BTC price hash for demo state."""
        with self.btc_lock:
            btc_hash = BTCPriceHash.from_price_data(price, volume, phase)
            self.btc_price_history.append(btc_hash)

            # Keep only last 1000 entries
            if len(self.btc_price_history) > 1000:
                self.btc_price_history = self.btc_price_history[-1000:]

            logger.info()
                f"Generated BTC price hash: {btc_hash.hash_value[:16]}... "
                f"(price: {price}, volume: {volume}, phase: {phase})"
            )
            return btc_hash

    def get_btc_price_history(self, limit: int = 100) -> List[BTCPriceHash]:
        """Get recent BTC price history."""
        with self.btc_lock:
            return self.btc_price_history[-limit:]

    def create_demo_state()
        self, btc_price: float, btc_volume: float, phase: int = 32, additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create demo state with BTC price hashing."""
        # Generate BTC price hash
        btc_hash = self.generate_btc_price_hash(btc_price, btc_volume, phase)

        # Create demo state
        demo_state = {
            "mode": self.mode.value,
            "btc_price_hash": {}
            "price": btc_hash.price,
            "volume": btc_hash.volume,
            "hash": btc_hash.hash_value,
            "phase": btc_hash.phase,
            "timestamp": btc_hash.timestamp.isoformat(),
}
                        },
                        "system_metrics": {}
                "memory_count": len(self.memories),
                    "backlog_size": self.backlog_queue.qsize(),
                        "btc_history_size": len(self.btc_price_history),
                        "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                        },
                        "additional_data": additional_data or {},
                    "timestamp": datetime.now().isoformat(),
}
        # Store in memory
        memory_id = f"demo_state_{btc_hash.hash_value[:16]}"
        self.store_memory(memory_id, demo_state, ttl=7200.0)  # 2 hours TTL

        logger.info(f"Created demo state: {memory_id}")
        return demo_state

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {}
            "mode": self.mode.value,
                "log_level": self.log_level.value,
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                    "memory": {"active_memories": len(self.memories), "memory_ids": list(self.memories.keys())},
                    "backlog": self.get_backlog_status(),
                    "btc_price": {}
                "history_size": len(self.btc_price_history),
                    "latest_hash": self.btc_price_history[-1].hash_value if self.btc_price_history else None,
                        },
                        "threads": {}
                "memory_cleanup": self.memory_cleanup_thread.is_alive(),
                    "backlog_processor": self.backlog_processor_thread.is_alive(),
                        "btc_hash_generator": self.btc_hash_generator_thread.is_alive(),
                        },
                        "timestamp": datetime.now().isoformat(),
}
    def _memory_cleanup_loop(self) -> None:
        """Background loop for memory cleanup."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_expired_memories()
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}")

    def _cleanup_expired_memories(self) -> None:
        """Clean up expired memories."""
        with self.memory_lock:
            expired_count = 0
            for memory_id, memory in list(self.memories.items()):
                if memory.is_expired():
                    del self.memories[memory_id]
                    expired_count += 1

            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired memories")

    def _backlog_processor_loop(self) -> None:
        """Background loop for backlog processing."""
        while True:
            try:
                time.sleep(1)  # Process every second
                self._process_backlog_entry()
            except Exception as e:
                logger.error(f"Backlog processing error: {e}")

    def _process_backlog_entry(self) -> None:
        """Process a single backlog entry."""
        try:
            if not self.backlog_queue.empty():
                priority, entry = self.backlog_queue.get_nowait()

                # Process the entry
                success = self._process_entry(entry)

                if success:
                    entry.processed = True
                    with self.backlog_lock:
                        self.backlog_processed.append(entry)
                    logger.info(f"Processed backlog entry: {entry.entry_id}")
                else:
                    entry.retry_count += 1
                    if entry.retry_count < entry.max_retries:
                        # Re-queue with lower priority
                        self.backlog_queue.put((priority - 1, entry))
                        logger.warning()
                            f"Re-queued backlog entry: {entry.entry_id} "
                            f"(retry {entry.retry_count}/{entry.max_retries})"
                        )
                    else:
                        logger.error(f"Failed to process backlog entry: {entry.entry_id} " f"(max retries exceeded)")

        except queue.Empty:
            pass  # No entries to process

    def _process_entry(self, entry: BacklogEntry) -> bool:
        """Process a single backlog entry."""
        try:
            # Example processing logic - can be customized based on source/target
            if entry.source == "demo" and entry.target == "memory":
                # Store demo data in memory
                memory_id = f"backlog_{entry.entry_id}"
                self.store_memory(memory_id, entry.data, ttl=1800.0)  # 30 minutes
                return True
            elif entry.source == "btc" and entry.target == "hash":
                # Process BTC data for hashing
                if "price" in entry.data and "volume" in entry.data:
                    self.generate_btc_price_hash(entry.data["price"], entry.data["volume"], entry.data.get("phase", 32))
                    return True
            else:
                # Default processing - just log
                logger.debug(f"Processing entry: {entry.source} -> {entry.target}")
                return True

        except Exception as e:
            logger.error(f"Error processing entry {entry.entry_id}: {e}")
            return False

    def _btc_hash_generator_loop(self) -> None:
        """Background loop for BTC hash generation (demo mode only)."""
        if self.mode != SystemMode.DEMO:
            return

        while True:
            try:
                time.sleep(5)  # Generate hash every 5 seconds in demo mode
                # Generate random BTC price data for demo
                price = 45000 + np.random.normal(0, 1000)  # Random price around 45k
                volume = 1000 + np.random.normal(0, 200)  # Random volume
                self.generate_btc_price_hash(price, volume, 32)
            except Exception as e:
                logger.error(f"BTC hash generation error: {e}")

    def export_system_state(self, filename: Optional[str] = None) -> str:
        """Export complete system state to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_state_{self.mode.value}_{timestamp}.json"

        system_state = {
            "mode": self.mode.value,
            "start_time": self.start_time.isoformat(),
            "memories": {k: asdict(v) for k, v in self.memories.items()},
            "backlog_processed": [asdict(entry) for entry in self.backlog_processed],
            "btc_price_history": [asdict(hash_obj) for hash_obj in self.btc_price_history[-100:]],
            "system_status": self.get_system_status(),
            "export_timestamp": datetime.now().isoformat(),
}
}
        with open(filename, "w") as f:
            json.dump(system_state, f, indent=2, default=str)

        logger.info(f"Exported system state to: {filename}")
        return filename

    def import_system_state(self, filename: str) -> bool:
        """Import system state from file."""
        try:
            with open(filename, "r") as f:
                system_state = json.load(f)

            # Import memories
            with self.memory_lock:
                self.memories.clear()
                for k, v in system_state.get("memories", {}).items():
                    memory = SystemMemory(**v)
                    memory.timestamp = datetime.fromisoformat(v["timestamp"])
                    memory.last_accessed = datetime.fromisoformat(v["last_accessed"])
                    self.memories[k] = memory

            # Import BTC price history
            with self.btc_lock:
                self.btc_price_history.clear()
                for hash_data in system_state.get("btc_price_history", []):
                    btc_hash = BTCPriceHash(**hash_data)
                    btc_hash.timestamp = datetime.fromisoformat(hash_data["timestamp"])
                    self.btc_price_history.append(btc_hash)

            logger.info(f"Imported system state from: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error importing system state: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Create enhanced state manager
    manager = EnhancedStateManager(mode=SystemMode.DEMO, log_level=LogLevel.INFO)

    # Test memory storage
    manager.store_memory("test_memory", {"key": "value", "number": 42})

    # Test backlog entry
    manager.add_backlog_entry(5, {"demo": "data"}, "demo", "memory")

    # Test BTC price hash generation
    btc_hash = manager.generate_btc_price_hash(50000.0, 1000.0, 32)

    # Test demo state creation
    demo_state = manager.create_demo_state(50000.0, 1000.0, 32, {"extra": "data"})

    # Print system status
    print("System Status:", manager.get_system_status())

    # Wait a bit for background processing
    time.sleep(10)

    # Export system state
    filename = manager.export_system_state()
    print(f"Exported system state to: {filename}")
