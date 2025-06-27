from dataclasses import dataclass, field, asdict
from datetime import datetime
from dual_unicore_handler import DualUnicoreHandler
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import gzip
import hashlib
import json
import logging
import os
import pickle

from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug


# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf - 8 -*-\\n# Import safe print for Windows compatibility
try:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
except Exception as e:
    pass

""""""
""""""
    pass

except ImportError:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    try:
    except Exception as e:
        pass

# from core.utils.windows_cli_compatibility import safe_print, info, warn,
# error, success, debug  # F811: duplicate import
    except ImportError:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass


def safe_print(message):
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""


""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    print(message)


def info(message):
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""


""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    print(f"[INFO] {message}")


def warn(message):
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""


""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    print(f"[WARN] {message}")


def error(message):
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""


""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    print(f"[ERROR] {message}")


def success(message):
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""


""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    print(f"[SUCCESS] {message}")


def debug(message):
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""


""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    print(f"[DEBUG] {message}")


# """"""
""""""
""""""
Echo Snapshot - System State Capture and Replay for Schwabot
== == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

This module implements the echo snapshot system for Schwabot, providing
capabilities to capture, store, and replay system states, market conditions,
and trading scenarios. It supports state persistence, versioning, and
deterministic replay for testing and analysis.

Core Functionality:
- System state capture and storage
- Market condition snapshots
- Deterministic replay capabilities
- State versioning and branching
- Performance analysis and comparison
- Scenario testing and validation
""""""
""""""
""""""


logger = logging.getLogger(__name__)


class SnapshotType(Enum):

    SYSTEM_STATE = "system_state"


MARKET_CONDITION = "market_condition"
TRADING_SCENARIO = "trading_scenario"
PERFORMANCE_METRIC = "performance_metric"
ERROR_STATE = "error_state"


class SnapshotStatus(Enum):

    ACTIVE = "active"


ARCHIVED = "archived"
DELETED = "deleted"
REPLAYING = "replaying"


@dataclass
class Placeholder:

    """[BRAIN] Placeholder class for recursive profit mapping"""


""""""
""""""
    pass
    snapshot_id: str


snapshot_type: SnapshotType
timestamp: datetime
description: str
tags: List[str] = field(default_factory=list)
    version: str = "1.0"
checksum: Optional[str] = None
size_bytes: int = 0
compression_ratio: float = 1.0


@dataclass
class Placeholder:

    """[BRAIN] Placeholder class for recursive profit mapping"""


""""""
""""""
    pass
    """Represents a complete system state snapshot."""
""""""
""""""


timestamp: datetime
components: Dict[str, Any]
configurations: Dict[str, Any]
memory_usage: Dict[str, float]
active_processes: List[str]
error_logs: List[str]
performance_metrics: Dict[str, float]
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Placeholder:

    """[BRAIN] Placeholder class for recursive profit mapping"""


""""""
""""""
    pass
    """Represents a market condition snapshot."""
""""""
""""""


timestamp: datetime
symbols: Dict[str, Dict[str, Any]]
market_sentiment: Dict[str, float]
volatility_metrics: Dict[str, float]
volume_metrics: Dict[str, float]
technical_indicators: Dict[str, Dict[str, Any]]
news_events: List[Dict[str, Any]]
metadata: Dict[str, Any] = field(default_factory=dict)


class Placeholder:

    """[BRAIN] Placeholder class for recursive profit mapping"""


""""""
""""""
    pass


def __init__(self, storage_path: str = "./snapshots"):
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""


""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        self.storage_path = storage_path


self.snapshots: Dict[str, SnapshotMetadata] = {}
self.active_snapshots: Dict[str, Any] = {}
self.replay_history: List[Dict[str, Any]] = []
self._ensure_storage_directory()
        self._load_existing_snapshots()
        logger.info()
    f"EchoSnapshot initialized with storage path: {storage_path}"


def _ensure_storage_directory(self) -> None:

    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Ensure the storage directory exists."""
""""""
""""""


os.makedirs(self.storage_path, exist_ok = True)
        os.makedirs(os.path.join(self.storage_path, "metadata"), exist_ok = True)
        os.makedirs(os.path.join(self.storage_path, "data"), exist_ok = True)


def _load_existing_snapshots(self) -> None:

    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Load existing snapshot metadata from storage."""
""""""
""""""


metadata_dir = os.path.join(self.storage_path, "metadata")
        if not os.path.exists(metadata_dir):
            return

        for filename in os.listdir(metadata_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(metadata_dir, filename), 'r') as f:
                        metadata_dict = json.load(f)
                        metadata = SnapshotMetadata(**metadata_dict)
                        self.snapshots[metadata.snapshot_id] = metadata
                except Exception as e:
logger.error(f"Error loading snapshot metadata {filename}: {e}")


def _generate_snapshot_id(self, snapshot_type: SnapshotType,):


                            description: str -> str:

"""Generate a unique snapshot ID."""
""""""
""""""
timestamp = datetime.now().isoformat()
        base_string = f"{snapshot_type.value}_{description}_{timestamp}"
#         return hashlib.md5(base_string.encode()).hexdigest()[:16]


def _calculate_checksum(self, data: Any) -> str:

    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Calculate checksum for data integrity."""
""""""
""""""


data_bytes = pickle.dumps(data)
#         return hashlib.sha256(data_bytes).hexdigest()


def _compress_data(self, data: Any) -> bytes:

    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Compress data for storage."""
""""""
""""""


data_bytes = pickle.dumps(data)
#         return gzip.compress(data_bytes)


def _decompress_data(self, compressed_data: bytes) -> Any:

    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Decompress data from storage."""
""""""
""""""


data_bytes = gzip.decompress(compressed_data)
#         return pickle.loads(data_bytes)


def create_system_snapshot(self, components: Dict[str, Any,]):

                                configurations: Dict[str, Any],


description: str = "",
tags: Optional[List[str]] = None -> str:
"""Create a system state snapshot."""
""""""
""""""
snapshot_id = self._generate_snapshot_id()
    SnapshotType.SYSTEM_STATE, description

# Create system state
system_state = SystemState()
            timestamp = datetime.now(),
            components = components,
configurations = configurations,
memory_usage = self._get_memory_usage(),
            active_processes = self._get_active_processes(),
            error_logs = self._get_error_logs(),
            performance_metrics = self._get_performance_metrics()

# Create metadata
metadata = SnapshotMetadata()
            snapshot_id = snapshot_id,
snapshot_type = SnapshotType.SYSTEM_STATE,
timestamp = datetime.now(),
            description = description,
tags = tags or []


# Store snapshot
self._store_snapshot(snapshot_id, system_state, metadata)
        logger.info(f"System snapshot created: {snapshot_id}")
#         return snapshot_id

def create_market_snapshot(self, symbols: Dict[str, Dict[str, Any],]):


                                market_sentiment: Dict[str, float],
description: str="",
tags: Optional[List[str]]=None -> str:
"""Create a market condition snapshot."""
""""""
""""""
snapshot_id = self._generate_snapshot_id()
    SnapshotType.MARKET_CONDITION, description

# Create market condition
market_condition = MarketCondition()
            timestamp = datetime.now(),
            symbols = symbols,
market_sentiment = market_sentiment,
volatility_metrics = self._calculate_volatility_metrics(symbols),
            volume_metrics = self._calculate_volume_metrics(symbols),
            technical_indicators = self._calculate_technical_indicators(symbols),
            news_events = self._get_news_events()


# Create metadata
metadata = SnapshotMetadata()
            snapshot_id = snapshot_id,
snapshot_type = SnapshotType.MARKET_CONDITION,
timestamp = datetime.now(),
            description = description,
tags = tags or []


# Store snapshot
self._store_snapshot(snapshot_id, market_condition, metadata)
        logger.info(f"Market snapshot created: {snapshot_id}")
#         return snapshot_id

def _store_snapshot(self, snapshot_id: str, data: Any,):


                        metadata: SnapshotMetadata -> None:
"""Store snapshot data and metadata."""
""""""
""""""
# Compress and store data
compressed_data = self._compress_data(data)
        data_path = os.path.join(self.storage_path, "data", f"{snapshot_id}.gz")

        with open(data_path, 'wb') as f:
            f.write(compressed_data)

# Calculate metadata
metadata.checksum = self._calculate_checksum(data)
        metadata.size_bytes = len(compressed_data)
        metadata.compression_ratio = len()
            compressed_data / len(pickle.dumps(data))

# Store metadata
metadata_path = os.path.join()
    self.storage_path,
    "metadata",
        f"{snapshot_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent = 2, default = str)

# Update in - memory storage
self.snapshots[snapshot_id]=metadata
self.active_snapshots[snapshot_id]=data

def load_snapshot(self, snapshot_id: str) -> Optional[Any]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Load a snapshot from storage."""
""""""
""""""
        if snapshot_id in self.active_snapshots:
#             return self.active_snapshots[snapshot_id]

        if snapshot_id not in self.snapshots:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
logger.error(f"Snapshot not found: {snapshot_id}")
#             return None

        try:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
        except Exception as e:
            pass

""""""
""""""
    pass
data_path = os.path.join(self.storage_path, "data", f"{snapshot_id}.gz")
            with open(data_path, 'rb') as f:
                compressed_data = f.read()

data = self._decompress_data(compressed_data)

# Verify checksum
calculated_checksum = self._calculate_checksum(data)
            if calculated_checksum != self.snapshots[snapshot_id].checksum:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
logger.error(f"Checksum mismatch for snapshot: {snapshot_id}")
#                 return None

self.active_snapshots[snapshot_id]=data
logger.debug(f"Snapshot loaded: {snapshot_id}")
#             return data

        except Exception as e:
logger.error(f"Error loading snapshot {snapshot_id}: {e}")
#             return None

def replay_snapshot(self, snapshot_id: str,):


                        replay_config: Optional[Dict[str, Any]]=None -> Dict[str, Any]:
"""Replay a snapshot with optional configuration."""
""""""
""""""
snapshot_data = self.load_snapshot(snapshot_id)
        if not snapshot_data:
#             return {"success": False, "error": "Snapshot not found"}

replay_config = replay_config or {}
replay_id = f"replay_{snapshot_id}_{int(datetime.now().timestamp())}"

        try:
        except Exception as e:
            pass

# Mark snapshot as replaying
            if snapshot_id in self.snapshots:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
self.snapshots[snapshot_id].status = SnapshotStatus.REPLAYING

# Perform replay based on snapshot type
            if isinstance(snapshot_data, SystemState):
                result = self._replay_system_state(snapshot_data, replay_config)
            elif isinstance(snapshot_data, MarketCondition):
                result = self._replay_market_condition()
                    snapshot_data, replay_config
            else:
result={"success": False, "error": "Unknown snapshot type"}

# Record replay history
replay_record={}
"replay_id": replay_id,
"snapshot_id": snapshot_id,
"timestamp": datetime.now(),
                "config": replay_config,
"result": result

self.replay_history.append(replay_record)

logger.info(f"Snapshot replayed: {snapshot_id}")
#             return result

        except Exception as e:
logger.error(f"Error replaying snapshot {snapshot_id}: {e}")
#             return {"success": False, "error": str(e)}

def _replay_system_state(self, system_state: SystemState,):


                            config: Dict[str, Any] -> Dict[str, Any]:
"""Replay a system state snapshot."""
""""""
""""""
# This would typically involve restoring system components
# to their captured state
#         return {}
"success": True,
"type": "system_state",
"components_restored": len(system_state.components),
            "configurations_applied": len(system_state.configurations),
            "timestamp": system_state.timestamp.isoformat()


def _replay_market_condition(self, market_condition: MarketCondition,):


                                config: Dict[str, Any] -> Dict[str, Any]:
"""Replay a market condition snapshot."""
""""""
""""""
# This would typically involve restoring market data
# to the captured state
#         return {}
"success": True,
"type": "market_condition",
"symbols_restored": len(market_condition.symbols),
            "sentiment_restored": len(market_condition.market_sentiment),
            "timestamp": market_condition.timestamp.isoformat()


def list_snapshots(self, snapshot_type: Optional[SnapshotType = None,]):


                        tags: Optional[List[str]]=None -> List[SnapshotMetadata]:
"""List available snapshots with optional filtering."""
""""""
""""""
snapshots = list(self.snapshots.values())

        if snapshot_type:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
snapshots=[s for s in snapshots if s.snapshot_type == snapshot_type]

        if tags:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
snapshots=[s for s in snapshots if any(tag in s.tags for tag in tags)]

#         return sorted(snapshots, key = lambda x: x.timestamp, reverse = True)

def delete_snapshot(self, snapshot_id: str) -> bool:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Delete a snapshot from storage."""
""""""
""""""
        if snapshot_id not in self.snapshots:
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
logger.error(f"Snapshot not found for deletion: {snapshot_id}")
#             return False

        try:
        except Exception as e:
            pass

# Remove data file
data_path = os.path.join(self.storage_path, "data", f"{snapshot_id}.gz")
            if os.path.exists(data_path):
                os.remove(data_path)

# Remove metadata file
metadata_path = os.path.join()
    self.storage_path,
    "metadata",
        f"{snapshot_id}.json"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

# Remove from memory
            if snapshot_id in self.active_snapshots:
                del self.active_snapshots[snapshot_id]

            del self.snapshots[snapshot_id]

logger.info(f"Snapshot deleted: {snapshot_id}")
#             return True

        except Exception as e:
logger.error(f"Error deleting snapshot {snapshot_id}: {e}")
#             return False

def get_storage_statistics(self) -> Dict[str, Any]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Get storage statistics."""
""""""
""""""
total_snapshots = len(self.snapshots)
        total_size = sum(s.size_bytes for s in self.snapshots.values())
        avg_compression = sum(s.compression_ratio for s in self.snapshots.values()) /
                            total_snapshots if total_snapshots > 0 else 0

type_counts={}
        for snapshot in self.snapshots.values():
            type_counts[snapshot.snapshot_type.value]=type_counts.get()
                snapshot.snapshot_type.value, 0 + 1

#         return {}
"total_snapshots": total_snapshots,
"total_size_bytes": total_size,
"average_compression_ratio": avg_compression,
"type_distribution": type_counts,
"replay_count": len(self.replay_history)


# Helper methods for system state capture
def _get_memory_usage(self) -> Dict[str, float]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Get current memory usage."""
""""""
""""""
import psutil

# Import core mathematical modules
from core.unified_math_system import unified_math
from core.bit_phase_sequencer import BitPhase, BitSequence
from core.symbolic_profit_router import ProfitTier, FlipBias, SymbolicState
from core.dual_error_handler import PhaseState, SickType, SickState

memory = psutil.virtual_memory()
#         return {}
"total": memory.total,
"available": memory.available,
"percent": memory.percent,
"used": memory.used,
"free": memory.free


def _get_active_processes(self) -> List[str]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Get list of active processes."""
""""""
""""""
#         return [p.name() for p in psutil.process_iter(['name'])]

def _get_error_logs(self) -> List[str]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Get recent error logs."""
""""""
""""""
# This would typically read from actual log files
#         return []

def _get_performance_metrics(self) -> Dict[str, float]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Get current performance metrics."""
""""""
""""""
cpu_percent = psutil.cpu_percent(interval = 1)
        memory = psutil.virtual_memory()
#         return {}
"cpu_percent": cpu_percent,
"memory_percent": memory.percent,
"disk_usage_percent": psutil.disk_usage('/').percent


# Helper methods for market condition capture
def _calculate_volatility_metrics():

    self, symbols: Dict[str, Dict[str, Any]] -> Dict[str, float]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Calculate volatility metrics for symbols."""
""""""
""""""
# Placeholder implementation
#         return {symbol: 0.1 for symbol in symbols.keys()}

def _calculate_volume_metrics():

    self, symbols: Dict[str, Dict[str, Any]] -> Dict[str, float]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Calculate volume metrics for symbols."""
""""""
""""""
# Placeholder implementation
#         return {symbol: 1000000.0 for symbol in symbols.keys()}

def _calculate_technical_indicators():

    self, symbols: Dict[str, Dict[str, Any]] -> Dict[str, Dict[str, Any]]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Calculate technical indicators for symbols."""
""""""
""""""
# Placeholder implementation
#         return {symbol: {"rsi": 50.0, "macd": 0.0}}
            for symbol in symbols.keys()

def _get_news_events(self) -> List[Dict[str, Any]]:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
        """Get recent news events."""
""""""
""""""
# Placeholder implementation
#         return []

def main() -> None:


    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """Main function for testing and demonstration."""
""""""
""""""
echo_snapshot = EchoSnapshot("./test_snapshots")

# Create a system snapshot
components={"engine": "running", "database": "connected"}
configurations={"risk_level": 0.5, "max_position_size": 1000}
system_snapshot_id = echo_snapshot.create_system_snapshot()
        components, configurations, "Test system state", ["test", "demo"]


# Create a market snapshot
symbols={"BTC": {"price": 50000, "volume": 1000000}}
market_sentiment={"BTC": 0.7, "ETH": 0.6}
market_snapshot_id = echo_snapshot.create_market_snapshot()
        symbols, market_sentiment, "Test market condition", ["test", "market"]


# List snapshots
all_snapshots = echo_snapshot.list_snapshots()
    safe_print(f"Total snapshots: {len(all_snapshots)}")

# Replay a snapshot
replay_result = echo_snapshot.replay_snapshot(system_snapshot_id)
    safe_print(f"Replay result: {replay_result}")

# Get statistics
stats = echo_snapshot.get_storage_statistics()
    safe_print(f"Storage statistics: {stats}")

if __name__ == "__main__":
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
    """[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""
""""""
""""""
    pass
main()



""""""
""""""
""""""
""""""
