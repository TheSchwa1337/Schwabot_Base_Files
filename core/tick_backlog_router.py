from core.unified_math_system import unified_math
import math
# #!/usr/bin/env python3
"""Tick Backlog Router - Full Tick-Linked Backlog Logic.

This module implements the complete tick-linked backlog logic that ensures
API outputs and internal tick memory are persistently matched.

Mathematical Foundation:
℘(t) = μ·Σ[T(i)×P(i)] + ∇²(T)

Where:
- ℘(t) = Backlog profit at time t
- μ = Memory persistence factor
- T(i) = Tick data at index i
- P(i) = Profit factor at index i
- ∇²(T) = Second derivative of tick data (acceleration)

Key Features:
- Persistent tick memory management
- API output synchronization
- Profit factor calculation
- Memory persistence validation
- Tick acceleration analysis
- Backlog state consistency checks

Flake8 compliant with comprehensive type hints and error handling.
"""

import logging
import time
# from core.unified_math_system import unified_math  # F811: duplicate import
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque
import json
import os

logger = logging.getLogger(__name__)


class BacklogState(Enum):
    """Backlog state types."""
ACTIVE = "active"
PERSISTENT = "persistent"
SYNCED = "synced"
DESYNCED = "desynced"
CORRUPTED = "corrupted"


class TickMemoryType(Enum):
    """Tick memory types."""
PRICE = "price"
VOLUME = "volume"
ORDER_BOOK = "order_book"
HASH = "hash"
PROFIT = "profit"
API_RESPONSE = "api_response"


@dataclass
class TickMemoryEntry:
    """Represents a tick memory entry."""
timestamp: float
tick_type: TickMemoryType
data: Dict[str, Any]
hash_value: str
profit_factor: float = 0.0
api_synced: bool = False
memory_id: str = field(default_factory=lambda: f"mem_{int(time.time() * 1000)}")


@dataclass
class BacklogProfit:
    """Represents backlog profit calculation."""
timestamp: float
total_profit: float
memory_persistence_factor: float
tick_profit_sum: float
acceleration_component: float
api_sync_score: float
state: BacklogState
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APISyncStatus:
    """Represents API synchronization status."""
api_name: str
last_sync_time: float
sync_success: bool
data_consistency: float
response_time: float
error_count: int = 0
metadata: Dict[str, Any] = field(default_factory=dict)


class TickBacklogRouter:
    """Core tick backlog router with persistent memory management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the tick backlog router."""
self.config = config or self._default_config()

        # Memory management
self.tick_memory: deque = deque(maxlen=self.config.get('max_memory_size', 10000))
        self.backlog_history: deque = deque(maxlen=self.config.get('max_backlog_history', 5000))
        self.api_sync_status: Dict[str, APISyncStatus] = {}

        # Performance tracking
self.total_ticks_processed = 0
self.total_profit_calculations = 0
self.api_sync_failures = 0

        # Memory persistence factor
self.memory_persistence_factor = self.config.get('memory_persistence_factor', 0.95)

        # File persistence
self.backlog_file_path = self.config.get('backlog_file_path', 'data/backlog_hash_state.json')
        self._ensure_data_directory()

logger.info("🔄 Tick Backlog Router initialized")

    def process_tick_data(self, tick_data: Dict[str, Any],
                         api_response: Optional[Dict[str, Any]] = None) -> BacklogProfit:
"""Process tick data and calculate backlog profit.

Args:
tick_data: Tick data containing price, volume, order book
api_response: Optional API response data

Returns:
BacklogProfit with calculation results
"""
        try:
            # Create tick memory entry
tick_entry = self._create_tick_memory_entry(tick_data, api_response)

            # Store in memory
self.tick_memory.append(tick_entry)

            # Calculate profit: ℘(t) = μ·Σ[T(i)×P(i)] + ∇²(T)
            backlog_profit = self._calculate_backlog_profit(tick_entry)

            # Update API sync status
            if api_response:
self._update_api_sync_status(api_response, tick_entry)

            # Validate memory persistence
persistence_valid = self._validate_memory_persistence()

            # Update backlog state
            if persistence_valid:
backlog_profit.state = BacklogState.SYNCED
            else:
backlog_profit.state = BacklogState.DESYNCED
self.api_sync_failures += 1

            # Store in history
self.backlog_history.append(backlog_profit)

            # Persist to file periodically
            if self.total_ticks_processed % 100 == 0:
self._persist_backlog_state()

            # Update performance tracking
self.total_ticks_processed += 1
self.total_profit_calculations += 1

logger.debug(f"Processed tick: profit={backlog_profit.total_profit:.6f}, "
                        f"state={backlog_profit.state.value}")

            return backlog_profit

        except Exception as e:
logger.error(f"Error processing tick data: {e}")
            return self._create_fallback_profit()

    def get_backlog_analytics(self) -> Dict[str, Any]:
        """Get backlog analytics and performance metrics."""
        try:
            if not self.backlog_history:
                return {
'total_ticks_processed': 0,
'total_profit_calculations': 0,
'average_profit': 0.0,
'api_sync_failures': 0,
'memory_persistence_factor': self.memory_persistence_factor,
'backlog_state': BacklogState.ACTIVE.value
}

            # Calculate statistics
profits = [profit.total_profit for profit in self.backlog_history]
sync_scores = [profit.api_sync_score for profit in self.backlog_history]

            # API sync statistics
api_sync_stats = {}
            for api_name, status in self.api_sync_status.items():
                api_sync_stats[api_name] = {
'last_sync_time': status.last_sync_time,
'sync_success': status.sync_success,
'data_consistency': status.data_consistency,
'response_time': status.response_time,
'error_count': status.error_count
}

            return {
'total_ticks_processed': self.total_ticks_processed,
'total_profit_calculations': self.total_profit_calculations,
'average_profit': unified_math.unified_math.mean(profits) if profits else 0.0,
                'profit_std': unified_math.unified_math.std(profits) if profits else 0.0,
                'average_sync_score': unified_math.unified_math.mean(sync_scores) if sync_scores else 0.0,
                'api_sync_failures': self.api_sync_failures,
'memory_persistence_factor': self.memory_persistence_factor,
'backlog_state': self._determine_overall_state().value,
                'api_sync_status': api_sync_stats,
'memory_size': len(self.tick_memory),
                'history_size': len(self.backlog_history)
            }

        except Exception as e:
logger.error(f"Error getting backlog analytics: {e}")
            return {}

    def validate_api_consistency(self, api_name: str, api_data: Dict[str, Any]) -> bool:
        """Validate API data consistency with internal memory."""
        try:
            if api_name not in self.api_sync_status:
                return False

            # Get recent memory entries
recent_entries = list(self.tick_memory)[-10:]  # Last 10 entries

            if not recent_entries:
                return False

            # Check consistency with API data
consistency_score = 0.0
valid_checks = 0

            for entry in recent_entries:
                if entry.api_synced:
                    # Compare with API data
                    if self._compare_with_api_data(entry, api_data):
                        consistency_score += 1.0
valid_checks += 1

            if valid_checks > 0:
consistency = consistency_score / valid_checks

                # Update API sync status
self.api_sync_status[api_name].data_consistency = consistency
self.api_sync_status[api_name].sync_success = consistency > 0.8

                return consistency > 0.8

            return False

        except Exception as e:
logger.error(f"Error validating API consistency: {e}")
            return False

    def _create_tick_memory_entry(self, tick_data: Dict[str, Any],
                                api_response: Optional[Dict[str, Any]]) -> TickMemoryEntry:
"""Create tick memory entry from data."""
        try:
            # Generate hash for tick data
hash_input = f"{tick_data.get('price', 0.0):.8f}|{tick_data.get('volume', 0.0):.6f}|{time.time():.3f}"
            hash_value = str(hash(hash_input))

            # Calculate profit factor
profit_factor = self._calculate_profit_factor(tick_data)

            # Determine if API synced
api_synced = api_response is not None

            return TickMemoryEntry(
                timestamp=tick_data.get('timestamp', time.time()),
                tick_type=TickMemoryType.PRICE,
data=tick_data,
hash_value=hash_value,
profit_factor=profit_factor,
api_synced=api_synced


        except Exception as e:
logger.error(f"Error creating tick memory entry: {e}")
            return TickMemoryEntry(
                timestamp=time.time(),
                tick_type=TickMemoryType.PRICE,
data={},
hash_value="",
profit_factor=0.0,
api_synced=False


    def _calculate_backlog_profit(self, tick_entry: TickMemoryEntry) -> BacklogProfit:
        """Calculate backlog profit: ℘(t) = μ·Σ[T(i)×P(i)] + ∇²(T)."""
        try:
            # Get recent memory entries for calculation
recent_entries = list(self.tick_memory)[-50:]  # Last 50 entries

            if not recent_entries:
                return self._create_fallback_profit()

            # Calculate tick profit sum: Σ[T(i)×P(i)]
            tick_profit_sum = 0.0
            for entry in recent_entries:
tick_value = self._extract_tick_value(entry)
                profit_factor = entry.profit_factor
tick_profit_sum += tick_value * profit_factor

            # Calculate acceleration component: ∇²(T)
            acceleration_component = self._calculate_tick_acceleration(recent_entries)

            # Apply memory persistence factor: μ·Σ[T(i)×P(i)]
            memory_persisted_sum = self.memory_persistence_factor * tick_profit_sum

            # Total profit: ℘(t) = μ·Σ[T(i)×P(i)] + ∇²(T)
            total_profit = memory_persisted_sum + acceleration_component

            # Calculate API sync score
api_sync_score = self._calculate_api_sync_score(recent_entries)

            # Determine state
state = self._determine_backlog_state(total_profit, api_sync_score)

            return BacklogProfit(
                timestamp=tick_entry.timestamp,
total_profit=total_profit,
memory_persistence_factor=self.memory_persistence_factor,
tick_profit_sum=tick_profit_sum,
acceleration_component=acceleration_component,
api_sync_score=api_sync_score,
state=state,
metadata={
'tick_memory_size': len(recent_entries),
                    'hash_value': tick_entry.hash_value,
'profit_factor': tick_entry.profit_factor
}


        except Exception as e:
logger.error(f"Error calculating backlog profit: {e}")
            return self._create_fallback_profit()

    def _extract_tick_value(self, entry: TickMemoryEntry) -> float:
        """Extract numerical value from tick entry."""
        try:
data = entry.data
            if 'price' in data:
                return float(data['price'])
            elif 'volume' in data:
                return float(data['volume'])
            else:
                return 0.0

        except Exception as e:
logger.error(f"Error extracting tick value: {e}")
            return 0.0

    def _calculate_profit_factor(self, tick_data: Dict[str, Any]) -> float:
        """Calculate profit factor for tick data."""
        try:
            # Base profit factor from price and volume
price = tick_data.get('price', 0.0)
            volume = tick_data.get('volume', 0.0)

            if price <= 0 or volume <= 0:
                return 0.0

            # Normalize price and volume
normalized_price = unified_math.min(price / 100000.0, 1.0)  # Normalize to 100k
            normalized_volume = unified_math.min(volume / 1000000.0, 1.0)  # Normalize to 1M

            # Calculate profit factor
profit_factor = (normalized_price * 0.6 + normalized_volume * 0.4)

            return unified_math.max(0.0, unified_math.min(1.0, profit_factor))

        except Exception as e:
logger.error(f"Error calculating profit factor: {e}")
            return 0.0

    def _calculate_tick_acceleration(self, entries: List[TickMemoryEntry]) -> float:
        """Calculate tick acceleration: ∇²(T)."""
        try:
            if len(entries) < 3:
                return 0.0

            # Extract tick values
tick_values = [self._extract_tick_value(entry) for entry in entries]

            if len(tick_values) < 3:
                return 0.0

            # Calculate second derivative (acceleration)
            # ∇²(T) ≈ T[i+2] - 2*T[i+1] + T[i]
            accelerations = []
            for i in range(len(tick_values) - 2):
                acceleration = tick_values[i+2] - 2*tick_values[i+1] + tick_values[i]
accelerations.append(acceleration)

            if not accelerations:
                return 0.0

            # Return average acceleration
            return unified_math.unified_math.mean(accelerations)

        except Exception as e:
logger.error(f"Error calculating tick acceleration: {e}")
            return 0.0

    def _calculate_api_sync_score(self, entries: List[TickMemoryEntry]) -> float:
        """Calculate API synchronization score."""
        try:
            if not entries:
                return 0.0

            # Count synced entries
synced_count = sum(1 for entry in entries if entry.api_synced)
            total_count = len(entries)

sync_score = synced_count / total_count if total_count > 0 else 0.0

            return sync_score

        except Exception as e:
logger.error(f"Error calculating API sync score: {e}")
            return 0.0

    def _determine_backlog_state(self, total_profit: float, api_sync_score: float) -> BacklogState:
        """Determine backlog state based on profit and sync score."""
        try:
            if api_sync_score < 0.5:
                return BacklogState.DESYNCED
            elif total_profit < 0:
                return BacklogState.CORRUPTED
            elif api_sync_score > 0.9:
                return BacklogState.SYNCED
            else:
                return BacklogState.PERSISTENT

        except Exception as e:
logger.error(f"Error determining backlog state: {e}")
            return BacklogState.ACTIVE

    def _update_api_sync_status(self, api_response: Dict[str, Any],
                              tick_entry: TickMemoryEntry) -> None:
"""Update API synchronization status."""
        try:
api_name = api_response.get('api_name', 'unknown')
            response_time = api_response.get('response_time', 0.0)

            if api_name not in self.api_sync_status:
self.api_sync_status[api_name] = APISyncStatus(
                    api_name=api_name,
last_sync_time=time.time(),
                    sync_success=True,
data_consistency=1.0,
response_time=response_time

            else:
status = self.api_sync_status[api_name]
status.last_sync_time = time.time()
                status.sync_success = True
status.response_time = response_time

        except Exception as e:
logger.error(f"Error updating API sync status: {e}")

    def _validate_memory_persistence(self) -> bool:
        """Validate memory persistence."""
        try:
            if len(self.tick_memory) < 10:
                return True  # Not enough data to validate

            # Check if recent entries are consistent
recent_entries = list(self.tick_memory)[-10:]

            # Validate hash consistency
hash_consistency = all(entry.hash_value for entry in recent_entries)

            # Validate timestamp consistency
timestamps = [entry.timestamp for entry in recent_entries]
timestamp_consistency = all(timestamps[i] <= timestamps[i+1]
                                      for i in range(len(timestamps)-1))

            return hash_consistency and timestamp_consistency

        except Exception as e:
logger.error(f"Error validating memory persistence: {e}")
            return False

    def _compare_with_api_data(self, entry: TickMemoryEntry, api_data: Dict[str, Any]) -> bool:
        """Compare memory entry with API data."""
        try:
            # Simple comparison - can be enhanced based on specific API structure
entry_price = entry.data.get('price', 0.0)
            api_price = api_data.get('price', 0.0)

            # Allow 1% tolerance
tolerance = 0.01
price_match = unified_math.abs(entry_price - api_price) / unified_math.max(entry_price, 1.0) < tolerance

            return price_match

        except Exception as e:
logger.error(f"Error comparing with API data: {e}")
            return False

    def _determine_overall_state(self) -> BacklogState:
        """Determine overall backlog state."""
        try:
            if not self.backlog_history:
                return BacklogState.ACTIVE

            # Check recent states
recent_states = [profit.state for profit in list(self.backlog_history)[-10:]]

            if BacklogState.CORRUPTED in recent_states:
                return BacklogState.CORRUPTED
            elif BacklogState.DESYNCED in recent_states:
                return BacklogState.DESYNCED
            elif all(state == BacklogState.SYNCED for state in recent_states):
                return BacklogState.SYNCED
            else:
                return BacklogState.PERSISTENT

        except Exception as e:
logger.error(f"Error determining overall state: {e}")
            return BacklogState.ACTIVE

    def _persist_backlog_state(self) -> None:
        """Persist backlog state to file."""
        try:
            # Create backup of existing file
            if os.path.exists(self.backlog_file_path):
                backup_path = f"{self.backlog_file_path}.backup"
os.rename(self.backlog_file_path, backup_path)

            # Prepare data for persistence
persistence_data = {
'metadata': {
'timestamp': time.time(),
                    'total_ticks_processed': self.total_ticks_processed,
'total_profit_calculations': self.total_profit_calculations,
'memory_persistence_factor': self.memory_persistence_factor,
'state': self._determine_overall_state().value
                },
'recent_memory': [
{
'timestamp': entry.timestamp,
'tick_type': entry.tick_type.value,
'hash_value': entry.hash_value,
'profit_factor': entry.profit_factor,
'api_synced': entry.api_synced
}
                    for entry in list(self.tick_memory)[-100:]  # Last 100 entries
                ],
'api_sync_status': {
api_name: {
'last_sync_time': status.last_sync_time,
'sync_success': status.sync_success,
'data_consistency': status.data_consistency,
'response_time': status.response_time,
'error_count': status.error_count
}
                    for api_name, status in self.api_sync_status.items()
                }
}

            # Write to file
            with open(self.backlog_file_path, 'w') as f:
                json.dump(persistence_data, f, indent=2)

logger.debug(f"Backlog state persisted to {self.backlog_file_path}")

        except Exception as e:
logger.error(f"Error persisting backlog state: {e}")

    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists."""
        try:
data_dir = os.path.dirname(self.backlog_file_path)
            if data_dir and not os.path.exists(data_dir):
                os.makedirs(data_dir)

        except Exception as e:
logger.error(f"Error ensuring data directory: {e}")

    def _create_fallback_profit(self) -> BacklogProfit:
        """Create fallback profit calculation."""
        return BacklogProfit(
            timestamp=time.time(),
            total_profit=0.0,
memory_persistence_factor=self.memory_persistence_factor,
tick_profit_sum=0.0,
acceleration_component=0.0,
api_sync_score=0.0,
state=BacklogState.ACTIVE


    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
'max_memory_size': 10000,
'max_backlog_history': 5000,
'memory_persistence_factor': 0.95,
'backlog_file_path': 'data/backlog_hash_state.json'
}


# Global instance for easy access
tick_backlog_router = TickBacklogRouter()


def process_tick_data(tick_data: Dict[str, Any],
                     api_response: Optional[Dict[str, Any]] = None) -> BacklogProfit:
"""Global function to process tick data."""
    return tick_backlog_router.process_tick_data(tick_data, api_response)


def get_backlog_analytics() -> Dict[str, Any]:
    """Global function to get backlog analytics."""
    return tick_backlog_router.get_backlog_analytics()


def validate_api_consistency(api_name: str, api_data: Dict[str, Any]) -> bool:
    """Global function to validate API consistency."""
    return tick_backlog_router.validate_api_consistency(api_name, api_data)
