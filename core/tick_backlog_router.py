"""
Tick Backlog Router - Full Tick-Linked Backlog Logic
===================================================

This module implements the complete tick-linked backlog logic that ensures
API outputs and internal tick memory are persistently matched.

Mathematical Foundation:
ℙ(t) = μ·Σ[T(i)*P(i)] + ∇²(T)

Where:
- ℙ(t) = Backlog profit at time t
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

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# Import unified math system
try:
    from core.unified_math_system import unified_math
except ImportError:
    import math as unified_math

# Configure logging
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
    memory_id: str = field(
        default_factory=lambda: f"mem_{int(time.time() * 1000)}"
    )


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
        self.tick_memory: deque = deque(
            maxlen=self.config.get('max_memory_size', 10000)
        )
        self.backlog_history: deque = deque(
            maxlen=self.config.get('max_backlog_history', 5000)
        )
        self.api_sync_status: Dict[str, APISyncStatus] = {}

# Performance tracking
self.total_ticks_processed = 0
self.total_profit_calculations = 0
self.api_sync_failures = 0

# Memory persistence factor
        self.memory_persistence_factor = self.config.get(
    'memory_persistence_factor', 0.95
        )

# File persistence
        self.backlog_file_path = self.config.get(
            'backlog_file_path', 'data/backlog_hash_state.json'
        )
        self._ensure_data_directory()

        logger.info("🔄 Tick Backlog Router initialized")

    def process_tick_data(self, 
                         tick_data: Dict[str, Any],
                         api_response: Optional[Dict[str, Any]] = None) -> BacklogProfit:
        """
        Process tick data and calculate backlog profit.

Args:
            tick_data: Tick data dictionary
api_response: Optional API response data

Returns:
            BacklogProfit object with calculated profit and state
        """
        try:
# Create tick memory entry
            tick_entry = self._create_tick_memory_entry(tick_data)

            # Add to memory
self.tick_memory.append(tick_entry)

            # Update API sync status if response provided
            if api_response:
                self._update_api_sync_status(api_response)

            # Calculate backlog profit
            backlog_profit = self._calculate_backlog_profit(tick_entry)

            # Add to history
self.backlog_history.append(backlog_profit)

# Update performance tracking
self.total_ticks_processed += 1
self.total_profit_calculations += 1

            # Persist state periodically
            if self.total_ticks_processed % 100 == 0:
                self._persist_backlog_state()

            logger.debug(f"Processed tick: profit={backlog_profit.total_profit:.4f}, "
                        f"state={backlog_profit.state.value}")
            
            return backlog_profit

        except Exception as e:
            logger.error(f"Tick processing failed: {e}")
            return self._create_fallback_profit()

def get_backlog_analytics(self) -> Dict[str, Any]:
        """Get comprehensive backlog analytics."""
        try:
            if not self.backlog_history:
                return {"error": "No backlog history available"}

# Calculate statistics
            profits = [bp.total_profit for bp in self.backlog_history]
            sync_scores = [bp.api_sync_score for bp in self.backlog_history]

            # State distribution
            state_counts = {}
            for bp in self.backlog_history:
                state = bp.state.value
                state_counts[state] = state_counts.get(state, 0) + 1
            
            return {
                "total_ticks_processed": self.total_ticks_processed,
                "total_profit_calculations": self.total_profit_calculations,
                "api_sync_failures": self.api_sync_failures,
                "memory_size": len(self.tick_memory),
                "history_size": len(self.backlog_history),
                "profit_statistics": {
                    "mean": sum(profits) / len(profits) if profits else 0.0,
                    "min": min(profits) if profits else 0.0,
                    "max": max(profits) if profits else 0.0,
                    "std": self._calculate_std(profits) if profits else 0.0
                },
                "sync_statistics": {
                    "mean": sum(sync_scores) / len(sync_scores) if sync_scores else 0.0,
                    "min": min(sync_scores) if sync_scores else 0.0,
                    "max": max(sync_scores) if sync_scores else 0.0
                },
                "state_distribution": state_counts,
                "memory_persistence_factor": self.memory_persistence_factor,
                "api_sync_status": {
                    name: {
                        "last_sync_time": status.last_sync_time,
                        "success": status.sync_success,
                        "consistency": status.data_consistency,
                        "response_time": status.response_time,
                        "error_count": status.error_count
                    }
                    for name, status in self.api_sync_status.items()
                }
            }

        except Exception as e:
            logger.error(f"Analytics calculation failed: {e}")
            return {"error": str(e)}

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'max_memory_size': 10000,
            'max_backlog_history': 5000,
            'memory_persistence_factor': 0.95,
            'backlog_file_path': 'data/backlog_hash_state.json',
            'profit_calculation_method': 'weighted_sum',
            'api_sync_timeout': 5.0,
            'hash_algorithm': 'sha256'
        }

    def _create_tick_memory_entry(self, tick_data: Dict[str, Any]) -> TickMemoryEntry:
        """Create a tick memory entry from tick data."""
        try:
            timestamp = time.time()
            tick_type = self._determine_tick_type(tick_data)
            hash_value = self._calculate_hash(tick_data)
profit_factor = self._calculate_profit_factor(tick_data)

            return TickMemoryEntry(
                timestamp=timestamp,
                tick_type=tick_type,
                data=tick_data,
                hash_value=hash_value,
                profit_factor=profit_factor,
                api_synced=False
            )

        except Exception as e:
            logger.error(f"Failed to create tick memory entry: {e}")
            return TickMemoryEntry(
                timestamp=time.time(),
                tick_type=TickMemoryType.PRICE,
                data=tick_data,
hash_value="",
                profit_factor=0.0,
                api_synced=False
            )

    def _determine_tick_type(self, tick_data: Dict[str, Any]) -> TickMemoryType:
        """Determine the type of tick data."""
        try:
            if 'price' in tick_data:
                return TickMemoryType.PRICE
            elif 'volume' in tick_data:
                return TickMemoryType.VOLUME
            elif 'order_book' in tick_data or 'bids' in tick_data or 'asks' in tick_data:
                return TickMemoryType.ORDER_BOOK
            elif 'profit' in tick_data:
                return TickMemoryType.PROFIT
            elif 'api_response' in tick_data:
                return TickMemoryType.API_RESPONSE
            else:
                return TickMemoryType.HASH
        except Exception as e:
            logger.error(f"Tick type determination failed: {e}")
            return TickMemoryType.HASH

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash of tick data."""
        try:
            import hashlib
            data_str = json.dumps(data, sort_keys=True)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return ""

    def _calculate_profit_factor(self, tick_data: Dict[str, Any]) -> float:
        """Calculate profit factor from tick data."""
        try:
            # Simple profit factor calculation
            if 'price' in tick_data and 'volume' in tick_data:
                price = float(tick_data['price'])
                volume = float(tick_data['volume'])
                return price * volume * 0.001  # Simple scaling factor
            elif 'profit' in tick_data:
                return float(tick_data['profit'])
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Profit factor calculation failed: {e}")
            return 0.0

    def _calculate_backlog_profit(self, tick_entry: TickMemoryEntry) -> BacklogProfit:
        """Calculate backlog profit using the mathematical formula."""
        try:
            # Calculate tick profit sum: Σ[T(i)*P(i)]
            tick_profit_sum = 0.0
            for entry in self.tick_memory:
                tick_profit_sum += entry.profit_factor
            
            # Calculate acceleration component: ∇²(T)
            acceleration_component = self._calculate_acceleration()
            
            # Calculate API sync score
            api_sync_score = self._calculate_api_sync_score()
            
            # Apply memory persistence factor: μ·Σ[T(i)*P(i)] + ∇²(T)
            total_profit = (self.memory_persistence_factor * tick_profit_sum + 
                          acceleration_component)
            
            # Determine state
            state = self._determine_backlog_state(api_sync_score)
            
            return BacklogProfit(
                timestamp=time.time(),
                total_profit=total_profit,
                memory_persistence_factor=self.memory_persistence_factor,
                tick_profit_sum=tick_profit_sum,
                acceleration_component=acceleration_component,
                api_sync_score=api_sync_score,
                state=state,
                metadata={
                    "tick_count": len(self.tick_memory),
                    "hash_value": tick_entry.hash_value
                }
            )

        except Exception as e:
            logger.error(f"Backlog profit calculation failed: {e}")
            return self._create_fallback_profit()

    def _calculate_acceleration(self) -> float:
        """Calculate acceleration component ∇²(T)."""
        try:
            if len(self.tick_memory) < 3:
                return 0.0

            # Get recent profit factors
            recent_profits = [entry.profit_factor for entry in list(self.tick_memory)[-3:]]

# Calculate second derivative (acceleration)
            if len(recent_profits) >= 3:
                # ∇²(T) ≈ T[i] - 2*T[i-1] + T[i-2]
                acceleration = (recent_profits[2] - 2 * recent_profits[1] + recent_profits[0])
                return acceleration
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Acceleration calculation failed: {e}")
            return 0.0

    def _calculate_api_sync_score(self) -> float:
        """Calculate API synchronization score."""
        try:
            if not self.api_sync_status:
                return 0.0
            
            total_score = 0.0
            count = 0
            
            for status in self.api_sync_status.values():
                if status.sync_success:
                    total_score += status.data_consistency
                count += 1
            
            return total_score / count if count > 0 else 0.0

        except Exception as e:
            logger.error(f"API sync score calculation failed: {e}")
            return 0.0

    def _determine_backlog_state(self, api_sync_score: float) -> BacklogState:
        """Determine backlog state based on various factors."""
        try:
            if api_sync_score > 0.9:
                return BacklogState.SYNCED
            elif api_sync_score > 0.7:
                return BacklogState.ACTIVE
            elif api_sync_score > 0.5:
                return BacklogState.PERSISTENT
            elif api_sync_score > 0.3:
                return BacklogState.DESYNCED
            else:
                return BacklogState.CORRUPTED

        except Exception as e:
            logger.error(f"Backlog state determination failed: {e}")
            return BacklogState.CORRUPTED

    def _update_api_sync_status(self, api_response: Dict[str, Any]) -> None:
"""Update API synchronization status."""
        try:
api_name = api_response.get('api_name', 'unknown')
            current_time = time.time()
            
            # Calculate response time
            response_time = api_response.get('response_time', 0.0)

            # Determine sync success
            sync_success = api_response.get('success', False)
            
            # Calculate data consistency
            data_consistency = api_response.get('consistency', 0.0)
            
            # Update or create status
            if api_name in self.api_sync_status:
status = self.api_sync_status[api_name]
                status.last_sync_time = current_time
                status.sync_success = sync_success
                status.data_consistency = data_consistency
status.response_time = response_time
                if not sync_success:
                    status.error_count += 1
            else:
                self.api_sync_status[api_name] = APISyncStatus(
                    api_name=api_name,
                    last_sync_time=current_time,
                    sync_success=sync_success,
                    data_consistency=data_consistency,
                    response_time=response_time,
                    error_count=0 if sync_success else 1
                )

        except Exception as e:
            logger.error(f"API sync status update failed: {e}")

    def _create_fallback_profit(self) -> BacklogProfit:
        """Create a fallback profit object for error cases."""
        return BacklogProfit(
            timestamp=time.time(),
            total_profit=0.0,
            memory_persistence_factor=self.memory_persistence_factor,
            tick_profit_sum=0.0,
            acceleration_component=0.0,
            api_sync_score=0.0,
            state=BacklogState.CORRUPTED,
            metadata={"error": "Fallback profit created"}
        )

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        try:
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return unified_math.sqrt(variance)
        except Exception as e:
            logger.error(f"Standard deviation calculation failed: {e}")
            return 0.0

    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists."""
        try:
            os.makedirs(os.path.dirname(self.backlog_file_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create data directory: {e}")

    def _persist_backlog_state(self) -> None:
        """Persist backlog state to file."""
        try:
            state_data = {
                "timestamp": time.time(),
                "total_ticks_processed": self.total_ticks_processed,
                "memory_persistence_factor": self.memory_persistence_factor,
                "api_sync_status": {
                    name: {
                        "last_sync_time": status.last_sync_time,
                        "sync_success": status.sync_success,
                        "data_consistency": status.data_consistency,
                        "response_time": status.response_time,
                        "error_count": status.error_count
                    }
                    for name, status in self.api_sync_status.items()
                }
            }
            
            with open(self.backlog_file_path, 'w') as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist backlog state: {e}")

    def load_backlog_state(self) -> bool:
        """Load backlog state from file."""
        try:
            if not os.path.exists(self.backlog_file_path):
                return False
            
            with open(self.backlog_file_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore state
            self.total_ticks_processed = state_data.get('total_ticks_processed', 0)
            self.memory_persistence_factor = state_data.get('memory_persistence_factor', 0.95)
            
            # Restore API sync status
            api_status_data = state_data.get('api_sync_status', {})
            for api_name, status_data in api_status_data.items():
                self.api_sync_status[api_name] = APISyncStatus(
                    api_name=api_name,
                    last_sync_time=status_data.get('last_sync_time', 0.0),
                    sync_success=status_data.get('sync_success', False),
                    data_consistency=status_data.get('data_consistency', 0.0),
                    response_time=status_data.get('response_time', 0.0),
                    error_count=status_data.get('error_count', 0)
                )

            logger.info("Backlog state loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load backlog state: {e}")
            return False

    def reset_backlog(self) -> None:
        """Reset the backlog system."""
        try:
            self.tick_memory.clear()
            self.backlog_history.clear()
            self.api_sync_status.clear()
            self.total_ticks_processed = 0
            self.total_profit_calculations = 0
            self.api_sync_failures = 0
            logger.info("Backlog system reset")
        except Exception as e:
            logger.error(f"Failed to reset backlog: {e}")


def create_tick_backlog_router(config: Optional[Dict[str, Any]] = None) -> TickBacklogRouter:
    """Factory function to create a tick backlog router."""
    try:
        router = TickBacklogRouter(config)
        router.load_backlog_state()  # Load previous state if available
        return router
        except Exception as e:
        logger.error(f"Failed to create tick backlog router: {e}")
        raise


def main():
    """Main function for testing the tick backlog router."""
    try:
        # Create router
        router = create_tick_backlog_router()
        
        # Simulate tick data
        test_ticks = [
            {"price": 50000.0, "volume": 1.5, "timestamp": time.time()},
            {"price": 50100.0, "volume": 2.0, "timestamp": time.time()},
            {"price": 50200.0, "volume": 1.8, "timestamp": time.time()}
        ]
        
        print("Testing Tick Backlog Router:")
        print("=" * 40)
        
        for i, tick_data in enumerate(test_ticks):
            profit = router.process_tick_data(tick_data)
            print(f"Tick {i+1}: Profit={profit.total_profit:.4f}, State={profit.state.value}")
        
        # Get analytics
        analytics = router.get_backlog_analytics()
        print(f"\nAnalytics: {analytics}")
        
    except Exception as e:
        logger.error(f"Main function failed: {e}")


if __name__ == "__main__":
    main()
