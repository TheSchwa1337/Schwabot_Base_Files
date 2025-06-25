from __future__ import annotations
import math

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
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
"""Demo Memory Core - In-Memory Simulation Pool for Self-Trade Testing.

This module provides in-memory simulation capabilities that enable Schwabot
to validate its own logic through recursive memory and historical data,
creating a self-referential testing environment.
"""


import asyncio
import logging
# from core.unified_math_system import unified_math  # F811: duplicate import
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
# from core.unified_math_system import unified_math  # F811: duplicate import

# Import unified mathematics
try:
    from core.unified_mathematics_config import get_unified_math
unified_math = get_unified_math()
    UNIFIED_MATH_AVAILABLE = True
except ImportError:
UNIFIED_MATH_AVAILABLE = False

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, safe_format_error, log_safe

CLI_HANDLER_AVAILABLE = True
except ImportError:
CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Memory types for different storage strategies."""
SHORT_TERM = "short_term"    # 16-bit memory for momentum
MID_TERM = "mid_term"        # 256-bit memory for patterns
LONG_TERM = "long_term"      # 10k-bit memory for cycles
LANTERN = "lantern"          # Textual hash memory


class SimulationMode(Enum):
    """Simulation modes for different testing scenarios."""
HISTORICAL = "historical"    # Use historical ledger data
SYNTHETIC = "synthetic"      # Generate synthetic data
HYBRID = "hybrid"            # Mix historical and synthetic
ADAPTIVE = "adaptive"        # Adaptive based on performance


@dataclass
class MemoryEntry:
    """Memory entry for storing trade and market data."""
tick_id: int
timestamp: datetime
market_data: Dict[str, Any]
trade_data: Dict[str, Any]
profit_result: float
strategy_used: str
phase_compression: float
entropy_field: float
zpe_resonance: float
memory_type: MemoryType
hash_id: str = ""
confidence_score: float = 0.0


@dataclass
class SimulationMemory:
    """Simulation memory pool for self-trade testing."""
short_term_memory: Dict[int, MemoryEntry] = field(default_factory=dict)
    mid_term_memory: Dict[int, MemoryEntry] = field(default_factory=dict)
    long_term_memory: Dict[str, MemoryEntry] = field(default_factory=dict)
    lantern_memory: Dict[str, MemoryEntry] = field(default_factory=dict)

    # Memory limits
short_term_limit: int = 65536  # 16-bit memory
mid_term_limit: int = 16777216  # 24-bit memory
long_term_limit: int = 10000    # 10k entries
lantern_limit: int = 5000       # 5k textual entries


class DemoMemoryCore:
    """
Demo Memory Core - In-memory simulation pool for self-trade testing.

Enables Schwabot to:
- Store and retrieve trade memory for validation
- Use historical data for simulation
- Self-validate through recursive memory
- Apply memory-based learning to improve strategies
"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize demo memory core."""
self.config = config or {}
self.simulation_mode = SimulationMode.HYBRID
self.memory = SimulationMemory()

        # Performance tracking
self.total_entries = 0
self.memory_hits = 0
self.memory_misses = 0

        # Memory management
self.auto_cleanup = True
self.cleanup_interval = 1000  # Cleanup every 1000 entries

safe_safe_print("🧠 Demo Memory Core initialized")

    def set_simulation_mode(self, mode: SimulationMode) -> None:
        """Set simulation mode."""
self.simulation_mode = mode
safe_safe_print(f"🔄 Simulation mode set to: {mode.value}")

    def store_memory_entry(
        self,
tick_id: int,
timestamp: datetime,
market_data: Dict[str, Any],
trade_data: Dict[str, Any],
profit_result: float,
strategy_used: str,
phase_compression: float,
entropy_field: float,
zpe_resonance: float,
memory_type: MemoryType = MemoryType.SHORT_TERM
) -> str:
"""
Store memory entry for future reference and learning.

This is the core function that enables Schwabot to learn from
its own trading history and improve future decisions.
"""
        try:
            # Generate hash ID
hash_id = self._generate_hash_id(tick_id, timestamp, market_data)

            # Create memory entry
entry = MemoryEntry(
                tick_id=tick_id,
timestamp=timestamp,
market_data=market_data,
trade_data=trade_data,
profit_result=profit_result,
strategy_used=strategy_used,
phase_compression=phase_compression,
entropy_field=entropy_field,
zpe_resonance=zpe_resonance,
memory_type=memory_type,
hash_id=hash_id,
confidence_score=self._calculate_confidence_score(
                    profit_result, phase_compression, entropy_field, zpe_resonance



            # Store based on memory type
            if memory_type == MemoryType.SHORT_TERM:
self.memory.short_term_memory[tick_id % self.memory.short_term_limit] = entry
            elif memory_type == MemoryType.MID_TERM:
self.memory.mid_term_memory[tick_id % self.memory.mid_term_limit] = entry
            elif memory_type == MemoryType.LONG_TERM:
                if len(self.memory.long_term_memory) < self.memory.long_term_limit:
                    self.memory.long_term_memory[hash_id] = entry
            elif memory_type == MemoryType.LANTERN:
                if len(self.memory.lantern_memory) < self.memory.lantern_limit:
                    self.memory.lantern_memory[hash_id] = entry

self.total_entries += 1

            # Auto cleanup if enabled
            if self.auto_cleanup and self.total_entries % self.cleanup_interval == 0:
self._cleanup_memory()

safe_safe_print(f"✅ Memory entry stored: {hash_id[:8]}...")
            return hash_id

        except Exception as e:
safe_safe_print(f"❌ Memory storage failed: {safe_format_error(e, 'memory_storage')}")
            return ""

    def retrieve_memory_entry(
        self,
tick_id: Optional[int] = None,
hash_id: Optional[str] = None,
memory_type: Optional[MemoryType] = None,
market_conditions: Optional[Dict[str, Any]] = None
) -> Optional[MemoryEntry]:
"""
Retrieve memory entry based on various criteria.

This enables Schwabot to find relevant historical data
        for current decision-making.
"""
        try:
            # Direct lookup by tick_id or hash_id
            if tick_id is not None:
                if memory_type == MemoryType.SHORT_TERM:
                    return self.memory.short_term_memory.get(tick_id % self.memory.short_term_limit)
                elif memory_type == MemoryType.MID_TERM:
                    return self.memory.mid_term_memory.get(tick_id % self.memory.mid_term_limit)

            if hash_id is not None:
                if memory_type == MemoryType.LONG_TERM:
                    return self.memory.long_term_memory.get(hash_id)
                elif memory_type == MemoryType.LANTERN:
                    return self.memory.lantern_memory.get(hash_id)

            # Similarity search based on market conditions
            if market_conditions is not None:
                return self._find_similar_memory(market_conditions, memory_type)

            return None

        except Exception as e:
safe_safe_print(f"❌ Memory retrieval failed: {safe_format_error(e, 'memory_retrieval')}")
            return None

    def _generate_hash_id(self, tick_id: int, timestamp: datetime, market_data: Dict[str, Any]) -> str:
        """Generate hash ID for memory entry."""
        try:
            import hashlib

            # Create hash data
hash_data = f"{tick_id}_{timestamp.isoformat()}_{str(sorted(market_data.items()))}"

            # Generate hash
hash_object = hashlib.sha256(hash_data.encode())
            return hash_object.hexdigest()

        except Exception as e:
safe_safe_print(f"⚠️ Hash generation failed: {safe_format_error(e, 'hash_generation')}")
            return f"fallback_{tick_id}_{int(time.time())}"

    def _calculate_confidence_score(
        self,
profit_result: float,
phase_compression: float,
entropy_field: float,
zpe_resonance: float
) -> float:
"""Calculate confidence score for memory entry."""
        try:
            # Profit-based confidence
profit_confidence = unified_math.min(1.0, unified_math.max(0.0, profit_result / 100.0))

            # Phase alignment confidence
phase_confidence = 1.0 - unified_math.abs(phase_compression)

            # Entropy stability confidence
entropy_confidence = 1.0 - unified_math.abs(entropy_field - 0.5) * 2.0

            # ZPE resonance confidence
resonance_confidence = unified_math.abs(zpe_resonance)

            # Combined confidence
confidence = (profit_confidence + phase_confidence +
                         entropy_confidence + resonance_confidence) / 4.0

            return unified_math.min(1.0, unified_math.max(0.0, confidence))

        except Exception as e:
safe_safe_print(f"⚠️ Confidence calculation failed: {safe_format_error(e, 'confidence_calculation')}")
            return 0.5

    def _find_similar_memory(
        self,
market_conditions: Dict[str, Any],
memory_type: Optional[MemoryType]
) -> Optional[MemoryEntry]:
"""Find similar memory entry based on market conditions."""
        try:
best_match = None
best_score = 0.0

            # Determine which memory pool to search
memory_pools = []
            if memory_type == MemoryType.SHORT_TERM:
memory_pools = [self.memory.short_term_memory]
            elif memory_type == MemoryType.MID_TERM:
memory_pools = [self.memory.mid_term_memory]
            elif memory_type == MemoryType.LONG_TERM:
memory_pools = [self.memory.long_term_memory]
            elif memory_type == MemoryType.LANTERN:
memory_pools = [self.memory.lantern_memory]
            else:
                # Search all pools
memory_pools = [
self.memory.short_term_memory,
self.memory.mid_term_memory,
self.memory.long_term_memory,
self.memory.lantern_memory
]

            # Search for best match
            for memory_pool in memory_pools:
                for entry in memory_pool.values():
                    similarity_score = self._calculate_similarity_score(
                        market_conditions, entry.market_data


                    if similarity_score > best_score:
best_score = similarity_score
best_match = entry

            # Only return if similarity is above threshold
            if best_score > 0.7:
self.memory_hits += 1
                return best_match
            else:
self.memory_misses += 1
                return None

        except Exception as e:
safe_safe_print(f"⚠️ Similarity search failed: {safe_format_error(e, 'similarity_search')}")
            return None

    def _calculate_similarity_score(
        self,
current_conditions: Dict[str, Any],
historical_conditions: Dict[str, Any]
) -> float:
"""Calculate similarity score between current and historical conditions."""
        try:
score = 0.0
total_factors = 0

            # Compare price factors
            for asset in ['btc_price', 'eth_price', 'xrp_price']:
                if asset in current_conditions and asset in historical_conditions:
current_price = current_conditions[asset]
historical_price = historical_conditions[asset]

                    if historical_price > 0:
price_diff = unified_math.abs(current_price - historical_price) / historical_price
                        price_similarity = unified_math.max(0.0, 1.0 - price_diff)
                        score += price_similarity
total_factors += 1

            # Compare volume factors
            for asset in ['volume_btc', 'volume_eth', 'volume_xrp']:
                if asset in current_conditions and asset in historical_conditions:
current_volume = current_conditions[asset]
historical_volume = historical_conditions[asset]

                    if historical_volume > 0:
volume_diff = unified_math.abs(current_volume - historical_volume) / historical_volume
                        volume_similarity = unified_math.max(0.0, 1.0 - volume_diff)
                        score += volume_similarity
total_factors += 1

            # Return average similarity
            return score / unified_math.max(total_factors, 1)

        except Exception as e:
safe_safe_print(f"⚠️ Similarity calculation failed: {safe_format_error(e, 'similarity_calculation')}")
            return 0.0

    def _cleanup_memory(self) -> None:
        """Clean up old memory entries."""
        try:
            # Remove low-confidence entries from long-term memory
low_confidence_entries = [
hash_id for hash_id, entry in self.memory.long_term_memory.items()
                if entry.confidence_score < 0.3
]

            for hash_id in low_confidence_entries:
                del self.memory.long_term_memory[hash_id]

            # Remove old entries from lantern memory
current_time = datetime.now()
            old_lantern_entries = [
hash_id for hash_id, entry in self.memory.lantern_memory.items()
                if (current_time - entry.timestamp).days > 30
            ]

            for hash_id in old_lantern_entries:
                del self.memory.lantern_memory[hash_id]

safe_safe_print(f"🗑️ Memory cleanup completed: {len(low_confidence_entries)} long-term, {len(old_lantern_entries)} lantern entries removed")

        except Exception as e:
safe_safe_print(f"⚠️ Memory cleanup failed: {safe_format_error(e, 'memory_cleanup')}")

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
'total_entries': self.total_entries,
'memory_hits': self.memory_hits,
'memory_misses': self.memory_misses,
'hit_rate': self.memory_hits / unified_math.max(self.memory_hits + self.memory_misses, 1),
            'short_term_size': len(self.memory.short_term_memory),
            'mid_term_size': len(self.memory.mid_term_memory),
            'long_term_size': len(self.memory.long_term_memory),
            'lantern_size': len(self.memory.lantern_memory),
            'simulation_mode': self.simulation_mode.value
}

    def clear_memory(self, memory_type: Optional[MemoryType] = None) -> None:
        """Clear memory entries."""
        try:
            if memory_type is None:
                # Clear all memory
self.memory.short_term_memory.clear()
                self.memory.mid_term_memory.clear()
                self.memory.long_term_memory.clear()
                self.memory.lantern_memory.clear()
                safe_safe_print("🗑️ All memory cleared")
            else:
                # Clear specific memory type
                if memory_type == MemoryType.SHORT_TERM:
self.memory.short_term_memory.clear()
                elif memory_type == MemoryType.MID_TERM:
self.memory.mid_term_memory.clear()
                elif memory_type == MemoryType.LONG_TERM:
self.memory.long_term_memory.clear()
                elif memory_type == MemoryType.LANTERN:
self.memory.lantern_memory.clear()
                safe_safe_print(f"🗑️ {memory_type.value} memory cleared")

        except Exception as e:
safe_safe_print(f"⚠️ Memory clear failed: {safe_format_error(e, 'memory_clear')}")


# Global demo memory core instance
demo_memory_core = DemoMemoryCore()


# Convenience functions for external access
def get_demo_memory_core() -> DemoMemoryCore:
    """Get global demo memory core instance."""
    return demo_memory_core


def store_memory_entry(
    tick_id: int,
timestamp: datetime,
market_data: Dict[str, Any],
trade_data: Dict[str, Any],
profit_result: float,
strategy_used: str,
phase_compression: float,
entropy_field: float,
zpe_resonance: float,
memory_type: MemoryType = MemoryType.SHORT_TERM
) -> str:
"""Store memory entry."""
    return demo_memory_core.store_memory_entry(
        tick_id, timestamp, market_data, trade_data, profit_result,
strategy_used, phase_compression, entropy_field, zpe_resonance, memory_type



def retrieve_memory_entry(
    tick_id: Optional[int] = None,
hash_id: Optional[str] = None,
memory_type: Optional[MemoryType] = None,
market_conditions: Optional[Dict[str, Any]] = None
) -> Optional[MemoryEntry]:
"""Retrieve memory entry."""
    return demo_memory_core.retrieve_memory_entry(
        tick_id, hash_id, memory_type, market_conditions



def get_memory_stats() -> Dict[str, Any]:
    """Get memory statistics."""
    return demo_memory_core.get_memory_statistics()


# Example usage

if __name__ == "__main__":
    # Test demo memory core
safe_print("🧪 Testing Demo Memory Core...")

    # Test market data
test_market_data = {
'btc_price': 50000.0,
'eth_price': 3000.0,
'xrp_price': 0.5,
'volume_btc': 1000.0,
'volume_eth': 500.0,
'volume_xrp': 100.0
}

test_trade_data = {
'strategy': 'momentum',
'volume': 100.0,
'entry_price': 50000.0,
'exit_price': 50100.0
}

    # Store memory entry
hash_id = store_memory_entry(
        tick_id=1,
timestamp=datetime.now(),
        market_data=test_market_data,
trade_data=test_trade_data,
profit_result=50.0,
strategy_used='momentum',
phase_compression=0.8,
entropy_field=0.6,
zpe_resonance=0.7,
memory_type=MemoryType.SHORT_TERM


safe_print(f"✅ Memory entry stored: {hash_id}")

    # Retrieve memory entry
retrieved_entry = retrieve_memory_entry(
        tick_id=1,
memory_type=MemoryType.SHORT_TERM


    if retrieved_entry:
safe_print(f"✅ Memory entry retrieved: {retrieved_entry.hash_id[:8]}...")
        safe_print(f"   Profit Result: {retrieved_entry.profit_result}")
        safe_print(f"   Confidence Score: {retrieved_entry.confidence_score:.3f}")

    # Get statistics
stats = get_memory_stats()
    safe_print(f"✅ Memory Statistics: {stats}")
