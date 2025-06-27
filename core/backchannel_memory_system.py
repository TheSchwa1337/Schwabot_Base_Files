# -*- coding: utf-8 -*-
"""
Backchannel Memory System

This module provides a comprehensive backchannel memory system that saves information
from individual states, collapses, and prints for market entry/exit logic with volume
and stop loss across CCXT and Coinbase integration.

Features:
- State persistence and recovery
- Memory pattern recognition
- Performance analysis and optimization
- Integration with CCXT and Coinbase
- Volume and stop loss management
- Recursive memory improvement
"""

from __future__ import annotations

import json
import logging
import gzip
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PATTERN = "pattern"
    PERFORMANCE = "performance"
    STATE = "state"
    COLLAPSE = "collapse"
    PRINT = "print"


class MemoryCategory(Enum):
    """Categories of memory data."""
    PROFIT_STATES = "profit_states"
    MARKET_CONDITIONS = "market_conditions"
    ENGINE_PERFORMANCE = "engine_performance"
    ERROR_LOGS = "error_logs"
    TRADING_DECISIONS = "trading_decisions"
    MEMORY_PATTERNS = "memory_patterns"
    VOLUME_ANALYSIS = "volume_analysis"
    STOP_LOSS_EVENTS = "stop_loss_events"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    entry_id: str
    memory_type: MemoryType
    category: MemoryCategory
    data: Dict[str, Any]
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """Snapshot of system state."""
    state_id: str
    timestamp: datetime
    profit_state: Dict[str, Any]
    market_conditions: Dict[str, Any]
    engine_performance: Dict[str, Any]
    trading_decisions: Dict[str, Any]
    volume_data: Dict[str, Any]
    stop_loss_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollapseEvent:
    """Market collapse event."""
    collapse_id: str
    timestamp: datetime
    trigger_symbol: str
    trigger_hash: str
    collapse_magnitude: float
    affected_assets: List[str]
    recovery_time: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrintEvent:
    """Print event for market entry/exit."""
    print_id: str
    timestamp: datetime
    event_type: str  # "entry", "exit", "adjustment"
    symbol: str
    price: float
    volume: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackchannelMemorySystem:
    """
    Backchannel memory system for Schwabot.

    This class manages the storage, retrieval, and analysis of memory data
    for the trading system, including state snapshots, collapse events,
    and print events for market entry/exit logic.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backchannel memory system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Memory storage configuration
        self.storage_config = self.config.get("storage", {
            "type": "json",
            "file_path": "backchannel/memory_stack",
            "max_file_size": "100MB",
            "compression": True,
            "encryption": False
        })

        # State management configuration
        self.state_config = self.config.get("states", {
            "save_interval": 60,
            "max_states": 10000,
            "state_compression": True
        })

        # Initialize storage
        self.storage_path = Path(self.storage_config["file_path"])
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Memory storage
        self.memory_entries: List[MemoryEntry] = []
        self.state_snapshots: List[StateSnapshot] = []
        self.collapse_events: List[CollapseEvent] = []
        self.print_events: List[PrintEvent] = []

        # Performance tracking
        self.total_entries_stored = 0
        self.total_states_saved = 0
        self.total_collapses_recorded = 0
        self.total_prints_logged = 0

        # Pattern recognition
        self.pattern_memory: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}

        # Load existing memory
        self._load_existing_memory()

        logger.info("🧠 Backchannel Memory System initialized")

    def _load_existing_memory(self) -> None:
        """Load existing memory from storage."""
        try:
            # Load memory entries
            memory_file = self.storage_path / "memory_entries.json.gz"
            if memory_file.exists():
                with gzip.open(memory_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = MemoryEntry(
                            entry_id=entry_data["entry_id"],
                            memory_type=MemoryType(entry_data["memory_type"]),
                            category=MemoryCategory(entry_data["category"]),
                            data=entry_data["data"],
                            timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                            importance=entry_data.get("importance", 0.5),
                            metadata=entry_data.get("metadata", {})
                        )
                        self.memory_entries.append(entry)
                logger.info(
                    f"✅ Loaded {len(self.memory_entries)} memory entries")

            # Load state snapshots
            states_file = self.storage_path / "state_snapshots.json.gz"
            if states_file.exists():
                with gzip.open(states_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    for state_data in data:
                        state = StateSnapshot(
                            state_id=state_data["state_id"],
                            timestamp=datetime.fromisoformat(
                                state_data["timestamp"]),
                            profit_state=state_data["profit_state"],
                            market_conditions=state_data["market_conditions"],
                            engine_performance=state_data["engine_performance"],
                            trading_decisions=state_data["trading_decisions"],
                            volume_data=state_data["volume_data"],
                            stop_loss_data=state_data["stop_loss_data"],
                            metadata=state_data.get(
                                "metadata",
                                {}))
                        self.state_snapshots.append(state)
                logger.info(
                    f"✅ Loaded {len(self.state_snapshots)} state snapshots")

            # Load collapse events
            collapses_file = self.storage_path / "collapse_events.json.gz"
            if collapses_file.exists():
                with gzip.open(collapses_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    for collapse_data in data:
                        collapse = CollapseEvent(
                            collapse_id=collapse_data["collapse_id"],
                            timestamp=datetime.fromisoformat(
                                collapse_data["timestamp"]),
                            trigger_symbol=collapse_data["trigger_symbol"],
                            trigger_hash=collapse_data["trigger_hash"],
                            collapse_magnitude=collapse_data["collapse_magnitude"],
                            affected_assets=collapse_data["affected_assets"],
                            recovery_time=timedelta(
                                seconds=collapse_data["recovery_time"]) if collapse_data.get("recovery_time") else None,
                            metadata=collapse_data.get(
                                "metadata",
                                {}))
                        self.collapse_events.append(collapse)
                logger.info(
                    f"✅ Loaded {len(self.collapse_events)} collapse events")

            # Load print events
            prints_file = self.storage_path / "print_events.json.gz"
            if prints_file.exists():
                with gzip.open(prints_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    for print_data in data:
                        print_event = PrintEvent(
                            print_id=print_data["print_id"],
                            timestamp=datetime.fromisoformat(
                                print_data["timestamp"]),
                            event_type=print_data["event_type"],
                            symbol=print_data["symbol"],
                            price=print_data["price"],
                            volume=print_data["volume"],
                            stop_loss=print_data.get("stop_loss"),
                            take_profit=print_data.get("take_profit"),
                            confidence=print_data["confidence"],
                            metadata=print_data.get(
                                "metadata",
                                {}))
                        self.print_events.append(print_event)
                logger.info(f"✅ Loaded {len(self.print_events)} print events")

        except Exception as e:
            logger.error(f"❌ Error loading existing memory: {e}")

    def save_memory_entry(
        self,
        memory_type: MemoryType,
        category: MemoryCategory,
        data: Dict[str, Any],
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a memory entry.

        Args:
            memory_type: Type of memory
            category: Category of memory
            data: Memory data
            importance: Importance level (0.0 to 1.0)
            metadata: Additional metadata

        Returns:
            Entry ID
        """
        entry_id = self._generate_entry_id(memory_type, category, data)

        entry = MemoryEntry(
            entry_id=entry_id,
            memory_type=memory_type,
            category=category,
            data=data,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )

        self.memory_entries.append(entry)
        self.total_entries_stored += 1

        # Auto-save if configured
        if self.config.get("auto_save", True):
            self._save_memory_to_storage()

        logger.debug(f"💾 Memory entry saved: {entry_id}")
        return entry_id

    def save_state_snapshot(
        self,
        profit_state: Dict[str, Any],
        market_conditions: Dict[str, Any],
        engine_performance: Dict[str, Any],
        trading_decisions: Dict[str, Any],
        volume_data: Dict[str, Any],
        stop_loss_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a state snapshot.

        Args:
            profit_state: Current profit state
            market_conditions: Current market conditions
            engine_performance: Engine performance data
            trading_decisions: Trading decision data
            volume_data: Volume analysis data
            stop_loss_data: Stop loss data
            metadata: Additional metadata

        Returns:
            State ID
        """
        state_id = self._generate_state_id(profit_state, market_conditions)

        state = StateSnapshot(
            state_id=state_id,
            timestamp=datetime.now(),
            profit_state=profit_state,
            market_conditions=market_conditions,
            engine_performance=engine_performance,
            trading_decisions=trading_decisions,
            volume_data=volume_data,
            stop_loss_data=stop_loss_data,
            metadata=metadata or {}
        )

        self.state_snapshots.append(state)
        self.total_states_saved += 1

        # Limit state snapshots
        max_states = self.state_config.get("max_states", 10000)
        if len(self.state_snapshots) > max_states:
            self.state_snapshots.pop(0)

        # Auto-save if configured
        if self.config.get("auto_save", True):
            self._save_states_to_storage()

        logger.debug(f"📸 State snapshot saved: {state_id}")
        return state_id

    def record_collapse_event(
        self,
        trigger_symbol: str,
        trigger_hash: str,
        collapse_magnitude: float,
        affected_assets: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a collapse event.

        Args:
            trigger_symbol: Symbol that triggered the collapse
            trigger_hash: Hash of the trigger
            collapse_magnitude: Magnitude of the collapse
            affected_assets: List of affected assets
            metadata: Additional metadata

        Returns:
            Collapse ID
        """
        collapse_id = self._generate_collapse_id(trigger_symbol, trigger_hash)

        collapse = CollapseEvent(
            collapse_id=collapse_id,
            timestamp=datetime.now(),
            trigger_symbol=trigger_symbol,
            trigger_hash=trigger_hash,
            collapse_magnitude=collapse_magnitude,
            affected_assets=affected_assets,
            metadata=metadata or {}
        )

        self.collapse_events.append(collapse)
        self.total_collapses_recorded += 1

        # Auto-save if configured
        if self.config.get("auto_save", True):
            self._save_collapses_to_storage()

        logger.info(
            f"💥 Collapse event recorded: {collapse_id} (magnitude: {
                collapse_magnitude:.4f})")
        return collapse_id

    def log_print_event(
        self,
        event_type: str,
        symbol: str,
        price: float,
        volume: float,
        confidence: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a print event for market entry/exit.

        Args:
            event_type: Type of event ("entry", "exit", "adjustment")
            symbol: Trading symbol
            price: Price at event
            volume: Volume at event
            confidence: Confidence level
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional metadata

        Returns:
            Print ID
        """
        print_id = self._generate_print_id(event_type, symbol, price)

        print_event = PrintEvent(
            print_id=print_id,
            timestamp=datetime.now(),
            event_type=event_type,
            symbol=symbol,
            price=price,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            metadata=metadata or {}
        )

        self.print_events.append(print_event)
        self.total_prints_logged += 1

        # Auto-save if configured
        if self.config.get("auto_save", True):
            self._save_prints_to_storage()

        logger.info(
            f"🖨️ Print event logged: {print_id} ({event_type} {symbol} @ {price:.4f})")
        return print_id

    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory patterns for optimization."""
        try:
            patterns = {}

            # Analyze profit patterns
            profit_entries = [
                e for e in self.memory_entries if e.category == MemoryCategory.PROFIT_STATES]
            if profit_entries:
                profit_data = [e.data.get("profit", 0) for e in profit_entries]
                patterns["profit"] = {
                    "mean": np.mean(profit_data),
                    "std": np.std(profit_data),
                    "min": np.min(profit_data),
                    "max": np.max(profit_data),
                    "trend": self._calculate_trend(profit_data)
                }

            # Analyze trading decision patterns
            decision_entries = [
                e for e in self.memory_entries if e.category == MemoryCategory.TRADING_DECISIONS]
            if decision_entries:
                decision_types = [
                    e.data.get(
                        "decision_type",
                        "unknown") for e in decision_entries]
                patterns["trading_decisions"] = {
                    "total_decisions": len(decision_entries),
                    "decision_types": self._count_occurrences(decision_types),
                    "success_rate": self._calculate_success_rate(decision_entries)}

            # Analyze collapse patterns
            if self.collapse_events:
                collapse_magnitudes = [
                    c.collapse_magnitude for c in self.collapse_events]
                patterns["collapses"] = {
                    "total_collapses": len(self.collapse_events),
                    "mean_magnitude": np.mean(collapse_magnitudes),
                    "max_magnitude": np.max(collapse_magnitudes),
                    "recovery_patterns": self._analyze_recovery_patterns()
                }

            # Analyze print event patterns
            if self.print_events:
                entry_events = [
                    p for p in self.print_events if p.event_type == "entry"]
                exit_events = [
                    p for p in self.print_events if p.event_type == "exit"]

                patterns["print_events"] = {
                    "total_prints": len(self.print_events),
                    "entries": len(entry_events),
                    "exits": len(exit_events),
                    "avg_confidence": np.mean([p.confidence for p in self.print_events]),
                    "symbol_distribution": self._count_occurrences([p.symbol for p in self.print_events])
                }

            self.pattern_memory = patterns
            logger.info("🔍 Memory pattern analysis completed")
            return patterns

        except Exception as e:
            logger.error(f"❌ Error analyzing memory patterns: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from memory."""
        try:
            metrics = {
                "memory_stats": {
                    "total_entries": len(self.memory_entries),
                    "total_states": len(self.state_snapshots),
                    "total_collapses": len(self.collapse_events),
                    "total_prints": len(self.print_events),
                    "entries_stored": self.total_entries_stored,
                    "states_saved": self.total_states_saved,
                    "collapses_recorded": self.total_collapses_recorded,
                    "prints_logged": self.total_prints_logged
                },
                "storage_stats": {
                    "storage_path": str(self.storage_path),
                    "storage_size": self._get_storage_size(),
                    "compression_enabled": self.storage_config.get("compression", True)
                },
                "pattern_analysis": self.pattern_memory,
                "recent_activity": self._get_recent_activity()
            }

            self.performance_metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return {}

    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory storage and performance."""
        try:
            optimization_results = {
                "entries_compressed": 0,
                "old_entries_removed": 0,
                "patterns_updated": 0,
                "storage_optimized": False
            }

            # Remove old entries based on importance
            old_entries = [
                e for e in self.memory_entries if e.importance < 0.1]
            self.memory_entries = [
                e for e in self.memory_entries if e.importance >= 0.1]
            optimization_results["old_entries_removed"] = len(old_entries)

            # Update pattern analysis
            self.analyze_memory_patterns()
            optimization_results["patterns_updated"] = 1

            # Optimize storage
            if self.storage_config.get("compression", True):
                self._save_memory_to_storage()
                optimization_results["storage_optimized"] = True

            logger.info("⚡ Memory optimization completed")
            return optimization_results

        except Exception as e:
            logger.error(f"❌ Error optimizing memory: {e}")
            return {"error": str(e)}

    def _generate_entry_id(self,
                           memory_type: MemoryType,
                           category: MemoryCategory,
                           data: Dict[str,
                                      Any]) -> str:
        """Generate unique entry ID."""
        data_str = json.dumps(data, sort_keys=True)
        hash_input = f"{memory_type.value}_{category.value}_{data_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _generate_state_id(
            self, profit_state: Dict[str, Any], market_conditions: Dict[str, Any]) -> str:
        """Generate unique state ID."""
        state_str = json.dumps(profit_state, sort_keys=True) + \
            json.dumps(market_conditions, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    def _generate_collapse_id(
            self,
            trigger_symbol: str,
            trigger_hash: str) -> str:
        """Generate unique collapse ID."""
        collapse_str = f"{trigger_symbol}_{trigger_hash}_{
            datetime.now().isoformat()}"
        return hashlib.sha256(collapse_str.encode()).hexdigest()[:16]

    def _generate_print_id(
            self,
            event_type: str,
            symbol: str,
            price: float) -> str:
        """Generate unique print ID."""
        print_str = f"{event_type}_{symbol}_{
            price:.4f}_{
            datetime.now().isoformat()}"
        return hashlib.sha256(print_str.encode()).hexdigest()[:16]

    def _save_memory_to_storage(self) -> None:
        """Save memory entries to storage."""
        try:
            memory_file = self.storage_path / "memory_entries.json.gz"
            data = []
            for entry in self.memory_entries:
                data.append({
                    "entry_id": entry.entry_id,
                    "memory_type": entry.memory_type.value,
                    "category": entry.category.value,
                    "data": entry.data,
                    "timestamp": entry.timestamp.isoformat(),
                    "importance": entry.importance,
                    "metadata": entry.metadata
                })

            with gzip.open(memory_file, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Error saving memory to storage: {e}")

    def _save_states_to_storage(self) -> None:
        """Save state snapshots to storage."""
        try:
            states_file = self.storage_path / "state_snapshots.json.gz"
            data = []
            for state in self.state_snapshots:
                data.append({
                    "state_id": state.state_id,
                    "timestamp": state.timestamp.isoformat(),
                    "profit_state": state.profit_state,
                    "market_conditions": state.market_conditions,
                    "engine_performance": state.engine_performance,
                    "trading_decisions": state.trading_decisions,
                    "volume_data": state.volume_data,
                    "stop_loss_data": state.stop_loss_data,
                    "metadata": state.metadata
                })

            with gzip.open(states_file, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Error saving states to storage: {e}")

    def _save_collapses_to_storage(self) -> None:
        """Save collapse events to storage."""
        try:
            collapses_file = self.storage_path / "collapse_events.json.gz"
            data = []
            for collapse in self.collapse_events:
                data.append({
                    "collapse_id": collapse.collapse_id,
                    "timestamp": collapse.timestamp.isoformat(),
                    "trigger_symbol": collapse.trigger_symbol,
                    "trigger_hash": collapse.trigger_hash,
                    "collapse_magnitude": collapse.collapse_magnitude,
                    "affected_assets": collapse.affected_assets,
                    "recovery_time": collapse.recovery_time.total_seconds() if collapse.recovery_time else None,
                    "metadata": collapse.metadata
                })

            with gzip.open(collapses_file, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Error saving collapses to storage: {e}")

    def _save_prints_to_storage(self) -> None:
        """Save print events to storage."""
        try:
            prints_file = self.storage_path / "print_events.json.gz"
            data = []
            for print_event in self.print_events:
                data.append({
                    "print_id": print_event.print_id,
                    "timestamp": print_event.timestamp.isoformat(),
                    "event_type": print_event.event_type,
                    "symbol": print_event.symbol,
                    "price": print_event.price,
                    "volume": print_event.volume,
                    "stop_loss": print_event.stop_loss,
                    "take_profit": print_event.take_profit,
                    "confidence": print_event.confidence,
                    "metadata": print_event.metadata
                })

            with gzip.open(prints_file, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Error saving prints to storage: {e}")

    def _calculate_trend(self, data: List[float]) -> str:
        """Calculate trend from data."""
        if len(data) < 2:
            return "insufficient_data"

        slope = np.polyfit(range(len(data)), data, 1)[0]
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _count_occurrences(self, items: List[str]) -> Dict[str, int]:
        """Count occurrences of items."""
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return counts

    def _calculate_success_rate(
            self, decision_entries: List[MemoryEntry]) -> float:
        """Calculate success rate of trading decisions."""
        successful = sum(
            1 for e in decision_entries if e.data.get(
                "success", False))
        total = len(decision_entries)
        return successful / total if total > 0 else 0.0

    def _analyze_recovery_patterns(self) -> Dict[str, Any]:
        """Analyze recovery patterns from collapse events."""
        recoveries = [
            c for c in self.collapse_events if c.recovery_time is not None]
        if not recoveries:
            return {"total_recoveries": 0}

        recovery_times = [r.recovery_time.total_seconds() for r in recoveries]
        return {
            "total_recoveries": len(recoveries),
            "avg_recovery_time": np.mean(recovery_times),
            "min_recovery_time": np.min(recovery_times),
            "max_recovery_time": np.max(recovery_times)
        }

    def _get_storage_size(self) -> str:
        """Get storage size."""
        try:
            total_size = 0
            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

            if total_size < 1024:
                return f"{total_size} B"
            elif total_size < 1024 * 1024:
                return f"{total_size / 1024:.1f} KB"
            else:
                return f"{total_size / (1024 * 1024):.1f} MB"
        except Exception:
            return "unknown"

    def _get_recent_activity(self) -> Dict[str, Any]:
        """Get recent activity summary."""
        now = datetime.now()
        recent_window = timedelta(hours=1)

        recent_entries = [
            e for e in self.memory_entries if now -
            e.timestamp < recent_window]
        recent_states = [
            s for s in self.state_snapshots if now -
            s.timestamp < recent_window]
        recent_collapses = [
            c for c in self.collapse_events if now -
            c.timestamp < recent_window]
        recent_prints = [
            p for p in self.print_events if now -
            p.timestamp < recent_window]

        return {
            "recent_entries": len(recent_entries),
            "recent_states": len(recent_states),
            "recent_collapses": len(recent_collapses),
            "recent_prints": len(recent_prints)
        }


# Global memory system instance
_memory_system: Optional[BackchannelMemorySystem] = None


def get_memory_system() -> BackchannelMemorySystem:
    """Get the global memory system instance."""
    global _memory_system
    if _memory_system is None:
        _memory_system = BackchannelMemorySystem()
    return _memory_system


def initialize_memory_system(
        config: Optional[Dict[str, Any]] = None) -> BackchannelMemorySystem:
    """Initialize the global memory system."""
    global _memory_system
    _memory_system = BackchannelMemorySystem(config)
    return _memory_system


def main() -> None:
    """Main function for testing the memory system."""
    try:
        # Initialize memory system
        memory_system = initialize_memory_system()

        # Test memory entry
        entry_id = memory_system.save_memory_entry(
            memory_type=MemoryType.SHORT_TERM,
            category=MemoryCategory.PROFIT_STATES,
            data={"profit": 0.05, "symbol": "BTC/USD"},
            importance=0.8
        )
        print(f"Memory entry saved: {entry_id}")

        # Test state snapshot
        state_id = memory_system.save_state_snapshot(
            profit_state={"total_profit": 100.0, "daily_profit": 10.0},
            market_conditions={"volatility": 0.02, "trend": "bullish"},
            engine_performance={"aleph_accuracy": 0.85, "alif_speed": 0.9},
            trading_decisions={"decisions_made": 50, "success_rate": 0.75},
            volume_data={"total_volume": 10000, "avg_volume": 500},
            stop_loss_data={"stop_losses_triggered": 5, "avg_loss": 0.02}
        )
        print(f"State snapshot saved: {state_id}")

        # Test collapse event
        collapse_id = memory_system.record_collapse_event(
            trigger_symbol="💰",
            trigger_hash="profit_trigger_hash",
            collapse_magnitude=0.15,
            affected_assets=["BTC/USD", "ETH/USD"]
        )
        print(f"Collapse event recorded: {collapse_id}")

        # Test print event
        print_id = memory_system.log_print_event(
            event_type="entry",
            symbol="BTC/USD",
            price=50000.0,
            volume=1000.0,
            confidence=0.85,
            stop_loss=49000.0,
            take_profit=52000.0
        )
        print(f"Print event logged: {print_id}")

        # Analyze patterns
        patterns = memory_system.analyze_memory_patterns()
        print(f"Pattern analysis: {patterns}")

        # Get performance metrics
        metrics = memory_system.get_performance_metrics()
        print(f"Performance metrics: {metrics}")

        # Optimize memory
        optimization = memory_system.optimize_memory()
        print(f"Memory optimization: {optimization}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
