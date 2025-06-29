""""""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core.pair_flip_orbit import get_pair_flip, update_pair_memory

Ghost Flip Executor
== == == == == == == == ==
Handles ghost trigger events, verifies memory, and executes strategy orbits.
Enhanced with backup logic from previous systems for comprehensive memory management.
""""""

MEMORY_BANK_DIR = os.path.join(os.path.dirname(__file__), "..", "hash_memory_bank")
GHOSTLOG_PATH = os.path.join(os.path.dirname(__file__), "..", ".ghostlogfile")
BACKUP_MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "backup_memory_stack")


@dataclass
class GhostEvent:
    """Ghost event with backup metadata."""


event_id: str
event_type: str
timestamp: float
bit_phase: int
trigger: str
outcome: Optional[str] = None
confidence: float = 0.0
backup_hash: str = field(default_factory=str)
memory_signature: str = field(default_factory=str)
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackupMemoryEntry:
    """Backup memory entry for ghost events."""


entry_id: str
memory_type: str
category: str
data: Dict[str, Any]
timestamp: float
importance: float
backup_signature: str
metadata: Dict[str, Any] = field(default_factory=dict)


class GhostFlipExecutor:
    """Enhanced ghost flip executor with backup logic integration."""

    def __init__(self, max_memory_events: int = 1000, decay_lambda: float = 0.1):
        """Initialize the ghost flip executor with backup capabilities."""
    self.max_memory_events = max_memory_events
    self.decay_lambda = decay_lambda
    self.ghost_events: List[GhostEvent] = []
    self.backup_memory: Dict[str, BackupMemoryEntry] = {}
    self.memory_patterns: Dict[str, Any] = {}
    self.performance_metrics: Dict[str, float] = {)}
        "total_triggers": 0,
            "successful_triggers": 0,
                "failed_triggers": 0,
                "average_confidence": 0.0,
                "memory_hit_rate": 0.0
}
    # Ensure backup directories exist
    os.makedirs(BACKUP_MEMORY_DIR, exist_ok=True)
    os.makedirs(MEMORY_BANK_DIR, exist_ok=True)

    # Load existing backup memory
    self._load_backup_memory()

    def _load_backup_memory(self) -> None:
    """Load backup memory from persistent storage."""
        try:
        backup_file = os.path.join(BACKUP_MEMORY_DIR, "ghost_backup_memory.json")
            if os.path.exists(backup_file):
                with open(backup_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.backup_memory = {)}
                        k: BackupMemoryEntry(**v) for k, v in data.get("entries", {}).items()
}
                self.memory_patterns = data.get("patterns", {})
                self.performance_metrics.update(data.get("metrics", {}))
            print(f"[BACKUP] Loaded {len(self.backup_memory)} backup memory entries")
            except Exception as e:
            print(f"Error loading backup memory: {e}")

    def _save_backup_memory(self) -> None:
    """Save backup memory to persistent storage."""
        try:
        backup_file = os.path.join(BACKUP_MEMORY_DIR, "ghost_backup_memory.json")
        data = {)}
                "entries": {k: v.__dict__ for k, v in self.backup_memory.items()},
                    "patterns": self.memory_patterns,
                        "metrics": self.performance_metrics,
                    "timestamp": time.time()
}
            with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        except Exception as e:
        print(f"Error saving backup memory: {e}")

    def _create_backup_signature(self, event_data: Dict[str, Any]) -> str:
        """Create a backup signature for event validation."""
    signature_data = f"{event_data.get('event', '')}_{event_data.get('trigger', '')}_{event_data.get('bit_phase', 4)}_{time.time()}"
    return hashlib.sha256(signature_data.encode()).hexdigest()

    def _verify_backup_consistency(self, event: GhostEvent) -> bool:
        """Verify backup consistency for a ghost event."""
        try:
            # Check if event exists in backup memory
        backup_entry = self.backup_memory.get(event.event_id)
            if not backup_entry:
            return False

        # Verify signature consistency
            if backup_entry.backup_signature != event.backup_hash:
            return False

        # Check timestamp validity (within 24 hours)
        time_diff = abs(time.time() - backup_entry.timestamp)
            if time_diff > 86400:  # 24 hours:
            return False

        return True
    except Exception as e:
        print(f"Error verifying backup consistency: {e}")
        return False

    def _update_backup_memory(self, event: GhostEvent, outcome: Dict[str, Any]) -> None:
        """Update backup memory with event outcome."""
        try:
        # Create backup memory entry
        backup_entry = BackupMemoryEntry()
            entry_id = event.event_id,
                memory_type="ghost_event",
                    category="trigger_outcome",
                    data={)}
                "event_type": event.event_type,
                    "trigger": event.trigger,
                        "bit_phase": event.bit_phase,
                        "outcome": outcome,
                        "confidence": event.confidence
            },
                timestamp = time.time(),
                    importance = event.confidence,
                    backup_signature = event.backup_hash,
                    metadata = event.metadata
        )

        # Store in backup memory
        self.backup_memory[event.event_id] = backup_entry

        # Update performance metrics
        self.performance_metrics["total_triggers"] += 1
            if outcome.get("success", False):
            self.performance_metrics["successful_triggers"] += 1
            else:
            self.performance_metrics["failed_triggers"] += 1

        # Update average confidence
        total_confidence = self.performance_metrics["average_confidence"] * (self.performance_metrics["total_triggers"] - 1)
        total_confidence += event.confidence
        self.performance_metrics["average_confidence"] = total_confidence / self.performance_metrics["total_triggers"]

        # Save backup memory periodically
                if self.performance_metrics["total_triggers"] % 10 == 0:
            self._save_backup_memory()

                except Exception as e:
                print(f"Error updating backup memory: {e}")

# Simulated strategy execution (replace with actual strategy_mapper integration)
def execute_strategy(pair: str, bit: str, bit_phase: int) -> None:
    print(f"[EXECUTE] Strategy for {pair} with bit {bit} (bit_phase {bit_phase})")

def ghost_trigger(event: Dict[str, Any]) -> None:
    """Process a ghost trigger event and execute the corresponding strategy with backup validation."""
    # Create ghost event with backup signature
backup_signature = hashlib.sha256(f"{event.get('event', '')}_{event.get('trigger', '')}_{time.time()}".encode()).hexdigest()

ghost_event = GhostEvent()
    event_id = f"ghost_{int(time.time() * 1000)}",
        event_type="ghost_trigger",
            timestamp = time.time(),
            bit_phase = event.get("bit_phase", 4),
            trigger = event.get("trigger", ""),
            confidence = event.get("confidence", 0.5),
            backup_hash = backup_signature,
            metadata = event
)

# Verify backup consistency
executor = GhostFlipExecutor()
    if executor._verify_backup_consistency(ghost_event):
        print(f"[BACKUP] Backup consistency verified for event {ghost_event.event_id}")

# Process the trigger
bit_phase = event.get("bit_phase", 4)
        if verify_memory_match(event["event"], event["trigger"], bit_phase):
    orbit_path = fetch_strategy_orbit(event["event"], bit_phase)
    execute_ferris_wheel(orbit_path, event["bit"], bit_phase)

        # Update backup memory with success
    outcome = {"success": True, "orbit_path": orbit_path, "timestamp": time.time()}
    executor._update_backup_memory(ghost_event, outcome)

    log_event("TRIGGERED", event, bit_phase)
            else:
            # Update backup memory with failure
        outcome = {"success": False, "reason": "memory_mismatch", "timestamp": time.time()}
        executor._update_backup_memory(ghost_event, outcome)

        log_event("IGNORED", event, bit_phase)

def verify_memory_match(event: str, trigger: str, bit_phase: int) -> bool:
    """Check hash_memory_bank for matching past success for a given bit phase with backup validation."""
memory_file = os.path.join(MEMORY_BANK_DIR, f"{event.replace('->', '_')}_bit{bit_phase}.json")
    try:
        if os.path.exists(memory_file):
            with open(memory_file, "r", encoding="utf-8") as f:
            memory = json.load(f)

            # Check for successful outcomes
                for e in memory.get("events", []):
                    if e.get("trigger") == trigger and e.get("outcome", "").startswith("+"):
                # Verify backup consistency
                backup_file = os.path.join(BACKUP_MEMORY_DIR, f"backup_{e.get('timestamp', '')}.json")
                        if os.path.exists(backup_file):
                            with open(backup_file, "r", encoding="utf-8") as bf:
                        backup_data = json.load(bf)
                                if backup_data.get("verified", False):
                            return True
                                else:
                            # If no backup exists, still consider it valid but log
                                print(f"[WARNING] No backup found for memory match: {event}")
                        return True
                    return False
                        except Exception as e:
                        print(f"Error verifying memory match: {e}")
                    return False

def fetch_strategy_orbit(event: str, bit_phase: int) -> List[str]:
    """Map event to strategy orbit path using pair flip data and bit phase with backup validation."""
flip_data = get_pair_flip(event, bit_phase)
    if not flip_data:
    return [event]

    # Check backup memory for orbit patterns
backup_patterns = {}
    try:
    patterns_file = os.path.join(BACKUP_MEMORY_DIR, "orbit_patterns.json")
        if os.path.exists(patterns_file):
            with open(patterns_file, "r", encoding="utf-8") as f:
            backup_patterns = json.load(f)
            except Exception as e:
            print(f"Error loading orbit patterns: {e}")

                # Use backup patterns if available
                    if event in backup_patterns:
            return backup_patterns[event].get("orbit_path", [event])

            # Fall back to flip matrix logic
            inverse = flip_data.get("inverse")
                    if inverse:
            return [event, inverse]
        return [event]

def execute_ferris_wheel(orbit_path: List[str], bit: str, bit_phase: int) -> None:
    """Call strategy_mapper to execute the trade sequence for a given bit phase with backup tracking."""
    for pair in orbit_path:
        # Create backup entry for execution
    execution_backup = {)}
        "pair": pair,
            "bit": bit,
                "bit_phase": bit_phase,
                "timestamp": time.time(),
                "execution_id": f"exec_{int(time.time() * 1000)}"
}
    # Save execution backup
    backup_file = os.path.join(BACKUP_MEMORY_DIR, f"execution_{execution_backup['execution_id']}.json")
        try:
            with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(execution_backup, f, indent=2)
            except Exception as e:
            print(f"Error saving execution backup: {e}")

            execute_strategy(pair, bit, bit_phase)

def log_event(event_type: str, event: Dict[str, Any], bit_phase: int) -> None:
"""Log to .ghostlogfile or RAM cache, including bit phase and backup information."""
log_entry = {)}
    "type": event_type,
        "event": event,
            "bit_phase": bit_phase,
            "timestamp": datetime.utcnow().isoformat(),
            "backup_signature": hashlib.sha256(f"{event_type}_{str(event)}_{time.time()}".encode()).hexdigest()
}
    try:
        with open(GHOSTLOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
        print(f"Error logging event: {e}")
        print(f"[{event_type}][bit_phase={bit_phase}] {event}")

def get_backup_statistics() -> Dict[str, Any]:
"""Get backup statistics and performance metrics."""
executor = GhostFlipExecutor()
return {)}
    "backup_memory_entries": len(executor.backup_memory),
        "performance_metrics": executor.performance_metrics,
            "memory_patterns": len(executor.memory_patterns),
            "backup_directory_size": _get_directory_size(BACKUP_MEMORY_DIR),
            "last_backup_save": time.time()
}
def _get_directory_size(directory: str) -> str:
"""Get directory size in human readable format."""
    try:
    total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)

                if total_size < 1024:
        return f"{total_size} B"
                elif total_size < 1024 * 1024:
        return f"{total_size / 1024:.1f} KB"
                else:
        return f"{total_size / (1024 * 1024):.1f} MB"
            except Exception:
        return "unknown"