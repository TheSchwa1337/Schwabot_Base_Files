""""""
Pair Flip Orbit
==============
Manages asset pair flip logic, bit states, and memory updates, with bit-phase awareness.
Enhanced with backup logic from previous systems for comprehensive memory management and flip pattern tracking.
""""""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

FLIPMATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "flipmatrix.json")
MEMORY_BANK_DIR = os.path.join(os.path.dirname(__file__), "..", "hash_memory_bank")
BACKUP_MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "backup_memory_stack")
FLIP_BACKUP_DIR = os.path.join(BACKUP_MEMORY_DIR, "flip_backups")


@dataclass
class FlipEvent:
    """Flip event with backup metadata."""

    event_id: str
    flip_type: str
    timestamp: float
    bit_phase: int
    pair: str
    flip_value: int
    outcome: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    backup_hash: str = field(default_factory=str)
    flip_pattern: str = field(default_factory=str)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlipBackupEntry:
    """Backup entry for flip events."""

    entry_id: str
    flip_type: str
    category: str
    data: Dict[str, Any]
    timestamp: float
    importance: float
    backup_signature: str
    flip_patterns: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PairFlipOrbit:
    """Enhanced pair flip orbit with backup logic integration."""

    def __init__(self, max_flip_events: int = 1000, decay_lambda: float = 0.1):
        """Initialize the pair flip orbit with backup capabilities."""
        self.max_flip_events = max_flip_events
        self.decay_lambda = decay_lambda
        self.flip_events: List[FlipEvent] = []
        self.backup_memory: Dict[str, FlipBackupEntry] = {}
        self.flip_patterns: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
            "total_flips": 0,
                "successful_flips": 0,
                    "failed_flips": 0,
                    "average_confidence": 0.0,
                    "pattern_hit_rate": 0.0,
                    "bit_phase_distribution": {},
}
        # Ensure backup directories exist
        os.makedirs(FLIP_BACKUP_DIR, exist_ok=True)
        os.makedirs(BACKUP_MEMORY_DIR, exist_ok=True)
        os.makedirs(MEMORY_BANK_DIR, exist_ok=True)

        # Load existing backup memory
        self._load_backup_memory()

    def _load_backup_memory(self) -> None:
        """Load backup memory from persistent storage."""
        try:
            backup_file = os.path.join(FLIP_BACKUP_DIR, "flip_backup_memory.json")
            if os.path.exists(backup_file):
                with open(backup_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.backup_memory = {k: FlipBackupEntry(**v) for k, v in data.get("entries", {}).items()}
                    self.flip_patterns = data.get("patterns", {})
                    self.performance_metrics.update(data.get("metrics", {}))
                print(f"[BACKUP] Loaded {len(self.backup_memory)} flip backup entries")
        except Exception as e:
            print(f"Error loading flip backup memory: {e}")

    def _save_backup_memory(self) -> None:
        """Save backup memory to persistent storage."""
        try:
            backup_file = os.path.join(FLIP_BACKUP_DIR, "flip_backup_memory.json")
            data = {
                "entries": {k: v.__dict__ for k, v in self.backup_memory.items()},
                "patterns": self.flip_patterns,
                "metrics": self.performance_metrics,
                "timestamp": time.time(),
}
}
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving flip backup memory: {e}")

    def _create_backup_signature(self, flip_data: Dict[str, Any]) -> str:
        """Create a backup signature for flip validation."""
        signature_data = f"{flip_data.get('pair', '')}_{flip_data.get('bit_phase', 4)}_{flip_data.get('flip_value', 0)}_{time.time()}"
        return hashlib.sha256(signature_data.encode()).hexdigest()

    def _verify_backup_consistency(self, event: FlipEvent) -> bool:
        """Verify backup consistency for a flip event."""
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
            if time_diff > 86400:  # 24 hours
                return False

            return True
        except Exception as e:
            print(f"Error verifying flip backup consistency: {e}")
            return False

    def _update_backup_memory(self, event: FlipEvent, outcome: Dict[str, Any]) -> None:
        """Update backup memory with flip outcome."""
        try:
            # Create backup memory entry
            backup_entry = FlipBackupEntry()
                entry_id=event.event_id,
                    flip_type=event.flip_type,
                        category="flip_outcome",
                        data = {
                            "pair": event.pair,
                            "bit_phase": event.bit_phase,
                            "flip_value": event.flip_value,
                            "outcome": outcome,
                            "confidence": event.confidence,
                            "flip_pattern": event.flip_pattern,
}
                            },
                            timestamp=time.time(),
                        importance=event.confidence,
                        backup_signature=event.backup_hash,
                        flip_patterns={event.pair: event.flip_pattern},
                        metadata=event.metadata,
                        )

            # Store in backup memory
            self.backup_memory[event.event_id] = backup_entry

            # Update performance metrics
            self.performance_metrics["total_flips"] += 1
            if outcome.get("success", False):
                self.performance_metrics["successful_flips"] += 1

            # Update bit phase distribution
            bit_phase = str(event.bit_phase)
            self.performance_metrics["bit_phase_distribution"][bit_phase] = ()
                self.performance_metrics["bit_phase_distribution"].get(bit_phase, 0) + 1
            )

            # Update average confidence
            total_confidence = self.performance_metrics["average_confidence"] * ()
                self.performance_metrics["total_flips"] - 1
            )
            total_confidence += event.confidence
            self.performance_metrics["average_confidence"] = total_confidence / self.performance_metrics["total_flips"]

            # Update flip patterns
            if event.pair not in self.flip_patterns:
                self.flip_patterns[event.pair] = []
            self.flip_patterns[event.pair].append()
                {}
                    "bit_phase": event.bit_phase,
                        "flip_value": event.flip_value,
                            "pattern": event.flip_pattern,
                            "timestamp": time.time(),
                            "success": outcome.get("success", False),
}
            )

            # Save backup memory periodically
            if self.performance_metrics["total_flips"] % 10 == 0:
                self._save_backup_memory()

        except Exception as e:
            print(f"Error updating flip backup memory: {e}")


def bit_flip(value: int, bits: int = 4) -> int:
    """Bitwise NOT for n-bit pattern with backup tracking."""
    flip_result = ~value & ((1 << bits) - 1)

    # Create flip event with backup signature
    backup_signature = hashlib.sha256(f"bit_flip_{value}_{bits}_{flip_result}_{time.time()}".encode()).hexdigest()

    flip_event = FlipEvent()
        event_id=f"flip_{int(time.time() * 1000)}",
            flip_type="bit_flip",
                timestamp=time.time(),
                bit_phase=bits,
                pair="bit_operation",
                flip_value=flip_result,
                confidence=0.9,
                backup_hash=backup_signature,
                flip_pattern=f"{value:0{bits}b}->{flip_result:0{bits}b}",
                metadata={"original_value": value, "bits": bits},
                )

    # Update backup memory
    orbit = PairFlipOrbit()
    outcome = {
        "success": True,
        "original_value": value,
        "flipped_value": flip_result,
        "bits": bits,
        "timestamp": time.time(),
}
}
    orbit._update_backup_memory(flip_event, outcome)

    return flip_result


def load_flipmatrix() -> Dict[str, Any]:
    """Load the flipmatrix.json file with backup validation."""
    try:
        with open(FLIPMATRIX_PATH, "r", encoding="utf-8") as f:
            flipmatrix = json.load(f)

        # Create backup of flipmatrix
        backup_signature = hashlib.sha256(str(flipmatrix).encode()).hexdigest()
        backup_file = os.path.join(FLIP_BACKUP_DIR, f"flipmatrix_backup_{int(time.time())}.json")

        backup_data = {
            "flipmatrix": flipmatrix,
            "backup_signature": backup_signature,
            "timestamp": time.time(),
            "entries_count": len(flipmatrix),
}
}
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, indent=2)

        print(f"[BACKUP] Flipmatrix loaded with {len(flipmatrix)} entries, backup created")
        return flipmatrix
    except Exception as e:
        print(f"Error loading flipmatrix.json: {e}")
        return {}


def get_pair_flip(pair: str, bit_phase: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve flip data for a given asset pair and bit phase from flipmatrix.json with backup validation."""
    flipmatrix = load_flipmatrix()
    pair_data = flipmatrix.get(pair, {})

    # Check backup memory for enhanced flip data
    backup_patterns = {}
    try:
        patterns_file = os.path.join(FLIP_BACKUP_DIR, "enhanced_flip_patterns.json")
        if os.path.exists(patterns_file):
            with open(patterns_file, "r", encoding="utf-8") as f:
                backup_patterns = json.load(f)
    except Exception as e:
        print(f"Error loading enhanced flip patterns: {e}")

    # Use backup patterns if available
    if pair in backup_patterns:
        enhanced_data = backup_patterns[pair]
        if bit_phase is not None and str(bit_phase) in enhanced_data.get("bit_phases", {}):
            return enhanced_data["bit_phases"][str(bit_phase)]
        return enhanced_data

    # Fall back to original logic
    if bit_phase is not None and "bit_phases" in pair_data:
        return pair_data["bit_phases"].get(str(bit_phase), pair_data)
    return pair_data


def update_pair_memory(pair: str, bit_phase: int, outcome: Dict[str, Any]) -> None:
    """Update memory for a given pair and bit phase with the latest outcome and backup tracking."""
    if not os.path.exists(MEMORY_BANK_DIR):
        os.makedirs(MEMORY_BANK_DIR)

    memory_file = os.path.join(MEMORY_BANK_DIR, f"{pair.replace('->', '_')}_bit{bit_phase}.json")

    # Create flip event with backup signature
    backup_signature = hashlib.sha256()
        f"memory_update_{pair}_{bit_phase}_{str(outcome)}_{time.time()}".encode()
    ).hexdigest()

    flip_event = FlipEvent()
        event_id=f"memory_{int(time.time() * 1000)}",
            flip_type="memory_update",
                timestamp=time.time(),
                bit_phase=bit_phase,
                pair=pair,
                flip_value=0,  # Not applicable for memory updates
        confidence=0.8,
            backup_hash=backup_signature,
                flip_pattern="memory_pattern",
                metadata={"outcome": outcome},
                )

    try:
        if os.path.exists(memory_file):
            with open(memory_file, "r", encoding="utf-8") as f:
                memory = json.load(f)
        else:
            memory = {"events": []}

        memory["events"].append(outcome)

        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)

        # Update backup memory
        orbit = PairFlipOrbit()
        backup_outcome = {
            "success": True,
            "memory_file": memory_file,
            "events_count": len(memory["events"]),
            "outcome": outcome,
            "timestamp": time.time(),
}
}
        orbit._update_backup_memory(flip_event, backup_outcome)

        print(f"[BACKUP] Memory updated for {pair} (bit_phase {bit_phase}) with {len(memory['events'])} events")

    except Exception as e:
        print(f"Error updating pair memory for {pair} (bit_phase {bit_phase}): {e}")

        # Update backup memory with failure
        orbit = PairFlipOrbit()
        failure_outcome = {"success": False, "error": str(e), "timestamp": time.time()}
        orbit._update_backup_memory(flip_event, failure_outcome)


def get_flip_backup_statistics() -> Dict[str, Any]:
    """Get flip backup statistics and performance metrics."""
    orbit = PairFlipOrbit()
    return {}
        "backup_memory_entries": len(orbit.backup_memory),
            "performance_metrics": orbit.performance_metrics,
                "flip_patterns": len(orbit.flip_patterns),
                "bit_phase_distribution": orbit.performance_metrics["bit_phase_distribution"],
                "backup_directory_size": _get_directory_size(FLIP_BACKUP_DIR),
                "last_backup_save": time.time(),
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
