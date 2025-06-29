""""""
Profit Orbit Engine
==================
Orchestrates multi-asset, multi-layer trade orbits using recursive bit-flip logic and memory, with bit-phase awareness.
Enhanced with backup logic from previous systems for comprehensive memory management and performance tracking.
""""""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

VOLUME_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "volume_weights.json")
BACKUP_MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "backup_memory_stack")
ORBIT_BACKUP_DIR = os.path.join(BACKUP_MEMORY_DIR, "orbit_backups")

@dataclass
class OrbitEvent:
    """Orbit event with backup metadata."""
event_id: str
orbit_type: str
timestamp: float
bit_phase: int
pairs: List[str]
outcome: Optional[Dict[str, Any]] = None
confidence: float = 0.0
backup_hash: str = field(default_factory = str)
performance_score: float = 0.0
metadata: Dict[str, Any] = field(default_factory = dict)

@dataclass
class OrbitBackupEntry:
    """Backup entry for orbit events."""
entry_id: str
orbit_type: str
category: str
data: Dict[str, Any]
timestamp: float
importance: float
backup_signature: str
performance_metrics: Dict[str, float] = field(default_factory = dict)
metadata: Dict[str, Any] = field(default_factory = dict)

class ProfitOrbitEngine:
    """Enhanced profit orbit engine with backup logic integration."""
    
    def __init__(self, max_orbit_events: int = 1000, decay_lambda: float = 0.01):
        """Initialize the profit orbit engine with backup capabilities."""
    self.max_orbit_events = max_orbit_events
    self.decay_lambda = decay_lambda
    self.orbit_events: List[OrbitEvent] = []
    self.backup_memory: Dict[str, OrbitBackupEntry] = {}
    self.orbit_patterns: Dict[str, Any] = {}
    self.performance_metrics: Dict[str, float] = {)
        "total_orbits": 0,
        "successful_orbits": 0,
        "failed_orbits": 0,
        "average_profit": 0.0,
        "total_volume": 0.0,
        "orbit_efficiency": 0.0
    }
        
        # In-memory weights for demonstration
    # Key: (pair, bit_phase)
    self.volume_weights: Dict[Tuple[str, int], float] = {}
        
    # Ensure backup directories exist
    os.makedirs(ORBIT_BACKUP_DIR, exist_ok = True)
    os.makedirs(BACKUP_MEMORY_DIR, exist_ok = True)
        
    # Load existing backup memory
    self._load_backup_memory()
        
    def _load_backup_memory(self) -> None:
    """Load backup memory from persistent storage."""
        try:
        backup_file = os.path.join(ORBIT_BACKUP_DIR, "orbit_backup_memory.json")
            if os.path.exists(backup_file):
                with open(backup_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.backup_memory = {)
                        k: OrbitBackupEntry(**v) for k, v in data.get("entries", {}).items()
                }
                self.orbit_patterns = data.get("patterns", {})
                self.performance_metrics.update(data.get("metrics", {}))
                self.volume_weights = {)
                        tuple(k.split('_')): v for k, v in data.get("volume_weights", {}).items()
                }
            print(f"[BACKUP] Loaded {len(self.backup_memory)} orbit backup entries")
            except Exception as e:
            print(f"Error loading orbit backup memory: {e}")
    
    def _save_backup_memory(self) -> None:
    """Save backup memory to persistent storage."""
        try:
        backup_file = os.path.join(ORBIT_BACKUP_DIR, "orbit_backup_memory.json")
        data = {)
                "entries": {k: v.__dict__ for k, v in self.backup_memory.items()},
            "patterns": self.orbit_patterns,
            "metrics": self.performance_metrics,
                "volume_weights": {f"{k[0]}_{k[1]}": v for k, v in self.volume_weights.items()},
            "timestamp": time.time()
        }
            with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        except Exception as e:
        print(f"Error saving orbit backup memory: {e}")
    
    def _create_backup_signature(self, orbit_data: Dict[str, Any]) -> str:
        """Create a backup signature for orbit validation."""
    signature_data = f"{orbit_data.get('orbit_type', '')}_{str(orbit_data.get('pairs', []))}_{orbit_data.get('bit_phase', 4)}_{time.time()}"
    return hashlib.sha256(signature_data.encode()).hexdigest()
    
    def _verify_backup_consistency(self, event: OrbitEvent) -> bool:
        """Verify backup consistency for an orbit event."""
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
        print(f"Error verifying orbit backup consistency: {e}")
        return False
    
    def _update_backup_memory(self, event: OrbitEvent, outcome: Dict[str, Any]) -> None:
        """Update backup memory with orbit outcome."""
        try:
        # Create backup memory entry
        backup_entry = OrbitBackupEntry()
            entry_id = event.event_id,
            orbit_type = event.orbit_type,
            category="orbit_outcome",
            data={)
                "pairs": event.pairs,
                "bit_phase": event.bit_phase,
                "outcome": outcome,
                "confidence": event.confidence,
                "performance_score": event.performance_score
            },
            timestamp = time.time(),
            importance = event.confidence,
            backup_signature = event.backup_hash,
            performance_metrics={)
                "profit": outcome.get("profit", 0.0),
                "volume": outcome.get("volume", 0.0),
                "efficiency": outcome.get("efficiency", 0.0)
            },
            metadata = event.metadata
        )
            
        # Store in backup memory
        self.backup_memory[event.event_id] = backup_entry
            
        # Update performance metrics
        self.performance_metrics["total_orbits"] += 1
            if outcome.get("success", False):
            self.performance_metrics["successful_orbits"] += 1
            
        # Update profit metrics
        profit = outcome.get("profit", 0.0)
        volume = outcome.get("volume", 0.0)
        self.performance_metrics["total_volume"] += volume
            
        # Update average profit
        total_profit = self.performance_metrics["average_profit"] * (self.performance_metrics["total_orbits"] - 1)
        total_profit += profit
        self.performance_metrics["average_profit"] = total_profit / self.performance_metrics["total_orbits"]
            
        # Update orbit efficiency
            if self.performance_metrics["total_orbits"] > 0:
            self.performance_metrics["orbit_efficiency"] = ()
                self.performance_metrics["successful_orbits"] / self.performance_metrics["total_orbits"]
            )
            
        # Save backup memory periodically
                if self.performance_metrics["total_orbits"] % 10 == 0:
            self._save_backup_memory()
                
                except Exception as e:
                print(f"Error updating orbit backup memory: {e}")

# In-memory weights for demonstration
# Key: (pair, bit_phase)
                volume_weights: Dict[Tuple[str, int], float] = {}

def run_orbit_cycle(trade_layers: List[List[Tuple[str, int]]], market_data: Dict[str, Any]) -> None:
    """Run a full orbit cycle across all trade layers and bit phases given current market data with backup tracking."""
    # Create orbit event with backup signature
    orbit_pairs = [pair for layer in trade_layers for pair, _ in layer]
backup_signature = hashlib.sha256(f"orbit_cycle_{str(orbit_pairs)}_{time.time()}".encode()).hexdigest()
    
orbit_event = OrbitEvent()
    event_id = f"orbit_{int(time.time() * 1000)}",
    orbit_type="multi_layer_cycle",
    timestamp = time.time(),
    bit_phase=4,  # Default bit phase
    pairs = orbit_pairs,
    confidence=0.7,
    backup_hash = backup_signature,
    metadata={"market_data": market_data}
)
    
# Initialize engine
engine = ProfitOrbitEngine()
    
# Verify backup consistency
    if engine._verify_backup_consistency(orbit_event):
        print(f"[BACKUP] Backup consistency verified for orbit {orbit_event.event_id}")
    
# Track orbit execution
executed_trades = []
total_profit = 0.0
total_volume = 0.0
    
        for layer in trade_layers:
            for pair, bit_phase in layer:
        key = (pair, bit_phase)
        price = market_data.get(pair, {}).get("price", 0.0)
        trend = market_data.get(pair, {}).get("trend", "neutral")
        weight = volume_weights.get(key, 1.0)
            
                if trend == "up" and weight > 0.2:
                print(f"[ORBIT] Triggering trade for {pair} (bit_phase {bit_phase}) at price {price} (weight {weight})")
            executed_trades.append({))
                "pair": pair,
                "bit_phase": bit_phase,
                "price": price,
                "weight": weight,
                "action": "buy"
            })
            total_volume += weight
            total_profit += weight * 0.01  # Simulated profit
                    else:
                print(f"[ORBIT] Holding {pair} (bit_phase {bit_phase}) (trend: {trend}, weight: {weight})")
    
                    # Update backup memory with orbit outcome
                outcome = {)
                "success": len(executed_trades) > 0,
                "executed_trades": executed_trades,
                "profit": total_profit,
                "volume": total_volume,
                "efficiency": total_profit / max(total_volume, 1.0),
                "timestamp": time.time()
                }
                engine._update_backup_memory(orbit_event, outcome)

def update_volume_weights(asset: str, bit_phase: int, delta: float) -> None:
    """Update volume weights for an asset and bit phase based on performance delta with backup tracking."""
global volume_weights
key = (asset, bit_phase)
volume_weights[key] = max(0.0, volume_weights.get(key, 1.0) + delta)
    
    # Create backup entry for weight update
backup_signature = hashlib.sha256(f"weight_update_{asset}_{bit_phase}_{delta}_{time.time()}".encode()).hexdigest()
    
weight_event = OrbitEvent()
    event_id = f"weight_{int(time.time() * 1000)}",
    orbit_type="weight_update",
    timestamp = time.time(),
    bit_phase = bit_phase,
    pairs=[asset],
    confidence=0.8,
    backup_hash = backup_signature,
    metadata={"delta": delta, "new_weight": volume_weights[key]}
)
    
# Update backup memory
engine = ProfitOrbitEngine()
outcome = {)
    "success": True,
    "old_weight": volume_weights.get(key, 1.0) - delta,
    "new_weight": volume_weights[key],
    "delta": delta,
    "timestamp": time.time()
}
engine._update_backup_memory(weight_event, outcome)
    
# Persist to file
    try:
        # Convert tuple keys to string for JSON
        serializable = {f"{k[0]}_bit{k[1]}": v for k, v in volume_weights.items()}
        with open(VOLUME_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
        except Exception as e:
        print(f"Error updating volume weights: {e}")

def select_optimal_orbit(trade_layers: List[List[Tuple[str, int]]], memory: Dict[str, Any]) -> str:
    """Select the optimal orbit path based on memory and current state, considering bit phase with backup validation."""
    # Check backup memory for optimal orbit patterns
backup_patterns = {}
    try:
    patterns_file = os.path.join(ORBIT_BACKUP_DIR, "optimal_orbit_patterns.json")
        if os.path.exists(patterns_file):
            with open(patterns_file, "r", encoding="utf-8") as f:
            backup_patterns = json.load(f)
            except Exception as e:
            print(f"Error loading optimal orbit patterns: {e}")
    
                # Use backup patterns if available
            layer_signature = hashlib.sha256(str(trade_layers).encode()).hexdigest()
                    if layer_signature in backup_patterns:
            return backup_patterns[layer_signature].get("optimal_layer", "")
    
            # Fall back to original logic
            best_layer = ""
            best_score = -float("inf")
                    for i, layer in enumerate(trade_layers):
                score = 0
                        for pair, bit_phase in layer:
                    mem_key = f"{pair}_bit{bit_phase}"
                    pair_mem = memory.get(mem_key, {"events": []})
                            for event in pair_mem.get("events", []):
                                if event.get("outcome", "").startswith("+"):
                            score += 1
                                    if score > best_score:
                                best_score = score
                                best_layer = f"Layer {i+1}: {layer}"
    
                                # Save optimal orbit pattern to backup
                                        try:
                                            if not os.path.exists(ORBIT_BACKUP_DIR):
                                        os.makedirs(ORBIT_BACKUP_DIR, exist_ok = True)
        
                                        patterns_file = os.path.join(ORBIT_BACKUP_DIR, "optimal_orbit_patterns.json")
                                                if os.path.exists(patterns_file):
                                                    with open(patterns_file, "r", encoding="utf-8") as f:
                                                patterns = json.load(f)
                                                        else:
                                                    patterns = {}
        
                                                    patterns[layer_signature] = {)
                                                    "optimal_layer": best_layer,
                                                    "score": best_score,
                                                    "timestamp": time.time()
                                                    }
        
                                                            with open(patterns_file, "w", encoding="utf-8") as f:
                                                        json.dump(patterns, f, indent=2)
                                                            except Exception as e:
                                                            print(f"Error saving optimal orbit pattern: {e}")
    
                                                        return best_layer

def get_orbit_backup_statistics() -> Dict[str, Any]:
"""Get orbit backup statistics and performance metrics."""
engine = ProfitOrbitEngine()
return {)
    "backup_memory_entries": len(engine.backup_memory),
    "performance_metrics": engine.performance_metrics,
    "orbit_patterns": len(engine.orbit_patterns),
    "volume_weights_count": len(engine.volume_weights),
    "backup_directory_size": _get_directory_size(ORBIT_BACKUP_DIR),
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