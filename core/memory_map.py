"""
Shared Memory Map System
========================

Provides persistent storage and retrieval for agent memory, strategy successes,
and thermal-aware profit trajectory data. Used by the MemoryAgent system to
maintain state across system restarts.
"""

import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMap:
    """Thread-safe memory map implementation with persistent storage"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.memory_file = self.data_dir / "memory_map.json"
        self.backup_dir = self.data_dir / "backups"
        self._lock = threading.RLock()
        self._memory_data: Dict[str, Any] = {}
        
        # Initialize directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_memory_map()
        
    def _load_memory_map(self) -> None:
        """Load memory map from persistent storage"""
        with self._lock:
            if not self.memory_file.exists():
                self._memory_data = self._initialize_default_map()
                self._save_memory_map()
                return
                
            try:
                with open(self.memory_file, 'r') as f:
                    self._memory_data = json.load(f)
                logger.info(f"Loaded memory map with {len(self._memory_data)} entries")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Could not load memory map: {e}. Starting fresh.")
                self._memory_data = self._initialize_default_map()
                self._save_memory_map()
            except Exception as e:
                logger.error(f"Unexpected error loading memory map: {e}")
                self._memory_data = self._initialize_default_map()
                
    def _initialize_default_map(self) -> Dict[str, Any]:
        """Initialize default memory map structure"""
        return {
            "strategy_successes": {},
            "profit_trajectories": [],
            "thermal_states": [],
            "active_agents": {},
            "confidence_coefficients": {},
            "hash_performance": {},
            "system_metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        }
        
    def _save_memory_map(self) -> None:
        """Save memory map to persistent storage with backup"""
        try:
            # Create backup first
            if self.memory_file.exists():
                backup_name = f"memory_map_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                backup_path = self.backup_dir / backup_name
                self.memory_file.replace(backup_path)
                
            # Update metadata
            self._memory_data["system_metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Save new data
            with open(self.memory_file, 'w') as f:
                json.dump(self._memory_data, f, indent=2, default=str)
                
            logger.debug("Memory map saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving memory map: {e}")
            raise
            
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get value from memory map"""
        with self._lock:
            return self._memory_data.get(key, default)
            
    def set(self, key: str, value: Any) -> None:
        """Set value in memory map and persist"""
        with self._lock:
            self._memory_data[key] = value
            self._save_memory_map()
            
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values atomically"""
        with self._lock:
            self._memory_data.update(updates)
            self._save_memory_map()
            
    def append_to_list(self, key: str, value: Any) -> None:
        """Append value to a list in memory map"""
        with self._lock:
            if key not in self._memory_data:
                self._memory_data[key] = []
            elif not isinstance(self._memory_data[key], list):
                raise ValueError(f"Key '{key}' is not a list")
                
            self._memory_data[key].append(value)
            self._save_memory_map()
            
    def add_strategy_success(self, strategy_id: str, success_data: Dict[str, Any]) -> None:
        """Add a strategy success record"""
        with self._lock:
            if "strategy_successes" not in self._memory_data:
                self._memory_data["strategy_successes"] = {}
                
            if strategy_id not in self._memory_data["strategy_successes"]:
                self._memory_data["strategy_successes"][strategy_id] = []
                
            success_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": success_data
            }
            
            self._memory_data["strategy_successes"][strategy_id].append(success_record)
            
            # Keep only last 1000 records per strategy
            if len(self._memory_data["strategy_successes"][strategy_id]) > 1000:
                self._memory_data["strategy_successes"][strategy_id] = \
                    self._memory_data["strategy_successes"][strategy_id][-1000:]
                    
            self._save_memory_map()
            
    def get_strategy_successes(self, strategy_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get strategy success records"""
        with self._lock:
            successes = self._memory_data.get("strategy_successes", {}).get(strategy_id, [])
            if limit:
                return successes[-limit:]
            return successes
            
    def add_profit_trajectory(self, trajectory_data: Dict[str, Any]) -> None:
        """Add profit trajectory data"""
        trajectory_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": trajectory_data
        }
        self.append_to_list("profit_trajectories", trajectory_record)
        
    def get_profit_trajectories(self, limit: Optional[int] = None) -> List[Dict]:
        """Get profit trajectory records"""
        trajectories = self.get("profit_trajectories", [])
        if limit:
            return trajectories[-limit:]
        return trajectories
        
    def update_confidence_coefficient(self, agent_id: str, coefficient: float) -> None:
        """Update confidence coefficient for an agent"""
        with self._lock:
            if "confidence_coefficients" not in self._memory_data:
                self._memory_data["confidence_coefficients"] = {}
                
            self._memory_data["confidence_coefficients"][agent_id] = {
                "coefficient": coefficient,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            self._save_memory_map()
            
    def get_confidence_coefficient(self, agent_id: str) -> Optional[float]:
        """Get confidence coefficient for an agent"""
        coeffs = self.get("confidence_coefficients", {})
        agent_data = coeffs.get(agent_id)
        return agent_data["coefficient"] if agent_data else None
        
    def add_thermal_state(self, thermal_data: Dict[str, Any]) -> None:
        """Add thermal state record"""
        thermal_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": thermal_data
        }
        self.append_to_list("thermal_states", thermal_record)
        
    def get_thermal_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get thermal state history"""
        history = self.get("thermal_states", [])
        if limit:
            return history[-limit:]
        return history
        
    def clear_old_data(self, days_to_keep: int = 30) -> None:
        """Clear data older than specified days"""
        with self._lock:
            cutoff_date = datetime.now(timezone.utc).timestamp() - (days_to_keep * 24 * 3600)
            
            # Clear old trajectories
            trajectories = self.get("profit_trajectories", [])
            self._memory_data["profit_trajectories"] = [
                t for t in trajectories 
                if datetime.fromisoformat(t["timestamp"].replace('Z', '+00:00')).timestamp() > cutoff_date
            ]
            
            # Clear old thermal states
            thermal_states = self.get("thermal_states", [])
            self._memory_data["thermal_states"] = [
                t for t in thermal_states 
                if datetime.fromisoformat(t["timestamp"].replace('Z', '+00:00')).timestamp() > cutoff_date
            ]
            
            self._save_memory_map()
            logger.info(f"Cleared data older than {days_to_keep} days")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get memory map statistics"""
        with self._lock:
            return {
                "total_strategy_successes": sum(
                    len(successes) for successes in self.get("strategy_successes", {}).values()
                ),
                "profit_trajectory_count": len(self.get("profit_trajectories", [])),
                "thermal_state_count": len(self.get("thermal_states", [])),
                "active_agent_count": len(self.get("active_agents", {})),
                "confidence_coefficient_count": len(self.get("confidence_coefficients", {})),
                "memory_map_size_mb": self.memory_file.stat().st_size / (1024 * 1024) if self.memory_file.exists() else 0
            }

# Global instance
_memory_map_instance = None
_memory_map_lock = threading.Lock()

def get_memory_map(data_dir: str = "data") -> MemoryMap:
    """Get global memory map instance (singleton pattern)"""
    global _memory_map_instance
    with _memory_map_lock:
        if _memory_map_instance is None:
            _memory_map_instance = MemoryMap(data_dir)
        return _memory_map_instance

# Example usage and testing
if __name__ == "__main__":
    # Initialize memory map
    mem_map = get_memory_map("test_data")
    
    # Test strategy success recording
    mem_map.add_strategy_success("strategy_alpha_001", {
        "profit": 150.75,
        "confidence": 0.87,
        "hash_triggers": ["TRIGGER_001", "TRIGGER_002"],
        "execution_time": 2.3
    })
    
    # Test profit trajectory
    mem_map.add_profit_trajectory({
        "slope": 0.45,
        "vector": 0.82,
        "zone_state": "surging",
        "thermal_coefficient": 1.05
    })
    
    # Test confidence coefficients
    mem_map.update_confidence_coefficient("agent_001", 0.91)
    
    # Print stats
    stats = mem_map.get_stats()
    print("Memory Map Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    # Test retrieval
    successes = mem_map.get_strategy_successes("strategy_alpha_001", limit=5)
    print(f"\nStrategy successes: {len(successes)}")
    
    trajectories = mem_map.get_profit_trajectories(limit=3)
    print(f"Profit trajectories: {len(trajectories)}")
    
    coeff = mem_map.get_confidence_coefficient("agent_001")
    print(f"Agent confidence coefficient: {coeff}") 