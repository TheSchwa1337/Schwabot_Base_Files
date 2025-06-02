"""
Profit Tensor Store
=================

Handles storage and mapping of profit tensors across different bit depths and patterns.
Integrates with ZBE temperature tensor for thermal-aware profit allocation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os

@dataclass
class TensorEntry:
    sha_key: str
    tensor: np.ndarray
    last_update: float
    profit_history: List[float]
    thermal_history: List[float]
    bit_depth: int

class ProfitTensorStore:
    def __init__(self, storage_path: str = "state/tensors"):
        self.tensor_map: Dict[str, TensorEntry] = {}
        self.storage_path = storage_path
        self.bit_depths = [4, 8, 16, 42, 81]
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing tensors if available
        self._load_tensors()

    def _load_tensors(self):
        """Load tensors from disk storage."""
        try:
            tensor_file = os.path.join(self.storage_path, "tensors.json")
            if os.path.exists(tensor_file):
                with open(tensor_file, 'r') as f:
                    data = json.load(f)
                    for sha_key, entry in data.items():
                        self.tensor_map[sha_key] = TensorEntry(
                            sha_key=sha_key,
                            tensor=np.array(entry['tensor']),
                            last_update=entry['last_update'],
                            profit_history=entry['profit_history'],
                            thermal_history=entry['thermal_history'],
                            bit_depth=entry['bit_depth']
                        )
        except Exception as e:
            print(f"Error loading tensors: {e}")

    def _save_tensors(self):
        """Save tensors to disk storage."""
        try:
            tensor_file = os.path.join(self.storage_path, "tensors.json")
            data = {
                sha_key: {
                    'tensor': entry.tensor.tolist(),
                    'last_update': entry.last_update,
                    'profit_history': entry.profit_history,
                    'thermal_history': entry.thermal_history,
                    'bit_depth': entry.bit_depth
                }
                for sha_key, entry in self.tensor_map.items()
            }
            with open(tensor_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving tensors: {e}")

    def lookup(self, sha_key: str) -> np.ndarray:
        """Lookup tensor for given SHA key."""
        if sha_key in self.tensor_map:
            return self.tensor_map[sha_key].tensor
        return np.ones(8)  # Default tensor

    def store(self, sha_key: str, tensor: np.ndarray, 
              profit: float = 0.0, thermal: float = 0.0,
              bit_depth: Optional[int] = None):
        """
        Store or update tensor with profit and thermal data.
        
        Args:
            sha_key: Hash key identifying the strategy/pattern
            tensor: The tensor to store
            profit: Profit value for this update
            thermal: Thermal value for this update
            bit_depth: Optional bit depth override
        """
        current_time = time.time()
        
        if sha_key in self.tensor_map:
            entry = self.tensor_map[sha_key]
            entry.tensor = tensor
            entry.last_update = current_time
            entry.profit_history.append(profit)
            entry.thermal_history.append(thermal)
            
            # Keep last 1000 entries
            entry.profit_history = entry.profit_history[-1000:]
            entry.thermal_history = entry.thermal_history[-1000:]
            
            if bit_depth is not None:
                entry.bit_depth = bit_depth
        else:
            self.tensor_map[sha_key] = TensorEntry(
                sha_key=sha_key,
                tensor=tensor,
                last_update=current_time,
                profit_history=[profit],
                thermal_history=[thermal],
                bit_depth=bit_depth or 4  # Default to 4-bit
            )
        
        # Save to disk periodically
        if len(self.tensor_map) % 10 == 0:  # Save every 10 updates
            self._save_tensors()

    def get_tensor_stats(self, sha_key: str) -> Dict:
        """Get statistics for a specific tensor."""
        if sha_key not in self.tensor_map:
            return {}
            
        entry = self.tensor_map[sha_key]
        return {
            'last_update': datetime.fromtimestamp(entry.last_update).isoformat(),
            'bit_depth': entry.bit_depth,
            'avg_profit': np.mean(entry.profit_history) if entry.profit_history else 0.0,
            'avg_thermal': np.mean(entry.thermal_history) if entry.thermal_history else 0.0,
            'profit_std': np.std(entry.profit_history) if entry.profit_history else 0.0,
            'thermal_std': np.std(entry.thermal_history) if entry.thermal_history else 0.0
        }

    def get_top_tensors(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N tensors by average profit."""
        tensor_scores = []
        for sha_key, entry in self.tensor_map.items():
            if entry.profit_history:
                avg_profit = np.mean(entry.profit_history)
                tensor_scores.append((sha_key, avg_profit))
        
        # Sort by average profit
        tensor_scores.sort(key=lambda x: x[1], reverse=True)
        return tensor_scores[:n]

    def cleanup_old_tensors(self, max_age_days: int = 7):
        """Remove tensors that haven't been updated in max_age_days."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        old_keys = [
            sha_key for sha_key, entry in self.tensor_map.items()
            if current_time - entry.last_update > max_age_seconds
        ]
        
        for sha_key in old_keys:
            del self.tensor_map[sha_key]
        
        if old_keys:
            self._save_tensors()

    def export_state(self) -> Dict:
        """Export current state for monitoring/debugging."""
        return {
            'tensor_count': len(self.tensor_map),
            'top_tensors': self.get_top_tensors(5),
            'storage_path': self.storage_path,
            'last_save': datetime.now().isoformat()
        } 