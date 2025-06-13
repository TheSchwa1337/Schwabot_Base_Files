"""
Strategy Snapshot Logger
======================

Captures and logs strategy state, including Ψ values, entropy,
coherence, actions, and confidence scores for analysis.
"""

from typing import Dict, Any, Optional
import json
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class StrategySnapshot:
    timestamp: float
    psi_value: float
    entropy: float
    coherence: float
    action: str
    confidence: float
    metadata: Dict[str, Any]

class SnapshotLogger:
    def __init__(self, output_dir: str = "snapshots"):
        self.output_dir = Path(__file__).resolve().parent / output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def create_snapshot(self,
                       psi_value: float,
                       entropy: float,
                       coherence: float,
                       action: str,
                       confidence: float,
                       metadata: Optional[Dict[str, Any]] = None) -> StrategySnapshot:
        """Create a new strategy snapshot"""
        return StrategySnapshot(
            timestamp=time.time(),
            psi_value=psi_value,
            entropy=entropy,
            coherence=coherence,
            action=action,
            confidence=confidence,
            metadata=metadata or {}
        )
        
    def save_snapshot(self, snapshot: StrategySnapshot) -> str:
        """Save snapshot to disk and return its hash"""
        # Convert to dict and compute hash
        snapshot_dict = asdict(snapshot)
        snapshot_json = json.dumps(snapshot_dict, sort_keys=True)
        snapshot_hash = hashlib.sha256(snapshot_json.encode()).hexdigest()
        
        # Save to file
        filename = f"snapshot_{int(snapshot.timestamp)}_{snapshot_hash[:8]}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(snapshot_dict, f, indent=2)
            
        return snapshot_hash
        
    def load_snapshot(self, snapshot_hash: str) -> Optional[StrategySnapshot]:
        """Load a snapshot by its hash"""
        for file in self.output_dir.glob("snapshot_*.json"):
            if file.stem.endswith(snapshot_hash[:8]):
                with open(file, 'r') as f:
                    data = json.load(f)
                    return StrategySnapshot(**data)
        return None
        
    def list_snapshots(self) -> Dict[str, float]:
        """List all snapshots with their timestamps"""
        snapshots = {}
        for file in self.output_dir.glob("snapshot_*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                snapshots[file.stem] = data['timestamp']
        return snapshots 