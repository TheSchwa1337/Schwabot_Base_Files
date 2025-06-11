"""
NCCO Generator
============

Generates and manages NCCOs (Nominal Channel Control Objects) for decision tracking.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import threading
from pathlib import Path
import time
from datetime import datetime
import json

from .event_bus import EventBus

logger = logging.getLogger(__name__)

@dataclass
class NCCOState:
    """Container for NCCO state"""
    timestamp: float
    price: float
    strategy: str
    ghost_hash: str
    entropy: float
    coherence: float
    paradox_phase: int
    velocity_class: str
    pattern_cluster: str
    liquidity_status: str
    smart_money_score: float
    panic_zone: bool
    metadata: Dict[str, Any]

class NCCOGenerator:
    """Generates and manages NCCOs for decision tracking"""
    
    def __init__(self, event_bus: EventBus, log_dir: str = "logs"):
        self.event_bus = event_bus
        self.entries: List[NCCOState] = []
        self.max_entries = 10000
        
        # Setup logging
        self.log_dir = Path(log_dir) / "ncco"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize index file
        self._init_index()
    
    def _init_index(self) -> None:
        """Initialize NCCO index file"""
        index_path = self.log_dir / "ncco_index.jsonl"
        if not index_path.exists():
            index_path.touch()
    
    def generate_ncco(self, ncco_type: str, current_price: float, metadata: Optional[Dict] = None) -> NCCOState:
        """
        Generate a new NCCO with current system state
        
        Args:
            ncco_type: Type of NCCO (e.g., "panic", "normal")
            current_price: Current market price
            metadata: Additional metadata
            
        Returns:
            Generated NCCO state
        """
        with self._lock:
            # Get current system state
            ghost_hash = self.event_bus.get("ghost_hash", str(uuid4()))
            strategy = self.event_bus.get("current_strategy", "UNSPECIFIED")
            timestamp = datetime.now().timestamp()
            entropy = self.event_bus.get("entropy_score", 0.0)
            coherence = self.event_bus.get("coherence_score", 1.0)
            paradox_phase = self.event_bus.get("paradox_phase", 0)
            velocity = self.event_bus.get("velocity_class", "UNKNOWN")
            pattern_cluster = self.event_bus.get("pattern_cluster", "none")
            panic_zone = self.event_bus.get("panic_zone", False)
            liquidity = self.event_bus.get("liquidity_status", "stable")
            smart_money_score = self.event_bus.get("smart_money_score", 0.0)
            
            # Create NCCO state
            ncco = NCCOState(
                timestamp=timestamp,
                price=current_price,
                strategy=strategy,
                ghost_hash=ghost_hash,
                entropy=entropy,
                coherence=coherence,
                paradox_phase=paradox_phase,
                velocity_class=velocity,
                pattern_cluster=pattern_cluster,
                liquidity_status=liquidity,
                smart_money_score=smart_money_score,
                panic_zone=panic_zone,
                metadata=metadata or {}
            )
            
            # Add to entries
            self.entries.append(ncco)
            
            # Trim entries if needed
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]
            
            # Save NCCO
            self._save_ncco(ncco)
            
            self.logger.info(f"Generated NCCO: {ncco.ncco_id} ({ncco_type})")
            return ncco
    
    def _save_ncco(self, ncco: NCCOState) -> None:
        """
        Save NCCO to disk
        
        Args:
            ncco: NCCO state to save
        """
        try:
            # Convert to dict
            ncco_dict = {
                "ncco_id": ncco.ncco_id,
                "ncco_type": ncco.ncco_type,
                "timestamp": ncco.timestamp,
                "price": ncco.price,
                "strategy": ncco.strategy,
                "ghost_hash": ncco.ghost_hash,
                "entropy": ncco.entropy,
                "coherence": ncco.coherence,
                "paradox_phase": ncco.paradox_phase,
                "velocity_class": ncco.velocity_class,
                "pattern_cluster": ncco.pattern_cluster,
                "liquidity_status": ncco.liquidity_status,
                "smart_money_score": ncco.smart_money_score,
                "panic_zone": ncco.panic_zone,
                "metadata": ncco.metadata
            }
            
            # Save full NCCO
            fname = self.log_dir / f"{ncco.ncco_id}_{int(ncco.timestamp)}.json"
            with open(fname, "w") as f:
                json.dump(ncco_dict, f, indent=2)
            
            # Update index
            index_line = json.dumps(ncco_dict)
            with open(self.log_dir / "ncco_index.jsonl", "a") as f:
                f.write(index_line + "\n")
            
        except Exception as e:
            self.logger.error(f"Error saving NCCO: {e}")
    
    def load_all_nccos(self) -> List[Dict]:
        """
        Load all NCCO entries from disk
        
        Returns:
            List of NCCO entries
        """
        try:
            loaded = []
            for fpath in self.log_dir.glob("*.json"):
                if fpath.name == "ncco_index.jsonl":
                    continue
                with open(fpath) as f:
                    loaded.append(json.load(f))
            return sorted(loaded, key=lambda x: x["timestamp"])
        except Exception as e:
            self.logger.error(f"Error loading NCCOs: {e}")
            return []
    
    def get_recent_nccos(self, limit: int = 100) -> List[Dict]:
        """
        Get most recent NCCO entries
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent NCCO entries
        """
        with self._lock:
            return [self._ncco_to_dict(n) for n in self.entries[-limit:]]
    
    def _ncco_to_dict(self, ncco: NCCOState) -> Dict:
        """
        Convert NCCO state to dictionary
        
        Args:
            ncco: NCCO state
            
        Returns:
            Dictionary representation
        """
        return {
            "ncco_id": ncco.ncco_id,
            "ncco_type": ncco.ncco_type,
            "timestamp": ncco.timestamp,
            "price": ncco.price,
            "strategy": ncco.strategy,
            "ghost_hash": ncco.ghost_hash,
            "entropy": ncco.entropy,
            "coherence": ncco.coherence,
            "paradox_phase": ncco.paradox_phase,
            "velocity_class": ncco.velocity_class,
            "pattern_cluster": ncco.pattern_cluster,
            "liquidity_status": ncco.liquidity_status,
            "smart_money_score": ncco.smart_money_score,
            "panic_zone": ncco.panic_zone,
            "metadata": ncco.metadata
        }
    
    def get_ncco_by_id(self, ncco_id: str) -> Optional[Dict]:
        """
        Get NCCO by ID
        
        Args:
            ncco_id: NCCO ID
            
        Returns:
            NCCO entry or None
        """
        try:
            for fpath in self.log_dir.glob(f"{ncco_id}_*.json"):
                with open(fpath) as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Error getting NCCO by ID: {e}")
            return None
    
    def get_nccos_by_ghost_hash(self, ghost_hash: str) -> List[Dict]:
        """
        Get all NCCOs with matching ghost hash
        
        Args:
            ghost_hash: Ghost hash to match
            
        Returns:
            List of matching NCCO entries
        """
        try:
            matches = []
            for fpath in self.log_dir.glob("*.json"):
                if fpath.name == "ncco_index.jsonl":
                    continue
                with open(fpath) as f:
                    ncco = json.load(f)
                    if ncco["ghost_hash"] == ghost_hash:
                        matches.append(ncco)
            return sorted(matches, key=lambda x: x["timestamp"])
        except Exception as e:
            self.logger.error(f"Error getting NCCOs by ghost hash: {e}")
            return []
    
    def clear_history(self) -> None:
        """Clear NCCO history"""
        with self._lock:
            self.entries.clear()
            for fpath in self.log_dir.glob("*.json"):
                fpath.unlink()
            self._init_index()
            self.logger.info("NCCO history cleared") 