"""
Phase Map
========

Tracks current phase state, urgency, and hash mapping.
Manages phase transitions and state persistence.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from .phase_loader import PhaseConfigLoader
from .sha_mapper import SHAMapper

logger = logging.getLogger(__name__)

@dataclass
class PhaseState:
    """Represents the current state of a phase"""
    phase: str
    urgency: float
    memory_coherence: float
    hash_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "phase": self.phase,
            "urgency": self.urgency,
            "memory_coherence": self.memory_coherence,
            "hash_id": self.hash_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhaseState':
        """Create state from dictionary"""
        return cls(
            phase=data["phase"],
            urgency=data["urgency"],
            memory_coherence=data["memory_coherence"],
            hash_id=data["hash_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=data["metrics"]
        )

class PhaseMap:
    """Manages phase state and transitions"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the phase map"""
        self.phase_loader = PhaseConfigLoader(config_path)
        self.sha_mapper = SHAMapper()
        self.state_log: List[PhaseState] = []
        self._load_state()
        
    def _load_state(self) -> None:
        """Load saved state from file"""
        try:
            state_path = Path("state/phase_state.json")
            if state_path.exists():
                with open(state_path, 'r') as f:
                    data = json.load(f)
                    self.state_log = [
                        PhaseState.from_dict(entry)
                        for entry in data.get("states", [])
                    ]
                logger.info(f"Loaded {len(self.state_log)} phase states")
                
        except Exception as e:
            logger.error(f"Error loading phase state: {e}")
            
    def _save_state(self) -> None:
        """Save current state to file"""
        try:
            state_path = Path("state/phase_state.json")
            state_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "states": [state.to_dict() for state in self.state_log],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(state_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving phase state: {e}")
            
    def update_phase(self, phase: str, urgency: float, coherence: float, 
                    hash_id: str, metrics: Dict[str, float]) -> None:
        """Update current phase state"""
        # Validate phase
        if not self.phase_loader.get_phase_region(phase):
            raise ValueError(f"Invalid phase: {phase}")
            
        # Create new state
        state = PhaseState(
            phase=phase,
            urgency=urgency,
            memory_coherence=coherence,
            hash_id=hash_id,
            timestamp=datetime.now(),
            metrics=metrics
        )
        
        # Add to log
        self.state_log.append(state)
        
        # Save state
        self._save_state()
        
        # Log transition
        self._log_transition(state)
        
    def _log_transition(self, state: PhaseState) -> None:
        """Log phase transition"""
        try:
            transition_path = Path("logs/phase_transitions.json")
            transition_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing transitions
            transitions = []
            if transition_path.exists():
                with open(transition_path, 'r') as f:
                    transitions = json.load(f)
                    
            # Add new transition
            transitions.append({
                "from_phase": self.state_log[-2].phase if len(self.state_log) > 1 else None,
                "to_phase": state.phase,
                "urgency": state.urgency,
                "memory_coherence": state.memory_coherence,
                "hash_id": state.hash_id,
                "timestamp": state.timestamp.isoformat(),
                "metrics": state.metrics
            })
            
            # Save transitions
            with open(transition_path, 'w') as f:
                json.dump(transitions, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging phase transition: {e}")
            
    def latest(self) -> Optional[PhaseState]:
        """Get latest phase state"""
        return self.state_log[-1] if self.state_log else None
        
    def get_phase_history(self, phase: Optional[str] = None) -> List[PhaseState]:
        """Get history of phase states"""
        if phase:
            return [state for state in self.state_log if state.phase == phase]
        return self.state_log.copy()
        
    def get_phase_statistics(self) -> Dict[str, Any]:
        """Get statistics about phase transitions"""
        stats = {
            "total_transitions": len(self.state_log),
            "phase_counts": {},
            "average_urgency": 0.0,
            "average_coherence": 0.0,
            "last_transition": None
        }
        
        if not self.state_log:
            return stats
            
        # Count phases
        for state in self.state_log:
            stats["phase_counts"][state.phase] = stats["phase_counts"].get(state.phase, 0) + 1
            stats["average_urgency"] += state.urgency
            stats["average_coherence"] += state.memory_coherence
            
        # Calculate averages
        stats["average_urgency"] /= len(self.state_log)
        stats["average_coherence"] /= len(self.state_log)
        
        # Get last transition
        if len(self.state_log) > 1:
            last = self.state_log[-1]
            prev = self.state_log[-2]
            stats["last_transition"] = {
                "from": prev.phase,
                "to": last.phase,
                "timestamp": last.timestamp.isoformat(),
                "urgency": last.urgency,
                "coherence": last.memory_coherence
            }
            
        return stats
        
    def is_valid_transition(self, from_phase: str, to_phase: str) -> bool:
        """Check if a phase transition is valid"""
        return self.phase_loader.is_valid_transition(from_phase, to_phase)
        
    def get_phase_metrics(self, phase: str) -> Dict[str, List[float]]:
        """Get historical metrics for a phase"""
        metrics = {}
        phase_states = [state for state in self.state_log if state.phase == phase]
        
        if not phase_states:
            return metrics
            
        # Initialize metrics
        for metric in phase_states[0].metrics.keys():
            metrics[metric] = []
            
        # Collect metrics
        for state in phase_states:
            for metric, value in state.metrics.items():
                metrics[metric].append(value)
                
        return metrics 