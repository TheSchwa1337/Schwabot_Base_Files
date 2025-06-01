"""
Bus Event Handler Module
======================

This module defines the event types and payloads for the Schwabot bus system,
including pattern matching, node activation, and memory updates.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from datetime import datetime

@dataclass
class BusEvent:
    """Base class for all bus events"""
    event_type: str
    source: str
    timestamp: datetime
    payload: Dict[str, Any]
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow()

@dataclass
class PatternMatchEvent(BusEvent):
    """Event for pattern matches in cyclic analysis"""
    pattern_hash: str
    confidence: float
    vector: np.ndarray
    
    def __init__(self, source: str, pattern_hash: str, confidence: float, vector: np.ndarray):
        super().__init__(
            event_type="pattern_match",
            source=source,
            timestamp=None,
            payload={
                "pattern_hash": pattern_hash,
                "confidence": confidence,
                "vector": vector.tolist()
            }
        )

@dataclass
class NodeActivationEvent(BusEvent):
    """Event for node activation/deactivation"""
    node_id: str
    status: str  # "active", "dormant", "locked"
    reason: str
    
    def __init__(self, source: str, node_id: str, status: str, reason: str):
        super().__init__(
            event_type="node_activation",
            source=source,
            timestamp=None,
            payload={
                "node_id": node_id,
                "status": status,
                "reason": reason
            }
        )

@dataclass
class MemoryUpdateEvent(BusEvent):
    """Event for memory lane updates"""
    lane_id: str
    data: Any
    operation: str  # "push", "pop", "update"
    
    def __init__(self, source: str, lane_id: str, data: Any, operation: str):
        super().__init__(
            event_type="memory_update",
            source=source,
            timestamp=None,
            payload={
                "lane_id": lane_id,
                "data": data,
                "operation": operation
            }
        )

@dataclass
class SymmetryBreakEvent(BusEvent):
    """Event for symmetry break detection"""
    vector: np.ndarray
    break_type: str  # "998", "inversion", "corruption"
    
    def __init__(self, source: str, vector: np.ndarray, break_type: str):
        super().__init__(
            event_type="symmetry_break",
            source=source,
            timestamp=None,
            payload={
                "vector": vector.tolist(),
                "break_type": break_type
            }
        )

# Event type registry
EVENT_TYPES = {
    "pattern_match": PatternMatchEvent,
    "node_activation": NodeActivationEvent,
    "memory_update": MemoryUpdateEvent,
    "symmetry_break": SymmetryBreakEvent
}

@dataclass
class BusCore:
    """Class to handle fractal state updates and manage CPU rendering"""
    
    def __init__(self):
        self.fractal_state = None
    
    def generate_fractal_vector(self, timestamp: datetime, phase_angle: float) -> np.ndarray:
        """
        Generate a fractal vector based on the current timestamp and phase angle
        
        Args:
            timestamp: Current timestamp
            phase_angle: Phase angle for fractal generation
            
        Returns:
            Fractal vector as a numpy array
        """
        # Example fractal generation logic
        fractal_vector = np.sin(timestamp.timestamp() * phase_angle)
        return fractal_vector
    
    def update_event_state(self, event_type: str, source: str, timestamp: datetime, payload: Dict[str, Any]):
        """
        Update the event state based on the current fractal vector
        
        Args:
            event_type: Type of event
            source: Source of the event
            timestamp: Current timestamp
            payload: Payload for the event
            
        Returns:
            Updated event state
        """
        if self.fractal_state is None:
            raise ValueError("Fractal state not set. Please generate a fractal vector first.")
        
        # Example event state update logic
        updated_payload = {
            **payload,
            "fractal_vector": self.fractal_state.tolist()
        }
        return create_event(event_type, source, timestamp=timestamp, payload=updated_payload)

def create_event(event_type: str, source: str, **kwargs) -> BusEvent:
    """
    Factory function to create bus events
    
    Args:
        event_type: Type of event to create
        source: Source of the event
        **kwargs: Additional arguments for the event
        
    Returns:
        Created BusEvent instance
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type}")
        
    return EVENT_TYPES[event_type](source=source, **kwargs)

# Example usage
if __name__ == "__main__":
    # Create a BusCore instance
    bus_core = BusCore()
    
    # Generate a fractal vector
    timestamp = datetime.utcnow()
    phase_angle = 0.123
    fractal_vector = bus_core.generate_fractal_vector(timestamp, phase_angle)
    print(f"Generated fractal vector: {fractal_vector}")
    
    # Update event state with the generated fractal vector
    event_type = "pattern_match"
    source = "cyclic_core"
    payload = {
        "pattern_hash": "abc123",
        "confidence": 0.95
    }
    updated_event = bus_core.update_event_state(event_type, source, timestamp, payload)
    print(f"Updated event state: {updated_event}") 