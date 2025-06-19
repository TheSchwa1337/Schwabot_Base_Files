"""
Bus Event Handler Module
======================

This module defines the event types and payloads for the Schwabot bus system,
including pattern matching, node activation, and memory updates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from jsonschema import validate
import pytest

@dataclass
class BusEvent:
    """Base class for all bus events"""
    event_type: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow()

@dataclass
class PatternMatchEvent(BusEvent):
    """Event for pattern matches in cyclic analysis"""
    def __init__(self, source: str, **kwargs):
        super().__init__(source=source)
        self.pattern_hash = kwargs.get('pattern_hash', None)
        self.confidence = kwargs.get('confidence', None)
        self.vector = kwargs.get('vector', None)

class NodeActivationEvent(BusEvent):
    """Event for node activation/deactivation"""
    
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
        self.node_id = node_id
        self.status = status
        self.reason = reason

class MemoryUpdateEvent(BusEvent):
    """Event for memory lane updates"""
    
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
        self.lane_id = lane_id
        self.data = data
        self.operation = operation

class SymmetryBreakEvent(BusEvent):
    """Event for symmetry break detection"""
    
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
        self.vector = vector
        self.break_type = break_type

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

# Centralized Config Loader Utility
def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    try:
        with path.open("r") as f:
            config = yaml.safe_load(f)
        
        # Define a JSON schema for the configuration
        schema = {
            "type": "object",
            "properties": {
                "fractal": {
                    "type": "object",
                    "properties": {
                        "decay_power": {"type": "number"}
                    },
                    "required": ["decay_power"]
                }
            },
            "required": ["fractal"]
        }
        
        # Validate the configuration against the schema
        validate(instance=config, schema=schema)
        
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error in {path}: {e}")

def create_default_config(path: Path, defaults: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(defaults, f)

# Unit Tests for Config Paths
def test_matrix_config_loads():
    from core.matrix_fault_resolver import MatrixFaultResolver
    resolver = MatrixFaultResolver()  # should not crash

def test_invalid_yaml_throws():
    with pytest.raises(ValueError):
        load_yaml_config(Path("tests/fixtures/bad_config.yaml"))

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
        "confidence": 0.95,
        "vector": np.array([1, 2, 3])  # Example vector
    }
    updated_event = bus_core.update_event_state(event_type, source, timestamp, payload)
    print(f"Updated event state: {updated_event}") 