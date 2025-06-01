"""
Bus Core
========

Implements the event bus system that coordinates between the cursor,
braid pattern engine, and fractal state management.
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from .cursor_engine import Cursor, CursorState
from .braid_pattern_engine import BraidPattern, BraidPatternEngine
from .fractal_core import ForeverFractalCore, FractalState
from .triplet_matcher import TripletMatcher, TripletMatch
from .cooldown_manager import CooldownManager, CooldownScope, FractalCooldownState

@dataclass
class BusEvent:
    """Represents an event in the bus system"""
    type: str
    data: Any
    timestamp: float
    fractal_state: Optional[FractalCooldownState] = None

class BusCore:
    """Core event bus implementation with fractal integration"""
    
    def __init__(self):
        self.cursor = Cursor()
        self.pattern_engine = BraidPatternEngine()
        self.handlers: Dict[str, List[Callable]] = {}
        self.event_history: List[BusEvent] = []
        
        # Initialize fractal components
        self.fractal_core = ForeverFractalCore(
            decay_power=2.0,
            terms=50,
            dimension=3
        )
        self.triplet_matcher = TripletMatcher(
            fractal_core=self.fractal_core,
            epsilon=0.1,
            min_coherence=0.7
        )
        self.cooldown_manager = CooldownManager([])
        
    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler for a specific event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
    def dispatch_event(self, event: BusEvent):
        """Dispatch an event to registered handlers with fractal state"""
        self.event_history.append(event)
        
        # Generate fractal state if not provided
        if event.fractal_state is None:
            fractal_vector = self.fractal_core.generate_fractal_vector(
                t=event.timestamp,
                phase_shift=event.data.get('phase_angle', 0.0)
            )
            event.fractal_state = FractalCooldownState(
                vector=fractal_vector,
                phase=event.data.get('phase_angle', 0.0),
                entropy=event.data.get('entropy', 0.0),
                timestamp=event.timestamp
            )
        
        # Check for triplet matches
        if len(self.event_history) >= 3:
            recent_events = self.event_history[-3:]
            recent_states = [e.fractal_state for e in recent_events]
            match = self.triplet_matcher.find_matching_triplet(recent_states)
            if match:
                event.fractal_state.coherence_score = match.coherence
                event.fractal_state.is_mirror = match.is_mirror
                
                # Apply fractal correction if needed
                if match.coherence > 0.9:
                    self._apply_fractal_correction(event, match)
        
        # Register with cooldown manager
        self.cooldown_manager.register_event(event.type, event.data)
        
        # Dispatch to handlers
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                handler(event)
                
    def _apply_fractal_correction(self, event: BusEvent, match: TripletMatch) -> None:
        """Apply fractal-based correction to event state"""
        # Get correction vector
        correction = self.fractal_core.compute_correction_vector(match.states)
        
        # Update event state
        event.fractal_state.vector = correction
        
        # Dispatch correction event
        self.dispatch_event(BusEvent(
            type="fractal_correction",
            data={
                "original_vector": event.fractal_state.vector,
                "corrected_vector": correction,
                "coherence": match.coherence
            },
            timestamp=event.timestamp,
            fractal_state=event.fractal_state
        ))
                
    def process_tick(self, triplet: tuple, timestamp: float):
        """Process a new tick through the cursor and pattern engine with fractal state"""
        # Process through cursor
        pattern = self.cursor.tick(triplet, timestamp)
        
        # Generate fractal state
        fractal_vector = self.fractal_core.generate_fractal_vector(
            t=timestamp,
            phase_shift=pattern.phase if pattern else 0.0
        )
        
        fractal_state = FractalCooldownState(
            vector=fractal_vector,
            phase=pattern.phase if pattern else 0.0,
            entropy=pattern.entropy if pattern else 0.0,
            timestamp=timestamp
        )
        
        # Dispatch cursor state event
        self.dispatch_event(BusEvent(
            type="cursor_state",
            data=self.cursor.state,
            timestamp=timestamp,
            fractal_state=fractal_state
        ))
        
        # If pattern detected, dispatch pattern event
        if pattern:
            self.dispatch_event(BusEvent(
                type="braid_pattern",
                data=pattern,
                timestamp=timestamp,
                fractal_state=fractal_state
            ))
            
    def get_cursor_state(self) -> Optional[CursorState]:
        """Get current cursor state"""
        return self.cursor.state
        
    def get_current_pattern(self) -> Optional[BraidPattern]:
        """Get current braid pattern"""
        return self.cursor.get_current_pattern()
        
    def get_pattern_frequency(self, pattern_name: str, window: int = 100) -> float:
        """Get frequency of a specific pattern"""
        return self.cursor.get_pattern_frequency(pattern_name, window)
        
    def get_event_history(self, event_type: Optional[str] = None, 
                         limit: Optional[int] = None) -> List[BusEvent]:
        """Get event history, optionally filtered by type and limited"""
        events = self.event_history
        if event_type:
            events = [e for e in events if e.type == event_type]
        if limit:
            events = events[-limit:]
        return events
        
    def get_fractal_metrics(self, event_type: Optional[str] = None) -> Dict[str, float]:
        """Get fractal metrics for events"""
        events = self.get_event_history(event_type)
        if not events:
            return {}
            
        # Calculate metrics from fractal states
        coherence_scores = [e.fractal_state.coherence_score for e in events if e.fractal_state]
        mirror_count = sum(1 for e in events if e.fractal_state and e.fractal_state.is_mirror)
        
        return {
            'avg_coherence': np.mean(coherence_scores) if coherence_scores else 0.0,
            'max_coherence': max(coherence_scores) if coherence_scores else 0.0,
            'mirror_ratio': mirror_count / len(events) if events else 0.0,
            'total_events': len(events)
        }
        
    def clear_history(self):
        """Clear all history"""
        self.event_history.clear()
        self.cursor.clear_history()
        self.pattern_engine.clear_history()
        
    def add_custom_pattern(self, name: str, pattern: List[tuple]):
        """Add a custom pattern to the pattern library"""
        self.cursor.add_custom_pattern(name, pattern)

# Example usage
if __name__ == "__main__":
    bus = BusCore()
    
    # Test event handler
    def handle_tick(event: BusEvent):
        print(f"Handling tick from {event.type}: {event.data}")
        if event.fractal_state:
            print(f"Fractal coherence: {event.fractal_state.coherence_score}")
    
    # Register handler
    bus.register_handler("tick", handle_tick)
    
    # Dispatch test events
    bus.dispatch_event(BusEvent(
        type="tick",
        data={"price": 100.0, "volume": 1000, "phase_angle": 0.5},
        timestamp=datetime.now().timestamp()
    ))
    
    # Show event history
    print("\nEvent History:")
    for event in bus.get_event_history():
        print(f"  {event.type}: {event.data}")
        
    # Show fractal metrics
    print("\nFractal Metrics:")
    metrics = bus.get_fractal_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}") 