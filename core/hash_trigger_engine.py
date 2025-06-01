"""
Hash Trigger Engine
=================

Implements the hash-based trigger system that processes 16-bit hash maps
and connects them to price patterns and Euler-coded triggers.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
import hashlib
from .cursor_engine import Cursor, CursorState
from .dormant_engine import DormantEngine
from .collapse_engine import CollapseEngine

@dataclass
class HashTrigger:
    """Represents a hash-based trigger"""
    trigger_id: str
    hash_value: int  # 16-bit hash
    price_map: Dict[float, float]  # price -> target mapping
    euler_phase: float  # Euler phase angle
    is_active: bool = False
    last_triggered: Optional[float] = None

class HashTriggerEngine:
    """Manages hash-based triggers and their connections to price patterns"""
    
    def __init__(self, dormant_engine: DormantEngine, collapse_engine: CollapseEngine):
        self.dormant_engine = dormant_engine
        self.collapse_engine = collapse_engine
        self.triggers: Dict[str, HashTrigger] = {}
        self.active_triggers: Set[str] = set()
        self.hash_history: List[Tuple[int, float]] = []  # (hash, timestamp)
        self.bit_map: Dict[int, List[str]] = {}  # bit position -> trigger_ids
        
    def register_trigger(self, trigger_id: str, price_map: Dict[float, float],
                        euler_phase: float = 0.0) -> None:
        """Register a new hash trigger"""
        # Generate 16-bit hash from trigger_id
        hash_bytes = hashlib.sha256(trigger_id.encode()).digest()
        hash_value = int.from_bytes(hash_bytes[:2], 'big')  # Use first 2 bytes for 16-bit
        
        trigger = HashTrigger(
            trigger_id=trigger_id,
            hash_value=hash_value,
            price_map=price_map,
            euler_phase=euler_phase,
            is_active=False
        )
        
        self.triggers[trigger_id] = trigger
        
        # Update bit map
        for bit_pos in range(16):
            if hash_value & (1 << bit_pos):
                if bit_pos not in self.bit_map:
                    self.bit_map[bit_pos] = []
                self.bit_map[bit_pos].append(trigger_id)
    
    def process_hash(self, hash_value: int, cursor_state: CursorState) -> List[str]:
        """Process a hash value and check for trigger activations"""
        triggered_ids = []
        
        # Store hash in history
        self.hash_history.append((hash_value, cursor_state.timestamp))
        
        # Check each bit position
        for bit_pos in range(16):
            if hash_value & (1 << bit_pos):
                # Get triggers mapped to this bit
                trigger_ids = self.bit_map.get(bit_pos, [])
                
                for trigger_id in trigger_ids:
                    trigger = self.triggers[trigger_id]
                    if not trigger.is_active:
                        # Check price mapping
                        if self._check_price_mapping(trigger, cursor_state):
                            # Check Euler phase alignment
                            if self._check_euler_phase(trigger, cursor_state):
                                # Activate trigger
                                trigger.is_active = True
                                trigger.last_triggered = cursor_state.timestamp
                                self.active_triggers.add(trigger_id)
                                triggered_ids.append(trigger_id)
                                
                                # Notify dormant engine
                                self.dormant_engine.check_activation(cursor_state)
                                
                                # Check for collapse
                                self.collapse_engine.check_collapse([cursor_state])
        
        return triggered_ids
    
    def _check_price_mapping(self, trigger: HashTrigger, 
                           cursor_state: CursorState) -> bool:
        """Check if current price matches trigger's price mapping"""
        current_price = cursor_state.triplet[0]
        
        # Find closest mapped price
        closest_price = min(trigger.price_map.keys(),
                          key=lambda p: abs(p - current_price))
        
        # Check if within acceptable range
        price_diff = abs(current_price - closest_price)
        return price_diff < 0.01  # 1% threshold
    
    def _check_euler_phase(self, trigger: HashTrigger, 
                          cursor_state: CursorState) -> bool:
        """Check if current state aligns with trigger's Euler phase"""
        current_angle = cursor_state.braid_angle
        phase_diff = abs(current_angle - trigger.euler_phase) % 360.0
        return phase_diff < 5.0 or phase_diff > 355.0
    
    def deactivate_trigger(self, trigger_id: str) -> None:
        """Deactivate a trigger"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].is_active = False
            if trigger_id in self.active_triggers:
                self.active_triggers.remove(trigger_id)
    
    def get_active_triggers(self) -> List[HashTrigger]:
        """Get list of currently active triggers"""
        return [self.triggers[trigger_id] 
                for trigger_id in self.active_triggers]
    
    def get_trigger_history(self, trigger_id: str) -> List[Tuple[float, bool]]:
        """Get activation history for a trigger"""
        if trigger_id in self.triggers:
            trigger = self.triggers[trigger_id]
            return [(trigger.last_triggered, trigger.is_active)]
        return []
    
    def clear_history(self) -> None:
        """Clear all trigger history"""
        self.hash_history.clear()
        for trigger in self.triggers.values():
            trigger.last_triggered = None
            trigger.is_active = False
        self.active_triggers.clear()

# Example usage
if __name__ == "__main__":
    # Initialize engines
    dormant_engine = DormantEngine()
    collapse_engine = CollapseEngine()
    hash_engine = HashTriggerEngine(dormant_engine, collapse_engine)
    
    # Register test trigger
    hash_engine.register_trigger(
        trigger_id="TRIGGER_001",
        price_map={100.0: 105.0, 200.0: 210.0},
        euler_phase=45.0
    )
    
    # Test hash processing
    test_hash = 0x1234  # Example 16-bit hash
    cursor_state = CursorState(
        triplet=(100.0, 100.1, 100.2),
        delta_idx=1,
        braid_angle=45.0,
        timestamp=1234567890.0
    )
    
    triggered = hash_engine.process_hash(test_hash, cursor_state)
    print(f"Triggered IDs: {triggered}") 