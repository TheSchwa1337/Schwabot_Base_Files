"""
Sequencer Engine
==============

Core orchestrator for Schwabot system that manages trigger sequences,
routes through Ferris → RDE → SFS/UFS → Cooldown → Profit Overlay,
and maintains recursive memory state.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import json
import logging
import threading
from datetime import datetime
import numpy as np
from pathlib import Path

from .hash_trigger_engine import HashTriggerEngine
from .dormant_engine import DormantEngine
from .collapse_engine import CollapseEngine
from .bitmap_engine import BitmapEngine
from .hash_recollection import HashRecollectionSystem

@dataclass
class TriggerState:
    """Represents the current state of a trigger"""
    trigger_id: str
    hash_value: int
    route: str
    last_triggered: Optional[float] = None
    cooldown: int = 0
    success_rate: float = 0.0
    is_active: bool = False

class SequencerEngine:
    """Core orchestrator for Schwabot system"""
    
    def __init__(self, 
                 config_path: str = "config/trigger_registry.json",
                 log_path: str = "logs/sequencer.log"):
        # Initialize components
        self.dormant_engine = DormantEngine()
        self.collapse_engine = CollapseEngine()
        self.hash_engine = HashTriggerEngine(self.dormant_engine, self.collapse_engine)
        self.bitmap_engine = BitmapEngine()
        self.hash_recollection = HashRecollectionSystem()
        
        # Load configuration
        self.config_path = Path(config_path)
        self.trigger_states: Dict[str, TriggerState] = {}
        self.load_config()
        
        # Setup logging
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SequencerEngine')
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Memory tracking
        self.sequence_history: List[Dict] = []
        self.last_update = datetime.now()
        
    def load_config(self) -> None:
        """Load trigger configuration from JSON"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    for trigger_id, data in config.items():
                        self.trigger_states[trigger_id] = TriggerState(
                            trigger_id=trigger_id,
                            hash_value=int(trigger_id, 16),
                            route=data['route'],
                            last_triggered=data.get('last_triggered'),
                            cooldown=data.get('cooldown', 0),
                            success_rate=data.get('success_rate', 0.0)
                        )
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            
    def save_config(self) -> None:
        """Save current trigger states to JSON"""
        try:
            config = {}
            for trigger_id, state in self.trigger_states.items():
                config[trigger_id] = {
                    'route': state.route,
                    'last_triggered': state.last_triggered,
                    'cooldown': state.cooldown,
                    'success_rate': state.success_rate
                }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            
    def process_trigger(self, trigger_id: str, market_data: Dict) -> bool:
        """Process a trigger through the system"""
        with self._lock:
            if trigger_id not in self.trigger_states:
                return False
                
            state = self.trigger_states[trigger_id]
            
            # Check cooldown
            if state.cooldown > 0:
                return False
                
            # Generate bitmap pattern
            bitmap = self.bitmap_engine.infer_current_bitmap(market_data)
            
            # Process through hash engine
            triggered = self.hash_engine.process_hash(state.hash_value, market_data)
            
            if triggered:
                # Update state
                state.last_triggered = datetime.now().timestamp()
                state.is_active = True
                
                # Log sequence
                sequence = {
                    'trigger_id': trigger_id,
                    'timestamp': state.last_triggered,
                    'route': state.route,
                    'market_data': market_data
                }
                self.sequence_history.append(sequence)
                
                # Save updated config
                self.save_config()
                
                return True
                
            return False
            
    def get_active_triggers(self) -> List[TriggerState]:
        """Get list of currently active triggers"""
        return [state for state in self.trigger_states.values() 
                if state.is_active]
                
    def deactivate_trigger(self, trigger_id: str) -> None:
        """Deactivate a trigger"""
        with self._lock:
            if trigger_id in self.trigger_states:
                self.trigger_states[trigger_id].is_active = False
                self.save_config()
                
    def update_trigger_stats(self, trigger_id: str, success: bool) -> None:
        """Update trigger statistics"""
        with self._lock:
            if trigger_id in self.trigger_states:
                state = self.trigger_states[trigger_id]
                
                # Update success rate with exponential moving average
                alpha = 0.1  # Learning rate
                state.success_rate = (1 - alpha) * state.success_rate + alpha * float(success)
                
                # Update cooldown based on success
                if success:
                    state.cooldown = max(0, state.cooldown - 1)
                else:
                    state.cooldown = min(10, state.cooldown + 1)
                    
                self.save_config()
                
    def get_sequence_history(self, limit: int = 100) -> List[Dict]:
        """Get recent sequence history"""
        return self.sequence_history[-limit:]
        
    def clear_history(self) -> None:
        """Clear sequence history"""
        with self._lock:
            self.sequence_history.clear()
            
    def get_trigger_stats(self) -> Dict[str, Dict]:
        """Get statistics for all triggers"""
        stats = {}
        for trigger_id, state in self.trigger_states.items():
            stats[trigger_id] = {
                'success_rate': state.success_rate,
                'cooldown': state.cooldown,
                'is_active': state.is_active,
                'last_triggered': state.last_triggered
            }
        return stats

# Example usage
if __name__ == "__main__":
    # Initialize sequencer
    sequencer = SequencerEngine()
    
    # Example market data
    market_data = {
        'price': 100.0,
        'volume': 1000.0,
        'timestamp': datetime.now().timestamp()
    }
    
    # Process trigger
    triggered = sequencer.process_trigger("0x8845", market_data)
    print(f"Trigger activated: {triggered}")
    
    # Get active triggers
    active = sequencer.get_active_triggers()
    print(f"Active triggers: {len(active)}") 