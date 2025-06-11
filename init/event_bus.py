"""
Event Bus for Schwabot System
Central state management and event distribution system
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EventState:
    """State container for event data"""
    value: Any
    timestamp: float
    source: str
    metadata: Dict[str, Any]

class EventBus:
    """Central event bus for state management and event distribution"""
    
    def __init__(self, log_dir: str = "logs"):
        self.state: Dict[str, EventState] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.history: List[Dict] = []
        self.max_history = 1000
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.Lock()
    
    def update(self, key: str, value: Any, source: str = "system", metadata: Optional[Dict] = None) -> None:
        """
        Update state and notify subscribers
        
        Args:
            key: State key
            value: New value
            source: Source of update
            metadata: Additional metadata
        """
        with self._lock:
            # Create event state
            event_state = EventState(
                value=value,
                timestamp=datetime.now().timestamp(),
                source=source,
                metadata=metadata or {}
            )
            
            # Update state
            self.state[key] = event_state
            
            # Log update
            self.logger.info(f"State updated: {key}={value} (source: {source})")
            
            # Record in history
            self.history.append({
                'key': key,
                'value': value,
                'timestamp': event_state.timestamp,
                'source': source,
                'metadata': metadata
            })
            
            # Trim history if needed
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Notify subscribers
            if key in self.subscribers:
                for callback in self.subscribers[key]:
                    try:
                        callback(event_state)
                    except Exception as e:
                        self.logger.error(f"Error in subscriber callback: {e}")
    
    def subscribe(self, key: str, callback: Callable) -> None:
        """
        Subscribe to state updates
        
        Args:
            key: State key to subscribe to
            callback: Callback function
        """
        with self._lock:
            if key not in self.subscribers:
                self.subscribers[key] = []
            self.subscribers[key].append(callback)
            self.logger.info(f"New subscriber for {key}")
    
    def unsubscribe(self, key: str, callback: Callable) -> None:
        """
        Unsubscribe from state updates
        
        Args:
            key: State key
            callback: Callback function to remove
        """
        with self._lock:
            if key in self.subscribers and callback in self.subscribers[key]:
                self.subscribers[key].remove(callback)
                self.logger.info(f"Removed subscriber for {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get current state value
        
        Args:
            key: State key
            default: Default value if key not found
            
        Returns:
            Current state value or default
        """
        with self._lock:
            if key in self.state:
                return self.state[key].value
            return default
    
    def get_state(self, key: str) -> Optional[EventState]:
        """
        Get full state object
        
        Args:
            key: State key
            
        Returns:
            EventState object or None
        """
        with self._lock:
            return self.state.get(key)
    
    def get_history(self, key: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get state history
        
        Args:
            key: Optional key to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of historical state entries
        """
        with self._lock:
            if key:
                filtered = [h for h in self.history if h['key'] == key]
                return filtered[-limit:]
            return self.history[-limit:]
    
    def clear_history(self) -> None:
        """Clear state history"""
        with self._lock:
            self.history.clear()
            self.logger.info("History cleared")
    
    def get_all_state(self) -> Dict[str, Any]:
        """
        Get all current state values
        
        Returns:
            Dictionary of all state values
        """
        with self._lock:
            return {k: v.value for k, v in self.state.items()} 