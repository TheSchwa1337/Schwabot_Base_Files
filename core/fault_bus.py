"""
Fault Bus
========

Handles system-wide event handling and fault responses.
Provides unified fault response mechanism and event propagation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Callable
import logging
import json
import os

@dataclass
class FaultBusEvent:
    tick: int
    module: str
    type: str
    severity: float
    timestamp: str = datetime.now().isoformat()
    metadata: Optional[Dict] = None

class FaultResolver:
    """Base class for fault resolution strategies."""
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None):
        """Handle a fault event."""
        raise NotImplementedError

class ThermalFaultResolver(FaultResolver):
    """Handles thermal-related faults."""
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None):
        if fault_type == "thermal_high":
            logging.warning(f"High thermal load detected: {severity}")
            # Implement thermal mitigation strategy
        elif fault_type == "thermal_critical":
            logging.error(f"Critical thermal condition: {severity}")
            # Implement emergency thermal response

class ProfitFaultResolver(FaultResolver):
    """Handles profit-related faults."""
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None):
        if fault_type == "profit_low":
            logging.warning(f"Low profit detected: {severity}")
            # Implement profit optimization strategy
        elif fault_type == "profit_critical":
            logging.error(f"Critical profit condition: {severity}")
            # Implement emergency profit response

class BitmapFaultResolver(FaultResolver):
    """Handles bitmap-related faults."""
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None):
        if fault_type == "bitmap_corrupt":
            logging.error(f"Bitmap corruption detected: {severity}")
            # Implement bitmap recovery strategy
        elif fault_type == "bitmap_overflow":
            logging.warning(f"Bitmap overflow detected: {severity}")
            # Implement bitmap cleanup strategy

class FaultBus:
    def __init__(self, log_path: str = "logs/faults"):
        self.queue: List[FaultBusEvent] = []
        self.resolvers: Dict[str, FaultResolver] = {}
        self.log_path = log_path
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Create log directory if it doesn't exist
        os.makedirs(log_path, exist_ok=True)
        
        # Initialize default resolvers
        self.register_resolver("thermal", ThermalFaultResolver())
        self.register_resolver("profit", ProfitFaultResolver())
        self.register_resolver("bitmap", BitmapFaultResolver())

    def register_resolver(self, fault_type: str, resolver: FaultResolver):
        """Register a resolver for a specific fault type."""
        self.resolvers[fault_type] = resolver

    def register_handler(self, event_type: str, handler: Callable):
        """Register an event handler for a specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def push(self, event: FaultBusEvent):
        """Push a new fault event to the queue."""
        self.queue.append(event)
        self._log_event(event)
        self._trigger_handlers(event)

    def dispatch(self):
        """Process all queued fault events."""
        while self.queue:
            event = self.queue.pop(0)
            self._handle_event(event)

    def _handle_event(self, event: FaultBusEvent):
        """Handle a single fault event."""
        # Get appropriate resolver
        resolver = self.resolvers.get(event.type.split('_')[0])
        if resolver:
            resolver.handle_fault(event.type, event.severity, event.metadata)
        else:
            logging.warning(f"No resolver found for fault type: {event.type}")

    def _trigger_handlers(self, event: FaultBusEvent):
        """Trigger registered event handlers."""
        handlers = self.event_handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logging.error(f"Error in event handler: {e}")

    def _log_event(self, event: FaultBusEvent):
        """Log fault event to file."""
        try:
            log_file = os.path.join(self.log_path, f"faults_{datetime.now().strftime('%Y%m%d')}.log")
            with open(log_file, 'a') as f:
                f.write(json.dumps({
                    'tick': event.tick,
                    'module': event.module,
                    'type': event.type,
                    'severity': event.severity,
                    'timestamp': event.timestamp,
                    'metadata': event.metadata
                }) + '\n')
        except Exception as e:
            logging.error(f"Error logging fault event: {e}")

    def get_event_stats(self) -> Dict:
        """Get statistics about fault events."""
        stats = {
            'queue_size': len(self.queue),
            'resolver_count': len(self.resolvers),
            'handler_count': sum(len(handlers) for handlers in self.event_handlers.values()),
            'event_types': list(self.event_handlers.keys())
        }
        return stats

    def cleanup_old_logs(self, max_age_days: int = 7):
        """Clean up old fault log files."""
        try:
            current_time = datetime.now()
            for filename in os.listdir(self.log_path):
                if filename.startswith('faults_') and filename.endswith('.log'):
                    file_path = os.path.join(self.log_path, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if (current_time - file_time).days > max_age_days:
                        os.remove(file_path)
        except Exception as e:
            logging.error(f"Error cleaning up old logs: {e}")

# Example usage:
if __name__ == "__main__":
    # Create fault bus
    fault_bus = FaultBus()
    
    # Register custom event handler
    def handle_thermal_event(event: FaultBusEvent):
        print(f"Thermal event detected: {event.severity}")
    
    fault_bus.register_handler("thermal_high", handle_thermal_event)
    
    # Push some test events
    fault_bus.push(FaultBusEvent(
        tick=1,
        module="thermal_monitor",
        type="thermal_high",
        severity=0.8,
        metadata={"temperature": 75.0}
    ))
    
    # Process events
    fault_bus.dispatch() 