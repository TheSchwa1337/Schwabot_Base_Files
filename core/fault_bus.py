"""
Fault Bus
========

Handles system-wide event handling and fault responses.
Provides unified fault response mechanism and event propagation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Callable
import logging
import json
import os
from enum import Enum
import yaml  # Ensure yaml is installed in requirements.txt
import asyncio

class FaultType(Enum):
    THERMAL_HIGH = "thermal_high"
    THERMAL_CRITICAL = "thermal_critical"
    PROFIT_LOW = "profit_low"
    PROFIT_CRITICAL = "profit_critical"
    BITMAP_CORRUPT = "bitmap_corrupt"
    BITMAP_OVERFLOW = "bitmap_overflow"
    GPU_OVERLOAD = "gpu_overload"
    GPU_DRIVER_CRASH = "gpu_driver_crash"
    # Extend this list with new categories as needed

@dataclass
class FaultBusEvent:
    tick: int
    module: str
    type: FaultType
    severity: float
    timestamp: str = datetime.now().isoformat()
    metadata: Optional[Dict] = None

class FaultResolver(ABC):
    @abstractmethod
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None):
        pass

class ThermalFaultResolver(FaultResolver):
    """Handles thermal-related faults."""
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None):
        if fault_type == FaultType.THERMAL_HIGH.value:
            logging.warning(f"High thermal load detected: {severity}")
            # Implement thermal mitigation strategy
        elif fault_type == FaultType.THERMAL_CRITICAL.value:
            logging.error(f"Critical thermal condition: {severity}")
            # Implement emergency thermal response

class ProfitFaultResolver(FaultResolver):
    """Handles profit-related faults."""
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None):
        if fault_type == FaultType.PROFIT_LOW.value:
            logging.warning(f"Low profit detected: {severity}")
            # Implement profit optimization strategy
        elif fault_type == FaultType.PROFIT_CRITICAL.value:
            logging.error(f"Critical profit condition: {severity}")
            # Implement emergency profit response

class BitmapFaultResolver(FaultResolver):
    """Handles bitmap-related faults."""
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None):
        if fault_type == FaultType.BITMAP_CORRUPT.value:
            logging.error(f"Bitmap corruption detected: {severity}")
            # Implement bitmap recovery strategy
        elif fault_type == FaultType.BITMAP_OVERFLOW.value:
            logging.warning(f"Bitmap overflow detected: {severity}")
            # Implement bitmap cleanup strategy

fault_resolver_registry = {}

def register_fault_resolver(name: str):
    def decorator(cls):
        fault_resolver_registry[name] = cls()
        return cls
    return decorator

@register_fault_resolver("gpu")
class GPUFaultResolver(FaultResolver):
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None):
        if fault_type == FaultType.GPU_OVERLOAD.value:
            logging.warning(f"GPU overload: {severity}")
        elif fault_type == FaultType.GPU_DRIVER_CRASH.value:
            logging.error(f"GPU driver crash detected!")

class EventSeverity:
    INFO = 0.1
    WARNING = 0.5
    CRITICAL = 0.9

class FaultBus:
    def __init__(self, log_path: str = "logs/faults"):
        self.queue: List[FaultBusEvent] = []
        self.resolvers: Dict[str, FaultResolver] = {}
        self.memory_log: List[FaultBusEvent] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.trigger_policies: Dict[str, Callable[[FaultBusEvent], bool]] = {}
        self.log_path = log_path
        
        # Create log directory if it doesn't exist
        os.makedirs(log_path, exist_ok=True)
        
        # Initialize default resolvers
        self.register_resolver("thermal", ThermalFaultResolver())
        self.register_resolver("profit", ProfitFaultResolver())
        self.register_resolver("bitmap", BitmapFaultResolver())
        self.register_resolver("gpu", GPUFaultResolver())

    def register_resolver(self, fault_type: str, resolver: FaultResolver):
        """Register a resolver for a specific fault type."""
        self.resolvers[fault_type] = resolver

    def register_handler(self, event_type: str):
        def decorator(func):
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append(func)
            return func
        return decorator

    def push(self, event: FaultBusEvent):
        self.queue.append(event)

    async def dispatch(self, severity_threshold: float = 0.5):
        while self.queue:
            event = self.queue.pop(0)
            if event.severity >= severity_threshold:
                await self._handle_event(event)
                self.memory_log.append(event)

    async def _handle_event(self, event: FaultBusEvent):
        resolver = self.resolvers.get(event.type.name.lower())
        if resolver:
            try:
                resolver.handle_fault(event.type.value, event.severity, event.metadata)
                logging.info(f"Resolver {resolver.__class__.__name__} handled {event.type}")
            except Exception as e:
                logging.error(f"Resolver failure: {e}")
        else:
            logging.warning(f"No resolver for {event.type.name}")

    def get_fault_buckets(self) -> Dict[str, int]:
        bucket = {}
        for event in self.memory_log:
            bucket[event.type.value] = bucket.get(event.type.value, 0) + 1
        return bucket

    def register_policy(self, event_type: str, condition: Callable[[FaultBusEvent], bool]):
        self.trigger_policies[event_type] = condition

    def export_memory_log(self, file_path: Optional[str] = None) -> str:
        log = [event.__dict__ for event in self.memory_log]
        output = json.dumps(log, indent=2)
        if file_path:
            with open(file_path, "w") as f:
                f.write(output)
        return output

# Example usage of the FaultBus class
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Create a new instance of FaultBus
    fault_bus = FaultBus()

    # Register an event handler
    @fault_bus.register_handler("thermal_high")
    def handle_thermal_high(event):
        print(f"🔥 Event handled: {event}")

    # Push a new event
    fault_bus.push(FaultBusEvent(tick=1, module="thermal_monitor", type=FaultType.THERMAL_HIGH, severity=0.8, metadata={"temperature": 75.0}))

    # Dispatch events with a severity threshold
    asyncio.run(fault_bus.dispatch(severity_threshold=0.6))

    # Export the memory log to JSON
    print(fault_bus.export_memory_log()) 