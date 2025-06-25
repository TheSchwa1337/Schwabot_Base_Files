from core.unified_math_system import unified_math
import math
# #!/usr/bin/env python3
"""
Profit Bridge Orchestrator - Schwabot Core Component

Manages the orchestration of profit-related operations across different
trading components and ensures proper coordination between profit tracking,
allocation, and execution systems.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BridgeState(Enum):
    """Bridge orchestration states."""
INITIALIZING = "initializing"
ACTIVE = "active"
PAUSED = "paused"
ERROR = "error"
SHUTDOWN = "shutdown"


@dataclass
class ProfitBridgeConfig:
    """Configuration for profit bridge orchestration."""
max_concurrent_operations: int = 10
operation_timeout: float = 30.0
retry_attempts: int = 3
health_check_interval: float = 60.0
enable_profit_tracking: bool = True
enable_allocation_optimization: bool = True
enable_execution_coordination: bool = True


@dataclass
class BridgeOperation:
    """Represents a profit bridge operation."""
operation_id: str
operation_type: str
status: str
start_time: float
end_time: Optional[float] = None
result: Optional[Dict[str, Any]] = None
error: Optional[str] = None
metadata: Dict[str, Any] = field(default_factory=dict)


class ProfitBridgeOrchestrator:
    """
Orchestrates profit-related operations across the Schwabot system.

Responsibilities:
- Coordinate profit tracking operations
- Manage profit allocation strategies
- Orchestrate execution coordination
- Monitor bridge health and performance
- Handle error recovery and fallback logic
"""

    def __init__(self, config: Optional[ProfitBridgeConfig] = None):
        """Initialize the profit bridge orchestrator."""
self.config = config or ProfitBridgeConfig()
        self.state = BridgeState.INITIALIZING

        # Operation tracking
self.active_operations: Dict[str, BridgeOperation] = {}
self.operation_history: List[BridgeOperation] = []
self.operation_counter = 0

        # Performance metrics
self.total_operations = 0
self.successful_operations = 0
self.failed_operations = 0
self.last_health_check = time.time()

        # Component references (will be set by core loop manager)
        self.profit_tracker = None
self.allocation_engine = None
self.execution_coordinator = None

logger.info("ProfitBridgeOrchestrator initialized")
        self.state = BridgeState.ACTIVE

    def register_profit_tracker(self, profit_tracker) -> None:
        """Register profit tracker component."""
self.profit_tracker = profit_tracker
logger.info("Profit tracker registered")

    def register_allocation_engine(self, allocation_engine) -> None:
        """Register allocation engine component."""
self.allocation_engine = allocation_engine
logger.info("Allocation engine registered")

    def register_execution_coordinator(self, execution_coordinator) -> None:
        """Register execution coordinator component."""
self.execution_coordinator = execution_coordinator
logger.info("Execution coordinator registered")

    def start_profit_tracking_operation(self, market_data: Dict[str, Any]) -> str:
        """Start a profit tracking operation."""
        if not self.config.enable_profit_tracking:
logger.warning("Profit tracking is disabled")
            return ""

operation_id = f"profit_track_{self.operation_counter}"
self.operation_counter += 1

operation = BridgeOperation(
            operation_id=operation_id,
operation_type="profit_tracking",
status="running",
start_time=time.time(),
            metadata={"market_data": market_data}


self.active_operations[operation_id] = operation

        try:
            if self.profit_tracker:
result = self.profit_tracker.track_profit(market_data)
                operation.result = result
operation.status = "completed"
operation.end_time = time.time()
                self.successful_operations += 1
            else:
operation.status = "failed"
operation.error = "Profit tracker not available"
operation.end_time = time.time()
                self.failed_operations += 1

        except Exception as e:
operation.status = "failed"
operation.error = str(e)
            operation.end_time = time.time()
            self.failed_operations += 1
logger.error(f"Profit tracking operation failed: {e}")

self.total_operations += 1
self.operation_history.append(operation)

        # Clean up completed operations
        if operation.status in ["completed", "failed"]:
            del self.active_operations[operation_id]

        return operation_id

    def start_allocation_optimization(self, portfolio_state: Dict[str, Any]) -> str:
        """Start an allocation optimization operation."""
        if not self.config.enable_allocation_optimization:
logger.warning("Allocation optimization is disabled")
            return ""

operation_id = f"alloc_opt_{self.operation_counter}"
self.operation_counter += 1

operation = BridgeOperation(
            operation_id=operation_id,
operation_type="allocation_optimization",
status="running",
start_time=time.time(),
            metadata={"portfolio_state": portfolio_state}


self.active_operations[operation_id] = operation

        try:
            if self.allocation_engine:
result = self.allocation_engine.optimize_allocation(portfolio_state)
                operation.result = result
operation.status = "completed"
operation.end_time = time.time()
                self.successful_operations += 1
            else:
operation.status = "failed"
operation.error = "Allocation engine not available"
operation.end_time = time.time()
                self.failed_operations += 1

        except Exception as e:
operation.status = "failed"
operation.error = str(e)
            operation.end_time = time.time()
            self.failed_operations += 1
logger.error(f"Allocation optimization failed: {e}")

self.total_operations += 1
self.operation_history.append(operation)

        # Clean up completed operations
        if operation.status in ["completed", "failed"]:
            del self.active_operations[operation_id]

        return operation_id

    def start_execution_coordination(self, trade_signals: Dict[str, Any]) -> str:
        """Start an execution coordination operation."""
        if not self.config.enable_execution_coordination:
logger.warning("Execution coordination is disabled")
            return ""

operation_id = f"exec_coord_{self.operation_counter}"
self.operation_counter += 1

operation = BridgeOperation(
            operation_id=operation_id,
operation_type="execution_coordination",
status="running",
start_time=time.time(),
            metadata={"trade_signals": trade_signals}


self.active_operations[operation_id] = operation

        try:
            if self.execution_coordinator:
result = self.execution_coordinator.coordinate_execution(trade_signals)
                operation.result = result
operation.status = "completed"
operation.end_time = time.time()
                self.successful_operations += 1
            else:
operation.status = "failed"
operation.error = "Execution coordinator not available"
operation.end_time = time.time()
                self.failed_operations += 1

        except Exception as e:
operation.status = "failed"
operation.error = str(e)
            operation.end_time = time.time()
            self.failed_operations += 1
logger.error(f"Execution coordination failed: {e}")

self.total_operations += 1
self.operation_history.append(operation)

        # Clean up completed operations
        if operation.status in ["completed", "failed"]:
            del self.active_operations[operation_id]

        return operation_id

    def get_operation_status(self, operation_id: str) -> Optional[BridgeOperation]:
        """Get the status of a specific operation."""
        # Check active operations first
        if operation_id in self.active_operations:
            return self.active_operations[operation_id]

        # Check operation history
        for operation in reversed(self.operation_history):
            if operation.operation_id == operation_id:
                return operation

        return None

    def get_bridge_health(self) -> Dict[str, Any]:
        """Get the health status of the bridge orchestrator."""
current_time = time.time()

        # Check if health check is needed
        if current_time - self.last_health_check > self.config.health_check_interval:
self._perform_health_check()
            self.last_health_check = current_time

success_rate = (
            self.successful_operations / unified_math.max(self.total_operations, 1)


        return {
"state": self.state.value,
"active_operations": len(self.active_operations),
            "total_operations": self.total_operations,
"successful_operations": self.successful_operations,
"failed_operations": self.failed_operations,
"success_rate": success_rate,
"components_available": {
"profit_tracker": self.profit_tracker is not None,
"allocation_engine": self.allocation_engine is not None,
"execution_coordinator": self.execution_coordinator is not None
},
"last_health_check": self.last_health_check,
"config": {
"max_concurrent_operations": self.config.max_concurrent_operations,
"enable_profit_tracking": self.config.enable_profit_tracking,
"enable_allocation_optimization": self.config.enable_allocation_optimization,
"enable_execution_coordination": self.config.enable_execution_coordination
}
}

    def _perform_health_check(self) -> None:
        """Perform internal health check."""
        try:
            # Check if too many operations are active
            if len(self.active_operations) > self.config.max_concurrent_operations:
                logger.warning(f"Too many active operations: {len(self.active_operations)}")
                self.state = BridgeState.PAUSED

            # Check success rate
            if self.total_operations > 10:
success_rate = self.successful_operations / self.total_operations
                if success_rate < 0.5:  # Less than 50% success rate
logger.warning(f"Low success rate: {success_rate:.2%}")
                    self.state = BridgeState.ERROR

            # Check component availability
            if not any([self.profit_tracker, self.allocation_engine, self.execution_coordinator]):
                logger.warning("No components available")
                self.state = BridgeState.ERROR

            # If all checks pass, set to active
            if self.state != BridgeState.ERROR:
self.state = BridgeState.ACTIVE

        except Exception as e:
logger.error(f"Health check failed: {e}")
            self.state = BridgeState.ERROR

    def pause_bridge(self) -> None:
        """Pause bridge operations."""
self.state = BridgeState.PAUSED
logger.info("Bridge orchestrator paused")

    def resume_bridge(self) -> None:
        """Resume bridge operations."""
self.state = BridgeState.ACTIVE
logger.info("Bridge orchestrator resumed")

    def shutdown(self) -> None:
        """Shutdown the bridge orchestrator."""
self.state = BridgeState.SHUTDOWN

        # Cancel all active operations
        for operation in self.active_operations.values():
            operation.status = "cancelled"
operation.end_time = time.time()
            operation.error = "Bridge shutdown"

self.active_operations.clear()
        logger.info("Bridge orchestrator shutdown complete")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the bridge orchestrator."""
recent_operations = self.operation_history[-100:] if self.operation_history else []

operation_types = {}
        for op in recent_operations:
op_type = op.operation_type
            if op_type not in operation_types:
operation_types[op_type] = {"total": 0, "successful": 0, "failed": 0}

operation_types[op_type]["total"] += 1
            if op.status == "completed":
operation_types[op_type]["successful"] += 1
            else:
operation_types[op_type]["failed"] += 1

        return {
"total_operations": self.total_operations,
"successful_operations": self.successful_operations,
"failed_operations": self.failed_operations,
"success_rate": self.successful_operations / unified_math.max(self.total_operations, 1),
            "operation_types": operation_types,
"active_operations": len(self.active_operations),
            "bridge_state": self.state.value,
"last_health_check": self.last_health_check
}


def create_profit_bridge_orchestrator(config: Optional[ProfitBridgeConfig] = None) -> ProfitBridgeOrchestrator:
    """Factory function to create a profit bridge orchestrator."""
    return ProfitBridgeOrchestrator(config)
