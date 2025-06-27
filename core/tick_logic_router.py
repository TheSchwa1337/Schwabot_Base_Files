"""
Tick Logic Router - Command Cohesion for Schwabot Data Feed Management System

This module provides unified command routing and state management across all system
components, ensuring proper flow of entropy, bit, and phase states for cohesive
mathematical operations.

Mathematical Routing Logic:
- State synchronization: S(t) = Σ component_states × sync_weights
- Command routing: Route(cmd) = argmax(component_readiness × cmd_affinity)
- Entropy flow routing: H_route = entropy_gradient × flow_direction
- Bit phase routing: φ_route = phase_state × component_compatibility
- Error propagation: Error(t) = Error(t-1) × decay_factor + new_errors
"""

import logging
import asyncio
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import hashlib

from .math_core import (
    MathematicalCore, SystemComponent, MathematicalState, 
    get_math_core, ComponentState
)

logger = logging.getLogger(__name__)

class CommandType(Enum):
    """Command types for routing."""
    PROFIT_CALCULATION = "profit_calculation"
    STRATEGY_MAPPING = "strategy_mapping"
    ENTROPY_ANALYSIS = "entropy_analysis"
    BIT_PHASE_OPERATION = "bit_phase_operation"
    FRACTAL_RECURSION = "fractal_recursion"
    HASH_REGISTRATION = "hash_registration"
    FALLBACK_TRIGGER = "fallback_trigger"
    ECHO_MEMORY = "echo_memory"
    ALTITUDE_ADJUSTMENT = "altitude_adjustment"
    RING_CYCLING = "ring_cycling"

class RoutingPriority(Enum):
    """Routing priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

class CommandStatus(Enum):
    """Command execution status."""
    PENDING = "pending"
    ROUTING = "routing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class TickCommand:
    """Command structure for tick logic routing."""
    command_id: str
    command_type: CommandType
    priority: RoutingPriority
    source_component: SystemComponent
    target_components: List[SystemComponent]
    data: Dict[str, Any]
    mathematical_operation: str
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: CommandStatus = CommandStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class RoutingDecision:
    """Routing decision structure."""
    command_id: str
    selected_component: SystemComponent
    confidence_score: float
    routing_reason: str
    alternative_components: List[SystemComponent]
    expected_execution_time: float
    resource_requirements: Dict[str, Any]

@dataclass
class SystemState:
    """Global system state for routing decisions."""
    entropy_levels: Dict[SystemComponent, float]
    bit_phase_states: Dict[SystemComponent, float]
    component_loads: Dict[SystemComponent, float]
    error_rates: Dict[SystemComponent, float]
    last_update: float
    health_scores: Dict[SystemComponent, float]

class TickLogicRouter:
    """Unified command routing and state management system."""
    
    def __init__(self, math_core: Optional[MathematicalCore] = None):
        self.math_core = math_core or get_math_core()
        self.command_queue = deque()
        self.executing_commands = {}
        self.completed_commands = deque(maxlen=1000)  # Keep last 1000 commands
        self.failed_commands = deque(maxlen=100)  # Keep last 100 failed commands
        
        # Component routing tables
        self.component_capabilities = {}
        self.component_load_factors = {}
        self.routing_history = defaultdict(list)
        
        # System state tracking
        self.system_state = SystemState(
            entropy_levels={},
            bit_phase_states={},
            component_loads={},
            error_rates={},
            last_update=time.time(),
            health_scores={}
        )
        
        # Threading and async management
        self.router_lock = threading.RLock()
        self.event_loop = None
        self.router_task = None
        self.is_running = False
        
        # Performance metrics
        self.routing_metrics = {
            'commands_processed': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'average_execution_time': 0.0,
            'last_reset': time.time()
        }
        
        # Initialize component capabilities
        self._initialize_component_capabilities()
        
        logger.info("Tick Logic Router initialized with command cohesion")

    def _initialize_component_capabilities(self):
        """Initialize component capabilities for routing decisions."""
        
        self.component_capabilities = {
            SystemComponent.PROFIT_CYCLE_ALLOCATOR: {
                'command_types': [CommandType.PROFIT_CALCULATION],
                'mathematical_operations': ['profit_tier_navigation', 'roi_calculation'],
                'max_concurrent_commands': 5,
                'average_execution_time': 0.1,
                'reliability_score': 0.95
            },
            
            SystemComponent.STRATEGY_MAPPER: {
                'command_types': [CommandType.STRATEGY_MAPPING, CommandType.HASH_REGISTRATION],
                'mathematical_operations': ['hash_strategy_mapping', 'matrix_weighting'],
                'max_concurrent_commands': 3,
                'average_execution_time': 0.2,
                'reliability_score': 0.90
            },
            
            SystemComponent.ENTROPY_LANE_BUILDER: {
                'command_types': [CommandType.ENTROPY_ANALYSIS],
                'mathematical_operations': ['entropy_flow_detection', 'stream_analysis'],
                'max_concurrent_commands': 4,
                'average_execution_time': 0.15,
                'reliability_score': 0.88
            },
            
            SystemComponent.BIT_PHASE_ENGINE: {
                'command_types': [CommandType.BIT_PHASE_OPERATION],
                'mathematical_operations': ['bit_phase_collapse', 'phase_alignment'],
                'max_concurrent_commands': 6,
                'average_execution_time': 0.05,
                'reliability_score': 0.92
            },
            
            SystemComponent.FRACTAL_CORE: {
                'command_types': [CommandType.FRACTAL_RECURSION],
                'mathematical_operations': ['fractal_recursion', 'triplet_collapse'],
                'max_concurrent_commands': 2,
                'average_execution_time': 0.3,
                'reliability_score': 0.85
            },
            
            SystemComponent.HASH_REGISTRY: {
                'command_types': [CommandType.HASH_REGISTRATION, CommandType.ECHO_MEMORY],
                'mathematical_operations': ['hash_storage', 'echo_retrieval'],
                'max_concurrent_commands': 8,
                'average_execution_time': 0.08,
                'reliability_score': 0.94
            },
            
            SystemComponent.FALLBACK_VECTOR_GENERATOR: {
                'command_types': [CommandType.FALLBACK_TRIGGER],
                'mathematical_operations': ['fallback_calculation', 'vector_generation'],
                'max_concurrent_commands': 3,
                'average_execution_time': 0.25,
                'reliability_score': 0.87
            },
            
            SystemComponent.ECHO_TRIGGER_MANAGER: {
                'command_types': [CommandType.ECHO_MEMORY],
                'mathematical_operations': ['echo_trigger', 'memory_correlation'],
                'max_concurrent_commands': 4,
                'average_execution_time': 0.12,
                'reliability_score': 0.89
            },
            
            SystemComponent.ALTITUDE_GENERATOR: {
                'command_types': [CommandType.ALTITUDE_ADJUSTMENT, CommandType.RING_CYCLING],
                'mathematical_operations': ['altitude_calculation', 'volume_spike_detection'],
                'max_concurrent_commands': 5,
                'average_execution_time': 0.1,
                'reliability_score': 0.91
            }
        }
        
        # Initialize load factors
        for component in SystemComponent:
            self.component_load_factors[component] = 0.0

    async def start_router(self):
        """Start the async routing system."""
        if self.is_running:
            logger.warning("Router is already running")
            return
        
        self.is_running = True
        self.event_loop = asyncio.get_event_loop()
        self.router_task = asyncio.create_task(self._router_main_loop())
        logger.info("Tick Logic Router started")

    async def stop_router(self):
        """Stop the routing system."""
        self.is_running = False
        if self.router_task:
            self.router_task.cancel()
            try:
                await self.router_task
            except asyncio.CancelledError:
                pass
        logger.info("Tick Logic Router stopped")

    async def _router_main_loop(self):
        """Main routing loop for processing commands."""
        while self.is_running:
            try:
                await self._process_command_queue()
                await self._update_system_state()
                await self._cleanup_completed_commands()
                await asyncio.sleep(0.01)  # 10ms tick rate
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in router main loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_command_queue(self):
        """Process commands in the queue."""
        if not self.command_queue:
            return
        
        with self.router_lock:
            # Process up to 10 commands per tick
            commands_to_process = []
            for _ in range(min(10, len(self.command_queue))):
                if self.command_queue:
                    commands_to_process.append(self.command_queue.popleft())
        
        for command in commands_to_process:
            try:
                await self._route_command(command)
            except Exception as e:
                logger.error(f"Error routing command {command.command_id}: {e}")
                command.status = CommandStatus.FAILED
                command.error = str(e)
                self.failed_commands.append(command)

    async def _route_command(self, command: TickCommand):
        """Route a command to the appropriate component."""
        try:
            # Update command status
            command.status = CommandStatus.ROUTING
            command.started_at = time.time()
            
            # Make routing decision
            routing_decision = self._make_routing_decision(command)
            
            if not routing_decision:
                command.status = CommandStatus.FAILED
                command.error = "No suitable component found for routing"
                self.failed_commands.append(command)
                return
            
            # Execute command
            command.status = CommandStatus.EXECUTING
            self.executing_commands[command.command_id] = command
            
            # Simulate command execution (in real implementation, this would call actual components)
            result = await self._execute_command(command, routing_decision)
            
            # Update command with result
            command.result = result
            command.completed_at = time.time()
            command.status = CommandStatus.COMPLETED
            
            # Remove from executing and add to completed
            if command.command_id in self.executing_commands:
                del self.executing_commands[command.command_id]
            self.completed_commands.append(command)
            
            # Update metrics
            self.routing_metrics['commands_processed'] += 1
            self.routing_metrics['successful_routes'] += 1
            
            # Update routing history
            self.routing_history[routing_decision.selected_component].append({
                'command_id': command.command_id,
                'execution_time': command.completed_at - command.started_at,
                'success': True,
                'timestamp': time.time()
            })
            
        except Exception as e:
            command.status = CommandStatus.FAILED
            command.error = str(e)
            command.completed_at = time.time()
            
            if command.command_id in self.executing_commands:
                del self.executing_commands[command.command_id]
            self.failed_commands.append(command)
            
            self.routing_metrics['failed_routes'] += 1
            logger.error(f"Command execution failed: {e}")

    def _make_routing_decision(self, command: TickCommand) -> Optional[RoutingDecision]:
        """Make intelligent routing decision for a command."""
        try:
            # Find capable components
            capable_components = []
            
            for component, capabilities in self.component_capabilities.items():
                if command.command_type in capabilities['command_types']:
                    
                    # Calculate routing score
                    load_factor = self.component_load_factors.get(component, 0.0)
                    reliability = capabilities['reliability_score']
                    current_load = len([cmd for cmd in self.executing_commands.values() 
                                      if component in cmd.target_components])
                    max_concurrent = capabilities['max_concurrent_commands']
                    
                    # Avoid overloaded components
                    if current_load >= max_concurrent:
                        continue
                    
                    # Calculate confidence score
                    confidence_score = (
                        reliability * 0.4 +
                        (1.0 - load_factor) * 0.3 +
                        (1.0 - current_load / max_concurrent) * 0.3
                    )
                    
                    capable_components.append((component, confidence_score))
            
            if not capable_components:
                return None
            
            # Select best component
            capable_components.sort(key=lambda x: x[1], reverse=True)
            selected_component, confidence = capable_components[0]
            alternatives = [comp for comp, _ in capable_components[1:]]
            
            # Create routing decision
            routing_decision = RoutingDecision(
                command_id=command.command_id,
                selected_component=selected_component,
                confidence_score=confidence,
                routing_reason=f"Best match with confidence {confidence:.3f}",
                alternative_components=alternatives,
                expected_execution_time=self.component_capabilities[selected_component]['average_execution_time'],
                resource_requirements={'cpu': 0.1, 'memory': 0.05}
            )
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Error making routing decision: {e}")
            return None

    async def _execute_command(self, command: TickCommand, routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute command on selected component."""
        try:
            # Simulate component execution
            execution_time = routing_decision.expected_execution_time
            await asyncio.sleep(execution_time)
            
            # Generate result based on command type
            if command.command_type == CommandType.PROFIT_CALCULATION:
                result = await self._execute_profit_calculation(command)
            elif command.command_type == CommandType.STRATEGY_MAPPING:
                result = await self._execute_strategy_mapping(command)
            elif command.command_type == CommandType.ENTROPY_ANALYSIS:
                result = await self._execute_entropy_analysis(command)
            elif command.command_type == CommandType.BIT_PHASE_OPERATION:
                result = await self._execute_bit_phase_operation(command)
            elif command.command_type == CommandType.FRACTAL_RECURSION:
                result = await self._execute_fractal_recursion(command)
            else:
                result = {'status': 'executed', 'component': routing_decision.selected_component.value}
            
            result.update({
                'routing_decision': routing_decision.selected_component.value,
                'execution_time': execution_time,
                'confidence': routing_decision.confidence_score
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {'status': 'error', 'error': str(e)}

    async def _execute_profit_calculation(self, command: TickCommand) -> Dict[str, Any]:
        """Execute profit calculation command."""
        data = command.data
        base_profit = data.get('base_profit', 1000.0)
        tier_weights = data.get('tier_weights', [0.1, 0.3, 0.5])
        roi_rates = data.get('roi_rates', [0.02, 0.05, 0.1])
        
        result_profit = self.math_core.calculate_profit_tier_navigation(
            base_profit, tier_weights, roi_rates
        )
        
        return {
            'status': 'completed',
            'operation': 'profit_calculation',
            'result_profit': result_profit,
            'input_data': data
        }

    async def _execute_strategy_mapping(self, command: TickCommand) -> Dict[str, Any]:
        """Execute strategy mapping command."""
        data = command.data
        hash1 = data.get('hash1', '')
        hash2 = data.get('hash2', '')
        
        similarity = self.math_core.calculate_hash_strategy_similarity(hash1, hash2)
        
        return {
            'status': 'completed',
            'operation': 'strategy_mapping',
            'similarity_score': similarity,
            'input_data': data
        }

    async def _execute_entropy_analysis(self, command: TickCommand) -> Dict[str, Any]:
        """Execute entropy analysis command."""
        data = command.data
        data_stream = data.get('data_stream', [1.0, 2.0, 3.0])
        
        entropy = self.math_core.calculate_entropy_flow(data_stream)
        
        return {
            'status': 'completed',
            'operation': 'entropy_analysis',
            'entropy_value': entropy,
            'input_data': data
        }

    async def _execute_bit_phase_operation(self, command: TickCommand) -> Dict[str, Any]:
        """Execute bit phase operation command."""
        data = command.data
        amplitudes = data.get('amplitudes', [1.0, 0.8, 0.6])
        frequencies = data.get('frequencies', [1.0, 1.618, 3.14])
        time_point = data.get('time_point', time.time())
        
        phase_value = self.math_core.calculate_bit_phase_collapse(amplitudes, frequencies, time_point)
        
        return {
            'status': 'completed',
            'operation': 'bit_phase_operation',
            'phase_value': phase_value,
            'input_data': data
        }

    async def _execute_fractal_recursion(self, command: TickCommand) -> Dict[str, Any]:
        """Execute fractal recursion command."""
        data = command.data
        previous_value = data.get('previous_value', 1.0)
        tier_weights = data.get('tier_weights', [0.1, 0.3, 0.5])
        bit_phases = data.get('bit_phases', [0.2, 0.4, 0.6])
        
        fractal_value = self.math_core.calculate_fractal_recursion(previous_value, tier_weights, bit_phases)
        
        return {
            'status': 'completed',
            'operation': 'fractal_recursion',
            'fractal_value': fractal_value,
            'input_data': data
        }

    async def _update_system_state(self):
        """Update global system state."""
        try:
            current_time = time.time()
            
            # Update component loads
            for component in SystemComponent:
                current_load = len([cmd for cmd in self.executing_commands.values() 
                                  if component in cmd.target_components])
                max_load = self.component_capabilities.get(component, {}).get('max_concurrent_commands', 1)
                self.component_load_factors[component] = current_load / max_load
            
            # Update system state
            self.system_state.component_loads = self.component_load_factors.copy()
            self.system_state.last_update = current_time
            
            # Update health scores from math core
            for component in SystemComponent:
                component_state = self.math_core.get_component_state(component)
                if component_state:
                    self.system_state.health_scores[component] = component_state.health_score
                else:
                    self.system_state.health_scores[component] = 0.5  # Default health
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")

    async def _cleanup_completed_commands(self):
        """Clean up old completed commands."""
        try:
            current_time = time.time()
            cleanup_threshold = 300  # 5 minutes
            
            # Clean up executing commands that may have timed out
            timed_out_commands = []
            for command_id, command in self.executing_commands.items():
                if current_time - command.started_at > cleanup_threshold:
                    timed_out_commands.append(command_id)
            
            for command_id in timed_out_commands:
                command = self.executing_commands.pop(command_id)
                command.status = CommandStatus.TIMEOUT
                command.completed_at = current_time
                self.failed_commands.append(command)
                
        except Exception as e:
            logger.error(f"Error cleaning up commands: {e}")

    # Public API Methods
    def submit_command(self, command_type: CommandType, data: Dict[str, Any],
                      source_component: SystemComponent,
                      target_components: Optional[List[SystemComponent]] = None,
                      priority: RoutingPriority = RoutingPriority.MEDIUM,
                      mathematical_operation: str = "") -> str:
        """Submit a command for routing."""
        try:
            command_id = hashlib.sha256(
                f"{command_type.value}_{time.time()}_{source_component.value}".encode()
            ).hexdigest()[:16]
            
            if target_components is None:
                target_components = []
            
            command = TickCommand(
                command_id=command_id,
                command_type=command_type,
                priority=priority,
                source_component=source_component,
                target_components=target_components,
                data=data,
                mathematical_operation=mathematical_operation
            )
            
            with self.router_lock:
                if priority == RoutingPriority.CRITICAL:
                    self.command_queue.appendleft(command)  # High priority to front
                else:
                    self.command_queue.append(command)
            
            logger.debug(f"Command {command_id} submitted for routing")
            return command_id
            
        except Exception as e:
            logger.error(f"Error submitting command: {e}")
            return ""

    def get_command_status(self, command_id: str) -> Optional[CommandStatus]:
        """Get status of a command."""
        # Check executing commands
        if command_id in self.executing_commands:
            return self.executing_commands[command_id].status
        
        # Check completed commands
        for command in self.completed_commands:
            if command.command_id == command_id:
                return command.status
        
        # Check failed commands
        for command in self.failed_commands:
            if command.command_id == command_id:
                return command.status
        
        # Check pending commands
        for command in self.command_queue:
            if command.command_id == command_id:
                return command.status
        
        return None

    def get_command_result(self, command_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed command."""
        for command in self.completed_commands:
            if command.command_id == command_id:
                return command.result
        return None

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        return {
            'routing_metrics': self.routing_metrics.copy(),
            'queue_length': len(self.command_queue),
            'executing_commands': len(self.executing_commands),
            'completed_commands': len(self.completed_commands),
            'failed_commands': len(self.failed_commands),
            'component_loads': self.component_load_factors.copy(),
            'system_health': self.system_state.health_scores.copy(),
            'timestamp': time.time()
        }

    def route_entropy_state(self, entropy_data: Dict[str, Any]) -> str:
        """Route entropy state across components."""
        return self.submit_command(
            command_type=CommandType.ENTROPY_ANALYSIS,
            data=entropy_data,
            source_component=SystemComponent.ENTROPY_LANE_BUILDER,
            priority=RoutingPriority.HIGH,
            mathematical_operation="entropy_flow_detection"
        )

    def route_bit_phase_state(self, bit_phase_data: Dict[str, Any]) -> str:
        """Route bit phase state across components."""
        return self.submit_command(
            command_type=CommandType.BIT_PHASE_OPERATION,
            data=bit_phase_data,
            source_component=SystemComponent.BIT_PHASE_ENGINE,
            priority=RoutingPriority.HIGH,
            mathematical_operation="bit_phase_collapse"
        )

    def route_fractal_state(self, fractal_data: Dict[str, Any]) -> str:
        """Route fractal state across components."""
        return self.submit_command(
            command_type=CommandType.FRACTAL_RECURSION,
            data=fractal_data,
            source_component=SystemComponent.FRACTAL_CORE,
            priority=RoutingPriority.MEDIUM,
            mathematical_operation="fractal_recursion"
        )


# Global router instance
_tick_router = None

def get_tick_router() -> TickLogicRouter:
    """Get global tick router instance."""
    global _tick_router
    if _tick_router is None:
        _tick_router = TickLogicRouter()
    return _tick_router

async def start_global_router():
    """Start the global tick router."""
    router = get_tick_router()
    await router.start_router()

async def stop_global_router():
    """Stop the global tick router."""
    router = get_tick_router()
    await router.stop_router()

def main():
    """Main function for testing tick logic router."""
    async def test_router():
        print("🔄 Tick Logic Router Test")
        print("-" * 50)
        
        router = TickLogicRouter()
        await router.start_router()
        
        try:
            # Submit test commands
            profit_cmd = router.submit_command(
                CommandType.PROFIT_CALCULATION,
                {'base_profit': 1000.0, 'tier_weights': [0.1, 0.3], 'roi_rates': [0.02, 0.05]},
                SystemComponent.PROFIT_CYCLE_ALLOCATOR,
                priority=RoutingPriority.HIGH
            )
            
            entropy_cmd = router.submit_command(
                CommandType.ENTROPY_ANALYSIS,
                {'data_stream': [1.0, 2.0, 3.0, 2.0, 1.0]},
                SystemComponent.ENTROPY_LANE_BUILDER
            )
            
            print(f"📊 Submitted commands: {profit_cmd[:8]}, {entropy_cmd[:8]}")
            
            # Wait for execution
            await asyncio.sleep(1.0)
            
            # Check results
            profit_result = router.get_command_result(profit_cmd)
            entropy_result = router.get_command_result(entropy_cmd)
            
            print(f"💰 Profit Result: {profit_result}")
            print(f"🌀 Entropy Result: {entropy_result}")
            
            # Get system metrics
            metrics = router.get_system_metrics()
            print(f"📈 System Metrics: {metrics}")
            
        finally:
            await router.stop_router()
        
        print("\n✅ Tick Logic Router Test Complete")
    
    asyncio.run(test_router())

if __name__ == "__main__":
    main() 