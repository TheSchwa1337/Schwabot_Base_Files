"""
Fractal Command Dispatcher
=========================

Central command dispatcher for the three fractal systems:
- TFF (The Forever Fractals) - Infinite recursion and stability
- TPF (The Paradox Fractals) - Paradox resolution and stability correction
- TEF (The Echo Fractals) - Historical pattern memory and echo amplification

This module ensures proper command routing and mathematical synthesis between the systems.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import numpy as np
from datetime import datetime

from .fractal_core import ForeverFractalCore, FractalState
from .timing_manager import TimingManager, TimingState
from .recursive_profit import RecursiveMarketState, RecursiveProfitAllocationSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FractalSystemType(Enum):
    """Types of fractal systems"""
    TFF = "forever_fractals"      # The Forever Fractals
    TPF = "paradox_fractals"      # The Paradox Fractals  
    TEF = "echo_fractals"         # The Echo Fractals

class CommandType(Enum):
    """Types of fractal commands"""
    CALCULATE = "calculate"
    PREDICT = "predict"
    RESOLVE = "resolve"
    AMPLIFY = "amplify"
    STABILIZE = "stabilize"
    SYNCHRONIZE = "synchronize"

@dataclass
class FractalCommand:
    """Command structure for fractal operations"""
    command_id: str
    system_type: FractalSystemType
    command_type: CommandType
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    callback: Optional[Callable] = None
    result: Optional[Any] = None
    status: str = "pending"

@dataclass
class FractalSystemState:
    """State container for each fractal system"""
    system_type: FractalSystemType
    active: bool = True
    last_calculation: float = 0.0
    calculation_count: int = 0
    error_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class FractalCommandDispatcher:
    """Central dispatcher for fractal system commands"""
    
    def __init__(self):
        # Initialize fractal systems
        self.forever_fractal = ForeverFractalCore(decay_power=2.0, terms=50, dimension=3)
        self.timing_manager = TimingManager(
            recursion_coefficient=0.5,
            memory_decay_rate=0.1,
            phase_sync_rate=0.2
        )
        self.profit_system = RecursiveProfitAllocationSystem(max_memory_depth=1000)
        
        # Command queue and processing
        self.command_queue: List[FractalCommand] = []
        self.processing_lock = threading.Lock()
        self.system_states: Dict[FractalSystemType, FractalSystemState] = {
            FractalSystemType.TFF: FractalSystemState(FractalSystemType.TFF),
            FractalSystemType.TPF: FractalSystemState(FractalSystemType.TPF),
            FractalSystemType.TEF: FractalSystemState(FractalSystemType.TEF)
        }
        
        # Command handlers
        self.command_handlers = {
            (FractalSystemType.TFF, CommandType.CALCULATE): self._handle_tff_calculate,
            (FractalSystemType.TFF, CommandType.PREDICT): self._handle_tff_predict,
            (FractalSystemType.TFF, CommandType.STABILIZE): self._handle_tff_stabilize,
            
            (FractalSystemType.TPF, CommandType.RESOLVE): self._handle_tpf_resolve,
            (FractalSystemType.TPF, CommandType.CALCULATE): self._handle_tpf_calculate,
            (FractalSystemType.TPF, CommandType.STABILIZE): self._handle_tpf_stabilize,
            
            (FractalSystemType.TEF, CommandType.AMPLIFY): self._handle_tef_amplify,
            (FractalSystemType.TEF, CommandType.CALCULATE): self._handle_tef_calculate,
            (FractalSystemType.TEF, CommandType.PREDICT): self._handle_tef_predict,
            
            # Unified commands
            (FractalSystemType.TFF, CommandType.SYNCHRONIZE): self._handle_synchronize_all,
            (FractalSystemType.TPF, CommandType.SYNCHRONIZE): self._handle_synchronize_all,
            (FractalSystemType.TEF, CommandType.SYNCHRONIZE): self._handle_synchronize_all,
        }
        
        logger.info("Fractal Command Dispatcher initialized with all three systems")

    def dispatch_command(self, command: FractalCommand) -> str:
        """Dispatch a command to the appropriate fractal system"""
        with self.processing_lock:
            command.command_id = f"{command.system_type.value}_{command.command_type.value}_{int(time.time() * 1000)}"
            self.command_queue.append(command)
            logger.info(f"Dispatched command: {command.command_id}")
            return command.command_id

    def process_commands(self) -> List[FractalCommand]:
        """Process all pending commands"""
        processed_commands = []
        
        with self.processing_lock:
            # Sort by priority (higher priority first)
            self.command_queue.sort(key=lambda c: c.priority, reverse=True)
            
            for command in self.command_queue[:]:
                try:
                    self._execute_command(command)
                    processed_commands.append(command)
                    self.command_queue.remove(command)
                    
                    # Update system state
                    system_state = self.system_states[command.system_type]
                    system_state.last_calculation = time.time()
                    system_state.calculation_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing command {command.command_id}: {e}")
                    command.status = "error"
                    command.result = str(e)
                    
                    # Update error count
                    self.system_states[command.system_type].error_count += 1
        
        return processed_commands

    def _execute_command(self, command: FractalCommand):
        """Execute a single command"""
        handler_key = (command.system_type, command.command_type)
        
        if handler_key not in self.command_handlers:
            raise ValueError(f"No handler for {command.system_type.value} {command.command_type.value}")
        
        handler = self.command_handlers[handler_key]
        command.result = handler(command.parameters)
        command.status = "completed"
        
        # Execute callback if provided
        if command.callback:
            command.callback(command)

    # TFF (Forever Fractals) Handlers
    def _handle_tff_calculate(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle TFF calculation commands"""
        t = params.get('time', time.time())
        
        # Calculate forever fractal signal
        fractal_signal = self.forever_fractal.forever_fractal(t)
        
        # Generate fractal vector
        fractal_vector = self.forever_fractal.generate_fractal_vector(
            t, params.get('phase_shift', 0.0)
        )
        
        # Calculate stability metrics
        stability = self.timing_manager.calculate_forever_fractal(t)
        
        return {
            'fractal_signal': fractal_signal,
            'fractal_vector': fractal_vector,
            'stability_index': stability,
            'calculation_time': t
        }

    def _handle_tff_predict(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle TFF prediction commands"""
        current_state = params.get('market_state')
        horizon = params.get('horizon', 10)
        
        if not current_state:
            raise ValueError("Market state required for TFF prediction")
        
        # Use profit system's TFF prediction
        tff_movement = self.profit_system.predict_tff_movement(current_state, horizon)
        
        return {
            'predicted_movement': tff_movement,
            'prediction_horizon': horizon,
            'confidence': min(current_state.tff_stability_index, 1.0)
        }

    def _handle_tff_stabilize(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle TFF stabilization commands"""
        t = params.get('time', time.time())
        market_data = params.get('market_data', {})
        
        # Update timing state for stabilization
        self.timing_manager.update_timing_state(t, market_data)
        
        # Get stabilization metrics
        metrics = self.timing_manager.get_timing_metrics()
        
        return {
            'stabilization_factor': metrics['forever_fractal'],
            'phase_alignment': metrics['phase_alignment'],
            'recursion_depth': metrics['recursion_depth']
        }

    # TPF (Paradox Fractals) Handlers
    def _handle_tpf_resolve(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle TPF paradox resolution commands"""
        base_profit = params.get('base_profit', 0.0)
        tff_profit = params.get('tff_profit', 0.0)
        current_state = params.get('market_state')
        
        if not current_state:
            raise ValueError("Market state required for TPF resolution")
        
        # Use profit system's TPF resolution
        resolved_profit = self.profit_system.calculate_tpf_paradox_profit(
            base_profit, tff_profit, current_state
        )
        
        return {
            'resolved_profit': resolved_profit,
            'paradox_magnitude': abs(base_profit - tff_profit),
            'resolution_confidence': current_state.paradox_stability_score
        }

    def _handle_tpf_calculate(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle TPF calculation commands"""
        t = params.get('time', time.time())
        chaos_integral = params.get('chaos_integral', 0.0)
        
        # Calculate paradox fractal resolution
        paradox_resolution = self.timing_manager.calculate_paradox_fractal(t, chaos_integral)
        
        return {
            'paradox_resolution': paradox_resolution,
            'chaos_integral': chaos_integral,
            'calculation_time': t
        }

    def _handle_tpf_stabilize(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle TPF stabilization commands"""
        current_state = params.get('market_state')
        
        if not current_state:
            return {'stability_factor': 0.5}  # Default stability
        
        # Calculate stability based on paradox resolution count
        stability_factor = max(0.1, 1.0 - (current_state.paradox_resolution_count * 0.1))
        
        return {
            'stability_factor': stability_factor,
            'paradox_count': current_state.paradox_resolution_count,
            'stability_score': current_state.paradox_stability_score
        }

    # TEF (Echo Fractals) Handlers
    def _handle_tef_amplify(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle TEF echo amplification commands"""
        current_state = params.get('market_state')
        lookback_periods = params.get('lookback_periods', 50)
        
        if not current_state:
            raise ValueError("Market state required for TEF amplification")
        
        # Use profit system's TEF memory calculation
        memory_profit = self.profit_system.calculate_tef_memory_profit(
            current_state, lookback_periods
        )
        
        return {
            'amplified_signal': memory_profit,
            'echo_strength': current_state.historical_echo_strength,
            'memory_coherence': current_state.memory_coherence_level
        }

    def _handle_tef_calculate(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle TEF calculation commands"""
        t = params.get('time', time.time())
        
        # Calculate echo fractal memory preservation
        echo_memory = self.timing_manager.calculate_echo_fractal(t)
        
        return {
            'echo_memory': echo_memory,
            'calculation_time': t,
            'memory_strength': min(echo_memory, 1.0)
        }

    def _handle_tef_predict(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle TEF prediction commands"""
        current_state = params.get('market_state')
        horizon = params.get('horizon', 10)
        
        if not current_state:
            raise ValueError("Market state required for TEF prediction")
        
        # Use profit system's TEF prediction
        tef_movement = self.profit_system.predict_tef_movement(current_state, horizon)
        
        return {
            'predicted_movement': tef_movement,
            'prediction_horizon': horizon,
            'confidence': current_state.memory_coherence_level
        }

    def _handle_synchronize_all(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle synchronization across all fractal systems"""
        t = params.get('time', time.time())
        market_data = params.get('market_data', {})
        current_state = params.get('market_state')
        
        # Update timing manager
        self.timing_manager.update_timing_state(t, market_data)
        
        # Get unified metrics
        timing_metrics = self.timing_manager.get_timing_metrics()
        
        # Calculate unified predictions if market state provided
        unified_prediction = None
        if current_state:
            prediction_data = self.profit_system.calculate_predictive_movement_profit(current_state)
            unified_prediction = prediction_data['unified_movement']
        
        return {
            'tff_metrics': {
                'forever_fractal': timing_metrics['forever_fractal'],
                'recursion_depth': timing_metrics['recursion_depth']
            },
            'tpf_metrics': {
                'paradox_resolution': timing_metrics['paradox_resolution'],
                'phase_alignment': timing_metrics['phase_alignment']
            },
            'tef_metrics': {
                'echo_memory': timing_metrics['echo_memory'],
                'memory_weight': timing_metrics['memory_weight']
            },
            'unified_prediction': unified_prediction,
            'synchronization_time': t
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all fractal systems"""
        return {
            'systems': {
                system_type.value: {
                    'active': state.active,
                    'last_calculation': state.last_calculation,
                    'calculation_count': state.calculation_count,
                    'error_count': state.error_count,
                    'error_rate': state.error_count / max(state.calculation_count, 1)
                }
                for system_type, state in self.system_states.items()
            },
            'queue_size': len(self.command_queue),
            'total_commands_processed': sum(state.calculation_count for state in self.system_states.values())
        }

    def create_tff_command(self, command_type: CommandType, **params) -> FractalCommand:
        """Create a TFF command"""
        return FractalCommand(
            command_id="",
            system_type=FractalSystemType.TFF,
            command_type=command_type,
            parameters=params
        )

    def create_tpf_command(self, command_type: CommandType, **params) -> FractalCommand:
        """Create a TPF command"""
        return FractalCommand(
            command_id="",
            system_type=FractalSystemType.TPF,
            command_type=command_type,
            parameters=params
        )

    def create_tef_command(self, command_type: CommandType, **params) -> FractalCommand:
        """Create a TEF command"""
        return FractalCommand(
            command_id="",
            system_type=FractalSystemType.TEF,
            command_type=command_type,
            parameters=params
        )

# Example usage and testing
if __name__ == "__main__":
    dispatcher = FractalCommandDispatcher()
    
    # Test TFF calculation
    tff_cmd = dispatcher.create_tff_command(
        CommandType.CALCULATE,
        time=time.time(),
        phase_shift=0.1
    )
    dispatcher.dispatch_command(tff_cmd)
    
    # Test TPF resolution
    from .recursive_profit import RecursiveMarketState
    test_state = RecursiveMarketState(
        timestamp=datetime.now(),
        price=100.0,
        volume=1000.0,
        tff_stability_index=0.8,
        paradox_stability_score=0.7,
        memory_coherence_level=0.6
    )
    
    tpf_cmd = dispatcher.create_tpf_command(
        CommandType.RESOLVE,
        base_profit=10.0,
        tff_profit=12.0,
        market_state=test_state
    )
    dispatcher.dispatch_command(tpf_cmd)
    
    # Process commands
    processed = dispatcher.process_commands()
    
    print(f"Processed {len(processed)} commands")
    for cmd in processed:
        print(f"Command {cmd.command_id}: {cmd.status}")
        if cmd.result:
            print(f"Result: {cmd.result}")
    
    # Show system status
    status = dispatcher.get_system_status()
    print(f"System Status: {status}") 