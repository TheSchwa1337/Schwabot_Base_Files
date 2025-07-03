import asyncio
import logging
import time
import math
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict, deque

class MathVisualizer:
    """
    Mathematical visualization component for Schwabot.
    
    Handles visualization of:
    - Mathematical calculations
    - Tensor operations
    - Algorithm performance
    - Mathematical patterns
    - Computational complexity
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        update_interval: float = 0.5,
        enable_tensor_visualization: bool = True,
        enable_algorithm_tracking: bool = True
    ):
        """
        Initialize the mathematical visualizer.
        
        Args:
            max_history: Maximum number of data points to keep
            update_interval: Update interval in seconds
            enable_tensor_visualization: Enable tensor operation visualization
            enable_algorithm_tracking: Enable algorithm performance tracking
        """
        self.logger = logging.getLogger('MathVisualizer')
        
        # Configuration
        self.max_history = max_history
        self.update_interval = update_interval
        self.enable_tensor_visualization = enable_tensor_visualization
        self.enable_algorithm_tracking = enable_algorithm_tracking
        
        # Data storage
        self.calculation_history = deque(maxlen=max_history)
        self.tensor_operations = defaultdict(lambda: deque(maxlen=max_history))
        self.algorithm_performance = defaultdict(lambda: deque(maxlen=max_history))
        self.mathematical_patterns = defaultdict(lambda: deque(maxlen=max_history))
        self.complexity_metrics = defaultdict(lambda: deque(maxlen=max_history))
        
        # Current state
        self.active_calculations = {}
        self.completed_calculations = []
        self.tensor_cache = {}
        
        # Callbacks
        self.math_callbacks: List[Callable] = []
        
        # Async state
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None
    
    def add_calculation(
        self,
        operation: str,
        inputs: Dict[str, Any],
        result: Any,
        duration: float,
        complexity: str = 'O(1)'
    ):
        """Add a mathematical calculation"""
        calculation_data = {
            'timestamp': time.time(),
            'operation': operation,
            'inputs': inputs,
            'result': result,
            'duration': duration,
            'complexity': complexity,
            'input_size': self._calculate_input_size(inputs)
        }
        
        self.calculation_history.append(calculation_data)
        self.completed_calculations.append(calculation_data)
        
        # Update complexity metrics
        self._update_complexity_metrics(operation, duration, complexity, calculation_data['input_size'])
        
        # Trigger callbacks
        self._trigger_math_callbacks('calculation_completed', calculation_data)
    
    def add_tensor_operation(
        self,
        operation: str,
        tensor_shape: List[int],
        result_shape: List[int],
        duration: float,
        memory_usage: float = 0
    ):
        """Add a tensor operation"""
        if not self.enable_tensor_visualization:
            return
            
        tensor_data = {
            'timestamp': time.time(),
            'operation': operation,
            'input_shape': tensor_shape,
            'output_shape': result_shape,
            'duration': duration,
            'memory_usage': memory_usage,
            'flops': self._estimate_flops(operation, tensor_shape, result_shape)
        }
        
        self.tensor_operations[operation].append(tensor_data)
        
        # Update tensor cache
        self.tensor_cache[operation] = tensor_data
        
        # Trigger callbacks
        self._trigger_math_callbacks('tensor_operation', tensor_data)
    
    def add_algorithm_performance(
        self,
        algorithm: str,
        input_size: int,
        duration: float,
        memory_usage: float,
        iterations: int = 1
    ):
        """Add algorithm performance data"""
        if not self.enable_algorithm_tracking:
            return
            
        performance_data = {
            'timestamp': time.time(),
            'algorithm': algorithm,
            'input_size': input_size,
            'duration': duration,
            'memory_usage': memory_usage,
            'iterations': iterations,
            'throughput': input_size / duration if duration > 0 else 0
        }
        
        self.algorithm_performance[algorithm].append(performance_data)
        
        # Trigger callbacks
        self._trigger_math_callbacks('algorithm_performance', performance_data)
    
    def add_mathematical_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        confidence: float
    ):
        """Add mathematical pattern detection"""
        pattern_info = {
            'timestamp': time.time(),
            'pattern_type': pattern_type,
            'pattern_data': pattern_data,
            'confidence': confidence,
            'significance': self._calculate_pattern_significance(pattern_data)
        }
        
        self.mathematical_patterns[pattern_type].append(pattern_info)
        
        # Trigger callbacks
        self._trigger_math_callbacks('pattern_detected', pattern_info)
    
    def _calculate_input_size(self, inputs: Dict[str, Any]) -> int:
        """Calculate the size of inputs for complexity analysis"""
        total_size = 0
        
        for key, value in inputs.items():
            if isinstance(value, (list, tuple)):
                total_size += len(value)
            elif isinstance(value, dict):
                total_size += len(value)
            elif isinstance(value, (int, float)):
                total_size += 1
            else:
                total_size += 1
        
        return total_size
    
    def _estimate_flops(self, operation: str, input_shape: List[int], output_shape: List[int]) -> int:
        """Estimate floating point operations for tensor operations"""
        if operation in ['add', 'subtract']:
            return math.prod(output_shape)
        elif operation in ['multiply', 'elementwise_multiply']:
            return math.prod(output_shape)
        elif operation == 'matrix_multiply':
            if len(input_shape) >= 2 and len(output_shape) >= 2:
                return input_shape[0] * input_shape[1] * output_shape[1]
        elif operation == 'convolution':
            # Simplified convolution FLOP estimation
            return math.prod(output_shape) * 9  # Assuming 3x3 kernel
        
        return math.prod(output_shape)
    
    def _update_complexity_metrics(self, operation: str, duration: float, complexity: str, input_size: int):
        """Update complexity analysis metrics"""
        complexity_data = {
            'timestamp': time.time(),
            'operation': operation,
            'duration': duration,
            'complexity': complexity,
            'input_size': input_size,
            'efficiency': input_size / duration if duration > 0 else 0
        }
        
        self.complexity_metrics[operation].append(complexity_data)
    
    def _calculate_pattern_significance(self, pattern_data: Dict[str, Any]) -> float:
        """Calculate the significance of a detected pattern"""
        # Simplified significance calculation
        if 'strength' in pattern_data:
            return pattern_data['strength']
        elif 'frequency' in pattern_data:
            return min(pattern_data['frequency'] / 100, 1.0)
        else:
            return 0.5
    
    def get_calculation_summary(self) -> Dict[str, Any]:
        """Get mathematical calculation summary"""
        if not self.calculation_history:
            return {
                'total_calculations': 0,
                'avg_duration': 0,
                'total_operations': 0
            }
        
        total_calculations = len(self.calculation_history)
        total_duration = sum(calc['duration'] for calc in self.calculation_history)
        avg_duration = total_duration / total_calculations if total_calculations > 0 else 0
        
        # Group by operation type
        operation_counts = defaultdict(int)
        for calc in self.calculation_history:
            operation_counts[calc['operation']] += 1
        
        return {
            'total_calculations': total_calculations,
            'avg_duration': avg_duration,
            'total_duration': total_duration,
            'operation_distribution': dict(operation_counts),
            'recent_calculations': list(self.calculation_history)[-10:]  # Last 10
        }
    
    def get_tensor_summary(self) -> Dict[str, Any]:
        """Get tensor operations summary"""
        if not self.enable_tensor_visualization:
            return {}
        
        summary = {}
        total_flops = 0
        
        for operation, data in self.tensor_operations.items():
            if not data:
                continue
            
            recent_data = data[-10:]  # Last 10 operations
            total_ops = len(data)
            total_duration = sum(op['duration'] for op in data)
            total_flops += sum(op['flops'] for op in data)
            
            summary[operation] = {
                'total_operations': total_ops,
                'avg_duration': total_duration / total_ops if total_ops > 0 else 0,
                'total_flops': sum(op['flops'] for op in data),
                'recent_operations': recent_data
            }
        
        summary['total_flops'] = total_flops
        return summary
    
    def get_algorithm_summary(self) -> Dict[str, Any]:
        """Get algorithm performance summary"""
        if not self.enable_algorithm_tracking:
            return {}
        
        summary = {}
        
        for algorithm, data in self.algorithm_performance.items():
            if not data:
                continue
            
            recent_data = data[-10:]  # Last 10 runs
            total_runs = len(data)
            avg_duration = sum(run['duration'] for run in data) / total_runs
            avg_throughput = sum(run['throughput'] for run in data) / total_runs
            
            summary[algorithm] = {
                'total_runs': total_runs,
                'avg_duration': avg_duration,
                'avg_throughput': avg_throughput,
                'recent_runs': recent_data
            }
        
        return summary
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get mathematical pattern summary"""
        summary = {}
        
        for pattern_type, data in self.mathematical_patterns.items():
            if not data:
                continue
            
            recent_patterns = data[-10:]  # Last 10 patterns
            avg_confidence = sum(p['confidence'] for p in data) / len(data)
            avg_significance = sum(p['significance'] for p in data) / len(data)
            
            summary[pattern_type] = {
                'total_patterns': len(data),
                'avg_confidence': avg_confidence,
                'avg_significance': avg_significance,
                'recent_patterns': recent_patterns
            }
        
        return summary
    
    def get_complexity_summary(self) -> Dict[str, Any]:
        """Get complexity analysis summary"""
        summary = {}
        
        for operation, data in self.complexity_metrics.items():
            if not data:
                continue
            
            recent_data = data[-10:]  # Last 10 calculations
            avg_efficiency = sum(calc['efficiency'] for calc in data) / len(data)
            
            summary[operation] = {
                'total_calculations': len(data),
                'avg_efficiency': avg_efficiency,
                'complexity_classes': list(set(calc['complexity'] for calc in data)),
                'recent_calculations': recent_data
            }
        
        return summary
    
    def get_all_math_data(self) -> Dict[str, Any]:
        """Get all mathematical data"""
        return {
            'calculations': self.get_calculation_summary(),
            'tensor_operations': self.get_tensor_summary(),
            'algorithm_performance': self.get_algorithm_summary(),
            'patterns': self.get_pattern_summary(),
            'complexity': self.get_complexity_summary(),
            'active_calculations': len(self.active_calculations),
            'tensor_cache_size': len(self.tensor_cache)
        }
    
    def register_math_callback(self, callback: Callable):
        """Register a callback for mathematical events"""
        self.math_callbacks.append(callback)
    
    def _trigger_math_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Trigger mathematical callbacks"""
        for callback in self.math_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Math callback error: {e}")
    
    async def start(self):
        """Start the mathematical visualizer"""
        if self.is_running:
            return
            
        self.is_running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Math Visualizer started")
    
    async def stop(self):
        """Stop the mathematical visualizer"""
        self.is_running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Math Visualizer stopped")
    
    async def _update_loop(self):
        """Main update loop"""
        while self.is_running:
            try:
                # Get all mathematical data
                math_data = self.get_all_math_data()
                
                # Trigger periodic callbacks
                for callback in self.math_callbacks:
                    try:
                        callback('periodic_update', math_data)
                    except Exception as e:
                        self.logger.error(f"Periodic callback error: {e}")
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Math visualizer update loop error: {e}")
                await asyncio.sleep(1)
    
    def get_math_alerts(self) -> List[Dict[str, Any]]:
        """Get mathematical alerts"""
        alerts = []
        current_time = time.time()
        
        # Check for slow calculations
        recent_calculations = [c for c in self.calculation_history if current_time - c['timestamp'] <= 60]  # Last minute
        if recent_calculations:
            slow_calculations = [c for c in recent_calculations if c['duration'] > 1.0]  # > 1 second
            if slow_calculations:
                alerts.append({
                    'type': 'slow_calculation',
                    'severity': 'warning',
                    'message': f"Slow calculations detected: {len(slow_calculations)} calculations > 1s in last minute",
                    'timestamp': current_time
                })
        
        # Check for high memory usage in tensor operations
        if self.enable_tensor_visualization:
            recent_tensor_ops = []
            for ops in self.tensor_operations.values():
                recent_tensor_ops.extend([op for op in ops if current_time - op['timestamp'] <= 60])
            
            high_memory_ops = [op for op in recent_tensor_ops if op['memory_usage'] > 1000000000]  # > 1GB
            if high_memory_ops:
                alerts.append({
                    'type': 'high_memory_tensor',
                    'severity': 'warning',
                    'message': f"High memory tensor operations: {len(high_memory_ops)} operations > 1GB in last minute",
                    'timestamp': current_time
                })
        
        return alerts 