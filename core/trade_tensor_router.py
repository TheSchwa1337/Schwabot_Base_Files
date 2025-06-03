"""
Trade Tensor Router
=================

Coordinates tensor routing and execution across the Nominal Channel system.
Manages tensor state transitions, profit allocation, and thermal-aware execution.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging

from .profit_tensor import ProfitTensorStore
from .bitmap_engine import BitmapEngine
from .zbe_temperature_tensor import ZBETemperatureTensor
from .basket_tensor_feedback import BasketTensorFeedback
from .fault_bus import FaultBus, FaultBusEvent
from .memory_timing_orchestrator import MemoryTimingOrchestrator

@dataclass
class TensorRoute:
    """Represents a tensor routing decision with its components"""
    sha_key: str
    bit_depth: int
    confidence: float
    profit_potential: float
    thermal_cost: float
    memory_coherence: float
    phase_depth: float
    timestamp: datetime

class TradeTensorRouter:
    """Routes and manages tensor execution across the system"""
    
    def __init__(self):
        self.profit_store = ProfitTensorStore()
        self.bitmap_engine = BitmapEngine()
        self.zbe_tensor = ZBETemperatureTensor()
        self.tensor_feedback = BasketTensorFeedback()
        self.fault_bus = FaultBus()
        self.memory_orchestrator = MemoryTimingOrchestrator()
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Route history for analysis
        self.route_history: List[TensorRoute] = []
        
        # Performance metrics
        self.metrics = {
            'total_routes': 0,
            'successful_routes': 0,
            'avg_profit': 0.0,
            'avg_thermal_cost': 0.0,
            'bit_depth_stats': {
                depth: {
                    'count': 0,
                    'success_rate': 0.0,
                    'avg_profit': 0.0,
                    'avg_thermal': 0.0
                }
                for depth in [4, 8, 16, 42, 81]
            }
        }
        
        # Register fault handlers
        self._register_fault_handlers()

    def _register_fault_handlers(self):
        """Register handlers for various fault types"""
        def handle_thermal_fault(event: FaultBusEvent):
            if event.severity > 0.8:
                self._reduce_thermal_load()
        
        def handle_memory_fault(event: FaultBusEvent):
            if event.severity > 0.7:
                self._optimize_memory_usage()
        
        def handle_profit_fault(event: FaultBusEvent):
            if event.severity < 0.2:
                self._adjust_profit_thresholds()
        
        self.fault_bus.register_handler("thermal_high", handle_thermal_fault)
        self.fault_bus.register_handler("memory_coherence_low", handle_memory_fault)
        self.fault_bus.register_handler("profit_low", handle_profit_fault)

    def route_tensor(
        self,
        market_signature: Dict,
        current_state: np.ndarray,
        time_dilation: float = 1.0
    ) -> TensorRoute:
        """
        Route tensor through appropriate bit depth based on market conditions
        
        Args:
            market_signature: Current market state signature
            current_state: Current tensor state
            time_dilation: Time dilation factor for execution
            
        Returns:
            TensorRoute object containing routing decision
        """
        # Get current bitmap and generate SHA key
        bitmap = self.bitmap_engine.infer_current_bitmap(market_signature)
        sha_key = self.bitmap_engine.generate_sha_key(bitmap)
        
        # Get tensor feedback
        tensor_vector = self.profit_store.lookup(sha_key)
        if tensor_vector is None:
            tensor_vector = np.ones(8)  # Default tensor
            
        # Calculate routing parameters
        entropy_level = self._calculate_entropy_level(market_signature)
        trust_vector = self._calculate_trust_vector(tensor_vector)
        thermal_state = self.zbe_tensor.read_cpu_temperature()
        memory_coherence = self._calculate_memory_coherence(current_state)
        phase_depth = self._calculate_phase_depth(current_state)
        profit_gradient = self._calculate_profit_gradient(tensor_vector)
        
        # Get optimal bit depth
        bit_depth, confidence = self._select_bit_depth(
            entropy_level,
            trust_vector,
            thermal_state,
            memory_coherence,
            phase_depth,
            profit_gradient
        )
        
        # Calculate profit potential and thermal cost
        profit_potential = self._estimate_profit_potential(
            tensor_vector,
            bit_depth,
            time_dilation
        )
        thermal_cost = self._estimate_thermal_cost(
            bit_depth,
            tensor_vector.size,
            time_dilation
        )
        
        # Create route
        route = TensorRoute(
            sha_key=sha_key,
            bit_depth=bit_depth,
            confidence=confidence,
            profit_potential=profit_potential,
            thermal_cost=thermal_cost,
            memory_coherence=memory_coherence,
            phase_depth=phase_depth,
            timestamp=datetime.now()
        )
        
        # Update history and metrics
        self._update_route_history(route)
        
        return route

    def _calculate_entropy_level(self, market_signature: Dict) -> float:
        """Calculate entropy level from market signature"""
        entropy = market_signature.get('entropy', [0.2]*5)
        return np.mean(entropy)

    def _calculate_trust_vector(self, tensor_vector: np.ndarray) -> List[float]:
        """Calculate trust vector from tensor state"""
        if tensor_vector.size == 0:
            return [0.5] * 5
            
        # Calculate trust metrics
        stability = 1.0 - np.std(tensor_vector)
        coherence = np.mean(tensor_vector)
        consistency = 1.0 - np.max(np.abs(np.diff(tensor_vector)))
        
        return [stability, coherence, consistency, 0.5, 0.5]

    def _calculate_memory_coherence(self, current_state: np.ndarray) -> float:
        """Calculate memory coherence score"""
        if current_state.size == 0:
            return 0.0
            
        # Calculate coherence based on state stability
        stability = 1.0 - np.std(current_state)
        consistency = 1.0 - np.max(np.abs(np.diff(current_state)))
        
        return (stability + consistency) / 2.0

    def _calculate_phase_depth(self, current_state: np.ndarray) -> float:
        """Calculate phase depth from current state"""
        if current_state.size == 0:
            return 0.0
            
        # Calculate phase depth based on state complexity
        unique_values = len(np.unique(current_state))
        max_depth = 5.0
        
        return min(max_depth, unique_values / current_state.size * max_depth)

    def _calculate_profit_gradient(self, tensor_vector: np.ndarray) -> float:
        """Calculate profit gradient from tensor vector"""
        if tensor_vector.size < 2:
            return 0.0
            
        # Calculate gradient based on recent changes
        changes = np.diff(tensor_vector)
        return np.mean(changes)

    def _select_bit_depth(
        self,
        entropy_level: float,
        trust_vector: List[float],
        thermal_state: float,
        memory_coherence: float,
        phase_depth: float,
        profit_gradient: float
    ) -> Tuple[int, float]:
        """Select optimal bit depth based on system state"""
        # Define bit depth tiers
        tiers = {
            4: {'min_entropy': 0.0, 'max_entropy': 0.3, 'thermal_threshold': 80.0},
            8: {'min_entropy': 0.2, 'max_entropy': 0.5, 'thermal_threshold': 75.0},
            16: {'min_entropy': 0.4, 'max_entropy': 0.7, 'thermal_threshold': 70.0},
            42: {'min_entropy': 0.6, 'max_entropy': 0.9, 'thermal_threshold': 65.0},
            81: {'min_entropy': 0.8, 'max_entropy': 1.0, 'thermal_threshold': 60.0}
        }
        
        # Calculate scores for each tier
        scores = {}
        for depth, tier in tiers.items():
            # Check if tier is suitable
            if (entropy_level < tier['min_entropy'] or 
                entropy_level > tier['max_entropy'] or
                thermal_state > tier['thermal_threshold']):
                scores[depth] = 0.0
                continue
                
            # Calculate tier score
            entropy_score = 1.0 - abs(entropy_level - (tier['min_entropy'] + tier['max_entropy'])/2)
            trust_score = np.mean(trust_vector)
            thermal_score = 1.0 - (thermal_state / tier['thermal_threshold'])
            memory_score = memory_coherence
            phase_score = phase_depth / 5.0
            profit_score = 1.0 if profit_gradient > 0 else 0.0
            
            # Weighted combination
            scores[depth] = (
                entropy_score * 0.3 +
                trust_score * 0.2 +
                thermal_score * 0.2 +
                memory_score * 0.15 +
                phase_score * 0.1 +
                profit_score * 0.05
            )
        
        # Select best depth
        best_depth = max(scores.items(), key=lambda x: x[1])
        return best_depth[0], best_depth[1]

    def _estimate_profit_potential(
        self,
        tensor_vector: np.ndarray,
        bit_depth: int,
        time_dilation: float
    ) -> float:
        """Estimate profit potential for given tensor and bit depth"""
        if tensor_vector.size == 0:
            return 0.0
            
        # Calculate base potential
        stability = 1.0 - np.std(tensor_vector)
        coherence = np.mean(tensor_vector)
        
        # Adjust for bit depth
        depth_factor = 1.0 - (bit_depth / 81.0)  # Higher depth = lower factor
        
        # Apply time dilation
        return (stability * coherence * depth_factor) / time_dilation

    def _estimate_thermal_cost(
        self,
        bit_depth: int,
        tensor_size: int,
        time_dilation: float
    ) -> float:
        """Estimate thermal cost for given parameters"""
        # Base cost increases with bit depth and tensor size
        base_cost = (bit_depth / 81.0) * (tensor_size / 1000.0)
        
        # Apply time dilation
        return base_cost * time_dilation

    def _update_route_history(self, route: TensorRoute):
        """Update route history and metrics"""
        # Add to history
        self.route_history.append(route)
        if len(self.route_history) > 1000:
            self.route_history = self.route_history[-1000:]
            
        # Update metrics
        self.metrics['total_routes'] += 1
        depth_stats = self.metrics['bit_depth_stats'][route.bit_depth]
        depth_stats['count'] += 1
        
        # Update averages using exponential moving average
        alpha = 0.1
        self.metrics['avg_profit'] = (1 - alpha) * self.metrics['avg_profit'] + alpha * route.profit_potential
        self.metrics['avg_thermal_cost'] = (1 - alpha) * self.metrics['avg_thermal_cost'] + alpha * route.thermal_cost
        
        depth_stats['avg_profit'] = (1 - alpha) * depth_stats['avg_profit'] + alpha * route.profit_potential
        depth_stats['avg_thermal'] = (1 - alpha) * depth_stats['avg_thermal'] + alpha * route.thermal_cost

    def _reduce_thermal_load(self):
        """Reduce system thermal load"""
        # Increase thermal thresholds
        for depth in self.metrics['bit_depth_stats'].keys():
            self.metrics['bit_depth_stats'][depth]['thermal_threshold'] *= 0.9
            
        # Log thermal reduction
        self.logger.warning("Reducing thermal load - adjusting thresholds")

    def _optimize_memory_usage(self):
        """Optimize memory usage"""
        # Clear old route history
        if len(self.route_history) > 500:
            self.route_history = self.route_history[-500:]
            
        # Log memory optimization
        self.logger.info("Optimizing memory usage - cleared old routes")

    def _adjust_profit_thresholds(self):
        """Adjust profit thresholds based on performance"""
        # Calculate average profit by bit depth
        depth_profits = {}
        for depth, stats in self.metrics['bit_depth_stats'].items():
            if stats['count'] > 0:
                depth_profits[depth] = stats['avg_profit']
                
        # Adjust thresholds based on relative performance
        if depth_profits:
            max_profit = max(depth_profits.values())
            for depth, profit in depth_profits.items():
                if profit < max_profit * 0.5:
                    self.metrics['bit_depth_stats'][depth]['profit_threshold'] *= 0.9
                    
        # Log threshold adjustment
        self.logger.info("Adjusting profit thresholds based on performance")

    def get_route_stats(self) -> Dict:
        """Get statistics about route performance"""
        return {
            'total_routes': self.metrics['total_routes'],
            'avg_profit': self.metrics['avg_profit'],
            'avg_thermal_cost': self.metrics['avg_thermal_cost'],
            'bit_depth_stats': self.metrics['bit_depth_stats'],
            'recent_routes': len(self.route_history)
        } 