#!/usr/bin/env python3
"""Profit Routing Engine - Volumetric Profit Allocation Chains.

Implements the core mathematical framework for:
- P_v = ∑ Δv × (R_n(t) · P_t) | Recursive volumetric profit calculation
- Volume-scaled profit injections across tick cycles
- 2D/3D space measurement for profit allocation chains
- Multi-dimensional profit mapping and recursive bag growth
"""

import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal, getcontext
from enum import Enum

# Set high precision for financial calculations
getcontext().prec = 28


class ProfitAllocationStrategy(Enum):
    """Profit allocation strategies."""
    LINEAR = "LINEAR"
    EXPONENTIAL = "EXPONENTIAL"
    LOGARITHMIC = "LOGARITHMIC"
    SIGMOID = "SIGMOID"
    FRACTAL = "FRACTAL"


@dataclass
class VolumeProfile:
    """Volume profile for profit calculations."""
    
    volume_id: str
    base_volume: float
    delta_volume: float
    timestamp: float
    price_level: float
    allocation_weight: float = 1.0


@dataclass
class ProfitNode:
    """Node in the profit allocation chain."""
    
    node_id: str
    position: Tuple[float, float, float]  # 3D coordinates
    profit_value: Decimal
    volume_profile: VolumeProfile
    connections: List[str] = field(default_factory=list)
    allocation_history: List[Dict[str, Any]] = field(default_factory=list)
    recursive_depth: int = 0


@dataclass
class ProfitAllocationResult:
    """Result of profit allocation process."""
    
    allocation_id: str
    total_profit_allocated: Decimal
    volume_weighted_profit: Decimal
    spatial_distribution: Dict[str, Tuple[float, float, float]]
    allocation_efficiency: float
    recursive_growth_factor: float
    timestamp: float


class ProfitRoutingEngine:
    """Core profit routing with volumetric allocation chains."""
    
    def __init__(self, spatial_dimensions: Tuple[int, int, int] = (20, 20, 20)) -> None:
        """Initialize profit routing engine with spatial dimensions."""
        self.spatial_dimensions = spatial_dimensions
        self.profit_nodes: Dict[str, ProfitNode] = {}
        self.volume_profiles: Dict[str, VolumeProfile] = {}
        self.allocation_chains: Dict[str, List[str]] = {}
        self.profit_history: List[ProfitAllocationResult] = []
        self.spatial_grid: np.ndarray = np.zeros(spatial_dimensions)
        self.recursive_multipliers: Dict[int, float] = {}
        
    def initialize_profit_space(self) -> None:
        """Initialize the 3D profit allocation space."""
        x_dim, y_dim, z_dim = self.spatial_dimensions
        
        for x in range(x_dim):
            for y in range(y_dim):
                for z in range(z_dim):
                    node_id = f"profit_node_{x}_{y}_{z}"
                    position = (float(x), float(y), float(z))
                    
                    # Create volume profile
                    volume_profile = VolumeProfile(
                        volume_id=f"vol_{node_id}",
                        base_volume=np.random.uniform(100, 10000),
                        delta_volume=0.0,
                        timestamp=time.time(),
                        price_level=np.random.uniform(30000, 70000),  # BTC price range
                        allocation_weight=1.0
                    )
                    
                    # Create profit node
                    profit_node = ProfitNode(
                        node_id=node_id,
                        position=position,
                        profit_value=Decimal('0.0'),
                        volume_profile=volume_profile,
                        recursive_depth=0
                    )
                    
                    # Connect to adjacent nodes
                    profit_node.connections = self._get_adjacent_profit_nodes(x, y, z)
                    
                    self.profit_nodes[node_id] = profit_node
                    self.volume_profiles[volume_profile.volume_id] = volume_profile
        
        # Initialize allocation chains
        self._initialize_allocation_chains()
    
    def calculate_volumetric_profit(
        self, 
        volume_deltas: List[Tuple[str, float]], 
        price_tick: float,
        strategy: ProfitAllocationStrategy = ProfitAllocationStrategy.LINEAR
    ) -> ProfitAllocationResult:
        """Calculate P_v = ∑ Δv × (R_n(t) · P_t) | Recursive volumetric profit."""
        total_profit = Decimal('0.0')
        volume_weighted_profit = Decimal('0.0')
        spatial_distribution = {}
        
        for volume_id, delta_v in volume_deltas:
            if volume_id not in self.volume_profiles:
                continue
            
            volume_profile = self.volume_profiles[volume_id]
            
            # Update delta volume
            volume_profile.delta_volume = delta_v
            volume_profile.timestamp = time.time()
            
            # Calculate R_n(t) - recursive multiplier
            recursive_multiplier = self._calculate_recursive_multiplier(volume_profile)
            
            # Calculate profit component: Δv × (R_n(t) · P_t)
            profit_component = Decimal(str(delta_v)) * Decimal(str(recursive_multiplier)) * Decimal(str(price_tick))
            
            # Apply allocation strategy
            strategy_adjusted_profit = self._apply_allocation_strategy(profit_component, strategy)
            
            total_profit += strategy_adjusted_profit
            volume_weighted_profit += strategy_adjusted_profit * Decimal(str(volume_profile.allocation_weight))
            
            # Find associated profit node
            associated_node = self._find_node_by_volume(volume_id)
            if associated_node:
                associated_node.profit_value += strategy_adjusted_profit
                spatial_distribution[volume_id] = associated_node.position
        
        # Calculate allocation efficiency
        allocation_efficiency = self._calculate_allocation_efficiency(volume_deltas)
        
        # Calculate recursive growth factor
        recursive_growth_factor = self._calculate_recursive_growth_factor()
        
        result = ProfitAllocationResult(
            allocation_id=f"alloc_{len(self.profit_history)}_{int(time.time())}",
            total_profit_allocated=total_profit,
            volume_weighted_profit=volume_weighted_profit,
            spatial_distribution=spatial_distribution,
            allocation_efficiency=allocation_efficiency,
            recursive_growth_factor=recursive_growth_factor,
            timestamp=time.time()
        )
        
        self.profit_history.append(result)
        return result
    
    def _get_adjacent_profit_nodes(self, x: int, y: int, z: int) -> List[str]:
        """Get adjacent profit nodes in 3D space."""
        adjacent = []
        x_dim, y_dim, z_dim = self.spatial_dimensions
        
        # Check all 26 adjacent positions in 3D (including diagonals)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip self
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < x_dim and 0 <= ny < y_dim and 0 <= nz < z_dim:
                        adjacent.append(f"profit_node_{nx}_{ny}_{nz}")
        
        return adjacent
    
    def _initialize_allocation_chains(self) -> None:
        """Initialize default allocation chains."""
        # Create several default chains
        chain_count = min(10, len(self.profit_nodes) // 10)
        
        for i in range(chain_count):
            # Select random start node
            start_node_id = np.random.choice(list(self.profit_nodes.keys()))
            chain_id = f"default_chain_{i}"
            self.create_allocation_chain(chain_id, start_node_id, 8)
    
    def create_allocation_chain(
        self, 
        chain_id: str, 
        start_node_id: str, 
        chain_length: int = 10
    ) -> List[str]:
        """Create a profit allocation chain through the 3D space."""
        if start_node_id not in self.profit_nodes:
            return []
        
        chain = [start_node_id]
        current_node_id = start_node_id
        
        for _ in range(chain_length - 1):
            current_node = self.profit_nodes[current_node_id]
            
            # Find best next node based on profit potential
            best_next_node = self._find_best_next_node(current_node)
            
            if best_next_node and best_next_node not in chain:
                chain.append(best_next_node)
                current_node_id = best_next_node
            else:
                break
        
        self.allocation_chains[chain_id] = chain
        
        # Update recursive depths
        for i, node_id in enumerate(chain):
            self.profit_nodes[node_id].recursive_depth = i
        
        return chain
    
    def _calculate_recursive_multiplier(self, volume_profile: VolumeProfile) -> float:
        """Calculate R_n(t) - recursive multiplier for volume profile."""
        # Base multiplier
        base_multiplier = 1.0
        
        # Time-based component
        time_factor = math.sin(volume_profile.timestamp * 0.001) * 0.1 + 1.0
        
        # Volume-based component
        volume_factor = math.log(volume_profile.base_volume + 1) * 0.01
        
        # Price-based component
        price_factor = volume_profile.price_level / 50000.0  # Normalize around 50k
        
        # Delta volume impact
        delta_factor = 1.0 + (volume_profile.delta_volume / volume_profile.base_volume) * 0.1
        
        recursive_multiplier = base_multiplier * time_factor * volume_factor * price_factor * delta_factor
        
        return max(0.1, min(5.0, recursive_multiplier))  # Bound between 0.1 and 5.0
    
    def _apply_allocation_strategy(
        self, 
        profit_component: Decimal, 
        strategy: ProfitAllocationStrategy
    ) -> Decimal:
        """Apply allocation strategy to profit component."""
        base_value = float(profit_component)
        
        if strategy == ProfitAllocationStrategy.LINEAR:
            return profit_component
        elif strategy == ProfitAllocationStrategy.EXPONENTIAL:
            if base_value > 0:
                adjusted = base_value * math.exp(base_value / 1000000.0)  # Scale for stability
            else:
                adjusted = base_value
        elif strategy == ProfitAllocationStrategy.LOGARITHMIC:
            if base_value > 0:
                adjusted = base_value * math.log(1 + base_value / 1000.0)
            else:
                adjusted = base_value
        elif strategy == ProfitAllocationStrategy.SIGMOID:
            adjusted = base_value * (2 / (1 + math.exp(-base_value / 10000.0)) - 1)
        elif strategy == ProfitAllocationStrategy.FRACTAL:
            # Golden ratio based fractal scaling
            phi = (1 + math.sqrt(5)) / 2
            adjusted = base_value * (phi ** (base_value / 100000.0))
        else:
            adjusted = base_value
        
        return Decimal(str(adjusted))
    
    def _find_node_by_volume(self, volume_id: str) -> Optional[ProfitNode]:
        """Find profit node associated with volume profile."""
        for node in self.profit_nodes.values():
            if node.volume_profile.volume_id == volume_id:
                return node
        return None
    
    def _calculate_allocation_efficiency(self, volume_deltas: List[Tuple[str, float]]) -> float:
        """Calculate efficiency of profit allocation."""
        if not volume_deltas:
            return 0.0
        
        total_volume_delta = sum(abs(delta) for _, delta in volume_deltas)
        if total_volume_delta == 0:
            return 1.0
        
        # Calculate efficiency based on spatial distribution
        active_nodes = set()
        for volume_id, _ in volume_deltas:
            node = self._find_node_by_volume(volume_id)
            if node:
                active_nodes.add(node.node_id)
        
        # Efficiency is higher when profit is distributed across more nodes
        max_possible_nodes = len(self.profit_nodes)
        distribution_efficiency = len(active_nodes) / max_possible_nodes
        
        return min(1.0, distribution_efficiency * 2.0)  # Scale to reasonable range
    
    def _calculate_recursive_growth_factor(self) -> float:
        """Calculate overall recursive growth factor."""
        if not self.profit_history:
            return 1.0
        
        if len(self.profit_history) < 2:
            return 1.0
        
        # Compare recent profit allocations
        recent_profit = float(self.profit_history[-1].total_profit_allocated)
        previous_profit = float(self.profit_history[-2].total_profit_allocated)
        
        if previous_profit == 0:
            return 1.0
        
        growth_factor = recent_profit / previous_profit
        return max(0.1, min(10.0, growth_factor))  # Bound growth factor
    
    def _find_best_next_node(self, current_node: ProfitNode) -> Optional[str]:
        """Find the best next node for chain propagation."""
        if not current_node.connections:
            return None
        
        best_node_id = None
        best_score = -1.0
        
        for connected_node_id in current_node.connections:
            if connected_node_id in self.profit_nodes:
                connected_node = self.profit_nodes[connected_node_id]
                
                # Score based on profit potential and position
                distance_score = self._calculate_distance_score(current_node, connected_node)
                profit_score = float(connected_node.profit_value) / 1000.0  # Normalize
                volume_score = connected_node.volume_profile.base_volume / 10000.0  # Normalize
                
                total_score = distance_score + profit_score + volume_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_node_id = connected_node_id
        
        return best_node_id
    
    def _calculate_distance_score(self, node_a: ProfitNode, node_b: ProfitNode) -> float:
        """Calculate distance-based score between nodes."""
        pos_a = np.array(node_a.position)
        pos_b = np.array(node_b.position)
        distance = np.linalg.norm(pos_a - pos_b)
        
        # Prefer moderate distances (not too close, not too far)
        optimal_distance = 2.0
        distance_score = 1.0 / (1.0 + abs(distance - optimal_distance))
        
        return distance_score


# Convenience functions
def create_profit_routing_system(dimensions: Tuple[int, int, int] = (10, 10, 10)) -> ProfitRoutingEngine:
    """Create and initialize profit routing system."""
    engine = ProfitRoutingEngine(dimensions)
    engine.initialize_profit_space()
    return engine 