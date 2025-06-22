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
import logging

# Set high precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


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
                        allocation_weight=1.0,
                    )

                    # Create profit node
                    profit_node = ProfitNode(
                        node_id=node_id,
                        position=position,
                        profit_value=Decimal("0.0"),
                        volume_profile=volume_profile,
                        recursive_depth=0,
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
        strategy: ProfitAllocationStrategy = ProfitAllocationStrategy.LINEAR,
    ) -> ProfitAllocationResult:
        """Calculate P_v = ∑ Δv × (R_n(t) · P_t) | Recursive volumetric profit."""
        total_profit = Decimal("0.0")
        volume_weighted_profit = Decimal("0.0")
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
            profit_component = (
                Decimal(str(delta_v))
                * Decimal(str(recursive_multiplier))
                * Decimal(str(price_tick))
            )

            # Apply allocation strategy
            strategy_adjusted_profit = self._apply_allocation_strategy(
                profit_component, strategy
            )

            total_profit += strategy_adjusted_profit
            volume_weighted_profit += strategy_adjusted_profit * Decimal(
                str(volume_profile.allocation_weight)
            )

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
            timestamp=time.time(),
        )

        self.profit_history.append(result)
        return result

    def create_allocation_chain(
        self, chain_id: str, start_node_id: str, chain_length: int = 10
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

    def propagate_profit_through_chain(
        self, chain_id: str, initial_profit: Decimal, propagation_decay: float = 0.95
    ) -> Dict[str, Decimal]:
        """Propagate profit through an allocation chain with decay."""
        if chain_id not in self.allocation_chains:
            return {}

        chain = self.allocation_chains[chain_id]
        profit_distribution = {}
        current_profit = initial_profit

        for i, node_id in enumerate(chain):
            node = self.profit_nodes[node_id]

            # Calculate profit for this node
            node_profit = current_profit * Decimal(str(propagation_decay**i))

            # Add to node's profit value
            node.profit_value += node_profit
            profit_distribution[node_id] = node_profit

            # Record allocation history
            node.allocation_history.append(
                {
                    "timestamp": time.time(),
                    "profit_allocated": float(node_profit),
                    "chain_id": chain_id,
                    "chain_position": i,
                    "total_node_profit": float(node.profit_value),
                }
            )

        return profit_distribution

    def measure_2d_profit_density(self, z_level: int = 0) -> np.ndarray:
        """Measure profit density in 2D slice of the 3D space."""
        x_dim, y_dim, z_dim = self.spatial_dimensions

        if z_level >= z_dim:
            z_level = z_dim // 2  # Default to middle slice

        density_map = np.zeros((x_dim, y_dim))

        for x in range(x_dim):
            for y in range(y_dim):
                node_id = f"profit_node_{x}_{y}_{z_level}"
                if node_id in self.profit_nodes:
                    node = self.profit_nodes[node_id]
                    density_map[x, y] = float(node.profit_value)

        return density_map

    def measure_3d_profit_volume(self) -> Dict[str, Any]:
        """Measure volumetric profit distribution in 3D space."""
        x_dim, y_dim, z_dim = self.spatial_dimensions
        volume_data = np.zeros((x_dim, y_dim, z_dim))

        total_volume_profit = Decimal("0.0")
        max_profit_node = None
        max_profit_value = Decimal("0.0")

        for node in self.profit_nodes.values():
            x, y, z = node.position
            x, y, z = int(x), int(y), int(z)

            if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
                volume_data[x, y, z] = float(node.profit_value)
                total_volume_profit += node.profit_value

                if node.profit_value > max_profit_value:
                    max_profit_value = node.profit_value
                    max_profit_node = node.node_id

        # Calculate volume statistics
        volume_stats = {
            "total_volume_profit": float(total_volume_profit),
            "average_profit_density": float(total_volume_profit)
            / len(self.profit_nodes),
            "max_profit_node": max_profit_node,
            "max_profit_value": float(max_profit_value),
            "volume_data": volume_data,
            "profit_gradient": self._calculate_profit_gradient(volume_data),
            "profit_centroid": self._calculate_profit_centroid(volume_data),
        }

        return volume_stats

    def optimize_recursive_bag_growth(
        self, growth_target: float = 1.5, optimization_iterations: int = 100
    ) -> Dict[str, Any]:
        """Optimize recursive bag growth across profit chains."""
        optimization_results = {
            "initial_total_profit": float(
                sum(node.profit_value for node in self.profit_nodes.values())
            ),
            "optimized_chains": [],
            "growth_achieved": 0.0,
            "optimization_efficiency": 0.0,
        }

        for iteration in range(optimization_iterations):
            # Select random chain for optimization
            if not self.allocation_chains:
                break

            chain_id = np.random.choice(list(self.allocation_chains.keys()))
            chain = self.allocation_chains[chain_id]

            # Apply growth optimization
            growth_applied = self._apply_recursive_growth_optimization(
                chain, growth_target
            )

            if growth_applied > 0:
                optimization_results["optimized_chains"].append(
                    {
                        "chain_id": chain_id,
                        "growth_applied": growth_applied,
                        "iteration": iteration,
                    }
                )

        # Calculate final metrics
        final_total_profit = float(
            sum(node.profit_value for node in self.profit_nodes.values())
        )
        optimization_results["final_total_profit"] = final_total_profit
        optimization_results["growth_achieved"] = (
            final_total_profit / optimization_results["initial_total_profit"] - 1.0
        )
        optimization_results["optimization_efficiency"] = (
            optimization_results["growth_achieved"] / growth_target
        )

        return optimization_results

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
        delta_factor = (
            1.0 + (volume_profile.delta_volume / volume_profile.base_volume) * 0.1
        )

        recursive_multiplier = (
            base_multiplier * time_factor * volume_factor * price_factor * delta_factor
        )

        return max(0.1, min(5.0, recursive_multiplier))  # Bound between 0.1 and 5.0

    def _apply_allocation_strategy(
        self, profit_component: Decimal, strategy: ProfitAllocationStrategy
    ) -> Decimal:
        """Apply allocation strategy to profit component."""
        base_value = float(profit_component)

        if strategy == ProfitAllocationStrategy.LINEAR:
            return profit_component
        elif strategy == ProfitAllocationStrategy.EXPONENTIAL:
            if base_value > 0:
                adjusted = base_value * math.exp(
                    base_value / 1000000.0
                )  # Scale for stability
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

    def _calculate_allocation_efficiency(
        self, volume_deltas: List[Tuple[str, float]]
    ) -> float:
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
                distance_score = self._calculate_distance_score(
                    current_node, connected_node
                )
                profit_score = float(connected_node.profit_value) / 1000.0  # Normalize
                volume_score = (
                    connected_node.volume_profile.base_volume / 10000.0
                )  # Normalize

                total_score = distance_score + profit_score + volume_score

                if total_score > best_score:
                    best_score = total_score
                    best_node_id = connected_node_id

        return best_node_id

    def _calculate_distance_score(
        self, node_a: ProfitNode, node_b: ProfitNode
    ) -> float:
        """Calculate distance-based score between nodes."""
        pos_a = np.array(node_a.position)
        pos_b = np.array(node_b.position)
        distance = np.linalg.norm(pos_a - pos_b)

        # Prefer moderate distances (not too close, not too far)
        optimal_distance = 2.0
        distance_score = 1.0 / (1.0 + abs(distance - optimal_distance))

        return distance_score

    def _calculate_profit_gradient(self, volume_data: np.ndarray) -> np.ndarray:
        """Calculate profit gradient in 3D space."""
        gradient_x = np.gradient(volume_data, axis=0)
        gradient_y = np.gradient(volume_data, axis=1)
        gradient_z = np.gradient(volume_data, axis=2)

        # Magnitude of gradient
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)

        return gradient_magnitude

    def _calculate_profit_centroid(
        self, volume_data: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate centroid of profit distribution."""
        total_profit = np.sum(volume_data)

        if total_profit == 0:
            return (0.0, 0.0, 0.0)

        x_dim, y_dim, z_dim = volume_data.shape

        # Calculate weighted centroid
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(x_dim), np.arange(y_dim), np.arange(z_dim), indexing="ij"
        )

        centroid_x = np.sum(x_coords * volume_data) / total_profit
        centroid_y = np.sum(y_coords * volume_data) / total_profit
        centroid_z = np.sum(z_coords * volume_data) / total_profit

        return (float(centroid_x), float(centroid_y), float(centroid_z))

    def _apply_recursive_growth_optimization(
        self, chain: List[str], growth_target: float
    ) -> float:
        """Apply recursive growth optimization to a chain."""
        if not chain:
            return 0.0

        growth_applied = 0.0

        for i, node_id in enumerate(chain):
            if node_id not in self.profit_nodes:
                continue

            node = self.profit_nodes[node_id]

            # Calculate growth based on position in chain and recursive depth
            chain_position_factor = (len(chain) - i) / len(
                chain
            )  # Higher for earlier nodes
            recursive_factor = 1.0 + (node.recursive_depth * 0.1)

            # Apply growth
            growth_amount = (
                float(node.profit_value)
                * growth_target
                * 0.01
                * chain_position_factor
                * recursive_factor
            )
            node.profit_value += Decimal(str(growth_amount))

            growth_applied += growth_amount

        return growth_applied


# Convenience functions
def create_profit_routing_system(
    dimensions: Tuple[int, int, int] = (10, 10, 10)
) -> ProfitRoutingEngine:
    """Create and initialize profit routing system."""
    engine = ProfitRoutingEngine(dimensions)
    engine.initialize_profit_space()
    return engine


def simulate_profit_allocation(
    engine: ProfitRoutingEngine,
    simulation_steps: int = 100,
    base_profit_per_step: float = 1000.0,
) -> List[ProfitAllocationResult]:
    """Simulate profit allocation over multiple steps."""
    results = []

    for step in range(simulation_steps):
        # Generate random volume deltas
        volume_count = min(10, len(engine.volume_profiles))
        selected_volumes = np.random.choice(
            list(engine.volume_profiles.keys()), volume_count, replace=False
        )

        volume_deltas = [
            (vol_id, np.random.uniform(-500, 1500)) for vol_id in selected_volumes
        ]

        # Random price tick
        price_tick = np.random.uniform(30000, 70000)

        # Random strategy
        strategy = np.random.choice(list(ProfitAllocationStrategy))

        # Calculate profit allocation
        result = engine.calculate_volumetric_profit(volume_deltas, price_tick, strategy)
        results.append(result)

    return results
