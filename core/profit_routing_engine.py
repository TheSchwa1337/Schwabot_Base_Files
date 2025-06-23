#!/usr/bin/env python3
"""Profit Routing Engine - Volumetric Profit Allocation Chains.

Implements the core mathematical framework for:
- P_v = ∑ Δv × (R_n(t) · P_t) | Recursive volumetric profit calculation
- Volume-scaled profit injections across tick cycles
- 2D/3D space measurement for profit allocation chains
- Multi-dimensional profit mapping and recursive bag growth

Enhanced with shared math utilities and comprehensive error handling.
"""

import numpy as np
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal, getcontext
from enum import Enum

# Import shared math utilities
from core.utils.math_utils import (
    calculate_gradient,
    calculate_centroid,
    calculate_distance_score,
    calculate_recursive_multiplier,
    calculate_allocation_efficiency,
    calculate_recursive_growth_factor,
    apply_allocation_strategy,
    safe_decimal_operation,
    validate_spatial_dimensions,
    create_spatial_grid,
)

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

    def __post_init__(self) -> None:
        """Validate volume profile data."""
        if self.base_volume < 0:
            logger.warning(f"Negative base volume for {self.volume_id}, setting to 0")
            self.base_volume = 0.0
        if self.allocation_weight < 0:
            logger.warning(f"Negative allocation weight for {self.volume_id}, setting to 0")
            self.allocation_weight = 0.0


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

    def __post_init__(self) -> None:
        """Validate profit node data."""
        if len(self.position) != 3:
            raise ValueError(f"Position must be 3D tuple, got {len(self.position)}D")
        if self.recursive_depth < 0:
            logger.warning(f"Negative recursive depth for {self.node_id}, setting to 0")
            self.recursive_depth = 0


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

    def __post_init__(self) -> None:
        """Validate allocation result data."""
        if self.allocation_efficiency < 0 or self.allocation_efficiency > 1:
            logger.warning(f"Invalid allocation efficiency: {self.allocation_efficiency}")
            self.allocation_efficiency = np.clip(self.allocation_efficiency, 0.0, 1.0)


class ProfitRoutingEngine:
    """Core profit routing with volumetric allocation chains."""

    def __init__(self, spatial_dimensions: Tuple[int, int, int] = (20, 20, 20)) -> None:
        """Initialize profit routing engine with spatial dimensions."""
        # Validate spatial dimensions
        if not validate_spatial_dimensions(spatial_dimensions):
            raise ValueError(f"Invalid spatial dimensions: {spatial_dimensions}")
        
        self.spatial_dimensions = spatial_dimensions
        self.profit_nodes: Dict[str, ProfitNode] = {}
        self.volume_profiles: Dict[str, VolumeProfile] = {}
        self.allocation_chains: Dict[str, List[str]] = {}
        self.profit_history: List[ProfitAllocationResult] = []
        self.spatial_grid: np.ndarray = create_spatial_grid(spatial_dimensions)
        self.recursive_multipliers: Dict[int, float] = {}
        
        # Performance tracking
        self.operation_count = 0
        self.error_count = 0
        self.last_optimization_time = time.time()
        
        logger.info(f"ProfitRoutingEngine initialized with dimensions: {spatial_dimensions}")

    def initialize_profit_space(self) -> None:
        """Initialize the 3D profit allocation space."""
        try:
            x_dim, y_dim, z_dim = self.spatial_dimensions
            logger.info(f"Initializing profit space: {x_dim}x{y_dim}x{z_dim}")

            for x in range(x_dim):
                for y in range(y_dim):
                    for z in range(z_dim):
                        node_id = f"profit_node_{x}_{y}_{z}"
                        position = (float(x), float(y), float(z))

                        # Create volume profile with validation
                        try:
                            volume_profile = VolumeProfile(
                                volume_id=f"vol_{node_id}",
                                base_volume=np.random.uniform(100, 10000),
                                delta_volume=0.0,
                                timestamp=time.time(),
                                price_level=np.random.uniform(30000, 70000),  # BTC price range
                                allocation_weight=1.0,
                            )
                        except Exception as e:
                            logger.error(f"Error creating volume profile for {node_id}: {e}")
                            continue

                        # Create profit node with validation
                        try:
                            profit_node = ProfitNode(
                                node_id=node_id,
                                position=position,
                                profit_value=Decimal("0.0"),
                                volume_profile=volume_profile,
                                recursive_depth=0,
                            )
                        except Exception as e:
                            logger.error(f"Error creating profit node {node_id}: {e}")
                            continue

                        # Connect to adjacent nodes
                        try:
                            profit_node.connections = self._get_adjacent_profit_nodes(x, y, z)
                        except Exception as e:
                            logger.error(f"Error connecting node {node_id}: {e}")
                            profit_node.connections = []

                        self.profit_nodes[node_id] = profit_node
                        self.volume_profiles[volume_profile.volume_id] = volume_profile

            # Initialize allocation chains
            self._initialize_allocation_chains()
            logger.info(f"Profit space initialized with {len(self.profit_nodes)} nodes")

        except Exception as e:
            logger.error(f"Error initializing profit space: {e}")
            raise

    def calculate_volumetric_profit(
        self,
        volume_deltas: List[Tuple[str, float]],
        price_tick: float,
        strategy: ProfitAllocationStrategy = ProfitAllocationStrategy.LINEAR,
    ) -> ProfitAllocationResult:
        """Calculate P_v = ∑ Δv × (R_n(t) · P_t) | Recursive volumetric profit."""
        try:
            # Input validation
            if not volume_deltas:
                logger.warning("Empty volume deltas provided")
                return self._create_empty_allocation_result()
            
            if price_tick <= 0:
                logger.warning(f"Invalid price tick: {price_tick}")
                return self._create_empty_allocation_result()

            total_profit = Decimal("0.0")
            volume_weighted_profit = Decimal("0.0")
            spatial_distribution = {}
            processed_volumes = 0

            for volume_id, delta_v in volume_deltas:
                try:
                    if volume_id not in self.volume_profiles:
                        logger.warning(f"Volume profile not found: {volume_id}")
                        continue

                    volume_profile = self.volume_profiles[volume_id]

                    # Update delta volume
                    volume_profile.delta_volume = delta_v
                    volume_profile.timestamp = time.time()

                    # Calculate R_n(t) - recursive multiplier using utility
                    recursive_multiplier = calculate_recursive_multiplier(
                        base_value=1.0,
                        depth=volume_profile.allocation_weight,
                        decay_factor=0.95,
                        max_depth=10
                    )

                    # Calculate profit component: Δv × (R_n(t) · P_t)
                    profit_component = safe_decimal_operation(
                        "multiply", delta_v, recursive_multiplier, price_tick
                    )

                    # Apply allocation strategy using utility
                    strategy_adjusted_profit = apply_allocation_strategy(
                        profit_component, 
                        strategy.value,
                        {"multiplier": volume_profile.allocation_weight}
                    )

                    total_profit = safe_decimal_operation("add", total_profit, strategy_adjusted_profit)
                    volume_weighted_profit = safe_decimal_operation(
                        "add", 
                        volume_weighted_profit, 
                        safe_decimal_operation("multiply", strategy_adjusted_profit, volume_profile.allocation_weight)
                    )

                    # Find associated profit node
                    associated_node = self._find_node_by_volume(volume_id)
                    if associated_node:
                        associated_node.profit_value = safe_decimal_operation(
                            "add", associated_node.profit_value, strategy_adjusted_profit
                        )
                        spatial_distribution[volume_id] = associated_node.position

                    processed_volumes += 1

                except Exception as e:
                    logger.error(f"Error processing volume {volume_id}: {e}")
                    self.error_count += 1
                    continue

            # Calculate allocation efficiency using utility
            allocation_efficiency = calculate_allocation_efficiency(volume_deltas)

            # Calculate recursive growth factor using utility
            profit_history_floats = [float(result.total_profit_allocated) for result in self.profit_history[-10:]]
            recursive_growth_factor = calculate_recursive_growth_factor(profit_history_floats)

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
            self.operation_count += 1

            logger.debug(f"Volumetric profit calculated: {processed_volumes} volumes processed")
            return result

        except Exception as e:
            logger.error(f"Error in calculate_volumetric_profit: {e}")
            self.error_count += 1
            return self._create_empty_allocation_result()

    def _create_empty_allocation_result(self) -> ProfitAllocationResult:
        """Create an empty allocation result for error cases."""
        return ProfitAllocationResult(
            allocation_id=f"empty_{int(time.time())}",
            total_profit_allocated=Decimal("0.0"),
            volume_weighted_profit=Decimal("0.0"),
            spatial_distribution={},
            allocation_efficiency=0.0,
            recursive_growth_factor=1.0,
            timestamp=time.time(),
        )

    def create_allocation_chain(
        self, chain_id: str, start_node_id: str, chain_length: int = 10
    ) -> List[str]:
        """Create a profit allocation chain through the 3D space."""
        try:
            if start_node_id not in self.profit_nodes:
                logger.warning(f"Start node not found: {start_node_id}")
                return []

            if chain_length <= 0:
                logger.warning(f"Invalid chain length: {chain_length}")
                return []

            chain = [start_node_id]
            current_node_id = start_node_id

            for _ in range(chain_length - 1):
                try:
                    current_node = self.profit_nodes[current_node_id]
                    next_node_id = self._find_best_next_node(current_node)
                    
                    if next_node_id and next_node_id not in chain:
                        chain.append(next_node_id)
                        current_node_id = next_node_id
                    else:
                        # No more valid connections
                        break
                        
                except Exception as e:
                    logger.error(f"Error extending chain from {current_node_id}: {e}")
                    break

            self.allocation_chains[chain_id] = chain
            logger.info(f"Created allocation chain {chain_id} with {len(chain)} nodes")
            return chain

        except Exception as e:
            logger.error(f"Error creating allocation chain {chain_id}: {e}")
            return []

    def measure_2d_profit_density(self, z_level: int = 0) -> np.ndarray:
        """Measure 2D profit density at a specific z-level."""
        try:
            x_dim, y_dim, z_dim = self.spatial_dimensions
            
            if z_level < 0 or z_level >= z_dim:
                logger.warning(f"Invalid z_level: {z_level}, using 0")
                z_level = 0

            density_map = np.zeros((x_dim, y_dim))

            for x in range(x_dim):
                for y in range(y_dim):
                    node_id = f"profit_node_{x}_{y}_{z_level}"
                    if node_id in self.profit_nodes:
                        density_map[x, y] = float(self.profit_nodes[node_id].profit_value)

            logger.debug(f"2D profit density measured at z_level {z_level}")
            return density_map

        except Exception as e:
            logger.error(f"Error measuring 2D profit density: {e}")
            return np.zeros(self.spatial_dimensions[:2])

    def measure_3d_profit_volume(self) -> Dict[str, Any]:
        """Measure 3D profit volume and spatial characteristics."""
        try:
            x_dim, y_dim, z_dim = self.spatial_dimensions
            volume_data = np.zeros(self.spatial_dimensions)

            # Populate 3D volume data
            for node_id, node in self.profit_nodes.items():
                x, y, z = node.position
                x_idx, y_idx, z_idx = int(x), int(y), int(z)
                if (0 <= x_idx < x_dim and 0 <= y_idx < y_dim and 0 <= z_idx < z_dim):
                    volume_data[x_idx, y_idx, z_idx] = float(node.profit_value)

            # Calculate spatial characteristics using utilities
            try:
                gradient = calculate_gradient(volume_data)
                centroid = calculate_centroid(volume_data)
            except Exception as e:
                logger.error(f"Error calculating spatial characteristics: {e}")
                gradient = np.zeros_like(volume_data)
                centroid = (x_dim/2, y_dim/2, z_dim/2)

            result = {
                "total_volume": float(np.sum(volume_data)),
                "max_density": float(np.max(volume_data)),
                "mean_density": float(np.mean(volume_data)),
                "gradient_magnitude": float(np.mean(gradient)),
                "centroid": centroid,
                "volume_shape": volume_data.shape,
                "timestamp": time.time()
            }

            logger.debug("3D profit volume measured successfully")
            return result

        except Exception as e:
            logger.error(f"Error measuring 3D profit volume: {e}")
            return {
                "total_volume": 0.0,
                "max_density": 0.0,
                "mean_density": 0.0,
                "gradient_magnitude": 0.0,
                "centroid": (0, 0, 0),
                "volume_shape": self.spatial_dimensions,
                "timestamp": time.time(),
                "error": str(e)
            }

    def _get_adjacent_profit_nodes(self, x: int, y: int, z: int) -> List[str]:
        """Get adjacent profit nodes in 3D space."""
        try:
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

        except Exception as e:
            logger.error(f"Error getting adjacent nodes for ({x}, {y}, {z}): {e}")
            return []

    def _initialize_allocation_chains(self) -> None:
        """Initialize default allocation chains."""
        try:
            # Create several default chains
            chain_count = min(10, len(self.profit_nodes) // 10)
            
            for i in range(chain_count):
                chain_id = f"default_chain_{i}"
                start_node_id = f"profit_node_{i*2}_{i*2}_{i*2}"
                
                if start_node_id in self.profit_nodes:
                    self.create_allocation_chain(chain_id, start_node_id, 5)

            logger.info(f"Initialized {chain_count} default allocation chains")

        except Exception as e:
            logger.error(f"Error initializing allocation chains: {e}")

    def _find_node_by_volume(self, volume_id: str) -> Optional[ProfitNode]:
        """Find profit node associated with a volume ID."""
        try:
            for node in self.profit_nodes.values():
                if node.volume_profile.volume_id == volume_id:
                    return node
            return None

        except Exception as e:
            logger.error(f"Error finding node by volume {volume_id}: {e}")
            return None

    def _find_best_next_node(self, current_node: ProfitNode) -> Optional[str]:
        """Find the best next node in the allocation chain."""
        try:
            best_node_id = None
            best_score = float('-inf')

            for connection_id in current_node.connections:
                if connection_id in self.profit_nodes:
                    connected_node = self.profit_nodes[connection_id]
                    
                    # Calculate distance score using utility
                    distance_score = calculate_distance_score(
                        current_node.position, connected_node.position
                    )
                    
                    # Combine distance and profit potential
                    profit_potential = float(connected_node.profit_value)
                    combined_score = profit_potential - distance_score * 0.1
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_node_id = connection_id

            return best_node_id

        except Exception as e:
            logger.error(f"Error finding best next node: {e}")
            return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the routing engine."""
        return {
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.operation_count, 1),
            "node_count": len(self.profit_nodes),
            "chain_count": len(self.allocation_chains),
            "history_size": len(self.profit_history),
            "uptime": time.time() - self.last_optimization_time,
        }


def create_profit_routing_system(
    dimensions: Tuple[int, int, int] = (10, 10, 10)
) -> ProfitRoutingEngine:
    """Create and initialize a profit routing system."""
    try:
        engine = ProfitRoutingEngine(dimensions)
        engine.initialize_profit_space()
        return engine
    except Exception as e:
        logger.error(f"Error creating profit routing system: {e}")
        raise


def simulate_profit_allocation(
    engine: ProfitRoutingEngine,
    simulation_steps: int = 100,
    base_profit_per_step: float = 1000.0,
) -> List[ProfitAllocationResult]:
    """Simulate profit allocation over multiple steps."""
    results = []
    
    try:
        for step in range(simulation_steps):
            # Generate random volume deltas
            volume_deltas = []
            for i in range(5):  # 5 volume changes per step
                volume_id = f"sim_vol_{step}_{i}"
                delta = np.random.normal(base_profit_per_step, base_profit_per_step * 0.2)
                volume_deltas.append((volume_id, delta))

            # Calculate profit allocation
            result = engine.calculate_volumetric_profit(
                volume_deltas=volume_deltas,
                price_tick=50000.0 + np.random.normal(0, 1000),
                strategy=ProfitAllocationStrategy.LINEAR
            )
            
            results.append(result)

        logger.info(f"Simulation completed: {len(results)} steps")
        return results

    except Exception as e:
        logger.error(f"Error in profit allocation simulation: {e}")
        return results
