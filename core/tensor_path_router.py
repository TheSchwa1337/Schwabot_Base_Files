# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Tensor Path Router - Schwabot UROS v1.0
=====================================

Tensor path routing system for hash prefix to basket to tensor path mapping.
Provides integration with voltage lane mapper and hash registry for optimal routing.

Mathematical Foundation:
- Hash Prefix → Basket Mapping: basket_id = hash_prefix % total_baskets
- Tensor Path Generation: tensor_path = f"{asset_from}_{asset_to}_{strategy_type}_{basket_id}"
- Voltage Lane Integration: voltage_level = f(bit_depth) → compute_channel
- Routing Score: score = (priority * voltage_compatibility * basket_availability)

Features:
- Hash prefix to basket routing
- Tensor path generation and validation
- Voltage lane integration
- Performance optimization
- Safety validation
"""

import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
# from core.unified_math_system import unified_math  # F811: duplicate import
import threading
import queue

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Routing strategy types."""
    PRIORITY_BASED = "priority_based"
    VOLTAGE_OPTIMIZED = "voltage_optimized"
    LOAD_BALANCED = "load_balanced"
    HYBRID = "hybrid"

class TensorPathType(Enum):
    """Tensor path types."""
    LONG = "long"
    SHORT = "short"
    MID = "mid"
    QUANTUM = "quantum"
    HYBRID = "hybrid"

@dataclass
class HashPrefixMapping:
    """Hash prefix mapping configuration."""
    hash_prefix: str
    basket_id: int
    tensor_path: str
    bit_depth: int
    priority: float
    voltage_level: str
    routing_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TensorPathRoute:
    """Tensor path route result."""
    route_id: str
    hash_prefix: str
    basket_id: int
    tensor_path: str
    asset_from: str
    asset_to: str
    strategy_type: TensorPathType
    bit_depth: int
    voltage_level: str
    compute_channel: str
    routing_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoutingRequest:
    """Routing request structure."""
    request_id: str
    hash_prefix: str
    bit_depth: int
    priority: float
    strategy: RoutingStrategy
    timestamp: datetime
    timeout: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoutingResult:
    """Routing result structure."""
    request_id: str
    success: bool
    route: Optional[TensorPathRoute] = None
    error_message: Optional[str] = None
    routing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TensorPathRouter:
    """
    Tensor Path Router for Schwabot UROS v1.0.

    Mathematical Foundation:
    - Hash Prefix → Basket: basket_id = hash_prefix % total_baskets
    - Tensor Path: tensor_path = f"{asset_from}_{asset_to}_{strategy_type}_{basket_id}"
    - Voltage Integration: voltage_level = f(bit_depth) → compute_channel
    - Routing Score: score = (priority * voltage_compatibility * basket_availability)
    """

    def __init__(self, hash_registry_manager=None, voltage_lane_mapper=None, config_path: str = "./config/tensor_path_config.json"):
        self.config_path = config_path

        # Core components
        self.hash_registry_manager = hash_registry_manager
        self.voltage_lane_mapper = voltage_lane_mapper

        # Routing configuration
        self.total_baskets = 32
        self.assets = ["BTC", "USDC", "XRP", "ETH", "SOL"]
        self.strategy_types = [TensorPathType.LONG, TensorPathType.SHORT, TensorPathType.MID, TensorPathType.QUANTUM]

        # Routing state
        self.hash_prefix_mappings: Dict[str, HashPrefixMapping] = {}
        self.tensor_path_routes: Dict[str, TensorPathRoute] = {}
        self.basket_availability: Dict[int, float] = {i: 1.0 for i in range(self.total_baskets)}

        # Performance tracking
        self.routing_requests: List[RoutingRequest] = []
        self.routing_results: List[RoutingResult] = []
        self.routing_stats: Dict[str, int] = {}

        # Threading for async operations
        self.routing_queue = queue.Queue()
        self.routing_thread = None
        self.routing_running = False

        # Load configuration
        self._load_configuration()
        self._initialize_routing_tables()
        self._start_routing_processor()

        logger.info("Tensor Path Router initialized")

    def _load_configuration(self) -> None:
        """Load tensor path configuration."""
        try:
            # Default configuration
            config = {
                "routing_parameters": {
                    "total_baskets": 32,
                    "default_timeout": 5.0,
                    "max_retries": 3
                },
                "asset_configuration": {
                    "assets": ["BTC", "USDC", "XRP", "ETH", "SOL"],
                    "strategy_types": ["long", "short", "mid", "quantum"]
                },
                "voltage_integration": {
                    "enabled": True,
                    "voltage_threshold": 0.1,
                    "channel_preference": ["tensor", "gpu", "cpu"]
                }
            }

            self.config = config

            # Update parameters from config
            self.total_baskets = config["routing_parameters"]["total_baskets"]
            self.assets = config["asset_configuration"]["assets"]
            self.strategy_types = [TensorPathType(s) for s in config["asset_configuration"]["strategy_types"]]

            logger.info("Tensor path configuration loaded")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def _initialize_routing_tables(self) -> None:
        """Initialize routing tables with hash prefix mappings."""
        try:
            # Generate hash prefix mappings for all combinations
            for i in range(self.total_baskets):
                # Generate hash prefix
                hash_prefix = f"hash_{i:02d}"

                # Determine bit depth (4, 8, or 42)
                if i % 3 == 0:
                    bit_depth = 4
                elif i % 3 == 1:
                    bit_depth = 8
                else:
                    bit_depth = 42

                # Determine voltage level
                if bit_depth == 4:
                    voltage_level = "low"
                elif bit_depth == 8:
                    voltage_level = "medium"
                else:
                    voltage_level = "high"

                # Generate tensor path
                asset_from = self.assets[i % len(self.assets)]
                asset_to = self.assets[(i + 1) % len(self.assets)]
                strategy_type = self.strategy_types[i % len(self.strategy_types)]
                tensor_path = f"{asset_from}_to_{asset_to}_{strategy_type.value}_{i}"

                # Calculate priority
                priority = 0.1 + (i * 0.1)

                # Calculate routing score
                routing_score = priority * (1.0 - (i / self.total_baskets))

                # Create mapping
                mapping = HashPrefixMapping(
                    hash_prefix=hash_prefix,
                    basket_id=i,
                    tensor_path=tensor_path,
                    bit_depth=bit_depth,
                    priority=priority,
                    voltage_level=voltage_level,
                    routing_score=routing_score,
                    timestamp=datetime.now()
                )

                self.hash_prefix_mappings[hash_prefix] = mapping

            logger.info(f"Initialized {len(self.hash_prefix_mappings)} hash prefix mappings")

        except Exception as e:
            logger.error(f"Error initializing routing tables: {e}")

    def _start_routing_processor(self) -> None:
        """Start the routing processing thread."""
        try:
            self.routing_running = True
            self.routing_thread = threading.Thread(target=self._process_routing, daemon=True)
            self.routing_thread.start()
            logger.info("Routing processor started")

        except Exception as e:
            logger.error(f"Error starting routing processor: {e}")

    def _process_routing(self) -> None:
        """Process routing queue in background thread."""
        while self.routing_running:
            try:
                # Get routing request from queue with timeout
                request = self.routing_queue.get(timeout=1.0)

                if request:
                    result = self._execute_routing(request)
                    self.routing_results.append(result)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing routing: {e}")

    def route_hash_prefix(self, hash_prefix: str, bit_depth: int = None,
                         priority: float = 1.0, strategy: RoutingStrategy = RoutingStrategy.PRIORITY_BASED) -> str:
        """
        Route hash prefix to tensor path.

        Parameters:
        -----------
        hash_prefix : str
            Hash prefix to route
        bit_depth : int
            Bit depth (4, 8, or 42)
        priority : float
            Routing priority (0.1 to 3.2)
        strategy : RoutingStrategy
            Routing strategy

        Returns:
        --------
        str
            Routing request ID
        """
        try:
            # Create routing request
            request_id = f"route_{int(time.time() * 1000)}"
            request = RoutingRequest(
                request_id=request_id,
                hash_prefix=hash_prefix,
                bit_depth=bit_depth,
                priority=priority,
                strategy=strategy,
                timestamp=datetime.now(),
                timeout=self.config["routing_parameters"]["default_timeout"]
            )

            self.routing_requests.append(request)

            # Queue for processing
            self.routing_queue.put(request)

            logger.info(f"Routing request {request_id} queued for hash prefix {hash_prefix}")

            return request_id

        except Exception as e:
            logger.error(f"Error requesting routing: {e}")
            raise

    def _execute_routing(self, request: RoutingRequest) -> RoutingResult:
        """
        Execute routing operation.

        Parameters:
        -----------
        request : RoutingRequest
            Routing request

        Returns:
        --------
        RoutingResult
            Routing result
        """
        try:
            start_time = time.time()

            # Get hash prefix mapping
            mapping = self.hash_prefix_mappings.get(request.hash_prefix)
            if not mapping:
                return RoutingResult(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"Hash prefix {request.hash_prefix} not found in routing table"
                )

            # Update bit depth if provided
            if request.bit_depth:
                mapping.bit_depth = request.bit_depth
                # Recalculate voltage level
                if request.bit_depth == 4:
                    mapping.voltage_level = "low"
                elif request.bit_depth == 8:
                    mapping.voltage_level = "medium"
                else:
                    mapping.voltage_level = "high"

            # Update priority if provided
            if request.priority:
                mapping.priority = request.priority

            # Get compute channel from voltage lane mapper
            compute_channel = "cpu"  # Default
            if self.voltage_lane_mapper:
                try:
                    voltage_mapping = self.voltage_lane_mapper.calculate_voltage_for_bit_depth(mapping.bit_depth)
                    channel_assignment = self.voltage_lane_mapper.assign_channel_for_voltage(voltage_mapping, mapping.priority)
                    compute_channel = channel_assignment.channel_id
                except Exception as e:
                    logger.warning(f"Voltage lane mapping failed: {e}, using default channel")

            # Parse tensor path
            path_parts = mapping.tensor_path.split('_')
            if len(path_parts) >= 4:
                asset_from = path_parts[0]
                asset_to = path_parts[2]
                strategy_type = TensorPathType(path_parts[3])
            else:
                asset_from = "BTC"
                asset_to = "USDC"
                strategy_type = TensorPathType.LONG

            # Calculate routing score based on strategy
            routing_score = self._calculate_routing_score(mapping, request.strategy, compute_channel)

            # Create tensor path route
            route = TensorPathRoute(
                route_id=f"route_{mapping.basket_id}_{int(time.time() * 1000)}",
                hash_prefix=mapping.hash_prefix,
                basket_id=mapping.basket_id,
                tensor_path=mapping.tensor_path,
                asset_from=asset_from,
                asset_to=asset_to,
                strategy_type=strategy_type,
                bit_depth=mapping.bit_depth,
                voltage_level=mapping.voltage_level,
                compute_channel=compute_channel,
                routing_score=routing_score,
                timestamp=datetime.now()
            )

            # Store route
            self.tensor_path_routes[route.route_id] = route

            # Update basket availability
            self.basket_availability[mapping.basket_id] = unified_math.max(0.0,
                self.basket_availability[mapping.basket_id] - 0.1)

            # Success result
            result = RoutingResult(
                request_id=request.request_id,
                success=True,
                route=route,
                routing_time=time.time() - start_time
            )

            logger.info(f"Routing {request.request_id} successful: {mapping.hash_prefix} → {mapping.tensor_path}")

            return result

        except Exception as e:
            logger.error(f"Error executing routing {request.request_id}: {e}")
            return RoutingResult(
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )

    def _calculate_routing_score(self, mapping: HashPrefixMapping, strategy: RoutingStrategy,
                                compute_channel: str) -> float:
        """
        Calculate routing score based on strategy.

        Parameters:
        -----------
        mapping : HashPrefixMapping
            Hash prefix mapping
        strategy : RoutingStrategy
            Routing strategy
        compute_channel : str
            Assigned compute channel

        Returns:
        --------
        float
            Routing score
        """
        try:
            base_score = mapping.priority

            if strategy == RoutingStrategy.PRIORITY_BASED:
                return base_score

            elif strategy == RoutingStrategy.VOLTAGE_OPTIMIZED:
                # Factor in voltage compatibility
                voltage_compatibility = 1.0
                if self.voltage_lane_mapper:
                    try:
                        voltage_mapping = self.voltage_lane_mapper.calculate_voltage_for_bit_depth(mapping.bit_depth)
                        voltage_compatibility = voltage_mapping.safety_margin
                    except:
                        pass
                return base_score * voltage_compatibility

            elif strategy == RoutingStrategy.LOAD_BALANCED:
                # Factor in basket availability
                basket_availability = self.basket_availability.get(mapping.basket_id, 1.0)
                return base_score * basket_availability

            elif strategy == RoutingStrategy.HYBRID:
                # Combine all factors
                voltage_compatibility = 1.0
                if self.voltage_lane_mapper:
                    try:
                        voltage_mapping = self.voltage_lane_mapper.calculate_voltage_for_bit_depth(mapping.bit_depth)
                        voltage_compatibility = voltage_mapping.safety_margin
                    except:
                        pass

                basket_availability = self.basket_availability.get(mapping.basket_id, 1.0)
                channel_efficiency = 1.0 if compute_channel == "tensor" else 0.8

                return base_score * voltage_compatibility * basket_availability * channel_efficiency

            return base_score

        except Exception as e:
            logger.error(f"Error calculating routing score: {e}")
            return mapping.priority

    def get_routing_status(self, request_id: str) -> Optional[RoutingResult]:
        """
        Get routing status by request ID.

        Parameters:
        -----------
        request_id : str
            Routing request ID

        Returns:
        --------
        Optional[RoutingResult]
            Routing result if found
        """
        for result in self.routing_results:
            if result.request_id == request_id:
                return result
        return None

    def get_tensor_path_route(self, route_id: str) -> Optional[TensorPathRoute]:
        """
        Get tensor path route by route ID.

        Parameters:
        -----------
        route_id : str
            Route ID

        Returns:
        --------
        Optional[TensorPathRoute]
            Tensor path route if found
        """
        return self.tensor_path_routes.get(route_id)

    def get_routes_by_hash_prefix(self, hash_prefix: str) -> List[TensorPathRoute]:
        """
        Get all routes for a hash prefix.

        Parameters:
        -----------
        hash_prefix : str
            Hash prefix

        Returns:
        --------
        List[TensorPathRoute]
            List of tensor path routes
        """
        return [route for route in self.tensor_path_routes.values() if route.hash_prefix == hash_prefix]

    def get_routes_by_asset_pair(self, asset_from: str, asset_to: str) -> List[TensorPathRoute]:
        """
        Get all routes for an asset pair.

        Parameters:
        -----------
        asset_from : str
            Source asset
        asset_to : str
            Target asset

        Returns:
        --------
        List[TensorPathRoute]
            List of tensor path routes
        """
        return [route for route in self.tensor_path_routes.values()
                if route.asset_from == asset_from and route.asset_to == asset_to]

    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get routing statistics.

        Returns:
        --------
        Dict[str, Any]
            Routing statistics
        """
        try:
            stats = {
                "total_mappings": len(self.hash_prefix_mappings),
                "total_routes": len(self.tensor_path_routes),
                "total_requests": len(self.routing_requests),
                "successful_routes": len([r for r in self.routing_results if r.success]),
                "failed_routes": len([r for r in self.routing_results if not r.success]),
                "average_routing_time": unified_math.mean([r.routing_time for r in self.routing_results]) if self.routing_results else 0.0,
                "basket_availability": self.basket_availability.copy(),
                "asset_distribution": {},
                "strategy_distribution": {}
            }

            # Calculate asset distribution
            for route in self.tensor_path_routes.values():
                asset_pair = f"{route.asset_from}_{route.asset_to}"
                stats["asset_distribution"][asset_pair] = stats["asset_distribution"].get(asset_pair, 0) + 1

                strategy = route.strategy_type.value
                stats["strategy_distribution"][strategy] = stats["strategy_distribution"].get(strategy, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Error getting routing statistics: {e}")
            return {}

    def export_routing_data(self, output_path: str = "tensor_path_routing_data.json") -> None:
        """
        Export tensor path routing data.

        Parameters:
        -----------
        output_path : str
            Output file path
        """
        try:
            data = {
                "hash_prefix_mappings": [
                    {
                        "hash_prefix": m.hash_prefix,
                        "basket_id": m.basket_id,
                        "tensor_path": m.tensor_path,
                        "bit_depth": m.bit_depth,
                        "priority": m.priority,
                        "voltage_level": m.voltage_level,
                        "routing_score": m.routing_score,
                        "timestamp": m.timestamp.isoformat()
                    }
                    for m in self.hash_prefix_mappings.values()
                ],
                "tensor_path_routes": [
                    {
                        "route_id": r.route_id,
                        "hash_prefix": r.hash_prefix,
                        "basket_id": r.basket_id,
                        "tensor_path": r.tensor_path,
                        "asset_from": r.asset_from,
                        "asset_to": r.asset_to,
                        "strategy_type": r.strategy_type.value,
                        "bit_depth": r.bit_depth,
                        "voltage_level": r.voltage_level,
                        "compute_channel": r.compute_channel,
                        "routing_score": r.routing_score,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in self.tensor_path_routes.values()
                ],
                "routing_results": [
                    {
                        "request_id": r.request_id,
                        "success": r.success,
                        "routing_time": r.routing_time,
                        "error_message": r.error_message,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in self.routing_results
                ],
                "statistics": self.get_routing_statistics()
            }

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Tensor path routing data exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting routing data: {e}")

def main():
    """Main function for testing tensor path router."""
    try:
        # Initialize tensor path router
        router = TensorPathRouter()

        # Test routing for different hash prefixes
        test_prefixes = ["hash_00", "hash_15", "hash_31"]

        for prefix in test_prefixes:
            request_id = router.route_hash_prefix(prefix, bit_depth=8, priority=2.0)
            safe_print(f"Routing request: {request_id} for {prefix}")

        # Wait for routing completion
        time.sleep(2)

        # Check routing results
        for prefix in test_prefixes:
            routes = router.get_routes_by_hash_prefix(prefix)
            for route in routes:
                safe_print(f"Route: {route.tensor_path} (score: {route.routing_score:.3f})")

        # Export data
        router.export_routing_data()

        # Print statistics
        stats = router.get_routing_statistics()
        safe_print(f"Routing statistics: {stats}")

    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
