"""Module for Schwabot trading system."""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..cpu_handlers import run_cpu_strategy
from ..gpu_handlers import run_gpu_strategy

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual State Router - Profit-Tiered CUDA Orchestration System

Implements ZPE (Zero Point, Efficiency) vs ZBE (Zero Bottleneck, Entropy) routing
based on profit density and strategy context, not just raw performance.

    Core Concept:
    - ZPE: CPU-based, low-latency, single-shot logic for short-term decisions
    - ZBE: CUDA-accelerated, batch-matrix, parallel strategy engines for mid/long-term

        The router learns which strategies "deserve" GPU based on:
        - Historical profit margins
        - Compute time vs profit ratio
        - Strategy tier (short/mid/long)
        - Current market conditions
        """

        # CUDA Helper Integration
            try:
            USING_CUDA,
            get_cuda_status,
            report_cuda_status,
            safe_cuda_operation,
            xp,
            )

            CUDA_AVAILABLE = True
            logger = logging.getLogger(__name__)
            logger.info("⚡ CUDA acceleration available for Dual State Router")
                except ImportError:
                xp = np
                USING_CUDA = False
                CUDA_AVAILABLE = False
                logger = logging.getLogger(__name__)
                logger.warning("🔄 CUDA not available - CPU-only mode for Dual State Router")

                logger = logging.getLogger(__name__)


                    class StrategyTier(Enum):
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """Strategy time tiers for routing decisions."""

                    SHORT = "short"  # <300ms, tick-level, high-urgency
                    MID = "mid"  # 300ms-2s, volume curves, pattern matching
                    LONG = "long"  # >2s, fractal sequences, timing overlays


                        class ComputeMode(Enum):
    """Class for Schwabot trading functionality."""
                        """Class for Schwabot trading functionality."""
                        """Available computation modes."""

                        ZPE = "zpe"  # CPU-based, Zero Point Efficiency
                        ZBE = "zbe"  # GPU-based, Zero Bottleneck Entropy


                        @ dataclass
                            class StrategyMetadata:
    """Class for Schwabot trading functionality."""
                            """Class for Schwabot trading functionality."""
                            """Metadata for strategy routing decisions."""

                            strategy_id: str
                            tier: StrategyTier
                            priority: float  # 0.0 to 1.0 (profit, density)
                            avg_compute_time_ms: float
                            avg_profit_margin: float
                            success_rate: float
                            last_execution: datetime
                            preferred_mode: ComputeMode
                            execution_count: int = 0
                            total_profit: float = 0.0

                                def __post_init__(self) -> None:
                                    if not (0.0 <= self.priority <= 1.0):
                                raise ValueError("Priority must be in [0.0, 1.0]")
                                    if not (0.0 <= self.success_rate <= 1.0):
                                raise ValueError("Success rate must be in [0.0, 1.0]")


                                @ dataclass
                                    class ExecutionResult:
    """Class for Schwabot trading functionality."""
                                    """Class for Schwabot trading functionality."""
                                    """Result of strategy execution with performance metrics."""

                                    strategy_id: str
                                    compute_mode: ComputeMode
                                    execution_time_ms: float
                                    profit_delta: float
                                    success: bool
                                    timestamp: datetime
                                    metadata: Dict[str, Any] = field(default_factory=dict)


                                    @ dataclass
                                        class ProfitRegistry:
    """Class for Schwabot trading functionality."""
                                        """Class for Schwabot trading functionality."""
                                        """Registry for tracking profit density and strategy performance."""

                                        strategies: Dict[str, StrategyMetadata] = field(default_factory=dict)
                                        execution_history: List[ExecutionResult] = field(default_factory=list)
                                        performance_window: int = 100  # Number of executions to track

                                            def update_strategy_performance(self, result: ExecutionResult) -> None:
                                            """Update strategy performance based on execution result."""
                                                if result.strategy_id not in self.strategies:
                                                # Create new strategy metadata
                                                self.strategies[result.strategy_id] = StrategyMetadata()
                                                strategy_id = result.strategy_id,
                                                tier = self._infer_tier(result.execution_time_ms),
                                                priority = 0.5,  # Default priority
                                                avg_compute_time_ms = result.execution_time_ms,
                                                avg_profit_margin = result.profit_delta,
                                                success_rate = 1.0 if result.success else 0.0,
                                                last_execution = result.timestamp,
                                                preferred_mode = result.compute_mode,
                                                execution_count = 1,
                                                total_profit = result.profit_delta,
                                                )
                                                    else:
                                                    # Update existing strategy
                                                    strategy = self.strategies[result.strategy_id]
                                                    strategy.execution_count += 1

                                                    # Update running averages
                                                    strategy.avg_compute_time_ms = ()
                                                    strategy.avg_compute_time_ms * (strategy.execution_count - 1) + result.execution_time_ms
                                                    ) / strategy.execution_count
                                                    strategy.avg_profit_margin = ()
                                                    strategy.avg_profit_margin * (strategy.execution_count - 1) + result.profit_delta
                                                    ) / strategy.execution_count

                                                    # Update success rate
                                                    total_successes = sum()
                                                    1 for r in self.execution_history if r.strategy_id == result.strategy_id and r.success
                                                    )
                                                    strategy.success_rate = total_successes / strategy.execution_count

                                                    # Update total profit
                                                    strategy.total_profit += result.profit_delta

                                                    # Update priority based on profit density
                                                    strategy.priority = self._calculate_profit_density(strategy)

                                                    # Update preferred mode based on performance
                                                    strategy.preferred_mode = self._determine_preferred_mode(strategy)
                                                    strategy.last_execution = result.timestamp

                                                    # Add to execution history
                                                    self.execution_history.append(result)

                                                    # Maintain history window
                                                        if len(self.execution_history) > self.performance_window:
                                                        self.execution_history.pop(0)

                                                            def get_strategy_tier(self, strategy_id: str) -> StrategyTier:
                                                            """Get strategy tier from registry."""
                                                                if strategy_id in self.strategies:
                                                            return self.strategies[strategy_id].tier
                                                        return StrategyTier.MID  # Default tier

                                                            def get_profit_density(self, strategy_id: str) -> float:
                                                            """Get profit density (priority) from registry."""
                                                                if strategy_id in self.strategies:
                                                            return self.strategies[strategy_id].priority
                                                        return 0.5  # Default priority

                                                            def get_strategy_metadata(self, strategy_id: str) -> Optional[StrategyMetadata]:
                                                            """Get full strategy metadata."""
                                                        return self.strategies.get(strategy_id)

                                                            def _infer_tier(self, compute_time_ms: float) -> StrategyTier:
                                                            """Infer strategy tier from compute time."""
                                                                if compute_time_ms < 300:
                                                            return StrategyTier.SHORT
                                                                elif compute_time_ms < 2000:
                                                            return StrategyTier.MID
                                                                else:
                                                            return StrategyTier.LONG

                                                                def _calculate_profit_density(self, strategy: StrategyMetadata) -> float:
                                                                """Calculate profit density (ROI per, millisecond)."""
                                                                    if strategy.avg_compute_time_ms <= 0:
                                                                return 0.0

                                                                # Profit density = profit margin / compute time (normalized)
                                                                profit_density = strategy.avg_profit_margin / (strategy.avg_compute_time_ms / 1000.0)

                                                                # Normalize to [0, 1] range
                                                                # Assume max 100% profit per second
                                                            return float(np.clip(profit_density / 100.0, 0.0, 1.0))

                                                                def _determine_preferred_mode(self, strategy: StrategyMetadata) -> ComputeMode:
                                                                """Determine preferred compute mode based on performance."""
                                                                # Get recent executions for this strategy
                                                                recent_executions = []
                                                                r for r in self.execution_history[-20:] if r.strategy_id == strategy.strategy_id  # Last 20 executions
                                                                ]

                                                                    if len(recent_executions) < 2:
                                                                return strategy.preferred_mode

                                                                # Compare ZPE vs ZBE performance
                                                                zpe_executions = [r for r in recent_executions if r.compute_mode == ComputeMode.ZPE]
                                                                zbe_executions = [r for r in recent_executions if r.compute_mode == ComputeMode.ZBE]

                                                                    if not zpe_executions or not zbe_executions:
                                                                return strategy.preferred_mode

                                                                # Calculate average performance for each mode
                                                                zpe_avg_time = np.mean([r.execution_time_ms for r in zpe_executions])
                                                                zpe_avg_profit = np.mean([r.profit_delta for r in zpe_executions])
                                                                zpe_efficiency = zpe_avg_profit / (zpe_avg_time / 1000.0) if zpe_avg_time > 0 else 0

                                                                zbe_avg_time = np.mean([r.execution_time_ms for r in zbe_executions])
                                                                zbe_avg_profit = np.mean([r.profit_delta for r in zbe_executions])
                                                                zbe_efficiency = zbe_avg_profit / (zbe_avg_time / 1000.0) if zbe_avg_time > 0 else 0

                                                                # Choose mode with better efficiency
                                                                if zbe_efficiency > zpe_efficiency * 1.2:  # 20% threshold for GPU preference
                                                            return ComputeMode.ZBE
                                                                else:
                                                            return ComputeMode.ZPE


                                                                class DualStateRouter:
    """Class for Schwabot trading functionality."""
                                                                """Class for Schwabot trading functionality."""
                                                                """
                                                                Dual State Router for profit-tiered CUDA orchestration.

                                                                    Routes calculations between ZPE (CPU) and ZBE (GPU) based on:
                                                                    - Strategy tier (short/mid/long)
                                                                    - Profit density (ROI per compute, time)
                                                                    - Historical performance
                                                                    - Current system load
                                                                    """

                                                                        def __init__(self) -> None:
                                                                        """Initialize the dual state router."""
                                                                        self.registry = ProfitRegistry()
                                                                        self.lock = threading.Lock()

                                                                        # Performance thresholds
                                                                        self.zpe_time_threshold = 300  # ms
                                                                        self.zbe_profit_threshold = 0.85  # profit density
                                                                        self.gpu_load_threshold = 0.8  # GPU utilization threshold

                                                                        # Adaptive parameters
                                                                        self.learning_rate = 0.1
                                                                        self.performance_window = 50

                                                                        # System load tracking
                                                                        self.gpu_load = 0.0
                                                                        self.cpu_load = 0.0
                                                                        self.last_load_update = datetime.now()

                                                                        logger.info("🔄 Dual State Router initialized with profit-tiered orchestration")

                                                                            def route(self, task_id: str, data: Dict[str, Any], force_mode: Optional[ComputeMode]= None) -> Dict[str, Any]:
                                                                            """
                                                                            Route task to appropriate compute mode (ZPE or, ZBE).

                                                                                Args:
                                                                                task_id: Unique strategy identifier
                                                                                data: Task data to process
                                                                                force_mode: Force specific compute mode (for, testing)

                                                                                    Returns:
                                                                                    Task result with performance metrics
                                                                                    """
                                                                                    start_time = time.time()

                                                                                        try:
                                                                                        # Determine compute mode
                                                                                            if force_mode:
                                                                                            compute_mode = force_mode
                                                                                                else:
                                                                                                compute_mode = self._determine_compute_mode(task_id, data)

                                                                                                # Execute task
                                                                                                    if compute_mode == ComputeMode.ZPE:
                                                                                                    result = self._run_zpe(task_id, data)
                                                                                                        else:
                                                                                                        result = self._run_zbe(task_id, data)

                                                                                                        # Calculate performance metrics
                                                                                                        execution_time_ms = (time.time() - start_time) * 1000

                                                                                                        # Create execution result
                                                                                                        execution_result = ExecutionResult()
                                                                                                        strategy_id = task_id,
                                                                                                        compute_mode = compute_mode,
                                                                                                        execution_time_ms = execution_time_ms,
                                                                                                        profit_delta = result.get("profit_delta", 0.0),
                                                                                                        success = result.get("success", True),
                                                                                                        timestamp = datetime.now(),
                                                                                                        metadata = {}
                                                                                                        "input_data_size": len(str(data)),
                                                                                                        "output_data_size": len(str(result)),
                                                                                                        "gpu_load": self.gpu_load,
                                                                                                        "cpu_load": self.cpu_load,
                                                                                                        },
                                                                                                        )

                                                                                                        # Update registry
                                                                                                            with self.lock:
                                                                                                            self.registry.update_strategy_performance(execution_result)

                                                                                                            # Add performance metrics to result
                                                                                                            result["execution_metrics"] = {}
                                                                                                            "compute_mode": compute_mode.value,
                                                                                                            "execution_time_ms": execution_time_ms,
                                                                                                            "strategy_tier": self.registry.get_strategy_tier(task_id).value,
                                                                                                            "profit_density": self.registry.get_profit_density(task_id),
                                                                                                            }

                                                                                                            logger.debug()
                                                                                                            "Routed {0} to {1} in {2}ms".format(task_id,)
                                                                                                            compute_mode.value,
                                                                                                            execution_time_ms)
                                                                                                            )
                                                                                                        return result

                                                                                                            except Exception as e:
                                                                                                            logger.error("Error routing task {0}: {1}".format(task_id, e))
                                                                                                            # Fallback to ZPE
                                                                                                        return self._run_zpe(task_id, data)

                                                                                                            def _determine_compute_mode(self, task_id: str, data: Dict[str, Any]) -> ComputeMode:
                                                                                                            """Determine optimal compute mode for task."""
                                                                                                            # Get strategy metadata
                                                                                                            metadata = self.registry.get_strategy_metadata(task_id)

                                                                                                            # Check if CUDA is available
                                                                                                                if not CUDA_AVAILABLE:
                                                                                                            return ComputeMode.ZPE

                                                                                                            # Check GPU load
                                                                                                                if self.gpu_load > self.gpu_load_threshold:
                                                                                                                logger.debug()
                                                                                                                "GPU load high ({0}), preferring ZPE for {1}".format(self.gpu_load)
                                                                                                                )
                                                                                                            return ComputeMode.ZPE

                                                                                                            # Get strategy tier and profit density
                                                                                                            tier = self.registry.get_strategy_tier(task_id)
                                                                                                            profit_density = self.registry.get_profit_density(task_id)

                                                                                                            # Decision logic based on tier and profit density
                                                                                                                if tier == StrategyTier.SHORT:
                                                                                                                # Short-term tasks: prefer ZPE for low latency
                                                                                                                    if profit_density > self.zbe_profit_threshold:
                                                                                                                return ComputeMode.ZBE  # High-value short tasks get GPU
                                                                                                                    else:
                                                                                                                return ComputeMode.ZPE

                                                                                                                    elif tier == StrategyTier.MID:
                                                                                                                    # Mid-term tasks: balance between latency and throughput
                                                                                                                    if profit_density > 0.7:  # High profit density
                                                                                                                return ComputeMode.ZBE
                                                                                                                elif profit_density < 0.3:  # Low profit density
                                                                                                            return ComputeMode.ZPE
                                                                                                                else:
                                                                                                                # Use preferred mode from registry
                                                                                                                    if metadata and metadata.preferred_mode:
                                                                                                                return metadata.preferred_mode
                                                                                                                    else:
                                                                                                                return ComputeMode.ZBE  # Default to GPU for mid-tier

                                                                                                                else:  # StrategyTier.LONG
                                                                                                                # Long-term tasks: prefer ZBE for complex calculations
                                                                                                                if profit_density > 0.5:  # Moderate profit density
                                                                                                            return ComputeMode.ZBE
                                                                                                                else:
                                                                                                            return ComputeMode.ZPE

                                                                                                                def _run_zpe(self, task_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
                                                                                                                """Execute task using ZPE (CPU-based, computation)."""
                                                                                                                    try:
                                                                                                                    # Import CPU handlers
                                                                                                                    # Execute CPU strategy
                                                                                                                    result = run_cpu_strategy(task_id, data)

                                                                                                                    # Add ZPE metadata
                                                                                                                    result["compute_mode"] = "zpe"
                                                                                                                    result["execution_engine"] = "cpu"

                                                                                                                return result

                                                                                                                    except ImportError:
                                                                                                                    # Fallback CPU execution
                                                                                                                    logger.warning("CPU handlers not available, using fallback for {0}".format(task_id))
                                                                                                                return self._fallback_cpu_execution(task_id, data)

                                                                                                                    def _run_zbe(self, task_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
                                                                                                                    """Execute task using ZBE (GPU-based, computation)."""
                                                                                                                        try:
                                                                                                                        # Import GPU handlers
                                                                                                                        # Execute GPU strategy
                                                                                                                        result = run_gpu_strategy(task_id, data)

                                                                                                                        # Add ZBE metadata
                                                                                                                        result["compute_mode"] = "zbe"
                                                                                                                        result["execution_engine"] = "gpu"

                                                                                                                    return result

                                                                                                                        except ImportError:
                                                                                                                        # Fallback to CPU if GPU handlers not available
                                                                                                                        logger.warning("GPU handlers not available, falling back to CPU for {0}".format(task_id))
                                                                                                                    return self._run_zpe(task_id, data)

                                                                                                                        def _fallback_cpu_execution(self, task_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
                                                                                                                        """Fallback CPU execution when handlers are not available."""
                                                                                                                        # Simple fallback implementation
                                                                                                                        result = {}
                                                                                                                        "task_id": task_id,
                                                                                                                        "result": "fallback_cpu_execution",
                                                                                                                        "compute_mode": "zpe",
                                                                                                                        "execution_engine": "cpu_fallback",
                                                                                                                        "success": True,
                                                                                                                        "profit_delta": 0.0,
                                                                                                                        "metadata": {"fallback": True},
                                                                                                                        }

                                                                                                                    return result

                                                                                                                        def update_system_load(self, gpu_load: float, cpu_load: float) -> None:
                                                                                                                        """Update system load metrics."""
                                                                                                                            with self.lock:
                                                                                                                            self.gpu_load = np.clip(gpu_load, 0.0, 1.0)
                                                                                                                            self.cpu_load = np.clip(cpu_load, 0.0, 1.0)
                                                                                                                            self.last_load_update = datetime.now()

                                                                                                                                def get_performance_summary(self) -> Dict[str, Any]:
                                                                                                                                """Get comprehensive performance summary."""
                                                                                                                                    with self.lock:
                                                                                                                                    total_executions = len(self.registry.execution_history)
                                                                                                                                        if total_executions == 0:
                                                                                                                                    return {"error": "No execution history available"}

                                                                                                                                    # Calculate mode distribution
                                                                                                                                    zpe_executions = [r for r in self.registry.execution_history if r.compute_mode == ComputeMode.ZPE]
                                                                                                                                    zbe_executions = [r for r in self.registry.execution_history if r.compute_mode == ComputeMode.ZBE]

                                                                                                                                    # Calculate performance metrics
                                                                                                                                    zpe_avg_time = np.mean([r.execution_time_ms for r in zpe_executions]) if zpe_executions else 0
                                                                                                                                    zbe_avg_time = np.mean([r.execution_time_ms for r in zbe_executions]) if zbe_executions else 0

                                                                                                                                    zpe_avg_profit = np.mean([r.profit_delta for r in zpe_executions]) if zpe_executions else 0
                                                                                                                                    zbe_avg_profit = np.mean([r.profit_delta for r in zbe_executions]) if zbe_executions else 0

                                                                                                                                return {}
                                                                                                                                "total_executions": total_executions,
                                                                                                                                "zpe_executions": len(zpe_executions),
                                                                                                                                "zbe_executions": len(zbe_executions),
                                                                                                                                "zpe_avg_time_ms": float(zpe_avg_time),
                                                                                                                                "zbe_avg_time_ms": float(zbe_avg_time),
                                                                                                                                "zpe_avg_profit": float(zpe_avg_profit),
                                                                                                                                "zbe_avg_profit": float(zbe_avg_profit),
                                                                                                                                "active_strategies": len(self.registry.strategies),
                                                                                                                                "gpu_load": self.gpu_load,
                                                                                                                                "cpu_load": self.cpu_load,
                                                                                                                                "cuda_available": CUDA_AVAILABLE,
                                                                                                                                "performance_ratio": float(zbe_avg_time / zpe_avg_time) if zpe_avg_time > 0 else 0,
                                                                                                                                }

                                                                                                                                    def get_strategy_analytics(self, strategy_id: str) -> Dict[str, Any]:
                                                                                                                                    """Get detailed analytics for a specific strategy."""
                                                                                                                                    metadata = self.registry.get_strategy_metadata(strategy_id)
                                                                                                                                        if not metadata:
                                                                                                                                    return {"error": "Strategy {0} not found".format(strategy_id)}

                                                                                                                                    # Get recent executions
                                                                                                                                    recent_executions = [r for r in self.registry.execution_history[-20:] if r.strategy_id == strategy_id]

                                                                                                                                return {}
                                                                                                                                "strategy_id": strategy_id,
                                                                                                                                "tier": metadata.tier.value,
                                                                                                                                "priority": metadata.priority,
                                                                                                                                "avg_compute_time_ms": metadata.avg_compute_time_ms,
                                                                                                                                "avg_profit_margin": metadata.avg_profit_margin,
                                                                                                                                "success_rate": metadata.success_rate,
                                                                                                                                "preferred_mode": metadata.preferred_mode.value,
                                                                                                                                "execution_count": metadata.execution_count,
                                                                                                                                "total_profit": metadata.total_profit,
                                                                                                                                "recent_executions": len(recent_executions),
                                                                                                                                "last_execution": metadata.last_execution.isoformat(),
                                                                                                                                }


                                                                                                                                # Global router instance
                                                                                                                                _router = None


                                                                                                                                    def get_dual_state_router() -> DualStateRouter:
                                                                                                                                    """Get global dual state router instance."""
                                                                                                                                    global _router
                                                                                                                                        if _router is None:
                                                                                                                                        _router = DualStateRouter()
                                                                                                                                    return _router


                                                                                                                                        def route_task(task_id: str, data: Dict[str, Any], force_mode: Optional[ComputeMode] = None) -> Dict[str, Any]:
                                                                                                                                        """Convenience function to route a task."""
                                                                                                                                        router = get_dual_state_router()
                                                                                                                                    return router.route(task_id, data, force_mode)


                                                                                                                                    # Export key classes and functions
                                                                                                                                    __all__ = []
                                                                                                                                    "DualStateRouter",
                                                                                                                                    "StrategyTier",
                                                                                                                                    "ComputeMode",
                                                                                                                                    "StrategyMetadata",
                                                                                                                                    "ExecutionResult",
                                                                                                                                    "ProfitRegistry",
                                                                                                                                    "get_dual_state_router",
                                                                                                                                    "route_task",
                                                                                                                                    ]
