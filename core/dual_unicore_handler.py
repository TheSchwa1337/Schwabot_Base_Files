# -*- coding: utf-8 -*-
"""
Dual Unicore Handler
===================

Advanced dual-core processing system for mathematical operations and trading calculations.
Provides parallel processing capabilities, load balancing, and mathematical optimization
across dual processing cores.

Mathematical Foundation:
    - Dual Core Processing: P(x) = Core₁(x₁) ⊕ Core₂(x₂) where x = x₁ union x₂
    - Load Balancing: L(c₁,c₂) = min(load₁, load₂) / max(load₁, load₂)
    - Parallel Efficiency: E = (T_sequential / T_parallel) / N_cores
    - Mathematical Synchronization: S(t) = sync(Core₁_result, Core₂_result)
    - Error Correction: EC = verify(Core₁) and verify(Core₂) -> consensus
"""

import asyncio
import concurrent.futures
import hashlib
import logging
import math
import multiprocessing
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class CoreType(Enum):
    """Types of processing cores."""

    MATHEMATICAL = "mathematical"  # Mathematical computation core
    LOGICAL = "logical"  # Logical decision core
    ANALYTICAL = "analytical"  # Market analysis core
    OPTIMIZATION = "optimization"  # Portfolio optimization core
    RISK_MANAGEMENT = "risk_management"  # Risk calculation core
    EXECUTION = "execution"  # Trade execution core


class ProcessingMode(Enum):
    """Processing modes for dual core operations."""

    PARALLEL = "parallel"  # Full parallel processing
    SEQUENTIAL = "sequential"  # Sequential processing
    HYBRID = "hybrid"  # Mixed parallel/sequential
    REDUNDANT = "redundant"  # Redundant processing for verification
    LOAD_BALANCED = "load_balanced"  # Dynamic load balancing


@dataclass
class CoreStatus:
    """Status information for a processing core."""

    core_id: str
    core_type: CoreType
    is_active: bool = True
    current_load: float = 0.0
    total_operations: int = 0
    successful_operations: int = 0
    average_processing_time: float = 0.0
    error_count: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingTask:
    """Represents a task for dual core processing."""

    task_id: str
    function: Callable
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    requires_verification: bool = False
    mathematical_signature: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Generate mathematical signature for task."""
        if not self.mathematical_signature:
            task_data = f"{self.function.__name__}_{self.args}_{self.kwargs}"
            self.mathematical_signature = hashlib.sha256(task_data.encode()).hexdigest()[:12]


@dataclass
class ProcessingResult:
    """Result from dual core processing."""

    task_id: str
    core_results: Dict[str, Any]  # Results from each core
    consensus_result: Any
    processing_time: float
    verification_passed: bool
    load_distribution: Dict[str, float]
    mathematical_signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DualUnicoreHandler:
    """
    Advanced dual-core processing system for mathematical operations.

    Provides parallel processing, load balancing, and mathematical verification
    across dual processing cores with automatic error correction and optimization.
    """

    def __init__(self, max_workers: int = 2):
        """Initialize the dual unicore handler."""
        self.max_workers = min(max_workers, multiprocessing.cpu_count())
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

        # Core management
        self.cores: Dict[str, CoreStatus] = {}
        self.processing_queue: List[ProcessingTask] = []
        self.processing_history: List[ProcessingResult] = []

        # Initialize cores
        self._initialize_cores()

        # Performance metrics
        self.system_metrics = {
            "total_tasks_processed": 0,
            "successful_operations": 0,
            "parallel_efficiency": 0.0,
            "load_balance_ratio": 1.0,
            "error_rate": 0.0,
            "average_processing_time": 0.0,
        }

        # Threading for heartbeat monitoring
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self._running = True
        self._heartbeat_thread.start()

        logger.info(f"Dual Unicore Handler initialized with {self.max_workers} cores")

    def process_task(self, task: ProcessingTask, mode: ProcessingMode = ProcessingMode.PARALLEL) -> ProcessingResult:
        """
        Process a task using dual core architecture.

        Args:
            task: ProcessingTask to execute
            mode: Processing mode (parallel, sequential, etc.)

        Returns:
            ProcessingResult with core results and consensus
        """
        start_time = time.time()
        self.system_metrics["total_tasks_processed"] += 1

        try:
            if mode == ProcessingMode.PARALLEL:
                result = self._process_parallel(task)
            elif mode == ProcessingMode.SEQUENTIAL:
                result = self._process_sequential(task)
            elif mode == ProcessingMode.HYBRID:
                result = self._process_hybrid(task)
            elif mode == ProcessingMode.REDUNDANT:
                result = self._process_redundant(task)
            else:  # LOAD_BALANCED
                result = self._process_load_balanced(task)

            # Update processing time
            result.processing_time = time.time() - start_time

            # Update metrics
            self._update_system_metrics(result)

            # Store in history
            self.processing_history.append(result)

            # Keep history manageable
            if len(self.processing_history) > 1000:
                self.processing_history = self.processing_history[-500:]

            logger.debug(f"Task {task.task_id} processed in {result.processing_time:.6f}s")
            return result

        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            return self._create_fallback_result(task, start_time)

    def execute_mathematical_operation(
        self, operation: str, data: Union[Dict, List, np.ndarray], **kwargs
    ) -> Dict[str, Any]:
        """
        Execute mathematical operations using dual core processing.

        Args:
            operation: Mathematical operation name
            data: Data to process
            **kwargs: Additional operation parameters

        Returns:
            Mathematical operation results with verification
        """
        try:
            # Create processing task
            task = ProcessingTask(
                task_id=f"math_{operation}_{int(time.time() * 1000)}",
                function=self._execute_math_operation,
                args=(operation, data),
                kwargs=kwargs,
                requires_verification=True,
            )

            # Process with redundant mode for mathematical accuracy
            result = self.process_task(task, ProcessingMode.REDUNDANT)

            return {
                "operation": operation,
                "result": result.consensus_result,
                "verification_passed": result.verification_passed,
                "processing_time": result.processing_time,
                "mathematical_signature": result.mathematical_signature,
                "core_agreement": self._calculate_core_agreement(result.core_results),
            }

        except Exception as e:
            logger.error(f"Mathematical operation {operation} failed: {e}")
            return {"operation": operation, "result": None, "error": str(e), "verification_passed": False}

    def optimize_load_distribution(self) -> Dict[str, Any]:
        """
        Optimize load distribution across cores.

        Returns:
            Load optimization results and recommendations
        """
        try:
            # Analyze current core loads
            core_loads = {}
            for core_id, core_status in self.cores.items():
                core_loads[core_id] = core_status.current_load

            # Calculate load balance metrics
            loads = list(core_loads.values())
            if loads:
                min_load = min(loads)
                max_load = max(loads)
                balance_ratio = min_load / max(max_load, 0.001)
            else:
                balance_ratio = 1.0

            # Generate optimization recommendations
            recommendations = []

            if balance_ratio < 0.7:  # Significant imbalance
                overloaded_cores = [core_id for core_id, load in core_loads.items() if load > np.mean(loads) * 1.2]
                underutilized_cores = [core_id for core_id, load in core_loads.items() if load < np.mean(loads) * 0.8]

                recommendations.append(
                    {
                        "action": "redistribute_load",
                        "overloaded_cores": overloaded_cores,
                        "underutilized_cores": underutilized_cores,
                        "priority": "high",
                    }
                )

            # Update system load balance ratio
            self.system_metrics["load_balance_ratio"] = balance_ratio

            return {
                "current_loads": core_loads,
                "balance_ratio": balance_ratio,
                "recommendations": recommendations,
                "optimization_timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Load optimization failed: {e}")
            return {"error": str(e)}

    def get_core_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive core diagnostics and health status."""
        diagnostics = {"cores": {}, "system_health": "unknown", "performance_summary": {}, "recommendations": []}

        try:
            # Core-specific diagnostics
            healthy_cores = 0
            total_cores = len(self.cores)

            for core_id, core_status in self.cores.items():
                # Calculate core health score
                health_score = self._calculate_core_health(core_status)

                diagnostics["cores"][core_id] = {
                    "type": core_status.core_type.value,
                    "active": core_status.is_active,
                    "load": core_status.current_load,
                    "operations": core_status.total_operations,
                    "success_rate": (core_status.successful_operations / max(core_status.total_operations, 1)),
                    "avg_processing_time": core_status.average_processing_time,
                    "error_count": core_status.error_count,
                    "health_score": health_score,
                    "last_heartbeat": time.time() - core_status.last_heartbeat,
                }

                if health_score > 0.7:
                    healthy_cores += 1

            # Overall system health
            if total_cores > 0:
                health_ratio = healthy_cores / total_cores
                if health_ratio > 0.8:
                    diagnostics["system_health"] = "excellent"
                elif health_ratio > 0.6:
                    diagnostics["system_health"] = "good"
                elif health_ratio > 0.4:
                    diagnostics["system_health"] = "fair"
                else:
                    diagnostics["system_health"] = "poor"

            # Performance summary
            diagnostics["performance_summary"] = self.system_metrics.copy()

            # Generate recommendations
            if self.system_metrics["error_rate"] > 0.05:
                diagnostics["recommendations"].append(
                    {
                        "issue": "high_error_rate",
                        "description": f"Error rate is {self.system_metrics['error_rate']:.2%}",
                        "action": "investigate_error_sources",
                    }
                )

            if self.system_metrics["load_balance_ratio"] < 0.8:
                diagnostics["recommendations"].append(
                    {
                        "issue": "load_imbalance",
                        "description": f"Load balance ratio is {self.system_metrics['load_balance_ratio']:.2f}",
                        "action": "optimize_load_distribution",
                    }
                )

            return diagnostics

        except Exception as e:
            logger.error(f"Core diagnostics failed: {e}")
            diagnostics["error"] = str(e)
            return diagnostics

    def shutdown(self):
        """Gracefully shutdown the dual unicore handler."""
        try:
            logger.info("Shutting down Dual Unicore Handler...")

            self._running = False

            # Wait for heartbeat thread
            if self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=2.0)

            # Shutdown executor
            self.executor.shutdown(wait=True)

            # Mark all cores as inactive
            for core_status in self.cores.values():
                core_status.is_active = False

            logger.info("Dual Unicore Handler shutdown complete")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    # Internal implementation methods

    def _initialize_cores(self):
        """Initialize processing cores."""
        core_types = [CoreType.MATHEMATICAL, CoreType.LOGICAL]

        for i, core_type in enumerate(core_types[: self.max_workers]):
            core_id = f"core_{i}_{core_type.value}"

            self.cores[core_id] = CoreStatus(core_id=core_id, core_type=core_type, is_active=True)

        logger.info(f"Initialized {len(self.cores)} processing cores")

    def _process_parallel(self, task: ProcessingTask) -> ProcessingResult:
        """Process task in parallel across available cores."""
        futures = {}
        available_cores = [core_id for core_id, status in self.cores.items() if status.is_active]

        # Submit task to available cores
        for core_id in available_cores:
            future = self.executor.submit(self._execute_task_on_core, task, core_id)
            futures[core_id] = future

        # Collect results
        core_results = {}
        for core_id, future in futures.items():
            try:
                result = future.result(timeout=30.0)  # 30 second timeout
                core_results[core_id] = result
            except Exception as e:
                logger.error(f"Core {core_id} failed: {e}")
                core_results[core_id] = {"error": str(e)}

        # Generate consensus result
        consensus_result = self._generate_consensus(core_results)

        # Verify results if required
        verification_passed = True
        if task.requires_verification:
            verification_passed = self._verify_results(core_results)

        return ProcessingResult(
            task_id=task.task_id,
            core_results=core_results,
            consensus_result=consensus_result,
            processing_time=0.0,  # Will be set by caller
            verification_passed=verification_passed,
            load_distribution=self._calculate_load_distribution(),
            mathematical_signature=task.mathematical_signature,
        )

    def _process_sequential(self, task: ProcessingTask) -> ProcessingResult:
        """Process task sequentially across cores."""
        core_results = {}
        available_cores = [core_id for core_id, status in self.cores.items() if status.is_active]

        # Process on each core sequentially
        for core_id in available_cores:
            try:
                result = self._execute_task_on_core(task, core_id)
                core_results[core_id] = result
            except Exception as e:
                logger.error(f"Sequential processing on {core_id} failed: {e}")
                core_results[core_id] = {"error": str(e)}

        consensus_result = self._generate_consensus(core_results)
        verification_passed = True
        if task.requires_verification:
            verification_passed = self._verify_results(core_results)

        return ProcessingResult(
            task_id=task.task_id,
            core_results=core_results,
            consensus_result=consensus_result,
            processing_time=0.0,
            verification_passed=verification_passed,
            load_distribution=self._calculate_load_distribution(),
            mathematical_signature=task.mathematical_signature,
        )

    def _process_hybrid(self, task: ProcessingTask) -> ProcessingResult:
        """Process task using hybrid parallel/sequential approach."""
        # Use parallel for mathematical cores, sequential for others
        math_cores = [
            core_id
            for core_id, status in self.cores.items()
            if status.core_type == CoreType.MATHEMATICAL and status.is_active
        ]

        other_cores = [
            core_id
            for core_id, status in self.cores.items()
            if status.core_type != CoreType.MATHEMATICAL and status.is_active
        ]

        core_results = {}

        # Parallel processing for mathematical cores
        if math_cores:
            futures = {}
            for core_id in math_cores:
                future = self.executor.submit(self._execute_task_on_core, task, core_id)
                futures[core_id] = future

            for core_id, future in futures.items():
                try:
                    result = future.result(timeout=30.0)
                    core_results[core_id] = result
                except Exception as e:
                    core_results[core_id] = {"error": str(e)}

        # Sequential processing for other cores
        for core_id in other_cores:
            try:
                result = self._execute_task_on_core(task, core_id)
                core_results[core_id] = result
            except Exception as e:
                core_results[core_id] = {"error": str(e)}

        consensus_result = self._generate_consensus(core_results)
        verification_passed = True
        if task.requires_verification:
            verification_passed = self._verify_results(core_results)

        return ProcessingResult(
            task_id=task.task_id,
            core_results=core_results,
            consensus_result=consensus_result,
            processing_time=0.0,
            verification_passed=verification_passed,
            load_distribution=self._calculate_load_distribution(),
            mathematical_signature=task.mathematical_signature,
        )

    def _process_redundant(self, task: ProcessingTask) -> ProcessingResult:
        """Process task with redundancy for verification."""
        # Execute on all available cores for maximum verification
        return self._process_parallel(task)

    def _process_load_balanced(self, task: ProcessingTask) -> ProcessingResult:
        """Process task with dynamic load balancing."""
        # Select cores with lowest current load
        available_cores = [(core_id, status.current_load) for core_id, status in self.cores.items() if status.is_active]

        # Sort by load (ascending)
        available_cores.sort(key=lambda x: x[1])

        # Use top 2 least loaded cores
        selected_cores = [core_id for core_id, _ in available_cores[:2]]

        core_results = {}

        # Process on selected cores
        futures = {}
        for core_id in selected_cores:
            future = self.executor.submit(self._execute_task_on_core, task, core_id)
            futures[core_id] = future

        for core_id, future in futures.items():
            try:
                result = future.result(timeout=30.0)
                core_results[core_id] = result
            except Exception as e:
                core_results[core_id] = {"error": str(e)}

        consensus_result = self._generate_consensus(core_results)
        verification_passed = True
        if task.requires_verification:
            verification_passed = self._verify_results(core_results)

        return ProcessingResult(
            task_id=task.task_id,
            core_results=core_results,
            consensus_result=consensus_result,
            processing_time=0.0,
            verification_passed=verification_passed,
            load_distribution=self._calculate_load_distribution(),
            mathematical_signature=task.mathematical_signature,
        )

    def _execute_task_on_core(self, task: ProcessingTask, core_id: str) -> Any:
        """Execute a task on a specific core."""
        try:
            core_status = self.cores[core_id]

            # Update core load
            core_status.current_load += 0.1
            core_status.total_operations += 1

            start_time = time.time()

            # Execute the task
            result = task.function(*task.args, **task.kwargs)

            # Update core statistics
            processing_time = time.time() - start_time
            core_status.average_processing_time = (
                core_status.average_processing_time * (core_status.successful_operations) + processing_time
            ) / (core_status.successful_operations + 1)
            core_status.successful_operations += 1
            core_status.current_load = max(0.0, core_status.current_load - 0.1)
            core_status.last_heartbeat = time.time()

            return result

        except Exception as e:
            # Update error statistics
            if core_id in self.cores:
                self.cores[core_id].error_count += 1
                self.cores[core_id].current_load = max(0.0, self.cores[core_id].current_load - 0.1)

            logger.error(f"Task execution failed on {core_id}: {e}")
            raise

    def _execute_math_operation(self, operation: str, data: Any, **kwargs) -> Any:
        """Execute mathematical operations."""
        try:
            if operation == "add":
                if isinstance(data, (list, np.ndarray)):
                    return np.sum(data)
                else:
                    return sum(data.values()) if isinstance(data, dict) else data

            elif operation == "multiply":
                if isinstance(data, (list, np.ndarray)):
                    return np.prod(data)
                else:
                    result = 1
                    for v in data.values() if isinstance(data, dict) else [data]:
                        result *= v
                    return result

            elif operation == "mean":
                if isinstance(data, (list, np.ndarray)):
                    return np.mean(data)
                else:
                    values = list(data.values()) if isinstance(data, dict) else [data]
                    return sum(values) / len(values)

            elif operation == "std":
                if isinstance(data, (list, np.ndarray)):
                    return np.std(data)
                else:
                    values = list(data.values()) if isinstance(data, dict) else [data]
                    return np.std(values)

            elif operation == "correlation":
                matrix_size = kwargs.get("size", 3)
                if isinstance(data, (list, np.ndarray)) and len(data) >= matrix_size:
                    # Simple correlation calculation
                    return np.corrcoef(data[:matrix_size])
                return np.eye(matrix_size)  # Identity matrix fallback

            elif operation == "optimization":
                # Simple optimization (maximize sum with constraints)
                if isinstance(data, (list, np.ndarray)):
                    weights = np.array(data) / np.sum(np.abs(data))
                    return weights.tolist()
                return [1.0]

            else:
                return {"error": f"Unknown operation: {operation}"}

        except Exception as e:
            logger.error(f"Mathematical operation {operation} failed: {e}")
            return {"error": str(e)}

    def _generate_consensus(self, core_results: Dict[str, Any]) -> Any:
        """Generate consensus result from multiple core results."""
        try:
            if not core_results:
                return None

            # Filter out error results
            valid_results = {
                core_id: result
                for core_id, result in core_results.items()
                if not isinstance(result, dict) or "error" not in result
            }

            if not valid_results:
                return {"error": "No valid results from any core"}

            # If all results are identical, return that result
            results_list = list(valid_results.values())
            if all(result == results_list[0] for result in results_list):
                return results_list[0]

            # For numerical results, take the average
            if all(isinstance(result, (int, float)) for result in results_list):
                return sum(results_list) / len(results_list)

            # For array results, take element-wise average
            if all(isinstance(result, (list, np.ndarray)) for result in results_list):
                try:
                    arrays = [np.array(result) for result in results_list]
                    if all(arr.shape == arrays[0].shape for arr in arrays):
                        return np.mean(arrays, axis=0).tolist()
                except Exception:
                    pass

            # Default: return the first valid result
            return results_list[0]

        except Exception as e:
            logger.error(f"Consensus generation failed: {e}")
            return {"error": "Consensus generation failed"}

    def _verify_results(self, core_results: Dict[str, Any]) -> bool:
        """Verify consistency of results across cores."""
        try:
            valid_results = {
                core_id: result
                for core_id, result in core_results.items()
                if not isinstance(result, dict) or "error" not in result
            }

            if len(valid_results) < 2:
                return True  # Can't verify with less than 2 results

            results_list = list(valid_results.values())

            # For numerical results
            if all(isinstance(result, (int, float)) for result in results_list):
                mean_val = sum(results_list) / len(results_list)
                threshold = abs(mean_val) * 0.01  # 1% tolerance
                return all(abs(result - mean_val) <= threshold for result in results_list)

            # For arrays
            if all(isinstance(result, (list, np.ndarray)) for result in results_list):
                try:
                    arrays = [np.array(result) for result in results_list]
                    if all(arr.shape == arrays[0].shape for arr in arrays):
                        # Check element-wise similarity
                        for i in range(len(arrays)):
                            for j in range(i + 1, len(arrays)):
                                if not np.allclose(arrays[i], arrays[j], rtol=0.01):
                                    return False
                        return True
                except Exception:
                    pass

            # For identical results
            return all(result == results_list[0] for result in results_list)
        except Exception as e:
            logger.error(f"Result verification failed: {e}")
            return False

    def _calculate_load_distribution(self) -> Dict[str, float]:
        """Calculate current load distribution across cores."""
        return {core_id: status.current_load for core_id, status in self.cores.items()}

    def _calculate_core_agreement(self, core_results: Dict[str, Any]) -> float:
        """Calculate agreement score between core results."""
        try:
            if len(core_results) < 2:
                return 1.0

            valid_results = {
                core_id: result
                for core_id, result in core_results.items()
                if not isinstance(result, dict) or "error" not in result
            }

            if len(valid_results) < 2:
                return 0.5

            results_list = list(valid_results.values())

            # Perfect agreement
            if all(result == results_list[0] for result in results_list):
                return 1.0

            # Numerical agreement
            if all(isinstance(result, (int, float)) for result in results_list):
                mean_val = sum(results_list) / len(results_list)
                if mean_val != 0:
                    coefficient_of_variation = np.std(results_list) / abs(mean_val)
                    return max(0.0, 1.0 - coefficient_of_variation)

            return 0.5  # Moderate agreement for other cases

        except Exception:
            return 0.0

    def _calculate_core_health(self, core_status: CoreStatus) -> float:
        """Calculate health score for a core."""
        try:
            # Success rate component
            if core_status.total_operations > 0:
                success_rate = core_status.successful_operations / core_status.total_operations
            else:
                success_rate = 1.0

            # Load component (prefer moderate load)
            load_score = 1.0 - abs(core_status.current_load - 0.5)

            # Error rate component
            if core_status.total_operations > 0:
                error_rate = core_status.error_count / core_status.total_operations
                error_score = max(0.0, 1.0 - error_rate * 10)  # Heavily penalize errors
            else:
                error_score = 1.0

            # Heartbeat component
            time_since_heartbeat = time.time() - core_status.last_heartbeat
            heartbeat_score = max(0.0, 1.0 - time_since_heartbeat / 60.0)  # 1 minute timeout

            # Combined health score
            health_score = success_rate * 0.4 + load_score * 0.2 + error_score * 0.3 + heartbeat_score * 0.1

            return max(0.0, min(1.0, health_score))

        except Exception:
            return 0.5

    def _update_system_metrics(self, result: ProcessingResult):
        """Update system-wide performance metrics."""
        try:
            total_tasks = self.system_metrics["total_tasks_processed"]

            # Update success rate
            if result.verification_passed and result.consensus_result is not None:
                self.system_metrics["successful_operations"] += 1

            # Update average processing time
            current_avg = self.system_metrics["average_processing_time"]
            new_avg = ((current_avg * (total_tasks - 1)) + result.processing_time) / total_tasks
            self.system_metrics["average_processing_time"] = new_avg

            # Update error rate
            total_errors = sum(core.error_count for core in self.cores.values())
            total_operations = sum(core.total_operations for core in self.cores.values())
            if total_operations > 0:
                self.system_metrics["error_rate"] = total_errors / total_operations

            # Update parallel efficiency (simplified calculation)
            if result.processing_time > 0:
                theoretical_sequential_time = result.processing_time * len(result.core_results)
                efficiency = theoretical_sequential_time / (result.processing_time * self.max_workers)

                # Exponential moving average
                current_efficiency = self.system_metrics["parallel_efficiency"]
                alpha = 0.1  # Smoothing factor
                self.system_metrics["parallel_efficiency"] = alpha * efficiency + (1 - alpha) * current_efficiency

        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

    def _create_fallback_result(self, task: ProcessingTask, start_time: float) -> ProcessingResult:
        """Create fallback result for error conditions."""
        return ProcessingResult(
            task_id=task.task_id,
            core_results={"fallback": {"error": "Processing failed"}},
            consensus_result=None,
            processing_time=time.time() - start_time,
            verification_passed=False,
            load_distribution=self._calculate_load_distribution(),
            mathematical_signature=task.mathematical_signature,
            metadata={"status": "fallback"},
        )

    def _heartbeat_monitor(self):
        """Monitor core health with periodic heartbeats."""
        while self._running:
            try:
                current_time = time.time()

                for core_id, core_status in self.cores.items():
                    # Check if core is responsive
                    time_since_heartbeat = current_time - core_status.last_heartbeat

                    if time_since_heartbeat > 60.0:  # 1 minute timeout
                        logger.warning(f"Core {core_id} appears unresponsive")
                        core_status.is_active = False
                    elif time_since_heartbeat > 120.0:  # 2 minute hard timeout
                        logger.error(f"Core {core_id} marked as failed")
                        core_status.is_active = False
                        core_status.error_count += 1

                # Sleep for 10 seconds between checks
                for _ in range(100):  # 10 seconds in 0.1s increments
                    if not self._running:
                        break
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                time.sleep(1.0)


# Example usage
if __name__ == "__main__":
    print("Dual Unicore Handler Demonstration")
    print("=" * 40)

    # Initialize handler
    handler = DualUnicoreHandler(max_workers=2)

    try:
        # Test mathematical operation
        math_result = handler.execute_mathematical_operation("mean", [1.5, 2.3, 4.1, 3.7, 2.9, 5.2, 1.8, 3.4])

        print(f"Mathematical Operation Result:")
        print(f"  Operation: {math_result['operation']}")
        print(f"  Result: {math_result['result']:.3f}")
        print(f"  Verification: {'PASSED' if math_result['verification_passed'] else 'FAILED'}")
        print(f"  Core Agreement: {math_result['core_agreement']:.3f}")
        print(f"  Processing Time: {math_result['processing_time']:.6f}s")

        # Test load optimization
        load_result = handler.optimize_load_distribution()
        print(f"\nLoad Distribution:")
        for core_id, load in load_result["current_loads"].items():
            print(f"  {core_id}: {load:.3f}")
        print(f"  Balance Ratio: {load_result['balance_ratio']:.3f}")

        # Test diagnostics
        diagnostics = handler.get_core_diagnostics()
        print(f"\nSystem Health: {diagnostics['system_health'].upper()}")
        print(f"Active Cores: {len([c for c in diagnostics['cores'].values() if c['active']])}")

        # Test custom processing task
        def custom_calculation(x, y, operation="add"):
            if operation == "add":
                return x + y
            elif operation == "multiply":
                return x * y
            return x - y

        custom_task = ProcessingTask(
            task_id="custom_test",
            function=custom_calculation,
            args=(10.5, 5.2),
            kwargs={"operation": "multiply"},
            requires_verification=True,
        )

        custom_result = handler.process_task(custom_task, ProcessingMode.PARALLEL)
        print(f"\nCustom Task Result:")
        print(f"  Task ID: {custom_result.task_id}")
        print(f"  Consensus: {custom_result.consensus_result}")
        print(f"  Verification: {'PASSED' if custom_result.verification_passed else 'FAILED'}")
        print(f"  Processing Time: {custom_result.processing_time:.6f}s")

    finally:
        # Cleanup
        handler.shutdown()
