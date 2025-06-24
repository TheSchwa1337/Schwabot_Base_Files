#!/usr/bin/env python3
"""
Universal Schwabot Client - Schwabot UROS v1.0
============================================

Universal client that can run on any device, automatically detecting hardware
capabilities and connecting to the distributed Schwabot network for profit calculations.

Features:
- Automatic hardware detection and self-registration
- Connection to distributed Schwabot network
- Local profit calculations based on hardware capabilities
- Real-time synchronization with central coordinator
- Universal deployment across any hardware configuration
"""

import json
import time
import logging
import hashlib
import threading
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import psutil

logger = logging.getLogger(__name__)

class ClientMode(Enum):
    """Client operation modes."""
    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"
    MAINTENANCE = "maintenance"

class ClientStatus(Enum):
    """Client status types."""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class ClientTask:
    """Client task information."""
    task_id: str
    task_type: str
    priority: float
    data: Dict[str, Any]
    received_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClientPerformance:
    """Client performance metrics."""
    cpu_usage: float
    memory_usage: float
    calculations_since_last_heartbeat: int
    profit_contributed: float
    tasks_completed: int
    average_response_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class UniversalSchwabotClient:
    """
    Universal Schwabot Client for Schwabot UROS v1.0.
    
    Can run on any device and automatically connect to the distributed network.
    """
    
    def __init__(self, server_url: str = "http://localhost:5000", mode: ClientMode = ClientMode.DEMO):
        self.server_url = server_url
        self.mode = mode
        
        # Import hardware self-identifier
        from hardware_self_identifier import HardwareSelfIdentifier
        self.hardware_identifier = HardwareSelfIdentifier(server_url)
        
        # Client state
        self.client_status = ClientStatus.INITIALIZING
        self.device_id = None
        self.node_id = None
        self.profit_allocation = 0.0
        self.sync_interval = 30.0
        
        # Performance tracking
        self.current_task: Optional[ClientTask] = None
        self.completed_tasks: List[ClientTask] = []
        self.performance_history: List[ClientPerformance] = []
        self.total_profit_contributed = 0.0
        self.total_calculations = 0
        
        # Threading for background operations
        self.heartbeat_thread = None
        self.task_processor_thread = None
        self.performance_monitor_thread = None
        self.running = False
        
        # Network communication
        self.session = requests.Session()
        self.session.timeout = 10
        
        logger.info("Universal Schwabot Client initialized")

    def start(self) -> bool:
        """
        Start the universal Schwabot client.
        
        Returns:
        --------
        bool
            True if successfully started, False otherwise
        """
        try:
            logger.info("Starting Universal Schwabot Client...")
            
            # Step 1: Detect hardware capabilities
            logger.info("Detecting hardware capabilities...")
            hardware_profile = self.hardware_identifier.detect_hardware_capabilities()
            self.device_id = hardware_profile.device_id
            
            logger.info(f"Hardware detected: {hardware_profile.hardware_tier.value} tier, {hardware_profile.compute_capability.value}")
            
            # Step 2: Register with network
            logger.info("Registering with Schwabot network...")
            registration = self.hardware_identifier.register_with_network()
            
            if not registration.success:
                logger.error(f"Failed to register with network: {registration.error_message}")
                return False
            
            self.node_id = registration.assigned_node_id
            self.profit_allocation = registration.profit_allocation
            self.sync_interval = registration.sync_interval
            
            logger.info(f"Registered with network: {self.node_id}")
            
            # Step 3: Start background threads
            self.running = True
            self._start_background_threads()
            
            # Step 4: Update status
            self.client_status = ClientStatus.CONNECTED
            
            logger.info("Universal Schwabot Client started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting client: {e}")
            self.client_status = ClientStatus.ERROR
            return False

    def _start_background_threads(self) -> None:
        """Start background processing threads."""
        try:
            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            
            # Start task processor thread
            self.task_processor_thread = threading.Thread(target=self._task_processor_loop, daemon=True)
            self.task_processor_thread.start()
            
            # Start performance monitor thread
            self.performance_monitor_thread = threading.Thread(target=self._performance_monitor_loop, daemon=True)
            self.performance_monitor_thread.start()
            
            logger.info("Background threads started")
            
        except Exception as e:
            logger.error(f"Error starting background threads: {e}")

    def _heartbeat_loop(self) -> None:
        """Send heartbeat to server in background thread."""
        while self.running:
            try:
                # Get current performance metrics
                performance = self._get_current_performance()
                
                # Send heartbeat
                heartbeat_data = {
                    "device_id": self.device_id,
                    "performance_metrics": {
                        "cpu_usage": performance.cpu_usage,
                        "memory_usage": performance.memory_usage,
                        "calculations_since_last_heartbeat": performance.calculations_since_last_heartbeat,
                        "profit_contributed": performance.profit_contributed
                    }
                }
                
                response = self.session.post(f"{self.server_url}/api/heartbeat", json=heartbeat_data)
                
                if response.status_code == 200:
                    self.client_status = ClientStatus.CONNECTED
                else:
                    logger.warning(f"Heartbeat failed: {response.status_code}")
                    self.client_status = ClientStatus.DISCONNECTED
                
                # Reset counters
                self._reset_performance_counters()
                
                # Sleep for sync interval
                time.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                self.client_status = ClientStatus.DISCONNECTED
                time.sleep(60)  # Wait longer on error

    def _task_processor_loop(self) -> None:
        """Process tasks in background thread."""
        while self.running:
            try:
                # Request task from server
                task_data = {"device_id": self.device_id}
                response = self.session.post(f"{self.server_url}/api/task", json=task_data)
                
                if response.status_code == 200:
                    task_response = response.json()
                    
                    if task_response.get("task_available"):
                        # Process task
                        task = ClientTask(
                            task_id=task_response["task_id"],
                            task_type=task_response["task_type"],
                            priority=task_response["priority"],
                            data=task_response["data"],
                            received_at=datetime.now()
                        )
                        
                        self.current_task = task
                        result = self._process_task(task)
                        
                        # Complete task
                        complete_data = {
                            "task_id": task.task_id,
                            "device_id": self.device_id,
                            "result": result
                        }
                        
                        complete_response = self.session.post(f"{self.server_url}/api/task/complete", json=complete_data)
                        
                        if complete_response.status_code == 200:
                            task.completed_at = datetime.now()
                            task.result = result
                            self.completed_tasks.append(task)
                            self.total_calculations += 1
                            
                            # Update profit contribution
                            profit_contributed = result.get("profit_contributed", 0.0)
                            self.total_profit_contributed += profit_contributed
                            
                            logger.info(f"Task completed: {task.task_id} (profit: ${profit_contributed:.2f})")
                        else:
                            logger.warning(f"Failed to complete task: {complete_response.status_code}")
                
                # Sleep before next task request
                time.sleep(5)  # Check for tasks every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in task processor loop: {e}")
                time.sleep(30)  # Wait longer on error

    def _performance_monitor_loop(self) -> None:
        """Monitor performance in background thread."""
        while self.running:
            try:
                # Get current performance
                performance = self._get_current_performance()
                self.performance_history.append(performance)
                
                # Keep only last 1000 performance records
                if len(self.performance_history) > 1000:
                    self.performance_history.pop(0)
                
                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitor loop: {e}")
                time.sleep(60)

    def _process_task(self, task: ClientTask) -> Dict[str, Any]:
        """
        Process a task based on task type.
        
        Parameters:
        -----------
        task : ClientTask
            Task to process
            
        Returns:
        --------
        Dict[str, Any]
            Task result
        """
        try:
            start_time = time.time()
            
            if task.task_type == "profit_calculation":
                result = self._process_profit_calculation(task.data)
            elif task.task_type == "tensor_processing":
                result = self._process_tensor_processing(task.data)
            elif task.task_type == "hash_validation":
                result = self._process_hash_validation(task.data)
            elif task.task_type == "entropy_analysis":
                result = self._process_entropy_analysis(task.data)
            else:
                result = {"error": f"Unknown task type: {task.task_type}"}
            
            # Add processing metadata
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["device_id"] = self.device_id
            result["task_id"] = task.task_id
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            return {
                "error": str(e),
                "task_id": task.task_id,
                "device_id": self.device_id
            }

    def _process_profit_calculation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process profit calculation task."""
        try:
            # Extract input data
            price_data = data.get("price_data", [])
            volume_data = data.get("volume_data", [])
            volatility = data.get("volatility", 0.1)
            
            if not price_data or not volume_data:
                return {"profit_contributed": 0.0, "error": "Insufficient data"}
            
            # Calculate profit using hardware-appropriate algorithm
            profit_score = self._calculate_profit_score(price_data, volume_data, volatility)
            
            # Scale profit based on hardware capabilities
            hardware_profile = self.hardware_identifier.hardware_profile
            scaled_profit = profit_score * hardware_profile.overall_score * self.profit_allocation
            
            return {
                "profit_contributed": scaled_profit,
                "profit_score": profit_score,
                "hardware_score": hardware_profile.overall_score,
                "allocation_factor": self.profit_allocation
            }
            
        except Exception as e:
            logger.error(f"Error in profit calculation: {e}")
            return {"profit_contributed": 0.0, "error": str(e)}

    def _process_tensor_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process tensor processing task."""
        try:
            # Extract tensor data
            tensor_data = data.get("tensor_data", [])
            operation = data.get("operation", "multiply")
            
            if not tensor_data:
                return {"profit_contributed": 0.0, "error": "No tensor data"}
            
            # Perform tensor operation based on hardware capabilities
            hardware_profile = self.hardware_identifier.hardware_profile
            
            if hardware_profile.compute_capability.value in ["gpu_performance", "gpu_enterprise", "hybrid"]:
                # Use GPU-optimized processing
                result = self._gpu_tensor_operation(tensor_data, operation)
            else:
                # Use CPU processing
                result = self._cpu_tensor_operation(tensor_data, operation)
            
            # Calculate profit contribution based on processing complexity
            complexity_score = len(tensor_data) * len(tensor_data[0]) if tensor_data else 0
            profit_contribution = min(complexity_score * 0.001, 1.0) * self.profit_allocation
            
            return {
                "profit_contributed": profit_contribution,
                "tensor_result": result,
                "complexity_score": complexity_score,
                "processing_method": "gpu" if "gpu" in hardware_profile.compute_capability.value else "cpu"
            }
            
        except Exception as e:
            logger.error(f"Error in tensor processing: {e}")
            return {"profit_contributed": 0.0, "error": str(e)}

    def _process_hash_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process hash validation task."""
        try:
            # Extract hash data
            input_data = data.get("input_data", "")
            expected_hash = data.get("expected_hash", "")
            
            if not input_data:
                return {"profit_contributed": 0.0, "error": "No input data"}
            
            # Calculate hash
            calculated_hash = hashlib.sha256(input_data.encode()).hexdigest()
            
            # Validate hash
            is_valid = calculated_hash == expected_hash
            
            # Calculate profit contribution
            profit_contribution = 0.1 if is_valid else 0.0
            profit_contribution *= self.profit_allocation
            
            return {
                "profit_contributed": profit_contribution,
                "hash_valid": is_valid,
                "calculated_hash": calculated_hash,
                "expected_hash": expected_hash
            }
            
        except Exception as e:
            logger.error(f"Error in hash validation: {e}")
            return {"profit_contributed": 0.0, "error": str(e)}

    def _process_entropy_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process entropy analysis task."""
        try:
            # Extract entropy data
            entropy_data = data.get("entropy_data", [])
            
            if not entropy_data:
                return {"profit_contributed": 0.0, "error": "No entropy data"}
            
            # Calculate entropy metrics
            entropy_mean = np.mean(entropy_data)
            entropy_std = np.std(entropy_data)
            entropy_entropy = -np.sum(np.histogram(entropy_data, bins=10)[0] / len(entropy_data) * 
                                    np.log2(np.histogram(entropy_data, bins=10)[0] / len(entropy_data) + 1e-10))
            
            # Calculate profit contribution based on entropy complexity
            complexity_score = entropy_entropy / 10.0  # Normalize
            profit_contribution = min(complexity_score, 1.0) * self.profit_allocation
            
            return {
                "profit_contributed": profit_contribution,
                "entropy_mean": entropy_mean,
                "entropy_std": entropy_std,
                "entropy_entropy": entropy_entropy,
                "complexity_score": complexity_score
            }
            
        except Exception as e:
            logger.error(f"Error in entropy analysis: {e}")
            return {"profit_contributed": 0.0, "error": str(e)}

    def _calculate_profit_score(self, price_data: List[float], volume_data: List[float], volatility: float) -> float:
        """Calculate profit score from price and volume data."""
        try:
            if len(price_data) < 2 or len(volume_data) < 2:
                return 0.0
            
            # Calculate price momentum
            price_changes = np.diff(price_data)
            price_momentum = np.mean(price_changes)
            
            # Calculate volume momentum
            volume_changes = np.diff(volume_data)
            volume_momentum = np.mean(volume_changes)
            
            # Calculate volatility-adjusted profit score
            volatility_factor = 1.0 / (1.0 + volatility)
            momentum_score = abs(price_momentum) * abs(volume_momentum)
            
            profit_score = momentum_score * volatility_factor
            
            return min(profit_score, 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating profit score: {e}")
            return 0.0

    def _gpu_tensor_operation(self, tensor_data: List[List[float]], operation: str) -> List[List[float]]:
        """Perform GPU-optimized tensor operation."""
        try:
            # Convert to numpy arrays for efficient processing
            tensor = np.array(tensor_data)
            
            if operation == "multiply":
                result = tensor * tensor
            elif operation == "add":
                result = tensor + tensor
            elif operation == "subtract":
                result = tensor - tensor
            else:
                result = tensor
            
            return result.tolist()
            
        except Exception as e:
            logger.error(f"Error in GPU tensor operation: {e}")
            return tensor_data

    def _cpu_tensor_operation(self, tensor_data: List[List[float]], operation: str) -> List[List[float]]:
        """Perform CPU tensor operation."""
        try:
            # Simple CPU-based tensor operation
            result = []
            for row in tensor_data:
                new_row = []
                for element in row:
                    if operation == "multiply":
                        new_row.append(element * element)
                    elif operation == "add":
                        new_row.append(element + element)
                    elif operation == "subtract":
                        new_row.append(element - element)
                    else:
                        new_row.append(element)
                result.append(new_row)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in CPU tensor operation: {e}")
            return tensor_data

    def _get_current_performance(self) -> ClientPerformance:
        """Get current performance metrics."""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Calculate response time from recent tasks
            if self.completed_tasks:
                recent_tasks = self.completed_tasks[-10:]  # Last 10 tasks
                response_times = [
                    (task.completed_at - task.received_at).total_seconds()
                    for task in recent_tasks
                    if task.completed_at
                ]
                avg_response_time = np.mean(response_times) if response_times else 0.0
            else:
                avg_response_time = 0.0
            
            return ClientPerformance(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                calculations_since_last_heartbeat=self.total_calculations,
                profit_contributed=self.total_profit_contributed,
                tasks_completed=len(self.completed_tasks),
                average_response_time=avg_response_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting current performance: {e}")
            return ClientPerformance(
                cpu_usage=0.0,
                memory_usage=0.0,
                calculations_since_last_heartbeat=0,
                profit_contributed=0.0,
                tasks_completed=0,
                average_response_time=0.0,
                timestamp=datetime.now()
            )

    def _reset_performance_counters(self) -> None:
        """Reset performance counters after heartbeat."""
        self.total_calculations = 0
        self.total_profit_contributed = 0.0

    def get_client_status(self) -> Dict[str, Any]:
        """
        Get client status and statistics.
        
        Returns:
        --------
        Dict[str, Any]
            Client status information
        """
        try:
            performance = self._get_current_performance()
            
            return {
                "client_status": self.client_status.value,
                "device_id": self.device_id,
                "node_id": self.node_id,
                "mode": self.mode.value,
                "profit_allocation": self.profit_allocation,
                "sync_interval": self.sync_interval,
                "hardware_profile": {
                    "hardware_tier": self.hardware_identifier.hardware_profile.hardware_tier.value,
                    "compute_capability": self.hardware_identifier.hardware_profile.compute_capability.value,
                    "overall_score": self.hardware_identifier.hardware_profile.overall_score
                } if self.hardware_identifier.hardware_profile else None,
                "performance": {
                    "cpu_usage": performance.cpu_usage,
                    "memory_usage": performance.memory_usage,
                    "total_tasks_completed": performance.tasks_completed,
                    "average_response_time": performance.average_response_time
                },
                "total_profit_contributed": self.total_profit_contributed,
                "current_task": {
                    "task_id": self.current_task.task_id,
                    "task_type": self.current_task.task_type,
                    "priority": self.current_task.priority
                } if self.current_task else None
            }
            
        except Exception as e:
            logger.error(f"Error getting client status: {e}")
            return {"error": str(e)}

    def stop(self) -> None:
        """Stop the universal Schwabot client."""
        try:
            logger.info("Stopping Universal Schwabot Client...")
            self.running = False
            self.client_status = ClientStatus.DISCONNECTED
            
            # Stop hardware identifier monitoring
            if self.hardware_identifier:
                self.hardware_identifier.monitoring_running = False
            
            logger.info("Universal Schwabot Client stopped")
            
        except Exception as e:
            logger.error(f"Error stopping client: {e}")

def main():
    """Main function for testing universal Schwabot client."""
    try:
        # Initialize client
        client = UniversalSchwabotClient(server_url="http://localhost:5000", mode=ClientMode.DEMO)
        
        # Start client
        if client.start():
            print("Universal Schwabot Client started successfully!")
            print(f"Device ID: {client.device_id}")
            print(f"Node ID: {client.node_id}")
            print(f"Profit Allocation: {client.profit_allocation:.1%}")
            
            # Keep running
            try:
                while True:
                    time.sleep(10)
                    status = client.get_client_status()
                    print(f"Status: {status['client_status']}, CPU: {status['performance']['cpu_usage']:.1f}%")
            except KeyboardInterrupt:
                print("\nShutting down...")
                client.stop()
        else:
            print("Failed to start Universal Schwabot Client")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 