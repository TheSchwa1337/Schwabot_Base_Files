"""
Enhanced Thermal-Aware BTC Processing System
===========================================

Advanced BTC processor that integrates thermal awareness, dynamic load balancing,
and intelligent memory management for optimal performance under varying thermal conditions.

Key Features:
- Thermal-aware processing optimization
- Dynamic GPU/CPU load balancing based on temperature
- Intelligent memory pipeline integration
- Burst processing with thermal safeguards
- Predictive thermal management
- Real-time performance adaptation
- Integration with visual controller and pipeline manager

This system implements enhanced thermal management that automatically adjusts
BTC processing strategies based on real-time thermal conditions while maintaining
optimal performance and profit opportunities.
"""

import asyncio
import logging
import json
import time
import numpy as np
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path

# Core system imports
from .btc_data_processor import BTCDataProcessor, LoadBalancer, MemoryManager
from .thermal_system_integration import ThermalSystemIntegration, ThermalSystemConfig
from .pipeline_management_system import AdvancedPipelineManager, DataRetentionLevel
from .practical_visual_controller import PracticalVisualController, ControlMode, MappingBitLevel
from .thermal_zone_manager_mock import EnhancedThermalZoneManager, ThermalZone, ThermalState
from .thermal_performance_tracker import ThermalPerformanceTracker
from .unified_api_coordinator import UnifiedAPICoordinator

logger = logging.getLogger(__name__)

class ThermalProcessingMode(Enum):
    """BTC processing modes based on thermal state"""
    OPTIMAL_PERFORMANCE = "optimal_performance"    # Cool temps - max GPU utilization
    BALANCED_PROCESSING = "balanced_processing"    # Normal temps - balanced GPU/CPU
    THERMAL_EFFICIENT = "thermal_efficient"       # Warm temps - favor CPU
    EMERGENCY_THROTTLE = "emergency_throttle"      # Hot temps - minimal processing
    CRITICAL_PROTECTION = "critical_protection"   # Critical temps - emergency mode

class BTCProcessingStrategy(Enum):
    """BTC processing strategies for different thermal conditions"""
    HIGH_FREQUENCY_BURST = "high_frequency_burst"      # Short bursts, high performance
    SUSTAINED_THROUGHPUT = "sustained_throughput"      # Consistent processing
    THERMAL_CONSERVATIVE = "thermal_conservative"      # Temperature-first approach
    PROFIT_OPTIMIZED = "profit_optimized"             # Profit-first with thermal limits
    ADAPTIVE_HYBRID = "adaptive_hybrid"               # Dynamic strategy switching

@dataclass
class ThermalAwareBTCConfig:
    """Configuration for thermal-aware BTC processing"""
    # Thermal thresholds for processing mode switching
    temperature_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'optimal_max': 65.0,     # Above this, switch from optimal
        'balanced_max': 75.0,    # Above this, switch from balanced
        'efficient_max': 85.0,   # Above this, switch from efficient
        'throttle_max': 90.0,    # Above this, switch to emergency
        'critical_shutdown': 95.0 # Above this, emergency shutdown
    })
    
    # Processing allocation by thermal mode
    processing_allocations: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'optimal_performance': {'gpu': 0.85, 'cpu': 0.15, 'memory_intensive': True},
        'balanced_processing': {'gpu': 0.60, 'cpu': 0.40, 'memory_intensive': True},
        'thermal_efficient': {'gpu': 0.30, 'cpu': 0.70, 'memory_intensive': False},
        'emergency_throttle': {'gpu': 0.10, 'cpu': 0.90, 'memory_intensive': False},
        'critical_protection': {'gpu': 0.05, 'cpu': 0.95, 'memory_intensive': False}
    })
    
    # Burst processing parameters
    burst_config: Dict[str, Any] = field(default_factory=lambda: {
        'max_duration_seconds': 300,      # 5 minutes max burst
        'cooldown_ratio': 2.0,            # Cooldown = 2x burst duration
        'thermal_headroom_required': 10.0, # °C headroom needed for burst
        'profit_threshold_for_burst': 0.02 # 2% profit threshold to enable burst
    })
    
    # Memory pipeline integration
    memory_strategy: Dict[str, Any] = field(default_factory=lambda: {
        'thermal_priority_processing': True,    # Prioritize hot data processing
        'adaptive_retention_levels': True,     # Adjust retention based on thermal state
        'emergency_compression': True,         # Enable emergency compression
        'thermal_aware_caching': True         # Cache frequently accessed data when hot
    })

@dataclass
class BTCProcessingMetrics:
    """Real-time metrics for thermal-aware BTC processing"""
    current_thermal_mode: ThermalProcessingMode
    current_strategy: BTCProcessingStrategy
    temperature_cpu: float
    temperature_gpu: float
    processing_efficiency: float
    thermal_headroom: float
    burst_available: bool
    profit_rate_btc_per_hour: float
    memory_utilization_percent: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float
    operations_per_second: float
    thermal_drift_coefficient: float
    
    # Performance tracking
    successful_operations: int = 0
    failed_operations: int = 0
    thermal_throttling_events: int = 0
    burst_activations: int = 0
    emergency_shutdowns: int = 0

class EnhancedThermalAwareBTCProcessor:
    """
    Enhanced BTC processor with comprehensive thermal awareness and dynamic
    optimization capabilities. Integrates with all existing system components
    for unified thermal-aware operation.
    """
    
    def __init__(self,
                 btc_processor: Optional[BTCDataProcessor] = None,
                 thermal_system: Optional[ThermalSystemIntegration] = None,
                 pipeline_manager: Optional[AdvancedPipelineManager] = None,
                 visual_controller: Optional[PracticalVisualController] = None,
                 api_coordinator: Optional[UnifiedAPICoordinator] = None,
                 config: Optional[ThermalAwareBTCConfig] = None):
        """
        Initialize enhanced thermal-aware BTC processor
        
        Args:
            btc_processor: Existing BTC data processor
            thermal_system: Thermal system integration
            pipeline_manager: Advanced pipeline manager
            visual_controller: Practical visual controller
            api_coordinator: Unified API coordinator
            config: Thermal-aware BTC configuration
        """
        self.config = config or ThermalAwareBTCConfig()
        
        # Core system components
        self.btc_processor = btc_processor
        self.thermal_system = thermal_system
        self.pipeline_manager = pipeline_manager
        self.visual_controller = visual_controller
        self.api_coordinator = api_coordinator
        
        # Initialize thermal zone manager if not provided
        if not self.thermal_system and not hasattr(self, 'thermal_manager'):
            self.thermal_manager = EnhancedThermalZoneManager()
        else:
            self.thermal_manager = self.thermal_system.thermal_manager if self.thermal_system else None
        
        # Processing state
        self.current_mode = ThermalProcessingMode.BALANCED_PROCESSING
        self.current_strategy = BTCProcessingStrategy.ADAPTIVE_HYBRID
        self.is_running = False
        self.burst_active = False
        
        # Performance tracking
        self.metrics = BTCProcessingMetrics(
            current_thermal_mode=self.current_mode,
            current_strategy=self.current_strategy,
            temperature_cpu=70.0,
            temperature_gpu=65.0,
            processing_efficiency=1.0,
            thermal_headroom=20.0,
            burst_available=True,
            profit_rate_btc_per_hour=0.0,
            memory_utilization_percent=50.0,
            gpu_utilization_percent=40.0,
            cpu_utilization_percent=30.0,
            operations_per_second=100.0,
            thermal_drift_coefficient=1.0
        )
        
        # Processing queues with thermal prioritization
        self.thermal_priority_queue = asyncio.PriorityQueue()
        self.normal_processing_queue = asyncio.Queue()
        self.batch_processing_queue = asyncio.Queue()
        
        # Background tasks
        self.background_tasks = []
        self.monitoring_tasks = []
        
        # Performance history for optimization
        self.performance_history = []
        self.thermal_events = []
        self.optimization_decisions = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("EnhancedThermalAwareBTCProcessor initialized")
    
    async def start_enhanced_processing(self) -> bool:
        """Start the enhanced thermal-aware BTC processing system"""
        try:
            with self._lock:
                if self.is_running:
                    logger.warning("Enhanced BTC processor already running")
                    return False
                
                logger.info("🚀 Starting Enhanced Thermal-Aware BTC Processing...")
                
                # Start thermal monitoring
                await self._start_thermal_monitoring()
                
                # Initialize BTC processor if not already running
                if self.btc_processor and not getattr(self.btc_processor, 'is_running', False):
                    await self.btc_processor.start_processing_pipeline()
                
                # Start pipeline manager if not running
                if self.pipeline_manager and not self.pipeline_manager.is_running:
                    await self.pipeline_manager.start_pipeline()
                
                # Start thermal system if available
                if self.thermal_system and not self.thermal_system.is_running:
                    await self.thermal_system.start_system()
                
                # Start background processing tasks
                await self._start_background_tasks()
                
                # Initialize thermal-aware optimization
                await self._initialize_thermal_optimization()
                
                self.is_running = True
                logger.info("✅ Enhanced Thermal-Aware BTC Processing started successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error starting enhanced BTC processor: {e}")
            return False
    
    async def stop_enhanced_processing(self) -> bool:
        """Stop the enhanced thermal-aware BTC processing system"""
        try:
            with self._lock:
                if not self.is_running:
                    logger.warning("Enhanced BTC processor not running")
                    return False
                
                logger.info("🛑 Stopping Enhanced Thermal-Aware BTC Processing...")
                
                # Stop background tasks
                await self._stop_background_tasks()
                
                # Stop burst processing if active
                if self.burst_active:
                    await self._stop_burst_processing()
                
                self.is_running = False
                logger.info("✅ Enhanced Thermal-Aware BTC Processing stopped")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error stopping enhanced BTC processor: {e}")
            return False
    
    async def _start_thermal_monitoring(self) -> None:
        """Start comprehensive thermal monitoring"""
        if self.thermal_manager:
            self.thermal_manager.start_monitoring(interval=5.0)  # Monitor every 5 seconds
        
        # Start thermal performance tracking
        if self.thermal_system:
            # Thermal system already handles this
            pass
        else:
            # Create our own thermal monitoring task
            monitoring_task = asyncio.create_task(self._thermal_monitoring_loop())
            self.monitoring_tasks.append(monitoring_task)
        
        logger.info("🌡️ Thermal monitoring started")
    
    async def _thermal_monitoring_loop(self) -> None:
        """Independent thermal monitoring loop"""
        while self.is_running:
            try:
                # Get current thermal state
                if self.thermal_manager:
                    thermal_state = self.thermal_manager.get_current_state()
                    
                    # Update metrics
                    self.metrics.temperature_cpu = thermal_state.get('cpu_temp', 70.0)
                    self.metrics.temperature_gpu = thermal_state.get('gpu_temp', 65.0)
                    self.metrics.thermal_drift_coefficient = thermal_state.get('drift_coefficient', 1.0)
                    
                    # Calculate thermal headroom
                    max_temp = max(self.metrics.temperature_cpu, self.metrics.temperature_gpu)
                    self.metrics.thermal_headroom = max(0, self.config.temperature_thresholds['throttle_max'] - max_temp)
                    
                    # Determine optimal processing mode
                    await self._update_thermal_processing_mode()
                    
                    # Check for thermal events that require immediate action
                    await self._handle_thermal_events(thermal_state)
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in thermal monitoring loop: {e}")
                await asyncio.sleep(10.0)  # Longer sleep on error
    
    async def _update_thermal_processing_mode(self) -> None:
        """Update processing mode based on current thermal state"""
        max_temp = max(self.metrics.temperature_cpu, self.metrics.temperature_gpu)
        
        # Determine new mode based on temperature thresholds
        new_mode = self.current_mode
        
        if max_temp <= self.config.temperature_thresholds['optimal_max']:
            new_mode = ThermalProcessingMode.OPTIMAL_PERFORMANCE
        elif max_temp <= self.config.temperature_thresholds['balanced_max']:
            new_mode = ThermalProcessingMode.BALANCED_PROCESSING
        elif max_temp <= self.config.temperature_thresholds['efficient_max']:
            new_mode = ThermalProcessingMode.THERMAL_EFFICIENT
        elif max_temp <= self.config.temperature_thresholds['throttle_max']:
            new_mode = ThermalProcessingMode.EMERGENCY_THROTTLE
        else:
            new_mode = ThermalProcessingMode.CRITICAL_PROTECTION
        
        # Apply mode change if needed
        if new_mode != self.current_mode:
            await self._switch_thermal_mode(new_mode)
    
    async def _switch_thermal_mode(self, new_mode: ThermalProcessingMode) -> None:
        """Switch to a new thermal processing mode"""
        old_mode = self.current_mode
        self.current_mode = new_mode
        self.metrics.current_thermal_mode = new_mode
        
        logger.info(f"🔄 Switching thermal mode: {old_mode.value} → {new_mode.value}")
        
        # Get processing allocation for new mode
        allocation = self.config.processing_allocations[new_mode.value]
        
        # Update BTC processor load balancing
        if self.btc_processor and hasattr(self.btc_processor, 'load_balancer'):
            await self._apply_thermal_load_balancing(allocation)
        
        # Update memory pipeline strategy
        if self.pipeline_manager:
            await self._adjust_memory_pipeline_for_thermal_mode(new_mode)
        
        # Update visual controller if available
        if self.visual_controller:
            await self._update_visual_controller_thermal_mode(new_mode)
        
        # Record thermal event
        thermal_event = {
            'timestamp': time.time(),
            'old_mode': old_mode.value,
            'new_mode': new_mode.value,
            'temperature_cpu': self.metrics.temperature_cpu,
            'temperature_gpu': self.metrics.temperature_gpu,
            'thermal_headroom': self.metrics.thermal_headroom
        }
        self.thermal_events.append(thermal_event)
        
        # Keep only recent events
        if len(self.thermal_events) > 1000:
            self.thermal_events = self.thermal_events[-500:]
    
    async def _apply_thermal_load_balancing(self, allocation: Dict[str, Any]) -> None:
        """Apply thermal-aware load balancing to BTC processor"""
        try:
            # Update load balancer with thermal allocation
            if hasattr(self.btc_processor, 'load_balancer'):
                load_balancer = self.btc_processor.load_balancer
                
                # Force specific processing mode
                if allocation['gpu'] > 0.6:
                    load_balancer.current_mode = 'gpu'
                elif allocation['cpu'] > 0.6:
                    load_balancer.current_mode = 'cpu'
                else:
                    load_balancer.current_mode = 'auto'
                
                # Adjust batch sizes based on memory intensity
                if allocation.get('memory_intensive', True):
                    # Normal batch sizes
                    pass
                else:
                    # Reduce batch sizes to minimize memory usage
                    if hasattr(self.btc_processor, 'batch_size_gpu'):
                        self.btc_processor.batch_size_gpu = min(50, self.btc_processor.batch_size_gpu)
                    if hasattr(self.btc_processor, 'batch_size_cpu'):
                        self.btc_processor.batch_size_cpu = min(25, self.btc_processor.batch_size_cpu)
                
                logger.info(f"📊 Applied thermal load balancing: GPU {allocation['gpu']:.1%}, CPU {allocation['cpu']:.1%}")
                
        except Exception as e:
            logger.error(f"Error applying thermal load balancing: {e}")
    
    async def _adjust_memory_pipeline_for_thermal_mode(self, mode: ThermalProcessingMode) -> None:
        """Adjust memory pipeline based on thermal mode"""
        try:
            if not self.pipeline_manager:
                return
            
            # Adjust retention levels based on thermal state
            if mode in [ThermalProcessingMode.EMERGENCY_THROTTLE, ThermalProcessingMode.CRITICAL_PROTECTION]:
                # Aggressive memory cleanup for hot conditions
                await self.pipeline_manager._cleanup_expired_data()
                await self.pipeline_manager._compress_long_term_data()
                
                # Reduce retention times
                self.pipeline_manager.memory_config.retention_hours['short_term'] = 0.5  # 30 minutes
                self.pipeline_manager.memory_config.retention_hours['mid_term'] = 48    # 2 days
                
            elif mode == ThermalProcessingMode.THERMAL_EFFICIENT:
                # Moderate memory optimization
                self.pipeline_manager.memory_config.retention_hours['short_term'] = 1    # 1 hour
                self.pipeline_manager.memory_config.retention_hours['mid_term'] = 120   # 5 days
                
            else:
                # Normal or optimal conditions - standard retention
                self.pipeline_manager.memory_config.retention_hours['short_term'] = 1    # 1 hour
                self.pipeline_manager.memory_config.retention_hours['mid_term'] = 168   # 1 week
            
            logger.info(f"💾 Adjusted memory pipeline for thermal mode: {mode.value}")
            
        except Exception as e:
            logger.error(f"Error adjusting memory pipeline: {e}")
    
    async def _update_visual_controller_thermal_mode(self, mode: ThermalProcessingMode) -> None:
        """Update visual controller with current thermal mode"""
        try:
            if not self.visual_controller:
                return
            
            # Update thermal monitoring toggle
            self.visual_controller.visual_state.toggle_states['thermal_monitoring'] = True
            
            # Adjust bit mapping intensity based on thermal mode
            if mode == ThermalProcessingMode.OPTIMAL_PERFORMANCE:
                self.visual_controller.visual_state.slider_values['bit_mapping_intensity'] = 0.9
            elif mode == ThermalProcessingMode.BALANCED_PROCESSING:
                self.visual_controller.visual_state.slider_values['bit_mapping_intensity'] = 0.7
            elif mode == ThermalProcessingMode.THERMAL_EFFICIENT:
                self.visual_controller.visual_state.slider_values['bit_mapping_intensity'] = 0.5
            else:  # Emergency or critical
                self.visual_controller.visual_state.slider_values['bit_mapping_intensity'] = 0.3
            
            # Update thermal threshold
            max_temp = max(self.metrics.temperature_cpu, self.metrics.temperature_gpu)
            self.visual_controller.visual_state.slider_values['thermal_threshold'] = max_temp + 5.0
            
            logger.info(f"🎛️ Updated visual controller for thermal mode: {mode.value}")
            
        except Exception as e:
            logger.error(f"Error updating visual controller: {e}")
    
    async def _handle_thermal_events(self, thermal_state: Dict[str, Any]) -> None:
        """Handle immediate thermal events requiring action"""
        max_temp = max(
            thermal_state.get('cpu_temp', 70.0),
            thermal_state.get('gpu_temp', 65.0)
        )
        
        # Critical temperature - emergency shutdown
        if max_temp >= self.config.temperature_thresholds['critical_shutdown']:
            await self._emergency_thermal_shutdown()
            self.metrics.emergency_shutdowns += 1
            
        # Throttle temperature - emergency throttling
        elif max_temp >= self.config.temperature_thresholds['throttle_max']:
            await self._emergency_thermal_throttling()
            self.metrics.thermal_throttling_events += 1
            
        # Check if burst processing should be stopped
        if self.burst_active and max_temp >= self.config.temperature_thresholds['balanced_max']:
            await self._stop_burst_processing()
    
    async def _emergency_thermal_shutdown(self) -> None:
        """Emergency thermal shutdown procedure"""
        logger.critical("🚨 EMERGENCY THERMAL SHUTDOWN - Critical temperature reached!")
        
        try:
            # Stop burst processing immediately
            if self.burst_active:
                await self._stop_burst_processing()
            
            # Switch to critical protection mode
            await self._switch_thermal_mode(ThermalProcessingMode.CRITICAL_PROTECTION)
            
            # Pause non-critical operations
            if self.pipeline_manager:
                await self.pipeline_manager._pause_non_critical_operations()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Log emergency event
            emergency_event = {
                'timestamp': time.time(),
                'event_type': 'emergency_shutdown',
                'temperature_cpu': self.metrics.temperature_cpu,
                'temperature_gpu': self.metrics.temperature_gpu,
                'action_taken': 'critical_protection_mode'
            }
            self.thermal_events.append(emergency_event)
            
        except Exception as e:
            logger.error(f"Error during emergency thermal shutdown: {e}")
    
    async def _emergency_thermal_throttling(self) -> None:
        """Emergency thermal throttling procedure"""
        logger.warning("⚠️ EMERGENCY THERMAL THROTTLING - High temperature detected!")
        
        try:
            # Switch to emergency throttle mode
            await self._switch_thermal_mode(ThermalProcessingMode.EMERGENCY_THROTTLE)
            
            # Reduce processing intensity
            if self.btc_processor and hasattr(self.btc_processor, 'load_balancer'):
                self.btc_processor.load_balancer.current_mode = 'cpu'
            
            # Trigger memory optimization
            if self.pipeline_manager:
                await self.pipeline_manager._trigger_memory_optimization()
            
        except Exception as e:
            logger.error(f"Error during emergency thermal throttling: {e}")
    
    async def _initialize_thermal_optimization(self) -> None:
        """Initialize thermal optimization algorithms"""
        logger.info("🧠 Initializing thermal optimization algorithms...")
        
        # Set up predictive thermal management
        if self.config.memory_strategy.get('thermal_aware_caching', True):
            await self._setup_thermal_aware_caching()
        
        # Initialize adaptive processing strategies
        await self._setup_adaptive_strategies()
        
        logger.info("✅ Thermal optimization initialized")
    
    async def _setup_thermal_aware_caching(self) -> None:
        """Setup thermal-aware caching strategy"""
        # Implement caching strategy that prioritizes frequently accessed data
        # when system is running hot to reduce processing overhead
        pass
    
    async def _setup_adaptive_strategies(self) -> None:
        """Setup adaptive processing strategies"""
        # Initialize strategy selection based on thermal and profit conditions
        self.strategy_weights = {
            BTCProcessingStrategy.HIGH_FREQUENCY_BURST: 0.2,
            BTCProcessingStrategy.SUSTAINED_THROUGHPUT: 0.3,
            BTCProcessingStrategy.THERMAL_CONSERVATIVE: 0.2,
            BTCProcessingStrategy.PROFIT_OPTIMIZED: 0.2,
            BTCProcessingStrategy.ADAPTIVE_HYBRID: 0.1
        }
    
    async def _start_background_tasks(self) -> None:
        """Start background processing and monitoring tasks"""
        # Performance monitoring task
        perf_task = asyncio.create_task(self._performance_monitoring_loop())
        self.background_tasks.append(perf_task)
        
        # Thermal optimization task
        thermal_opt_task = asyncio.create_task(self._thermal_optimization_loop())
        self.background_tasks.append(thermal_opt_task)
        
        # Memory pipeline coordination task
        pipeline_task = asyncio.create_task(self._memory_pipeline_coordination_loop())
        self.background_tasks.append(pipeline_task)
        
        # Burst processing management task
        burst_task = asyncio.create_task(self._burst_management_loop())
        self.background_tasks.append(burst_task)
        
        logger.info("🔄 Background tasks started")
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks"""
        all_tasks = self.background_tasks + self.monitoring_tasks
        
        for task in all_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
        self.monitoring_tasks.clear()
        
        logger.info("🔄 Background tasks stopped")
    
    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring and metrics collection"""
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check for optimization opportunities
                await self._check_optimization_opportunities()
                
                # Update visual controller metrics if available
                if self.visual_controller:
                    await self._update_visual_metrics()
                
                await asyncio.sleep(10.0)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _update_performance_metrics(self) -> None:
        """Update current performance metrics"""
        try:
            # Get BTC processor stats if available
            if self.btc_processor:
                btc_stats = self.btc_processor.get_processing_stats()
                self.metrics.operations_per_second = btc_stats.get('hash_rate', 100.0)
                self.metrics.gpu_utilization_percent = btc_stats.get('gpu_utilization', 40.0)
                self.metrics.cpu_utilization_percent = btc_stats.get('cpu_utilization', 30.0)
            
            # Get memory utilization from pipeline manager
            if self.pipeline_manager:
                pipeline_status = self.pipeline_manager.get_pipeline_status()
                self.metrics.memory_utilization_percent = pipeline_status.get('memory_utilization', 50.0)
            
            # Calculate processing efficiency
            self.metrics.processing_efficiency = self._calculate_processing_efficiency()
            
            # Update burst availability
            if self.thermal_manager:
                thermal_state = self.thermal_manager.get_current_state()
                self.metrics.burst_available = thermal_state.get('burst_available', True)
            
            # Add to performance history
            performance_snapshot = {
                'timestamp': time.time(),
                'thermal_mode': self.current_mode.value,
                'efficiency': self.metrics.processing_efficiency,
                'operations_per_second': self.metrics.operations_per_second,
                'temperature_max': max(self.metrics.temperature_cpu, self.metrics.temperature_gpu),
                'memory_utilization': self.metrics.memory_utilization_percent
            }
            self.performance_history.append(performance_snapshot)
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_processing_efficiency(self) -> float:
        """Calculate overall processing efficiency score"""
        try:
            # Base efficiency from operations per second
            ops_efficiency = min(1.0, self.metrics.operations_per_second / 200.0)
            
            # Thermal efficiency - penalty for high temperatures
            max_temp = max(self.metrics.temperature_cpu, self.metrics.temperature_gpu)
            thermal_efficiency = max(0.1, 1.0 - (max_temp - 60.0) / 40.0)
            
            # Memory efficiency
            memory_efficiency = max(0.1, 1.0 - self.metrics.memory_utilization_percent / 100.0)
            
            # Combined efficiency with weights
            efficiency = (
                0.4 * ops_efficiency +
                0.4 * thermal_efficiency +
                0.2 * memory_efficiency
            )
            
            return max(0.1, min(1.0, efficiency))
            
        except Exception as e:
            logger.error(f"Error calculating processing efficiency: {e}")
            return 0.5
    
    async def _thermal_optimization_loop(self) -> None:
        """Background task for thermal optimization"""
        while self.is_running:
            try:
                # Analyze thermal trends
                await self._analyze_thermal_trends()
                
                # Optimize processing strategy
                await self._optimize_processing_strategy()
                
                # Check for burst opportunities
                await self._evaluate_burst_opportunities()
                
                await asyncio.sleep(30.0)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in thermal optimization loop: {e}")
                await asyncio.sleep(60.0)
    
    async def _memory_pipeline_coordination_loop(self) -> None:
        """Background task for memory pipeline coordination"""
        while self.is_running:
            try:
                # Coordinate with pipeline manager
                if self.pipeline_manager:
                    await self._coordinate_memory_pipeline()
                
                # Optimize data retention based on thermal state
                await self._optimize_thermal_data_retention()
                
                await asyncio.sleep(60.0)  # Coordinate every minute
                
            except Exception as e:
                logger.error(f"Error in memory pipeline coordination: {e}")
                await asyncio.sleep(120.0)
    
    async def _burst_management_loop(self) -> None:
        """Background task for burst processing management"""
        while self.is_running:
            try:
                # Check if burst is available and profitable
                if await self._should_activate_burst():
                    await self._activate_burst_processing()
                
                # Monitor active burst and stop if needed
                if self.burst_active:
                    await self._monitor_active_burst()
                
                await asyncio.sleep(15.0)  # Check every 15 seconds
                
            except Exception as e:
                logger.error(f"Error in burst management loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _should_activate_burst(self) -> bool:
        """Determine if burst processing should be activated"""
        if self.burst_active:
            return False
        
        if not self.metrics.burst_available:
            return False
        
        # Check thermal headroom
        if self.metrics.thermal_headroom < self.config.burst_config['thermal_headroom_required']:
            return False
        
        # Check if we're in a suitable thermal mode
        if self.current_mode not in [ThermalProcessingMode.OPTIMAL_PERFORMANCE, ThermalProcessingMode.BALANCED_PROCESSING]:
            return False
        
        # TODO: Check profit opportunity (integrate with profit calculations)
        # For now, activate burst periodically when conditions are good
        return True
    
    async def _activate_burst_processing(self) -> None:
        """Activate burst processing mode"""
        try:
            logger.info("⚡ Activating burst processing mode")
            
            # Start burst in thermal manager
            if self.thermal_manager:
                burst_started = self.thermal_manager.start_burst()
                if not burst_started:
                    logger.warning("Failed to start burst in thermal manager")
                    return
            
            self.burst_active = True
            self.metrics.burst_activations += 1
            
            # Increase processing intensity for burst
            if self.btc_processor and hasattr(self.btc_processor, 'load_balancer'):
                self.btc_processor.load_balancer.current_mode = 'gpu'  # Force GPU for burst
            
            # Adjust memory pipeline for burst
            if self.pipeline_manager:
                # Temporarily increase short-term memory allocation
                original_short_term = self.pipeline_manager.memory_config.short_term_limit_mb
                self.pipeline_manager.memory_config.short_term_limit_mb = int(original_short_term * 1.5)
            
            logger.info("✅ Burst processing activated")
            
        except Exception as e:
            logger.error(f"Error activating burst processing: {e}")
            self.burst_active = False
    
    async def _stop_burst_processing(self) -> None:
        """Stop burst processing mode"""
        try:
            logger.info("🛑 Stopping burst processing mode")
            
            self.burst_active = False
            
            # Return to normal processing mode
            if self.btc_processor and hasattr(self.btc_processor, 'load_balancer'):
                self.btc_processor.load_balancer.current_mode = 'auto'
            
            # Restore normal memory configuration
            if self.pipeline_manager:
                self.pipeline_manager.memory_config.short_term_limit_mb = 512  # Default value
            
            logger.info("✅ Burst processing stopped")
            
        except Exception as e:
            logger.error(f"Error stopping burst processing: {e}")
    
    async def _monitor_active_burst(self) -> None:
        """Monitor active burst and stop if conditions change"""
        if not self.burst_active:
            return
        
        # Check thermal conditions
        max_temp = max(self.metrics.temperature_cpu, self.metrics.temperature_gpu)
        if max_temp >= self.config.temperature_thresholds['balanced_max']:
            await self._stop_burst_processing()
            return
        
        # Check thermal headroom
        if self.metrics.thermal_headroom < 5.0:  # Stop if less than 5°C headroom
            await self._stop_burst_processing()
            return
        
        # Check burst duration (implemented in thermal manager)
        if self.thermal_manager:
            thermal_state = self.thermal_manager.get_current_state()
            if not thermal_state.get('burst_available', True):
                await self._stop_burst_processing()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'is_running': self.is_running,
            'thermal_mode': self.current_mode.value,
            'processing_strategy': self.current_strategy.value,
            'burst_active': self.burst_active,
            'metrics': asdict(self.metrics),
            'thermal_events_count': len(self.thermal_events),
            'performance_history_length': len(self.performance_history),
            'thermal_thresholds': self.config.temperature_thresholds,
            'system_components': {
                'btc_processor': self.btc_processor is not None,
                'thermal_system': self.thermal_system is not None,
                'pipeline_manager': self.pipeline_manager is not None,
                'visual_controller': self.visual_controller is not None,
                'api_coordinator': self.api_coordinator is not None
            }
        }
    
    async def get_thermal_recommendations(self) -> List[str]:
        """Get thermal optimization recommendations"""
        recommendations = []
        
        max_temp = max(self.metrics.temperature_cpu, self.metrics.temperature_gpu)
        
        if max_temp > self.config.temperature_thresholds['efficient_max']:
            recommendations.append("🌡️ Consider reducing processing intensity - high temperature detected")
        
        if self.metrics.thermal_headroom < 10.0:
            recommendations.append("⚠️ Low thermal headroom - burst processing not recommended")
        
        if self.metrics.processing_efficiency < 0.7:
            recommendations.append("📊 Processing efficiency below optimal - consider thermal optimization")
        
        if not self.burst_active and self.metrics.thermal_headroom > 15.0:
            recommendations.append("⚡ Good thermal conditions for burst processing")
        
        if self.metrics.memory_utilization_percent > 80.0:
            recommendations.append("💾 High memory utilization - consider memory pipeline optimization")
        
        return recommendations

# Integration function for easy system creation
async def create_enhanced_thermal_btc_processor(
    btc_processor: Optional[BTCDataProcessor] = None,
    thermal_system: Optional[ThermalSystemIntegration] = None,
    pipeline_manager: Optional[AdvancedPipelineManager] = None,
    visual_controller: Optional[PracticalVisualController] = None,
    api_coordinator: Optional[UnifiedAPICoordinator] = None,
    config: Optional[ThermalAwareBTCConfig] = None
) -> EnhancedThermalAwareBTCProcessor:
    """
    Create and initialize enhanced thermal-aware BTC processor
    
    Args:
        btc_processor: Existing BTC data processor
        thermal_system: Thermal system integration
        pipeline_manager: Advanced pipeline manager
        visual_controller: Practical visual controller
        api_coordinator: Unified API coordinator
        config: Thermal-aware BTC configuration
        
    Returns:
        Initialized enhanced thermal-aware BTC processor
    """
    processor = EnhancedThermalAwareBTCProcessor(
        btc_processor=btc_processor,
        thermal_system=thermal_system,
        pipeline_manager=pipeline_manager,
        visual_controller=visual_controller,
        api_coordinator=api_coordinator,
        config=config
    )
    
    # Start the enhanced processing
    success = await processor.start_enhanced_processing()
    
    if not success:
        raise RuntimeError("Failed to start enhanced thermal-aware BTC processor")
    
    logger.info("✅ Enhanced Thermal-Aware BTC Processor created and started successfully")
    return processor 