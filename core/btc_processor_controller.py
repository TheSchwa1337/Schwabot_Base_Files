"""
BTC Processor Controller
========================

Provides dynamic control over BTC data processor features to manage system load,
memory usage, and processing efficiency. Allows toggling of specific functionalities
to prevent system overload during live testing and hash processing.
"""

import logging
import time
import psutil
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ProcessorConfig:
    """Configuration state for BTC processor features"""
    # Core processing controls
    mining_analysis_enabled: bool = True
    block_timing_analysis: bool = True
    nonce_sequence_analysis: bool = True
    difficulty_tracking: bool = True
    
    # Memory management controls
    memory_management_enabled: bool = True
    short_term_memory: bool = True
    mid_term_memory: bool = True
    long_term_memory: bool = True
    
    # Hash generation controls
    hash_generation_enabled: bool = True
    entropy_validation: bool = True
    timing_monitoring: bool = True
    
    # Load balancing controls
    load_balancing_enabled: bool = True
    gpu_processing: bool = True
    cpu_processing: bool = True
    
    # Storage controls
    historical_data_storage: bool = True
    pattern_storage: bool = True
    statistics_tracking: bool = True
    
    # Performance monitoring
    performance_monitoring: bool = True
    error_tracking: bool = True
    
    # System resource limits
    max_memory_usage_gb: float = 10.0
    max_cpu_usage_percent: float = 80.0
    max_gpu_usage_percent: float = 85.0
    
    # Backlog management
    backlog_processing: bool = True
    max_backlog_size: int = 10000
    auto_cleanup_enabled: bool = True

@dataclass
class SystemMetrics:
    """Current system resource usage metrics"""
    cpu_usage: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_usage: float = 0.0
    disk_usage_gb: float = 0.0
    active_processes: int = 0
    timestamp: float = field(default_factory=time.time)

class BTCProcessorController:
    """Controls BTC processor features and manages system resources"""
    
    def __init__(self, processor=None):
        self.processor = processor
        self.config = ProcessorConfig()
        self.is_monitoring = False
        self.metrics_history = []
        self.control_tasks = []
        self.emergency_shutdown_triggered = False
        
        # Initialize feature control flags
        self.feature_states = {
            'mining_analysis': True,
            'block_timing': True,
            'nonce_sequences': True,
            'difficulty_tracking': True,
            'memory_management': True,
            'hash_generation': True,
            'load_balancing': True,
            'storage': True,
            'monitoring': True
        }
        
        # Initialize system thresholds
        self.system_thresholds = {
            'memory_warning': 8.0,  # GB
            'memory_critical': 12.0,  # GB
            'cpu_warning': 70.0,  # %
            'cpu_critical': 90.0,  # %
            'gpu_warning': 80.0,  # %
            'gpu_critical': 95.0  # %
        }
        
    async def start_monitoring(self):
        """Start monitoring system resources and managing processor features"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
            
        self.is_monitoring = True
        logger.info("Starting BTC processor monitoring and control")
        
        # Start monitoring tasks
        self.control_tasks = [
            asyncio.create_task(self._monitor_system_resources()),
            asyncio.create_task(self._manage_processor_features()),
            asyncio.create_task(self._cleanup_data()),
            asyncio.create_task(self._emergency_monitoring())
        ]
        
        await asyncio.gather(*self.control_tasks, return_exceptions=True)
        
    async def stop_monitoring(self):
        """Stop monitoring and cleanup tasks"""
        self.is_monitoring = False
        
        # Cancel all control tasks
        for task in self.control_tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        await asyncio.gather(*self.control_tasks, return_exceptions=True)
        
        logger.info("BTC processor monitoring stopped")
        
    async def _monitor_system_resources(self):
        """Monitor system resource usage"""
        while self.is_monitoring:
            try:
                # Get current system metrics
                metrics = await self._get_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 1000 readings)
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                    
                # Check for resource warnings
                await self._check_resource_thresholds(metrics)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)
                
    async def _get_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_gb = memory.used / (1024**3)
            
            # GPU usage (if available)
            gpu_usage = 0.0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
                
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_gb = disk.used / (1024**3)
            
            # Active processes
            active_processes = len(psutil.pids())
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage_gb=memory_usage_gb,
                gpu_usage=gpu_usage,
                disk_usage_gb=disk_usage_gb,
                active_processes=active_processes
            )
            
        except Exception as e:
            logger.error(f"System metrics error: {e}")
            return SystemMetrics()
            
    async def _check_resource_thresholds(self, metrics: SystemMetrics):
        """Check if system resources exceed thresholds and take action"""
        try:
            # Memory threshold checks
            if metrics.memory_usage_gb > self.system_thresholds['memory_critical']:
                logger.warning(f"Critical memory usage: {metrics.memory_usage_gb:.1f}GB")
                await self._emergency_memory_cleanup()
            elif metrics.memory_usage_gb > self.system_thresholds['memory_warning']:
                logger.warning(f"High memory usage: {metrics.memory_usage_gb:.1f}GB")
                await self._reduce_memory_usage()
                
            # CPU threshold checks
            if metrics.cpu_usage > self.system_thresholds['cpu_critical']:
                logger.warning(f"Critical CPU usage: {metrics.cpu_usage:.1f}%")
                await self._reduce_cpu_load()
            elif metrics.cpu_usage > self.system_thresholds['cpu_warning']:
                logger.warning(f"High CPU usage: {metrics.cpu_usage:.1f}%")
                await self._optimize_cpu_usage()
                
            # GPU threshold checks
            if metrics.gpu_usage > self.system_thresholds['gpu_critical']:
                logger.warning(f"Critical GPU usage: {metrics.gpu_usage:.1f}%")
                await self._reduce_gpu_load()
            elif metrics.gpu_usage > self.system_thresholds['gpu_warning']:
                logger.warning(f"High GPU usage: {metrics.gpu_usage:.1f}%")
                await self._optimize_gpu_usage()
                
        except Exception as e:
            logger.error(f"Threshold check error: {e}")
            
    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup procedures"""
        try:
            logger.info("Initiating emergency memory cleanup")
            
            # Disable non-essential features
            await self.disable_feature('mining_analysis')
            await self.disable_feature('block_timing')
            await self.disable_feature('nonce_sequences')
            
            # Clear memory buffers if processor available
            if self.processor:
                # Clear short and mid-term memory
                if hasattr(self.processor, 'memory_manager'):
                    self.processor.memory_manager.short_term_buffer.clear()
                    self.processor.memory_manager.mid_term_buffer.clear()
                    
                # Clear mining data storage
                if hasattr(self.processor, 'mining_data_storage'):
                    for storage_type in self.processor.mining_data_storage:
                        self.processor.mining_data_storage[storage_type].clear()
                        
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Emergency memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency memory cleanup error: {e}")
            
    async def _reduce_memory_usage(self):
        """Reduce memory usage by disabling non-essential features"""
        try:
            logger.info("Reducing memory usage")
            
            # Disable pattern storage
            await self.disable_feature('pattern_storage')
            
            # Reduce buffer sizes
            if self.processor and hasattr(self.processor, 'memory_manager'):
                # Clear oldest data from buffers
                if len(self.processor.memory_manager.short_term_buffer) > 1000:
                    # Keep only newest 1000 entries
                    new_buffer = list(self.processor.memory_manager.short_term_buffer)[-1000:]
                    self.processor.memory_manager.short_term_buffer.clear()
                    self.processor.memory_manager.short_term_buffer.extend(new_buffer)
                    
        except Exception as e:
            logger.error(f"Memory reduction error: {e}")
            
    async def _reduce_cpu_load(self):
        """Reduce CPU load by disabling CPU-intensive features"""
        try:
            logger.info("Reducing CPU load")
            
            # Disable CPU-intensive analysis
            await self.disable_feature('mining_analysis')
            await self.disable_feature('difficulty_tracking')
            
            # Reduce processing workers
            if self.processor and hasattr(self.processor, 'executor'):
                self.processor.executor._max_workers = 2  # Reduce workers
                
        except Exception as e:
            logger.error(f"CPU load reduction error: {e}")
            
    async def _reduce_gpu_load(self):
        """Reduce GPU load by switching to CPU processing"""
        try:
            logger.info("Reducing GPU load")
            
            # Switch to CPU processing
            await self.disable_feature('gpu_processing')
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"GPU load reduction error: {e}")
            
    async def _optimize_cpu_usage(self):
        """Optimize CPU usage without disabling features"""
        try:
            # Reduce update frequencies
            if self.processor:
                # Increase sleep intervals in monitoring tasks
                pass  # Implementation depends on processor structure
                
        except Exception as e:
            logger.error(f"CPU optimization error: {e}")
            
    async def _optimize_gpu_usage(self):
        """Optimize GPU usage without switching to CPU"""
        try:
            # Reduce GPU memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"GPU optimization error: {e}")
            
    async def _manage_processor_features(self):
        """Manage processor features based on current configuration"""
        while self.is_monitoring:
            try:
                if not self.processor:
                    await asyncio.sleep(5)
                    continue
                    
                # Apply current configuration to processor
                await self._apply_feature_configuration()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Feature management error: {e}")
                await asyncio.sleep(10)
                
    async def _apply_feature_configuration(self):
        """Apply current feature configuration to the processor"""
        try:
            # This would interface with the actual processor
            # Implementation depends on processor structure
            pass
            
        except Exception as e:
            logger.error(f"Feature configuration error: {e}")
            
    async def _cleanup_data(self):
        """Periodic data cleanup to manage storage"""
        while self.is_monitoring:
            try:
                if self.config.auto_cleanup_enabled:
                    await self._perform_data_cleanup()
                    
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")
                await asyncio.sleep(300)
                
    async def _perform_data_cleanup(self):
        """Perform data cleanup operations"""
        try:
            if not self.processor:
                return
                
            # Clean up old metrics
            if len(self.metrics_history) > 500:
                self.metrics_history = self.metrics_history[-500:]
                
            # Clean up processor data if available
            if hasattr(self.processor, 'memory_manager'):
                self.processor.memory_manager.cleanup_old_data()
                
            logger.debug("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Data cleanup performance error: {e}")
            
    async def _emergency_monitoring(self):
        """Monitor for emergency shutdown conditions"""
        while self.is_monitoring:
            try:
                if len(self.metrics_history) > 0:
                    latest_metrics = self.metrics_history[-1]
                    
                    # Check for emergency conditions
                    if (latest_metrics.memory_usage_gb > 15.0 or 
                        latest_metrics.cpu_usage > 95.0):
                        
                        if not self.emergency_shutdown_triggered:
                            logger.critical("Emergency shutdown conditions detected")
                            await self._trigger_emergency_shutdown()
                            
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Emergency monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def _trigger_emergency_shutdown(self):
        """Trigger emergency shutdown of non-essential features"""
        try:
            self.emergency_shutdown_triggered = True
            logger.critical("Triggering emergency shutdown")
            
            # Disable all non-essential features
            await self.disable_all_analysis_features()
            
            # Clear all buffers
            await self._emergency_memory_cleanup()
            
            # Notify that manual intervention may be required
            logger.critical("Emergency shutdown complete - manual intervention may be required")
            
        except Exception as e:
            logger.error(f"Emergency shutdown error: {e}")
            
    # Feature control methods
    async def enable_feature(self, feature_name: str):
        """Enable a specific processor feature"""
        try:
            if feature_name in self.feature_states:
                self.feature_states[feature_name] = True
                logger.info(f"Enabled feature: {feature_name}")
                
                # Apply to processor if available
                await self._apply_feature_change(feature_name, True)
                
            else:
                logger.warning(f"Unknown feature: {feature_name}")
                
        except Exception as e:
            logger.error(f"Feature enable error: {e}")
            
    async def disable_feature(self, feature_name: str):
        """Disable a specific processor feature"""
        try:
            if feature_name in self.feature_states:
                self.feature_states[feature_name] = False
                logger.info(f"Disabled feature: {feature_name}")
                
                # Apply to processor if available
                await self._apply_feature_change(feature_name, False)
                
            else:
                logger.warning(f"Unknown feature: {feature_name}")
                
        except Exception as e:
            logger.error(f"Feature disable error: {e}")
            
    async def _apply_feature_change(self, feature_name: str, enabled: bool):
        """Apply feature change to the processor"""
        try:
            if not self.processor:
                return
                
            # This would interface with the actual processor
            # Implementation depends on processor structure
            
            if feature_name == 'mining_analysis' and hasattr(self.processor, 'config'):
                if 'mining_analysis' in self.processor.config:
                    self.processor.config['mining_analysis']['enabled'] = enabled
                    
            # Add more feature-specific implementations as needed
            
        except Exception as e:
            logger.error(f"Feature change application error: {e}")
            
    async def disable_all_analysis_features(self):
        """Disable all analysis features to reduce system load"""
        analysis_features = [
            'mining_analysis',
            'block_timing',
            'nonce_sequences',
            'difficulty_tracking'
        ]
        
        for feature in analysis_features:
            await self.disable_feature(feature)
            
    async def enable_all_analysis_features(self):
        """Enable all analysis features"""
        analysis_features = [
            'mining_analysis',
            'block_timing',
            'nonce_sequences',
            'difficulty_tracking'
        ]
        
        for feature in analysis_features:
            await self.enable_feature(feature)
            
    def get_current_status(self) -> Dict:
        """Get current status of all features and system metrics"""
        try:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else SystemMetrics()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'monitoring_active': self.is_monitoring,
                'emergency_shutdown': self.emergency_shutdown_triggered,
                'feature_states': self.feature_states.copy(),
                'system_metrics': {
                    'cpu_usage': latest_metrics.cpu_usage,
                    'memory_usage_gb': latest_metrics.memory_usage_gb,
                    'gpu_usage': latest_metrics.gpu_usage,
                    'disk_usage_gb': latest_metrics.disk_usage_gb
                },
                'configuration': {
                    'max_memory_gb': self.config.max_memory_usage_gb,
                    'max_cpu_percent': self.config.max_cpu_usage_percent,
                    'max_gpu_percent': self.config.max_gpu_usage_percent
                },
                'thresholds': self.system_thresholds.copy()
            }
            
        except Exception as e:
            logger.error(f"Status retrieval error: {e}")
            return {'error': str(e)}
            
    def update_configuration(self, new_config: Dict):
        """Update processor configuration"""
        try:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Updated config: {key} = {value}")
                    
        except Exception as e:
            logger.error(f"Configuration update error: {e}")
            
    def update_thresholds(self, new_thresholds: Dict):
        """Update system resource thresholds"""
        try:
            self.system_thresholds.update(new_thresholds)
            logger.info(f"Updated thresholds: {new_thresholds}")
            
        except Exception as e:
            logger.error(f"Threshold update error: {e}")
            
    def save_configuration(self, file_path: str = "btc_processor_control_config.json"):
        """Save current configuration to file"""
        try:
            config_data = {
                'feature_states': self.feature_states,
                'thresholds': self.system_thresholds,
                'processor_config': {
                    'max_memory_usage_gb': self.config.max_memory_usage_gb,
                    'max_cpu_usage_percent': self.config.max_cpu_usage_percent,
                    'max_gpu_usage_percent': self.config.max_gpu_usage_percent,
                    'auto_cleanup_enabled': self.config.auto_cleanup_enabled
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Configuration save error: {e}")
            
    def load_configuration(self, file_path: str = "btc_processor_control_config.json"):
        """Load configuration from file"""
        try:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                    
                if 'feature_states' in config_data:
                    self.feature_states.update(config_data['feature_states'])
                    
                if 'thresholds' in config_data:
                    self.system_thresholds.update(config_data['thresholds'])
                    
                if 'processor_config' in config_data:
                    for key, value in config_data['processor_config'].items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                            
                logger.info(f"Configuration loaded from {file_path}")
                
        except Exception as e:
            logger.error(f"Configuration load error: {e}") 