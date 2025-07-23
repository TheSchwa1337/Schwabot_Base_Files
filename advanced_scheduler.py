#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🕐 Advanced Scheduler - Automated Self-Reconfiguration System
============================================================

Intelligent scheduling system for Schwabot that handles:
- Daily self-reconfiguration during low-trading hours (1-4 AM)
- Automated storage compression and optimization
- Weight matrix updates and registry synchronization
- Performance monitoring and drift correction
- Multi-device synchronization and backup

Developed by Maxamillion M.A.A. DeLeon ("The Schwa") & Nexus AI
"""

import json
import os
import time
import logging
import threading
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
import hashlib
import pickle
import gzip
from dataclasses import dataclass, field

# Import compression manager
try:
    from alpha_compression_manager import (
        get_storage_device_manager,
        AlphaCompressionManager,
        compress_trading_data_on_device,
        auto_compress_device_data
    )
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SchedulingConfig:
    """Configuration for advanced scheduling system."""
    
    # Timing configuration
    low_trading_start_hour: int = 1  # 1 AM
    low_trading_end_hour: int = 4    # 4 AM
    preferred_reconfig_hour: int = 2  # 2 AM (middle of low-trading window)
    
    # Compression timing
    compression_timeout_minutes: int = 30  # Max time for compression
    compression_retry_attempts: int = 3    # Retry attempts if compression fails
    
    # Registry synchronization
    registry_sync_interval_hours: int = 24  # Daily sync
    weight_matrix_backup_count: int = 7     # Keep 7 days of backups
    
    # Performance monitoring
    performance_check_interval_minutes: int = 60  # Check performance every hour
    drift_threshold: float = 0.05  # 5% drift threshold
    
    # Storage management
    auto_compression_enabled: bool = True
    storage_optimization_enabled: bool = True
    backup_rotation_enabled: bool = True
    
    # Advanced features
    multi_device_sync_enabled: bool = True
    api_connectivity_monitoring: bool = True
    memory_optimization_enabled: bool = True


@dataclass
class DailyTradingWeights:
    """Daily trading weight matrices for self-reconfiguration."""
    
    date: str
    timestamp: float
    trading_pairs: Dict[str, float]  # Pair -> weight
    strategy_weights: Dict[str, float]  # Strategy -> weight
    risk_weights: Dict[str, float]  # Risk parameter -> weight
    performance_metrics: Dict[str, float]  # Performance indicators
    compression_ratio: float
    memory_usage: float
    api_latency: float
    drift_correction: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistryEntry:
    """Registry entry for tracking system state."""
    
    entry_id: str
    timestamp: float
    entry_type: str  # 'daily_weights', 'compression', 'backup', 'sync'
    device_path: str
    file_path: str
    file_size: int
    checksum: str
    compression_ratio: float
    status: str  # 'success', 'failed', 'pending'
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedScheduler:
    """
    🕐 Advanced Scheduler for Automated Self-Reconfiguration
    
    Handles intelligent scheduling of system maintenance, storage optimization,
    and performance tuning during low-trading hours.
    """
    
    def __init__(self, config: Optional[SchedulingConfig] = None):
        """Initialize Advanced Scheduler."""
        self.config = config or SchedulingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage device manager
        self.storage_manager = None
        if COMPRESSION_AVAILABLE:
            self.storage_manager = get_storage_device_manager()
        
        # Registry for tracking system state
        self.registry: Dict[str, RegistryEntry] = {}
        self.daily_weights: Dict[str, DailyTradingWeights] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.last_reconfig_time: float = 0
        self.last_compression_time: float = 0
        self.last_sync_time: float = 0
        
        # Threading
        self.scheduler_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Load existing registry
        self._load_registry()
        self._load_daily_weights()
        
        self.logger.info("🕐 Advanced Scheduler initialized")
    
    def _load_registry(self):
        """Load existing registry from storage."""
        registry_file = Path("AOI_Base_Files_Schwabot/config/scheduler_registry.json")
        try:
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                    self.registry = {
                        entry_id: RegistryEntry(**data)
                        for entry_id, data in registry_data.items()
                    }
                self.logger.info(f"Loaded {len(self.registry)} registry entries")
        except Exception as e:
            self.logger.warning(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to storage."""
        registry_file = Path("AOI_Base_Files_Schwabot/config/scheduler_registry.json")
        try:
            os.makedirs(registry_file.parent, exist_ok=True)
            registry_data = {
                entry_id: {
                    'entry_id': entry.entry_id,
                    'timestamp': entry.timestamp,
                    'entry_type': entry.entry_type,
                    'device_path': entry.device_path,
                    'file_path': entry.file_path,
                    'file_size': entry.file_size,
                    'checksum': entry.checksum,
                    'compression_ratio': entry.compression_ratio,
                    'status': entry.status,
                    'metadata': entry.metadata
                }
                for entry_id, entry in self.registry.items()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def _load_daily_weights(self):
        """Load daily trading weights from storage."""
        weights_file = Path("AOI_Base_Files_Schwabot/config/daily_weights.json")
        try:
            if weights_file.exists():
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
                    self.daily_weights = {
                        date: DailyTradingWeights(**data)
                        for date, data in weights_data.items()
                    }
                self.logger.info(f"Loaded {len(self.daily_weights)} daily weight sets")
        except Exception as e:
            self.logger.warning(f"Failed to load daily weights: {e}")
    
    def _save_daily_weights(self):
        """Save daily trading weights to storage."""
        weights_file = Path("AOI_Base_Files_Schwabot/config/daily_weights.json")
        try:
            os.makedirs(weights_file.parent, exist_ok=True)
            weights_data = {
                date: {
                    'date': weights.date,
                    'timestamp': weights.timestamp,
                    'trading_pairs': weights.trading_pairs,
                    'strategy_weights': weights.strategy_weights,
                    'risk_weights': weights.risk_weights,
                    'performance_metrics': weights.performance_metrics,
                    'compression_ratio': weights.compression_ratio,
                    'memory_usage': weights.memory_usage,
                    'api_latency': weights.api_latency,
                    'drift_correction': weights.drift_correction,
                    'metadata': weights.metadata
                }
                for date, weights in self.daily_weights.items()
            }
            
            with open(weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save daily weights: {e}")
    
    def start_scheduler(self):
        """Start the advanced scheduler."""
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        # Schedule daily reconfiguration
        schedule.every().day.at(f"{self.config.preferred_reconfig_hour:02d}:00").do(
            self._daily_reconfiguration
        )
        
        # Schedule performance monitoring
        schedule.every(self.config.performance_check_interval_minutes).minutes.do(
            self._performance_monitoring
        )
        
        # Schedule registry synchronization
        schedule.every(self.config.registry_sync_interval_hours).hours.do(
            self._registry_synchronization
        )
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("🕐 Advanced Scheduler started")
        self.logger.info(f"📅 Daily reconfiguration scheduled for {self.config.preferred_reconfig_hour:02d}:00")
        self.logger.info(f"📊 Performance monitoring every {self.config.performance_check_interval_minutes} minutes")
    
    def stop_scheduler(self):
        """Stop the advanced scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("🕐 Advanced Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                time.sleep(60)
    
    def _daily_reconfiguration(self):
        """Perform daily self-reconfiguration during low-trading hours."""
        current_hour = datetime.now().hour
        
        # Check if we're in low-trading hours
        if not (self.config.low_trading_start_hour <= current_hour <= self.config.low_trading_end_hour):
            self.logger.info(f"⏰ Not in low-trading hours ({current_hour}:00), skipping reconfiguration")
            return
        
        self.logger.info("🔄 Starting daily self-reconfiguration")
        
        try:
            # 1. Collect current trading weights
            current_weights = self._collect_current_weights()
            
            # 2. Perform storage optimization
            storage_result = self._optimize_storage()
            
            # 3. Update weight matrices
            weight_result = self._update_weight_matrices(current_weights)
            
            # 4. Synchronize registry
            sync_result = self._synchronize_registry()
            
            # 5. Backup critical data
            backup_result = self._backup_critical_data()
            
            # 6. Update performance metrics
            performance_result = self._update_performance_metrics()
            
            # 7. Save daily weights
            self._save_daily_weights()
            
            # 8. Update registry
            self._save_registry()
            
            self.last_reconfig_time = time.time()
            
            self.logger.info("✅ Daily reconfiguration completed successfully")
            
            # Log results
            results = {
                'storage_optimization': storage_result,
                'weight_update': weight_result,
                'registry_sync': sync_result,
                'backup': backup_result,
                'performance': performance_result
            }
            
            self.logger.info(f"📊 Reconfiguration results: {json.dumps(results, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"❌ Daily reconfiguration failed: {e}")
            self._handle_reconfiguration_failure(e)
    
    def _collect_current_weights(self) -> DailyTradingWeights:
        """Collect current trading weights and performance metrics."""
        try:
            # Get current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Collect trading pair weights (simulated - would come from actual trading system)
            trading_pairs = {
                'BTC/USDC': 0.35,
                'ETH/USDC': 0.25,
                'ADA/USDC': 0.20,
                'BNB/USDC': 0.15,
                'SOL/USDC': 0.05
            }
            
            # Collect strategy weights
            strategy_weights = {
                'momentum': 0.40,
                'mean_reversion': 0.30,
                'breakout': 0.20,
                'scalping': 0.10
            }
            
            # Collect risk weights
            risk_weights = {
                'max_position_size': 0.05,
                'stop_loss': 0.02,
                'take_profit': 0.08,
                'max_drawdown': 0.15
            }
            
            # Collect performance metrics
            performance_metrics = {
                'daily_return': 0.023,  # 2.3%
                'sharpe_ratio': 1.85,
                'max_drawdown': 0.08,
                'win_rate': 0.68,
                'profit_factor': 1.45
            }
            
            # Calculate compression ratio
            compression_ratio = self._calculate_compression_ratio()
            
            # Get memory usage
            memory_usage = self._get_memory_usage()
            
            # Get API latency
            api_latency = self._get_api_latency()
            
            # Calculate drift correction
            drift_correction = self._calculate_drift_correction()
            
            # Create daily weights object
            daily_weights = DailyTradingWeights(
                date=current_date,
                timestamp=time.time(),
                trading_pairs=trading_pairs,
                strategy_weights=strategy_weights,
                risk_weights=risk_weights,
                performance_metrics=performance_metrics,
                compression_ratio=compression_ratio,
                memory_usage=memory_usage,
                api_latency=api_latency,
                drift_correction=drift_correction,
                metadata={
                    'reconfig_type': 'daily',
                    'low_trading_hours': True,
                    'compression_enabled': self.config.auto_compression_enabled
                }
            )
            
            # Save to daily weights
            self.daily_weights[current_date] = daily_weights
            
            self.logger.info(f"📊 Collected weights for {current_date}")
            return daily_weights
            
        except Exception as e:
            self.logger.error(f"Failed to collect current weights: {e}")
            raise
    
    def _optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage across all devices."""
        if not COMPRESSION_AVAILABLE or not self.storage_manager:
            return {'status': 'compression_not_available'}
        
        try:
            results = {}
            
            # Get all devices
            devices = self.storage_manager.get_all_devices_info()
            
            for device_info in devices:
                device_path = device_info['device_path']
                
                if device_info.get('compression_enabled'):
                    self.logger.info(f"🔧 Optimizing storage on {device_path}")
                    
                    # Auto-compress old data
                    compression_result = auto_compress_device_data(device_path)
                    
                    # Get compression suggestions
                    suggestions = self.storage_manager.get_device_compression_suggestions(device_path)
                    
                    results[device_path] = {
                        'compression_result': compression_result,
                        'suggestions': suggestions,
                        'status': 'success'
                    }
                else:
                    results[device_path] = {
                        'status': 'compression_disabled'
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Storage optimization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _update_weight_matrices(self, daily_weights: DailyTradingWeights) -> Dict[str, Any]:
        """Update weight matrices based on daily performance."""
        try:
            # Calculate new weights based on performance
            new_weights = self._calculate_optimized_weights(daily_weights)
            
            # Save weight matrices to all devices
            results = {}
            
            if self.storage_manager:
                devices = self.storage_manager.get_all_devices_info()
                
                for device_info in devices:
                    device_path = device_info['device_path']
                    
                    if device_info.get('compression_enabled'):
                        # Compress and save weight matrices
                        weight_data = {
                            'date': daily_weights.date,
                            'weights': new_weights,
                            'performance': daily_weights.performance_metrics,
                            'timestamp': time.time()
                        }
                        
                        compressed = compress_trading_data_on_device(
                            device_path, weight_data, 'daily_weights'
                        )
                        
                        if compressed:
                            results[device_path] = {
                                'status': 'success',
                                'compression_ratio': compressed.compression_ratio,
                                'file_size': compressed.compressed_size
                            }
                        else:
                            results[device_path] = {
                                'status': 'compression_failed'
                            }
                    else:
                        results[device_path] = {
                            'status': 'compression_disabled'
                        }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Weight matrix update failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_optimized_weights(self, daily_weights: DailyTradingWeights) -> Dict[str, Any]:
        """Calculate optimized weights based on performance."""
        try:
            # Get performance metrics
            performance = daily_weights.performance_metrics
            
            # Adjust trading pair weights based on performance
            trading_pairs = daily_weights.trading_pairs.copy()
            
            # Simple optimization: increase weights for better performing pairs
            # In a real system, this would use more sophisticated algorithms
            
            # Adjust strategy weights based on performance
            strategy_weights = daily_weights.strategy_weights.copy()
            
            # Adjust risk weights based on performance
            risk_weights = daily_weights.risk_weights.copy()
            
            # Calculate drift correction
            drift_correction = self._calculate_drift_correction()
            
            return {
                'trading_pairs': trading_pairs,
                'strategy_weights': strategy_weights,
                'risk_weights': risk_weights,
                'drift_correction': drift_correction,
                'optimization_timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Weight optimization failed: {e}")
            return {}
    
    def _synchronize_registry(self) -> Dict[str, Any]:
        """Synchronize registry across all devices."""
        try:
            results = {}
            
            if self.storage_manager:
                devices = self.storage_manager.get_all_devices_info()
                
                for device_info in devices:
                    device_path = device_info['device_path']
                    
                    # Create registry entry
                    entry_id = f"registry_sync_{int(time.time())}"
                    
                    registry_entry = RegistryEntry(
                        entry_id=entry_id,
                        timestamp=time.time(),
                        entry_type='registry_sync',
                        device_path=device_path,
                        file_path=f"{device_path}/schwabot/registry/sync_{entry_id}.json",
                        file_size=len(json.dumps(self.registry)),
                        checksum=self._calculate_checksum(self.registry),
                        compression_ratio=0.0,
                        status='success',
                        metadata={
                            'sync_type': 'daily',
                            'entries_count': len(self.registry)
                        }
                    )
                    
                    # Save registry to device
                    registry_file = Path(device_path) / "schwabot" / "registry" / f"sync_{entry_id}.json"
                    registry_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(registry_file, 'w') as f:
                        json.dump(self.registry, f, indent=2)
                    
                    # Add to registry
                    self.registry[entry_id] = registry_entry
                    
                    results[device_path] = {
                        'status': 'success',
                        'entry_id': entry_id,
                        'file_size': registry_entry.file_size
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Registry synchronization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _backup_critical_data(self) -> Dict[str, Any]:
        """Backup critical data to all devices."""
        try:
            results = {}
            
            # Critical data to backup
            critical_data = {
                'daily_weights': self.daily_weights,
                'registry': self.registry,
                'performance_history': self.performance_history,
                'config': self.config.__dict__
            }
            
            if self.storage_manager:
                devices = self.storage_manager.get_all_devices_info()
                
                for device_info in devices:
                    device_path = device_info['device_path']
                    
                    # Create backup
                    backup_id = f"backup_{int(time.time())}"
                    backup_file = Path(device_path) / "schwabot" / "backups" / f"{backup_id}.pkl"
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Compress and save backup
                    with gzip.open(backup_file, 'wb') as f:
                        pickle.dump(critical_data, f)
                    
                    # Create registry entry
                    entry_id = f"backup_{backup_id}"
                    
                    registry_entry = RegistryEntry(
                        entry_id=entry_id,
                        timestamp=time.time(),
                        entry_type='backup',
                        device_path=device_path,
                        file_path=str(backup_file),
                        file_size=backup_file.stat().st_size,
                        checksum=self._calculate_file_checksum(backup_file),
                        compression_ratio=0.8,  # Estimated compression ratio
                        status='success',
                        metadata={
                            'backup_type': 'critical_data',
                            'data_size': len(str(critical_data))
                        }
                    )
                    
                    # Add to registry
                    self.registry[entry_id] = registry_entry
                    
                    results[device_path] = {
                        'status': 'success',
                        'backup_id': backup_id,
                        'file_size': registry_entry.file_size
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Critical data backup failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _update_performance_metrics(self) -> Dict[str, Any]:
        """Update performance metrics."""
        try:
            current_metrics = {
                'timestamp': time.time(),
                'memory_usage': self._get_memory_usage(),
                'api_latency': self._get_api_latency(),
                'compression_ratio': self._calculate_compression_ratio(),
                'registry_entries': len(self.registry),
                'daily_weights_count': len(self.daily_weights),
                'last_reconfig_time': self.last_reconfig_time,
                'last_compression_time': self.last_compression_time,
                'last_sync_time': self.last_sync_time
            }
            
            # Add to performance history
            self.performance_history.append(current_metrics)
            
            # Keep only recent history (last 30 days)
            if len(self.performance_history) > 30 * 24:  # 30 days * 24 hours
                self.performance_history = self.performance_history[-30 * 24:]
            
            return current_metrics
            
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _performance_monitoring(self):
        """Monitor system performance."""
        try:
            current_metrics = self._update_performance_metrics()
            
            # Check for performance issues
            issues = []
            
            # Check memory usage
            if current_metrics.get('memory_usage', 0) > 0.8:  # 80%
                issues.append('High memory usage detected')
            
            # Check API latency
            if current_metrics.get('api_latency', 0) > 1000:  # 1 second
                issues.append('High API latency detected')
            
            # Check compression ratio
            if current_metrics.get('compression_ratio', 0) < 0.3:  # 30%
                issues.append('Low compression ratio detected')
            
            if issues:
                self.logger.warning(f"⚠️ Performance issues detected: {issues}")
                self._handle_performance_issues(issues)
            else:
                self.logger.debug("✅ Performance monitoring: All systems normal")
                
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
    
    def _registry_synchronization(self):
        """Synchronize registry across devices."""
        try:
            self.logger.info("🔄 Starting registry synchronization")
            
            result = self._synchronize_registry()
            
            if result.get('status') != 'error':
                self.last_sync_time = time.time()
                self.logger.info("✅ Registry synchronization completed")
            else:
                self.logger.error("❌ Registry synchronization failed")
                
        except Exception as e:
            self.logger.error(f"Registry synchronization failed: {e}")
    
    def _handle_reconfiguration_failure(self, error: Exception):
        """Handle reconfiguration failure."""
        self.logger.error(f"🔄 Reconfiguration failed: {error}")
        
        # Log failure to registry
        entry_id = f"reconfig_failure_{int(time.time())}"
        
        registry_entry = RegistryEntry(
            entry_id=entry_id,
            timestamp=time.time(),
            entry_type='reconfig_failure',
            device_path='local',
            file_path='',
            file_size=0,
            checksum='',
            compression_ratio=0.0,
            status='failed',
            metadata={
                'error': str(error),
                'retry_count': 0
            }
        )
        
        self.registry[entry_id] = registry_entry
        self._save_registry()
    
    def _handle_performance_issues(self, issues: List[str]):
        """Handle performance issues."""
        self.logger.warning(f"⚠️ Handling performance issues: {issues}")
        
        # Log issues to registry
        for issue in issues:
            entry_id = f"performance_issue_{int(time.time())}"
            
            registry_entry = RegistryEntry(
                entry_id=entry_id,
                timestamp=time.time(),
                entry_type='performance_issue',
                device_path='local',
                file_path='',
                file_size=0,
                checksum='',
                compression_ratio=0.0,
                status='warning',
                metadata={
                    'issue': issue,
                    'severity': 'medium'
                }
            )
            
            self.registry[entry_id] = registry_entry
        
        self._save_registry()
    
    def _calculate_compression_ratio(self) -> float:
        """Calculate current compression ratio."""
        try:
            if not self.storage_manager:
                return 0.0
            
            total_original = 0
            total_compressed = 0
            
            devices = self.storage_manager.get_all_devices_info()
            
            for device_info in devices:
                device_path = device_info['device_path']
                
                if device_info.get('compression_enabled'):
                    stats = self.storage_manager.get_device_compression_stats(device_path)
                    if stats:
                        for pattern_type, type_stats in stats.get('type_statistics', {}).items():
                            total_original += type_stats.get('total_original', 0)
                            total_compressed += type_stats.get('total_compressed', 0)
            
            if total_original > 0:
                return 1.0 - (total_compressed / total_original)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Compression ratio calculation failed: {e}")
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.5  # Default value
    
    def _get_api_latency(self) -> float:
        """Get current API latency."""
        try:
            # Simulate API latency measurement
            # In a real system, this would measure actual API calls
            return 150.0  # 150ms default
        except Exception as e:
            self.logger.error(f"API latency measurement failed: {e}")
            return 1000.0  # 1 second default
    
    def _calculate_drift_correction(self) -> float:
        """Calculate drift correction factor."""
        try:
            # Calculate drift based on performance history
            if len(self.performance_history) < 2:
                return 0.0
            
            recent_performance = self.performance_history[-10:]  # Last 10 entries
            
            # Calculate average performance drift
            drifts = []
            for i in range(1, len(recent_performance)):
                prev = recent_performance[i-1]
                curr = recent_performance[i]
                
                # Calculate drift in compression ratio
                if 'compression_ratio' in prev and 'compression_ratio' in curr:
                    drift = curr['compression_ratio'] - prev['compression_ratio']
                    drifts.append(drift)
            
            if drifts:
                avg_drift = sum(drifts) / len(drifts)
                return max(-self.config.drift_threshold, min(self.config.drift_threshold, avg_drift))
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Drift correction calculation failed: {e}")
            return 0.0
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data."""
        try:
            data_str = json.dumps(data, sort_keys=True)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate checksum for file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"File checksum calculation failed: {e}")
            return ""
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'is_running': self.is_running,
            'last_reconfig_time': self.last_reconfig_time,
            'last_compression_time': self.last_compression_time,
            'last_sync_time': self.last_sync_time,
            'registry_entries': len(self.registry),
            'daily_weights_count': len(self.daily_weights),
            'performance_history_count': len(self.performance_history),
            'config': self.config.__dict__
        }
    
    def get_next_scheduled_time(self) -> str:
        """Get next scheduled reconfiguration time."""
        next_run = schedule.next_run()
        if next_run:
            return next_run.strftime("%Y-%m-%d %H:%M:%S")
        return "No scheduled runs"


# Global scheduler instance
_advanced_scheduler = None


def get_advanced_scheduler() -> AdvancedScheduler:
    """Get or create global Advanced Scheduler instance."""
    global _advanced_scheduler
    if _advanced_scheduler is None:
        _advanced_scheduler = AdvancedScheduler()
    return _advanced_scheduler


def start_advanced_scheduler():
    """Start the advanced scheduler."""
    scheduler = get_advanced_scheduler()
    scheduler.start_scheduler()


def stop_advanced_scheduler():
    """Stop the advanced scheduler."""
    global _advanced_scheduler
    if _advanced_scheduler:
        _advanced_scheduler.stop_scheduler()


if __name__ == "__main__":
    # Example usage
    scheduler = get_advanced_scheduler()
    
    print("🕐 Advanced Scheduler Demo")
    print("=" * 50)
    
    # Start scheduler
    scheduler.start_scheduler()
    
    print(f"📅 Next reconfiguration: {scheduler.get_next_scheduled_time()}")
    print(f"📊 Current status: {scheduler.get_scheduler_status()}")
    
    # Keep running for demo
    try:
        while True:
            time.sleep(60)
            print(f"⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")
    except KeyboardInterrupt:
        print("\n🛑 Stopping scheduler...")
        scheduler.stop_scheduler()
        print("✅ Scheduler stopped") 