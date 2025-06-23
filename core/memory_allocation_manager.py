#!/usr/bin/env python3
"""Memory Allocation Manager - Intelligent Memory Management System.

This module provides comprehensive memory allocation management including:
- Short, mid, and long-term memory allocation
- Memory key allocation based on data type and importance
- Reflective allocator for BTC hashing data (3.75 minutes)
- User interface settings integration
- Memory state management and optimization
- Integration with all Schwabot core systems
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import os
import hashlib
from pathlib import Path
from collections import defaultdict, deque

# Import core systems
try:
    from core.ops_observability import log_operation, LogLevel
    from core.persistent_state_manager import get_persistent_state_manager, MemoryAllocationType
    from core.demo_memory_core import get_demo_memory_core, MemoryType
    from core.exchange_plumbing import ExchangeType
    CORE_SYSTEMS_AVAILABLE = True
except ImportError:
    CORE_SYSTEMS_AVAILABLE = False

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, safe_format_error, log_safe
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)


class MemoryPriority(Enum):
    """Memory priority levels."""
    CRITICAL = "critical"      # System critical data
    HIGH = "high"             # Important trading data
    MEDIUM = "medium"         # Regular analysis data
    LOW = "low"               # Background data
    ARCHIVE = "archive"       # Historical data


class DataCategory(Enum):
    """Data categories for allocation."""
    BTC_HASHING = "btc_hashing"           # 3.75 minute BTC data
    TRADING_SIGNALS = "trading_signals"   # Trading signals
    MARKET_DATA = "market_data"           # Market data
    RISK_METRICS = "risk_metrics"         # Risk calculations
    PORTFOLIO_STATE = "portfolio_state"   # Portfolio state
    ANALYSIS_RESULTS = "analysis_results" # Analysis results
    SYSTEM_LOGS = "system_logs"           # System logs
    AUDIT_TRAIL = "audit_trail"           # Audit trail
    USER_SETTINGS = "user_settings"       # User interface settings


@dataclass
class MemoryKey:
    """Memory key with allocation metadata."""
    key_id: str
    category: DataCategory
    priority: MemoryPriority
    allocation_type: MemoryAllocationType
    timestamp: datetime
    size_bytes: int
    retention_days: int
    compression_ratio: float = 1.0
    encryption_enabled: bool = True
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryAllocationConfig:
    """Memory allocation configuration."""
    category: DataCategory
    priority: MemoryPriority
    short_term_limit: int
    mid_term_limit: int
    long_term_limit: int
    retention_days: int
    compression_enabled: bool = True
    encryption_enabled: bool = True
    auto_cleanup: bool = True
    user_configurable: bool = True


@dataclass
class MemoryUsage:
    """Memory usage statistics."""
    total_entries: int
    total_size_bytes: int
    short_term_usage: float  # Percentage
    mid_term_usage: float    # Percentage
    long_term_usage: float   # Percentage
    compression_savings: float  # Percentage
    oldest_entry: datetime
    newest_entry: datetime


class ReflectiveAllocator:
    """Reflective allocator for intelligent memory management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reflective allocator."""
        self.config = config or {}
        self.btc_hashing_interval = 3.75 * 60  # 3.75 minutes in seconds
        self.last_btc_allocation = time.time()
        
        # Performance tracking
        self.total_allocations = 0
        self.successful_allocations = 0
        self.failed_allocations = 0
        
        # Memory patterns
        self.allocation_patterns = defaultdict(list)
        self.access_patterns = defaultdict(list)
        
        safe_print("🔄 Reflective Allocator initialized")
    
    def should_allocate_btc_data(self) -> bool:
        """Check if it's time to allocate BTC hashing data."""
        current_time = time.time()
        if current_time - self.last_btc_allocation >= self.btc_hashing_interval:
            self.last_btc_allocation = current_time
            return True
        return False
    
    def get_optimal_allocation_type(self, category: DataCategory, 
                                  priority: MemoryPriority, 
                                  data_size: int) -> MemoryAllocationType:
        """Get optimal allocation type based on data characteristics."""
        try:
            # Critical data goes to long-term
            if priority == MemoryPriority.CRITICAL:
                return MemoryAllocationType.LONG_TERM
            
            # BTC hashing data goes to short-term
            if category == DataCategory.BTC_HASHING:
                return MemoryAllocationType.SHORT_TERM
            
            # Trading signals go to mid-term
            if category == DataCategory.TRADING_SIGNALS:
                return MemoryAllocationType.MID_TERM
            
            # Large data goes to long-term
            if data_size > 1024 * 1024:  # 1MB
                return MemoryAllocationType.LONG_TERM
            
            # Medium priority goes to mid-term
            if priority == MemoryPriority.MEDIUM:
                return MemoryAllocationType.MID_TERM
            
            # Low priority goes to short-term
            if priority == MemoryPriority.LOW:
                return MemoryAllocationType.SHORT_TERM
            
            # Default to mid-term
            return MemoryAllocationType.MID_TERM
            
        except Exception as e:
            safe_print(f"❌ Allocation type determination failed: {safe_format_error(e, 'allocation_type')}")
            return MemoryAllocationType.MID_TERM
    
    def calculate_compression_ratio(self, data: Dict[str, Any]) -> float:
        """Calculate potential compression ratio."""
        try:
            # Simple compression estimation
            data_json = json.dumps(data, separators=(',', ':'))
            original_size = len(data_json.encode())
            
            # Estimate compressed size (rough approximation)
            compressed_size = original_size * 0.7  # Assume 30% compression
            
            return compressed_size / original_size
            
        except Exception as e:
            safe_print(f"⚠️ Compression calculation failed: {safe_format_error(e, 'compression')}")
            return 1.0
    
    def record_allocation_pattern(self, category: DataCategory, 
                                allocation_type: MemoryAllocationType, 
                                success: bool) -> None:
        """Record allocation pattern for learning."""
        try:
            pattern = {
                'timestamp': datetime.now(),
                'category': category.value,
                'allocation_type': allocation_type.value,
                'success': success
            }
            
            self.allocation_patterns[category.value].append(pattern)
            
            # Keep only recent patterns
            if len(self.allocation_patterns[category.value]) > 1000:
                self.allocation_patterns[category.value] = \
                    self.allocation_patterns[category.value][-1000:]
                    
        except Exception as e:
            safe_print(f"⚠️ Pattern recording failed: {safe_format_error(e, 'pattern_record')}")
    
    def get_allocation_recommendations(self) -> Dict[str, Any]:
        """Get allocation recommendations based on patterns."""
        try:
            recommendations = {}
            
            for category, patterns in self.allocation_patterns.items():
                if not patterns:
                    continue
                
                # Analyze success rates by allocation type
                success_rates = defaultdict(list)
                for pattern in patterns[-100:]:  # Last 100 patterns
                    allocation_type = pattern['allocation_type']
                    success_rates[allocation_type].append(pattern['success'])
                
                # Calculate success rates
                category_recommendations = {}
                for allocation_type, successes in success_rates.items():
                    success_rate = sum(successes) / len(successes)
                    category_recommendations[allocation_type] = success_rate
                
                recommendations[category] = category_recommendations
            
            return recommendations
            
        except Exception as e:
            safe_print(f"❌ Recommendation generation failed: {safe_format_error(e, 'recommendations')}")
            return {}


class MemoryAllocationManager:
    """Comprehensive memory allocation management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize memory allocation manager."""
        self.config = config or {}
        self.reflective_allocator = ReflectiveAllocator(config)
        self.memory_keys: Dict[str, MemoryKey] = {}
        self.allocation_configs: Dict[DataCategory, MemoryAllocationConfig] = {}
        
        # Memory usage tracking
        self.memory_usage = {
            MemoryAllocationType.SHORT_TERM: {'entries': 0, 'size_bytes': 0},
            MemoryAllocationType.MID_TERM: {'entries': 0, 'size_bytes': 0},
            MemoryAllocationType.LONG_TERM: {'entries': 0, 'size_bytes': 0}
        }
        
        # User interface settings
        self.ui_settings = self._load_ui_settings()
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.running = False
        self._start_background_cleanup()
        
        safe_print("🧠 Memory Allocation Manager initialized")
    
    def _load_ui_settings(self) -> Dict[str, Any]:
        """Load user interface settings."""
        try:
            settings_file = Path("config/memory_allocation_settings.json")
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    return json.load(f)
            else:
                # Create default settings
                default_settings = {
                    'btc_hashing_enabled': True,
                    'btc_hashing_interval_minutes': 3.75,
                    'auto_cleanup_enabled': True,
                    'compression_enabled': True,
                    'encryption_enabled': True,
                    'memory_limits': {
                        'short_term_mb': 100,
                        'mid_term_mb': 500,
                        'long_term_mb': 1000
                    },
                    'retention_policies': {
                        'short_term_days': 1,
                        'mid_term_days': 7,
                        'long_term_days': 30
                    }
                }
                
                # Save default settings
                settings_file.parent.mkdir(parents=True, exist_ok=True)
                with open(settings_file, 'w') as f:
                    json.dump(default_settings, f, indent=2)
                
                return default_settings
                
        except Exception as e:
            safe_print(f"⚠️ UI settings load failed: {safe_format_error(e, 'ui_settings')}")
            return {}
    
    def _initialize_default_configs(self) -> None:
        """Initialize default allocation configurations."""
        default_configs = {
            DataCategory.BTC_HASHING: MemoryAllocationConfig(
                category=DataCategory.BTC_HASHING,
                priority=MemoryPriority.HIGH,
                short_term_limit=10000,
                mid_term_limit=50000,
                long_term_limit=100000,
                retention_days=1,
                compression_enabled=True,
                encryption_enabled=True,
                auto_cleanup=True,
                user_configurable=True
            ),
            DataCategory.TRADING_SIGNALS: MemoryAllocationConfig(
                category=DataCategory.TRADING_SIGNALS,
                priority=MemoryPriority.CRITICAL,
                short_term_limit=5000,
                mid_term_limit=25000,
                long_term_limit=50000,
                retention_days=7,
                compression_enabled=True,
                encryption_enabled=True,
                auto_cleanup=True,
                user_configurable=True
            ),
            DataCategory.MARKET_DATA: MemoryAllocationConfig(
                category=DataCategory.MARKET_DATA,
                priority=MemoryPriority.HIGH,
                short_term_limit=15000,
                mid_term_limit=75000,
                long_term_limit=150000,
                retention_days=3,
                compression_enabled=True,
                encryption_enabled=True,
                auto_cleanup=True,
                user_configurable=True
            ),
            DataCategory.RISK_METRICS: MemoryAllocationConfig(
                category=DataCategory.RISK_METRICS,
                priority=MemoryPriority.CRITICAL,
                short_term_limit=3000,
                mid_term_limit=15000,
                long_term_limit=30000,
                retention_days=7,
                compression_enabled=True,
                encryption_enabled=True,
                auto_cleanup=True,
                user_configurable=True
            ),
            DataCategory.PORTFOLIO_STATE: MemoryAllocationConfig(
                category=DataCategory.PORTFOLIO_STATE,
                priority=MemoryPriority.HIGH,
                short_term_limit=2000,
                mid_term_limit=10000,
                long_term_limit=20000,
                retention_days=7,
                compression_enabled=True,
                encryption_enabled=True,
                auto_cleanup=True,
                user_configurable=True
            ),
            DataCategory.ANALYSIS_RESULTS: MemoryAllocationConfig(
                category=DataCategory.ANALYSIS_RESULTS,
                priority=MemoryPriority.MEDIUM,
                short_term_limit=5000,
                mid_term_limit=25000,
                long_term_limit=50000,
                retention_days=14,
                compression_enabled=True,
                encryption_enabled=True,
                auto_cleanup=True,
                user_configurable=True
            ),
            DataCategory.SYSTEM_LOGS: MemoryAllocationConfig(
                category=DataCategory.SYSTEM_LOGS,
                priority=MemoryPriority.LOW,
                short_term_limit=10000,
                mid_term_limit=50000,
                long_term_limit=100000,
                retention_days=3,
                compression_enabled=True,
                encryption_enabled=True,
                auto_cleanup=True,
                user_configurable=True
            ),
            DataCategory.AUDIT_TRAIL: MemoryAllocationConfig(
                category=DataCategory.AUDIT_TRAIL,
                priority=MemoryPriority.CRITICAL,
                short_term_limit=1000,
                mid_term_limit=5000,
                long_term_limit=10000,
                retention_days=365,
                compression_enabled=False,
                encryption_enabled=True,
                auto_cleanup=False,
                user_configurable=False
            ),
            DataCategory.USER_SETTINGS: MemoryAllocationConfig(
                category=DataCategory.USER_SETTINGS,
                priority=MemoryPriority.HIGH,
                short_term_limit=100,
                mid_term_limit=500,
                long_term_limit=1000,
                retention_days=365,
                compression_enabled=False,
                encryption_enabled=True,
                auto_cleanup=False,
                user_configurable=True
            )
        }
        
        for category, config in default_configs.items():
            self.allocation_configs[category] = config
    
    def allocate_memory(self, data: Dict[str, Any], category: DataCategory, 
                       priority: Optional[MemoryPriority] = None) -> Optional[str]:
        """Allocate memory for data."""
        try:
            # Get configuration
            config = self.allocation_configs.get(category)
            if not config:
                safe_print(f"❌ No configuration for category: {category.value}")
                return None
            
            # Use provided priority or default from config
            if priority is None:
                priority = config.priority
            
            # Calculate data size
            data_json = json.dumps(data, separators=(',', ':'))
            data_size = len(data_json.encode())
            
            # Get optimal allocation type
            allocation_type = self.reflective_allocator.get_optimal_allocation_type(
                category, priority, data_size
            )
            
            # Check memory limits
            if not self._check_memory_limits(allocation_type, data_size):
                safe_print(f"⚠️ Memory limit reached for {allocation_type.value}")
                return None
            
            # Generate memory key
            key_id = str(uuid.uuid4())
            compression_ratio = self.reflective_allocator.calculate_compression_ratio(data)
            
            memory_key = MemoryKey(
                key_id=key_id,
                category=category,
                priority=priority,
                allocation_type=allocation_type,
                timestamp=datetime.now(),
                size_bytes=data_size,
                retention_days=config.retention_days,
                compression_ratio=compression_ratio,
                encryption_enabled=config.encryption_enabled,
                metadata={'config_priority': config.priority.value}
            )
            
            # Store in persistent state
            if CORE_SYSTEMS_AVAILABLE:
                persistent_manager = get_persistent_state_manager()
                entry_id = persistent_manager.memory_manager.allocate_memory(
                    data=data,
                    data_type=category.value,
                    allocation_type=allocation_type
                )
                
                if entry_id:
                    # Store memory key
                    self.memory_keys[key_id] = memory_key
                    
                    # Update memory usage
                    self._update_memory_usage(allocation_type, data_size, 1)
                    
                    # Record pattern
                    self.reflective_allocator.record_allocation_pattern(
                        category, allocation_type, True
                    )
                    
                    # Log operation
                    log_operation(
                        operation="memory_allocation",
                        component="memory_allocation_manager",
                        level=LogLevel.INFO,
                        success=True,
                        key_id=key_id,
                        category=category.value,
                        allocation_type=allocation_type.value,
                        data_size=data_size
                    )
                    
                    safe_print(f"✅ Memory allocated: {key_id[:8]}... ({category.value})")
                    return key_id
                else:
                    # Record failed pattern
                    self.reflective_allocator.record_allocation_pattern(
                        category, allocation_type, False
                    )
                    
                    safe_print(f"❌ Persistent storage failed for {category.value}")
                    return None
            else:
                # Fallback to in-memory storage
                self.memory_keys[key_id] = memory_key
                self._update_memory_usage(allocation_type, data_size, 1)
                
                safe_print(f"✅ Memory allocated (in-memory): {key_id[:8]}... ({category.value})")
                return key_id
                
        except Exception as e:
            safe_print(f"❌ Memory allocation failed: {safe_format_error(e, 'memory_allocate')}")
            return None
    
    def _check_memory_limits(self, allocation_type: MemoryAllocationType, data_size: int) -> bool:
        """Check if memory allocation is within limits."""
        try:
            # Get UI settings limits
            limits = self.ui_settings.get('memory_limits', {})
            
            if allocation_type == MemoryAllocationType.SHORT_TERM:
                limit_mb = limits.get('short_term_mb', 100)
            elif allocation_type == MemoryAllocationType.MID_TERM:
                limit_mb = limits.get('mid_term_mb', 500)
            elif allocation_type == MemoryAllocationType.LONG_TERM:
                limit_mb = limits.get('long_term_mb', 1000)
            else:
                return True  # No limit for other types
            
            # Convert to bytes
            limit_bytes = limit_mb * 1024 * 1024
            
            # Check current usage
            current_usage = self.memory_usage[allocation_type]['size_bytes']
            
            return (current_usage + data_size) <= limit_bytes
            
        except Exception as e:
            safe_print(f"⚠️ Memory limit check failed: {safe_format_error(e, 'memory_limit')}")
            return True  # Allow allocation on error
    
    def _update_memory_usage(self, allocation_type: MemoryAllocationType, 
                           size_bytes: int, entry_count: int) -> None:
        """Update memory usage statistics."""
        try:
            usage = self.memory_usage[allocation_type]
            usage['size_bytes'] += size_bytes
            usage['entries'] += entry_count
            
        except Exception as e:
            safe_print(f"⚠️ Memory usage update failed: {safe_format_error(e, 'usage_update')}")
    
    def get_memory_key(self, key_id: str) -> Optional[MemoryKey]:
        """Get memory key by ID."""
        memory_key = self.memory_keys.get(key_id)
        if memory_key:
            # Update access statistics
            memory_key.access_count += 1
            memory_key.last_accessed = datetime.now()
            
            # Record access pattern
            self.reflective_allocator.access_patterns[memory_key.category.value].append({
                'timestamp': datetime.now(),
                'key_id': key_id,
                'allocation_type': memory_key.allocation_type.value
            })
        
        return memory_key
    
    def get_memory_usage(self) -> MemoryUsage:
        """Get comprehensive memory usage statistics."""
        try:
            total_entries = sum(usage['entries'] for usage in self.memory_usage.values())
            total_size_bytes = sum(usage['size_bytes'] for usage in self.memory_usage.values())
            
            # Calculate usage percentages
            limits = self.ui_settings.get('memory_limits', {})
            short_term_limit = limits.get('short_term_mb', 100) * 1024 * 1024
            mid_term_limit = limits.get('mid_term_mb', 500) * 1024 * 1024
            long_term_limit = limits.get('long_term_mb', 1000) * 1024 * 1024
            
            short_term_usage = (self.memory_usage[MemoryAllocationType.SHORT_TERM]['size_bytes'] / short_term_limit) * 100
            mid_term_usage = (self.memory_usage[MemoryAllocationType.MID_TERM]['size_bytes'] / mid_term_limit) * 100
            long_term_usage = (self.memory_usage[MemoryAllocationType.LONG_TERM]['size_bytes'] / long_term_limit) * 100
            
            # Calculate compression savings
            total_compressed_size = sum(
                key.size_bytes * key.compression_ratio 
                for key in self.memory_keys.values()
            )
            compression_savings = ((total_size_bytes - total_compressed_size) / total_size_bytes) * 100 if total_size_bytes > 0 else 0
            
            # Get oldest and newest entries
            timestamps = [key.timestamp for key in self.memory_keys.values()]
            oldest_entry = min(timestamps) if timestamps else datetime.now()
            newest_entry = max(timestamps) if timestamps else datetime.now()
            
            return MemoryUsage(
                total_entries=total_entries,
                total_size_bytes=total_size_bytes,
                short_term_usage=short_term_usage,
                mid_term_usage=mid_term_usage,
                long_term_usage=long_term_usage,
                compression_savings=compression_savings,
                oldest_entry=oldest_entry,
                newest_entry=newest_entry
            )
            
        except Exception as e:
            safe_print(f"❌ Memory usage calculation failed: {safe_format_error(e, 'usage_calc')}")
            return MemoryUsage(0, 0, 0.0, 0.0, 0.0, 0.0, datetime.now(), datetime.now())
    
    def update_ui_settings(self, new_settings: Dict[str, Any]) -> bool:
        """Update user interface settings."""
        try:
            # Update settings
            self.ui_settings.update(new_settings)
            
            # Save to file
            settings_file = Path("config/memory_allocation_settings.json")
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(settings_file, 'w') as f:
                json.dump(self.ui_settings, f, indent=2)
            
            safe_print("✅ UI settings updated")
            return True
            
        except Exception as e:
            safe_print(f"❌ UI settings update failed: {safe_format_error(e, 'ui_update')}")
            return False
    
    def _start_background_cleanup(self) -> None:
        """Start background cleanup thread."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_worker(self) -> None:
        """Background cleanup worker."""
        while self.running:
            try:
                self._perform_cleanup()
                time.sleep(3600)  # Run every hour
            except Exception as e:
                safe_print(f"⚠️ Cleanup error: {safe_format_error(e, 'cleanup')}")
                time.sleep(3600)
    
    def _perform_cleanup(self) -> None:
        """Perform memory cleanup."""
        try:
            current_time = datetime.now()
            keys_to_remove = []
            
            for key_id, memory_key in self.memory_keys.items():
                # Check retention policy
                retention_until = memory_key.timestamp + timedelta(days=memory_key.retention_days)
                
                if current_time > retention_until:
                    keys_to_remove.append(key_id)
            
            # Remove expired keys
            for key_id in keys_to_remove:
                memory_key = self.memory_keys.pop(key_id)
                self._update_memory_usage(memory_key.allocation_type, -memory_key.size_bytes, -1)
            
            if keys_to_remove:
                safe_print(f"🗑️ Cleaned up {len(keys_to_remove)} expired memory keys")
                
        except Exception as e:
            safe_print(f"❌ Cleanup failed: {safe_format_error(e, 'cleanup_perform')}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            memory_usage = self.get_memory_usage()
            recommendations = self.reflective_allocator.get_allocation_recommendations()
            
            return {
                'memory_usage': asdict(memory_usage),
                'allocation_recommendations': recommendations,
                'total_memory_keys': len(self.memory_keys),
                'allocation_configs': {
                    category.value: asdict(config) 
                    for category, config in self.allocation_configs.items()
                },
                'ui_settings': self.ui_settings,
                'reflective_allocator_stats': {
                    'total_allocations': self.reflective_allocator.total_allocations,
                    'successful_allocations': self.reflective_allocator.successful_allocations,
                    'failed_allocations': self.reflective_allocator.failed_allocations,
                    'success_rate': self.reflective_allocator.successful_allocations / max(self.reflective_allocator.total_allocations, 1)
                }
            }
            
        except Exception as e:
            safe_print(f"❌ Status generation failed: {safe_format_error(e, 'status')}")
            return {}


# Global memory allocation manager instance
memory_allocation_manager = MemoryAllocationManager()


# Convenience functions for external access
def get_memory_allocation_manager() -> MemoryAllocationManager:
    """Get global memory allocation manager instance."""
    return memory_allocation_manager


def allocate_memory(data: Dict[str, Any], category: DataCategory, 
                   priority: Optional[MemoryPriority] = None) -> Optional[str]:
    """Allocate memory for data."""
    return memory_allocation_manager.allocate_memory(data, category, priority)


def get_memory_key(key_id: str) -> Optional[MemoryKey]:
    """Get memory key by ID."""
    return memory_allocation_manager.get_memory_key(key_id)


def get_memory_usage() -> MemoryUsage:
    """Get memory usage statistics."""
    return memory_allocation_manager.get_memory_usage()


def update_ui_settings(new_settings: Dict[str, Any]) -> bool:
    """Update user interface settings."""
    return memory_allocation_manager.update_ui_settings(new_settings)


def get_memory_allocation_status() -> Dict[str, Any]:
    """Get memory allocation system status."""
    return memory_allocation_manager.get_system_status()


# Example usage
if __name__ == "__main__":
    # Test memory allocation manager
    print("🧪 Testing Memory Allocation Manager...")
    
    # Test BTC hashing data allocation
    btc_data = {
        'btc_price': 50000.0,
        'hash_rate': 150.5,
        'difficulty': 25.6,
        'block_height': 800000,
        'timestamp': datetime.now().isoformat()
    }
    
    key_id = allocate_memory(btc_data, DataCategory.BTC_HASHING)
    print(f"✅ BTC data allocated: {key_id}")
    
    # Test trading signals allocation
    trading_data = {
        'signal_type': 'buy',
        'confidence': 0.85,
        'price_target': 52000.0,
        'timestamp': datetime.now().isoformat()
    }
    
    trading_key = allocate_memory(trading_data, DataCategory.TRADING_SIGNALS)
    print(f"✅ Trading signal allocated: {trading_key}")
    
    # Get memory usage
    usage = get_memory_usage()
    print(f"✅ Memory usage: {usage.total_entries} entries, {usage.total_size_bytes} bytes")
    
    # Get system status
    status = get_memory_allocation_status()
    print(f"✅ System status: {status}")
    
    print("✅ Memory Allocation Manager test completed") 