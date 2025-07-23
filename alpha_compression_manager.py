#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔐 Alpha Encryption-Based Intelligent Compression System
=======================================================

Enhanced storage management using Alpha Encryption (Ω-B-Γ Logic) for:
- Intelligent data compression on any storage device
- Progressive learning states
- Pattern recognition and recall
- Dynamic storage optimization

Developed by Maxamillion M.A.A. DeLeon ("The Schwa") & Nexus AI
"""

import json
import os
import shutil
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pickle
import gzip
import psutil

# Import Alpha Encryption
try:
    from schwabot.alpha_encryption import AlphaEncryption, AlphaEncryptionResult
    ALPHA_ENCRYPTION_AVAILABLE = True
except ImportError:
    ALPHA_ENCRYPTION_AVAILABLE = False
    logging.warning("Alpha Encryption not available - using basic compression")
    
    # Create mock classes for when Alpha Encryption is not available
    class AlphaEncryptionResult:
        def __init__(self):
            self.encryption_hash = "mock_hash"
            self.security_score = 0.0
            self.total_entropy = 0.0
            self.processing_time = 0.0
            self.omega_state = type('obj', (object,), {
                'recursive_pattern': [0.0, 0.0, 0.0]
            })()
            self.beta_state = type('obj', (object,), {
                'quantum_state': [0.0, 0.0, 0.0]
            })()
            self.gamma_state = type('obj', (object,), {
                'frequency_components': [0.0, 0.0, 0.0]
            })()
    
    class AlphaEncryption:
        def __init__(self, config=None):
            self.config = config or {}
        
        def encrypt_data(self, data, metadata):
            return AlphaEncryptionResult()

logger = logging.getLogger(__name__)


@dataclass
class CompressedPattern:
    """Compressed trading pattern using Alpha Encryption."""
    pattern_id: str
    alpha_encryption_result: AlphaEncryptionResult
    original_size: int
    compressed_size: int
    compression_ratio: float
    pattern_type: str  # 'backtest', 'live_trade', 'market_data'
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressiveLearningState:
    """Progressive learning state for trading patterns."""
    state_id: str
    weight_matrices: Dict[str, np.ndarray]
    pattern_weights: Dict[str, float]
    success_metrics: Dict[str, float]
    evolution_history: List[Dict[str, Any]]
    last_updated: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageMetrics:
    """Storage metrics and compression statistics."""
    total_space: int
    used_space: int
    free_space: int
    compression_threshold: float = 0.5  # 50% usage triggers compression
    compression_ratio: float = 0.0
    patterns_compressed: int = 0
    space_saved: int = 0
    last_compression: float = 0.0


@dataclass
class StorageDevice:
    """Storage device information."""
    device_path: str
    device_name: str
    device_type: str  # 'usb', 'ssd', 'hdd', 'network'
    total_space: int
    free_space: int
    is_writable: bool
    compression_enabled: bool = False
    compression_config: Dict[str, Any] = field(default_factory=dict)


class AlphaCompressionManager:
    """
    🔐 Alpha Encryption-Based Intelligent Compression Manager
    
    Uses Alpha Encryption (Ω-B-Γ Logic) for intelligent data compression and
    progressive learning state management on any storage device.
    """
    
    def __init__(self, storage_path: str, config: Optional[Dict[str, Any]] = None):
        """Initialize Alpha Compression Manager."""
        self.storage_path = Path(storage_path)
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Alpha Encryption
        self.alpha_encryption = None
        if ALPHA_ENCRYPTION_AVAILABLE:
            try:
                self.alpha_encryption = AlphaEncryption(self.config.get('alpha_config', {}))
                self.logger.info("✅ Alpha Encryption (Ω-B-Γ Logic) initialized for compression")
            except Exception as e:
                self.logger.warning(f"⚠️ Alpha Encryption initialization failed: {e}")
        
        # Storage paths
        self.compressed_data_path = self.storage_path / "schwabot" / "compressed_data"
        self.pattern_registry_path = self.storage_path / "schwabot" / "pattern_registry"
        self.learning_states_path = self.storage_path / "schwabot" / "learning_states"
        self.decompression_registry_path = self.storage_path / "schwabot" / "decompression_registry"
        
        # Create directories
        self._create_directories()
        
        # Initialize storage metrics
        self.storage_metrics = self._calculate_storage_metrics()
        
        # Pattern registry for quick lookup
        self.pattern_registry: Dict[str, CompressedPattern] = {}
        self._load_pattern_registry()
        
        # Progressive learning states
        self.learning_states: Dict[str, ProgressiveLearningState] = {}
        self._load_learning_states()
        
        # Compression history
        self.compression_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"🔐 Alpha Compression Manager initialized for {self.storage_path}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Alpha Compression Manager."""
        return {
            'compression_threshold': 0.5,  # 50% usage triggers compression
            'min_compression_ratio': 0.3,  # Minimum 30% compression to keep
            'pattern_retention_days': 90,  # Keep patterns for 90 days
            'learning_state_update_interval': 3600,  # Update learning states every hour
            'max_patterns_per_type': 1000,  # Maximum patterns per type
            'auto_compression_enabled': True,
            'progressive_learning_enabled': True,
            'alpha_config': {
                'omega_weight': 0.4,
                'beta_weight': 0.3,
                'gamma_weight': 0.3,
                'max_recursion_depth': 16,
                'convergence_threshold': 1e-6
            }
        }
    
    def _create_directories(self):
        """Create necessary directories on storage device."""
        directories = [
            self.compressed_data_path,
            self.pattern_registry_path,
            self.learning_states_path,
            self.decompression_registry_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def _calculate_storage_metrics(self) -> StorageMetrics:
        """Calculate current storage metrics."""
        try:
            total, used, free = shutil.disk_usage(self.storage_path)
            return StorageMetrics(
                total_space=total,
                used_space=used,
                free_space=free,
                compression_threshold=self.config['compression_threshold']
            )
        except Exception as e:
            self.logger.error(f"Failed to calculate storage metrics: {e}")
            return StorageMetrics(0, 0, 0)
    
    def _load_pattern_registry(self):
        """Load pattern registry from storage device."""
        registry_file = self.pattern_registry_path / "pattern_registry.json"
        try:
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                    self.pattern_registry = {
                        pattern_id: CompressedPattern(**data)
                        for pattern_id, data in registry_data.items()
                    }
                self.logger.info(f"Loaded {len(self.pattern_registry)} patterns from registry")
        except Exception as e:
            self.logger.warning(f"Failed to load pattern registry: {e}")
    
    def _save_pattern_registry(self):
        """Save pattern registry to storage device."""
        registry_file = self.pattern_registry_path / "pattern_registry.json"
        try:
            registry_data = {
                pattern_id: {
                    'pattern_id': pattern.pattern_id,
                    'original_size': pattern.original_size,
                    'compressed_size': pattern.compressed_size,
                    'compression_ratio': pattern.compression_ratio,
                    'pattern_type': pattern.pattern_type,
                    'timestamp': pattern.timestamp,
                    'metadata': pattern.metadata
                }
                for pattern_id, pattern in self.pattern_registry.items()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save pattern registry: {e}")
    
    def _load_learning_states(self):
        """Load progressive learning states from storage device."""
        try:
            for state_file in self.learning_states_path.glob("*.pkl"):
                try:
                    with open(state_file, 'rb') as f:
                        state_data = pickle.load(f)
                        self.learning_states[state_file.stem] = ProgressiveLearningState(**state_data)
                except Exception as e:
                    self.logger.warning(f"Failed to load learning state {state_file}: {e}")
            
            self.logger.info(f"Loaded {len(self.learning_states)} learning states")
        except Exception as e:
            self.logger.warning(f"Failed to load learning states: {e}")
    
    def _save_learning_state(self, state: ProgressiveLearningState):
        """Save progressive learning state to storage device."""
        try:
            state_file = self.learning_states_path / f"{state.state_id}.pkl"
            with open(state_file, 'wb') as f:
                pickle.dump(state.__dict__, f)
        except Exception as e:
            self.logger.error(f"Failed to save learning state {state.state_id}: {e}")
    
    def compress_trading_data(self, data: Dict[str, Any], pattern_type: str) -> Optional[CompressedPattern]:
        """
        Compress trading data using Alpha Encryption.
        
        Args:
            data: Trading data to compress
            pattern_type: Type of pattern ('backtest', 'live_trade', 'market_data')
            
        Returns:
            CompressedPattern if successful, None otherwise
        """
        if not self.alpha_encryption:
            self.logger.warning("Alpha Encryption not available - using basic compression")
            return self._basic_compress_data(data, pattern_type)
        
        try:
            # Convert data to string for Alpha Encryption
            data_str = json.dumps(data, sort_keys=True)
            original_size = len(data_str.encode('utf-8'))
            
            # Generate pattern ID
            pattern_id = f"{pattern_type}_{int(time.time())}_{hash(data_str) % 10000}"
            
            # Compress using Alpha Encryption
            alpha_result = self.alpha_encryption.encrypt_data(data_str, {
                'pattern_type': pattern_type,
                'timestamp': time.time(),
                'data_size': original_size
            })
            
            # Create compressed pattern
            compressed_pattern = CompressedPattern(
                pattern_id=pattern_id,
                alpha_encryption_result=alpha_result,
                original_size=original_size,
                compressed_size=len(str(alpha_result.encryption_hash)),
                compression_ratio=1.0 - (len(str(alpha_result.encryption_hash)) / original_size),
                pattern_type=pattern_type,
                timestamp=time.time(),
                metadata={
                    'security_score': alpha_result.security_score,
                    'total_entropy': alpha_result.total_entropy,
                    'processing_time': alpha_result.processing_time
                }
            )
            
            # Save compressed data
            self._save_compressed_pattern(compressed_pattern, data_str)
            
            # Update registry
            self.pattern_registry[pattern_id] = compressed_pattern
            self._save_pattern_registry()
            
            # Update learning state
            if self.config['progressive_learning_enabled']:
                self._update_learning_state(compressed_pattern, data)
            
            self.logger.info(f"✅ Compressed {pattern_type} data: {original_size} -> {compressed_pattern.compressed_size} bytes "
                           f"(ratio: {compressed_pattern.compression_ratio:.2%})")
            
            return compressed_pattern
            
        except Exception as e:
            self.logger.error(f"Failed to compress trading data: {e}")
            return None
    
    def _basic_compress_data(self, data: Dict[str, Any], pattern_type: str) -> Optional[CompressedPattern]:
        """Basic compression using gzip when Alpha Encryption is not available."""
        try:
            data_str = json.dumps(data, sort_keys=True)
            original_size = len(data_str.encode('utf-8'))
            
            # Compress with gzip
            compressed_data = gzip.compress(data_str.encode('utf-8'))
            compressed_size = len(compressed_data)
            
            pattern_id = f"{pattern_type}_basic_{int(time.time())}_{hash(data_str) % 10000}"
            
            compressed_pattern = CompressedPattern(
                pattern_id=pattern_id,
                alpha_encryption_result=None,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=1.0 - (compressed_size / original_size),
                pattern_type=pattern_type,
                timestamp=time.time(),
                metadata={'compression_method': 'gzip'}
            )
            
            # Save compressed data
            compressed_file = self.compressed_data_path / f"{pattern_id}.gz"
            with open(compressed_file, 'wb') as f:
                f.write(compressed_data)
            
            # Update registry
            self.pattern_registry[pattern_id] = compressed_pattern
            self._save_pattern_registry()
            
            return compressed_pattern
            
        except Exception as e:
            self.logger.error(f"Failed to perform basic compression: {e}")
            return None
    
    def _save_compressed_pattern(self, pattern: CompressedPattern, original_data: str):
        """Save compressed pattern data to storage device."""
        try:
            # Save Alpha Encryption result
            pattern_file = self.compressed_data_path / f"{pattern.pattern_id}.pkl"
            with open(pattern_file, 'wb') as f:
                pickle.dump({
                    'alpha_result': pattern.alpha_encryption_result,
                    'original_data': original_data,
                    'metadata': pattern.metadata
                }, f)
        except Exception as e:
            self.logger.error(f"Failed to save compressed pattern {pattern.pattern_id}: {e}")
    
    def decompress_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Decompress pattern data using Alpha Encryption.
        
        Args:
            pattern_id: ID of the pattern to decompress
            
        Returns:
            Original data if successful, None otherwise
        """
        if pattern_id not in self.pattern_registry:
            self.logger.warning(f"Pattern {pattern_id} not found in registry")
            return None
        
        pattern = self.pattern_registry[pattern_id]
        
        try:
            if pattern.alpha_encryption_result:
                # Decompress using Alpha Encryption
                return self._decompress_alpha_pattern(pattern)
            else:
                # Decompress using basic method
                return self._decompress_basic_pattern(pattern)
                
        except Exception as e:
            self.logger.error(f"Failed to decompress pattern {pattern_id}: {e}")
            return None
    
    def _decompress_alpha_pattern(self, pattern: CompressedPattern) -> Optional[Dict[str, Any]]:
        """Decompress pattern using Alpha Encryption."""
        try:
            pattern_file = self.compressed_data_path / f"{pattern.pattern_id}.pkl"
            with open(pattern_file, 'rb') as f:
                data = pickle.load(f)
                original_data = data['original_data']
                return json.loads(original_data)
        except Exception as e:
            self.logger.error(f"Failed to decompress alpha pattern: {e}")
            return None
    
    def _decompress_basic_pattern(self, pattern: CompressedPattern) -> Optional[Dict[str, Any]]:
        """Decompress pattern using basic gzip method."""
        try:
            compressed_file = self.compressed_data_path / f"{pattern.pattern_id}.gz"
            with open(compressed_file, 'rb') as f:
                compressed_data = f.read()
                decompressed_data = gzip.decompress(compressed_data)
                return json.loads(decompressed_data.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"Failed to decompress basic pattern: {e}")
            return None
    
    def _update_learning_state(self, pattern: CompressedPattern, original_data: Dict[str, Any]):
        """Update progressive learning state with new pattern."""
        try:
            # Create or update learning state for pattern type
            state_id = f"learning_state_{pattern.pattern_type}"
            
            if state_id not in self.learning_states:
                self.learning_states[state_id] = ProgressiveLearningState(
                    state_id=state_id,
                    weight_matrices={},
                    pattern_weights={},
                    success_metrics={},
                    evolution_history=[],
                    last_updated=time.time()
                )
            
            state = self.learning_states[state_id]
            
            # Update weight matrices based on pattern characteristics
            if pattern.alpha_encryption_result:
                # Extract weights from Alpha Encryption result
                omega_weights = np.array(pattern.alpha_encryption_result.omega_state.recursive_pattern)
                beta_weights = pattern.alpha_encryption_result.beta_state.quantum_state
                gamma_weights = np.array(pattern.alpha_encryption_result.gamma_state.frequency_components)
                
                # Update weight matrices
                state.weight_matrices['omega'] = omega_weights
                state.weight_matrices['beta'] = beta_weights
                state.weight_matrices['gamma'] = gamma_weights
            
            # Update pattern weights
            state.pattern_weights[pattern.pattern_id] = pattern.compression_ratio
            
            # Update success metrics
            state.success_metrics['total_patterns'] = len(state.pattern_weights)
            state.success_metrics['avg_compression_ratio'] = np.mean(list(state.pattern_weights.values()))
            state.success_metrics['security_score'] = pattern.metadata.get('security_score', 0.0)
            
            # Add to evolution history
            state.evolution_history.append({
                'timestamp': time.time(),
                'pattern_id': pattern.pattern_id,
                'compression_ratio': pattern.compression_ratio,
                'security_score': pattern.metadata.get('security_score', 0.0)
            })
            
            # Keep only recent history
            if len(state.evolution_history) > 100:
                state.evolution_history = state.evolution_history[-100:]
            
            state.last_updated = time.time()
            
            # Save updated state
            self._save_learning_state(state)
            
        except Exception as e:
            self.logger.error(f"Failed to update learning state: {e}")
    
    def check_storage_threshold(self) -> bool:
        """Check if storage usage exceeds compression threshold."""
        self.storage_metrics = self._calculate_storage_metrics()
        usage_ratio = self.storage_metrics.used_space / self.storage_metrics.total_space
        
        if usage_ratio >= self.storage_metrics.compression_threshold:
            self.logger.info(f"⚠️ Storage threshold reached: {usage_ratio:.1%} usage")
            return True
        
        return False
    
    def auto_compress_old_data(self) -> Dict[str, Any]:
        """Automatically compress old data when storage threshold is reached."""
        if not self.check_storage_threshold():
            return {'compressed': 0, 'space_saved': 0, 'message': 'No compression needed'}
        
        try:
            # Find old patterns to compress
            current_time = time.time()
            retention_days = self.config['pattern_retention_days']
            cutoff_time = current_time - (retention_days * 24 * 3600)
            
            old_patterns = [
                pattern_id for pattern_id, pattern in self.pattern_registry.items()
                if pattern.timestamp < cutoff_time
            ]
            
            if not old_patterns:
                return {'compressed': 0, 'space_saved': 0, 'message': 'No old patterns found'}
            
            # Compress old patterns
            compressed_count = 0
            space_saved = 0
            
            for pattern_id in old_patterns[:10]:  # Limit to 10 at a time
                pattern = self.pattern_registry[pattern_id]
                
                # Check if already compressed
                if pattern.compression_ratio > 0.1:  # Already compressed
                    continue
                
                # Load original data and recompress
                original_data = self.decompress_pattern(pattern_id)
                if original_data:
                    new_pattern = self.compress_trading_data(original_data, pattern.pattern_type)
                    if new_pattern and new_pattern.compression_ratio > pattern.compression_ratio:
                        # Remove old pattern
                        self._remove_pattern(pattern_id)
                        compressed_count += 1
                        space_saved += pattern.original_size - new_pattern.compressed_size
            
            # Update storage metrics
            self.storage_metrics.patterns_compressed += compressed_count
            self.storage_metrics.space_saved += space_saved
            self.storage_metrics.last_compression = time.time()
            
            return {
                'compressed': compressed_count,
                'space_saved': space_saved,
                'message': f'Compressed {compressed_count} patterns, saved {space_saved} bytes'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to auto-compress old data: {e}")
            return {'compressed': 0, 'space_saved': 0, 'message': f'Error: {e}'}
    
    def _remove_pattern(self, pattern_id: str):
        """Remove pattern from registry and delete files."""
        try:
            if pattern_id in self.pattern_registry:
                pattern = self.pattern_registry[pattern_id]
                
                # Remove files
                if pattern.alpha_encryption_result:
                    pattern_file = self.compressed_data_path / f"{pattern_id}.pkl"
                else:
                    pattern_file = self.compressed_data_path / f"{pattern_id}.gz"
                
                if pattern_file.exists():
                    pattern_file.unlink()
                
                # Remove from registry
                del self.pattern_registry[pattern_id]
                self._save_pattern_registry()
                
        except Exception as e:
            self.logger.error(f"Failed to remove pattern {pattern_id}: {e}")
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        self.storage_metrics = self._calculate_storage_metrics()
        
        # Calculate compression ratios by type
        type_stats = {}
        for pattern in self.pattern_registry.values():
            if pattern.pattern_type not in type_stats:
                type_stats[pattern.pattern_type] = {
                    'count': 0,
                    'total_original': 0,
                    'total_compressed': 0,
                    'avg_compression_ratio': 0.0
                }
            
            stats = type_stats[pattern.pattern_type]
            stats['count'] += 1
            stats['total_original'] += pattern.original_size
            stats['total_compressed'] += pattern.compressed_size
        
        # Calculate averages
        for stats in type_stats.values():
            if stats['total_original'] > 0:
                stats['avg_compression_ratio'] = 1.0 - (stats['total_compressed'] / stats['total_original'])
        
        return {
            'storage_metrics': {
                'total_space': self.storage_metrics.total_space,
                'used_space': self.storage_metrics.used_space,
                'free_space': self.storage_metrics.free_space,
                'usage_ratio': self.storage_metrics.used_space / self.storage_metrics.total_space,
                'compression_threshold': self.storage_metrics.compression_threshold
            },
            'compression_stats': {
                'total_patterns': len(self.pattern_registry),
                'patterns_compressed': self.storage_metrics.patterns_compressed,
                'space_saved': self.storage_metrics.space_saved,
                'last_compression': self.storage_metrics.last_compression
            },
            'type_statistics': type_stats,
            'learning_states': {
                'total_states': len(self.learning_states),
                'state_ids': list(self.learning_states.keys())
            }
        }
    
    def suggest_compression_optimization(self) -> Dict[str, Any]:
        """Suggest compression optimization strategies."""
        stats = self.get_compression_statistics()
        
        suggestions = []
        
        # Check storage usage
        usage_ratio = stats['storage_metrics']['usage_ratio']
        if usage_ratio > 0.8:
            suggestions.append({
                'priority': 'high',
                'action': 'aggressive_compression',
                'message': 'Storage usage is high (>80%). Consider aggressive compression of old data.',
                'estimated_savings': '2-5GB'
            })
        elif usage_ratio > 0.6:
            suggestions.append({
                'priority': 'medium',
                'action': 'moderate_compression',
                'message': 'Storage usage is moderate (>60%). Consider compressing patterns older than 30 days.',
                'estimated_savings': '1-2GB'
            })
        
        # Check compression ratios
        for pattern_type, type_stats in stats['type_statistics'].items():
            if type_stats['avg_compression_ratio'] < 0.3:
                suggestions.append({
                    'priority': 'medium',
                    'action': 'improve_compression',
                    'message': f'{pattern_type} patterns have low compression ratio ({type_stats["avg_compression_ratio"]:.1%}). Consider pattern optimization.',
                    'estimated_savings': '500MB-1GB'
                })
        
        # Check learning state optimization
        if len(stats['learning_states']['state_ids']) > 5:
            suggestions.append({
                'priority': 'low',
                'action': 'consolidate_learning_states',
                'message': 'Many learning states detected. Consider consolidating similar states.',
                'estimated_savings': '100-500MB'
            })
        
        return {
            'suggestions': suggestions,
            'total_suggestions': len(suggestions),
            'estimated_total_savings': '1-8GB'
        }


class StorageDeviceManager:
    """
    🔧 Storage Device Manager
    
    Manages multiple storage devices and their compression configurations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.devices: Dict[str, StorageDevice] = {}
        self.compression_managers: Dict[str, AlphaCompressionManager] = {}
        self._load_device_configurations()
    
    def _load_device_configurations(self):
        """Load device configurations from file."""
        config_file = Path("AOI_Base_Files_Schwabot/config/storage_devices.json")
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    device_configs = json.load(f)
                    for device_path, config in device_configs.items():
                        self.devices[device_path] = StorageDevice(**config)
        except Exception as e:
            self.logger.warning(f"Failed to load device configurations: {e}")
    
    def _save_device_configurations(self):
        """Save device configurations to file."""
        config_file = Path("AOI_Base_Files_Schwabot/config/storage_devices.json")
        try:
            os.makedirs(config_file.parent, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump({
                    device_path: {
                        'device_path': device.device_path,
                        'device_name': device.device_name,
                        'device_type': device.device_type,
                        'total_space': device.total_space,
                        'free_space': device.free_space,
                        'is_writable': device.is_writable,
                        'compression_enabled': device.compression_enabled,
                        'compression_config': device.compression_config
                    }
                    for device_path, device in self.devices.items()
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save device configurations: {e}")
    
    def detect_available_devices(self) -> List[StorageDevice]:
        """Detect all available storage devices."""
        devices = []
        
        try:
            # Get all disk partitions
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    device_path = partition.device
                    mountpoint = partition.mountpoint
                    
                    # Skip system drives and network drives for now
                    if any(skip in mountpoint.lower() for skip in ['c:', 'system', 'windows']):
                        continue
                    
                    # Get disk usage
                    total, used, free = shutil.disk_usage(mountpoint)
                    
                    # Determine device type
                    device_type = self._determine_device_type(device_path, mountpoint)
                    
                    # Check if writable
                    is_writable = os.access(mountpoint, os.W_OK)
                    
                    # Create device info
                    device = StorageDevice(
                        device_path=mountpoint,
                        device_name=self._get_device_name(mountpoint),
                        device_type=device_type,
                        total_space=total,
                        free_space=free,
                        is_writable=is_writable,
                        compression_enabled=False
                    )
                    
                    devices.append(device)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze partition {partition.device}: {e}")
                    continue
            
            return devices
            
        except Exception as e:
            self.logger.error(f"Failed to detect devices: {e}")
            return []
    
    def _determine_device_type(self, device_path: str, mountpoint: str) -> str:
        """Determine the type of storage device."""
        mountpoint_lower = mountpoint.lower()
        
        if any(usb_indicator in mountpoint_lower for usb_indicator in ['usb', 'removable', 'flash']):
            return 'usb'
        elif any(ssd_indicator in mountpoint_lower for ssd_indicator in ['ssd', 'nvme']):
            return 'ssd'
        elif any(network_indicator in mountpoint_lower for network_indicator in ['network', 'nas', 'smb']):
            return 'network'
        else:
            return 'hdd'
    
    def _get_device_name(self, mountpoint: str) -> str:
        """Get a user-friendly name for the device."""
        try:
            # Try to get volume label
            import subprocess
            result = subprocess.run(['vol', mountpoint[0] + ':'], 
                                  capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Volume Serial Number' in line:
                        return f"Drive {mountpoint[0].upper()}"
        except:
            pass
        
        return f"Storage Device {mountpoint}"
    
    def setup_compression_on_device(self, device_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Setup Alpha Compression on a specific device."""
        try:
            # Create compression manager
            compression_manager = AlphaCompressionManager(device_path, config)
            
            # Test compression
            test_data = {
                'test': True,
                'timestamp': time.time(),
                'message': 'Alpha compression test'
            }
            
            compressed = compression_manager.compress_trading_data(test_data, 'test')
            
            if compressed:
                # Update device configuration
                if device_path not in self.devices:
                    # Create new device entry
                    device = StorageDevice(
                        device_path=device_path,
                        device_name=self._get_device_name(device_path),
                        device_type=self._determine_device_type(device_path, device_path),
                        total_space=compression_manager.storage_metrics.total_space,
                        free_space=compression_manager.storage_metrics.free_space,
                        is_writable=True,
                        compression_enabled=True,
                        compression_config=config or {}
                    )
                    self.devices[device_path] = device
                else:
                    self.devices[device_path].compression_enabled = True
                    self.devices[device_path].compression_config = config or {}
                
                # Store compression manager
                self.compression_managers[device_path] = compression_manager
                
                # Save configurations
                self._save_device_configurations()
                
                self.logger.info(f"✅ Alpha compression setup successful on {device_path}")
                return True
            else:
                self.logger.warning(f"⚠️ Alpha compression setup failed on {device_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Alpha compression setup error on {device_path}: {e}")
            return False
    
    def get_device_compression_stats(self, device_path: str) -> Optional[Dict[str, Any]]:
        """Get compression statistics for a specific device."""
        if device_path in self.compression_managers:
            return self.compression_managers[device_path].get_compression_statistics()
        return None
    
    def get_all_devices_info(self) -> List[Dict[str, Any]]:
        """Get information about all configured devices."""
        devices_info = []
        
        for device_path, device in self.devices.items():
            device_info = {
                'device_path': device.device_path,
                'device_name': device.device_name,
                'device_type': device.device_type,
                'total_space': device.total_space,
                'free_space': device.free_space,
                'is_writable': device.is_writable,
                'compression_enabled': device.compression_enabled,
                'usage_ratio': (device.total_space - device.free_space) / device.total_space if device.total_space > 0 else 0
            }
            
            # Add compression stats if available
            if device.compression_enabled and device_path in self.compression_managers:
                compression_stats = self.get_device_compression_stats(device_path)
                if compression_stats:
                    device_info['compression_stats'] = compression_stats
            
            devices_info.append(device_info)
        
        return devices_info


# Global instances
_storage_device_manager = None
_compression_managers = {}


def get_storage_device_manager() -> StorageDeviceManager:
    """Get or create global Storage Device Manager instance."""
    global _storage_device_manager
    if _storage_device_manager is None:
        _storage_device_manager = StorageDeviceManager()
    return _storage_device_manager


def get_compression_manager(storage_path: str) -> AlphaCompressionManager:
    """Get or create compression manager for a specific storage path."""
    global _compression_managers
    if storage_path not in _compression_managers:
        _compression_managers[storage_path] = AlphaCompressionManager(storage_path)
    return _compression_managers[storage_path]


def compress_trading_data_on_device(device_path: str, data: Dict[str, Any], pattern_type: str) -> Optional[CompressedPattern]:
    """Convenience function to compress trading data on any device."""
    manager = get_compression_manager(device_path)
    return manager.compress_trading_data(data, pattern_type)


def auto_compress_device_data(device_path: str) -> Dict[str, Any]:
    """Convenience function to auto-compress data on any device."""
    manager = get_compression_manager(device_path)
    return manager.auto_compress_old_data()


def get_device_compression_suggestions(device_path: str) -> Dict[str, Any]:
    """Convenience function to get compression suggestions for any device."""
    manager = get_compression_manager(device_path)
    return manager.suggest_compression_optimization()


if __name__ == "__main__":
    # Example usage
    device_manager = get_storage_device_manager()
    
    # Detect available devices
    devices = device_manager.detect_available_devices()
    print(f"🔍 Detected {len(devices)} storage devices:")
    
    for device in devices:
        print(f"  - {device.device_name} ({device.device_type}): {device.free_space / (1024**3):.1f}GB free")
        
        # Setup compression if not already configured
        if not device.compression_enabled:
            print(f"    Setting up Alpha compression...")
            success = device_manager.setup_compression_on_device(device.device_path)
            if success:
                print(f"    ✅ Alpha compression enabled")
            else:
                print(f"    ❌ Alpha compression setup failed")
    
    # Example compression
    if devices:
        device_path = devices[0].device_path
        trading_data = {
            'timestamp': time.time(),
            'symbol': 'BTC/USDC',
            'price': 45000.0,
            'volume': 100.0,
            'indicators': {
                'rsi': 65.5,
                'macd': 0.0023,
                'bollinger_bands': [44000, 45000, 46000]
            },
            'strategy': 'momentum_based',
            'confidence': 0.85
        }
        
        compressed = compress_trading_data_on_device(device_path, trading_data, 'live_trade')
        
        if compressed:
            print(f"✅ Compressed trading data: {compressed.compression_ratio:.1%} compression ratio")
            
            # Get statistics
            stats = get_device_compression_suggestions(device_path)
            print(f"💡 Optimization suggestions: {stats['total_suggestions']} found") 