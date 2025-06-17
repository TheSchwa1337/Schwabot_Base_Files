"""
SECR Configuration Injector
===========================

Applies configuration patches to the live system without interrupting
trading operations. Handles hot-swapping of parameters and safe rollback.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from copy import deepcopy
from pathlib import Path
import json

from .resolver_matrix import PatchConfig

logger = logging.getLogger(__name__)

@dataclass
class ConfigSnapshot:
    """Snapshot of configuration state"""
    timestamp: float
    config_state: Dict[str, Any]
    patch_applied: Optional[str] = None
    reason: str = "baseline"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ConfigValidator:
    """Validates configuration changes before application"""
    
    def __init__(self, validation_rules: Optional[Dict[str, Any]] = None):
        self.validation_rules = validation_rules or self._get_default_rules()
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Default validation rules"""
        return {
            'strategy': {
                'required_fields': ['trading_mode'],
                'numeric_ranges': {
                    'batch_size_multiplier': (0.1, 2.0),
                    'risk_tolerance': (0.0, 1.0),
                    'icap_adjustment': (-0.5, 0.5)
                },
                'boolean_fields': [
                    'emergency_mode', 'learning_mode', 'thermal_throttling',
                    'memory_optimization', 'hybrid_processing'
                ]
            },
            'engine': {
                'numeric_ranges': {
                    'gpu_queue_limit': (1, 1000),
                    'thread_pool_size': (1, 64),
                    'max_memory_usage': (0.1, 0.95),
                    'cpu_throttle_factor': (0.1, 1.0),
                    'gpu_throttle_factor': (0.1, 1.0)
                },
                'positive_integers': [
                    'buffer_size_limit', 'thermal_monitoring_interval'
                ]
            },
            'risk': {
                'numeric_ranges': {
                    'position_size_reduction': (0.0, 0.8),
                    'corridor_width_multiplier': (0.5, 3.0),
                    'slippage_tolerance': (0.001, 0.5),
                    'spread_buffer_multiplier': (0.5, 5.0)
                },
                'positive_integers': [
                    'max_concurrent_orders', 'max_order_attempts'
                ]
            },
            'timing': {
                'positive_integers': [
                    'tick_buffer_ms', 'order_confirmation_timeout',
                    'batch_dispatch_delay', 'api_timeout_base',
                    'retry_delay_ms', 'connection_timeout'
                ],
                'numeric_ranges': {
                    'api_timeout_multiplier': (0.5, 5.0)
                }
            }
        }
    
    def validate_patch(self, patch: PatchConfig) -> tuple[bool, List[str]]:
        """
        Validate a patch configuration
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate each section
        if patch.strategy_mod:
            errors.extend(self._validate_section('strategy', patch.strategy_mod))
        if patch.engine_mod:
            errors.extend(self._validate_section('engine', patch.engine_mod))
        if patch.risk_mod:
            errors.extend(self._validate_section('risk', patch.risk_mod))
        if patch.timing_mod:
            errors.extend(self._validate_section('timing', patch.timing_mod))
        
        return len(errors) == 0, errors
    
    def _validate_section(self, section_name: str, config: Dict[str, Any]) -> List[str]:
        """Validate a configuration section"""
        errors = []
        
        if section_name not in self.validation_rules:
            return errors
        
        rules = self.validation_rules[section_name]
        
        # Check numeric ranges
        if 'numeric_ranges' in rules:
            for field, (min_val, max_val) in rules['numeric_ranges'].items():
                if field in config:
                    value = config[field]
                    if not isinstance(value, (int, float)):
                        errors.append(f"{section_name}.{field} must be numeric, got {type(value)}")
                    elif not (min_val <= value <= max_val):
                        errors.append(f"{section_name}.{field} must be between {min_val} and {max_val}, got {value}")
        
        # Check positive integers
        if 'positive_integers' in rules:
            for field in rules['positive_integers']:
                if field in config:
                    value = config[field]
                    if not isinstance(value, int) or value <= 0:
                        errors.append(f"{section_name}.{field} must be a positive integer, got {value}")
        
        # Check boolean fields
        if 'boolean_fields' in rules:
            for field in rules['boolean_fields']:
                if field in config:
                    value = config[field]
                    if not isinstance(value, bool):
                        errors.append(f"{section_name}.{field} must be boolean, got {type(value)}")
        
        # Check required fields
        if 'required_fields' in rules:
            for field in rules['required_fields']:
                if field not in config:
                    errors.append(f"{section_name}.{field} is required but missing")
        
        return errors

class LiveConfigManager:
    """Manages live configuration with thread-safe updates"""
    
    def __init__(self, initial_config: Dict[str, Any]):
        self.config = deepcopy(initial_config)
        self.lock = threading.RLock()
        self.snapshots: List[ConfigSnapshot] = []
        self.max_snapshots = 100
        
        # Take initial snapshot
        self._take_snapshot("initial_state")
    
    def _take_snapshot(self, reason: str, patch_hash: Optional[str] = None) -> None:
        """Take a configuration snapshot"""
        snapshot = ConfigSnapshot(
            timestamp=time.time(),
            config_state=deepcopy(self.config),
            patch_applied=patch_hash,
            reason=reason
        )
        
        self.snapshots.append(snapshot)
        
        # Trim snapshots if too many
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get current configuration (or specific section)"""
        with self.lock:
            if section:
                return deepcopy(self.config.get(section, {}))
            return deepcopy(self.config)
    
    def update_config(self, patch: PatchConfig, patch_hash: str) -> bool:
        """Apply configuration patch"""
        with self.lock:
            try:
                # Take snapshot before change
                self._take_snapshot(f"before_patch_{patch_hash}", patch_hash)
                
                # Apply patch sections
                if patch.strategy_mod:
                    self._merge_section('strategy', patch.strategy_mod)
                if patch.engine_mod:
                    self._merge_section('engine', patch.engine_mod)
                if patch.risk_mod:
                    self._merge_section('risk', patch.risk_mod)
                if patch.timing_mod:
                    self._merge_section('timing', patch.timing_mod)
                
                # Take snapshot after change
                self._take_snapshot(f"after_patch_{patch_hash}", patch_hash)
                
                logger.info(f"Applied patch {patch_hash} to live config")
                return True
                
            except Exception as e:
                logger.error(f"Error applying patch {patch_hash}: {e}")
                return False
    
    def _merge_section(self, section_name: str, updates: Dict[str, Any]) -> None:
        """Merge updates into a configuration section"""
        if section_name not in self.config:
            self.config[section_name] = {}
        
        # Deep merge the updates
        self._deep_merge(self.config[section_name], updates)
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source into target dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = deepcopy(value)
    
    def rollback_to_snapshot(self, snapshot_index: int = -1) -> bool:
        """Rollback to a previous configuration snapshot"""
        with self.lock:
            try:
                if not self.snapshots:
                    logger.error("No snapshots available for rollback")
                    return False
                
                if abs(snapshot_index) > len(self.snapshots):
                    logger.error(f"Snapshot index {snapshot_index} out of range")
                    return False
                
                target_snapshot = self.snapshots[snapshot_index]
                self.config = deepcopy(target_snapshot.config_state)
                
                # Take snapshot of rollback
                self._take_snapshot(f"rollback_to_{target_snapshot.timestamp}")
                
                logger.info(f"Rolled back to snapshot from {target_snapshot.timestamp}")
                return True
                
            except Exception as e:
                logger.error(f"Error during rollback: {e}")
                return False
    
    def get_snapshot_history(self) -> List[Dict[str, Any]]:
        """Get history of configuration snapshots"""
        with self.lock:
            return [snapshot.to_dict() for snapshot in self.snapshots]

class ConfigInjector:
    """Main configuration injection system"""
    
    def __init__(self, 
                 initial_config: Dict[str, Any],
                 validation_rules: Optional[Dict[str, Any]] = None,
                 backup_path: Optional[Path] = None):
        
        self.config_manager = LiveConfigManager(initial_config)
        self.validator = ConfigValidator(validation_rules)
        self.backup_path = Path(backup_path) if backup_path else Path("data/config_backups")
        
        # Create backup directory
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Active patches tracking
        self.active_patches: Dict[str, PatchConfig] = {}
        self.patch_callbacks: List[Callable[[str, PatchConfig], None]] = []
        
        # Statistics
        self.injection_stats = {
            'total_applied': 0,
            'successful_applied': 0,
            'validation_failures': 0,
            'rollbacks': 0,
            'last_injection': None
        }
    
    def register_callback(self, callback: Callable[[str, PatchConfig], None]) -> None:
        """Register callback for patch application events"""
        self.patch_callbacks.append(callback)
    
    def _notify_callbacks(self, patch_hash: str, patch: PatchConfig) -> None:
        """Notify registered callbacks of patch application"""
        for callback in self.patch_callbacks:
            try:
                callback(patch_hash, patch)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def apply_patch(self, patch: PatchConfig, patch_hash: str) -> tuple[bool, List[str]]:
        """
        Apply a configuration patch with validation
        
        Args:
            patch: Patch configuration to apply
            patch_hash: Unique identifier for the patch
            
        Returns:
            Tuple of (success, error_messages)
        """
        self.injection_stats['total_applied'] += 1
        
        try:
            # Validate patch
            is_valid, validation_errors = self.validator.validate_patch(patch)
            if not is_valid:
                self.injection_stats['validation_failures'] += 1
                logger.warning(f"Patch {patch_hash} validation failed: {validation_errors}")
                return False, validation_errors
            
            # Create backup before applying
            self._create_backup(f"before_patch_{patch_hash}")
            
            # Apply patch
            success = self.config_manager.update_config(patch, patch_hash)
            
            if success:
                self.active_patches[patch_hash] = patch
                self.injection_stats['successful_applied'] += 1
                self.injection_stats['last_injection'] = time.time()
                
                # Notify callbacks
                self._notify_callbacks(patch_hash, patch)
                
                logger.info(f"Successfully applied patch {patch_hash}")
                return True, []
            else:
                return False, ["Failed to apply patch to configuration"]
        
        except Exception as e:
            logger.error(f"Error applying patch {patch_hash}: {e}")
            return False, [str(e)]
    
    def remove_patch(self, patch_hash: str) -> bool:
        """Remove an active patch (conceptual - requires full config reload)"""
        if patch_hash in self.active_patches:
            del self.active_patches[patch_hash]
            logger.info(f"Removed patch {patch_hash} from tracking")
            return True
        return False
    
    def get_current_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get current live configuration"""
        return self.config_manager.get_config(section)
    
    def rollback_to_snapshot(self, steps_back: int = 1) -> bool:
        """Rollback configuration to previous state"""
        success = self.config_manager.rollback_to_snapshot(-steps_back)
        if success:
            self.injection_stats['rollbacks'] += 1
            # Clear active patches since they're no longer valid
            self.active_patches.clear()
        return success
    
    def _create_backup(self, backup_name: str) -> None:
        """Create a backup of current configuration"""
        try:
            backup_file = self.backup_path / f"{backup_name}_{int(time.time())}.json"
            current_config = self.config_manager.get_config()
            
            with open(backup_file, 'w') as f:
                json.dump(current_config, f, indent=2, default=str)
            
            logger.debug(f"Created config backup: {backup_file}")
            
            # Cleanup old backups (keep last 50)
            backup_files = sorted(self.backup_path.glob("*.json"))
            if len(backup_files) > 50:
                for old_backup in backup_files[:-50]:
                    old_backup.unlink()
                    
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    def emergency_reset(self) -> bool:
        """Emergency reset to initial configuration"""
        try:
            # Get initial snapshot
            snapshots = self.config_manager.get_snapshot_history()
            if not snapshots:
                logger.error("No initial snapshot available for emergency reset")
                return False
            
            # Rollback to first snapshot (initial state)
            success = self.config_manager.rollback_to_snapshot(0)
            if success:
                self.active_patches.clear()
                logger.warning("Emergency reset to initial configuration completed")
            
            return success
            
        except Exception as e:
            logger.error(f"Emergency reset failed: {e}")
            return False
    
    def get_injection_stats(self) -> Dict[str, Any]:
        """Get injection statistics"""
        stats = deepcopy(self.injection_stats)
        stats.update({
            'active_patches_count': len(self.active_patches),
            'active_patch_hashes': list(self.active_patches.keys()),
            'snapshot_count': len(self.config_manager.snapshots),
            'success_rate': (
                self.injection_stats['successful_applied'] / max(1, self.injection_stats['total_applied'])
            ) * 100
        })
        return stats
    
    def get_active_patches(self) -> Dict[str, PatchConfig]:
        """Get currently active patches"""
        return deepcopy(self.active_patches)
    
    def validate_current_config(self) -> tuple[bool, List[str]]:
        """Validate current configuration state"""
        current_config = self.config_manager.get_config()
        
        # Create a dummy patch with current config to validate
        dummy_patch = PatchConfig(
            strategy_mod=current_config.get('strategy'),
            engine_mod=current_config.get('engine'),
            risk_mod=current_config.get('risk'),
            timing_mod=current_config.get('timing')
        )
        
        return self.validator.validate_patch(dummy_patch) 