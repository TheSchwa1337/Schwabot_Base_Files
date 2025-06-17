"""
SECR Coordinator
================

Main coordinator for the Sustainment-Encoded Collapse Resolver system.
Orchestrates all SECR components and provides the primary API for
integration with Schwabot's trading infrastructure.
"""

import logging
import yaml
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass

from .failure_logger import FailureLogger, FailureKey, FailureGroup, FailureSubGroup
from .allocator import ResourceAllocator, AllocationDecision
from .resolver_matrix import ResolverMatrix, PatchConfig
from .injector import ConfigInjector
from .watchdog import SECRWatchdog, OutcomeMetrics
from .adaptive_icap import AdaptiveICAPTuner

logger = logging.getLogger(__name__)

@dataclass
class SECRStats:
    """Overall SECR system statistics"""
    failures_logged: int
    failures_resolved: int
    patches_applied: int
    patches_active: int
    avg_resolution_time: float
    system_stability: float
    icap_threshold: float
    uptime_seconds: float

class SECRCoordinator:
    """
    Main SECR system coordinator
    
    Provides:
    - Unified API for failure reporting and resolution
    - Component lifecycle management
    - Configuration management
    - Integration hooks for existing systems
    - Monitoring and statistics
    """
    
    def __init__(self, 
                 config_path: Optional[Path] = None,
                 initial_config: Optional[Dict[str, Any]] = None):
        
        # Load configuration
        self.config = self._load_config(config_path, initial_config)
        
        # Initialize components
        self.failure_logger = FailureLogger(
            max_keys=self.config['global']['max_failure_keys'],
            persistence_file=Path(self.config['integration']['data_paths']['phantom_corridors'])
        )
        
        self.resource_allocator = ResourceAllocator(
            config=self.config['allocator']
        )
        
        self.resolver_matrix = ResolverMatrix(
            config=self.config
        )
        
        self.config_injector = ConfigInjector(
            initial_config=self._get_initial_system_config(),
            validation_rules=self.config['injector']['validation'],
            backup_path=Path(self.config['integration']['data_paths']['config_backups'])
        )
        
        self.adaptive_icap = AdaptiveICAPTuner(
            initial_threshold=self.config['adaptive_icap']['initial_threshold'],
            adjustment_alpha=self.config['adaptive_icap']['adjustment_alpha'],
            bounds=tuple(self.config['adaptive_icap']['bounds']),
            learning_window=self.config['adaptive_icap']['learning_window']
        )
        
        self.watchdog = SECRWatchdog(
            failure_logger=self.failure_logger,
            resolver_matrix=self.resolver_matrix,
            config_injector=self.config_injector,
            evaluation_window=self.config['watchdog']['evaluation_window'],
            stability_threshold=self.config['watchdog']['stability_threshold']
        )
        
        # System state
        self.running = False
        self.start_time = time.time()
        self.last_activity = time.time()
        
        # Integration hooks
        self.failure_hooks: List[Callable[[FailureKey], None]] = []
        self.resolution_hooks: List[Callable[[FailureKey, PatchConfig], None]] = []
        self.icap_hooks: List[Callable[[float], None]] = []
        
        # Statistics
        self.stats_cache: Optional[SECRStats] = None
        self.stats_cache_time = 0.0
        
        logger.info("SECR Coordinator initialized")
    
    def _load_config(self, 
                    config_path: Optional[Path], 
                    initial_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load SECR configuration"""
        
        if initial_config:
            return initial_config
        
        # Default config path
        if config_path is None:
            config_path = Path("config/secr.yaml")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded SECR config from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            'global': {
                'enabled': True,
                'log_level': 'INFO',
                'max_failure_keys': 10000,
                'evaluation_window_ticks': 16,
                'stability_threshold': 0.8
            },
            'allocator': {'max_history': 1000},
            'injector': {
                'max_snapshots': 100,
                'validation': {}
            },
            'watchdog': {
                'evaluation_window': 16,
                'stability_threshold': 0.8,
                'baseline_window': 100
            },
            'adaptive_icap': {
                'initial_threshold': 0.4,
                'adjustment_alpha': 0.05,
                'bounds': [0.1, 0.85],
                'learning_window': 100
            },
            'integration': {
                'data_paths': {
                    'phantom_corridors': 'data/phantom_corridors.json',
                    'config_backups': 'data/config_backups',
                    'training_data': 'data/secr_training'
                }
            }
        }
    
    def _get_initial_system_config(self) -> Dict[str, Any]:
        """Get initial system configuration for the injector"""
        return {
            'strategy': {
                'trading_mode': 'adaptive',
                'risk_tolerance': 0.3,
                'batch_size_multiplier': 1.0
            },
            'engine': {
                'gpu_queue_limit': 32,
                'thread_pool_size': 8,
                'max_memory_usage': 0.8
            },
            'risk': {
                'corridor_width_multiplier': 1.0,
                'slippage_tolerance': 0.05
            },
            'timing': {
                'api_timeout_multiplier': 1.0
            }
        }
    
    async def start(self) -> None:
        """Start the SECR system"""
        if self.running:
            logger.warning("SECR already running")
            return
        
        try:
            # Start watchdog monitoring
            self.watchdog.start_monitoring()
            
            # Register integration callbacks
            self._setup_integration_callbacks()
            
            self.running = True
            self.start_time = time.time()
            
            logger.info("SECR system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start SECR system: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the SECR system"""
        if not self.running:
            return
        
        try:
            # Stop watchdog
            self.watchdog.stop_monitoring()
            
            # Save state
            await self._save_system_state()
            
            self.running = False
            
            logger.info("SECR system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping SECR system: {e}")
    
    def _setup_integration_callbacks(self) -> None:
        """Setup integration callbacks"""
        
        # ICAP adjustment callback
        def icap_callback(patch_hash: str, patch: PatchConfig) -> None:
            if patch.risk_mod and 'icap_adjustment' in patch.risk_mod:
                new_threshold = self.adaptive_icap.current_threshold + patch.risk_mod['icap_adjustment']
                for hook in self.icap_hooks:
                    try:
                        hook(new_threshold)
                    except Exception as e:
                        logger.warning(f"ICAP hook error: {e}")
        
        self.config_injector.register_callback(icap_callback)
    
    def report_failure(self, 
                      group: FailureGroup,
                      subgroup: FailureSubGroup,
                      severity: float,
                      context: Dict[str, Any],
                      timestamp: Optional[float] = None) -> FailureKey:
        """
        Report a system failure for SECR processing
        
        Args:
            group: Failure group (PERF, ORDER, ENTROPY, etc.)
            subgroup: Specific failure type
            severity: Severity score (0.0 to 1.0)
            context: Additional failure context
            timestamp: Failure timestamp (defaults to current time)
            
        Returns:
            FailureKey for tracking the failure resolution
        """
        
        if not self.running:
            logger.warning("SECR not running, failure not processed")
            return None
        
        try:
            # Log the failure
            failure_key = self.failure_logger.log_failure(
                group=group,
                subgroup=subgroup,
                severity=severity,
                context=context,
                timestamp=timestamp
            )
            
            # Update activity timestamp
            self.last_activity = time.time()
            
            # Trigger resolution pipeline
            asyncio.create_task(self._process_failure(failure_key))
            
            # Notify hooks
            for hook in self.failure_hooks:
                try:
                    hook(failure_key)
                except Exception as e:
                    logger.warning(f"Failure hook error: {e}")
            
            logger.debug(f"Reported failure: {failure_key.hash}")
            return failure_key
            
        except Exception as e:
            logger.error(f"Error reporting failure: {e}")
            return None
    
    async def _process_failure(self, failure_key: FailureKey) -> None:
        """Process a failure through the resolution pipeline"""
        
        try:
            # 1. Resource allocation decision
            allocation = self.resource_allocator.allocate_for_failure(
                failure_key, 
                {'current_load': 0.6}  # TODO: Get actual system metrics
            )
            
            # 2. Generate resolution patch
            patch = self.resolver_matrix.resolve_failure(failure_key, allocation)
            
            # 3. Apply configuration patch
            success, errors = self.config_injector.apply_patch(patch, failure_key.hash)
            
            if success:
                # 4. Register for outcome monitoring
                self.watchdog.register_patch_application(failure_key, patch)
                
                # 5. Update adaptive ICAP if relevant
                if failure_key.group == FailureGroup.ENTROPY:
                    new_threshold = self.adaptive_icap.process_failure(failure_key)
                    if new_threshold is not None:
                        # Apply ICAP adjustment
                        icap_patch = PatchConfig(
                            risk_mod={'icap_threshold': new_threshold},
                            persistence_ticks=patch.persistence_ticks,
                            priority=patch.priority + 1,
                            metadata={'source': 'adaptive_icap', 'auto_generated': True}
                        )
                        self.config_injector.apply_patch(icap_patch, f"icap_{failure_key.hash}")
                
                # Notify resolution hooks
                for hook in self.resolution_hooks:
                    try:
                        hook(failure_key, patch)
                    except Exception as e:
                        logger.warning(f"Resolution hook error: {e}")
                
                logger.info(f"Successfully processed failure {failure_key.hash}")
                
            else:
                logger.error(f"Failed to apply patch for {failure_key.hash}: {errors}")
                
        except Exception as e:
            logger.error(f"Error processing failure {failure_key.hash}: {e}")
    
    def update_performance_feedback(self, 
                                  profit_delta: float, 
                                  latency_ms: float,
                                  error_rate: float,
                                  stability_score: float = 1.0) -> None:
        """
        Update SECR with performance feedback
        
        Args:
            profit_delta: Profit change since last update
            latency_ms: Current system latency
            error_rate: Current error rate (0.0 to 1.0)
            stability_score: System stability score (0.0 to 1.0)
        """
        
        try:
            # Update adaptive ICAP
            self.adaptive_icap.update_performance(profit_delta, stability_score)
            
            # Update activity timestamp
            self.last_activity = time.time()
            
        except Exception as e:
            logger.error(f"Error updating performance feedback: {e}")
    
    def suggest_market_based_adjustments(self, market_metrics: Dict[str, float]) -> Optional[float]:
        """
        Get market-based ICAP adjustments
        
        Args:
            market_metrics: Dict with volatility, volume, price_momentum, etc.
            
        Returns:
            New ICAP threshold if adjustment suggested, None otherwise
        """
        
        try:
            return self.adaptive_icap.suggest_market_based_adjustment(market_metrics)
        except Exception as e:
            logger.error(f"Error suggesting market adjustments: {e}")
            return None
    
    def get_current_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get current live system configuration"""
        return self.config_injector.get_current_config(section)
    
    def get_system_stats(self, use_cache: bool = True) -> SECRStats:
        """Get comprehensive SECR system statistics"""
        
        # Check cache
        if use_cache and self.stats_cache and (time.time() - self.stats_cache_time) < 5.0:
            return self.stats_cache
        
        try:
            # Gather stats from components
            failure_stats = self.failure_logger.get_statistics()
            injector_stats = self.config_injector.get_injection_stats()
            resolver_stats = self.resolver_matrix.get_patch_stats()
            watchdog_stats = self.watchdog.get_monitoring_status()
            
            # Calculate metrics
            total_failures = failure_stats.get('total_failures', 0)
            resolved_failures = failure_stats.get('closed_keys', 0)
            
            resolution_rate = (resolved_failures / max(1, total_failures)) * 100
            avg_resolution_time = failure_stats.get('avg_closure_time', 0.0)
            
            system_stability = min(1.0, resolution_rate / 100.0)
            
            stats = SECRStats(
                failures_logged=total_failures,
                failures_resolved=resolved_failures,
                patches_applied=injector_stats.get('successful_applied', 0),
                patches_active=resolver_stats.get('active_patches', 0),
                avg_resolution_time=avg_resolution_time,
                system_stability=system_stability,
                icap_threshold=self.adaptive_icap.current_threshold,
                uptime_seconds=time.time() - self.start_time
            )
            
            # Cache stats
            self.stats_cache = stats
            self.stats_cache_time = time.time()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error gathering system stats: {e}")
            return SECRStats(0, 0, 0, 0, 0.0, 0.0, 0.4, 0.0)
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed system status"""
        
        stats = self.get_system_stats()
        
        return {
            'system': {
                'running': self.running,
                'uptime_seconds': stats.uptime_seconds,
                'last_activity': self.last_activity,
                'config_loaded': bool(self.config)
            },
            'performance': {
                'failures_logged': stats.failures_logged,
                'failures_resolved': stats.failures_resolved,
                'resolution_rate': (stats.failures_resolved / max(1, stats.failures_logged)) * 100,
                'patches_applied': stats.patches_applied,
                'patches_active': stats.patches_active,
                'avg_resolution_time': stats.avg_resolution_time,
                'system_stability': stats.system_stability
            },
            'components': {
                'failure_logger': self.failure_logger.get_statistics(),
                'resource_allocator': self.resource_allocator.get_allocation_stats(),
                'resolver_matrix': self.resolver_matrix.get_patch_stats(),
                'config_injector': self.config_injector.get_injection_stats(),
                'adaptive_icap': self.adaptive_icap.get_current_status(),
                'watchdog': self.watchdog.get_monitoring_status()
            },
            'integration': {
                'failure_hooks': len(self.failure_hooks),
                'resolution_hooks': len(self.resolution_hooks),
                'icap_hooks': len(self.icap_hooks)
            }
        }
    
    def register_failure_hook(self, hook: Callable[[FailureKey], None]) -> None:
        """Register a hook to be called when failures are reported"""
        self.failure_hooks.append(hook)
        
    def register_resolution_hook(self, hook: Callable[[FailureKey, PatchConfig], None]) -> None:
        """Register a hook to be called when failures are resolved"""
        self.resolution_hooks.append(hook)
        
    def register_icap_hook(self, hook: Callable[[float], None]) -> None:
        """Register a hook to be called when ICAP threshold changes"""
        self.icap_hooks.append(hook)
    
    def force_icap_adjustment(self, new_threshold: float, reason: str = "Manual override") -> bool:
        """Force ICAP threshold adjustment"""
        return self.adaptive_icap.manual_override(new_threshold, reason)
    
    def emergency_reset(self) -> bool:
        """Emergency reset to initial configuration"""
        try:
            success = self.config_injector.emergency_reset()
            if success:
                logger.warning("SECR emergency reset completed")
            return success
        except Exception as e:
            logger.error(f"Emergency reset failed: {e}")
            return False
    
    def get_schwafit_training_data(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Get training data for SchwaFit system"""
        return self.watchdog.get_schwafit_training_data(batch_size)
    
    async def _save_system_state(self) -> None:
        """Save system state for persistence"""
        try:
            # Save failure logger state
            self.failure_logger.save_state()
            
            logger.debug("SECR system state saved")
            
        except Exception as e:
            logger.warning(f"Failed to save SECR state: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.running:
            asyncio.create_task(self.stop()) 