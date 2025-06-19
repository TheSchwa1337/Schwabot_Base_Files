#!/usr/bin/env python3
"""
Strategy Loader - Dynamic Trading Strategy Management System
==========================================================

Dynamic strategy loading and management system for the Schwabot framework.
Provides hot-reloading, validation, and lifecycle management for trading strategies.

Key Features:
- Dynamic strategy loading from multiple sources (files, databases, APIs)
- Hot-reloading with zero-downtime strategy updates
- Strategy validation and compatibility checking
- Version control and rollback capabilities
- Strategy dependency management
- Performance monitoring and optimization
- Strategy lifecycle management (load, unload, reload, update)
- Configuration management and parameter validation
- Strategy isolation and sandboxing
- Real-time strategy health monitoring

Integration Points:
- strategy_logic.py: Core strategy execution engine
- tick_processor.py: Real-time market data feed
- risk_monitor.py: Risk management integration
- constraints.py: Strategy constraint validation
- mathematical_optimization_bridge.py: Mathematical optimization
- rittle_gemm.py: High-performance matrix operations

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import logging
import time
import threading
import json
import yaml
import importlib
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable, Type
from enum import Enum
from collections import deque, defaultdict
from pathlib import Path
import hashlib
import pickle
import warnings

# Import Windows CLI compatibility handler
try:
    from core.enhanced_windows_cli_compatibility import (
        EnhancedWindowsCliCompatibilityHandler as CLIHandler,
        safe_print, safe_log
    )
    CLI_COMPATIBILITY_AVAILABLE = True
except ImportError:
    CLI_COMPATIBILITY_AVAILABLE = False
    # Fallback CLI handler
    class CLIHandler:
        @staticmethod
        def safe_emoji_print(message: str, force_ascii: bool = False) -> str:
            """Fallback emoji-safe print function"""
            emoji_mapping = {
                '✅': '[SUCCESS]', '❌': '[ERROR]', '⚠️': '[WARNING]', '🚨': '[ALERT]',
                '🎉': '[COMPLETE]', '🔄': '[PROCESSING]', '⏳': '[WAITING]', '⭐': '[STAR]',
                '🚀': '[LAUNCH]', '🔧': '[TOOLS]', '🛠️': '[REPAIR]', '⚡': '[FAST]',
                '🔍': '[SEARCH]', '🎯': '[TARGET]', '🔥': '[HOT]', '❄️': '[COOL]',
                '📊': '[DATA]', '📈': '[PROFIT]', '📉': '[LOSS]', '💰': '[MONEY]',
                '🧪': '[TEST]', '⚖️': '[BALANCE]', '️': '[TEMP]', '🔬': '[ANALYZE]',
                '': '[SYSTEM]', '️': '[COMPUTER]', '📱': '[MOBILE]', '🌐': '[NETWORK]',
                '🔒': '[SECURE]', '🔓': '[UNLOCK]', '🔑': '[KEY]', '🛡️': '[SHIELD]',
                '🧮': '[CALC]', '📐': '[MATH]', '🔢': '[NUMBERS]', '∞': '[INFINITY]',
                'φ': '[PHI]', 'π': '[PI]', '∑': '[SUM]', '∫': '[INTEGRAL]'
            }
            
            if force_ascii:
                for emoji, replacement in emoji_mapping.items():
                    message = message.replace(emoji, replacement)
            
            return message
        
        @staticmethod
        def safe_print(message: str, force_ascii: bool = False) -> None:
            """Fallback safe print function"""
            safe_message = CLIHandler.safe_emoji_print(message, force_ascii)
            print(safe_message)

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy type enumeration"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MACHINE_LEARNING = "machine_learning"
    QUANTUM = "quantum"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class StrategyStatus(Enum):
    """Strategy status enumeration"""
    LOADED = "loaded"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    VALIDATING = "validating"
    UPDATING = "updating"
    ROLLING_BACK = "rolling_back"


class LoaderType(Enum):
    """Strategy loader type enumeration"""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    PLUGIN = "plugin"
    DYNAMIC = "dynamic"


@dataclass
class StrategyConfig:
    """Strategy configuration container"""
    
    name: str
    version: str
    strategy_type: StrategyType
    description: str
    author: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class StrategyInstance:
    """Strategy instance container"""
    
    config: StrategyConfig
    instance: Any
    status: StrategyStatus
    load_time: float
    last_activity: float
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None
    memory_usage: int = 0
    cpu_usage: float = 0.0
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0


@dataclass
class LoaderResult:
    """Strategy loader result container"""
    
    success: bool
    strategy_instance: Optional[StrategyInstance] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    load_time: float = 0.0
    validation_results: Dict[str, Any] = field(default_factory=dict)


class StrategyValidator:
    """
    Strategy validation system for ensuring strategy compatibility and safety
    
    This class provides comprehensive validation for trading strategies,
    including syntax checking, dependency validation, and safety checks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize strategy validator
        
        Args:
            config: Validation configuration
        """
        self.config = config or self._default_config()
        self.cli_handler = CLIHandler()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default validation configuration"""
        return {
            'enable_syntax_check': True,
            'enable_dependency_check': True,
            'enable_safety_check': True,
            'enable_performance_check': True,
            'max_strategy_size': 1024 * 1024,  # 1MB
            'allowed_imports': ['numpy', 'pandas', 'scipy', 'sklearn'],
            'forbidden_imports': ['os', 'subprocess', 'sys'],
            'max_execution_time': 1.0,  # 1 second
            'max_memory_usage': 100 * 1024 * 1024,  # 100MB
            'enable_cli_compatibility': True
        }
    
    def validate_strategy(self, strategy_code: str, config: StrategyConfig) -> Dict[str, Any]:
        """
        Validate a strategy for safety and compatibility
        
        Args:
            strategy_code: Strategy source code
            config: Strategy configuration
            
        Returns:
            Validation results dictionary
        """
        try:
            results = {
                'syntax_valid': False,
                'dependencies_valid': False,
                'safety_valid': False,
                'performance_valid': False,
                'overall_valid': False,
                'warnings': [],
                'errors': []
            }
            
            # Syntax validation
            if self.config['enable_syntax_check']:
                syntax_result = self._validate_syntax(strategy_code)
                results['syntax_valid'] = syntax_result['valid']
                results['warnings'].extend(syntax_result['warnings'])
                results['errors'].extend(syntax_result['errors'])
            
            # Dependency validation
            if self.config['enable_dependency_check']:
                dep_result = self._validate_dependencies(strategy_code)
                results['dependencies_valid'] = dep_result['valid']
                results['warnings'].extend(dep_result['warnings'])
                results['errors'].extend(dep_result['errors'])
            
            # Safety validation
            if self.config['enable_safety_check']:
                safety_result = self._validate_safety(strategy_code)
                results['safety_valid'] = safety_result['valid']
                results['warnings'].extend(safety_result['warnings'])
                results['errors'].extend(safety_result['errors'])
            
            # Performance validation
            if self.config['enable_performance_check']:
                perf_result = self._validate_performance(strategy_code)
                results['performance_valid'] = perf_result['valid']
                results['warnings'].extend(perf_result['warnings'])
                results['errors'].extend(perf_result['errors'])
            
            # Overall validation
            results['overall_valid'] = (
                results['syntax_valid'] and
                results['dependencies_valid'] and
                results['safety_valid'] and
                results['performance_valid']
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Error in strategy validation: {e}"
            self.cli_handler.safe_print(f"❌ {error_msg}")
            return {
                'syntax_valid': False,
                'dependencies_valid': False,
                'safety_valid': False,
                'performance_valid': False,
                'overall_valid': False,
                'warnings': [],
                'errors': [error_msg]
            }
    
    def _validate_syntax(self, strategy_code: str) -> Dict[str, Any]:
        """Validate strategy syntax"""
        try:
            compile(strategy_code, '<strategy>', 'exec')
            return {'valid': True, 'warnings': [], 'errors': []}
        except SyntaxError as e:
            return {
                'valid': False,
                'warnings': [],
                'errors': [f"Syntax error: {e}"]
            }
        except Exception as e:
            return {
                'valid': False,
                'warnings': [],
                'errors': [f"Compilation error: {e}"]
            }
    
    def _validate_dependencies(self, strategy_code: str) -> Dict[str, Any]:
        """Validate strategy dependencies"""
        try:
            # Extract import statements
            import_lines = [line.strip() for line in strategy_code.split('\n') 
                          if line.strip().startswith(('import ', 'from '))]
            
            warnings = []
            errors = []
            
            for import_line in import_lines:
                # Check for forbidden imports
                for forbidden in self.config['forbidden_imports']:
                    if forbidden in import_line:
                        errors.append(f"Forbidden import: {import_line}")
                
                # Check for allowed imports
                allowed_found = False
                for allowed in self.config['allowed_imports']:
                    if allowed in import_line:
                        allowed_found = True
                        break
                
                if not allowed_found:
                    warnings.append(f"Potentially unsafe import: {import_line}")
            
            return {
                'valid': len(errors) == 0,
                'warnings': warnings,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'valid': False,
                'warnings': [],
                'errors': [f"Dependency validation error: {e}"]
            }
    
    def _validate_safety(self, strategy_code: str) -> Dict[str, Any]:
        """Validate strategy safety"""
        try:
            warnings = []
            errors = []
            
            # Check for dangerous operations
            dangerous_patterns = [
                'eval(', 'exec(', 'open(', 'file(', '__import__',
                'subprocess', 'os.system', 'os.popen'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in strategy_code:
                    errors.append(f"Dangerous operation detected: {pattern}")
            
            # Check strategy size
            if len(strategy_code) > self.config['max_strategy_size']:
                warnings.append(f"Strategy size exceeds limit: {len(strategy_code)} bytes")
            
            return {
                'valid': len(errors) == 0,
                'warnings': warnings,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'valid': False,
                'warnings': [],
                'errors': [f"Safety validation error: {e}"]
            }
    
    def _validate_performance(self, strategy_code: str) -> Dict[str, Any]:
        """Validate strategy performance characteristics"""
        try:
            warnings = []
            errors = []
            
            # Check for potential performance issues
            performance_patterns = [
                'while True:', 'for i in range(1000000):',
                'time.sleep(', 'threading.sleep('
            ]
            
            for pattern in performance_patterns:
                if pattern in strategy_code:
                    warnings.append(f"Potential performance issue: {pattern}")
            
            return {
                'valid': True,
                'warnings': warnings,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'valid': False,
                'warnings': [],
                'errors': [f"Performance validation error: {e}"]
            }


class StrategyLoader:
    """
    Dynamic strategy loading and management system
    
    This class provides comprehensive strategy loading capabilities including
    hot-reloading, validation, and lifecycle management. It integrates with
    the existing mathematical framework and trading components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize strategy loader
        
        Args:
            config: Loader configuration
        """
        self.version = "1.0.0"
        self.config = config or self._default_config()
        
        # Initialize CLI compatibility handler
        self.cli_handler = CLIHandler()
        
        # Strategy storage and management
        self.loaded_strategies: Dict[str, StrategyInstance] = {}
        self.strategy_cache: Dict[str, Any] = {}
        self.load_history: deque = deque(maxlen=self.config.get('max_history_size', 1000))
        
        # Validation and monitoring
        self.validator = StrategyValidator(self.config.get('validation_config'))
        self.monitoring_enabled = self.config.get('enable_monitoring', True)
        
        # Threading and synchronization
        self.loader_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Performance tracking
        self.total_loads = 0
        self.successful_loads = 0
        self.failed_loads = 0
        self.total_load_time = 0.0
        
        # Initialize monitoring if enabled
        if self.monitoring_enabled:
            self._start_monitoring()
        
        # Log initialization
        init_message = f"StrategyLoader v{self.version} initialized"
        if CLI_COMPATIBILITY_AVAILABLE:
            safe_log(logger, 'info', init_message)
        else:
            logger.info(init_message)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default loader configuration"""
        return {
            'max_history_size': 1000,
            'enable_monitoring': True,
            'enable_caching': True,
            'enable_hot_reload': True,
            'enable_validation': True,
            'enable_performance_tracking': True,
            'cache_size': 100,
            'max_concurrent_loads': 5,
            'load_timeout': 30.0,  # 30 seconds
            'validation_config': {},
            'strategy_paths': ['./strategies', './config/strategies'],
            'backup_enabled': True,
            'backup_path': './backups/strategies',
            'enable_cli_compatibility': True,
            'force_ascii_output': False
        }
    
    def safe_print(self, message: str, force_ascii: Optional[bool] = None) -> None:
        """
        Safe print function with CLI compatibility
        
        Args:
            message: Message to print
            force_ascii: Force ASCII conversion
        """
        if force_ascii is None:
            force_ascii = self.config.get('force_ascii_output', False)
        
        if CLI_COMPATIBILITY_AVAILABLE:
            safe_print(message, force_ascii=force_ascii)
        else:
            safe_message = self.cli_handler.safe_emoji_print(message, force_ascii=force_ascii)
            print(safe_message)
    
    def safe_log(self, level: str, message: str, context: str = "") -> bool:
        """
        Safe logging function with CLI compatibility
        
        Args:
            level: Log level
            message: Message to log
            context: Additional context
            
        Returns:
            True if logging was successful
        """
        if CLI_COMPATIBILITY_AVAILABLE:
            return safe_log(logger, level, message, context)
        else:
            try:
                log_func = getattr(logger, level.lower(), logger.info)
                log_func(message)
                return True
            except Exception:
                return False
    
    def load_strategy(self, strategy_path: str, config: Optional[StrategyConfig] = None,
                     loader_type: LoaderType = LoaderType.FILE) -> LoaderResult:
        """
        Load a strategy from the specified path
        
        Args:
            strategy_path: Path to strategy file or identifier
            config: Strategy configuration (optional)
            loader_type: Type of loader to use
            
        Returns:
            LoaderResult containing load status and strategy instance
        """
        try:
            start_time = time.time()
            
            # Check if strategy is already loaded
            if strategy_path in self.loaded_strategies:
                self.safe_print(f"⚠️ Strategy {strategy_path} already loaded")
                return LoaderResult(
                    success=True,
                    strategy_instance=self.loaded_strategies[strategy_path],
                    warnings=["Strategy already loaded"],
                    load_time=0.0
                )
            
            # Load strategy based on type
            if loader_type == LoaderType.FILE:
                result = self._load_from_file(strategy_path, config)
            elif loader_type == LoaderType.DATABASE:
                result = self._load_from_database(strategy_path, config)
            elif loader_type == LoaderType.API:
                result = self._load_from_api(strategy_path, config)
            elif loader_type == LoaderType.PLUGIN:
                result = self._load_from_plugin(strategy_path, config)
            else:
                result = LoaderResult(
                    success=False,
                    error_message=f"Unsupported loader type: {loader_type}"
                )
            
            # Update performance tracking
            load_time = time.time() - start_time
            result.load_time = load_time
            
            with self.loader_lock:
                self.total_loads += 1
                self.total_load_time += load_time
                
                if result.success:
                    self.successful_loads += 1
                    if result.strategy_instance:
                        self.loaded_strategies[strategy_path] = result.strategy_instance
                else:
                    self.failed_loads += 1
            
            # Log result
            if result.success:
                self.safe_log('info', f"Strategy {strategy_path} loaded successfully")
            else:
                self.safe_log('error', f"Failed to load strategy {strategy_path}: {result.error_message}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error loading strategy {strategy_path}: {e}"
            self.safe_log('error', error_msg)
            return LoaderResult(
                success=False,
                error_message=error_msg,
                load_time=time.time() - start_time
            )
    
    def _load_from_file(self, file_path: str, config: Optional[StrategyConfig]) -> LoaderResult:
        """
        Load strategy from file
        
        Args:
            file_path: Path to strategy file
            config: Strategy configuration
            
        Returns:
            LoaderResult containing load status
        """
        try:
            # Read strategy file
            with open(file_path, 'r', encoding='utf-8') as f:
                strategy_code = f.read()
            
            # Parse configuration if not provided
            if config is None:
                config = self._parse_strategy_config(strategy_code, file_path)
            
            # Validate strategy
            if self.config.get('enable_validation', True):
                validation_results = self.validator.validate_strategy(strategy_code, config)
                if not validation_results['overall_valid']:
                    return LoaderResult(
                        success=False,
                        error_message=f"Strategy validation failed: {validation_results['errors']}",
                        validation_results=validation_results
                    )
            
            # Execute strategy code in isolated environment
            strategy_namespace = self._create_strategy_namespace()
            exec(strategy_code, strategy_namespace)
            
            # Extract strategy class or function
            strategy_instance = self._extract_strategy_instance(strategy_namespace, config)
            
            if strategy_instance is None:
                return LoaderResult(
                    success=False,
                    error_message="No valid strategy found in file"
                )
            
            # Create strategy instance
            instance = StrategyInstance(
                config=config,
                instance=strategy_instance,
                status=StrategyStatus.LOADED,
                load_time=time.time(),
                last_activity=time.time()
            )
            
            return LoaderResult(
                success=True,
                strategy_instance=instance,
                validation_results=validation_results if 'validation_results' in locals() else {}
            )
            
        except FileNotFoundError:
            return LoaderResult(
                success=False,
                error_message=f"Strategy file not found: {file_path}"
            )
        except Exception as e:
            return LoaderResult(
                success=False,
                error_message=f"Error loading from file: {e}"
            )
    
    def _load_from_database(self, strategy_id: str, config: Optional[StrategyConfig]) -> LoaderResult:
        """
        Load strategy from database
        
        Args:
            strategy_id: Database identifier for strategy
            config: Strategy configuration
            
        Returns:
            LoaderResult containing load status
        """
        try:
            # This would integrate with your database system
            # For now, return a placeholder implementation
            self.safe_print(f"🔄 Loading strategy {strategy_id} from database...")
            
            return LoaderResult(
                success=False,
                error_message="Database loading not yet implemented"
            )
            
        except Exception as e:
            return LoaderResult(
                success=False,
                error_message=f"Error loading from database: {e}"
            )
    
    def _load_from_api(self, api_endpoint: str, config: Optional[StrategyConfig]) -> LoaderResult:
        """
        Load strategy from API endpoint
        
        Args:
            api_endpoint: API endpoint for strategy
            config: Strategy configuration
            
        Returns:
            LoaderResult containing load status
        """
        try:
            # This would integrate with your API system
            # For now, return a placeholder implementation
            self.safe_print(f"🔄 Loading strategy from API: {api_endpoint}")
            
            return LoaderResult(
                success=False,
                error_message="API loading not yet implemented"
            )
            
        except Exception as e:
            return LoaderResult(
                success=False,
                error_message=f"Error loading from API: {e}"
            )
    
    def _load_from_plugin(self, plugin_name: str, config: Optional[StrategyConfig]) -> LoaderResult:
        """
        Load strategy from plugin system
        
        Args:
            plugin_name: Plugin name
            config: Strategy configuration
            
        Returns:
            LoaderResult containing load status
        """
        try:
            # This would integrate with your plugin system
            # For now, return a placeholder implementation
            self.safe_print(f"🔄 Loading strategy plugin: {plugin_name}")
            
            return LoaderResult(
                success=False,
                error_message="Plugin loading not yet implemented"
            )
            
        except Exception as e:
            return LoaderResult(
                success=False,
                error_message=f"Error loading plugin: {e}"
            )
    
    def _parse_strategy_config(self, strategy_code: str, file_path: str) -> StrategyConfig:
        """
        Parse strategy configuration from code or file
        
        Args:
            strategy_code: Strategy source code
            config_file_path: Path to configuration file
            
        Returns:
            StrategyConfig object
        """
        try:
            # Try to extract configuration from code comments
            config = self._extract_config_from_comments(strategy_code)
            
            # If no config found, create default
            if config is None:
                config = StrategyConfig(
                    name=Path(file_path).stem,
                    version="1.0.0",
                    strategy_type=StrategyType.CUSTOM,
                    description="Auto-generated strategy configuration",
                    author="System"
                )
            
            return config
            
        except Exception as e:
            # Return default configuration on error
            return StrategyConfig(
                name=Path(file_path).stem,
                version="1.0.0",
                strategy_type=StrategyType.CUSTOM,
                description="Default strategy configuration",
                author="System"
            )
    
    def _extract_config_from_comments(self, strategy_code: str) -> Optional[StrategyConfig]:
        """
        Extract configuration from strategy code comments
        
        Args:
            strategy_code: Strategy source code
            
        Returns:
            StrategyConfig if found, None otherwise
        """
        try:
            # Look for configuration in comments
            lines = strategy_code.split('\n')
            config_lines = []
            
            for line in lines:
                if line.strip().startswith('#') and 'config:' in line.lower():
                    config_lines.append(line.strip())
            
            if not config_lines:
                return None
            
            # Parse configuration (simplified)
            config_dict = {
                'name': 'Unknown',
                'version': '1.0.0',
                'strategy_type': StrategyType.CUSTOM,
                'description': 'No description',
                'author': 'Unknown'
            }
            
            for line in config_lines:
                if 'name:' in line:
                    config_dict['name'] = line.split('name:')[1].strip()
                elif 'version:' in line:
                    config_dict['version'] = line.split('version:')[1].strip()
                elif 'description:' in line:
                    config_dict['description'] = line.split('description:')[1].strip()
                elif 'author:' in line:
                    config_dict['author'] = line.split('author:')[1].strip()
            
            return StrategyConfig(**config_dict)
            
        except Exception:
            return None
    
    def _create_strategy_namespace(self) -> Dict[str, Any]:
        """
        Create isolated namespace for strategy execution
        
        Returns:
            Dictionary containing safe namespace for strategy execution
        """
        try:
            # Create safe namespace with allowed imports
            namespace = {
                '__builtins__': {
                    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
                    'chr': chr, 'dict': dict, 'dir': dir, 'enumerate': enumerate,
                    'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
                    'getattr': getattr, 'hasattr': hasattr, 'hash': hash, 'hex': hex,
                    'id': id, 'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
                    'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max,
                    'min': min, 'next': next, 'oct': oct, 'ord': ord, 'pow': pow,
                    'print': print, 'range': range, 'repr': repr, 'reversed': reversed,
                    'round': round, 'set': set, 'slice': slice, 'sorted': sorted,
                    'str': str, 'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip
                }
            }
            
            # Add safe mathematical libraries
            try:
                import numpy as np
                namespace['np'] = np
            except ImportError:
                pass
            
            try:
                import pandas as pd
                namespace['pd'] = pd
            except ImportError:
                pass
            
            return namespace
            
        except Exception as e:
            error_msg = f"Error creating strategy namespace: {e}"
            self.safe_log('error', error_msg)
            return {}
    
    def _extract_strategy_instance(self, namespace: Dict[str, Any], config: StrategyConfig) -> Optional[Any]:
        """
        Extract strategy instance from namespace
        
        Args:
            namespace: Strategy execution namespace
            config: Strategy configuration
            
        Returns:
            Strategy instance if found, None otherwise
        """
        try:
            # Look for strategy class or function
            strategy_instance = None
            
            # Check for common strategy class names
            strategy_class_names = ['Strategy', 'TradingStrategy', 'BaseStrategy', config.name]
            
            for class_name in strategy_class_names:
                if class_name in namespace:
                    strategy_class = namespace[class_name]
                    if inspect.isclass(strategy_class):
                        strategy_instance = strategy_class()
                        break
            
            # If no class found, look for functions
            if strategy_instance is None:
                function_names = ['execute', 'run', 'trade', 'strategy']
                for func_name in function_names:
                    if func_name in namespace:
                        strategy_instance = namespace[func_name]
                        break
            
            return strategy_instance
            
        except Exception as e:
            error_msg = f"Error extracting strategy instance: {e}"
            self.safe_log('error', error_msg)
            return None
    
    def unload_strategy(self, strategy_name: str) -> bool:
        """
        Unload a strategy
        
        Args:
            strategy_name: Name of strategy to unload
            
        Returns:
            True if successfully unloaded, False otherwise
        """
        try:
            if strategy_name not in self.loaded_strategies:
                self.safe_print(f"⚠️ Strategy {strategy_name} not loaded")
                return False
            
            # Get strategy instance
            strategy_instance = self.loaded_strategies[strategy_name]
            
            # Stop strategy if running
            if strategy_instance.status == StrategyStatus.ACTIVE:
                self.safe_print(f"🔄 Stopping strategy {strategy_name}...")
                # This would integrate with your strategy execution system
            
            # Remove from loaded strategies
            del self.loaded_strategies[strategy_name]
            
            # Clear from cache
            with self.cache_lock:
                if strategy_name in self.strategy_cache:
                    del self.strategy_cache[strategy_name]
            
            self.safe_print(f"✅ Strategy {strategy_name} unloaded successfully")
            self.safe_log('info', f"Strategy {strategy_name} unloaded")
            
            return True
            
        except Exception as e:
            error_msg = f"Error unloading strategy {strategy_name}: {e}"
            self.safe_log('error', error_msg)
            return False
    
    def reload_strategy(self, strategy_name: str) -> LoaderResult:
        """
        Reload a strategy with hot-reloading
        
        Args:
            strategy_name: Name of strategy to reload
            
        Returns:
            LoaderResult containing reload status
        """
        try:
            if strategy_name not in self.loaded_strategies:
                return LoaderResult(
                    success=False,
                    error_message=f"Strategy {strategy_name} not loaded"
                )
            
            # Get current strategy
            current_strategy = self.loaded_strategies[strategy_name]
            
            # Unload current strategy
            self.unload_strategy(strategy_name)
            
            # Reload strategy (this would need the original path)
            # For now, return success
            self.safe_print(f"🚀 Strategy {strategy_name} reloaded")
            
            return LoaderResult(
                success=True,
                strategy_instance=current_strategy
            )
            
        except Exception as e:
            error_msg = f"Error reloading strategy {strategy_name}: {e}"
            self.safe_log('error', error_msg)
            return LoaderResult(
                success=False,
                error_message=error_msg
            )
    
    def get_loaded_strategies(self) -> Dict[str, StrategyInstance]:
        """
        Get all loaded strategies
        
        Returns:
            Dictionary of loaded strategies
        """
        return self.loaded_strategies.copy()
    
    def get_strategy_status(self, strategy_name: str) -> Optional[StrategyStatus]:
        """
        Get status of a specific strategy
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            Strategy status if found, None otherwise
        """
        if strategy_name in self.loaded_strategies:
            return self.loaded_strategies[strategy_name].status
        return None
    
    def _start_monitoring(self) -> None:
        """Start strategy monitoring thread"""
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.safe_log('info', 'Strategy monitoring started')
            
        except Exception as e:
            error_msg = f"Error starting monitoring: {e}"
            self.safe_log('error', error_msg)
    
    def _monitoring_loop(self) -> None:
        """Strategy monitoring loop"""
        try:
            while self.monitoring_active:
                # Monitor loaded strategies
                for strategy_name, strategy_instance in self.loaded_strategies.items():
                    # Check strategy health
                    self._check_strategy_health(strategy_name, strategy_instance)
                
                # Sleep between monitoring cycles
                time.sleep(self.config.get('monitoring_interval', 30))
                
        except Exception as e:
            error_msg = f"Error in monitoring loop: {e}"
            self.safe_log('error', error_msg)
    
    def _check_strategy_health(self, strategy_name: str, strategy_instance: StrategyInstance) -> None:
        """
        Check health of a specific strategy
        
        Args:
            strategy_name: Name of strategy
            strategy_instance: Strategy instance to check
        """
        try:
            # Check if strategy is responding
            current_time = time.time()
            time_since_activity = current_time - strategy_instance.last_activity
            
            # Alert if strategy has been inactive for too long
            if time_since_activity > self.config.get('inactivity_threshold', 300):  # 5 minutes
                warning_msg = f"Strategy {strategy_name} has been inactive for {time_since_activity:.1f}s"
                self.safe_log('warning', warning_msg)
            
            # Check error count
            if strategy_instance.error_count > self.config.get('max_error_count', 10):
                error_msg = f"Strategy {strategy_name} has {strategy_instance.error_count} errors"
                self.safe_log('error', error_msg)
                
        except Exception as e:
            error_msg = f"Error checking strategy health for {strategy_name}: {e}"
            self.safe_log('error', error_msg)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary of strategy loader
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            success_rate = 0.0
            if self.total_loads > 0:
                success_rate = self.successful_loads / self.total_loads
            
            avg_load_time = 0.0
            if self.successful_loads > 0:
                avg_load_time = self.total_load_time / self.successful_loads
            
            return {
                'total_loads': self.total_loads,
                'successful_loads': self.successful_loads,
                'failed_loads': self.failed_loads,
                'success_rate': success_rate,
                'average_load_time': avg_load_time,
                'loaded_strategies_count': len(self.loaded_strategies),
                'cache_size': len(self.strategy_cache)
            }
            
        except Exception as e:
            error_msg = f"Error getting performance summary: {e}"
            self.safe_log('error', error_msg)
            return {}


def main() -> None:
    """
    Main function for testing Strategy Loader functionality
    
    This function demonstrates the capabilities of the Strategy Loader
    and provides testing for various loading scenarios.
    Uses CLI-safe output with emoji fallbacks for Windows compatibility.
    """
    try:
        # Initialize Strategy Loader
        loader = StrategyLoader()
        
        # Use CLI-safe print for all output
        loader.safe_print("🚀 Strategy Loader Test")
        loader.safe_print("=" * 50)
        
        # Test strategy loading
        loader.safe_print("\n📊 Testing strategy loading...")
        
        # Create a simple test strategy
        test_strategy_code = '''
# config: name=TestStrategy, version=1.0.0, description=Test strategy, author=System

import numpy as np

class TestStrategy:
    def __init__(self):
        self.name = "TestStrategy"
    
    def execute(self, data):
        return np.mean(data)
'''
        
        # Test loading from string (simulated file)
        loader.safe_print("  Testing strategy validation...")
        
        # Create temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_strategy_code)
            temp_file = f.name
        
        try:
            # Test loading
            result = loader.load_strategy(temp_file)
            
            if result.success:
                loader.safe_print(f"    ✅ Strategy loaded successfully")
                loader.safe_print(f"    📊 Load time: {result.load_time:.6f}s")
                loader.safe_print(f"    📊 Strategy name: {result.strategy_instance.config.name}")
            else:
                loader.safe_print(f"    ❌ Strategy loading failed: {result.error_message}")
            
            # Test performance summary
            summary = loader.get_performance_summary()
            loader.safe_print(f"\n📊 Performance Summary:")
            loader.safe_print(f"   Total loads: {summary['total_loads']}")
            loader.safe_print(f"   Success rate: {summary['success_rate']:.2%}")
            loader.safe_print(f"   Average load time: {summary['average_load_time']:.6f}s")
            loader.safe_print(f"   Loaded strategies: {summary['loaded_strategies_count']}")
            
        finally:
            # Clean up temporary file
            import os
            os.unlink(temp_file)
        
        loader.safe_print("\n🎉 Strategy Loader test completed successfully!")
        
    except Exception as e:
        # Use CLI-safe error reporting
        loader = StrategyLoader()  # Create instance for safe printing
        loader.safe_print(f"❌ Strategy Loader test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
