#!/usr/bin/env python3
"""
Mathematical Framework Configuration
===================================

Configuration system for the unified mathematics framework.
This provides configuration for:
- Recursive function parameters
- BTC256SH-A pipeline settings
- Ferris Wheel visualizer settings
- Mathematical validation thresholds
- Error handling and logging

Based on systematic elimination of Flake8 issues and SP 1.27-AE framework.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RecursionConfig:
    """Configuration for recursive function management"""
    max_depth: int = 50
    convergence_threshold: float = 1e-6
    memoization_cache_size: int = 128
    enable_depth_guards: bool = True
    enable_convergence_checking: bool = True
    enable_memoization: bool = True


@dataclass
class DriftShellConfig:
    """Configuration for drift shell operations"""
    shell_radius: float = 144.44
    ring_count: int = 12
    cycle_duration: float = 3.75  # minutes
    psi_infinity: float = 1.618033988749  # Golden ratio
    drift_coefficient: float = 0.1
    enable_ring_allocation: bool = True
    enable_depth_mapping: bool = True


@dataclass
class QuantumConfig:
    """Configuration for quantum operations"""
    energy_scale: float = 1.0
    planck_constant: float = 1.054571817e-34
    enable_phase_harmonization: bool = True
    enable_quantum_entropy: bool = True
    enable_wave_functions: bool = True


@dataclass
class ThermalConfig:
    """Configuration for thermal operations"""
    thermal_conductivity: float = 0.024  # W/(m·K) - air
    heat_capacity: float = 1005.0  # J/(kg·K) - air
    boltzmann_constant: float = 1.380649e-23
    enable_thermal_pressure: bool = True
    enable_thermal_gradients: bool = True
    enable_entropy_mapping: bool = True


@dataclass
class BTC256SHAPipelineConfig:
    """Configuration for BTC256SH-A pipeline"""
    price_history_size: int = 1000
    hash_history_size: int = 1000
    enable_price_processing: bool = True
    enable_hash_generation: bool = True
    enable_mathematical_analysis: bool = True
    enable_drift_field_computation: bool = True
    price_normalization_factor: float = 100000.0
    time_normalization_factor: float = 3600.0  # seconds to hours


@dataclass
class FerrisWheelConfig:
    """Configuration for Ferris Wheel visualizer"""
    time_points_count: int = 100
    enable_recursive_visualization: bool = True
    enable_entropy_stabilization: bool = True
    enable_drift_field_visualization: bool = True
    enable_data_export: bool = True
    export_format: str = "json"
    visualization_cache_size: int = 50


@dataclass
class ValidationConfig:
    """Configuration for mathematical validation"""
    enable_scalar_validation: bool = True
    enable_vector_validation: bool = True
    enable_matrix_validation: bool = True
    enable_tensor_validation: bool = True
    enable_quantum_state_validation: bool = True
    normalization_tolerance: float = 1e-6
    enable_operation_validation: bool = True


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling"""
    enable_exception_logging: bool = True
    enable_error_recovery: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    enable_graceful_degradation: bool = True
    log_error_details: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    log_file_path: str = "logs/mathematical_framework.log"
    max_log_file_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class MathematicalFrameworkConfig:
    """Complete configuration for the mathematical framework"""

    # Component configurations
    recursion: RecursionConfig = field(default_factory=RecursionConfig)
    drift_shell: DriftShellConfig = field(default_factory=DriftShellConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    btc_pipeline: BTC256SHAPipelineConfig = field(default_factory=BTC256SHAPipelineConfig)
    ferris_wheel: FerrisWheelConfig = field(default_factory=FerrisWheelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Framework settings
    enable_all_components: bool = True
    enable_advanced_integration: bool = True
    enable_system_monitoring: bool = True
    config_file_path: str = "config/mathematical_framework.json"

    def __post_init__(self) -> None:
        """Post-initialization setup"""
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.logging.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.logging.log_level),
            format=self.logging.log_format,
            handlers=[]
        )

        # Add console handler
        if self.logging.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.logging.log_level))
            console_handler.setFormatter(logging.Formatter(self.logging.log_format))
            logging.getLogger().addHandler(console_handler)

        # Add file handler
        if self.logging.enable_file_logging:
            file_handler = RotatingFileHandler(
                self.logging.log_file_path,
                maxBytes=self.logging.max_log_file_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setLevel(getattr(logging, self.logging.log_level))
            file_handler.setFormatter(logging.Formatter(self.logging.log_format))
            logging.getLogger().addHandler(file_handler)

    def save_config(self, file_path: Optional[str] = None) -> None:
        """
        Save configuration to file.

        Args:
            file_path: Optional file path (uses default if None)
        """
        if file_path is None:
            file_path = self.config_file_path

        # Create config directory if it doesn't exist
        config_dir = Path(file_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dictionary
        config_dict = self._to_dict()

        # Save to file
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Configuration saved to {file_path}")

    def load_config(self, file_path: Optional[str] = None) -> None:
        """
        Load configuration from file.

        Args:
            file_path: Optional file path (uses default if None)
        """
        if file_path is None:
            file_path = self.config_file_path

        if not Path(file_path).exists():
            logger.warning(f"Configuration file {file_path} not found, using defaults")
            return

        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)

            self._from_dict(config_dict)
            logger.info(f"Configuration loaded from {file_path}")

        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            logger.info("Using default configuration")

    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}

        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                # Handle dataclass fields
                config_dict[field_name] = field_value.__dict__
            else:
                # Handle simple fields
                config_dict[field_name] = field_value

        return config_dict

    def _from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary"""
        for field_name, field_value in config_dict.items():
            if hasattr(self, field_name):
                current_value = getattr(self, field_name)
                if hasattr(current_value, '__dict__'):
                    # Handle dataclass fields
                    if isinstance(field_value, dict):
                        for sub_field_name, sub_field_value in field_value.items():
                            if hasattr(current_value, sub_field_name):
                                setattr(current_value, sub_field_name, sub_field_value)
                else:
                    # Handle simple fields
                    setattr(self, field_name, field_value)

    def get_component_config(self, component_name: str) -> Optional[Any]:
        """
        Get configuration for a specific component.

        Args:
            component_name: Name of the component

        Returns:
            Component configuration or None if not found
        """
        return getattr(self, component_name, None)

    def update_component_config(self, component_name: str,
                              config_updates: Dict[str, Any]) -> bool:
        """
        Update configuration for a specific component.

        Args:
            component_name: Name of the component
            config_updates: Dictionary of configuration updates

        Returns:
            True if successful, False otherwise
        """
        component_config = getattr(self, component_name, None)
        if component_config is None:
            logger.error(f"Component {component_name} not found")
            return False

        try:
            for key, value in config_updates.items():
                if hasattr(component_config, key):
                    setattr(component_config, key, value)
                else:
                    logger.warning(f"Unknown configuration key {key} for component {component_name}")

            logger.info(f"Updated configuration for component {component_name}")
            return True

        except Exception as e:
            logger.error(f"Error updating configuration for component {component_name}: {e}")
            return False

    def validate_config(self) -> bool:
        """
        Validate the configuration.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate recursion config
            if self.recursion.max_depth <= 0:
                logger.error("Recursion max_depth must be positive")
                return False

            if self.recursion.convergence_threshold <= 0:
                logger.error("Recursion convergence_threshold must be positive")
                return False

            # Validate drift shell config
            if self.drift_shell.shell_radius <= 0:
                logger.error("Drift shell radius must be positive")
                return False

            if self.drift_shell.cycle_duration <= 0:
                logger.error("Drift shell cycle_duration must be positive")
                return False

            # Validate quantum config
            if self.quantum.energy_scale <= 0:
                logger.error("Quantum energy_scale must be positive")
                return False

            # Validate thermal config
            if self.thermal.thermal_conductivity <= 0:
                logger.error("Thermal conductivity must be positive")
                return False

            if self.thermal.heat_capacity <= 0:
                logger.error("Heat capacity must be positive")
                return False

            # Validate BTC pipeline config
            if self.btc_pipeline.price_history_size <= 0:
                logger.error("BTC pipeline price_history_size must be positive")
                return False

            # Validate Ferris Wheel config
            if self.ferris_wheel.time_points_count <= 0:
                logger.error("Ferris Wheel time_points_count must be positive")
                return False

            # Validate error handling config
            if self.error_handling.max_retry_attempts < 0:
                logger.error("Error handling max_retry_attempts must be non-negative")
                return False

            if self.error_handling.retry_delay < 0:
                logger.error("Error handling retry_delay must be non-negative")
                return False

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.

        Returns:
            Configuration summary dictionary
        """
        return {
            'recursion': {
                'max_depth': self.recursion.max_depth,
                'convergence_threshold': self.recursion.convergence_threshold,
                'enable_depth_guards': self.recursion.enable_depth_guards
            },
            'drift_shell': {
                'shell_radius': self.drift_shell.shell_radius,
                'ring_count': self.drift_shell.ring_count,
                'cycle_duration': self.drift_shell.cycle_duration
            },
            'quantum': {
                'energy_scale': self.quantum.energy_scale,
                'enable_phase_harmonization': self.quantum.enable_phase_harmonization
            },
            'thermal': {
                'thermal_conductivity': self.thermal.thermal_conductivity,
                'heat_capacity': self.thermal.heat_capacity
            },
            'btc_pipeline': {
                'price_history_size': self.btc_pipeline.price_history_size,
                'enable_price_processing': self.btc_pipeline.enable_price_processing
            },
            'ferris_wheel': {
                'time_points_count': self.ferris_wheel.time_points_count,
                'enable_recursive_visualization': self.ferris_wheel.enable_recursive_visualization
            },
            'validation': {
                'enable_scalar_validation': self.validation.enable_scalar_validation,
                'enable_vector_validation': self.validation.enable_vector_validation
            },
            'error_handling': {
                'enable_exception_logging': self.error_handling.enable_exception_logging,
                'max_retry_attempts': self.error_handling.max_retry_attempts
            },
            'logging': {
                'log_level': self.logging.log_level,
                'enable_file_logging': self.logging.enable_file_logging
            }
        }


def create_default_config() -> MathematicalFrameworkConfig:
    """
    Create default configuration.

    Returns:
        Default configuration
    """
    return MathematicalFrameworkConfig()


def load_config_from_file(file_path: str) -> MathematicalFrameworkConfig:
    """
    Load configuration from file.

    Args:
        file_path: Path to configuration file

    Returns:
        Loaded configuration
    """
    config = MathematicalFrameworkConfig()
    config.load_config(file_path)
    return config


def main() -> None:
    """Main function for testing configuration"""
    # Create default configuration
    config = create_default_config()

    # Validate configuration
    is_valid = config.validate_config()
    print(f"Configuration valid: {is_valid}")

    # Get configuration summary
    summary = config.get_summary()
    print("Configuration Summary:")
    for component, settings in summary.items():
        print(f"  {component}:")
        for key, value in settings.items():
            print(f"    {key}: {value}")

    # Save configuration
    config.save_config()

    # Load configuration
    loaded_config = load_config_from_file(config.config_file_path)
    print(f"Loaded configuration valid: {loaded_config.validate_config()}")

    # Test component configuration update
    success = config.update_component_config('recursion', {'max_depth': 100})
    print(f"Updated recursion config: {success}")
    print(f"New max_depth: {config.recursion.max_depth}")


if __name__ == "__main__":
    main()