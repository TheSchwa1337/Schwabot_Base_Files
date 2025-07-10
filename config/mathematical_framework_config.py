import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from utils.safe_print import safe_print

"""mathematical_framework_config module."""


"""mathematical_framework_config module."""



# -*- coding: utf-8 -*-


"""



Mathematical Framework Configuration.







Configuration system for the unified mathematics framework.







This provides configuration for:



- Recursive function parameters



- BTC256SHA pipeline settings



- Ferris Wheel visualizer settings



- Mathematical validation thresholds



- Error handling and logging







Based on systematic elimination of Flake8 issues and SP 1.27-AE framework.



"""


# Configure logging


logger = logging.getLogger(__name__)


@dataclass
class RecursionConfig:
    """Configuration for recursive function management."""

    max_depth: int = 50

    convergence_threshold: float = 1e-6

    memoization_cache_size: int = 128

    enable_depth_guards: bool = True

    enable_convergence_checking: bool = True

    enable_memoization: bool = True


@dataclass
class DriftShellConfig:
    """Configuration for drift shell operations."""

    shell_radius: float = 144.44

    ring_count: int = 12

    cycle_duration: float = 3.75  # minutes

    psi_infinity: float = 1.618033988749  # Golden ratio

    drift_coefficient: float = 0.1

    enable_ring_allocation: bool = True

    enable_depth_mapping: bool = True


@dataclass
class QuantumConfig:
    """Configuration for quantum operations."""

    energy_scale: float = 1.0

    planck_constant: float = 1.054571817e-34

    enable_phase_harmonization: bool = True

    enable_quantum_entropy: bool = True

    enable_wave_functions: bool = True


@dataclass
class ThermalConfig:
    """Configuration for thermal operations."""

    thermal_conductivity: float = 0.024  # W/(mK) - air

    heat_capacity: float = 1005.0  # J/(kgK) - air

    boltzmann_constant: float = 1.380649e-23

    enable_thermal_pressure: bool = True

    enable_thermal_gradients: bool = True

    enable_entropy_mapping: bool = True


@dataclass
class BTC256SHAPipelineConfig:
    """Configuration for BTC256SHA pipeline."""

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
    """Configuration for Ferris Wheel visualizer."""

    time_points_count: int = 100

    enable_recursive_visualization: bool = True

    enable_entropy_stabilization: bool = True

    enable_drift_field_visualization: bool = True

    enable_data_export: bool = True

    export_format: str = "json"

    visualization_cache_size: int = 50


@dataclass
class ValidationConfig:
    """Configuration for mathematical validation."""

    enable_scalar_validation: bool = True

    enable_vector_validation: bool = True

    enable_matrix_validation: bool = True

    enable_tensor_validation: bool = True

    enable_quantum_state_validation: bool = True

    normalization_tolerance: float = 1e-6

    enable_operation_validation: bool = True


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling."""

    enable_exception_logging: bool = True

    enable_error_recovery: bool = True

    max_retry_attempts: int = 3

    retry_delay: float = 1.0  # seconds

    enable_graceful_degradation: bool = True

    log_error_details: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    log_level: str = "INFO"

    enable_file_logging: bool = True

    enable_console_logging: bool = True

    log_file_path: str = "logs/mathematical_framework.log"

    max_log_file_size: int = 10 * 1024 * 1024  # 10 MB

    backup_count: int = 5

    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class MathematicalFrameworkConfig:
    """Complete configuration for the mathematical framework."""

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
        """Post-initialization setup."""

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""

        # Create logs directory if it doesn't exist

        log_dir = Path(self.logging.log_file_path).parent

        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging

        logging.basicConfig(
            level=getattr(logging, self.logging.log_level),
            format=self.logging.log_format,
            handlers=[],
        )

        if self.logging.enable_console_logging:

            console_handler = logging.StreamHandler()

            console_handler.setLevel(getattr(logging, self.logging.log_level))

            logging.getLogger().addHandler(console_handler)

        if self.logging.enable_file_logging:

            file_handler = logging.handlers.RotatingFileHandler(
                self.logging.log_file_path,
                maxBytes=self.logging.max_log_file_size,
                backupCount=self.logging.backup_count,
            )

            file_handler.setLevel(getattr(logging, self.logging.log_level))

            logging.getLogger().addHandler(file_handler)

    def save_config(self,   file_path: Optional[str] = None) -> None:
        """Save configuration to JSON file."""

        file_path = file_path or self.config_file_path

        config_dict = self._to_dict()

        with open(file_path, "w") as f:

            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Configuration saved to {file_path}")

    def load_config(self,   file_path: Optional[str] = None) -> None:
        """Load configuration from JSON file."""

        file_path = file_path or self.config_file_path

        if not Path(file_path).exists():

            logger.warning(f"Config file {file_path} not found, using defaults")

            return

        with open(file_path, "r") as f:

            config_dict = json.load(f)

        self._from_dict(config_dict)

        logger.info(f"Configuration loaded from {file_path}")

    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""

        return {
            "recursion": self.recursion.__dict__,
            "drift_shell": self.drift_shell.__dict__,
            "quantum": self.quantum.__dict__,
            "thermal": self.thermal.__dict__,
            "btc_pipeline": self.btc_pipeline.__dict__,
            "ferris_wheel": self.ferris_wheel.__dict__,
            "validation": self.validation.__dict__,
            "error_handling": self.error_handling.__dict__,
            "logging": self.logging.__dict__,
            "enable_all_components": self.enable_all_components,
            "enable_advanced_integration": self.enable_advanced_integration,
            "enable_system_monitoring": self.enable_system_monitoring,
        }

    def _from_dict(self,   config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""

        for key, value in config_dict.items():

            if hasattr(self, key):

                if hasattr(getattr(self, key), "__dict__"):

                    # Update dataclass fields

                    for subkey, subvalue in value.items():

                        if hasattr(getattr(self, key), subkey):

                            setattr(getattr(self, key), subkey, subvalue)

                else:

                    # Update simple attributes

                    setattr(self, key, value)


# Default configuration instance


default_config = MathematicalFrameworkConfig()


def create_default_config() -> MathematicalFrameworkConfig:
    """Create default configuration."""

    return MathematicalFrameworkConfig()


def load_config_from_file(file_path: str) -> MathematicalFrameworkConfig:
    """Load configuration from file."""

    config = MathematicalFrameworkConfig()

    config.load_config(file_path)

    return config


def main() -> None:
    """Test configuration handling."""

    # Create default configuration

    config = create_default_config()

    # Save configuration

    config.save_config()

    # Load configuration

    loaded_config = load_config_from_file(config.config_file_path)

    print(f"Configuration loaded successfully")


if __name__ == "__main__":

    main()
