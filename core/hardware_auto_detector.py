"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware Auto-Detector for Schwabot Trading System
=================================================

Automatically detects hardware capabilities and configures optimal settings
for memory management, GPU acceleration, and performance optimization.

Key Features:
- GPU model detection (RTX 3060 Ti, RTX 4090, etc.)
- Memory capacity detection and optimization
- Automatic configuration generation
- Performance-preserving memory management
- Cross-platform compatibility
"""

import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any

import numpy as np
import psutil

logger = logging.getLogger(__name__)

# =============================================================================
# HARDWARE DETECTION ENUMS AND DATA STRUCTURES
# =============================================================================

class GPUTier(Enum):
    """GPU performance tiers for optimization."""
    INTEGRATED = "integrated"      # Integrated graphics
    LOW_END = "low_end"           # GTX 1050, GTX 1650, etc.
    MID_RANGE = "mid_range"       # RTX 3060, RTX 4060, etc.
    HIGH_END = "high_end"         # RTX 3070, RTX 4070, etc.
    ULTRA = "ultra"               # RTX 3080, RTX 4080, etc.
    EXTREME = "extreme"           # RTX 3090, RTX 4090, etc.


class MemoryTier(Enum):
    """Memory capacity tiers for optimization."""
    LOW = "low"                   # < 8GB RAM
    MEDIUM = "medium"             # 8-16GB RAM
    HIGH = "high"                 # 16-32GB RAM
    ULTRA = "ultra"               # > 32GB RAM


class OptimizationMode(Enum):
    """Optimization modes based on hardware capabilities."""
    CONSERVATIVE = "conservative"  # Memory-constrained systems
    BALANCED = "balanced"         # Standard optimization
    PERFORMANCE = "performance"   # High-performance systems
    MAXIMUM = "maximum"           # Maximum performance


@dataclass
class GPUInfo:
    """GPU information structure."""
    name: str = ""
    memory_gb: float = 0.0
    compute_capability: str = ""
    driver_version: str = ""
    tier: GPUTier = GPUTier.INTEGRATED
    cuda_cores: int = 0
    memory_bandwidth_gbps: float = 0.0
    boost_clock_mhz: float = 0.0


@dataclass
class SystemInfo:
    """System information structure."""
    platform: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_frequency_mhz: float = 0.0
    ram_gb: float = 0.0
    ram_tier: MemoryTier = MemoryTier.LOW
    storage_gb: float = 0.0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED


@dataclass
class MemoryConfig:
    """Memory configuration for different hardware tiers."""
    # TIC Map configurations
    tic_map_sizes: Dict[str, int] = field(default_factory=dict)

    # Cache configurations
    cache_sizes: Dict[str, int] = field(default_factory=dict)

    # Bit-depth configurations
    bit_depth_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Memory pool configurations
    memory_pools: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Optimization settings
    optimization_settings: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# GPU DATABASE
# =============================================================================

GPU_DATABASE = {
# RTX 30 Series
"RTX 3060": {
"tier": GPUTier.MID_RANGE,
"memory_gb": 12.0,
"cuda_cores": 3584,
"memory_bandwidth_gbps": 360.0,
"boost_clock_mhz": 1777.0
},
"RTX 3060 Ti": {
"tier": GPUTier.MID_RANGE,
"memory_gb": 8.0,
"cuda_cores": 4864,
"memory_bandwidth_gbps": 448.0,
"boost_clock_mhz": 1665.0
},
"RTX 3070": {
"tier": GPUTier.HIGH_END,
"memory_gb": 8.0,
"cuda_cores": 5888,
"memory_bandwidth_gbps": 448.0,
"boost_clock_mhz": 1725.0
},
"RTX 3070 Ti": {
"tier": GPUTier.HIGH_END,
"memory_gb": 8.0,
"cuda_cores": 6144,
"memory_bandwidth_gbps": 608.0,
"boost_clock_mhz": 1770.0
},
"RTX 3080": {
"tier": GPUTier.ULTRA,
"memory_gb": 10.0,
"cuda_cores": 8704,
"memory_bandwidth_gbps": 760.0,
"boost_clock_mhz": 1710.0
},
"RTX 3080 Ti": {
"tier": GPUTier.ULTRA,
"memory_gb": 12.0,
"cuda_cores": 10240,
"memory_bandwidth_gbps": 912.0,
"boost_clock_mhz": 1665.0
},
"RTX 3090": {
"tier": GPUTier.EXTREME,
"memory_gb": 24.0,
"cuda_cores": 10496,
"memory_bandwidth_gbps": 936.0,
"boost_clock_mhz": 1695.0
},

# RTX 40 Series
"RTX 4060": {
"tier": GPUTier.MID_RANGE,
"memory_gb": 8.0,
"cuda_cores": 3072,
"memory_bandwidth_gbps": 272.0,
"boost_clock_mhz": 2460.0
},
"RTX 4060 Ti": {
"tier": GPUTier.MID_RANGE,
"memory_gb": 8.0,
"cuda_cores": 4352,
"memory_bandwidth_gbps": 288.0,
"boost_clock_mhz": 2535.0
},
"RTX 4070": {
"tier": GPUTier.HIGH_END,
"memory_gb": 12.0,
"cuda_cores": 5888,
"memory_bandwidth_gbps": 504.0,
"boost_clock_mhz": 2475.0
},
"RTX 4070 Ti": {
"tier": GPUTier.HIGH_END,
"memory_gb": 12.0,
"cuda_cores": 7680,
"memory_bandwidth_gbps": 504.0,
"boost_clock_mhz": 2610.0
},
"RTX 4080": {
"tier": GPUTier.ULTRA,
"memory_gb": 16.0,
"cuda_cores": 9728,
"memory_bandwidth_gbps": 716.8,
"boost_clock_mhz": 2505.0
},
"RTX 4090": {
"tier": GPUTier.EXTREME,
"memory_gb": 24.0,
"cuda_cores": 16384,
"memory_bandwidth_gbps": 1008.0,
"boost_clock_mhz": 2520.0
},

# GTX Series (for fallback detection)
"GTX 1050": {
"tier": GPUTier.LOW_END,
"memory_gb": 2.0,
"cuda_cores": 640,
"memory_bandwidth_gbps": 112.0,
"boost_clock_mhz": 1455.0
},
"GTX 1650": {
"tier": GPUTier.LOW_END,
"memory_gb": 4.0,
"cuda_cores": 896,
"memory_bandwidth_gbps": 128.0,
"boost_clock_mhz": 1665.0
},
"GTX 1660": {
"tier": GPUTier.LOW_END,
"memory_gb": 6.0,
"cuda_cores": 1408,
"memory_bandwidth_gbps": 192.0,
"boost_clock_mhz": 1785.0
},
"GTX 1660 Ti": {
"tier": GPUTier.LOW_END,
"memory_gb": 6.0,
"cuda_cores": 1536,
"memory_bandwidth_gbps": 288.0,
"boost_clock_mhz": 1500.0
}
}

# =============================================================================
# HARDWARE AUTO-DETECTOR
# =============================================================================

class HardwareAutoDetector:
    """Automatic hardware detection and configuration system."""


    def __init__(self) -> None:
        self.system_info = SystemInfo()
        self.memory_config = MemoryConfig()
        self.detected = False

    def detect_hardware(self) -> SystemInfo:
        """Detect all hardware components and generate system profile."""
        logger.info("🔍 Starting hardware auto-detection...")

        try:
            # Detect platform and basic system info
            self._detect_platform()

            # Detect CPU
            self._detect_cpu()

            # Detect RAM
            self._detect_ram()

            # Detect storage
            self._detect_storage()

            # Detect GPU
            self._detect_gpu()

            # Determine optimization mode
            self._determine_optimization_mode()

            self.detected = True
            logger.info("✅ Hardware detection completed")

            return self.system_info

        except Exception as e:
            logger.error(f"❌ Hardware detection failed: {e}")
            # Return default configuration
            return self._get_default_system_info()


    def _detect_platform(self) -> None:
        """Detect platform information."""
        self.system_info.platform = platform.system()
        logger.info(f"Platform: {self.system_info.platform}")


    def _detect_cpu(self) -> None:
        """Detect CPU information."""
        try:
            if self.system_info.platform == "Windows":
                # Windows CPU detection
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name,numberofcores,maxclockspeed"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        cpu_info = lines[1].strip()
                        self.system_info.cpu_model = cpu_info.split()[0]
                        # Extract cores and frequency from the output
                        # This is a simplified version
                        self.system_info.cpu_cores = psutil.cpu_count(logical=False)
                        self.system_info.cpu_frequency_mhz = psutil.cpu_freq().current
                else:
                    # Fallback for Windows
                    self.system_info.cpu_model = platform.processor()
                    self.system_info.cpu_cores = psutil.cpu_count(logical=False)
                    self.system_info.cpu_frequency_mhz = psutil.cpu_freq().current
            else:
                # Linux/macOS CPU detection
                self.system_info.cpu_model = platform.processor()
                self.system_info.cpu_cores = psutil.cpu_count(logical=False)
                self.system_info.cpu_frequency_mhz = psutil.cpu_freq().current

        except Exception as e:
            logger.warning(f"CPU detection failed: {e}")
            self.system_info.cpu_cores = psutil.cpu_count(logical=False)
            self.system_info.cpu_frequency_mhz = 2000.0  # Default

        logger.info(f"CPU: {self.system_info.cpu_model}")
        logger.info(f"CPU Cores: {self.system_info.cpu_cores}")
        logger.info(f"CPU Frequency: {self.system_info.cpu_frequency_mhz:.0f} MHz")


    def _detect_ram(self) -> None:
        """Detect RAM information."""
        try:
            ram_bytes = psutil.virtual_memory().total
            self.system_info.ram_gb = ram_bytes / (1024**3)

            # Determine RAM tier
            if self.system_info.ram_gb < 8:
                self.system_info.ram_tier = MemoryTier.LOW
            elif self.system_info.ram_gb < 16:
                self.system_info.ram_tier = MemoryTier.MEDIUM
            elif self.system_info.ram_gb < 32:
                self.system_info.ram_tier = MemoryTier.HIGH
            else:
                self.system_info.ram_tier = MemoryTier.ULTRA

        except Exception as e:
            logger.warning(f"RAM detection failed: {e}")
            self.system_info.ram_gb = 8.0
            self.system_info.ram_tier = MemoryTier.MEDIUM

        logger.info(f"RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")


    def _detect_storage(self) -> None:
        """Detect storage information."""
        try:
            storage_bytes = psutil.disk_usage('/').total
            self.system_info.storage_gb = storage_bytes / (1024**3)
        except Exception as e:
            logger.warning(f"Storage detection failed: {e}")
            self.system_info.storage_gb = 100.0

        logger.info(f"Storage: {self.system_info.storage_gb:.1f} GB")


    def _detect_gpu(self) -> None:
        """Detect GPU information."""
        try:
            if self.system_info.platform == "Windows":
                self._detect_gpu_windows()
            else:
                self._detect_gpu_linux()

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self.system_info.gpu = GPUInfo()


    def _detect_gpu_windows(self) -> None:
        """Detect GPU on Windows."""
        try:
            # Use nvidia-smi if available
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(',')
                if len(gpu_info) >= 3:
                    gpu_name = gpu_info[0].strip()
                    memory_mb = float(gpu_info[1].strip())
                    driver_version = gpu_info[2].strip()

                    self.system_info.gpu.name = gpu_name
                    self.system_info.gpu.memory_gb = memory_mb / 1024
                    self.system_info.gpu.driver_version = driver_version

                    # Look up GPU in database
                    self._lookup_gpu_in_database(gpu_name)

                    logger.info(f"GPU: {gpu_name}")
                    logger.info(f"GPU Memory: {self.system_info.gpu.memory_gb:.1f} GB")
                    logger.info(f"GPU Tier: {self.system_info.gpu.tier.value}")

                    return

                # Fallback to WMI
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name,adapterram"],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        if "NVIDIA" in line:
                            parts = line.strip().split()
                            gpu_name = " ".join(parts[:-1])  # Everything except last part
                            memory_bytes = int(parts[-1])

                            self.system_info.gpu.name = gpu_name
                            self.system_info.gpu.memory_gb = memory_bytes / (1024**3)

                            self._lookup_gpu_in_database(gpu_name)

                            logger.info(f"GPU: {gpu_name}")
                            logger.info(f"GPU Memory: {self.system_info.gpu.memory_gb:.1f} GB")
                            logger.info(f"GPU Tier: {self.system_info.gpu.tier.value}")
                            break

        except Exception as e:
            logger.warning(f"Windows GPU detection failed: {e}")


    def _detect_gpu_linux(self) -> None:
        """Detect GPU on Linux."""
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(',')
                if len(gpu_info) >= 3:
                    gpu_name = gpu_info[0].strip()
                    memory_mb = float(gpu_info[1].strip())
                    driver_version = gpu_info[2].strip()

                    self.system_info.gpu.name = gpu_name
                    self.system_info.gpu.memory_gb = memory_mb / 1024
                    self.system_info.gpu.driver_version = driver_version

                    self._lookup_gpu_in_database(gpu_info[0].strip())

                    logger.info(f"GPU: {gpu_name}")
                    logger.info(f"GPU Memory: {self.system_info.gpu.memory_gb:.1f} GB")
                    logger.info(f"GPU Tier: {self.system_info.gpu.tier.value}")

        except Exception as e:
            logger.warning(f"Linux GPU detection failed: {e}")


    def _lookup_gpu_in_database(self, gpu_name: str) -> None:
        """Look up GPU in database and populate additional info."""
        # Try exact match first
        if gpu_name in GPU_DATABASE:
            gpu_data = GPU_DATABASE[gpu_name]
            self.system_info.gpu.tier = gpu_data["tier"]
            self.system_info.gpu.cuda_cores = gpu_data["cuda_cores"]
            self.system_info.gpu.memory_bandwidth_gbps = gpu_data["memory_bandwidth_gbps"]
            self.system_info.gpu.boost_clock_mhz = gpu_data["boost_clock_mhz"]
            return

        # Try partial match
        for db_name, gpu_data in GPU_DATABASE.items():
            if db_name.lower() in gpu_name.lower() or gpu_name.lower() in db_name.lower():
                self.system_info.gpu.tier = gpu_data["tier"]
                self.system_info.gpu.cuda_cores = gpu_data["cuda_cores"]
                self.system_info.gpu.memory_bandwidth_gbps = gpu_data["memory_bandwidth_gbps"]
                self.system_info.gpu.boost_clock_mhz = gpu_data["boost_clock_mhz"]
                logger.info(f"Matched GPU '{gpu_name}' to database entry '{db_name}'")
                return

        # Default to integrated if no match
        logger.warning(f"GPU '{gpu_name}' not found in database, using integrated tier")
        self.system_info.gpu.tier = GPUTier.INTEGRATED


    def _determine_optimization_mode(self) -> None:
        """Determine optimal optimization mode based on hardware."""
        # Score based on hardware capabilities
        score = 0

        # GPU scoring
        gpu_scores = {
            GPUTier.INTEGRATED: 1,
            GPUTier.LOW_END: 2,
            GPUTier.MID_RANGE: 4,
            GPUTier.HIGH_END: 6,
            GPUTier.ULTRA: 8,
            GPUTier.EXTREME: 10
        }
        score += gpu_scores.get(self.system_info.gpu.tier, 1)

        # RAM scoring
        ram_scores = {
            MemoryTier.LOW: 1,
            MemoryTier.MEDIUM: 2,
            MemoryTier.HIGH: 4,
            MemoryTier.ULTRA: 6
        }
        score += ram_scores.get(self.system_info.ram_tier, 1)

        # CPU scoring (simplified)
        if self.system_info.cpu_cores >= 8:
            score += 2
        elif self.system_info.cpu_cores >= 4:
            score += 1

        # Determine mode
        if score <= 3:
            self.system_info.optimization_mode = OptimizationMode.CONSERVATIVE
        elif score <= 6:
            self.system_info.optimization_mode = OptimizationMode.BALANCED
        elif score <= 10:
            self.system_info.optimization_mode = OptimizationMode.PERFORMANCE
        else:
            self.system_info.optimization_mode = OptimizationMode.MAXIMUM

        logger.info(f"Optimization Mode: {self.system_info.optimization_mode.value}")

    def generate_memory_config(self) -> MemoryConfig:
        """Generate optimal memory configuration based on detected hardware."""
        if not self.detected:
            self.detect_hardware()

        logger.info("🔧 Generating memory configuration...")

        # Base configurations for different optimization modes
        base_configs = {
            OptimizationMode.CONSERVATIVE: {
                "tic_map_sizes": {
                    "4bit": 1000,
                    "8bit": 500,
                    "16bit": 200,
                    "42bit": 50,
                    "81bit": 10
                },
                "cache_sizes": {
                    "pattern_cache": 1000,
                    "signal_cache": 500,
                    "hash_cache": 200
                },
                "bit_depth_limits": {
                    "max_42bit_operations": 10,
                    "max_81bit_operations": 2,
                    "preferred_bit_depth": "16bit"
                }
            },
            OptimizationMode.BALANCED: {
                "tic_map_sizes": {
                    "4bit": 2000,
                    "8bit": 1000,
                    "16bit": 500,
                    "42bit": 100,
                    "81bit": 20
                },
                "cache_sizes": {
                    "pattern_cache": 2000,
                    "signal_cache": 1000,
                    "hash_cache": 500
                },
                "bit_depth_limits": {
                    "max_42bit_operations": 50,
                    "max_81bit_operations": 10,
                    "preferred_bit_depth": "16bit"
                }
            },
            OptimizationMode.PERFORMANCE: {
                "tic_map_sizes": {
                    "4bit": 5000,
                    "8bit": 2500,
                    "16bit": 1000,
                    "42bit": 250,
                    "81bit": 50
                },
                "cache_sizes": {
                    "pattern_cache": 5000,
                    "signal_cache": 2500,
                    "hash_cache": 1000
                },
                "bit_depth_limits": {
                    "max_42bit_operations": 200,
                    "max_81bit_operations": 50,
                    "preferred_bit_depth": "42bit"
                }
            },
            OptimizationMode.MAXIMUM: {
                "tic_map_sizes": {
                    "4bit": 10000,
                    "8bit": 5000,
                    "16bit": 2000,
                    "42bit": 500,
                    "81bit": 100
                },
                "cache_sizes": {
                    "pattern_cache": 10000,
                    "signal_cache": 5000,
                    "hash_cache": 2000
                },
                "bit_depth_limits": {
                    "max_42bit_operations": 500,
                    "max_81bit_operations": 100,
                    "preferred_bit_depth": "81bit"
                }
            }
        }

        # Get base configuration
        base_config = base_configs[self.system_info.optimization_mode]

        # Apply GPU-specific adjustments
        gpu_multipliers = {
            GPUTier.INTEGRATED: 0.5,
            GPUTier.LOW_END: 0.75,
            GPUTier.MID_RANGE: 1.0,
            GPUTier.HIGH_END: 1.25,
            GPUTier.ULTRA: 1.5,
            GPUTier.EXTREME: 2.0
        }

        gpu_multiplier = gpu_multipliers.get(self.system_info.gpu.tier, 1.0)

        # Apply RAM-specific adjustments
        ram_multipliers = {
            MemoryTier.LOW: 0.5,
            MemoryTier.MEDIUM: 0.75,
            MemoryTier.HIGH: 1.0,
            MemoryTier.ULTRA: 1.25
        }

        ram_multiplier = ram_multipliers.get(self.system_info.ram_tier, 1.0)

        # Calculate final multiplier (use the more restrictive one)
        final_multiplier = min(gpu_multiplier, ram_multiplier)

        # Apply multipliers to TIC map sizes
        self.memory_config.tic_map_sizes = {
            bit_depth: int(size * final_multiplier)
            for bit_depth, size in base_config["tic_map_sizes"].items()
        }

        # Apply multipliers to cache sizes
        self.memory_config.cache_sizes = {
            cache_type: int(size * final_multiplier)
            for cache_type, size in base_config["cache_sizes"].items()
        }

        # Copy bit depth limits
        self.memory_config.bit_depth_limits = base_config["bit_depth_limits"].copy()

        # Generate memory pool configuration
        self.memory_config.memory_pools = {
            "high_frequency": {
                "size_mb": int(512 * final_multiplier),
                "priority": "high",
                "bit_depth": "16bit"
            },
            "pattern_recognition": {
                "size_mb": int(1024 * final_multiplier),
                "priority": "medium",
                "bit_depth": "42bit"
            },
            "deep_analysis": {
                "size_mb": int(2048 * final_multiplier),
                "priority": "low",
                "bit_depth": "81bit"
            }
        }

        # Generate optimization settings
        self.memory_config.optimization_settings = {
            "enable_predictive_caching": True,
            "enable_thermal_scaling": True,
            "enable_profit_based_allocation": True,
            "memory_cleanup_threshold": 0.8,
            "cache_compression_enabled": True,
            "gpu_memory_fraction": min(0.8, self.system_info.gpu.memory_gb / 8.0),
            "max_concurrent_operations": int(10 * final_multiplier),
            "batch_size_optimization": True,
            "adaptive_precision_scaling": True
        }

        logger.info("✅ Memory configuration generated")
        return self.memory_config


    def save_configuration(self, config_path: str = "config/hardware_auto_config.json") -> None:
        """Save the generated configuration to file."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            config_data = {
                "system_info": {
                    "platform": self.system_info.platform,
                    "cpu_model": self.system_info.cpu_model,
                    "cpu_cores": self.system_info.cpu_cores,
                    "cpu_frequency_mhz": self.system_info.cpu_frequency_mhz,
                    "ram_gb": self.system_info.ram_gb,
                    "ram_tier": self.system_info.ram_tier.value,
                    "storage_gb": self.system_info.storage_gb,
                    "gpu": {
                        "name": self.system_info.gpu.name,
                        "memory_gb": self.system_info.gpu.memory_gb,
                        "tier": self.system_info.gpu.tier.value,
                        "cuda_cores": self.system_info.gpu.cuda_cores,
                        "memory_bandwidth_gbps": self.system_info.gpu.memory_bandwidth_gbps,
                        "boost_clock_mhz": self.system_info.gpu.boost_clock_mhz
                    },
                    "optimization_mode": self.system_info.optimization_mode.value
                },
                "memory_config": {
                    "tic_map_sizes": self.memory_config.tic_map_sizes,
                    "cache_sizes": self.memory_config.cache_sizes,
                    "bit_depth_limits": self.memory_config.bit_depth_limits,
                    "memory_pools": self.memory_config.memory_pools,
                    "optimization_settings": self.memory_config.optimization_settings
                },
                "detection_timestamp": str(np.datetime64('now')),
                "version": "1.0.0"
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"✅ Configuration saved to {config_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            return False

    def load_configuration(self, config_path: str = "config/hardware_auto_config.json") -> bool:
        """Load configuration from file."""
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Configuration file not found: {config_path}")
                return False

            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Restore system info
            self.system_info.platform = config_data["system_info"]["platform"]
            self.system_info.cpu_model = config_data["system_info"]["cpu_model"]
            self.system_info.cpu_cores = config_data["system_info"]["cpu_cores"]
            self.system_info.cpu_frequency_mhz = config_data["system_info"]["cpu_frequency_mhz"]
            self.system_info.ram_gb = config_data["system_info"]["ram_gb"]
            self.system_info.ram_tier = MemoryTier(config_data["system_info"]["ram_tier"])
            self.system_info.storage_gb = config_data["system_info"]["storage_gb"]

            # Restore GPU info
            gpu_data = config_data["system_info"]["gpu"]
            self.system_info.gpu.name = gpu_data["name"]
            self.system_info.gpu.memory_gb = gpu_data["memory_gb"]
            self.system_info.gpu.tier = GPUTier(gpu_data["tier"])
            self.system_info.gpu.cuda_cores = gpu_data["cuda_cores"]
            self.system_info.gpu.memory_bandwidth_gbps = gpu_data["memory_bandwidth_gbps"]
            self.system_info.gpu.boost_clock_mhz = gpu_data["boost_clock_mhz"]

            self.system_info.optimization_mode = OptimizationMode(config_data["system_info"]["optimization_mode"])

            # Restore memory config
            self.memory_config.tic_map_sizes = config_data["memory_config"]["tic_map_sizes"]
            self.memory_config.cache_sizes = config_data["memory_config"]["cache_sizes"]
            self.memory_config.bit_depth_limits = config_data["memory_config"]["bit_depth_limits"]
            self.memory_config.memory_pools = config_data["memory_config"]["memory_pools"]
            self.memory_config.optimization_settings = config_data["memory_config"]["optimization_settings"]

            self.detected = True
            logger.info(f"✅ Configuration loaded from {config_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            return False

    def _get_default_system_info(self) -> SystemInfo:
        """Get default system info when detection fails."""
        return SystemInfo(
            platform=platform.system(),
            cpu_cores=psutil.cpu_count(logical=False),
            cpu_frequency_mhz=2000.0,
            ram_gb=8.0,
            ram_tier=MemoryTier.MEDIUM,
            storage_gb=100.0,
            gpu=GPUInfo(),
            optimization_mode=OptimizationMode.BALANCED
        )


    def print_system_summary(self) -> None:
        """Print a summary of detected hardware and configuration."""
        if not self.detected:
            logger.warning("Hardware not detected yet. Run detect_hardware() first.")
            return

        print("\n" + "=" * 60)
        print("🔍 HARDWARE AUTO-DETECTION SUMMARY")
        print("=" * 60)

        print(f"Platform: {self.system_info.platform}")
        print(f"CPU: {self.system_info.cpu_model}")
        print(f"CPU Cores: {self.system_info.cpu_cores}")
        print(f"CPU Frequency: {self.system_info.cpu_frequency_mhz:.0f} MHz")
        print(f"RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
        print(f"Storage: {self.system_info.storage_gb:.1f} GB")

        if self.system_info.gpu.name:
            print(f"GPU: {self.system_info.gpu.name}")
            print(f"GPU Memory: {self.system_info.gpu.memory_gb:.1f} GB")
            print(f"GPU Tier: {self.system_info.gpu.tier.value}")
            print(f"CUDA Cores: {self.system_info.gpu.cuda_cores:,}")
            print(f"Memory Bandwidth: {self.system_info.gpu.memory_bandwidth_gbps:.0f} GB/s")
            print(f"Boost Clock: {self.system_info.gpu.boost_clock_mhz:.0f} MHz")

        print(f"Optimization Mode: {self.system_info.optimization_mode.value}")

        print("\n📊 MEMORY CONFIGURATION")
        print("-" * 30)
        print("TIC Map Sizes:")
        for bit_depth, size in self.memory_config.tic_map_sizes.items():
            print(f"  {bit_depth}: {size:,}")

        print("\nCache Sizes:")
        for cache_type, size in self.memory_config.cache_sizes.items():
            print(f"  {cache_type}: {size:,}")

        print("\nBit Depth Limits:")
        for limit, value in self.memory_config.bit_depth_limits.items():
            print(f"  {limit}: {value}")

        print("\nMemory Pools:")
        for pool_name, pool_config in self.memory_config.memory_pools.items():
            print(f"  {pool_name}: {pool_config['size_mb']} MB ({pool_config['bit_depth']})")

        print("=" * 60)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for hardware auto-detection."""
    logging.basicConfig(level=logging.INFO)

    detector = HardwareAutoDetector()

    # Try to load existing configuration first
    if detector.load_configuration():
        detector.print_system_summary()
        return detector.memory_config

    # Perform hardware detection
    system_info = detector.detect_hardware()

    # Generate memory configuration
    memory_config = detector.generate_memory_config()

    # Save configuration
    detector.save_configuration()

    # Print summary
    detector.print_system_summary()

    return memory_config

if __name__ == "__main__":
    main()
