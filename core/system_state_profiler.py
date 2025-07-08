import hashlib
import json
import logging
import os
import platform
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import psutil
    import cpuinfo
    from OpenGL.GL import *
    from OpenGL.GL.shaders import *
from .unified_math_system import UnifiedMathSystem
            import pygame

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System State Profiler for Schwabot Trading System
================================================

Comprehensive hardware detection and performance profiling system that:
- Auto-detects CPU model and maps to performance tiers
- Identifies GPU capabilities and shader limitations
- Determines optimal matrix sizes for GPU operations
- Creates system profiles for consistent cross-device operation
- Generates system hash for hardware fingerprinting

Key Features:
- Supports Pi 4 through RTX 5090 GPU range
- Auto-scales shader complexity based on detected hardware
- Creates portable system profiles for USB deployment
- Integrates with Schwabot's mathematical foundation
- Provides fallback logic for weak/integrated GPUs
"""

try:
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False
    logging.warning("cpuinfo not available - using platform fallback")

try:
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

logger = logging.getLogger(__name__)


class SystemTier(Enum):
    """System performance tiers for Schwabot operations."""

    EMBEDDED = "embedded"  # Pi 4, embedded systems
    APPLE_SILICON = "apple_silicon"  # M1, M2, M3 series
    TIER_LOW_SYSTEM = "low_system"  # Integrated graphics
    TIER_MID_SYSTEM = "mid_system"  # Mid-range discrete GPU
    TIER_HIGH_SYSTEM = "high_system"  # High-end discrete GPU
    TIER_ELITE = "elite"  # RTX 3080+, top-tier systems
    TIER_UNKNOWN = "unknown"  # Unidentified hardware


class CPUTier(Enum):
    """CPU performance tiers based on generation and model."""

    TIER_PI = "pi"  # ARM Cortex (Pi 4, Pi 5)
    TIER_LOW = "low"  # Older/budget CPUs
    TIER_MID = "mid"  # Mainstream CPUs
    TIER_HIGH = "high"  # High-performance CPUs
    TIER_APPLE = "apple"  # Apple Silicon
    TIER_UNKNOWN = "unknown"  # Unidentified CPU


class GPUTier(Enum):
    """GPU performance tiers for shader operations."""

    TIER_PI4 = "pi4"  # VideoCore VI (Pi 4)
    TIER_LOW = "low"  # Integrated graphics
    TIER_MID = "mid"  # Mid-range discrete
    TIER_HIGH = "high"  # High-end discrete
    TIER_ULTRA = "ultra"  # Top-tier GPUs
    TIER_UNKNOWN = "unknown"  # Unidentified GPU


@dataclass
class CPUProfile:
    """CPU hardware profile."""

    model: str
    architecture: str
    logical_cores: int
    physical_cores: int
    base_frequency: float
    max_frequency: float
    cpu_tier: CPUTier
    cache_l3_size: Optional[int] = None
    features: List[str] = field(default_factory=list)


@dataclass
class GPUProfile:
    """GPU hardware profile."""

    vendor: str
    renderer: str
    gl_version: str
    glsl_version: str
    gpu_tier: GPUTier
    max_matrix_size: int
    use_half_precision: bool
    shader_morph_enabled: bool
    memory_mb: Optional[int] = None
    compute_units: Optional[int] = None


@dataclass
class SystemProfile:
    """Complete system hardware profile."""

    device_type: str
    device_id: str
    os_info: str
    cpu: CPUProfile
    gpu: GPUProfile
    ram_total_gb: float
    ram_available_gb: float
    system_tier: SystemTier
    profile_timestamp: str
    system_hash: str


class CPUDetector:
    """Detects and classifies CPU hardware."""

    # CPU model mapping for tier classification
    CPU_TIER_MAP = {
        # Intel Desktop - High Tier
        'i5-13600': CPUTier.TIER_HIGH,
        'i7-13700': CPUTier.TIER_HIGH,
        'i9-13900': CPUTier.TIER_HIGH,
        'i5-12600': CPUTier.TIER_HIGH,
        'i7-12700': CPUTier.TIER_HIGH,
        'Ultra 7-265': CPUTier.TIER_HIGH,
        'Ultra 9-285': CPUTier.TIER_HIGH,
        # Intel Desktop - Mid Tier
        'i5-10400': CPUTier.TIER_MID,
        'i7-10700': CPUTier.TIER_MID,
        'i5-6600': CPUTier.TIER_MID,
        'i7-6700': CPUTier.TIER_MID,
        'i5-2500': CPUTier.TIER_LOW,
        'i7-2600': CPUTier.TIER_LOW,
        # AMD Desktop - High Tier
        'Ryzen 7 7800X3D': CPUTier.TIER_HIGH,
        'Ryzen 9 9800X3D': CPUTier.TIER_HIGH,
        'Ryzen 9 9950X3D': CPUTier.TIER_HIGH,
        'Ryzen 5 7600X': CPUTier.TIER_HIGH,
        'Ryzen 7 7700X': CPUTier.TIER_HIGH,
        # AMD Desktop - Mid Tier
        'Ryzen 5 5600X': CPUTier.TIER_MID,
        'Ryzen 7 5800X3D': CPUTier.TIER_MID,
        'Ryzen 5 3600': CPUTier.TIER_MID,
        'Ryzen 7 3700X': CPUTier.TIER_MID,
        'Ryzen 5 1600': CPUTier.TIER_LOW,
        # Intel Mobile
        'i7-13620H': CPUTier.TIER_HIGH,
        'i9-13900HK': CPUTier.TIER_HIGH,
        'i7-12700H': CPUTier.TIER_MID,
        'i9-12900H': CPUTier.TIER_MID,
        'i7-10750H': CPUTier.TIER_MID,
        'i7-7700HQ': CPUTier.TIER_LOW,
        # AMD Mobile
        'Ryzen 9 8900HX': CPUTier.TIER_HIGH,
        'Ryzen 9 7940HS': CPUTier.TIER_HIGH,
        'Ryzen 5 5600H': CPUTier.TIER_MID,
        'Ryzen 7 6800H': CPUTier.TIER_MID,
        # ARM/Pi
        'Cortex-A76': CPUTier.TIER_PI,  # Pi 5
        'Cortex-A72': CPUTier.TIER_PI,  # Pi 4
        'Cortex-A53': CPUTier.TIER_PI,  # Pi 3
        # Apple Silicon
        'M1': CPUTier.TIER_APPLE,
        'M2': CPUTier.TIER_APPLE,
        'M3': CPUTier.TIER_APPLE,
        'M4': CPUTier.TIER_APPLE,
    }

    def detect_cpu_profile(self) -> CPUProfile:
        """Detect comprehensive CPU profile."""
        try:
            if CPUINFO_AVAILABLE:
                info = cpuinfo.get_cpu_info()
                cpu_model = info.get('brand_raw', 'Unknown CPU')
                arch = info.get('arch_string_raw', platform.machine())
                features = info.get('flags', [])
            else:
                cpu_model = platform.processor() or 'Unknown CPU'
                arch = platform.machine()
                features = []

            # Get core counts
            logical_cores = os.cpu_count() or 1
            physical_cores = psutil.cpu_count(logical=False) or logical_cores

            # Get frequency info
            try:
                freq_info = psutil.cpu_freq()
                base_freq = freq_info.current / 1000 if freq_info else 0.0  # Convert to GHz
                max_freq = freq_info.max / 1000 if freq_info else 0.0
            except:
                base_freq = max_freq = 0.0

            # Determine CPU tier
            cpu_tier = self._classify_cpu_tier(cpu_model)

            return CPUProfile(
                model=cpu_model,
                architecture=arch,
                logical_cores=logical_cores,
                physical_cores=physical_cores,
                base_frequency=base_freq,
                max_frequency=max_freq,
                cpu_tier=cpu_tier,
                features=features,
            )

        except Exception as e:
            logger.error("CPU detection failed: {0}".format(e))
            return CPUProfile(
                model="Unknown CPU",
                architecture=platform.machine(),
                logical_cores=1,
                physical_cores=1,
                base_frequency=0.0,
                max_frequency=0.0,
                cpu_tier=CPUTier.TIER_UNKNOWN,
            )

    def _classify_cpu_tier(self, cpu_model: str) -> CPUTier:
        """Classify CPU tier based on model string."""
        cpu_lower = cpu_model.lower()

        # Check for exact matches first
        for model_key, tier in self.CPU_TIER_MAP.items():
            if model_key.lower() in cpu_lower:
                return tier

        # Fallback pattern matching
        if any(x in cpu_lower for x in ['cortex-a', 'arm', 'raspberry']):
            return CPUTier.TIER_PI
        elif any(x in cpu_lower for x in ['m1', 'm2', 'm3', 'm4', 'apple']):
            return CPUTier.TIER_APPLE
        elif any(x in cpu_lower for x in ['i9', 'ryzen 9', 'threadripper']):
            return CPUTier.TIER_HIGH
        elif any(x in cpu_lower for x in ['i7', 'ryzen 7']):
            return CPUTier.TIER_MID
        elif any(x in cpu_lower for x in ['i5', 'ryzen 5']):
            return CPUTier.TIER_MID
        elif any(x in cpu_lower for x in ['i3', 'ryzen 3', 'celeron', 'pentium']):
            return CPUTier.TIER_LOW
        else:
            return CPUTier.TIER_UNKNOWN


class GPUDetector:
    """Detects and classifies GPU hardware for shader operations."""

    # GPU tier mapping based on your comprehensive list
    GPU_TIER_MAP = {
        # VideoCore (Pi 4)
        'videocore': GPUTier.TIER_PI4,
        # Integrated GPUs - Low Tier
        'uhd': GPUTier.TIER_LOW,
        'iris xe': GPUTier.TIER_LOW,
        'vega 8': GPUTier.TIER_LOW,
        'vega 10': GPUTier.TIER_LOW,
        'radeon 780m': GPUTier.TIER_LOW,
        'radeon 890m': GPUTier.TIER_LOW,
        # Mid Tier GPUs
        'gtx 760': GPUTier.TIER_MID,
        'gtx 960m': GPUTier.TIER_MID,
        'gtx 1060': GPUTier.TIER_MID,
        'rx 580': GPUTier.TIER_MID,
        'arc a380': GPUTier.TIER_MID,
        'm1': GPUTier.TIER_MID,
        'gtx 1650': GPUTier.TIER_MID,
        # High Tier GPUs
        'gtx 1660': GPUTier.TIER_HIGH,
        'rtx 2060': GPUTier.TIER_HIGH,
        'rtx 2070': GPUTier.TIER_HIGH,
        'rtx 3060': GPUTier.TIER_HIGH,
        'rx 6600': GPUTier.TIER_HIGH,
        'rx 6700': GPUTier.TIER_HIGH,
        'arc a750': GPUTier.TIER_HIGH,
        'arc a770': GPUTier.TIER_HIGH,
        # Ultra Tier GPUs
        'rtx 3070': GPUTier.TIER_ULTRA,
        'rtx 3080': GPUTier.TIER_ULTRA,
        'rtx 3090': GPUTier.TIER_ULTRA,
        'rtx 4060': GPUTier.TIER_ULTRA,
        'rtx 4070': GPUTier.TIER_ULTRA,
        'rtx 4080': GPUTier.TIER_ULTRA,
        'rtx 4090': GPUTier.TIER_ULTRA,
        'rtx 5060': GPUTier.TIER_ULTRA,
        'rtx 5070': GPUTier.TIER_ULTRA,
        'rtx 5080': GPUTier.TIER_ULTRA,
        'rtx 5090': GPUTier.TIER_ULTRA,
        'rx 7800': GPUTier.TIER_ULTRA,
        'rx 7900': GPUTier.TIER_ULTRA,
        'rx 9060': GPUTier.TIER_ULTRA,
        'rx 9070': GPUTier.TIER_ULTRA,
        'm2': GPUTier.TIER_ULTRA,
        'm3': GPUTier.TIER_ULTRA,
        'm4': GPUTier.TIER_ULTRA,
    }

    def detect_gpu_profile(self) -> GPUProfile:
        """Detect comprehensive GPU profile."""
        try:
            if OPENGL_AVAILABLE:
                return self._detect_opengl_gpu()
            else:
                return self._detect_fallback_gpu()
        except Exception as e:
            logger.error("GPU detection failed: {0}".format(e))
            return self._create_fallback_profile()

    def _detect_opengl_gpu(self) -> GPUProfile:
        """Detect GPU using OpenGL queries."""
        try:
            # Initialize minimal OpenGL context
            pygame.init()
            pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.HIDDEN)

            vendor = glGetString(GL_VENDOR).decode()
            renderer = glGetString(GL_RENDERER).decode()
            version = glGetString(GL_VERSION).decode()
            shading_lang = glGetString(GL_SHADING_LANGUAGE_VERSION).decode()

            pygame.quit()

            gpu_tier = self._classify_gpu_tier(renderer)
            matrix_size = self._get_matrix_size_for_tier(gpu_tier)

            return GPUProfile(
                vendor=vendor,
                renderer=renderer,
                gl_version=version,
                glsl_version=shading_lang,
                gpu_tier=gpu_tier,
                max_matrix_size=matrix_size,
                use_half_precision=gpu_tier in [GPUTier.TIER_PI4, GPUTier.TIER_LOW],
                shader_morph_enabled=gpu_tier in [GPUTier.TIER_HIGH, GPUTier.TIER_ULTRA],
            )

        except Exception as e:
            logger.warning("OpenGL detection failed: {0}".format(e))
            return self._detect_fallback_gpu()

    def _detect_fallback_gpu(self) -> GPUProfile:
        """Fallback GPU detection using system information."""
        # Try to detect GPU from system info
        system = platform.system().lower()
        machine = platform.machine().lower()

        if 'raspberry' in platform.uname().node.lower() or 'pi' in machine:
            gpu_tier = GPUTier.TIER_PI4
        elif 'arm' in machine:
            gpu_tier = GPUTier.TIER_LOW
        else:
            gpu_tier = GPUTier.TIER_UNKNOWN

        matrix_size = self._get_matrix_size_for_tier(gpu_tier)

        return GPUProfile(
            vendor="Unknown",
            renderer="Fallback Detection",
            gl_version="Unknown",
            glsl_version="Unknown",
            gpu_tier=gpu_tier,
            max_matrix_size=matrix_size,
            use_half_precision=True,
            shader_morph_enabled=False,
        )

    def _create_fallback_profile(self) -> GPUProfile:
        """Create minimal fallback GPU profile."""
        return GPUProfile(
            vendor="Unknown",
            renderer="Detection Failed",
            gl_version="Unknown",
            glsl_version="Unknown",
            gpu_tier=GPUTier.TIER_UNKNOWN,
            max_matrix_size=16,
            use_half_precision=True,
            shader_morph_enabled=False,
        )

    def _classify_gpu_tier(self, renderer: str) -> GPUTier:
        """Classify GPU tier based on renderer string."""
        renderer_lower = renderer.lower()

        # Check for exact matches
        for gpu_key, tier in self.GPU_TIER_MAP.items():
            if gpu_key in renderer_lower:
                return tier

        return GPUTier.TIER_UNKNOWN

    def _get_matrix_size_for_tier(self, tier: GPUTier) -> int:
        """Get optimal matrix size for GPU tier."""
        size_map = {
            GPUTier.TIER_PI4: 8,
            GPUTier.TIER_LOW: 16,
            GPUTier.TIER_MID: 32,
            GPUTier.TIER_HIGH: 64,
            GPUTier.TIER_ULTRA: 128,
            GPUTier.TIER_UNKNOWN: 16,
        }
        return size_map.get(tier, 16)


class SystemStateProfiler:
    """Main system state profiler for Schwabot."""

    def __init__(self):
        self.cpu_detector = CPUDetector()
        self.gpu_detector = GPUDetector()
        self.profiles_dir = "init/system_profiles"
        self._ensure_profiles_dir()

    def _ensure_profiles_dir(self):
        """Ensure system profiles directory exists."""
        os.makedirs(self.profiles_dir, exist_ok=True)

    def build_full_system_profile(self) -> SystemProfile:
        """Build comprehensive system profile."""
        logger.info("🔍 Building comprehensive system profile...")

        # Detect device type
        device_type = self._detect_device_type()
        device_id = self._get_device_id()
        os_info = "{0} {1}".format(platform.system(), platform.release())

        # Detect hardware components
        cpu = self.cpu_detector.detect_cpu_profile()
        gpu = self.gpu_detector.detect_gpu_profile()

        # Get memory info
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024**3)
        ram_available_gb = memory.available / (1024**3)

        # Determine system tier
        system_tier = self._derive_system_tier(cpu.cpu_tier, gpu.gpu_tier)

        # Create profile
        profile_data = {
            "device_type": device_type,
            "device_id": device_id,
            "os_info": os_info,
            "cpu": {
                "model": cpu.model,
                "architecture": cpu.architecture,
                "logical_cores": cpu.logical_cores,
                "physical_cores": cpu.physical_cores,
                "base_frequency": cpu.base_frequency,
                "max_frequency": cpu.max_frequency,
                "cpu_tier": cpu.cpu_tier.value,
                "features": cpu.features,
            },
            "gpu": {
                "vendor": gpu.vendor,
                "renderer": gpu.renderer,
                "gl_version": gpu.gl_version,
                "glsl_version": gpu.glsl_version,
                "gpu_tier": gpu.gpu_tier.value,
                "max_matrix_size": gpu.max_matrix_size,
                "use_half_precision": gpu.use_half_precision,
                "shader_morph_enabled": gpu.shader_morph_enabled,
            },
            "ram_total_gb": ram_total_gb,
            "ram_available_gb": ram_available_gb,
            "system_tier": system_tier.value,
            "profile_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Generate system hash
        system_hash = hashlib.sha256(json.dumps(profile_data, sort_keys=True).encode()).hexdigest()

        profile_data["system_hash"] = system_hash

        # Create SystemProfile object
        profile = SystemProfile(
            device_type=device_type,
            device_id=device_id,
            os_info=os_info,
            cpu=cpu,
            gpu=gpu,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            system_tier=system_tier,
            profile_timestamp=profile_data["profile_timestamp"],
            system_hash=system_hash,
        )

        # Save profile
        self._save_profile(profile_data)

        logger.info("✅ System Profile Complete - Tier: {0}".format(system_tier.value))
        logger.info("🔧 CPU: {0} ({1})".format(cpu.model, cpu.cpu_tier.value))
        logger.info("🎮 GPU: {0} ({1})".format(gpu.renderer, gpu.gpu_tier.value))
        logger.info("📊 Matrix Size: {0}x{0}".format(gpu.max_matrix_size, gpu.max_matrix_size))
        logger.info("🔐 System Hash: {0}...".format(system_hash[:16]))

        return profile

    def _detect_device_type(self) -> str:
        """Detect device type from system information."""
        node = platform.uname().node.lower()
        system = platform.system().lower()

        if os.path.exists('/boot/config.txt') or "raspberrypi" in node or "pi" in node:
            return "Raspberry Pi"
        elif "microsoft" in platform.platform().lower():
            return "Windows Desktop"
        elif "apple" in platform.platform().lower() or "darwin" in system:
            return "Apple System"
        elif os.path.exists("/sys/class/power_supply/BAT0"):
            return "Laptop"
        else:
            return "Desktop"

    def _get_device_id(self) -> str:
        """Get device identifier."""
        try:
            return platform.uname().node
        except:
            return "Unknown Device"

    def _derive_system_tier(self, cpu_tier: CPUTier, gpu_tier: GPUTier) -> SystemTier:
        """Derive overall system tier from CPU and GPU tiers."""
        if cpu_tier == CPUTier.TIER_PI:
            return SystemTier.EMBEDDED
        elif cpu_tier == CPUTier.TIER_APPLE:
            return SystemTier.APPLE_SILICON
        elif cpu_tier == CPUTier.TIER_HIGH and gpu_tier == GPUTier.TIER_ULTRA:
            return SystemTier.TIER_ELITE
        elif cpu_tier in [CPUTier.TIER_MID, CPUTier.TIER_HIGH] and gpu_tier in [GPUTier.TIER_MID, GPUTier.TIER_HIGH]:
            return SystemTier.TIER_HIGH_SYSTEM
        elif gpu_tier == GPUTier.TIER_LOW:
            return SystemTier.TIER_LOW_SYSTEM
        else:
            return SystemTier.TIER_UNKNOWN

    def _save_profile(self, profile_data: Dict[str, Any]):
        """Save system profile to disk."""
        try:
            # Save latest profile
            latest_path = os.path.join(self.profiles_dir, "latest_profile.json")
            with open(latest_path, "w") as f:
                json.dump(profile_data, f, indent=2)

            # Save timestamped profile
            timestamp = profile_data["profile_timestamp"].replace(":", "-")
            timestamped_path = os.path.join(
                self.profiles_dir, "profile_{0}_{1}.json".format(timestamp, profile_data['system_hash'][:8])
            )
            with open(timestamped_path, "w") as f:
                json.dump(profile_data, f, indent=2)

            logger.info("💾 Profile saved: {0}".format(latest_path))

        except Exception as e:
            logger.error("Failed to save system profile: {0}".format(e))

    def load_latest_profile(self) -> Optional[SystemProfile]:
        """Load the latest system profile."""
        try:
            latest_path = os.path.join(self.profiles_dir, "latest_profile.json")
            if os.path.exists(latest_path):
                with open(latest_path, "r") as f:
                    data = json.load(f)

                # Reconstruct SystemProfile object
                cpu = CPUProfile(
                    model=data["cpu"]["model"],
                    architecture=data["cpu"]["architecture"],
                    logical_cores=data["cpu"]["logical_cores"],
                    physical_cores=data["cpu"]["physical_cores"],
                    base_frequency=data["cpu"]["base_frequency"],
                    max_frequency=data["cpu"]["max_frequency"],
                    cpu_tier=CPUTier(data["cpu"]["cpu_tier"]),
                    features=data["cpu"].get("features", []),
                )

                gpu = GPUProfile(
                    vendor=data["gpu"]["vendor"],
                    renderer=data["gpu"]["renderer"],
                    gl_version=data["gpu"]["gl_version"],
                    glsl_version=data["gpu"]["glsl_version"],
                    gpu_tier=GPUTier(data["gpu"]["gpu_tier"]),
                    max_matrix_size=data["gpu"]["max_matrix_size"],
                    use_half_precision=data["gpu"]["use_half_precision"],
                    shader_morph_enabled=data["gpu"]["shader_morph_enabled"],
                )

                return SystemProfile(
                    device_type=data["device_type"],
                    device_id=data["device_id"],
                    os_info=data["os_info"],
                    cpu=cpu,
                    gpu=gpu,
                    ram_total_gb=data["ram_total_gb"],
                    ram_available_gb=data["ram_available_gb"],
                    system_tier=SystemTier(data["system_tier"]),
                    profile_timestamp=data["profile_timestamp"],
                    system_hash=data["system_hash"],
                )

        except Exception as e:
            logger.error("Failed to load system profile: {0}".format(e))

        return None


# Factory functions
def create_system_profiler() -> SystemStateProfiler:
    """Create a new system state profiler."""
    return SystemStateProfiler()


def get_system_profile(force_rebuild: bool = False) -> SystemProfile:
    """Get current system profile, building if necessary."""
    profiler = create_system_profiler()

    if not force_rebuild:
        existing_profile = profiler.load_latest_profile()
        if existing_profile:
            logger.info("📋 Loaded existing profile: {0}".format(existing_profile.system_tier.value))
            return existing_profile

    logger.info("🔄 Building new system profile...")
    return profiler.build_full_system_profile()


def get_gpu_shader_config() -> Dict[str, Any]:
    """Get GPU shader configuration for current system."""
    profile = get_system_profile()

    return {
        "matrix_size": profile.gpu.max_matrix_size,
        "use_half_precision": profile.gpu.use_half_precision,
        "shader_morph_enabled": profile.gpu.shader_morph_enabled,
        "gpu_tier": profile.gpu.gpu_tier.value,
        "system_tier": profile.system_tier.value,
    }


# Export key components
__all__ = [
    "SystemStateProfiler",
    "SystemProfile",
    "CPUProfile",
    "GPUProfile",
    "SystemTier",
    "CPUTier",
    "GPUTier",
    "create_system_profiler",
    "get_system_profile",
    "get_gpu_shader_config",
]
