"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hash Configuration Manager
==========================
Centralized hash configuration system for Schwabot trading system.

Provides:
- Hardware auto-detection for hash truncation
- CLI override support
- Consistent hash settings across all modules
- Performance optimization for low-power hardware

Hardware Tiers:
- Pi Zero/Pico: Auto-enable truncated hashes (8-12 bytes)
- Pi 3: Auto-enable truncated hashes (12-16 bytes)
- Pi 4/Mobile: Auto-enable truncated hashes (16-24 bytes)
- Desktop/Server: Full hashes (64 bytes) unless overridden
"""

import hashlib
import logging
import platform
import psutil
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import subprocess
import sys

logger = logging.getLogger(__name__)


class HardwareTier(Enum):
"""Hardware capability tiers for hash optimization."""
PI_ZERO_PICO = "pi_zero_pico"      # Pi Zero, Pi Pico, very low power
PI_3 = "pi_3"                      # Pi 3, low power
PI_4_MOBILE = "pi_4_mobile"        # Pi 4, mobile devices
DESKTOP_SERVER = "desktop_server"  # Desktop, server, high power


@dataclass
class HashConfig:
"""Hash configuration settings."""
truncated_hash: bool = False
hash_length: int = 64
hardware_tier: HardwareTier = HardwareTier.DESKTOP_SERVER
auto_detected: bool = True
cli_override: bool = False


class HashConfigManager:
"""Centralized hash configuration manager."""

_instance = None
_initialized = False

def __new__(cls):
if cls._instance is None:
cls._instance = super(HashConfigManager, cls).__new__(cls)
return cls._instance

def __init__(self) -> None:
if not self._initialized:
self.config = HashConfig()
self._initialized = True

def initialize(self, cli_truncated_hash: bool = False, cli_hash_length: Optional[int] = None) -> None:
"""Initialize hash configuration with CLI options."""
try:
# CLI override takes precedence
if cli_truncated_hash or cli_hash_length is not None:
self.config.cli_override = True
self.config.truncated_hash = cli_truncated_hash
if cli_hash_length is not None:
self.config.hash_length = cli_hash_length
else:
self.config.hash_length = 16 if cli_truncated_hash else 64
logger.info(
f"🔧 Hash config: CLI override - truncated={
self.config.truncated_hash}, length={
self.config.hash_length}")
return

# Auto-detect hardware tier
hardware_tier = self._detect_hardware_tier()
self.config.hardware_tier = hardware_tier

# Apply hardware-based defaults
if hardware_tier == HardwareTier.PI_ZERO_PICO:
self.config.truncated_hash = True
self.config.hash_length = 8
elif hardware_tier == HardwareTier.PI_3:
self.config.truncated_hash = True
self.config.hash_length = 12
elif hardware_tier == HardwareTier.PI_4_MOBILE:
self.config.truncated_hash = True
self.config.hash_length = 16
else:  # DESKTOP_SERVER
self.config.truncated_hash = False
self.config.hash_length = 64

self.config.auto_detected = True
logger.info(
f"🔧 Hash config: Auto-detected {hardware_tier.value} - truncated={self.config.truncated_hash}, length={self.config.hash_length}")

except Exception as e:
logger.error(f"❌ Error initializing hash config: {e}")
# Fallback to safe defaults
self.config.truncated_hash = False
self.config.hash_length = 64

def _detect_hardware_tier(self) -> HardwareTier:
"""Detect hardware tier for hash optimization."""
try:
# Get system information
cpu_count = psutil.cpu_count()
memory_gb = psutil.virtual_memory().total / (1024**3)

# Check for Raspberry Pi
if self._is_raspberry_pi():
if self._is_pi_zero_or_pico():
return HardwareTier.PI_ZERO_PICO
elif self._is_pi_3():
return HardwareTier.PI_3
else:
return HardwareTier.PI_4_MOBILE

# Check for mobile/low-power devices
if cpu_count <= 4 and memory_gb <= 4:
return HardwareTier.PI_4_MOBILE

# Default to desktop/server
return HardwareTier.DESKTOP_SERVER

except Exception as e:
logger.warning(f"⚠️ Hardware detection failed: {e}, using desktop/server tier")
return HardwareTier.DESKTOP_SERVER

def _is_raspberry_pi(self) -> bool:
"""Check if running on Raspberry Pi."""
try:
# Check CPU info
with open('/proc/cpuinfo', 'r') as f:
cpuinfo = f.read()
return 'Raspberry Pi' in cpuinfo or 'BCM2708' in cpuinfo or 'BCM2709' in cpuinfo or 'BCM2835' in cpuinfo
except Exception:
# Check platform
return platform.system() == 'Linux' and 'arm' in platform.machine().lower()

def _is_pi_zero_or_pico(self) -> bool:
"""Check if running on Pi Zero or Pi Pico."""
try:
# Check for Pi Zero/Pico specific identifiers
with open('/proc/cpuinfo', 'r') as f:
cpuinfo = f.read()
return 'BCM2835' in cpuinfo and ('Pi Zero' in cpuinfo or 'Pico' in cpuinfo)
except Exception:
return False

def _is_pi_3(self) -> bool:
"""Check if running on Pi 3."""
try:
with open('/proc/cpuinfo', 'r') as f:
cpuinfo = f.read()
return 'BCM2709' in cpuinfo or 'BCM2837' in cpuinfo
except Exception:
return False

def get_config(self) -> HashConfig:
"""Get current hash configuration."""
return self.config

def get_hash_settings(self) -> Dict[str, Any]:
"""Get hash settings for module configuration."""
return {
'truncated_hash': self.config.truncated_hash,
'hash_length': self.config.hash_length,
'hardware_tier': self.config.hardware_tier.value,
'auto_detected': self.config.auto_detected,
'cli_override': self.config.cli_override
}

def generate_hash(self, data: bytes) -> str:
"""Generate hash using current configuration."""
try:
full_hash = hashlib.sha256(data).hexdigest()
if self.config.truncated_hash:
return full_hash[:self.config.hash_length]
else:
return full_hash
except Exception as e:
logger.error(f"❌ Error generating hash: {e}")
return "fallback_hash"

def generate_hash_from_string(self, data: str) -> str:
"""Generate hash from string using current configuration."""
return self.generate_hash(data.encode('utf-8'))

def get_status(self) -> Dict[str, Any]:
"""Get hash configuration status."""
return {
'hardware_tier': self.config.hardware_tier.value,
'truncated_hash': self.config.truncated_hash,
'hash_length': self.config.hash_length,
'auto_detected': self.config.auto_detected,
'cli_override': self.config.cli_override,
'initialized': self._initialized
}


# Global instance
hash_config_manager = HashConfigManager()

# Convenience functions


def get_hash_config() -> HashConfig:
"""Get hash configuration."""
return hash_config_manager.get_config()


def get_hash_settings() -> Dict[str, Any]:
"""Get hash settings."""
return hash_config_manager.get_hash_settings()


def generate_hash(data: bytes) -> str:
"""Generate hash."""
return hash_config_manager.generate_hash(data)


def generate_hash_from_string(data: str) -> str:
"""Generate hash from string."""
return hash_config_manager.generate_hash_from_string(data)
