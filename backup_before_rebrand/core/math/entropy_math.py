"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy Math Module for Schwabot
================================
Provides core entropy mathematical calculations and utilities.
"""

import time
import logging
from dataclasses import dataclass, field
import numpy as np


logger = logging.getLogger(__name__)

# Check for math infrastructure availability
try:
from .math_cache import MathResultCache
from .math_config_manager import MathConfigManager
from .math_orchestrator import MathOrchestrator

MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
MATH_INFRASTRUCTURE_AVAILABLE = False
logger.warning("Math infrastructure not available")


@dataclass
class Config:
"""Configuration data class."""

enabled: bool = True
timeout: float = 30.0
retries: int = 3
debug: bool = False


@dataclass
class Result:
"""Result data class."""

success: bool = False
data: Optional[Dict[str, Any]] = None
error: Optional[str] = None
timestamp: float = field(default_factory=time.time)


class EntropyMath:
"""
EntropyMath Implementation
Provides core entropy math functionality.
"""

def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
"""Initialize EntropyMath with configuration."""
self.config = config or self._default_config()
self.logger = logging.getLogger(__name__)
self.active = False
self.initialized = False

# Initialize math infrastructure if available
if MATH_INFRASTRUCTURE_AVAILABLE:
self.math_config = MathConfigManager()
self.math_cache = MathResultCache()
self.math_orchestrator = MathOrchestrator()

self._initialize_system()

def _default_config(self) -> Dict[str, Any]:
"""Default configuration."""
return {
"enabled": True,
"timeout": 30.0,
"retries": 3,
"debug": False,
"log_level": "INFO",
}

def _initialize_system(self) -> None:
"""Initialize the system."""
try:
self.logger.info(f"Initializing {self.__class__.__name__}")
self.initialized = True
self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
except Exception as e:
self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
self.initialized = False

def activate(self) -> bool:
"""Activate the system."""
if not self.initialized:
self.logger.error("System not initialized")
return False

try:
self.active = True
self.logger.info(f"✅ {self.__class__.__name__} activated")
return True
except Exception as e:
self.logger.error(f"❌ Error activating {self.__class__.__name__}: {e}")
return False

def deactivate(self) -> bool:
"""Deactivate the system."""
try:
self.active = False
self.logger.info(f"✅ {self.__class__.__name__} deactivated")
return True
except Exception as e:
self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
return False

def get_status(self) -> Dict[str, Any]:
"""Get system status."""
return {
"active": self.active,
"initialized": self.initialized,
"config": self.config,
}

def calculate_entropy(self, probabilities) -> float:
"""Calculate entropy."""
if not isinstance(probabilities, (list, tuple, np.ndarray)):
raise ValueError("Probabilities must be array-like")

probs = np.array(probabilities)
probs = probs[probs > 0]  # Remove zero probabilities
if len(probs) == 0:
return 0.0

return -np.sum(probs * np.log2(probs))

def calculate_shannon_entropy(self, probabilities: List[float]) -> float:
"""
Calculate Shannon entropy: H = -Σ p_i * log2(p_i)

Args:
probabilities: List of probabilities

Returns:
Shannon entropy value
"""
try:
probs = np.array(probabilities)
probs = probs[probs > 0]  # Remove zero probabilities

if len(probs) == 0:
return 0.0

# Normalize probabilities
probs = probs / np.sum(probs)

# Calculate Shannon entropy
entropy = -np.sum(probs * np.log2(probs + 1e-10))

return float(entropy)

except Exception as e:
self.logger.error(f"Shannon entropy calculation failed: {e}")
return 0.0

def calculate_market_entropy(self, price_changes: List[float]) -> float:
"""
Calculate market entropy from price changes.

Args:
price_changes: List of price changes

Returns:
Market entropy value
"""
try:
changes = np.array(price_changes)

# Calculate absolute changes
abs_changes = np.abs(changes)
total_change = np.sum(abs_changes)

if total_change == 0:
return 0.0

# Calculate probabilities
probabilities = abs_changes / total_change

# Calculate entropy
entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

return float(entropy)

except Exception as e:
self.logger.error(f"Market entropy calculation failed: {e}")
return 0.0

def calculate_zbe_entropy(self, price_changes: List[float]) -> float:
"""
Calculate ZBE (Zero-Based Entropy) from price changes.

Args:
price_changes: List of price changes

Returns:
ZBE entropy value
"""
try:
changes = np.array(price_changes)

# Calculate zero-based probabilities
positive_changes = changes[changes > 0]
negative_changes = changes[changes < 0]

total_positive = (
np.sum(positive_changes) if len(positive_changes) > 0 else 0
)
total_negative = (
np.sum(np.abs(negative_changes)) if len(negative_changes) > 0 else 0
)

if total_positive == 0 and total_negative == 0:
return 0.0

# Calculate ZBE
zbe = (
total_positive / (total_positive + total_negative)
if (total_positive + total_negative) > 0
else 0.5
)

return float(zbe)

except Exception as e:
self.logger.error(f"ZBE entropy calculation failed: {e}")
return 0.5


# Factory function


def create_entropy_math(config: Optional[Dict[str, Any]] = None) -> EntropyMath:
"""Create an EntropyMath instance."""
return EntropyMath(config)
