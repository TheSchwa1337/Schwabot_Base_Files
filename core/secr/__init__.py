"""
SECR - Sustainment-Encoded Collapse Resolver
===========================================

Core module for recursive failure analysis and adaptive resolution.
Transforms every system failure into forward momentum through intelligent
learning and real-time parameter adjustment.

Author: Schwabot Development Team
Version: 0.5.0-alpha
"""

from .failure_logger import FailureLogger, FailureKey, FailureGroup
from .resolver_matrix import ResolverMatrix, ResolverRegistry
from .injector import ConfigInjector, PatchConfig
from .watchdog import SECRWatchdog
from .allocator import ResourceAllocator, PressureIndex
from .adaptive_icap import AdaptiveICAPTuner

__version__ = "0.5.0-alpha"
__all__ = [
    "FailureLogger",
    "FailureKey", 
    "FailureGroup",
    "ResolverMatrix",
    "ResolverRegistry",
    "ConfigInjector",
    "PatchConfig",
    "SECRWatchdog",
    "ResourceAllocator",
    "PressureIndex",
    "AdaptiveICAPTuner"
] 