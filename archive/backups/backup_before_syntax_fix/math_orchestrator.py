#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Orchestrator Implementation ⚡

Provides orchestration for mathematical operations:
• Hardware selection (CPU/GPU/Auto)
• Mathematical system initialization and activation
• Hardware info and status reporting

Features:
- Hardware selection for tensor and matrix operations
- System activation/deactivation
- Hardware info and memory reporting
- Minimal, production-ready orchestration logic
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class MathOrchestrator:
    """
    Mathematical orchestrator for hardware and operation management.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False
        self._initialize_system()

    def _default_config(self) -> Dict[str, Any]:
        return {
            'enabled': True,
            'timeout': 30.0,
            'retries': 3,
            'debug': False,
            'log_level': 'INFO',
        }

    def _initialize_system(self) -> None:
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            self.initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False

    def activate(self) -> bool:
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
        try:
            self.active = False
            self.logger.info(f"✅ {self.__class__.__name__} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
        }

    def select_hardware(self, operation_type: str) -> str:
        """Select appropriate hardware for the given operation type."""
        if not self.active:
            self.logger.warning("Orchestrator not active, defaulting to CPU")
            return 'cpu'
        # Simple hardware selection logic
        if operation_type in ['tensor_process', 'tensor_score', 'tensor_recursion', 'tensor_weights', 'matrix_process', 'data_process']:
            return 'gpu' if self._check_gpu_availability() else 'cpu'
        elif operation_type in ['profit_optimization', 'profit_calculation', 'profit_allocation']:
            return 'cpu'
        else:
            return 'cpu'

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for computation."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import cupy
                return True
            except ImportError:
                return False

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about available hardware."""
        return {
            'cpu_available': True,
            'gpu_available': self._check_gpu_availability(),
            'memory_available': self._get_memory_info(),
            'preferred_hardware': 'gpu' if self._check_gpu_availability() else 'cpu'
        }

    def _get_memory_info(self) -> Dict[str, Any]:
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent
            }
        except ImportError:
            return {'total': 0, 'available': 0, 'percent': 0}

# Factory function
def create_math_orchestrator(config: Optional[Dict[str, Any]] = None) -> MathOrchestrator:
    """Create a math orchestrator instance."""
    return MathOrchestrator(config)

# Singleton instance for global use
math_orchestrator = MathOrchestrator()
