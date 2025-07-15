#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎮 ENHANCED GPU AUTO-DETECTION SYSTEM - SCHWABOT UNIVERSAL GPU SUPPORT
=====================================================================

Comprehensive GPU auto-detection system that handles any GPU type and automatically
switches to more advanced systems when available.

Features:
- Universal GPU detection (CUDA, OpenCL, Integrated Graphics)
- Automatic backend switching (CuPy, PyTorch, OpenCL, NumPy)
- Intelligent fallback chain for maximum reliability
- Support for older laptop GPUs (980m, 970m, etc.)
- Cross-platform compatibility (Windows, Linux, macOS)
- Performance monitoring and optimization
- Zero-downtime operation with automatic failover

Mathematical Core:
GPU_Score = Memory_GB × 10 + CUDA_Cores/1000 + Tier_Score
Where Tier_Score = {extreme: 100, ultra: 80, high_end: 60, mid_range: 40, low_end: 20, integrated: 5}

Author: Schwabot Team
Date: 2025-01-02
"""

import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class GPUTier(Enum):
    """GPU performance tiers."""
    EXTREME = "extreme"
    ULTRA = "ultra"
    HIGH_END = "high_end"
    MID_RANGE = "mid_range"
    LOW_END = "low_end"
    INTEGRATED = "integrated"
    CPU = "cpu"


class BackendType(Enum):
    """Computation backend types."""
    CUPY = "cupy"
    TORCH = "torch"
    OPENCL = "opencl"
    NUMPY = "numpy"


# Enhanced GPU Database with comprehensive coverage
ENHANCED_GPU_DATABASE = {
    # Laptop GPUs (including older models like 980m)
    "GTX 980M": {
        "tier": GPUTier.LOW_END,
        "memory_gb": 4.0,
        "cuda_cores": 1536,
        "memory_bandwidth_gbps": 160.0,
        "boost_clock_mhz": 1038.0,
        "type": "laptop",
        "compute_capability": "5.2"
    },
    "GTX 970M": {
        "tier": GPUTier.LOW_END,
        "memory_gb": 3.0,
        "cuda_cores": 1280,
        "memory_bandwidth_gbps": 120.0,
        "boost_clock_mhz": 1038.0,
        "type": "laptop",
        "compute_capability": "5.2"
    },
    "GTX 960M": {
        "tier": GPUTier.LOW_END,
        "memory_gb": 2.0,
        "cuda_cores": 640,
        "memory_bandwidth_gbps": 80.0,
        "boost_clock_mhz": 1176.0,
        "type": "laptop",
        "compute_capability": "5.0"
    },
    "RTX 3060 Mobile": {
        "tier": GPUTier.MID_RANGE,
        "memory_gb": 6.0,
        "cuda_cores": 3840,
        "memory_bandwidth_gbps": 336.0,
        "boost_clock_mhz": 1703.0,
        "type": "laptop",
        "compute_capability": "8.6"
    },
    "RTX 3070 Mobile": {
        "tier": GPUTier.HIGH_END,
        "memory_gb": 8.0,
        "cuda_cores": 5120,
        "memory_bandwidth_gbps": 448.0,
        "boost_clock_mhz": 1620.0,
        "type": "laptop",
        "compute_capability": "8.6"
    },
    "RTX 3080 Mobile": {
        "tier": GPUTier.ULTRA,
        "memory_gb": 8.0,
        "cuda_cores": 6144,
        "memory_bandwidth_gbps": 448.0,
        "boost_clock_mhz": 1710.0,
        "type": "laptop",
        "compute_capability": "8.6"
    },
    
    # Desktop GPUs (existing + enhanced)
    "GTX 980": {
        "tier": GPUTier.LOW_END,
        "memory_gb": 4.0,
        "cuda_cores": 2048,
        "memory_bandwidth_gbps": 224.0,
        "boost_clock_mhz": 1216.0,
        "type": "desktop",
        "compute_capability": "5.2"
    },
    "GTX 970": {
        "tier": GPUTier.LOW_END,
        "memory_gb": 4.0,
        "cuda_cores": 1664,
        "memory_bandwidth_gbps": 224.0,
        "boost_clock_mhz": 1178.0,
        "type": "desktop",
        "compute_capability": "5.2"
    },
    "GTX 960": {
        "tier": GPUTier.LOW_END,
        "memory_gb": 2.0,
        "cuda_cores": 1024,
        "memory_bandwidth_gbps": 112.0,
        "boost_clock_mhz": 1178.0,
        "type": "desktop",
        "compute_capability": "5.2"
    },
    "RTX 3060": {
        "tier": GPUTier.MID_RANGE,
        "memory_gb": 12.0,
        "cuda_cores": 3584,
        "memory_bandwidth_gbps": 360.0,
        "boost_clock_mhz": 1777.0,
        "type": "desktop",
        "compute_capability": "8.6"
    },
    "RTX 3070": {
        "tier": GPUTier.HIGH_END,
        "memory_gb": 8.0,
        "cuda_cores": 5888,
        "memory_bandwidth_gbps": 448.0,
        "boost_clock_mhz": 1725.0,
        "type": "desktop",
        "compute_capability": "8.6"
    },
    "RTX 3080": {
        "tier": GPUTier.ULTRA,
        "memory_gb": 10.0,
        "cuda_cores": 8704,
        "memory_bandwidth_gbps": 760.0,
        "boost_clock_mhz": 1710.0,
        "type": "desktop",
        "compute_capability": "8.6"
    },
    "RTX 4090": {
        "tier": GPUTier.EXTREME,
        "memory_gb": 24.0,
        "cuda_cores": 16384,
        "memory_bandwidth_gbps": 1008.0,
        "boost_clock_mhz": 2520.0,
        "type": "desktop",
        "compute_capability": "8.9"
    },
    
    # AMD GPUs (for OpenCL support)
    "RX 580": {
        "tier": GPUTier.LOW_END,
        "memory_gb": 8.0,
        "cuda_cores": 2304,  # Stream processors
        "memory_bandwidth_gbps": 256.0,
        "boost_clock_mhz": 1340.0,
        "type": "desktop",
        "compute_capability": "opencl"
    },
    "RX 6600": {
        "tier": GPUTier.MID_RANGE,
        "memory_gb": 8.0,
        "cuda_cores": 1792,
        "memory_bandwidth_gbps": 224.0,
        "boost_clock_mhz": 2491.0,
        "type": "desktop",
        "compute_capability": "opencl"
    },
    "RX 6700 XT": {
        "tier": GPUTier.HIGH_END,
        "memory_gb": 12.0,
        "cuda_cores": 2560,
        "memory_bandwidth_gbps": 384.0,
        "boost_clock_mhz": 2424.0,
        "type": "desktop",
        "compute_capability": "opencl"
    },
    
    # Integrated Graphics
    "Intel UHD Graphics": {
        "tier": GPUTier.INTEGRATED,
        "memory_gb": 0.5,
        "cuda_cores": 32,
        "memory_bandwidth_gbps": 50.0,
        "boost_clock_mhz": 1200.0,
        "type": "integrated",
        "compute_capability": "opencl"
    },
    "AMD Radeon Graphics": {
        "tier": GPUTier.INTEGRATED,
        "memory_gb": 0.5,
        "cuda_cores": 512,
        "memory_bandwidth_gbps": 50.0,
        "boost_clock_mhz": 2000.0,
        "type": "integrated",
        "compute_capability": "opencl"
    }
}


@dataclass
class GPUInfo:
    """GPU information structure."""
    name: str
    device_id: int = 0
    memory_gb: float = 0.0
    cuda_cores: int = 0
    compute_capability: str = ""
    backend: str = "numpy"
    type: str = "unknown"
    tier: GPUTier = GPUTier.INTEGRATED
    memory_bandwidth_gbps: float = 0.0
    boost_clock_mhz: float = 0.0
    platform: str = ""


@dataclass
class GPUConfig:
    """GPU configuration structure."""
    backend: str
    gpu_name: str
    gpu_tier: str
    memory_limit_gb: float
    device_id: int
    matrix_size_limit: int
    batch_size: int
    precision: str
    use_tensor_cores: bool


@dataclass
class FallbackLevel:
    """Fallback level structure."""
    type: str
    backend: str
    gpu_name: str
    gpu_tier: str
    memory_limit_gb: float
    device_id: int


class EnhancedGPUAutoDetector:
    """
    Enhanced GPU Auto-Detection System
    Handles any GPU type and automatically switches to more advanced systems
    """
    
    def __init__(self):
        self.detected_gpus = []
        self.primary_gpu = None
        self.fallback_gpus = []
        self.available_backends = []
        self.optimal_config = {}
        
    def detect_all_gpus(self) -> Dict[str, Any]:
        """Detect all available GPUs and their capabilities."""
        logger.info("🔍 Enhanced GPU Auto-Detection Starting...")
        
        detection_results = {
            'cuda_gpus': self._detect_cuda_gpus(),
            'opencl_gpus': self._detect_opencl_gpus(),
            'integrated_graphics': self._detect_integrated_graphics(),
            'available_backends': [],
            'optimal_config': {},
            'fallback_chain': [],
            'ranked_gpus': []  # Always present for test compatibility
        }
        
        # Determine available backends
        detection_results['available_backends'] = self._determine_available_backends()
        
        # Select optimal configuration
        detection_results['optimal_config'] = self._select_optimal_config(detection_results)
        
        # Create fallback chain
        detection_results['fallback_chain'] = self._create_fallback_chain(detection_results)

        # Compute ranked_gpus for detailed analysis compatibility
        all_gpus = (
            detection_results['cuda_gpus'] + 
            detection_results['opencl_gpus'] + 
            detection_results['integrated_graphics']
        )
        performance_scores = []
        for gpu in all_gpus:
            memory_score = gpu.get('memory_gb', 0) * 10
            compute_score = gpu.get('cuda_cores', 0) / 1000
            tier_score = {
                'extreme': 100,
                'ultra': 80,
                'high_end': 60,
                'mid_range': 40,
                'low_end': 20,
                'integrated': 5
            }.get(gpu.get('tier', 'integrated'), 5)
            total_score = memory_score + compute_score + tier_score
            performance_scores.append({
                'name': gpu.get('name', 'Unknown'),
                'total_score': total_score,
                'memory_score': memory_score,
                'compute_score': compute_score,
                'tier_score': tier_score
            })
        performance_scores.sort(key=lambda x: x['total_score'], reverse=True)
        detection_results['ranked_gpus'] = performance_scores
        
        logger.info(f"✅ GPU Detection Complete: {len(detection_results['cuda_gpus'])} CUDA, {len(detection_results['opencl_gpus'])} OpenCL")
        return detection_results
    
    def _detect_cuda_gpus(self) -> List[Dict[str, Any]]:
        """Detect all CUDA-capable GPUs."""
        cuda_gpus = []
        
        try:
            # Try CuPy detection
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()
            
            for device_id in range(device_count):
                try:
                    device = cp.cuda.Device(device_id)
                    props = device.attributes
                    
                    gpu_info = {
                        'device_id': device_id,
                        'name': props['Name'].decode('utf-8'),
                        'memory_gb': props['TotalGlobalMem'] / (1024**3),
                        'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                        'cuda_cores': props.get('MultiProcessorCount', 0) * 128,  # Estimate
                        'backend': 'cupy',
                        'type': 'cuda'
                    }
                    
                    # Lookup in database for detailed specs
                    db_info = self._lookup_gpu_in_database(gpu_info['name'])
                    if db_info:
                        gpu_info.update(db_info)
                    
                    cuda_gpus.append(gpu_info)
                    logger.info(f"🎮 CUDA GPU detected: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
                    
                except Exception as e:
                    logger.warning(f"Failed to get info for CUDA device {device_id}: {e}")
                    
        except ImportError:
            logger.info("CuPy not available for CUDA detection")
        except Exception as e:
            logger.warning(f"CUDA detection failed: {e}")
        
        # Try PyTorch CUDA detection as fallback
        try:
            import torch
            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(device_id)
                    
                    gpu_info = {
                        'device_id': device_id,
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3),
                        'compute_capability': f"{props.major}.{props.minor}",
                        'cuda_cores': props.multi_processor_count * 128,  # Estimate
                        'backend': 'torch',
                        'type': 'cuda'
                    }
                    
                    # Avoid duplicates
                    if not any(g['name'] == gpu_info['name'] for g in cuda_gpus):
                        db_info = self._lookup_gpu_in_database(gpu_info['name'])
                        if db_info:
                            gpu_info.update(db_info)
                        
                        cuda_gpus.append(gpu_info)
                        logger.info(f"🎮 PyTorch CUDA GPU detected: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
                        
        except ImportError:
            logger.info("PyTorch not available for CUDA detection")
        except Exception as e:
            logger.warning(f"PyTorch CUDA detection failed: {e}")
        
        return cuda_gpus
    
    def _detect_opencl_gpus(self) -> List[Dict[str, Any]]:
        """Detect all OpenCL-capable GPUs."""
        opencl_gpus = []
        
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                
                for device in devices:
                    gpu_info = {
                        'name': device.name,
                        'memory_gb': device.global_mem_size / (1024**3),
                        'compute_capability': 'opencl',
                        'cuda_cores': device.max_compute_units * 64,  # Estimate
                        'backend': 'opencl',
                        'type': 'opencl',
                        'platform': platform.name
                    }
                    
                    # Lookup in database
                    db_info = self._lookup_gpu_in_database(gpu_info['name'])
                    if db_info:
                        gpu_info.update(db_info)
                    
                    opencl_gpus.append(gpu_info)
                    logger.info(f"🎮 OpenCL GPU detected: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
                    
        except ImportError:
            logger.info("PyOpenCL not available for OpenCL detection")
        except Exception as e:
            logger.warning(f"OpenCL detection failed: {e}")
        
        return opencl_gpus
    
    def _detect_integrated_graphics(self) -> List[Dict[str, Any]]:
        """Detect integrated graphics solutions."""
        integrated_gpus = []
        
        try:
            # Windows detection
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
                    capture_output=True, text=True
                )
                
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            name = ' '.join(parts[:-1])
                            memory_bytes = int(parts[-1]) if parts[-1].isdigit() else 0
                            
                            # Check if it's integrated
                            if any(keyword in name.lower() for keyword in ['intel', 'uhd', 'radeon', 'vega', 'integrated']):
                                gpu_info = {
                                    'name': name,
                                    'memory_gb': memory_bytes / (1024**3) if memory_bytes > 0 else 0.5,
                                    'compute_capability': 'opencl',
                                    'cuda_cores': 32,  # Conservative estimate
                                    'backend': 'opencl',
                                    'type': 'integrated'
                                }
                                
                                db_info = self._lookup_gpu_in_database(gpu_info['name'])
                                if db_info:
                                    gpu_info.update(db_info)
                                
                                integrated_gpus.append(gpu_info)
                                logger.info(f"🎮 Integrated GPU detected: {gpu_info['name']}")
            
            # Linux detection
            elif platform.system() == "Linux":
                try:
                    result = subprocess.run(["lspci"], capture_output=True, text=True)
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line or 'Display' in line:
                            if any(keyword in line.lower() for keyword in ['intel', 'amd', 'radeon']):
                                gpu_info = {
                                    'name': line.split(':')[-1].strip(),
                                    'memory_gb': 0.5,  # Conservative estimate
                                    'compute_capability': 'opencl',
                                    'cuda_cores': 32,
                                    'backend': 'opencl',
                                    'type': 'integrated'
                                }
                                
                                integrated_gpus.append(gpu_info)
                                logger.info(f"🎮 Integrated GPU detected: {gpu_info['name']}")
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Integrated graphics detection failed: {e}")
        
        return integrated_gpus
    
    def _lookup_gpu_in_database(self, gpu_name: str) -> Optional[Dict[str, Any]]:
        """Lookup GPU in enhanced database."""
        # Normalize GPU name for matching
        normalized_name = gpu_name.upper()
        
        # Try exact match first
        if normalized_name in ENHANCED_GPU_DATABASE:
            return ENHANCED_GPU_DATABASE[normalized_name]
        
        # Try partial matches
        for db_name, specs in ENHANCED_GPU_DATABASE.items():
            if db_name in normalized_name or normalized_name in db_name:
                return specs
        
        # Try pattern matching for common variations
        patterns = {
            'GTX 980M': ['980M', 'GTX 980M', 'GEFORCE GTX 980M'],
            'GTX 970M': ['970M', 'GTX 970M', 'GEFORCE GTX 970M'],
            'RTX 3060': ['3060', 'RTX 3060', 'GEFORCE RTX 3060'],
            'RTX 3070': ['3070', 'RTX 3070', 'GEFORCE RTX 3070'],
            'RTX 3080': ['3080', 'RTX 3080', 'GEFORCE RTX 3080'],
            'RTX 4090': ['4090', 'RTX 4090', 'GEFORCE RTX 4090']
        }
        
        for pattern_name, pattern_list in patterns.items():
            if any(pattern in normalized_name for pattern in pattern_list):
                if pattern_name in ENHANCED_GPU_DATABASE:
                    return ENHANCED_GPU_DATABASE[pattern_name]
        
        return None
    
    def _determine_available_backends(self) -> List[str]:
        """Determine available computation backends."""
        backends = ['numpy']  # Always available
        
        # Check CUDA backends
        try:
            import cupy as cp
            if cp.cuda.is_available():
                backends.append('cupy')
        except:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                backends.append('torch')
        except:
            pass
        
        # Check OpenCL backends
        try:
            import pyopencl as cl
            if cl.get_platforms():
                backends.append('opencl')
        except:
            pass
        
        logger.info(f"Available backends: {backends}")
        return backends
    
    def _select_optimal_config(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal configuration based on available GPUs."""
        all_gpus = (
            detection_results['cuda_gpus'] + 
            detection_results['opencl_gpus'] + 
            detection_results['integrated_graphics']
        )
        
        if not all_gpus:
            return {
                'backend': 'numpy',
                'gpu_name': 'CPU',
                'gpu_tier': 'cpu',
                'memory_limit_gb': 0.0,
                'device_id': -1,
                'matrix_size_limit': 1000,
                'batch_size': 1,
                'precision': 'float32',
                'use_tensor_cores': False
            }
        
        # Sort by performance (memory + compute capability)
        def gpu_score(gpu):
            memory_score = gpu.get('memory_gb', 0) * 10
            compute_score = gpu.get('cuda_cores', 0) / 1000
            tier_score = {
                'extreme': 100,
                'ultra': 80,
                'high_end': 60,
                'mid_range': 40,
                'low_end': 20,
                'integrated': 5
            }.get(gpu.get('tier', 'integrated'), 5)
            
            return memory_score + compute_score + tier_score
        
        # Select best GPU
        best_gpu = max(all_gpus, key=gpu_score)
        
        # Determine configuration based on GPU tier
        tier = best_gpu.get('tier', 'integrated')
        
        config = {
            'backend': best_gpu.get('backend', 'numpy'),
            'gpu_name': best_gpu.get('name', 'Unknown'),
            'gpu_tier': tier,
            'memory_limit_gb': best_gpu.get('memory_gb', 0) * 0.8,  # Use 80% of GPU memory
            'device_id': best_gpu.get('device_id', 0)
        }
        
        # Set processing limits based on tier
        if tier == 'extreme':
            config.update({
                'matrix_size_limit': 10000,
                'batch_size': 20,
                'precision': 'float32',
                'use_tensor_cores': True
            })
        elif tier == 'ultra':
            config.update({
                'matrix_size_limit': 8000,
                'batch_size': 15,
                'precision': 'float32',
                'use_tensor_cores': True
            })
        elif tier == 'high_end':
            config.update({
                'matrix_size_limit': 6000,
                'batch_size': 10,
                'precision': 'float32',
                'use_tensor_cores': False
            })
        elif tier == 'mid_range':
            config.update({
                'matrix_size_limit': 4000,
                'batch_size': 8,
                'precision': 'float32',
                'use_tensor_cores': False
            })
        elif tier == 'low_end':
            config.update({
                'matrix_size_limit': 2000,
                'batch_size': 5,
                'precision': 'float32',
                'use_tensor_cores': False
            })
        else:  # integrated
            config.update({
                'matrix_size_limit': 1000,
                'batch_size': 3,
                'precision': 'float32',
                'use_tensor_cores': False
            })
        
        logger.info(f"Optimal config selected: {config['gpu_name']} ({config['gpu_tier']})")
        return config
    
    def _create_fallback_chain(self, detection_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback chain for different computation methods."""
        fallback_chain = []
        
        # Add all detected GPUs in order of preference
        all_gpus = (
            detection_results['cuda_gpus'] + 
            detection_results['opencl_gpus'] + 
            detection_results['integrated_graphics']
        )
        
        # Sort by performance
        def gpu_score(gpu):
            memory_score = gpu.get('memory_gb', 0) * 10
            compute_score = gpu.get('cuda_cores', 0) / 1000
            tier_score = {
                'extreme': 100,
                'ultra': 80,
                'high_end': 60,
                'mid_range': 40,
                'low_end': 20,
                'integrated': 5
            }.get(gpu.get('tier', 'integrated'), 5)
            
            return memory_score + compute_score + tier_score
        
        sorted_gpus = sorted(all_gpus, key=gpu_score, reverse=True)
        
        # Add GPU fallbacks
        for gpu in sorted_gpus:
            fallback_chain.append({
                'type': 'gpu',
                'backend': gpu.get('backend', 'numpy'),
                'gpu_name': gpu.get('name', 'Unknown'),
                'gpu_tier': gpu.get('tier', 'integrated'),
                'memory_limit_gb': gpu.get('memory_gb', 0) * 0.8,
                'device_id': gpu.get('device_id', 0)
            })
        
        # Add CPU fallback
        fallback_chain.append({
            'type': 'cpu',
            'backend': 'numpy',
            'gpu_name': 'CPU',
            'gpu_tier': 'cpu',
            'memory_limit_gb': 0.0,
            'device_id': -1
        })
        
        logger.info(f"Fallback chain created with {len(fallback_chain)} levels")
        return fallback_chain


class EnhancedGPULogicMapper:
    """
    Enhanced GPU Logic Mapper with automatic switching between systems
    """
    
    def __init__(self):
        self.auto_detector = EnhancedGPUAutoDetector()
        self.detection_results = self.auto_detector.detect_all_gpus()
        self.current_backend = self.detection_results['optimal_config']['backend']
        self.fallback_chain = self.detection_results['fallback_chain']
        self.current_fallback_index = 0
        
        # Initialize current backend
        self._initialize_current_backend()
        
        logger.info(f"🎮 Enhanced GPU Logic Mapper initialized with {self.current_backend}")
    
    def _initialize_current_backend(self):
        """Initialize the current computation backend."""
        try:
            if self.current_backend == 'cupy':
                import cupy as cp
                self.xp = cp
                logger.info("✅ CuPy backend initialized")
            elif self.current_backend == 'torch':
                import torch
                self.xp = torch
                logger.info("✅ PyTorch backend initialized")
            elif self.current_backend == 'opencl':
                import pyopencl as cl
                self.xp = cl
                logger.info("✅ OpenCL backend initialized")
            else:
                import numpy as np
                self.xp = np
                logger.info("✅ NumPy backend initialized")
        except Exception as e:
            logger.warning(f"Backend initialization failed: {e}")
            self._switch_to_fallback()
    
    def _switch_to_fallback(self):
        """Switch to next available fallback."""
        if self.current_fallback_index < len(self.fallback_chain) - 1:
            self.current_fallback_index += 1
            fallback = self.fallback_chain[self.current_fallback_index]
            self.current_backend = fallback['backend']
            
            logger.info(f"🔄 Switching to fallback: {fallback['gpu_name']} ({fallback['backend']})")
            self._initialize_current_backend()
        else:
            logger.error("❌ All fallbacks exhausted, using CPU-only mode")
            import numpy as np
            self.xp = np
            self.current_backend = 'numpy'
    
    def map_strategy_to_gpu(self, strategy_hash: str, strategy_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Map strategy to GPU with automatic fallback."""
        max_retries = len(self.fallback_chain)
        
        for attempt in range(max_retries):
            try:
                result = self._map_strategy_with_current_backend(strategy_hash, strategy_matrix)
                result['backend_used'] = self.current_backend
                result['gpu_name'] = self.fallback_chain[self.current_fallback_index]['gpu_name']
                return result
            except Exception as e:
                logger.warning(f"Mapping failed with {self.current_backend}: {e}")
                if attempt < max_retries - 1:
                    self._switch_to_fallback()
                else:
                    # Final fallback to CPU
                    return self._map_strategy_to_cpu(strategy_hash, strategy_matrix)
    
    def _map_strategy_with_current_backend(self, strategy_hash: str, strategy_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Map strategy using current backend."""
        if strategy_matrix is None:
            strategy_matrix = self._generate_strategy_matrix(strategy_hash)
        
        # Convert to current backend's array type
        if self.current_backend == 'cupy':
            gpu_matrix = self.xp.asarray(strategy_matrix)
        elif self.current_backend == 'torch':
            gpu_matrix = self.xp.tensor(strategy_matrix, dtype=self.xp.float32)
        elif self.current_backend == 'opencl':
            # OpenCL implementation would go here
            gpu_matrix = strategy_matrix  # Fallback to numpy for now
        else:
            gpu_matrix = strategy_matrix
        
        # Perform tensor analysis
        analysis_results = self._perform_tensor_analysis(gpu_matrix)
        
        return {
            "status": "success",
            "strategy_hash": strategy_hash,
            "matrix_size": strategy_matrix.size,
            "tensor_analysis_results": analysis_results,
            "backend_used": self.current_backend
        }
    
    def _generate_strategy_matrix(self, strategy_hash: str) -> np.ndarray:
        """Generate strategy matrix from hash."""
        # Convert hash to numerical values
        hash_bytes = strategy_hash.encode('utf-8')
        hash_int = int.from_bytes(hash_bytes, byteorder='big')
        
        # Create matrix based on hash
        size = 64  # 64x64 matrix
        matrix = np.zeros((size, size), dtype=np.float32)
        
        for i in range(size):
            for j in range(size):
                # Use hash to generate matrix values
                seed = (hash_int + i * size + j) % 1000000
                matrix[i, j] = (seed / 1000000.0) * 2.0 - 1.0  # Range [-1, 1]
        
        return matrix
    
    def _perform_tensor_analysis(self, matrix) -> Dict[str, Any]:
        """Perform tensor analysis with current backend."""
        analysis_results = {}
        
        try:
            if self.current_backend == 'cupy':
                # CuPy tensor operations
                analysis_results['eigenvalues'] = self.xp.linalg.eigvals(matrix)
                analysis_results['singular_values'] = self.xp.linalg.svd(matrix)[1]
                analysis_results['correlation'] = self.xp.corrcoef(matrix)
                
            elif self.current_backend == 'torch':
                # PyTorch tensor operations
                analysis_results['eigenvalues'] = self.xp.linalg.eigvals(matrix)
                analysis_results['singular_values'] = self.xp.linalg.svd(matrix)[1]
                analysis_results['correlation'] = self.xp.corrcoef(matrix)
                
            else:
                # NumPy operations
                analysis_results['eigenvalues'] = np.linalg.eigvals(matrix)
                analysis_results['singular_values'] = np.linalg.svd(matrix)[1]
                analysis_results['correlation'] = np.corrcoef(matrix)
                
        except Exception as e:
            logger.warning(f"Tensor analysis failed: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _map_strategy_to_cpu(self, strategy_hash: str, strategy_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Map strategy to CPU as final fallback."""
        if strategy_matrix is None:
            strategy_matrix = self._generate_strategy_matrix(strategy_hash)
        
        analysis_results = self._perform_tensor_analysis(strategy_matrix)
        
        return {
            "status": "success",
            "strategy_hash": strategy_hash,
            "matrix_size": strategy_matrix.size,
            "tensor_analysis_results": analysis_results,
            "backend_used": "numpy",
            "gpu_name": "CPU"
        }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get detailed GPU information."""
        return {
            'detection_results': self.detection_results,
            'current_backend': self.current_backend,
            'current_fallback_index': self.current_fallback_index,
            'fallback_chain': self.fallback_chain
        }


# Factory functions
def create_enhanced_gpu_auto_detector() -> EnhancedGPUAutoDetector:
    """Create an enhanced GPU auto-detector instance."""
    return EnhancedGPUAutoDetector()


def create_enhanced_gpu_logic_mapper() -> EnhancedGPULogicMapper:
    """Create an enhanced GPU logic mapper instance."""
    return EnhancedGPULogicMapper()


def main():
    """Main function for testing the enhanced GPU auto-detection system."""
    logger.info("🎮 Testing Enhanced GPU Auto-Detection System")
    
    # Test auto-detector
    detector = create_enhanced_gpu_auto_detector()
    results = detector.detect_all_gpus()
    
    print("\n🎮 GPU DETECTION RESULTS")
    print("=" * 50)
    
    print(f"Primary Configuration:")
    print(f"  GPU: {results['optimal_config']['gpu_name']}")
    print(f"  Backend: {results['optimal_config']['backend']}")
    print(f"  Tier: {results['optimal_config']['gpu_tier']}")
    print(f"  Memory Limit: {results['optimal_config']['memory_limit_gb']:.1f} GB")
    print(f"  Matrix Size Limit: {results['optimal_config']['matrix_size_limit']}")
    
    print(f"\nAvailable Backends: {results['available_backends']}")
    
    print(f"\nFallback Chain:")
    for i, fallback in enumerate(results['fallback_chain']):
        status = "🟢 ACTIVE" if i == 0 else "⚪ FALLBACK"
        print(f"  {i+1}. {fallback['gpu_name']} ({fallback['backend']}) {status}")
    
    print(f"\nDetected GPUs:")
    for gpu in results['cuda_gpus']:
        print(f"  CUDA: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    for gpu in results['opencl_gpus']:
        print(f"  OpenCL: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    for gpu in results['integrated_graphics']:
        print(f"  Integrated: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    
    # Test logic mapper
    mapper = create_enhanced_gpu_logic_mapper()
    test_hash = "test_strategy_hash_12345"
    result = mapper.map_strategy_to_gpu(test_hash)
    
    print(f"\n🎯 Strategy Mapping Test:")
    print(f"  Hash: {test_hash}")
    print(f"  Backend Used: {result['backend_used']}")
    print(f"  GPU Name: {result['gpu_name']}")
    print(f"  Matrix Size: {result['matrix_size']}")
    print(f"  Status: {result['status']}")


if __name__ == "__main__":
    main() 