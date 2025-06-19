"""
Enhanced GPU Hash Processor with Critical Error Integration
==========================================================

Advanced GPU hash processing system that integrates with the critical error handler
to provide robust, fault-tolerant hash computation for the news correlation profit matrix.

Key Features:
- Thermal-aware GPU utilization with automatic throttling
- Comprehensive error handling with automatic recovery
- Profit-optimized hash correlation algorithms
- Memory management with intelligent batching
- Real-time performance monitoring and adjustment
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from collections import deque
import numpy as np
import hashlib
import json

# GPU imports with fallback
try:
    import cupy as cp
    import torch
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    torch = None

# System monitoring
try:
    import GPUtil
    import psutil
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

from .critical_error_handler import CriticalErrorHandler, ErrorCategory, ErrorSeverity
from .news_profit_mathematical_bridge import MathematicalEventSignature

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    GPU_ACCELERATED = "gpu_accelerated"
    CPU_FALLBACK = "cpu_fallback"
    HYBRID_PROCESSING = "hybrid_processing"
    THERMAL_THROTTLED = "thermal_throttled"
    EMERGENCY_MODE = "emergency_mode"

class ThermalZone(Enum):
    COOL = "cool"
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"

@dataclass
class HashProcessingResult:
    """Result of hash processing operation"""
    hash_value: str
    processing_time: float
    mode_used: ProcessingMode
    correlation_strength: float
    profit_potential: float
    error_occurred: bool = False
    error_message: str = None
    fallback_used: bool = False
    thermal_state: str = None

@dataclass
class ThermalState:
    """Current thermal state of the system"""
    cpu_temp: float
    gpu_temp: float
    zone: ThermalZone
    throttle_factor: float
    cooling_time_remaining: float
    emergency_shutdown_triggered: bool
    processing_recommendation: Dict[str, float]

class EnhancedGPUHashProcessor:
    """
    Enhanced GPU hash processor with comprehensive error handling and thermal management
    """
    
    def __init__(self, config: Optional[Dict] = None, error_handler: Optional[CriticalErrorHandler] = None):
        """Initialize enhanced GPU hash processor"""
        
        self.config = config or self._default_config()
        self.error_handler = error_handler or CriticalErrorHandler()
        
        # System state
        self.gpu_available = GPU_AVAILABLE and self._check_gpu_health()
        self.current_mode = ProcessingMode.GPU_ACCELERATED if self.gpu_available else ProcessingMode.CPU_FALLBACK
        self.thermal_state = self._get_thermal_state()
        
        # Processing queues and buffers
        self.gpu_queue = asyncio.Queue(maxsize=self.config['gpu_queue_size'])
        self.cpu_queue = asyncio.Queue(maxsize=self.config['cpu_queue_size'])
        self.result_buffer: deque = deque(maxlen=self.config['result_buffer_size'])
        
        # Hash correlation tracking
        self.correlation_cache: Dict[str, float] = {}
        self.profit_correlation_history: deque = deque(maxlen=1000)
        
        # Performance monitoring
        self.processing_stats = {
            'total_hashes_processed': 0,
            'gpu_hash_count': 0,
            'cpu_hash_count': 0,
            'avg_processing_time': 0.0,
            'error_rate': 0.0,
            'thermal_throttle_count': 0
        }
        
        # Threading
        self._running = False
        self._workers = []
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # GPU memory management
        self.gpu_memory_pool = None
        if self.gpu_available:
            self._initialize_gpu_memory()
        
        logger.info(f"Enhanced GPU Hash Processor initialized - Mode: {self.current_mode.value}")

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'gpu_queue_size': 1000,
            'cpu_queue_size': 2000,
            'result_buffer_size': 5000,
            'batch_size_gpu': 100,
            'batch_size_cpu': 50,
            'thermal_monitoring_interval': 10.0,
            'max_gpu_temperature': 80.0,
            'max_cpu_temperature': 75.0,
            'thermal_throttle_threshold': 75.0,
            'emergency_shutdown_threshold': 85.0,
            'correlation_cache_size': 10000,
            'performance_window': 1000,
            'error_recovery_attempts': 3,
            'memory_pool_size_mb': 2048,
            'hash_precision_bits': 256,
            'profit_correlation_threshold': 0.3
        }

    def _check_gpu_health(self) -> bool:
        """Check if GPU is healthy and available"""
        if not GPU_AVAILABLE or not MONITORING_AVAILABLE:
            return False
        
        try:
            # Test basic GPU operation
            if cp is not None:
                test_array = cp.array([1, 2, 3, 4, 5])
                cp.cuda.Stream.null.synchronize()
            
            # Check GPU status
            gpus = GPUtil.getGPUs()
            if not gpus:
                return False
            
            gpu = gpus[0]
            if (gpu.temperature > self.config['max_gpu_temperature'] or 
                gpu.load > 0.95 or
                gpu.memoryUtil > 0.95):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"GPU health check failed: {e}")
            return False

    def _get_thermal_state(self) -> ThermalState:
        """Get current thermal state"""
        try:
            # CPU temperature
            cpu_temp = 0.0
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            cpu_temp = max(cpu_temp, entries[0].current)
            
            # GPU temperature
            gpu_temp = 0.0
            if MONITORING_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_temp = gpus[0].temperature
            
            # Determine thermal zone
            max_temp = max(cpu_temp, gpu_temp)
            if max_temp >= self.config['emergency_shutdown_threshold']:
                zone = ThermalZone.CRITICAL
            elif max_temp >= self.config['max_gpu_temperature']:
                zone = ThermalZone.HOT
            elif max_temp >= self.config['thermal_throttle_threshold']:
                zone = ThermalZone.WARM
            elif max_temp >= 60.0:
                zone = ThermalZone.NORMAL
            else:
                zone = ThermalZone.COOL
            
            # Calculate throttle factor and recommendations
            throttle_factor = self._calculate_throttle_factor(zone, max_temp)
            processing_recommendation = self._get_processing_recommendation(zone, throttle_factor)
            
            return ThermalState(
                cpu_temp=cpu_temp,
                gpu_temp=gpu_temp,
                zone=zone,
                throttle_factor=throttle_factor,
                cooling_time_remaining=0.0,
                emergency_shutdown_triggered=(zone == ThermalZone.CRITICAL),
                processing_recommendation=processing_recommendation
            )
            
        except Exception as e:
            logger.error(f"Error getting thermal state: {e}")
            # Return safe defaults
            return ThermalState(
                cpu_temp=70.0,
                gpu_temp=70.0,
                zone=ThermalZone.WARM,
                throttle_factor=0.5,
                cooling_time_remaining=0.0,
                emergency_shutdown_triggered=False,
                processing_recommendation={'cpu': 0.8, 'gpu': 0.2}
            )

    def _calculate_throttle_factor(self, zone: ThermalZone, temperature: float) -> float:
        """Calculate thermal throttle factor"""
        if zone == ThermalZone.CRITICAL:
            return 0.1  # Minimal processing
        elif zone == ThermalZone.HOT:
            return 0.3  # Heavy throttling
        elif zone == ThermalZone.WARM:
            return 0.6  # Moderate throttling
        elif zone == ThermalZone.NORMAL:
            return 0.8  # Light throttling
        else:  # COOL
            return 1.0  # Full processing

    def _get_processing_recommendation(self, zone: ThermalZone, throttle_factor: float) -> Dict[str, float]:
        """Get processing allocation recommendation based on thermal state"""
        if zone == ThermalZone.CRITICAL:
            return {'cpu': 0.2, 'gpu': 0.0}  # Emergency CPU-only
        elif zone == ThermalZone.HOT:
            return {'cpu': 0.7, 'gpu': 0.1}  # Mostly CPU
        elif zone == ThermalZone.WARM:
            return {'cpu': 0.6, 'gpu': 0.3}  # CPU-preferred
        elif zone == ThermalZone.NORMAL:
            return {'cpu': 0.4, 'gpu': 0.6}  # GPU-preferred
        else:  # COOL
            return {'cpu': 0.3, 'gpu': 0.8}  # Mostly GPU

    def _initialize_gpu_memory(self):
        """Initialize GPU memory pool"""
        if not self.gpu_available or cp is None:
            return
        
        try:
            # Create memory pool
            pool_size = self.config['memory_pool_size_mb'] * 1024 * 1024
            self.gpu_memory_pool = cp.cuda.MemoryPool()
            
            # Set as default allocator
            cp.cuda.set_allocator(self.gpu_memory_pool.malloc)
            
            logger.info(f"GPU memory pool initialized: {self.config['memory_pool_size_mb']}MB")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU memory pool: {e}")
            self.gpu_available = False

    async def process_news_hash_correlation(self, 
                                          news_signatures: List[MathematicalEventSignature],
                                          btc_patterns: Dict[str, Any]) -> Dict[str, HashProcessingResult]:
        """
        Process news hash correlations with BTC patterns using optimal processing mode
        """
        start_time = time.time()
        results = {}
        
        try:
            # Update thermal state
            self.thermal_state = self._get_thermal_state()
            
            # Adjust processing mode based on thermal state
            await self._adjust_processing_mode()
            
            # Process each news signature
            for signature in news_signatures:
                try:
                    result = await self._process_single_correlation(signature, btc_patterns)
                    results[signature.combined_signature] = result
                    
                    # Cache correlation for future use
                    if result.correlation_strength > self.config['profit_correlation_threshold']:
                        self.correlation_cache[signature.combined_signature] = result.correlation_strength
                    
                except Exception as e:
                    # Handle individual processing error
                    error_context = {
                        'signature_hash': signature.combined_signature,
                        'thermal_zone': self.thermal_state.zone.value,
                        'gpu_temperature': self.thermal_state.gpu_temp,
                        'processing_mode': self.current_mode.value
                    }
                    
                    recovery_successful = await self.error_handler.handle_critical_error(
                        ErrorCategory.NEWS_CORRELATION,
                        'EnhancedGPUHashProcessor',
                        e,
                        error_context
                    )
                    
                    if recovery_successful:
                        # Retry with fallback mode
                        result = await self._process_single_correlation_fallback(signature, btc_patterns)
                        results[signature.combined_signature] = result
                    else:
                        # Create error result
                        results[signature.combined_signature] = HashProcessingResult(
                            hash_value="",
                            processing_time=0.0,
                            mode_used=self.current_mode,
                            correlation_strength=0.0,
                            profit_potential=0.0,
                            error_occurred=True,
                            error_message=str(e),
                            fallback_used=True,
                            thermal_state=self.thermal_state.zone.value
                        )
            
            # Update performance statistics
            processing_time = time.time() - start_time
            self._update_performance_stats(len(news_signatures), processing_time)
            
            logger.info(f"Processed {len(news_signatures)} correlations in {processing_time:.3f}s using {self.current_mode.value}")
            
            return results
            
        except Exception as e:
            # Handle batch processing error
            error_context = {
                'batch_size': len(news_signatures),
                'thermal_zone': self.thermal_state.zone.value,
                'processing_mode': self.current_mode.value
            }
            
            await self.error_handler.handle_critical_error(
                ErrorCategory.GPU_HASH_COMPUTATION,
                'EnhancedGPUHashProcessor',
                e,
                error_context
            )
            
            # Return empty results on batch failure
            return {}

    async def _process_single_correlation(self, 
                                        signature: MathematicalEventSignature,
                                        btc_patterns: Dict[str, Any]) -> HashProcessingResult:
        """Process a single news-BTC correlation"""
        
        start_time = time.time()
        
        # Check cache first
        cached_correlation = self.correlation_cache.get(signature.combined_signature)
        if cached_correlation is not None:
            return HashProcessingResult(
                hash_value=signature.combined_signature,
                processing_time=time.time() - start_time,
                mode_used=ProcessingMode.CPU_FALLBACK,  # Cache lookup
                correlation_strength=cached_correlation,
                profit_potential=self._estimate_profit_potential(cached_correlation, signature),
                thermal_state=self.thermal_state.zone.value
            )
        
        # Choose processing method based on current mode and thermal state
        if (self.current_mode == ProcessingMode.GPU_ACCELERATED and 
            self.thermal_state.zone in [ThermalZone.COOL, ThermalZone.NORMAL]):
            
            try:
                correlation = await self._compute_correlation_gpu(signature, btc_patterns)
                mode_used = ProcessingMode.GPU_ACCELERATED
                
            except Exception as e:
                logger.warning(f"GPU correlation failed, falling back to CPU: {e}")
                correlation = await self._compute_correlation_cpu(signature, btc_patterns)
                mode_used = ProcessingMode.CPU_FALLBACK
                
        else:
            correlation = await self._compute_correlation_cpu(signature, btc_patterns)
            mode_used = ProcessingMode.CPU_FALLBACK
        
        processing_time = time.time() - start_time
        profit_potential = self._estimate_profit_potential(correlation, signature)
        
        return HashProcessingResult(
            hash_value=signature.combined_signature,
            processing_time=processing_time,
            mode_used=mode_used,
            correlation_strength=correlation,
            profit_potential=profit_potential,
            thermal_state=self.thermal_state.zone.value
        )

    async def _process_single_correlation_fallback(self,
                                                 signature: MathematicalEventSignature,
                                                 btc_patterns: Dict[str, Any]) -> HashProcessingResult:
        """Fallback correlation processing with minimal computation"""
        
        start_time = time.time()
        
        # Simplified correlation calculation
        try:
            # Use basic hash similarity
            correlation = self._simple_hash_correlation(signature.combined_signature, btc_patterns)
            
            return HashProcessingResult(
                hash_value=signature.combined_signature,
                processing_time=time.time() - start_time,
                mode_used=ProcessingMode.EMERGENCY_MODE,
                correlation_strength=correlation,
                profit_potential=self._estimate_profit_potential(correlation, signature),
                fallback_used=True,
                thermal_state=self.thermal_state.zone.value
            )
            
        except Exception as e:
            # Ultimate fallback - return minimal safe result
            return HashProcessingResult(
                hash_value=signature.combined_signature,
                processing_time=time.time() - start_time,
                mode_used=ProcessingMode.EMERGENCY_MODE,
                correlation_strength=0.1,  # Minimal correlation
                profit_potential=0.0,
                error_occurred=True,
                error_message=str(e),
                fallback_used=True,
                thermal_state=self.thermal_state.zone.value
            )

    async def _compute_correlation_gpu(self, 
                                     signature: MathematicalEventSignature,
                                     btc_patterns: Dict[str, Any]) -> float:
        """Compute correlation using GPU acceleration"""
        
        if not self.gpu_available or cp is None:
            raise Exception("GPU not available for correlation computation")
        
        try:
            # Convert signature to GPU arrays
            signature_bytes = signature.combined_signature.encode()
            signature_hash = hashlib.sha256(signature_bytes).digest()
            
            # Convert to CuPy array
            sig_array = cp.frombuffer(signature_hash, dtype=cp.uint8)
            
            # Process BTC patterns
            correlations = []
            for pattern_id, pattern_data in btc_patterns.items():
                btc_hash = pattern_data.get('hash', '')
                if not btc_hash:
                    continue
                
                # Convert BTC hash to array
                btc_bytes = bytes.fromhex(btc_hash) if len(btc_hash) == 64 else btc_hash[:32].encode()
                btc_array = cp.frombuffer(btc_bytes, dtype=cp.uint8)
                
                # Ensure same length
                min_len = min(len(sig_array), len(btc_array))
                sig_truncated = sig_array[:min_len]
                btc_truncated = btc_array[:min_len]
                
                # Calculate correlation on GPU
                correlation = self._gpu_correlation_calculation(sig_truncated, btc_truncated)
                correlations.append(correlation)
            
            # Return average correlation
            if correlations:
                avg_correlation = float(cp.mean(cp.array(correlations)))
                
                # Apply thermal throttling if needed
                if self.thermal_state.throttle_factor < 1.0:
                    avg_correlation *= self.thermal_state.throttle_factor
                
                return avg_correlation
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"GPU correlation computation failed: {e}")
            raise

    def _gpu_correlation_calculation(self, array1: cp.ndarray, array2: cp.ndarray) -> float:
        """Perform correlation calculation on GPU"""
        try:
            # Hamming distance-based correlation
            xor_result = cp.bitwise_xor(array1, array2)
            hamming_distance = cp.sum(cp.unpackbits(xor_result))
            max_distance = len(array1) * 8
            
            # Convert to similarity (inverse of normalized distance)
            similarity = 1.0 - (float(hamming_distance) / max_distance)
            
            # Apply bit pattern analysis for enhanced correlation
            pattern_correlation = self._gpu_bit_pattern_analysis(array1, array2)
            
            # Combine correlations
            final_correlation = 0.7 * similarity + 0.3 * pattern_correlation
            
            return float(final_correlation)
            
        except Exception as e:
            logger.error(f"GPU correlation calculation error: {e}")
            return 0.0

    def _gpu_bit_pattern_analysis(self, array1: cp.ndarray, array2: cp.ndarray) -> float:
        """Analyze bit patterns for correlation on GPU"""
        try:
            # Convert to bit patterns
            bits1 = cp.unpackbits(array1)
            bits2 = cp.unpackbits(array2)
            
            # Calculate correlation coefficient
            if len(bits1) > 1 and len(bits2) > 1:
                # Use CuPy's correlation function
                correlation_matrix = cp.corrcoef(bits1.astype(cp.float32), bits2.astype(cp.float32))
                correlation = float(correlation_matrix[0, 1])
                
                # Handle NaN
                if cp.isnan(correlation):
                    correlation = 0.0
                
                return abs(correlation)
            else:
                return 0.0
                
        except Exception:
            return 0.0

    async def _compute_correlation_cpu(self, 
                                     signature: MathematicalEventSignature,
                                     btc_patterns: Dict[str, Any]) -> float:
        """Compute correlation using CPU"""
        
        try:
            signature_hash = signature.combined_signature
            correlations = []
            
            for pattern_id, pattern_data in btc_patterns.items():
                btc_hash = pattern_data.get('hash', '')
                if not btc_hash:
                    continue
                
                # Calculate Hamming similarity
                hamming_sim = self._hamming_similarity(signature_hash[:32], btc_hash[:32])
                
                # Calculate bit pattern correlation
                bit_correlation = self._bit_pattern_correlation_cpu(signature_hash, btc_hash)
                
                # Temporal correlation
                temporal_correlation = self._temporal_correlation(pattern_data.get('timestamp', time.time()))
                
                # Combined correlation
                combined = (
                    0.5 * hamming_sim +
                    0.3 * bit_correlation +
                    0.2 * temporal_correlation
                )
                
                correlations.append(combined)
            
            if correlations:
                avg_correlation = np.mean(correlations)
                
                # Apply thermal throttling
                if self.thermal_state.throttle_factor < 1.0:
                    avg_correlation *= self.thermal_state.throttle_factor
                
                return float(avg_correlation)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"CPU correlation computation failed: {e}")
            return 0.0

    def _hamming_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate Hamming distance similarity"""
        if len(hash1) != len(hash2):
            min_len = min(len(hash1), len(hash2))
            hash1, hash2 = hash1[:min_len], hash2[:min_len]
        
        if not hash1:
            return 0.0
        
        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)

    def _bit_pattern_correlation_cpu(self, hash1: str, hash2: str) -> float:
        """Calculate bit pattern correlation on CPU"""
        try:
            # Convert to binary patterns
            bin1 = bin(int(hash1[:16], 16))[2:].zfill(64)
            bin2 = bin(int(hash2[:16], 16))[2:].zfill(64)
            
            # Create pattern arrays
            patterns1 = [int(bin1[i:i+4], 2) for i in range(0, len(bin1), 4)]
            patterns2 = [int(bin2[i:i+4], 2) for i in range(0, len(bin2), 4)]
            
            if len(patterns1) > 1 and len(patterns2) > 1:
                correlation = np.corrcoef(patterns1, patterns2)[0, 1]
                return 0.0 if np.isnan(correlation) else abs(correlation)
            else:
                return 0.0
                
        except Exception:
            return 0.0

    def _temporal_correlation(self, btc_timestamp: float) -> float:
        """Calculate temporal correlation"""
        current_time = time.time()
        time_diff = abs(current_time - btc_timestamp)
        
        # Correlation decreases with time difference
        max_diff = 3600  # 1 hour
        
        if time_diff > max_diff:
            return 0.0
        
        return 1.0 - (time_diff / max_diff)

    def _simple_hash_correlation(self, signature_hash: str, btc_patterns: Dict[str, Any]) -> float:
        """Simple hash correlation for emergency fallback"""
        if not btc_patterns:
            return 0.1  # Minimal default
        
        try:
            # Simple character-based correlation
            correlations = []
            for pattern_data in btc_patterns.values():
                btc_hash = pattern_data.get('hash', '')
                if btc_hash:
                    # Count matching characters at same positions
                    matches = sum(1 for i, (c1, c2) in enumerate(zip(signature_hash[:16], btc_hash[:16])) if c1 == c2)
                    correlation = matches / 16.0  # Normalize
                    correlations.append(correlation)
            
            return np.mean(correlations) if correlations else 0.1
            
        except Exception:
            return 0.1

    def _estimate_profit_potential(self, correlation: float, signature: MathematicalEventSignature) -> float:
        """Estimate profit potential from correlation and signature"""
        try:
            # Base profit from correlation strength
            base_profit = correlation * 100.0  # Convert to basis points
            
            # Adjust for signature properties
            entropy_bonus = signature.entropy_class * 10.0
            profit_weight_bonus = signature.profit_weight * 50.0
            
            # Combined profit potential
            total_profit = base_profit + entropy_bonus + profit_weight_bonus
            
            # Apply thermal penalty if throttled
            if self.thermal_state.throttle_factor < 1.0:
                total_profit *= self.thermal_state.throttle_factor
            
            return max(0.0, total_profit)
            
        except Exception:
            return 0.0

    async def _adjust_processing_mode(self):
        """Adjust processing mode based on thermal state"""
        
        if self.thermal_state.emergency_shutdown_triggered:
            self.current_mode = ProcessingMode.EMERGENCY_MODE
            self.gpu_available = False
            
        elif self.thermal_state.zone == ThermalZone.CRITICAL:
            self.current_mode = ProcessingMode.EMERGENCY_MODE
            
        elif self.thermal_state.zone == ThermalZone.HOT:
            self.current_mode = ProcessingMode.THERMAL_THROTTLED
            
        elif self.thermal_state.zone == ThermalZone.WARM:
            self.current_mode = ProcessingMode.HYBRID_PROCESSING
            
        elif self.gpu_available and self.thermal_state.zone in [ThermalZone.COOL, ThermalZone.NORMAL]:
            self.current_mode = ProcessingMode.GPU_ACCELERATED
            
        else:
            self.current_mode = ProcessingMode.CPU_FALLBACK

    def _update_performance_stats(self, processed_count: int, processing_time: float):
        """Update performance statistics"""
        
        with self._lock:
            self.processing_stats['total_hashes_processed'] += processed_count
            
            if self.current_mode == ProcessingMode.GPU_ACCELERATED:
                self.processing_stats['gpu_hash_count'] += processed_count
            else:
                self.processing_stats['cpu_hash_count'] += processed_count
            
            # Update average processing time
            current_avg = self.processing_stats['avg_processing_time']
            total_processed = self.processing_stats['total_hashes_processed']
            
            if total_processed > 1:
                self.processing_stats['avg_processing_time'] = (
                    (current_avg * (total_processed - processed_count) + processing_time) / total_processed
                )
            else:
                self.processing_stats['avg_processing_time'] = processing_time

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        with self._lock:
            stats = dict(self.processing_stats)
        
        # Add thermal information
        stats.update({
            'current_mode': self.current_mode.value,
            'thermal_zone': self.thermal_state.zone.value,
            'gpu_temperature': self.thermal_state.gpu_temp,
            'cpu_temperature': self.thermal_state.cpu_temp,
            'throttle_factor': self.thermal_state.throttle_factor,
            'gpu_available': self.gpu_available,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'correlation_cache_size': len(self.correlation_cache)
        })
        
        return stats

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate correlation cache hit rate"""
        # This would be implemented based on actual cache usage tracking
        return 0.0  # Placeholder

    async def start_processing(self):
        """Start background processing threads"""
        if self._running:
            return
        
        self._running = True
        
        # Start thermal monitoring
        self._monitor_thread = threading.Thread(target=self._thermal_monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Enhanced GPU Hash Processor started")

    async def stop_processing(self):
        """Stop background processing"""
        self._running = False
        
        # Stop monitoring thread
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        # Cleanup GPU memory
        if self.gpu_memory_pool:
            try:
                self.gpu_memory_pool.free_all_blocks()
            except Exception as e:
                logger.error(f"Error cleaning up GPU memory: {e}")
        
        logger.info("Enhanced GPU Hash Processor stopped")

    def _thermal_monitor_loop(self):
        """Background thermal monitoring loop"""
        while self._running:
            try:
                # Update thermal state
                self.thermal_state = self._get_thermal_state()
                
                # Check for thermal emergencies
                if self.thermal_state.emergency_shutdown_triggered:
                    logger.critical("THERMAL EMERGENCY - GPU processing disabled")
                    self.gpu_available = False
                    
                    # Trigger emergency error handling
                    asyncio.create_task(self.error_handler.handle_critical_error(
                        ErrorCategory.THERMAL_MANAGEMENT,
                        'EnhancedGPUHashProcessor',
                        Exception("Thermal emergency shutdown triggered"),
                        {
                            'gpu_temperature': self.thermal_state.gpu_temp,
                            'cpu_temperature': self.thermal_state.cpu_temp,
                            'thermal_zone': self.thermal_state.zone.value
                        }
                    ))
                
                time.sleep(self.config['thermal_monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Error in thermal monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error 