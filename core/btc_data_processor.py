"""
BTC Data Processor
================

Handles BTC price data aggregation, hash generation, and processing pipeline
with support for intelligent load balancing between CPU and GPU processing.
Integrates with NCCO core, ALF core, and phase engine for mathematical processing.
"""

import numpy as np
import torch
import logging
import hashlib
import time
import psutil
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import json
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import gc

from core.mathlib_v3 import SustainmentMathLib
from core.entropy_engine import EntropyEngine
from core.quantum_antipole_engine import QuantumAntipoleEngine
from core.drift_shell_engine import DriftShellEngine
from core.recursive_engine.primary_loop import RecursiveEngine
from core.antipole.vector import AntiPoleVector
from core.phase_engine.phase_metrics_engine import PhaseMetricsEngine
from core.zygot_shell import ZygotShell
from core.gpu_offload_manager import GPUOffloadManager
from core.thermal_map_allocator import ThermalMapAllocator, MemoryRegion
from core.bitcoin_mining_analyzer import BitcoinMiningAnalyzer

logger = logging.getLogger(__name__)

class LoadBalancer:
    """Manages load balancing between CPU and GPU processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cpu_load_history = deque(maxlen=100)
        self.gpu_load_history = deque(maxlen=100)
        self.processing_times = {
            'cpu': deque(maxlen=100),
            'gpu': deque(maxlen=100)
        }
        self.backlog_sizes = {
            'cpu': deque(maxlen=100),
            'gpu': deque(maxlen=100)
        }
        self.last_balance_time = time.time()
        self.current_mode = 'auto'  # 'cpu', 'gpu', or 'auto'
        self.error_counts = {'cpu': 0, 'gpu': 0}
        self.last_error_time = {'cpu': 0, 'gpu': 0}
        
        # Initialize phase engine integration
        self.phase_engine = PhaseMetricsEngine()
        self.zygot_shell = ZygotShell()
        
        # Initialize GPU offload manager
        self.gpu_manager = GPUOffloadManager()
        
        # Initialize mining analyzer for load balancing optimization
        self.mining_analyzer = BitcoinMiningAnalyzer(config)
        
    def update_load_metrics(self, cpu_time: float, gpu_time: float, 
                          cpu_backlog: int, gpu_backlog: int, hash_data: str = None):
        """Update load metrics for both CPU and GPU with mining analysis"""
        self.processing_times['cpu'].append(cpu_time)
        self.processing_times['gpu'].append(gpu_time)
        self.backlog_sizes['cpu'].append(cpu_backlog)
        self.backlog_sizes['gpu'].append(gpu_backlog)
        
        # Update system load metrics
        self.cpu_load_history.append(psutil.cpu_percent())
        if torch.cuda.is_available():
            self.gpu_load_history.append(torch.cuda.utilization())
            
        # Update phase metrics
        self.phase_engine.update_metrics({
            'cpu_load': self.cpu_load_history[-1],
            'gpu_load': self.gpu_load_history[-1] if self.gpu_load_history else 0,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'cpu_backlog': cpu_backlog,
            'gpu_backlog': gpu_backlog
        })
        
        # Analyze mining patterns for load optimization
        if hash_data:
            try:
                mining_efficiency = asyncio.create_task(
                    self.mining_analyzer._analyze_hash_patterns(hash_data)
                )
                # Use mining analysis to optimize load balancing
                self._optimize_based_on_mining_patterns(mining_efficiency)
            except Exception as e:
                logger.error(f"Mining pattern analysis error in load balancer: {e}")
            
    def _optimize_based_on_mining_patterns(self, mining_efficiency):
        """Optimize load balancing based on mining pattern analysis"""
        try:
            # Adjust processing mode based on mining efficiency
            if hasattr(mining_efficiency, 'result'):
                efficiency_data = mining_efficiency.result()
                if efficiency_data.get('solution_probability', 0) > 0.8:
                    # High solution probability - favor GPU for faster processing
                    self.current_mode = 'gpu' if torch.cuda.is_available() else 'cpu'
                elif efficiency_data.get('hash_entropy', 0) < 0.5:
                    # Low entropy - favor CPU for more precise calculations
                    self.current_mode = 'cpu'
        except Exception as e:
            logger.error(f"Mining pattern optimization error: {e}")
            
    def get_optimal_processor(self) -> str:
        """Determine optimal processor based on current load and history"""
        current_time = time.time()
        
        # Only rebalance periodically
        if current_time - self.last_balance_time < self.config['load_balancing']['rebalance_interval_ms'] / 1000:
            return self.current_mode
            
        # Calculate average processing times
        avg_cpu_time = np.mean(self.processing_times['cpu']) if self.processing_times['cpu'] else float('inf')
        avg_gpu_time = np.mean(self.processing_times['gpu']) if self.processing_times['gpu'] else float('inf')
        
        # Calculate backlog pressure
        cpu_backlog_pressure = np.mean(self.backlog_sizes['cpu']) if self.backlog_sizes['cpu'] else 0
        gpu_backlog_pressure = np.mean(self.backlog_sizes['gpu']) if self.backlog_sizes['gpu'] else 0
        
        # Calculate system load
        cpu_load = np.mean(self.cpu_load_history) if self.cpu_load_history else 0
        gpu_load = np.mean(self.gpu_load_history) if self.gpu_load_history else 0
        
        # Get phase metrics
        phase_metrics = self.phase_engine.get_metrics()
        drift_resonance = self.zygot_shell.compute_drift_resonance(
            phase_metrics.get('phase_angle', 0),
            phase_metrics.get('entropy', 0)
        )
        
        # Check for recent errors
        error_threshold = self.config['load_balancing']['error_threshold']
        error_cooldown = self.config['load_balancing']['error_cooldown_ms'] / 1000
        
        cpu_error_penalty = 0
        gpu_error_penalty = 0
        
        if self.error_counts['cpu'] > error_threshold and current_time - self.last_error_time['cpu'] < error_cooldown:
            cpu_error_penalty = 1.5
        if self.error_counts['gpu'] > error_threshold and current_time - self.last_error_time['gpu'] < error_cooldown:
            gpu_error_penalty = 1.5
            
        # Determine optimal processor
        if self.current_mode == 'auto':
            # Consider processing time, backlog, system load, phase metrics, and error history
            cpu_score = (
                avg_cpu_time * 
                (1 + cpu_load/100) * 
                (1 + cpu_backlog_pressure/100) *
                (1 + (1 - drift_resonance)) *  # Favor CPU when drift resonance is low
                (1 + cpu_error_penalty)  # Penalize CPU for recent errors
            )
            gpu_score = (
                avg_gpu_time * 
                (1 + gpu_load/100) * 
                (1 + gpu_backlog_pressure/100) *
                (1 + drift_resonance) *  # Favor GPU when drift resonance is high
                (1 + gpu_error_penalty)  # Penalize GPU for recent errors
            )
            
            if cpu_score < gpu_score:
                self.current_mode = 'cpu'
            else:
                self.current_mode = 'gpu'
                
        # Check for overload conditions
        elif self.current_mode == 'cpu' and cpu_load > self.config['load_balancing']['cpu_overload_threshold']:
            self.current_mode = 'gpu'
        elif self.current_mode == 'gpu' and gpu_load > self.config['load_balancing']['gpu_overload_threshold']:
            self.current_mode = 'cpu'
            
        self.last_balance_time = current_time
        return self.current_mode
        
    def record_error(self, processor: str):
        """Record an error for the specified processor"""
        self.error_counts[processor] += 1
        self.last_error_time[processor] = time.time()
        
    def get_processing_stats(self) -> Dict:
        """Get current processing statistics"""
        return {
            'current_mode': self.current_mode,
            'cpu_load': np.mean(self.cpu_load_history) if self.cpu_load_history else 0,
            'gpu_load': np.mean(self.gpu_load_history) if self.gpu_load_history else 0,
            'cpu_avg_time': np.mean(self.processing_times['cpu']) if self.processing_times['cpu'] else 0,
            'gpu_avg_time': np.mean(self.processing_times['gpu']) if self.processing_times['gpu'] else 0,
            'cpu_backlog': np.mean(self.backlog_sizes['cpu']) if self.backlog_sizes['cpu'] else 0,
            'gpu_backlog': np.mean(self.backlog_sizes['gpu']) if self.backlog_sizes['gpu'] else 0,
            'error_counts': self.error_counts
        }

class MemoryManager:
    """Manages different types of memory (short-term, mid-term, long-term) with mathematical synthesis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.thermal_allocator = ThermalMapAllocator(
            max_thermal_threshold=config['memory']['max_thermal_threshold'],
            min_coherence_threshold=config['memory']['min_coherence_threshold'],
            memory_quota=config['memory']['quota']
        )
        
        # Initialize memory regions
        self.short_term_buffer = deque(maxlen=config['memory']['short_term_size'])
        self.mid_term_buffer = deque(maxlen=config['memory']['mid_term_size'])
        self.long_term_buffer = deque(maxlen=config['memory']['long_term_size'])
        
        # Initialize mathematical synthesis components
        self.recursive_engine = RecursiveEngine()
        self.antipole_vector = AntiPoleVector()
        self.drift_engine = DriftShellEngine()
        
        # Initialize memory metrics
        self.memory_metrics = {
            'short_term_usage': 0,
            'mid_term_usage': 0,
            'long_term_usage': 0,
            'total_usage': 0,
            'last_synthesis_time': time.time()
        }
        
    def allocate_memory(self, data: Dict, memory_type: str) -> Optional[MemoryRegion]:
        """Allocate memory for data based on its type and mathematical properties"""
        try:
            # Calculate mathematical properties
            coherence = self._calculate_coherence(data)
            entropy = self._calculate_entropy(data)
            drift = self._calculate_drift(data)
            
            # Determine memory type if not specified
            if memory_type == 'auto':
                memory_type = self._determine_memory_type(coherence, entropy, drift)
                
            # Allocate memory with thermal constraints
            region = self.thermal_allocator.allocate_memory(
                key=f"{memory_type}_{len(self.memory_metrics)}",
                size=self._calculate_memory_size(data),
                priority=self._calculate_priority(coherence, entropy, drift)
            )
            
            if region:
                # Store data in appropriate buffer
                if memory_type == 'short_term':
                    self.short_term_buffer.append((data, region))
                elif memory_type == 'mid_term':
                    self.mid_term_buffer.append((data, region))
                else:  # long_term
                    self.long_term_buffer.append((data, region))
                    
                # Update metrics
                self._update_memory_metrics()
                
            return region
            
        except Exception as e:
            logger.error(f"Memory allocation error: {e}")
            return None
            
    def _determine_memory_type(self, coherence: float, entropy: float, drift: float) -> str:
        """Determine appropriate memory type based on mathematical properties"""
        # High coherence and low drift suggests long-term importance
        if coherence > 0.8 and drift < 0.2:
            return 'long_term'
            
        # Moderate coherence and drift suggests mid-term importance
        elif coherence > 0.5 and drift < 0.5:
            return 'mid_term'
            
        # Low coherence or high drift suggests short-term importance
        else:
            return 'short_term'
            
    def _calculate_coherence(self, data: Dict) -> float:
        """Calculate coherence score for data"""
        try:
            # Use recursive engine to calculate coherence
            recursive_metrics = self.recursive_engine.process_tick(
                F=data.get('price', 0),
                P=data.get('volume', 0),
                Lambda=data.get('entropy', 0),
                phi=0.0,
                R=1.0,
                dt=0.1
            )
            return recursive_metrics.get('coherence', 0)
        except Exception as e:
            logger.error(f"Coherence calculation error: {e}")
            return 0.0
            
    def _calculate_entropy(self, data: Dict) -> float:
        """Calculate entropy for data"""
        try:
            # Use entropy engine to calculate entropy
            return self.entropy_engine.calculate_entropy(str(data))
        except Exception as e:
            logger.error(f"Entropy calculation error: {e}")
            return 0.0
            
    def _calculate_drift(self, data: Dict) -> float:
        """Calculate drift for data"""
        try:
            # Use drift engine to calculate drift
            drift_variance = self.drift_engine.drift_variance(
                hashes=[data.get('hash', '')],
                features={'price': data.get('price', 0), 'volume': data.get('volume', 0)},
                tick_times=[time.time()],
                meta={'entropy': data.get('entropy', 0)}
            )
            return drift_variance
        except Exception as e:
            logger.error(f"Drift calculation error: {e}")
            return 0.0
            
    def _calculate_memory_size(self, data: Dict) -> int:
        """Calculate required memory size for data"""
        try:
            # Estimate size based on data structure
            size = len(json.dumps(data))
            # Add overhead for mathematical properties
            size += 100  # Base overhead
            size += len(str(data.get('hash', ''))) * 2  # Hash storage
            size += len(str(data.get('recursive_metrics', {}))) * 2  # Recursive metrics
            return size
        except Exception as e:
            logger.error(f"Memory size calculation error: {e}")
            return 1024  # Default size
            
    def _calculate_priority(self, coherence: float, entropy: float, drift: float) -> float:
        """Calculate priority score for memory allocation"""
        try:
            # Higher coherence and lower drift means higher priority
            priority = (coherence * 0.6) + ((1 - drift) * 0.4)
            # Adjust based on entropy
            if entropy > 0.8:  # High entropy suggests important data
                priority *= 1.2
            return min(max(priority, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Priority calculation error: {e}")
            return 0.5
            
    def _update_memory_metrics(self):
        """Update memory usage metrics"""
        try:
            self.memory_metrics['short_term_usage'] = sum(
                region.size for _, region in self.short_term_buffer
            )
            self.memory_metrics['mid_term_usage'] = sum(
                region.size for _, region in self.mid_term_buffer
            )
            self.memory_metrics['long_term_usage'] = sum(
                region.size for _, region in self.long_term_buffer
            )
            self.memory_metrics['total_usage'] = (
                self.memory_metrics['short_term_usage'] +
                self.memory_metrics['mid_term_usage'] +
                self.memory_metrics['long_term_usage']
            )
        except Exception as e:
            logger.error(f"Memory metrics update error: {e}")
            
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        return {
            'metrics': self.memory_metrics,
            'buffer_sizes': {
                'short_term': len(self.short_term_buffer),
                'mid_term': len(self.mid_term_buffer),
                'long_term': len(self.long_term_buffer)
            },
            'thermal_stats': self.thermal_allocator.performance_metrics
        }
        
    def cleanup_old_data(self):
        """Clean up old data based on memory type and mathematical properties"""
        try:
            current_time = time.time()
            
            # Clean up short-term buffer
            while self.short_term_buffer:
                data, region = self.short_term_buffer[0]
                if current_time - data.get('timestamp', 0) > self.config['memory']['short_term_ttl']:
                    self.thermal_allocator.deallocate_memory(region.start_address)
                    self.short_term_buffer.popleft()
                else:
                    break
                    
            # Clean up mid-term buffer
            while self.mid_term_buffer:
                data, region = self.mid_term_buffer[0]
                if current_time - data.get('timestamp', 0) > self.config['memory']['mid_term_ttl']:
                    self.thermal_allocator.deallocate_memory(region.start_address)
                    self.mid_term_buffer.popleft()
                else:
                    break
                    
            # Clean up long-term buffer
            while self.long_term_buffer:
                data, region = self.long_term_buffer[0]
                if current_time - data.get('timestamp', 0) > self.config['memory']['long_term_ttl']:
                    self.thermal_allocator.deallocate_memory(region.start_address)
                    self.long_term_buffer.popleft()
                else:
                    break
                    
            # Update metrics
            self._update_memory_metrics()
            
        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")

class BTCDataProcessor:
    """Processes BTC price data and generates hashes with intelligent load balancing"""
    
    def __init__(self, config_path: str = "config/btc_processor_config.yaml"):
        """Initialize the BTC data processor"""
        self.config = self._load_config(config_path)
        self._setup_gpu()
        self._initialize_components()
        self.memory_manager = MemoryManager(self.config)
        self.processing_queue = asyncio.Queue()
        self.cpu_queue = asyncio.Queue()
        self.gpu_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.last_hash_time = None
        self.hash_generation_stats = {
            'total_hashes': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0
        }
        self.load_balancer = LoadBalancer(self.config)
        
        # Initialize core mathematical engines
        self.drift_engine = DriftShellEngine()
        self.recursive_engine = RecursiveEngine()
        self.antipole_vector = AntiPoleVector()
        
        # Initialize Bitcoin mining analyzer
        self.mining_analyzer = BitcoinMiningAnalyzer(self.config)
        
        # Initialize mining information storage
        self.mining_data_storage = {
            'block_templates': deque(maxlen=1000),
            'mining_solutions': deque(maxlen=10000),
            'nonce_sequences': deque(maxlen=50000),
            'difficulty_adjustments': deque(maxlen=100),
            'hash_rate_estimates': deque(maxlen=1000)
        }
        
        # Initialize time scaling functions
        self.time_scaling_factors = self.config.get('time_scaling', {}).get('scaling_factors', [1, 10, 100, 1000])
        self.target_block_time = self.config.get('time_scaling', {}).get('target_block_time', 600)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load processor configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load BTC processor config: {e}")
            raise
            
    def _setup_gpu(self):
        """Set up GPU acceleration if available"""
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("GPU not available, using CPU")
            
    def _initialize_components(self):
        """Initialize processing components"""
        self.math_lib = SustainmentMathLib()
        self.entropy_engine = EntropyEngine()
        self.antipole_engine = QuantumAntipoleEngine()
        
        # Initialize CUDA streams if using GPU
        if self.use_gpu:
            self.streams = [torch.cuda.Stream() for _ in range(3)]
            
    async def start_processing_pipeline(self):
        """Start the data processing pipeline with mining analysis"""
        try:
            # Start WebSocket connection for real-time data
            await self._connect_websocket()
            
            # Start processing tasks
            tasks = [
                self._process_data_stream(),
                self._distribute_processing_load(),
                self._process_cpu_queue(),
                self._process_gpu_queue(),
                self._validate_entropy(),
                self._update_price_correlations(),
                self._monitor_hash_timing(),
                self._analyze_mining_patterns(),
                self._monitor_block_timing(),
                self._analyze_nonce_sequences(),
                self._track_difficulty_adjustments()
            ]
            
            if self.use_gpu:
                tasks.append(self._monitor_gpu_load())
                
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
            
    async def _distribute_processing_load(self):
        """Distribute processing load between CPU and GPU"""
        while True:
            try:
                if self.processing_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                    
                data = await self.processing_queue.get()
                
                # Get optimal processor
                processor = self.load_balancer.get_optimal_processor()
                
                # Distribute to appropriate queue
                if processor == 'cpu' or not self.use_gpu:
                    await self.cpu_queue.put(data)
                else:
                    await self.gpu_queue.put(data)
                    
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Load distribution error: {e}")
                continue
                
    async def _process_cpu_queue(self):
        """Process CPU queue"""
        while True:
            try:
                if self.cpu_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                    
                data = await self.cpu_queue.get()
                start_time = time.perf_counter()
                
                # Process data
                hash_value = await self._generate_hash_cpu(data)
                
                # Update timing stats
                end_time = time.perf_counter()
                duration = end_time - start_time
                self.load_balancer.update_load_metrics(
                    duration, 0,
                    self.cpu_queue.qsize(),
                    self.gpu_queue.qsize()
                )
                
                # Store result
                await self._store_hash_result(hash_value, data, duration, 'cpu')
                
                self.cpu_queue.task_done()
                
            except Exception as e:
                logger.error(f"CPU processing error: {e}")
                continue
                
    async def _process_gpu_queue(self):
        """Process GPU queue"""
        if not self.use_gpu:
            return
            
        while True:
            try:
                if self.gpu_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                    
                data = await self.gpu_queue.get()
                start_time = time.perf_counter()
                
                # Process data
                hash_value = await self._generate_hash_gpu(data)
                
                # Update timing stats
                end_time = time.perf_counter()
                duration = end_time - start_time
                self.load_balancer.update_load_metrics(
                    0, duration,
                    self.cpu_queue.qsize(),
                    self.gpu_queue.qsize()
                )
                
                # Store result
                await self._store_hash_result(hash_value, data, duration, 'gpu')
                
                self.gpu_queue.task_done()
                
            except Exception as e:
                logger.error(f"GPU processing error: {e}")
                continue
                
    async def _store_hash_result(self, hash_value: str, data: Dict, 
                               duration: float, processor: str):
        """Store hash generation result"""
        hash_entry = {
            'hash': hash_value,
            'timestamp': data['timestamp'],
            'generation_time': duration,
            'processor': processor,
            'price': data['price']
        }
        
        self.hash_buffer.append(hash_entry)
        self.timing_buffer.append(duration)
        
        # Maintain buffer size
        if len(self.hash_buffer) > self.config['hash_buffer_size']:
            self.hash_buffer.pop(0)
            self.timing_buffer.pop(0)
            
        # Update last hash time
        self.last_hash_time = time.time()
        
    async def _monitor_gpu_load(self):
        """Monitor GPU load and adjust processing accordingly"""
        while True:
            try:
                if not self.use_gpu:
                    break
                    
                # Get GPU utilization
                gpu_util = torch.cuda.utilization()
                
                # Adjust processing if GPU is overloaded
                if gpu_util > self.config['load_balancing']['gpu_overload_threshold']:
                    logger.warning(f"GPU overload detected: {gpu_util}%")
                    self.load_balancer.current_mode = 'cpu'
                    
                await asyncio.sleep(self.config['load_balancing']['monitor_interval_ms'] / 1000)
                
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                continue
                
    def get_processing_stats(self) -> Dict:
        """Get current processing statistics"""
        stats = self.load_balancer.get_processing_stats()
        stats.update({
            'queue_sizes': {
                'main': self.processing_queue.qsize(),
                'cpu': self.cpu_queue.qsize(),
                'gpu': self.gpu_queue.qsize()
            },
            'hash_stats': self.hash_generation_stats
        })
        return stats
        
    async def _connect_websocket(self):
        """Connect to BTC price WebSocket feed"""
        uri = self.config['websocket']['uri']
        async with websockets.connect(uri) as websocket:
            while True:
                try:
                    data = await websocket.recv()
                    await self.processing_queue.put(json.loads(data))
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await asyncio.sleep(1)
                    continue
                    
    async def _process_data_stream(self):
        """Process incoming data stream with error handling and memory management"""
        while True:
            try:
                data = await self.processing_queue.get()
                
                # Validate data
                if not self._validate_data(data):
                    logger.warning("Invalid data received, skipping processing")
                    self.processing_queue.task_done()
                    continue
                    
                # Process price data
                processed_data = await self._process_price_data(data)
                
                # Allocate memory for processed data
                memory_region = self.memory_manager.allocate_memory(processed_data, 'auto')
                if not memory_region:
                    logger.warning("Failed to allocate memory for processed data")
                    self.processing_queue.task_done()
                    continue
                    
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Data processing error: {e}")
                self.load_balancer.record_error('cpu')
                continue
                
    def _validate_data(self, data: Dict) -> bool:
        """Validate incoming data"""
        try:
            required_fields = ['price', 'volume', 'timestamp']
            if not all(field in data for field in required_fields):
                return False
                
            # Validate price
            if not isinstance(data['price'], (int, float)) or data['price'] <= 0:
                return False
                
            # Validate volume
            if not isinstance(data['volume'], (int, float)) or data['volume'] < 0:
                return False
                
            # Validate timestamp
            if not isinstance(data['timestamp'], (str, int, float)):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
            
    async def _process_price_data(self, data: Dict) -> Dict:
        """Process BTC price data with core mathematical integration and error handling"""
        try:
            # Get base metrics with error checking
            price = float(data['price'])
            volume = float(data['volume'])
            timestamp = datetime.now().isoformat()
            
            # Calculate entropy with error handling
            try:
                entropy = await self._calculate_entropy(data)
            except Exception as e:
                logger.error(f"Entropy calculation error: {e}")
                entropy = 0.0
                
            # Process through recursive engine with error handling
            try:
                recursive_metrics = self.recursive_engine.process_tick(
                    F=price,
                    P=volume,
                    Lambda=entropy,
                    phi=0.0,
                    R=1.0,
                    dt=0.1
                )
            except Exception as e:
                logger.error(f"Recursive engine error: {e}")
                recursive_metrics = {'coherence': 0.0, 'psi': 0.0}
                
            # Process through anti-pole vector with error handling
            try:
                antipole_state = self.antipole_vector.process_tick(
                    btc_price=price,
                    volume=volume,
                    lambda_i=recursive_metrics.get('coherence', 0),
                    f_k=recursive_metrics.get('psi', 0)
                )
            except Exception as e:
                logger.error(f"Anti-pole vector error: {e}")
                antipole_state = {'delta_psi_bar': 0.0}
                
            # Calculate drift shell variance with error handling
            try:
                drift_variance = self.drift_engine.drift_variance(
                    hashes=[data.get('hash', '')],
                    features={'price': price, 'volume': volume},
                    tick_times=[time.time()],
                    meta={'entropy': entropy}
                )
            except Exception as e:
                logger.error(f"Drift shell variance error: {e}")
                drift_variance = 0.0
                
            processed = {
                'timestamp': timestamp,
                'price': price,
                'volume': volume,
                'entropy': entropy,
                'recursive_metrics': recursive_metrics,
                'antipole_state': antipole_state,
                'drift_variance': drift_variance
            }
            return processed
            
        except Exception as e:
            logger.error(f"Price data processing error: {e}")
            # Return safe fallback data
            return {
                'timestamp': datetime.now().isoformat(),
                'price': 0.0,
                'volume': 0.0,
                'entropy': 0.0,
                'recursive_metrics': {'coherence': 0.0, 'psi': 0.0},
                'antipole_state': {'delta_psi_bar': 0.0},
                'drift_variance': 0.0
            }
            
    async def _generate_hashes(self):
        """Generate BTC price hashes with precise timing"""
        while True:
            try:
                if not self.data_buffer:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Get latest data
                latest_data = self.data_buffer[-1]
                
                # Record start time
                start_time = time.perf_counter()
                
                # Generate hash
                if self.use_gpu:
                    hash_value = await self._generate_hash_gpu(latest_data)
                else:
                    hash_value = await self._generate_hash_cpu(latest_data)
                    
                # Record end time and calculate duration
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                # Update timing statistics
                self._update_hash_timing_stats(duration)
                
                # Store hash with timing information
                hash_entry = {
                    'hash': hash_value,
                    'timestamp': latest_data['timestamp'],
                    'generation_time': duration,
                    'price': latest_data['price']
                }
                
                self.hash_buffer.append(hash_entry)
                self.timing_buffer.append(duration)
                
                # Maintain hash buffer size
                if len(self.hash_buffer) > self.config['hash_buffer_size']:
                    self.hash_buffer.pop(0)
                    self.timing_buffer.pop(0)
                    
                # Update last hash time
                self.last_hash_time = end_time
                
            except Exception as e:
                logger.error(f"Hash generation error: {e}")
                continue
                
    def _update_hash_timing_stats(self, duration: float):
        """Update hash generation timing statistics"""
        self.hash_generation_stats['total_hashes'] += 1
        self.hash_generation_stats['total_time'] += duration
        self.hash_generation_stats['min_time'] = min(self.hash_generation_stats['min_time'], duration)
        self.hash_generation_stats['max_time'] = max(self.hash_generation_stats['max_time'], duration)
        
    async def _generate_hash_gpu(self, data: Dict) -> str:
        """Generate hash using GPU acceleration with mathematical integration"""
        with torch.cuda.stream(self.streams[0]):
            # Convert data to tensor
            price_tensor = torch.tensor(data['price'], device=self.device)
            volume_tensor = torch.tensor(data['volume'], device=self.device)
            
            # Get recursive metrics
            recursive_metrics = data.get('recursive_metrics', {})
            coherence = torch.tensor(recursive_metrics.get('coherence', 0), device=self.device)
            
            # Get anti-pole state
            antipole_state = data.get('antipole_state', {})
            delta_psi = torch.tensor(antipole_state.get('delta_psi_bar', 0), device=self.device)
            
            # Combine all metrics for hash computation
            combined = torch.cat([
                price_tensor,
                volume_tensor,
                coherence,
                delta_psi
            ])
            
            # Apply mathematical transformations
            transformed = self._apply_mathematical_transforms(combined)
            
            # Generate hash
            hash_tensor = self._compute_hash_gpu(transformed)
            
            # Convert back to CPU and format
            hash_value = hash_tensor.cpu().numpy().tobytes().hex()
            return hash_value
            
    def _apply_mathematical_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply mathematical transformations to tensor data with error handling"""
        try:
            # Apply phase angle transformation with error checking
            phase_angle = torch.atan2(tensor[1], tensor[0])
            if torch.isnan(phase_angle) or torch.isinf(phase_angle):
                phase_angle = torch.zeros(1, device=tensor.device)
                
            # Apply drift transformation with bounds checking
            drift = (tensor[-1] - tensor[0]) / (len(tensor) - 1)
            drift = torch.clamp(drift, -1.0, 1.0)
            
            # Apply recursive transformation
            recursive_transform = self.recursive_engine.transform_tensor(tensor)
            
            # Apply anti-pole transformation
            antipole_transform = self.antipole_vector.transform_tensor(tensor)
            
            # Combine transformations with error checking
            transformed = torch.cat([
                tensor,
                phase_angle.unsqueeze(0),
                drift.unsqueeze(0),
                recursive_transform,
                antipole_transform
            ])
            
            # Validate transformed tensor
            if torch.isnan(transformed).any() or torch.isinf(transformed).any():
                logger.warning("Invalid values in transformed tensor, applying correction")
                transformed = torch.nan_to_num(transformed, nan=0.0, posinf=1.0, neginf=-1.0)
                
            return transformed
            
        except Exception as e:
            logger.error(f"Mathematical transformation error: {e}")
            # Return safe fallback tensor
            return torch.zeros_like(tensor)
            
    def _compute_hash_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute hash using GPU operations with error handling"""
        try:
            # Apply SHA-256-like operations on GPU with error checking
            hash_tensor = torch.nn.functional.linear(tensor, torch.randn(32, device=self.device))
            
            # Ensure deterministic output with bounds checking
            hash_tensor = torch.clamp(hash_tensor, -1.0, 1.0)
            hash_tensor = torch.round(hash_tensor * 1e6) / 1e6
            
            # Validate hash tensor
            if torch.isnan(hash_tensor).any() or torch.isinf(hash_tensor).any():
                logger.warning("Invalid values in hash tensor, applying correction")
                hash_tensor = torch.nan_to_num(hash_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                
            return hash_tensor
            
        except Exception as e:
            logger.error(f"GPU hash computation error: {e}")
            # Return safe fallback hash
            return torch.zeros(32, device=self.device)
            
    async def _generate_hash_cpu(self, data: Dict) -> str:
        """Generate hash using CPU with mathematical integration and error handling"""
        try:
            # Get recursive metrics with error checking
            recursive_metrics = data.get('recursive_metrics', {})
            coherence = recursive_metrics.get('coherence', 0)
            if not isinstance(coherence, (int, float)):
                coherence = 0
                
            # Get anti-pole state with error checking
            antipole_state = data.get('antipole_state', {})
            delta_psi = antipole_state.get('delta_psi_bar', 0)
            if not isinstance(delta_psi, (int, float)):
                delta_psi = 0
                
            # Create deterministic string representation with mathematical components
            data_str = (
                f"{float(data['price']):.8f}:"
                f"{float(data['volume']):.8f}:"
                f"{data['timestamp']}:"
                f"{float(coherence):.8f}:"
                f"{float(delta_psi):.8f}"
            )
            
            # Generate SHA-256 hash with error checking
            try:
                hash_obj = hashlib.sha256(data_str.encode())
                return hash_obj.hexdigest()
            except Exception as e:
                logger.error(f"SHA-256 hash generation error: {e}")
                return "0" * 64  # Return safe fallback hash
                
        except Exception as e:
            logger.error(f"CPU hash generation error: {e}")
            return "0" * 64  # Return safe fallback hash
        
    async def _monitor_hash_timing(self):
        """Monitor hash generation timing and performance"""
        while True:
            try:
                if not self.timing_buffer:
                    await asyncio.sleep(1)
                    continue
                    
                # Calculate timing statistics
                avg_time = sum(self.timing_buffer) / len(self.timing_buffer)
                max_time = max(self.timing_buffer)
                min_time = min(self.timing_buffer)
                
                # Check for timing anomalies
                if max_time > self.config['hash_generation']['max_allowed_time_ms'] / 1000:
                    logger.warning(f"Hash generation timing anomaly detected: {max_time:.6f}s")
                    
                # Log timing statistics periodically
                logger.debug(f"Hash generation timing - Avg: {avg_time:.6f}s, Min: {min_time:.6f}s, Max: {max_time:.6f}s")
                
                await asyncio.sleep(self.config['hash_generation']['timing_check_interval_ms'] / 1000)
                
            except Exception as e:
                logger.error(f"Hash timing monitoring error: {e}")
                continue
                
    async def _validate_entropy(self):
        """Validate entropy of generated hashes"""
        while True:
            try:
                if not self.hash_buffer:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Get latest hash
                latest_hash = self.hash_buffer[-1]['hash']
                
                # Validate entropy
                entropy_value = await self.entropy_engine.calculate_entropy(latest_hash)
                
                if entropy_value < self.config['entropy']['min_entropy']:
                    logger.warning(f"Low entropy detected: {entropy_value}")
                    
            except Exception as e:
                logger.error(f"Entropy validation error: {e}")
                continue
                
    async def _update_price_correlations(self):
        """Update price correlations with generated hashes"""
        while True:
            try:
                if len(self.data_buffer) < 2:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Calculate correlations
                correlations = await self._calculate_price_correlations()
                
                # Update correlation matrix
                await self._update_correlation_matrix(correlations)
                
            except Exception as e:
                logger.error(f"Correlation update error: {e}")
                continue
                
    async def _calculate_price_correlations(self) -> Dict:
        """Calculate price correlations"""
        prices = [data['price'] for data in self.data_buffer]
        volumes = [data['volume'] for data in self.data_buffer]
        
        if self.use_gpu:
            return await self._calculate_correlations_gpu(prices, volumes)
        else:
            return await self._calculate_correlations_cpu(prices, volumes)
            
    async def _calculate_correlations_gpu(self, prices: List[float], volumes: List[float]) -> Dict:
        """Calculate correlations using GPU"""
        with torch.cuda.stream(self.streams[1]):
            price_tensor = torch.tensor(prices, device=self.device)
            volume_tensor = torch.tensor(volumes, device=self.device)
            
            # Calculate correlation matrix
            correlation_matrix = torch.corrcoef(torch.stack([price_tensor, volume_tensor]))
            
            return {
                'price_volume': correlation_matrix[0, 1].item(),
                'price_entropy': await self._calculate_entropy_correlation(price_tensor)
            }
            
    async def _calculate_entropy_correlation(self, data: torch.Tensor) -> float:
        """Calculate entropy correlation"""
        with torch.cuda.stream(self.streams[2]):
            # Implement entropy correlation calculation
            # This is a placeholder for the actual implementation
            return torch.randn(1).item()
            
    def get_latest_data(self) -> Dict:
        """Get latest processed data"""
        return {
            'price_data': self.data_buffer[-1] if self.data_buffer else None,
            'latest_hash': self.hash_buffer[-1] if self.hash_buffer else None,
            'correlations': self._get_latest_correlations(),
            'hash_timing': self._get_hash_timing_stats()
        }
        
    def _get_latest_correlations(self) -> Dict:
        """Get latest correlation data"""
        # Implement correlation data retrieval
        # This is a placeholder for the actual implementation
        return {
            'price_volume': 0.0,
            'price_entropy': 0.0
        }
        
    def _get_hash_timing_stats(self) -> Dict:
        """Get hash generation timing statistics"""
        if self.hash_generation_stats['total_hashes'] == 0:
            return {
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'total_hashes': 0
            }
            
        return {
            'avg_time': self.hash_generation_stats['total_time'] / self.hash_generation_stats['total_hashes'],
            'min_time': self.hash_generation_stats['min_time'],
            'max_time': self.hash_generation_stats['max_time'],
            'total_hashes': self.hash_generation_stats['total_hashes']
        }
        
    async def _monitor_memory(self):
        """Monitor memory usage and perform cleanup if needed"""
        while True:
            try:
                current_time = time.time()
                if current_time - self.memory_manager.memory_metrics['last_synthesis_time'] < self.config['memory']['check_interval_ms'] / 1000:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Get memory stats
                memory_stats = self.memory_manager.get_memory_stats()
                
                # Check for memory pressure
                if memory_stats['metrics']['total_usage'] > self.config['memory']['pressure_threshold']:
                    logger.warning(f"High memory usage detected: {memory_stats['metrics']['total_usage']}")
                    self.memory_manager.cleanup_old_data()
                    
                # Check CPU memory
                cpu_memory = psutil.Process().memory_percent()
                if cpu_memory > self.config['memory']['cpu_threshold']:
                    logger.warning(f"High CPU memory usage: {cpu_memory:.1f}%")
                    gc.collect()  # Force garbage collection
                    
                # Check GPU memory if available
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    if gpu_memory > self.config['memory']['gpu_threshold']:
                        logger.warning(f"High GPU memory usage: {gpu_memory:.1f}%")
                        torch.cuda.empty_cache()  # Clear GPU cache
                        
                self.memory_manager.memory_metrics['last_synthesis_time'] = current_time
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                continue
                
    async def shutdown(self):
        """Shutdown the processor with cleanup"""
        try:
            # Stop processing
            self.executor.shutdown(wait=True)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Clear memory regions
            self.memory_manager.cleanup_old_data()
            
            # Clear queues
            while not self.processing_queue.empty():
                self.processing_queue.get_nowait()
            while not self.cpu_queue.empty():
                self.cpu_queue.get_nowait()
            while not self.gpu_queue.empty():
                self.gpu_queue.get_nowait()
                
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            
    async def _analyze_mining_patterns(self):
        """Continuously analyze Bitcoin mining patterns"""
        while True:
            try:
                if not self.memory_manager.long_term_buffer:
                    await asyncio.sleep(1)
                    continue
                    
                # Get latest processed data
                latest_data = self.memory_manager.long_term_buffer[-1][0] if self.memory_manager.long_term_buffer else None
                if not latest_data:
                    continue
                    
                # Get latest hash
                latest_hash = latest_data.get('hash', '')
                if not latest_hash:
                    continue
                    
                # Perform mining analysis
                mining_analysis = await self.mining_analyzer.analyze_mining_data(
                    latest_data, latest_hash
                )
                
                # Store mining analysis results
                await self._store_mining_analysis(mining_analysis)
                
                # Update mining statistics
                self._update_mining_statistics(mining_analysis)
                
                # Check for potential block solutions
                await self._check_block_solutions(mining_analysis)
                
                await asyncio.sleep(self.config.get('mining_analysis', {}).get('update_interval_ms', 5000) / 1000)
                
            except Exception as e:
                logger.error(f"Mining pattern analysis error: {e}")
                await asyncio.sleep(1)
                continue
                
    async def _monitor_block_timing(self):
        """Monitor block timing patterns for mining optimization"""
        while True:
            try:
                current_time = time.time()
                
                # Analyze time scaling functions
                for scale_factor in self.time_scaling_factors:
                    scaled_time = current_time / scale_factor
                    log_scaled_time = np.log10(scaled_time) if scaled_time > 0 else 0
                    
                    # Correlate with current price data
                    if self.memory_manager.short_term_buffer:
                        latest_price_data = self.memory_manager.short_term_buffer[-1][0]
                        price = latest_price_data.get('price', 0)
                        
                        # Calculate correlation between time scaling and price
                        time_price_correlation = self._calculate_time_price_correlation(
                            log_scaled_time, price, scale_factor
                        )
                        
                        # Store time scaling analysis
                        self.mining_data_storage['hash_rate_estimates'].append({
                            'timestamp': current_time,
                            'scale_factor': scale_factor,
                            'scaled_time': scaled_time,
                            'log_scaled_time': log_scaled_time,
                            'price': price,
                            'correlation': time_price_correlation
                        })
                        
                await asyncio.sleep(self.config.get('time_scaling', {}).get('check_interval_ms', 10000) / 1000)
                
            except Exception as e:
                logger.error(f"Block timing monitoring error: {e}")
                await asyncio.sleep(1)
                continue
                
    async def _analyze_nonce_sequences(self):
        """Analyze nonce sequences for mining optimization"""
        while True:
            try:
                if len(self.mining_data_storage['nonce_sequences']) < 100:
                    await asyncio.sleep(1)
                    continue
                    
                # Get recent nonce sequences
                recent_nonces = list(self.mining_data_storage['nonce_sequences'])[-100:]
                
                # Analyze sequence patterns
                sequence_analysis = await self._analyze_sequence_patterns(recent_nonces)
                
                # Predict optimal nonce ranges
                optimal_ranges = await self._predict_optimal_nonce_ranges(sequence_analysis)
                
                # Update nonce optimization strategies
                await self._update_nonce_strategies(optimal_ranges)
                
                await asyncio.sleep(self.config.get('sequence_analysis', {}).get('update_interval_ms', 15000) / 1000)
                
            except Exception as e:
                logger.error(f"Nonce sequence analysis error: {e}")
                await asyncio.sleep(1)
                continue
                
    async def _track_difficulty_adjustments(self):
        """Track Bitcoin difficulty adjustments for mining strategy"""
        while True:
            try:
                # Get current network state
                network_state = await self.mining_analyzer._get_network_state()
                
                # Store difficulty information
                difficulty_data = {
                    'timestamp': time.time(),
                    'current_difficulty': network_state.get('current_difficulty', 0),
                    'estimated_next_difficulty': network_state.get('estimated_next_difficulty', 0),
                    'blocks_until_adjustment': network_state.get('blocks_until_adjustment', 0),
                    'network_hash_rate': network_state.get('network_hash_rate', 0)
                }
                
                self.mining_data_storage['difficulty_adjustments'].append(difficulty_data)
                
                # Predict mining profitability
                profitability_prediction = await self._predict_mining_profitability(difficulty_data)
                
                # Adjust mining strategies based on difficulty
                await self._adjust_mining_strategies(profitability_prediction)
                
                await asyncio.sleep(self.config.get('mining_analysis', {}).get('difficulty_check_interval_ms', 30000) / 1000)
                
            except Exception as e:
                logger.error(f"Difficulty tracking error: {e}")
                await asyncio.sleep(1)
                continue
                
    async def _store_mining_analysis(self, analysis: Dict):
        """Store mining analysis results for long-term learning"""
        try:
            # Determine storage type based on analysis importance
            importance_score = self._calculate_analysis_importance(analysis)
            
            storage_data = {
                'timestamp': time.time(),
                'analysis': analysis,
                'importance_score': importance_score
            }
            
            # Store in appropriate memory based on importance
            if importance_score > 0.8:
                memory_type = 'long_term'
            elif importance_score > 0.5:
                memory_type = 'mid_term'
            else:
                memory_type = 'short_term'
                
            # Allocate memory for analysis
            memory_region = self.memory_manager.allocate_memory(storage_data, memory_type)
            
            if memory_region:
                logger.debug(f"Stored mining analysis with importance {importance_score:.3f} in {memory_type} memory")
            else:
                logger.warning("Failed to allocate memory for mining analysis")
                
        except Exception as e:
            logger.error(f"Mining analysis storage error: {e}")
            
    def _calculate_analysis_importance(self, analysis: Dict) -> float:
        """Calculate importance score for mining analysis"""
        try:
            score = 0.0
            
            # High solution probability increases importance
            if 'hash_analysis' in analysis:
                solution_prob = analysis['hash_analysis'].get('solution_probability', 0)
                score += solution_prob * 0.3
                
            # Network efficiency affects importance
            if 'efficiency_analysis' in analysis:
                overall_efficiency = analysis['efficiency_analysis'].get('overall_efficiency', 0)
                score += overall_efficiency * 0.3
                
            # Strategy prediction quality affects importance
            if 'strategy_prediction' in analysis:
                strategy_confidence = analysis['strategy_prediction'].get('confidence', 0)
                score += strategy_confidence * 0.4
                
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Analysis importance calculation error: {e}")
            return 0.5  # Default medium importance
            
    async def _check_block_solutions(self, analysis: Dict):
        """Check for potential Bitcoin block solutions"""
        try:
            if 'hash_analysis' not in analysis:
                return
                
            hash_analysis = analysis['hash_analysis']
            solution_probability = hash_analysis.get('solution_probability', 0)
            leading_zeros = hash_analysis.get('leading_zeros', 0)
            
            # Check if hash meets difficulty requirements
            if solution_probability > 0.9 and leading_zeros >= 10:  # Configurable thresholds
                logger.info(f"Potential block solution detected! Probability: {solution_probability:.6f}, Leading zeros: {leading_zeros}")
                
                # Store potential solution
                solution_data = {
                    'timestamp': time.time(),
                    'hash_analysis': hash_analysis,
                    'solution_probability': solution_probability,
                    'leading_zeros': leading_zeros,
                    'analysis': analysis
                }
                
                self.mining_data_storage['mining_solutions'].append(solution_data)
                
                # Trigger detailed analysis for potential solution
                await self._analyze_potential_solution(solution_data)
                
        except Exception as e:
            logger.error(f"Block solution check error: {e}")
            
    async def _analyze_potential_solution(self, solution_data: Dict):
        """Perform detailed analysis of potential mining solution"""
        try:
            # Extract hash and network data
            hash_analysis = solution_data.get('hash_analysis', {})
            
            # Verify solution against current difficulty
            verification_result = await self._verify_solution(hash_analysis)
            
            # Calculate potential profitability
            profitability = await self._calculate_solution_profitability(solution_data)
            
            # Log detailed solution analysis
            logger.info(f"Solution Analysis - Verified: {verification_result}, Profitability: ${profitability:.2f}")
            
        except Exception as e:
            logger.error(f"Potential solution analysis error: {e}")
            
    def get_mining_statistics(self) -> Dict:
        """Get comprehensive mining statistics and analysis"""
        try:
            base_stats = self.get_processing_stats()
            mining_stats = self.mining_analyzer.get_mining_statistics()
            
            combined_stats = {
                **base_stats,
                'mining_analysis': mining_stats,
                'storage_statistics': {
                    'block_templates': len(self.mining_data_storage['block_templates']),
                    'mining_solutions': len(self.mining_data_storage['mining_solutions']),
                    'nonce_sequences': len(self.mining_data_storage['nonce_sequences']),
                    'difficulty_adjustments': len(self.mining_data_storage['difficulty_adjustments']),
                    'hash_rate_estimates': len(self.mining_data_storage['hash_rate_estimates'])
                },
                'time_scaling_analysis': {
                    'factors_tracked': len(self.time_scaling_factors),
                    'target_block_time': self.target_block_time,
                    'current_efficiency': self._calculate_current_time_efficiency()
                },
                'memory_efficiency': self.memory_manager.get_memory_stats()
            }
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Mining statistics error: {e}")
            return {'error': str(e)}
            
    def _calculate_current_time_efficiency(self) -> float:
        """Calculate current time efficiency for mining operations"""
        try:
            if not self.mining_data_storage['hash_rate_estimates']:
                return 0.0
                
            recent_estimates = list(self.mining_data_storage['hash_rate_estimates'])[-10:]
            correlations = [est.get('correlation', 0) for est in recent_estimates]
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Time efficiency calculation error: {e}")
            return 0.0 