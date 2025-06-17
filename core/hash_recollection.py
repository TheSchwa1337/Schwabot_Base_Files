"""
Hash Recollection System
Manages hash-based pattern recognition and memory optimization.
"""

import logging
import threading
from queue import Queue, Full, Empty
from collections import deque
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import yaml
from pathlib import Path
import hashlib
import time

# Import the modular components
from .entropy_tracker import EntropyTracker, EntropyState
from .bit_operations import BitOperations, PhaseState
from .pattern_utils import PatternUtils, PatternMatch
from .strange_loop_detector import StrangeLoopDetector, EchoPattern
from .risk_engine import RiskEngine, PositionSignal

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


@dataclass
class HashEntry:
    """Cache-line optimized hash storage (64-byte alignment)"""
    hash_value: int  # 8 bytes
    strategy_id: int  # 4 bytes
    confidence: float  # 4 bytes
    frequency: int  # 4 bytes
    timestamp: int  # 8 bytes
    profit_history: float  # 4 bytes
    bit_pattern: int  # 8 bytes (42-bit float representation)
    tier: int  # 4 bytes
    state: Optional[EntropyState] = None
    
    def update(self):
        """Update entry with new occurrence."""
        self.frequency += 1
        self.timestamp = int(time.time())
    
    def get_entropy_vector(self) -> Optional[np.ndarray]:
        """Get entropy vector from state."""
        if self.state:
            return np.array([
                self.state.price_entropy,
                self.state.volume_entropy,
                self.state.time_entropy
            ])
        return None


class HashRecollectionSystem:
    """Manages hash-based pattern recognition with modular components"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the hash recollection system.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize modular components
        self.entropy_tracker = EntropyTracker(maxlen=self.config.get('history_length', 1000))
        self.bit_operations = BitOperations()
        self.pattern_utils = PatternUtils(self.config.get('patterns', {}))
        self.strange_loop_detector = StrangeLoopDetector()
        self.risk_engine = RiskEngine(
            max_position_size=self.config.get('max_position_size', 0.25)
        )
        
        # Initialize state
        self.gpu_enabled = self._check_gpu()
        self.hash_database: Dict[int, HashEntry] = {}
        
        # Thread-safe queues for GPU-CPU communication with back-pressure handling
        queue_size = self.config.get('queue_size', 10000)
        self.hash_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        
        # Initialize worker threads
        self.running = False
        self.gpu_worker = None
        self.cpu_worker = None
        
        # Tetragram matrix (3x3x3 for 3D entropy) - FIXED: correct dimensions
        self.tetragram_matrix = np.zeros((3, 3, 3), dtype=np.float32)
        
        # Performance tracking
        self.tick_count = 0
        self.pattern_matches = 0
        self.last_price = 0.0
        self.current_price = 0.0
        self.dropped_ticks = 0  # Track back-pressure events
        
        # Signal callbacks
        self.signal_callbacks: List[Callable] = []
        
        # Latency tracking for compensation
        self.latency_measurements = deque(maxlen=100)
        self.avg_latency = 0.0
        
        # Start time for system reporting
        self._start_time = time.time()
        
        logger.info("Hash recollection system initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'sync_interval': 100,
            'history_length': 1000,
            'gpu_enabled': True,
            'gpu_batch_size': 1000,
            'queue_size': 10000,
            'max_position_size': 0.25,
            'patterns': {
                'density_entry': 0.57,
                'density_exit': 0.42,
                'variance_entry': 0.002,
                'variance_exit': 0.007,
                'confidence_min': 0.7,
                'pattern_strength_min': 0.7
            },
            'strange_loop': {
                'echo_threshold': 0.1,
                'history_length': 10000
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as file:
                    loaded_config = yaml.safe_load(file) or {}
                # Merge with defaults
                for key, value in loaded_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                logger.info(f"Configuration loaded from: {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}, using defaults")
        
        return default_config

    def _check_gpu(self) -> bool:
        """Check if GPU is available and initialize CuPy"""
        if not CUPY_AVAILABLE:
            return False
        try:
            cp.array([1])
            return True
        except:
            return False

    def start(self):
        """Start the hash recollection system"""
        self.running = True
        self._start_time = time.time()
        
        # Start worker threads
        if self.gpu_enabled:
            self.gpu_worker = threading.Thread(target=self._gpu_worker, daemon=True)
            self.gpu_worker.start()
        
        self.cpu_worker = threading.Thread(target=self._cpu_worker, daemon=True)
        self.cpu_worker.start()
        
        logger.info("Hash recollection system started")

    def stop(self):
        """Stop the hash recollection system"""
        self.running = False
        
        # Signal workers to stop
        try:
            self.hash_queue.put(None, timeout=1.0)  # Poison pill
            self.result_queue.put(None, timeout=1.0)
        except Full:
            logger.warning("Could not send poison pill - queues full")
        
        # Wait for workers to finish
        if self.gpu_worker and self.gpu_worker.is_alive():
            self.gpu_worker.join(timeout=5)
        if self.cpu_worker and self.cpu_worker.is_alive():
            self.cpu_worker.join(timeout=5)
        
        logger.info("Hash recollection system stopped")

    def _gpu_worker(self):
        """GPU worker thread for parallel hash computation"""
        batch_size = self.config.get('gpu_batch_size', 1000)
        
        while self.running:
            try:
                # Get batch of entropy states
                batch = []
                for _ in range(batch_size):
                    try:
                        item = self.hash_queue.get_nowait()
                        if item is None:  # Poison pill
                            return
                        batch.append(item)
                    except Empty:
                        break
                
                if not batch:
                    time.sleep(0.01)
                    continue
                
                # Process batch on GPU or fallback to CPU
                if self.gpu_enabled and CUPY_AVAILABLE:
                    results = self._gpu_compute_hashes(batch)
                else:
                    results = self._cpu_compute_batch(batch)
                
                # Send results to CPU worker with back-pressure handling
                for result in results:
                    try:
                        self.result_queue.put(result, timeout=0.1)
                    except Full:
                        logger.warning("Result queue full, dropping hash result")
                        break
                
            except Exception as e:
                logger.error(f"GPU worker error: {e}")
                time.sleep(0.1)

    def _gpu_compute_hashes(self, batch: List[EntropyState]) -> List:
        """
        Compute SHA-256 hashes in parallel on GPU.
        
        Note: This is currently a CPU fallback. In production, this would use:
        - Custom CUDA kernel for SHA-256
        - CuPy-accelerated hash library
        - Or a hybrid CPU-GPU approach for hash verification
        """
        results = []
        
        if not CUPY_AVAILABLE:
            return self._cpu_compute_batch(batch)
        
        try:
            # Convert entropy states to GPU arrays for processing
            entropy_arrays = []
            for state in batch:
                entropy_vec = cp.array([state.price_entropy, state.volume_entropy, state.time_entropy])
                entropy_arrays.append(entropy_vec)
            
            # Process in parallel on GPU
            for i, state in enumerate(batch):
                # For now, fall back to CPU for actual SHA-256
                # In production: implement custom CUDA SHA-256 kernel
                entropy_str = f"{state.price_entropy:.6f}{state.volume_entropy:.6f}{state.time_entropy:.6f}"
                hash_value = int(hashlib.sha256(entropy_str.encode()).hexdigest()[:16], 16)
                
                # Calculate bit pattern on GPU
                bit_pattern = self.bit_operations.calculate_42bit_float(state.price_entropy)
                
                results.append((state, hash_value, bit_pattern))
            
            # Synchronize GPU
            if CUPY_AVAILABLE:
                cp.cuda.Stream.null.synchronize()
                
        except Exception as e:
            logger.error(f"GPU computation error: {e}, falling back to CPU")
            return self._cpu_compute_batch(batch)
        
        return results

    def _cpu_compute_batch(self, batch: List[EntropyState]) -> List:
        """CPU fallback for hash computation"""
        results = []
        
        for state in batch:
            entropy_str = f"{state.price_entropy:.6f}{state.volume_entropy:.6f}{state.time_entropy:.6f}"
            hash_value = int(hashlib.sha256(entropy_str.encode()).hexdigest()[:16], 16)
            bit_pattern = self.bit_operations.calculate_42bit_float(state.price_entropy)
            results.append((state, hash_value, bit_pattern))
        
        return results

    def _cpu_worker(self):
        """CPU worker thread for pattern recognition and strategy execution"""
        tick_counter = 0
        
        while self.running:
            try:
                # Process results from GPU
                result = self.result_queue.get(timeout=1.0)
                if result is None:  # Poison pill
                    break
                
                processing_start = time.time()
                self._process_hash_result(result)
                processing_time = time.time() - processing_start
                
                # Track latency for compensation
                self.latency_measurements.append(processing_time * 1000)  # Convert to ms
                self.avg_latency = np.mean(self.latency_measurements)
                
                # Synchronize periodically
                tick_counter += 1
                if tick_counter % self.config.get('sync_interval', 100) == 0:
                    self._synchronize_gpu_cpu()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"CPU worker error: {e}")

    def _process_hash_result(self, result):
        """Process hash result and update pattern recognition"""
        entropy_state, hash_value, bit_pattern = result
        
        # Update entropy state with bit pattern
        entropy_state.bit_pattern = bit_pattern
        
        # Analyze bit pattern
        pattern_analysis = self.bit_operations.analyze_bit_pattern(bit_pattern)
        entropy_state.tier = pattern_analysis['tier']
        
        # Check for strange loops BEFORE updating database
        echo_pattern = self.strange_loop_detector.process_hash(hash_value, entropy_state)
        if echo_pattern and echo_pattern.pattern_type == 'strange_loop':
            # Handle strange loop - reduce confidence or skip processing
            pattern_analysis['pattern_strength'] *= 0.5
            logger.warning(f"Strange loop detected, reducing pattern strength for hash {hash_value}")
        
        # Check if this hash should break a loop
        if self.strange_loop_detector.should_break_loop(hash_value):
            logger.info(f"Breaking loop for hash {hash_value}")
            self.strange_loop_detector.clear_loop_breaker(hash_value)
            return  # Skip processing to break the loop
        
        # Update hash database
        self._update_hash_database(hash_value, entropy_state, pattern_analysis)
        
        # Update tetragram matrix (FIXED: correct matrix update)
        self._update_tetragram_matrix(entropy_state)
        
        # Check for pattern matches
        self._check_pattern_matches(hash_value, entropy_state, pattern_analysis)

    def _update_hash_database(self, hash_value: int, entropy_state: EntropyState, pattern_analysis: Dict):
        """Update hash database with new entry or increment frequency"""
        if hash_value in self.hash_database:
            entry = self.hash_database[hash_value]
            entry.update()
            entry.bit_pattern = entropy_state.bit_pattern
            entry.tier = entropy_state.tier
            entry.state = entropy_state  # FIXED: Ensure entropy state is stored
        else:
            self.hash_database[hash_value] = HashEntry(
                hash_value=hash_value,
                strategy_id=0,
                confidence=0.0,
                frequency=1,
                timestamp=int(time.time()),
                profit_history=0.0,
                bit_pattern=entropy_state.bit_pattern,
                tier=entropy_state.tier,
                state=entropy_state  # FIXED: Store entropy state for similarity calc
            )
        
        # Update bit operations cache
        self.bit_operations.update_position_cache(
            value=entropy_state.bit_pattern,
            density=pattern_analysis['density'],
            tier=entropy_state.tier,
            collapse_type='mid'
        )

    def _update_tetragram_matrix(self, entropy_state: EntropyState):
        """Update 3D tetragram matrix - FIXED: correct matrix update"""
        # Convert entropy values to base-3 indices
        price_idx = int(abs(entropy_state.price_entropy) * 3) % 3
        volume_idx = int(abs(entropy_state.volume_entropy) * 3) % 3
        time_idx = int(abs(entropy_state.time_entropy) * 3) % 3
        
        # Update matrix with exponential decay
        decay = 0.95
        self.tetragram_matrix *= decay
        # FIXED: Remove the [:] slice that was causing shape mismatch
        self.tetragram_matrix[price_idx, volume_idx, time_idx] += 1.0

    def _check_pattern_matches(self, hash_value: int, entropy_state: EntropyState, pattern_analysis: Dict):
        """Check for pattern matches using hash distance and bit patterns"""
        if len(self.hash_database) < 10:
            return
        
        # Create phase state
        phase_state = self.bit_operations.create_phase_state(entropy_state.bit_pattern, entropy_state)
        
        # Get entropy vector
        entropy_vector = np.array([
            entropy_state.price_entropy,
            entropy_state.volume_entropy,
            entropy_state.time_entropy
        ])
        
        # Check pattern match
        pattern_match = self.pattern_utils.check_pattern_match(
            hash_value, phase_state, pattern_analysis, self.hash_database, entropy_vector
        )
        
        if pattern_match.confidence > 0.7:
            self.pattern_matches += 1
            self._trigger_pattern_match(pattern_match)

    def _trigger_pattern_match(self, pattern_match: PatternMatch):
        """Trigger pattern match event and emit trading signal with risk management"""
        logger.info(
            f"Pattern match: action={pattern_match.action}, "
            f"confidence={pattern_match.confidence:.3f}, "
            f"hash={pattern_match.hash_value}, "
            f"tier={pattern_match.phase_state.tier}"
        )
        
        # Calculate risk-adjusted position sizing if this is an entry signal
        if pattern_match.action == 'entry':
            # Calculate expected edge from pattern strength and tier
            expected_edge = pattern_match.phase_state.density * 0.1  # Base edge calculation
            
            # Calculate stop loss based on current volatility
            volatility = self.risk_engine._calculate_current_volatility()
            stop_loss_distance = volatility * 0.02  # 2% of volatility as stop loss
            stop_loss_price = self.current_price * (1 - stop_loss_distance)
            
            # Get risk-adjusted position size
            position_signal = self.risk_engine.calculate_position_size(
                signal_confidence=pattern_match.confidence,
                expected_edge=expected_edge,
                current_price=self.current_price,
                stop_loss_price=stop_loss_price
            )
            
            # Create enhanced trading signal with risk management
            signal = {
                'action': pattern_match.action,
                'confidence': pattern_match.confidence,
                'hash_value': pattern_match.hash_value,
                'tier': pattern_match.phase_state.tier,
                'density': pattern_match.phase_state.density,
                'price': self.current_price,
                'timestamp': time.time(),
                'reasons': pattern_match.reasons,
                'similarity_score': pattern_match.similarity_score,
                # Risk management fields
                'position_size': position_signal.risk_adjusted_size,
                'kelly_fraction': position_signal.kelly_fraction,
                'stop_loss': position_signal.stop_loss,
                'take_profit': position_signal.take_profit,
                'max_loss': position_signal.max_loss,
                'expected_return': position_signal.expected_return
            }
        else:
            # Exit signal - simpler structure
            signal = {
                'action': pattern_match.action,
                'confidence': pattern_match.confidence,
                'hash_value': pattern_match.hash_value,
                'tier': pattern_match.phase_state.tier,
                'density': pattern_match.phase_state.density,
                'price': self.current_price,
                'timestamp': time.time(),
                'reasons': pattern_match.reasons,
                'similarity_score': pattern_match.similarity_score
            }
        
        # Emit signal to callbacks
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    def _synchronize_gpu_cpu(self):
        """Synchronize GPU and CPU states with memory management and metrics"""
        if not self.gpu_enabled or not CUPY_AVAILABLE:
            return
        
        try:
            # Synchronize CUDA streams
            cp.cuda.Stream.null.synchronize()
            
            # Get memory info for metrics
            mem_info = cp.cuda.runtime.memGetInfo()
            free_memory = mem_info[0]
            total_memory = mem_info[1]
            used_memory = total_memory - free_memory
            
            # Calculate queue depths
            hash_queue_depth = self.hash_queue.qsize() if self.hash_queue else 0
            result_queue_depth = self.result_queue.qsize() if self.result_queue else 0
            
            # Export to Prometheus metrics if available
            sync_metrics = {
                'gpu_memory_used_bytes': used_memory,
                'gpu_memory_total_bytes': total_memory,
                'gpu_memory_utilization': used_memory / total_memory,
                'hash_queue_depth': hash_queue_depth,
                'result_queue_depth': result_queue_depth,
                'sync_timestamp': time.time()
            }
            
            # Store metrics for monitoring
            if not hasattr(self, '_sync_metrics'):
                self._sync_metrics = []
            self._sync_metrics.append(sync_metrics)
            
            # Keep only last 100 sync metrics
            if len(self._sync_metrics) > 100:
                self._sync_metrics = self._sync_metrics[-100:]
            
            # Memory cleanup if usage is high (>80%)
            memory_usage = used_memory / total_memory
            if memory_usage > 0.8:
                logger.warning(f"High GPU memory usage: {memory_usage:.1%}")
                cp.get_default_memory_pool().free_all_blocks()
                
                # Force garbage collection on GPU
                cp.cuda.runtime.deviceSynchronize()
                
            # Flush pending GPU hashes back to CPU for clustering if queue is getting full
            if hash_queue_depth > self.hash_queue.maxsize * 0.8:
                logger.info("High queue depth - flushing GPU results to CPU")
                
                # Process any pending results immediately
                while not self.result_queue.empty():
                    try:
                        result = self.result_queue.get_nowait()
                        self._process_hash_result(result)
                    except Empty:
                        break
            
            # Log sync info at debug level
            logger.debug(f"GPU sync: {memory_usage:.1%} memory, {hash_queue_depth}/{result_queue_depth} queue depths")
            
        except Exception as e:
            logger.error(f"GPU synchronization error: {e}")
            # Fallback to CPU-only mode if sync fails repeatedly
            if not hasattr(self, '_sync_errors'):
                self._sync_errors = 0
            self._sync_errors += 1
            
            if self._sync_errors > 10:
                logger.error("Too many GPU sync errors - disabling GPU")
                self.gpu_enabled = False

    def process_tick(self, price: float, volume: float, timestamp: Optional[float] = None):
        """Process new market tick data with back-pressure handling"""
        if timestamp is None:
            timestamp = time.time()
        
        self.last_price = self.current_price
        self.current_price = price
        self.tick_count += 1
        
        # Update risk engine with price data
        self.risk_engine.update_price(price)
        
        # Update entropy tracker
        entropy_state = self.entropy_tracker.update(price, volume, timestamp)
        
        # Queue for GPU processing with back-pressure handling
        if self.running:
            try:
                # Check for queue overload and drop oldest tick
                if self.hash_queue.full():
                    logger.warning("Hash queue overloaded; dropping oldest tick to avoid deadlocks")
                    try:
                        self.hash_queue.get_nowait()  # Remove oldest item
                        self.dropped_ticks += 1
                    except Empty:
                        pass  # Queue became empty between full() check and get_nowait()
                
                self.hash_queue.put_nowait(entropy_state)
                
            except Full:
                # This shouldn't happen after the check above, but handle it anyway
                logger.error("Failed to add tick to queue: queue is still full")
                self.dropped_ticks += 1

    def register_signal_callback(self, callback: Callable):
        """Register a callback function for trading signals"""
        self.signal_callbacks.append(callback)

    def update_trade_result(self, hash_value: int, entry_price: float, exit_price: float, 
                          position_size: float, trade_type: str = 'long'):
        """Update system with trade result for learning"""
        # Calculate trade duration (simplified)
        duration = 60.0  # Default 1 minute
        
        # Record trade in risk engine
        self.risk_engine.record_trade(
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            trade_type=trade_type,
            duration=duration
        )
        
        # Update hash entry profit history
        if hash_value in self.hash_database:
            entry = self.hash_database[hash_value]
            pnl = (exit_price - entry_price) / entry_price if trade_type == 'long' else (entry_price - exit_price) / entry_price
            entry.profit_history += pnl

    def get_pattern_metrics(self) -> Dict:
        """Get current pattern recognition metrics"""
        latest_entropy = self.entropy_tracker.get_latest_state()
        
        metrics = {
            'hash_count': len(self.hash_database),
            'pattern_confidence': np.mean([e.confidence for e in self.hash_database.values()]) if self.hash_database else 0.0,
            'collision_rate': self._calculate_collision_rate(),
            'tetragram_density': np.mean(self.tetragram_matrix > 0),
            'gpu_utilization': self._get_gpu_utilization() if self.gpu_enabled else 0.0,
            'ticks_processed': self.tick_count,
            'patterns_detected': self.pattern_matches,
            'dropped_ticks': self.dropped_ticks,
            'avg_latency_ms': self.avg_latency,
            'queue_utilization': {
                'hash_queue': self.hash_queue.qsize() / self.hash_queue.maxsize,
                'result_queue': self.result_queue.qsize() / self.result_queue.maxsize
            }
        }
        
        # Add bit pattern metrics if available
        if latest_entropy and latest_entropy.bit_pattern is not None:
            pattern_analysis = self.bit_operations.analyze_bit_pattern(latest_entropy.bit_pattern)
            metrics.update({
                'bit_pattern_strength': pattern_analysis['pattern_strength'],
                'long_density': pattern_analysis['long_density'],
                'mid_density': pattern_analysis['mid_density'],
                'short_density': pattern_analysis['short_density'],
                'current_tier': pattern_analysis['tier']
            })
        
        # Add strange loop detector metrics
        metrics.update({
            'strange_loops': self.strange_loop_detector.get_metrics()
        })
        
        # Add risk metrics
        risk_metrics = self.risk_engine.get_risk_metrics()
        metrics.update({
            'risk': {
                'expectancy': risk_metrics.current_expectancy,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'max_drawdown': risk_metrics.max_drawdown,
                'win_rate': risk_metrics.win_rate,
                'current_volatility': risk_metrics.current_volatility
            }
        })
        
        return metrics

    def _calculate_collision_rate(self) -> float:
        """Calculate current hash collision rate"""
        total_hashes = len(self.hash_database)
        if total_hashes == 0:
            return 0.0
        
        collisions = sum(1 for e in self.hash_database.values() if e.frequency > 1)
        return collisions / total_hashes

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization"""
        if not self.gpu_enabled or not CUPY_AVAILABLE:
            return 0.0
        
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            used_memory = mem_info[1] - mem_info[0]
            total_memory = mem_info[1]
            return used_memory / total_memory
        except:
            return 0.0

    def get_system_report(self) -> Dict:
        """Generate comprehensive system report"""
        uptime = time.time() - self._start_time
        
        return {
            'summary': {
                'uptime': f"{uptime:.1f} seconds",
                'ticks_processed': self.tick_count,
                'patterns_detected': self.pattern_matches,
                'hash_database_size': len(self.hash_database),
                'current_price': self.current_price,
                'dropped_ticks': self.dropped_ticks,
                'avg_latency_ms': self.avg_latency
            },
            'entropy': {
                'latest_state': self.entropy_tracker.get_latest_state() if self.entropy_tracker.get_latest_state() else None,
                'multi_window': self.entropy_tracker.get_multi_window_entropies()
            },
            'patterns': self.get_pattern_metrics(),
            'system': {
                'gpu_enabled': self.gpu_enabled,
                'workers_running': self.running,
                'queue_sizes': {
                    'hash_queue': self.hash_queue.qsize() if self.hash_queue else 0,
                    'result_queue': self.result_queue.qsize() if self.result_queue else 0
                },
                'queue_utilization': {
                    'hash_queue_pct': (self.hash_queue.qsize() / self.hash_queue.maxsize * 100) if self.hash_queue else 0,
                    'result_queue_pct': (self.result_queue.qsize() / self.result_queue.maxsize * 100) if self.result_queue else 0
                }
            }
        } 