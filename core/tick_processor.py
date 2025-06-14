"""
Tick-by-Tick Processing System
=============================

Integrates GAN filter with broader Schwabot system for real-time
market data processing, error correction, and profit routing.

Provides:
- Real-time tick processing pipeline
- Integration with matrix clusters and parent-child adoption
- Profit signal routing and accumulation
- Fault tolerance and fallback mechanisms
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
from collections import deque
import time

from .gan_filter import GANFilter, TickProcessingResult, GANAnomalyMetrics
from .hooks import ncco_manager, sfsss_router, echo_logger

logger = logging.getLogger(__name__)

@dataclass
class TickProfile:
    """Profile for incoming market tick"""
    timestamp: float
    symbol: str
    price: float
    volume: float
    raw_vector: np.ndarray
    source: str = "market"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingPipeline:
    """Processing pipeline configuration"""
    gan_filter: GANFilter
    fallback_handlers: List[Callable] = field(default_factory=list)
    profit_accumulators: Dict[str, float] = field(default_factory=dict)
    processing_stats: Dict[str, int] = field(default_factory=lambda: {
        'total_ticks': 0,
        'anomalies_detected': 0,
        'corrections_applied': 0,
        'profit_signals_generated': 0
    })

class TickProcessor:
    """
    Main tick processing engine integrating GAN filter with Schwabot system.
    Handles real-time processing, profit routing, and system integration.
    """
    
    def __init__(self, gan_config: Dict[str, Any], input_dim: int = 64):
        self.gan_filter = GANFilter(gan_config, input_dim)
        self.pipeline = ProcessingPipeline(gan_filter=self.gan_filter)
        
        # Processing queues
        self.tick_queue = deque(maxlen=10000)
        self.result_queue = deque(maxlen=1000)
        
        # System integration points
        self.profit_router = None
        self.fault_handler = None
        self.dashboard_updater = None
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.last_health_check = time.time()
        
    def register_profit_router(self, router: Callable[[Dict[str, float]], None]) -> None:
        """Register profit routing callback"""
        self.profit_router = router
        logger.info("Profit router registered")
        
    def register_fault_handler(self, handler: Callable[[Exception, TickProfile], None]) -> None:
        """Register fault handling callback"""
        self.fault_handler = handler
        logger.info("Fault handler registered")
        
    def register_dashboard_updater(self, updater: Callable[[Dict[str, Any]], None]) -> None:
        """Register dashboard update callback"""  
        self.dashboard_updater = updater
        logger.info("Dashboard updater registered")
        
    def process_tick(self, tick_profile: TickProfile) -> Optional[TickProcessingResult]:
        """
        Process single market tick through GAN filter pipeline.
        
        Args:
            tick_profile: Market tick data
            
        Returns:
            Processing result or None if failed
        """
        start_time = time.time()
        
        try:
            # Add to processing queue
            self.tick_queue.append(tick_profile)
            
            # Extract vector from tick data
            if tick_profile.raw_vector is None:
                # Synthesize vector from price/volume data
                vector = self._synthesize_vector(tick_profile)
            else:
                vector = tick_profile.raw_vector
                
            # Run GAN detection
            anomaly_metrics = self.gan_filter.detect(vector)
            
            # Get full processing result from cache
            tick_hash = self.gan_filter._create_tick_hash(vector)
            result = self.gan_filter.get_tick_result(tick_hash)
            
            if result:
                # Update pipeline stats
                self.pipeline.processing_stats['total_ticks'] += 1
                if anomaly_metrics.is_anomaly:
                    self.pipeline.processing_stats['anomalies_detected'] += 1
                if result.correction_applied:
                    self.pipeline.processing_stats['corrections_applied'] += 1
                if result.profit_signal:
                    self.pipeline.processing_stats['profit_signals_generated'] += 1
                    
                # Route profit signals
                if self.profit_router and result.profit_signal:
                    try:
                        enriched_signals = {
                            **result.profit_signal,
                            'symbol': tick_profile.symbol,
                            'timestamp': tick_profile.timestamp,
                            'price': tick_profile.price,
                            'volume': tick_profile.volume
                        }
                        self.profit_router(enriched_signals)
                    except Exception as e:
                        logger.error(f"Profit routing failed: {e}")
                        
                # Add to result queue
                self.result_queue.append(result)
                
                # Update dashboard
                if self.dashboard_updater:
                    try:
                        dashboard_data = {
                            'tick_hash': tick_hash,
                            'anomaly_score': anomaly_metrics.anomaly_score,
                            'is_anomaly': anomaly_metrics.is_anomaly,
                            'correction_applied': result.correction_applied,
                            'profit_signals': result.profit_signal,
                            'processing_time': result.processing_time,
                            'symbol': tick_profile.symbol
                        }
                        self.dashboard_updater(dashboard_data)
                    except Exception as e:
                        logger.error(f"Dashboard update failed: {e}")
                        
                # Log significant events
                if anomaly_metrics.is_anomaly or result.correction_applied:
                    echo_logger.log_event({
                        'type': 'gan_anomaly' if anomaly_metrics.is_anomaly else 'matrix_correction',
                        'symbol': tick_profile.symbol,
                        'score': anomaly_metrics.anomaly_score,
                        'correction': result.correction_applied,
                        'profit': result.profit_signal.get('anomaly_profit', 0.0) if result.profit_signal else 0.0
                    })
                    
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Tick processing failed for {tick_profile.symbol}: {e}")
            
            # Trigger fault handler
            if self.fault_handler:
                try:
                    self.fault_handler(e, tick_profile)
                except Exception as handler_error:
                    logger.error(f"Fault handler failed: {handler_error}")
                    
            return None
            
    def _synthesize_vector(self, tick_profile: TickProfile) -> np.ndarray:
        """Synthesize vector from price/volume data when raw vector not available"""
        # Create 64-dimensional vector from available data
        base_vector = np.zeros(64)
        
        # Price features (first 20 elements)
        price_normalized = (tick_profile.price % 1000) / 1000.0  # Normalize price
        base_vector[:20] = np.sin(np.linspace(0, 2*np.pi*price_normalized, 20))
        
        # Volume features (next 20 elements)  
        volume_normalized = min(tick_profile.volume / 10000.0, 1.0)  # Cap at 10k
        base_vector[20:40] = np.cos(np.linspace(0, np.pi*volume_normalized, 20))
        
        # Time features (next 12 elements)
        hour = datetime.fromtimestamp(tick_profile.timestamp).hour
        minute = datetime.fromtimestamp(tick_profile.timestamp).minute
        base_vector[40:52] = np.sin(np.linspace(0, 2*np.pi*hour/24, 12))
        
        # Symbol hash features (remaining 12 elements)
        symbol_hash = hash(tick_profile.symbol) % 1000
        base_vector[52:64] = np.cos(np.linspace(0, 2*np.pi*symbol_hash/1000, 12))
        
        return base_vector
        
    async def process_tick_stream(self, tick_stream: List[TickProfile]) -> List[TickProcessingResult]:
        """Process stream of ticks asynchronously"""
        results = []
        
        for tick_profile in tick_stream:
            result = self.process_tick(tick_profile)
            if result:
                results.append(result)
                
            # Yield control for other coroutines
            await asyncio.sleep(0.001)
            
        return results
        
    def train_from_tick_history(self, tick_history: List[TickProfile]) -> None:
        """Train GAN filter from historical tick data"""
        logger.info(f"Training GAN from {len(tick_history)} historical ticks")
        
        # Convert tick profiles to vectors
        vectors = []
        for tick in tick_history:
            if tick.raw_vector is not None:
                vectors.append(tick.raw_vector)
            else:
                vectors.append(self._synthesize_vector(tick))
                
        # Train GAN filter
        self.gan_filter.train(vectors)
        logger.info("GAN training completed from tick history")
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        current_time = time.time()
        
        stats = {
            **self.pipeline.processing_stats,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0.0,
            'queue_size': len(self.tick_queue),
            'result_queue_size': len(self.result_queue),
            'cluster_count': len(self.gan_filter.cluster_db),
            'uptime_seconds': current_time - (self.last_health_check or current_time)
        }
        
        return stats
        
    def get_profit_summary(self) -> Dict[str, float]:
        """Get accumulated profit signals"""
        profit_routing = self.gan_filter.get_profit_routing()
        
        summary = {
            'total_anomaly_profit': 0.0,
            'total_correction_bonus': 0.0,
            'total_parent_boost': 0.0,
            'active_signals': len(profit_routing)
        }
        
        for signals in profit_routing.values():
            summary['total_anomaly_profit'] += signals.get('anomaly_profit', 0.0)
            summary['total_correction_bonus'] += signals.get('correction_bonus', 0.0)
            summary['total_parent_boost'] += signals.get('parent_boost', 0.0)
            
        return summary
        
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        current_time = time.time()
        
        health = {
            'status': 'healthy',
            'gan_model_loaded': self.gan_filter.generator is not None,
            'cluster_count': len(self.gan_filter.cluster_db),
            'recent_processing_avg': np.mean(list(self.processing_times)[-10:]) if len(self.processing_times) >= 10 else 0.0,
            'queue_health': len(self.tick_queue) < 9000,  # Not overflowing
            'last_check': current_time
        }
        
        # Check for system issues
        if len(self.tick_queue) > 9500:
            health['status'] = 'warning'
            health['warning'] = 'tick_queue_near_full'
            
        if health['recent_processing_avg'] > 0.1:  # 100ms threshold
            health['status'] = 'warning' 
            health['warning'] = 'slow_processing'
            
        self.last_health_check = current_time
        return health

# Integration factory function
def create_tick_processor(gan_config: Optional[Dict[str, Any]] = None) -> TickProcessor:
    """Create configured tick processor"""
    if gan_config is None:
        gan_config = {
            'latent_dim': 32,
            'hidden_dim': 64,
            'num_layers': 3,
            'anomaly_threshold': 0.75,
            'use_gpu': True
        }
        
    processor = TickProcessor(gan_config)
    
    # Wire up system integrations
    def profit_router(signals: Dict[str, float]) -> None:
        """Route profit signals to SFSSS system"""
        try:
            sfsss_router.route_profit_signal(signals)
        except Exception as e:
            logger.error(f"SFSSS profit routing failed: {e}")
            
    def fault_handler(error: Exception, tick: TickProfile) -> None:
        """Handle processing faults"""
        try:
            ncco_manager.report_fault({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'symbol': tick.symbol,
                'timestamp': tick.timestamp
            })
        except Exception as e:
            logger.error(f"NCCO fault reporting failed: {e}")
            
    processor.register_profit_router(profit_router)
    processor.register_fault_handler(fault_handler)
    
    return processor 