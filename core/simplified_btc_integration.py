#!/usr/bin/env python3
"""
Simplified BTC Integration Layer
===============================

Bridges the simplified API with the existing BTC processor system for live data integration.
Handles the 16-bit tick aggregator and 10,000 ticks/hour processing requirements.

Key Features:
- Live integration with existing BTC data processor
- High-frequency tick processing pipeline
- CPU optimization for scaling
- Real-time sustainment monitoring
- Error prevention and recovery mechanisms
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path
import threading
import queue
import json

# Import existing BTC processor components
try:
    from btc_data_processor import BTCDataProcessor
    from btc_processor_controller import BTCProcessorController  
    from quantum_btc_intelligence_core import QuantumBTCIntelligenceCore
    from schwabot_unified_math_v2 import (
        UnifiedQuantumTradingController,
        calculate_btc_processor_metrics
    )
    CORE_SYSTEMS_AVAILABLE = True
except ImportError:
    CORE_SYSTEMS_AVAILABLE = False
    print("⚠️ Core BTC systems not available - running in standalone mode")

logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Individual tick data structure"""
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    source: str = "live"
    sequence: int = 0

@dataclass
class ProcessingMetrics:
    """Real-time processing metrics"""
    ticks_processed: int = 0
    ticks_per_second: float = 0.0
    average_latency_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    errors_count: int = 0
    last_error: Optional[str] = None
    sustainment_index: float = 0.0
    confidence: float = 0.0

class SimplifiedBTCIntegration:
    """
    Simplified integration layer for BTC data processing with live tick aggregation
    """
    
    def __init__(self, 
                 simplified_api=None,
                 tick_buffer_size: int = 1000,
                 processing_batch_size: int = 100,
                 max_ticks_per_hour: int = 10000):
        """Initialize BTC integration layer"""
        self.simplified_api = simplified_api
        self.tick_buffer_size = tick_buffer_size
        self.processing_batch_size = processing_batch_size
        self.max_ticks_per_hour = max_ticks_per_hour
        
        # Initialize core systems if available
        if CORE_SYSTEMS_AVAILABLE:
            self.btc_processor = BTCDataProcessor()
            self.processor_controller = BTCProcessorController() 
            self.quantum_core = QuantumBTCIntelligenceCore()
            self.trading_controller = UnifiedQuantumTradingController()
        else:
            self.btc_processor = None
            self.processor_controller = None
            self.quantum_core = None
            self.trading_controller = None
        
        # Tick processing pipeline
        self.tick_queue = queue.Queue(maxsize=tick_buffer_size)
        self.processing_queue = queue.Queue(maxsize=processing_batch_size)
        self.result_queue = queue.Queue()
        
        # Processing threads
        self.tick_ingestion_thread = None
        self.processing_thread = None
        self.output_thread = None
        
        # State management
        self.is_running = False
        self.processing_active = False
        self.tick_counter = 0
        self.start_time = None
        
        # Performance tracking
        self.metrics = ProcessingMetrics()
        self.performance_history = []
        
        # Rate limiting for high-frequency processing
        self.last_tick_time = 0.0
        self.min_tick_interval = 3600.0 / max_ticks_per_hour  # seconds between ticks
        
        # Callbacks for simplified API
        self.data_callbacks = []
        
        logger.info("SimplifiedBTCIntegration initialized")
    
    def start_integration(self) -> bool:
        """Start the BTC integration system"""
        try:
            self.is_running = True
            self.processing_active = True
            self.start_time = datetime.now(timezone.utc)
            
            # Start processing threads
            self.tick_ingestion_thread = threading.Thread(
                target=self._tick_ingestion_loop, daemon=True
            )
            self.processing_thread = threading.Thread(
                target=self._processing_loop, daemon=True
            )
            self.output_thread = threading.Thread(
                target=self._output_loop, daemon=True
            )
            
            self.tick_ingestion_thread.start()
            self.processing_thread.start()
            self.output_thread.start()
            
            logger.info("BTC integration system started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start BTC integration: {e}")
            return False
    
    def stop_integration(self) -> None:
        """Stop the BTC integration system"""
        self.is_running = False
        self.processing_active = False
        
        # Wait for threads to finish
        if self.tick_ingestion_thread and self.tick_ingestion_thread.is_alive():
            self.tick_ingestion_thread.join(timeout=5.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=5.0)
        
        logger.info("BTC integration system stopped")
    
    def ingest_tick(self, tick_data: TickData) -> bool:
        """Ingest a new tick for processing"""
        try:
            # Rate limiting check
            current_time = time.time()
            if current_time - self.last_tick_time < self.min_tick_interval:
                # Skip this tick to maintain rate limit
                return False
            
            self.last_tick_time = current_time
            
            # Add to queue for processing
            if not self.tick_queue.full():
                self.tick_queue.put(tick_data, block=False)
                self.tick_counter += 1
                return True
            else:
                logger.warning("Tick queue full - dropping tick")
                return False
                
        except Exception as e:
            logger.error(f"Error ingesting tick: {e}")
            self.metrics.errors_count += 1
            self.metrics.last_error = str(e)
            return False
    
    def ingest_live_tick(self, price: float, volume: float, 
                        bid: Optional[float] = None, ask: Optional[float] = None) -> bool:
        """Convenience method for ingesting live tick data"""
        tick = TickData(
            timestamp=datetime.now(timezone.utc),
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            spread=(ask - bid) if (ask and bid) else None,
            source="live",
            sequence=self.tick_counter
        )
        return self.ingest_tick(tick)
    
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for processed data"""
        self.data_callbacks.append(callback)
    
    def get_current_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics"""
        # Update real-time metrics
        if self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            if elapsed > 0:
                self.metrics.ticks_per_second = self.tick_counter / elapsed
        
        return self.metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        metrics = self.get_current_metrics()
        
        return {
            'integration_active': self.is_running,
            'processing_active': self.processing_active,
            'core_systems_available': CORE_SYSTEMS_AVAILABLE,
            'ticks_processed': metrics.ticks_processed,
            'ticks_per_second': metrics.ticks_per_second,
            'average_latency_ms': metrics.average_latency_ms,
            'cpu_usage': metrics.cpu_usage,
            'memory_usage_mb': metrics.memory_usage_mb,
            'errors_count': metrics.errors_count,
            'last_error': metrics.last_error,
            'sustainment_index': metrics.sustainment_index,
            'confidence': metrics.confidence,
            'queue_sizes': {
                'tick_queue': self.tick_queue.qsize(),
                'processing_queue': self.processing_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            }
        }
    
    def _tick_ingestion_loop(self) -> None:
        """Main tick ingestion loop"""
        logger.info("Tick ingestion loop started")
        
        while self.is_running:
            try:
                # Get tick from queue (blocking with timeout)
                try:
                    tick = self.tick_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Basic validation
                if not self._validate_tick(tick):
                    continue
                
                # Add to processing queue
                if not self.processing_queue.full():
                    self.processing_queue.put(tick, block=False)
                else:
                    logger.warning("Processing queue full - dropping tick")
                
                self.tick_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in tick ingestion loop: {e}")
                self.metrics.errors_count += 1
                self.metrics.last_error = str(e)
        
        logger.info("Tick ingestion loop stopped")
    
    def _processing_loop(self) -> None:
        """Main processing loop for tick data"""
        logger.info("Processing loop started")
        
        batch_buffer = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Get tick from processing queue
                try:
                    tick = self.processing_queue.get(timeout=0.1)
                    batch_buffer.append(tick)
                except queue.Empty:
                    pass
                
                current_time = time.time()
                
                # Process batch when full or timeout reached
                if (len(batch_buffer) >= self.processing_batch_size or 
                    (batch_buffer and current_time - last_batch_time > 1.0)):
                    
                    self._process_tick_batch(batch_buffer)
                    batch_buffer.clear()
                    last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.metrics.errors_count += 1
                self.metrics.last_error = str(e)
        
        # Process remaining ticks
        if batch_buffer:
            self._process_tick_batch(batch_buffer)
        
        logger.info("Processing loop stopped")
    
    def _process_tick_batch(self, ticks: List[TickData]) -> None:
        """Process a batch of ticks"""
        if not ticks:
            return
        
        processing_start = time.time()
        
        try:
            # Extract latest tick for processing
            latest_tick = ticks[-1]
            
            if CORE_SYSTEMS_AVAILABLE and self.trading_controller:
                # Use real BTC processor integration
                result = self._process_with_btc_system(latest_tick, ticks)
            else:
                # Use mock processing
                result = self._process_mock_tick(latest_tick, ticks)
            
            # Calculate processing metrics
            processing_time = time.time() - processing_start
            self.metrics.average_latency_ms = processing_time * 1000
            self.metrics.ticks_processed += len(ticks)
            
            # Send to output queue
            if not self.result_queue.full():
                self.result_queue.put(result, block=False)
            
        except Exception as e:
            logger.error(f"Error processing tick batch: {e}")
            self.metrics.errors_count += 1
            self.metrics.last_error = str(e)
    
    def _process_with_btc_system(self, latest_tick: TickData, batch: List[TickData]) -> Dict[str, Any]:
        """Process tick using real BTC system integration"""
        try:
            # Prepare market state from tick batch
            prices = [tick.price for tick in batch]
            volumes = [tick.volume for tick in batch]
            
            # Calculate metrics using unified math system
            market_state = {
                'latencies': [25.0],  # Mock latency
                'operations': [len(batch)],
                'profit_deltas': [0.02],  # Mock profit delta
                'resource_costs': [1.0],
                'utility_values': [0.8],
                'predictions': prices,
                'subsystem_scores': [0.8, 0.75, 0.9, 0.85],
                'system_states': [0.8],
                'uptime_ratio': 0.99,
                'iteration_states': [[0.8, 0.7]]
            }
            
            # Get trading evaluation
            result = self.trading_controller.evaluate_trade_opportunity(
                price=latest_tick.price,
                volume=latest_tick.volume,
                market_state=market_state
            )
            
            # Calculate BTC processor metrics
            btc_metrics = calculate_btc_processor_metrics(
                volume=latest_tick.volume,
                price_velocity=prices[-1] - prices[0] if len(prices) > 1 else 0.0,
                profit_residual=0.02,  # Mock profit residual
                current_hash="mock_hash",
                pool_hash="mock_pool_hash",
                echo_memory=["mem1", "mem2"],
                tick_entropy=0.5,
                phase_confidence=result['confidence'],
                current_xi=result['sustainment_metrics']['sustainment_index'],
                previous_xi=0.8,
                previous_entropy=0.5,
                time_delta=1.0
            )
            
            # Update metrics
            self.metrics.sustainment_index = result['sustainment_metrics']['sustainment_index']
            self.metrics.confidence = result['confidence']
            
            return {
                'timestamp': latest_tick.timestamp.isoformat(),
                'tick_data': {
                    'price': latest_tick.price,
                    'volume': latest_tick.volume,
                    'spread': latest_tick.spread
                },
                'trading_analysis': result,
                'btc_metrics': btc_metrics,
                'batch_size': len(batch),
                'processing_metrics': {
                    'ticks_per_second': self.metrics.ticks_per_second,
                    'latency_ms': self.metrics.average_latency_ms
                }
            }
            
        except Exception as e:
            logger.error(f"Error in BTC system processing: {e}")
            raise
    
    def _process_mock_tick(self, latest_tick: TickData, batch: List[TickData]) -> Dict[str, Any]:
        """Process tick using mock system when core systems unavailable"""
        # Mock trading analysis
        mock_analysis = {
            'should_execute': latest_tick.price > 50000,
            'confidence': 0.75,
            'position_size': 0.1,
            'sustainment_metrics': {
                'sustainment_index': 0.8
            },
            'fractal_metrics': {
                'hurst_exponent': 0.55,
                'hausdorff_dimension': 1.45
            }
        }
        
        # Update metrics
        self.metrics.sustainment_index = 0.8
        self.metrics.confidence = 0.75
        
        return {
            'timestamp': latest_tick.timestamp.isoformat(),
            'tick_data': {
                'price': latest_tick.price,
                'volume': latest_tick.volume,
                'spread': latest_tick.spread
            },
            'trading_analysis': mock_analysis,
            'batch_size': len(batch),
            'processing_metrics': {
                'ticks_per_second': self.metrics.ticks_per_second,
                'latency_ms': self.metrics.average_latency_ms
            },
            'mock_mode': True
        }
    
    def _output_loop(self) -> None:
        """Output loop for sending results to callbacks"""
        logger.info("Output loop started")
        
        while self.is_running:
            try:
                # Get result from queue
                try:
                    result = self.result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Send to all registered callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Error in data callback: {e}")
                
                self.result_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in output loop: {e}")
                self.metrics.errors_count += 1
                self.metrics.last_error = str(e)
        
        logger.info("Output loop stopped")
    
    def _validate_tick(self, tick: TickData) -> bool:
        """Validate tick data"""
        if tick.price <= 0:
            return False
        if tick.volume < 0:
            return False
        if tick.timestamp > datetime.now(timezone.utc):
            return False
        return True

# ===== INTEGRATION WITH SIMPLIFIED API =====

def integrate_with_simplified_api(simplified_api, btc_integration: SimplifiedBTCIntegration):
    """Integrate BTC processing with simplified API"""
    
    # Register callback to send data to API
    def api_data_callback(result: Dict[str, Any]):
        """Callback to send processed data to simplified API"""
        if hasattr(simplified_api, 'broadcast_trading_data'):
            simplified_api.broadcast_trading_data(result)
    
    btc_integration.register_data_callback(api_data_callback)
    
    # Add BTC integration methods to API
    simplified_api.btc_integration = btc_integration
    
    # Override API's _get_realtime_data to use live BTC data
    original_get_realtime_data = simplified_api._get_realtime_data
    
    async def enhanced_get_realtime_data():
        """Enhanced real-time data with BTC integration"""
        base_data = await original_get_realtime_data()
        
        # Add BTC integration status
        btc_status = btc_integration.get_system_status()
        base_data['btc_integration'] = btc_status
        
        return base_data
    
    simplified_api._get_realtime_data = enhanced_get_realtime_data

# ===== CONVENIENCE FUNCTIONS =====

def create_integrated_system(simplified_api=None, **kwargs) -> SimplifiedBTCIntegration:
    """Create fully integrated BTC processing system"""
    btc_integration = SimplifiedBTCIntegration(**kwargs)
    
    if simplified_api:
        integrate_with_simplified_api(simplified_api, btc_integration)
    
    return btc_integration

def run_live_demo(duration_minutes: int = 10, ticks_per_minute: int = 100):
    """Run live demo with synthetic tick data"""
    from core.simplified_api import create_simplified_api
    import random
    
    # Create integrated system
    api = create_simplified_api()
    btc_integration = create_integrated_system(api)
    
    # Start systems
    btc_integration.start_integration()
    
    # Generate synthetic ticks
    base_price = 50000.0
    base_volume = 1500.0
    
    async def generate_ticks():
        for i in range(duration_minutes * ticks_per_minute):
            # Generate realistic tick
            price_change = random.gauss(0, 0.001)  # 0.1% volatility
            volume_change = random.gauss(0, 0.05)  # 5% volume volatility
            
            price = base_price * (1 + price_change)
            volume = base_volume * (1 + volume_change)
            
            # Ingest tick
            btc_integration.ingest_live_tick(price, volume)
            
            # Wait for next tick
            await asyncio.sleep(60.0 / ticks_per_minute)
        
        # Stop integration
        btc_integration.stop_integration()
    
    # Run demo
    print(f"🚀 Running live BTC integration demo for {duration_minutes} minutes")
    print(f"📊 Generating {ticks_per_minute} ticks per minute")
    print(f"🔌 WebSocket: ws://localhost:8000/ws")
    
    # Start API server in background
    import threading
    api_thread = threading.Thread(target=lambda: api.run(port=8000), daemon=True)
    api_thread.start()
    
    # Run tick generation
    asyncio.run(generate_ticks())

if __name__ == "__main__":
    # Run live demo
    run_live_demo(duration_minutes=5, ticks_per_minute=120)  # 2 ticks per second 