"""
Integrated Profit Correlation System
====================================

Master integration system that combines:
- Critical Error Handler for robust error management
- Enhanced GPU Hash Processor for optimal performance
- News Profit Mathematical Bridge for core calculations
- Thermal management and GPU coordination
- Profit optimization and correlation analysis

This system provides the complete pipeline from news events to profit execution
with comprehensive error handling and recovery mechanisms.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import json
import numpy as np

from .critical_error_handler import CriticalErrorHandler, ErrorCategory, ErrorSeverity
from .enhanced_gpu_hash_processor import EnhancedGPUHashProcessor, HashProcessingResult, ProcessingMode
from .news_profit_mathematical_bridge import (
    NewsProfitMathematicalBridge, 
    NewsFactEvent, 
    MathematicalEventSignature,
    ProfitTiming
)

logger = logging.getLogger(__name__)

@dataclass
class SystemPerformanceMetrics:
    """Comprehensive system performance metrics"""
    total_news_events_processed: int
    total_correlations_calculated: int
    total_profit_opportunities_identified: int
    avg_processing_time_per_event: float
    gpu_utilization_rate: float
    error_recovery_success_rate: float
    thermal_throttle_incidents: int
    cache_hit_rate: float
    profit_prediction_accuracy: float
    system_uptime: float

@dataclass
class IntegratedProcessingResult:
    """Result of integrated processing pipeline"""
    event_id: str
    processing_start_time: datetime
    processing_end_time: datetime
    total_processing_time: float
    
    # News processing results
    fact_event: Optional[NewsFactEvent]
    mathematical_signature: Optional[MathematicalEventSignature]
    
    # Hash correlation results
    hash_correlation_result: Optional[HashProcessingResult]
    btc_correlation_strength: float
    
    # Profit analysis results
    profit_timing: Optional[ProfitTiming]
    estimated_profit_potential: float
    risk_assessment: float
    
    # System state
    processing_mode: ProcessingMode
    thermal_state: str
    errors_encountered: List[str]
    recovery_actions_taken: List[str]
    
    # Success indicators
    pipeline_successful: bool
    profit_opportunity_identified: bool

class IntegratedProfitCorrelationSystem:
    """
    Master system that integrates all components for profit correlation analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the integrated system"""
        
        self.config = config or self._default_config()
        
        # Initialize core components
        self.error_handler = CriticalErrorHandler(self.config.get('error_handler', {}))
        self.gpu_processor = EnhancedGPUHashProcessor(
            self.config.get('gpu_processor', {}), 
            self.error_handler
        )
        self.news_bridge = NewsProfitMathematicalBridge(
            profit_navigator=None,  # Would be injected
            btc_controller=None,    # Would be injected
            fractal_controller=None # Would be injected
        )
        
        # System state
        self.system_running = False
        self.processing_queue = asyncio.Queue(maxsize=self.config['processing_queue_size'])
        self.result_history: deque = deque(maxlen=self.config['result_history_size'])
        
        # Performance tracking
        self.performance_metrics = SystemPerformanceMetrics(
            total_news_events_processed=0,
            total_correlations_calculated=0,
            total_profit_opportunities_identified=0,
            avg_processing_time_per_event=0.0,
            gpu_utilization_rate=0.0,
            error_recovery_success_rate=0.0,
            thermal_throttle_incidents=0,
            cache_hit_rate=0.0,
            profit_prediction_accuracy=0.0,
            system_uptime=0.0
        )
        
        # Threading
        self._processing_workers = []
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # System start time
        self._start_time = time.time()
        
        logger.info("Integrated Profit Correlation System initialized")

    def _default_config(self) -> Dict:
        """Default system configuration"""
        return {
            'processing_queue_size': 5000,
            'result_history_size': 10000,
            'max_processing_workers': 4,
            'processing_timeout_seconds': 30.0,
            'correlation_batch_size': 50,
            'profit_threshold_basis_points': 25.0,
            'risk_tolerance': 0.7,
            'monitoring_interval_seconds': 10.0,
            'performance_window_size': 1000,
            'error_escalation_threshold': 10,  # errors per hour
            'thermal_monitoring_enabled': True,
            'gpu_fallback_enabled': True,
            'correlation_caching_enabled': True
        }

    async def start_system(self):
        """Start the integrated system"""
        if self.system_running:
            logger.warning("System already running")
            return
        
        try:
            # Start error handler monitoring
            self.error_handler.start_monitoring()
            
            # Start GPU processor
            await self.gpu_processor.start_processing()
            
            # Start processing workers
            await self._start_processing_workers()
            
            # Start system monitoring
            await self._start_system_monitoring()
            
            self.system_running = True
            self._start_time = time.time()
            
            logger.info("✅ Integrated Profit Correlation System started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start system: {e}")
            await self.error_handler.handle_critical_error(
                ErrorCategory.SYSTEM_INTEGRATION,
                'IntegratedProfitCorrelationSystem',
                e,
                {'startup_phase': 'system_start'}
            )
            raise

    async def stop_system(self):
        """Stop the integrated system"""
        if not self.system_running:
            return
        
        try:
            self.system_running = False
            
            # Stop processing workers
            await self._stop_processing_workers()
            
            # Stop GPU processor
            await self.gpu_processor.stop_processing()
            
            # Stop error handler
            self.error_handler.stop_monitoring()
            
            # Stop monitoring
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
            
            logger.info("✅ Integrated Profit Correlation System stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping system: {e}")

    async def process_news_events(self, news_events: List[Dict]) -> List[IntegratedProcessingResult]:
        """
        Process a batch of news events through the complete pipeline
        """
        if not self.system_running:
            raise RuntimeError("System not running. Call start_system() first.")
        
        start_time = time.time()
        results = []
        
        try:
            # Process events in batches for optimal performance
            batch_size = self.config['correlation_batch_size']
            
            for i in range(0, len(news_events), batch_size):
                batch = news_events[i:i + batch_size]
                batch_results = await self._process_event_batch(batch)
                results.extend(batch_results)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(len(news_events), processing_time, results)
            
            logger.info(f"Processed {len(news_events)} news events in {processing_time:.3f}s")
            
            return results
            
        except Exception as e:
            error_context = {
                'event_count': len(news_events),
                'processing_time': time.time() - start_time,
                'batch_size': batch_size
            }
            
            await self.error_handler.handle_critical_error(
                ErrorCategory.SYSTEM_INTEGRATION,
                'IntegratedProfitCorrelationSystem',
                e,
                error_context
            )
            
            return []

    async def _process_event_batch(self, events: List[Dict]) -> List[IntegratedProcessingResult]:
        """Process a batch of events through the pipeline"""
        
        results = []
        
        # Step 1: Extract fact events from news
        fact_events = []
        for event_data in events:
            try:
                fact_event = await self._extract_fact_event(event_data)
                if fact_event:
                    fact_events.append(fact_event)
            except Exception as e:
                logger.warning(f"Failed to extract fact event: {e}")
        
        if not fact_events:
            logger.warning("No valid fact events extracted from batch")
            return results
        
        # Step 2: Generate mathematical signatures
        try:
            signatures = await self.news_bridge.generate_mathematical_signatures(fact_events)
        except Exception as e:
            await self.error_handler.handle_critical_error(
                ErrorCategory.NEWS_CORRELATION,
                'news_bridge.generate_mathematical_signatures',
                e,
                {'fact_event_count': len(fact_events)}
            )
            signatures = []
        
        # Step 3: Get BTC patterns for correlation
        try:
            btc_patterns = await self._get_btc_patterns()
        except Exception as e:
            await self.error_handler.handle_critical_error(
                ErrorCategory.GPU_HASH_COMPUTATION,
                'get_btc_patterns',
                e,
                {}
            )
            btc_patterns = {}
        
        # Step 4: Calculate hash correlations
        correlation_results = {}
        if signatures and btc_patterns:
            try:
                correlation_results = await self.gpu_processor.process_news_hash_correlation(
                    signatures, btc_patterns
                )
            except Exception as e:
                await self.error_handler.handle_critical_error(
                    ErrorCategory.GPU_HASH_COMPUTATION,
                    'gpu_processor.process_news_hash_correlation',
                    e,
                    {'signature_count': len(signatures)}
                )
        
        # Step 5: Calculate profit timings
        profit_timings = {}
        if correlation_results:
            try:
                correlations_dict = {
                    sig: result.correlation_strength 
                    for sig, result in correlation_results.items()
                }
                profit_timings_list = await self.news_bridge.calculate_profit_timings(correlations_dict)
                profit_timings = {
                    timing.hash_correlation_strength: timing 
                    for timing in profit_timings_list
                }
            except Exception as e:
                await self.error_handler.handle_critical_error(
                    ErrorCategory.PROFIT_CALCULATION,
                    'news_bridge.calculate_profit_timings',
                    e,
                    {'correlation_count': len(correlation_results)}
                )
        
        # Step 6: Create integrated results
        for i, event_data in enumerate(events):
            result = await self._create_integrated_result(
                event_data, 
                fact_events[i] if i < len(fact_events) else None,
                signatures[i] if i < len(signatures) else None,
                correlation_results,
                profit_timings
            )
            results.append(result)
        
        return results

    async def _extract_fact_event(self, event_data: Dict) -> Optional[NewsFactEvent]:
        """Extract factual event from raw news data"""
        try:
            # This would integrate with the actual news processing pipeline
            return NewsFactEvent(
                event_id=event_data.get('id', f"event_{int(time.time())}"),
                timestamp=datetime.fromtimestamp(event_data.get('timestamp', time.time())),
                keywords=event_data.get('keywords', []),
                corroboration_count=event_data.get('corroboration_count', 1),
                trust_hierarchy=event_data.get('trust_score', 0.5),
                event_hash=event_data.get('hash', ''),
                block_timestamp=int(event_data.get('timestamp', time.time())),
                profit_correlation_potential=event_data.get('profit_potential', 0.0)
            )
        except Exception as e:
            logger.error(f"Error extracting fact event: {e}")
            return None

    async def _get_btc_patterns(self) -> Dict[str, Any]:
        """Get current BTC patterns for correlation"""
        try:
            # This would integrate with the actual BTC processor
            # For now, generate mock patterns
            current_time = time.time()
            patterns = {}
            
            for i in range(5):
                pattern_id = f"btc_pattern_{i}"
                patterns[pattern_id] = {
                    'hash': f"mock_hash_{current_time}_{i}",
                    'timestamp': current_time - i * 60,
                    'price': 45000.0 + i * 100
                }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting BTC patterns: {e}")
            return {}

    async def _create_integrated_result(self, 
                                      event_data: Dict,
                                      fact_event: Optional[NewsFactEvent],
                                      signature: Optional[MathematicalEventSignature],
                                      correlation_results: Dict[str, HashProcessingResult],
                                      profit_timings: Dict[str, ProfitTiming]) -> IntegratedProcessingResult:
        """Create comprehensive result for a single event"""
        
        processing_start = datetime.now()
        errors_encountered = []
        recovery_actions = []
        
        # Get correlation result
        hash_result = None
        btc_correlation = 0.0
        if signature and signature.combined_signature in correlation_results:
            hash_result = correlation_results[signature.combined_signature]
            btc_correlation = hash_result.correlation_strength
            
            if hash_result.error_occurred:
                errors_encountered.append(hash_result.error_message)
            if hash_result.fallback_used:
                recovery_actions.append("GPU fallback to CPU processing")
        
        # Get profit timing
        profit_timing = None
        profit_potential = 0.0
        risk_assessment = 0.5
        
        if btc_correlation in profit_timings:
            profit_timing = profit_timings[btc_correlation]
            profit_potential = profit_timing.profit_expectation
            risk_assessment = profit_timing.risk_factor
        
        # Determine success
        pipeline_successful = (
            fact_event is not None and 
            signature is not None and 
            hash_result is not None and 
            not hash_result.error_occurred
        )
        
        profit_opportunity = (
            pipeline_successful and
            profit_potential >= self.config['profit_threshold_basis_points'] and
            risk_assessment <= self.config['risk_tolerance']
        )
        
        processing_end = datetime.now()
        
        return IntegratedProcessingResult(
            event_id=event_data.get('id', f"event_{int(time.time())}"),
            processing_start_time=processing_start,
            processing_end_time=processing_end,
            total_processing_time=(processing_end - processing_start).total_seconds(),
            fact_event=fact_event,
            mathematical_signature=signature,
            hash_correlation_result=hash_result,
            btc_correlation_strength=btc_correlation,
            profit_timing=profit_timing,
            estimated_profit_potential=profit_potential,
            risk_assessment=risk_assessment,
            processing_mode=hash_result.mode_used if hash_result else ProcessingMode.CPU_FALLBACK,
            thermal_state=hash_result.thermal_state if hash_result else "unknown",
            errors_encountered=errors_encountered,
            recovery_actions_taken=recovery_actions,
            pipeline_successful=pipeline_successful,
            profit_opportunity_identified=profit_opportunity
        )

    def _update_performance_metrics(self, 
                                  event_count: int, 
                                  processing_time: float, 
                                  results: List[IntegratedProcessingResult]):
        """Update comprehensive performance metrics"""
        
        with self._lock:
            # Update basic counts
            self.performance_metrics.total_news_events_processed += event_count
            
            # Calculate correlation count
            correlation_count = sum(1 for r in results if r.hash_correlation_result is not None)
            self.performance_metrics.total_correlations_calculated += correlation_count
            
            # Calculate profit opportunities
            profit_opportunities = sum(1 for r in results if r.profit_opportunity_identified)
            self.performance_metrics.total_profit_opportunities_identified += profit_opportunities
            
            # Update average processing time
            current_avg = self.performance_metrics.avg_processing_time_per_event
            total_processed = self.performance_metrics.total_news_events_processed
            
            if total_processed > event_count:
                self.performance_metrics.avg_processing_time_per_event = (
                    (current_avg * (total_processed - event_count) + processing_time) / total_processed
                )
            else:
                self.performance_metrics.avg_processing_time_per_event = processing_time / event_count
            
            # Update GPU utilization
            gpu_results = sum(1 for r in results 
                            if r.hash_correlation_result and 
                               r.hash_correlation_result.mode_used == ProcessingMode.GPU_ACCELERATED)
            gpu_rate = gpu_results / len(results) if results else 0.0
            self.performance_metrics.gpu_utilization_rate = (
                0.9 * self.performance_metrics.gpu_utilization_rate + 0.1 * gpu_rate
            )
            
            # Update error recovery rate
            total_errors = sum(len(r.errors_encountered) for r in results)
            total_recoveries = sum(len(r.recovery_actions_taken) for r in results)
            if total_errors > 0:
                recovery_rate = total_recoveries / total_errors
                self.performance_metrics.error_recovery_success_rate = (
                    0.9 * self.performance_metrics.error_recovery_success_rate + 0.1 * recovery_rate
                )
            
            # Update system uptime
            self.performance_metrics.system_uptime = time.time() - self._start_time

    async def _start_processing_workers(self):
        """Start background processing workers"""
        worker_count = self.config['max_processing_workers']
        
        for i in range(worker_count):
            worker = asyncio.create_task(self._processing_worker(f"worker_{i}"))
            self._processing_workers.append(worker)
        
        logger.info(f"Started {worker_count} processing workers")

    async def _stop_processing_workers(self):
        """Stop background processing workers"""
        for worker in self._processing_workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self._processing_workers:
            await asyncio.gather(*self._processing_workers, return_exceptions=True)
        
        self._processing_workers.clear()
        logger.info("Stopped all processing workers")

    async def _processing_worker(self, worker_id: str):
        """Background processing worker"""
        logger.info(f"Processing worker {worker_id} started")
        
        while self.system_running:
            try:
                # Get work from queue (with timeout)
                try:
                    work_item = await asyncio.wait_for(
                        self.processing_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process work item
                if work_item:
                    await self._process_work_item(work_item, worker_id)
                
            except Exception as e:
                logger.error(f"Error in processing worker {worker_id}: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error
        
        logger.info(f"Processing worker {worker_id} stopped")

    async def _process_work_item(self, work_item: Dict, worker_id: str):
        """Process a single work item"""
        try:
            # This would handle different types of work items
            # For now, just log the processing
            logger.debug(f"Worker {worker_id} processing item: {work_item.get('type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} failed to process item: {e}")

    async def _start_system_monitoring(self):
        """Start system monitoring thread"""
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System monitoring started")

    def _monitoring_loop(self):
        """System monitoring loop"""
        while self.system_running:
            try:
                # Update performance metrics from components
                self._update_component_metrics()
                
                # Check system health
                self._check_system_health()
                
                # Log performance summary
                self._log_performance_summary()
                
                time.sleep(self.config['monitoring_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error

    def _update_component_metrics(self):
        """Update metrics from individual components"""
        try:
            # Get GPU processor stats
            gpu_stats = self.gpu_processor.get_performance_statistics()
            
            # Get error handler stats
            error_stats = self.error_handler.get_error_statistics()
            
            # Update thermal throttle incidents
            if 'thermal_throttle_count' in gpu_stats:
                self.performance_metrics.thermal_throttle_incidents = gpu_stats['thermal_throttle_count']
            
            # Update cache hit rate
            if 'cache_hit_rate' in gpu_stats:
                self.performance_metrics.cache_hit_rate = gpu_stats['cache_hit_rate']
            
        except Exception as e:
            logger.error(f"Error updating component metrics: {e}")

    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check error rates
            error_stats = self.error_handler.get_error_statistics()
            recent_errors = error_stats.get('recent_errors_1h', 0)
            
            if recent_errors > self.config['error_escalation_threshold']:
                logger.warning(f"High error rate detected: {recent_errors} errors in past hour")
            
            # Check GPU health
            gpu_stats = self.gpu_processor.get_performance_statistics()
            if not gpu_stats.get('gpu_available', False):
                logger.warning("GPU not available - running in CPU-only mode")
            
            # Check thermal state
            thermal_zone = gpu_stats.get('thermal_zone', 'unknown')
            if thermal_zone in ['hot', 'critical']:
                logger.warning(f"Thermal warning: {thermal_zone} zone detected")
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")

    def _log_performance_summary(self):
        """Log periodic performance summary"""
        try:
            with self._lock:
                metrics = self.performance_metrics
            
            logger.info(
                f"📊 System Performance: "
                f"Events: {metrics.total_news_events_processed}, "
                f"Correlations: {metrics.total_correlations_calculated}, "
                f"Opportunities: {metrics.total_profit_opportunities_identified}, "
                f"Avg Time: {metrics.avg_processing_time_per_event:.3f}s, "
                f"GPU: {metrics.gpu_utilization_rate:.1%}, "
                f"Recovery: {metrics.error_recovery_success_rate:.1%}, "
                f"Uptime: {metrics.system_uptime/3600:.1f}h"
            )
            
        except Exception as e:
            logger.error(f"Error logging performance summary: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            with self._lock:
                metrics_dict = {
                    'total_news_events_processed': self.performance_metrics.total_news_events_processed,
                    'total_correlations_calculated': self.performance_metrics.total_correlations_calculated,
                    'total_profit_opportunities_identified': self.performance_metrics.total_profit_opportunities_identified,
                    'avg_processing_time_per_event': self.performance_metrics.avg_processing_time_per_event,
                    'gpu_utilization_rate': self.performance_metrics.gpu_utilization_rate,
                    'error_recovery_success_rate': self.performance_metrics.error_recovery_success_rate,
                    'thermal_throttle_incidents': self.performance_metrics.thermal_throttle_incidents,
                    'cache_hit_rate': self.performance_metrics.cache_hit_rate,
                    'profit_prediction_accuracy': self.performance_metrics.profit_prediction_accuracy,
                    'system_uptime': self.performance_metrics.system_uptime
                }
            
            # Add component status
            gpu_stats = self.gpu_processor.get_performance_statistics()
            error_stats = self.error_handler.get_error_statistics()
            
            return {
                'system_running': self.system_running,
                'start_time': self._start_time,
                'performance_metrics': metrics_dict,
                'gpu_processor_status': gpu_stats,
                'error_handler_status': error_stats,
                'processing_queue_size': self.processing_queue.qsize(),
                'active_workers': len(self._processing_workers),
                'result_history_size': len(self.result_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'system_running': self.system_running} 