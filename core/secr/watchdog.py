"""
SECR Watchdog
=============

Monitors patch outcomes, closes failure keys, and feeds learning data
to SchwaFit for continuous improvement of the resolution system.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timezone

from .failure_logger import FailureLogger, FailureKey
from .resolver_matrix import ResolverMatrix, PatchConfig
from .injector import ConfigInjector

logger = logging.getLogger(__name__)

@dataclass
class OutcomeMetrics:
    """Metrics for evaluating patch outcomes"""
    profit_delta: float = 0.0
    latency_improvement: float = 0.0
    error_reduction: float = 0.0
    stability_score: float = 0.0
    resource_efficiency: float = 0.0
    
    def composite_score(self) -> float:
        """Calculate composite outcome score"""
        weights = {
            'profit': 0.4,
            'latency': 0.2,
            'error': 0.2,
            'stability': 0.1,
            'efficiency': 0.1
        }
        
        return (
            weights['profit'] * self.profit_delta +
            weights['latency'] * self.latency_improvement +
            weights['error'] * self.error_reduction +
            weights['stability'] * self.stability_score +
            weights['efficiency'] * self.resource_efficiency
        )

class OutcomeEvaluator:
    """Evaluates the effectiveness of applied patches"""
    
    def __init__(self, baseline_window: int = 100):
        self.baseline_window = baseline_window
        self.baseline_metrics: Dict[str, List[float]] = {
            'profit_per_trade': [],
            'avg_latency': [],
            'error_rate': [],
            'system_stability': [],
            'resource_usage': []
        }
        
    def update_baseline(self, metrics: Dict[str, float]) -> None:
        """Update baseline metrics with new data points"""
        for key, value in metrics.items():
            if key in self.baseline_metrics:
                self.baseline_metrics[key].append(value)
                # Keep only recent window
                if len(self.baseline_metrics[key]) > self.baseline_window:
                    self.baseline_metrics[key] = self.baseline_metrics[key][-self.baseline_window:]
    
    def evaluate_outcome(self, 
                        failure_key: FailureKey, 
                        patch: PatchConfig,
                        post_patch_metrics: Dict[str, float]) -> OutcomeMetrics:
        """
        Evaluate the outcome of a patch application
        
        Args:
            failure_key: Original failure that triggered the patch
            patch: Applied patch configuration  
            post_patch_metrics: System metrics after patch application
            
        Returns:
            OutcomeMetrics with evaluation results
        """
        outcome = OutcomeMetrics()
        
        # Calculate profit delta
        if 'profit_per_trade' in post_patch_metrics and self.baseline_metrics['profit_per_trade']:
            baseline_profit = np.mean(self.baseline_metrics['profit_per_trade'])
            current_profit = post_patch_metrics['profit_per_trade']
            outcome.profit_delta = (current_profit - baseline_profit) / abs(baseline_profit) if baseline_profit != 0 else 0
        
        # Calculate latency improvement
        if 'avg_latency' in post_patch_metrics and self.baseline_metrics['avg_latency']:
            baseline_latency = np.mean(self.baseline_metrics['avg_latency'])
            current_latency = post_patch_metrics['avg_latency']
            outcome.latency_improvement = (baseline_latency - current_latency) / baseline_latency if baseline_latency > 0 else 0
        
        # Calculate error reduction
        if 'error_rate' in post_patch_metrics and self.baseline_metrics['error_rate']:
            baseline_errors = np.mean(self.baseline_metrics['error_rate'])
            current_errors = post_patch_metrics['error_rate']
            outcome.error_reduction = (baseline_errors - current_errors) / baseline_errors if baseline_errors > 0 else 0
        
        # Calculate stability score
        if 'system_stability' in post_patch_metrics:
            outcome.stability_score = post_patch_metrics['system_stability']
        
        # Calculate resource efficiency
        if 'resource_usage' in post_patch_metrics and self.baseline_metrics['resource_usage']:
            baseline_usage = np.mean(self.baseline_metrics['resource_usage'])
            current_usage = post_patch_metrics['resource_usage']
            outcome.resource_efficiency = (baseline_usage - current_usage) / baseline_usage if baseline_usage > 0 else 0
        
        return outcome

class SchwaFitInterface:
    """Interface for feeding data to SchwaFit learning system"""
    
    def __init__(self, queue_size: int = 1000):
        self.training_queue: List[Dict[str, Any]] = []
        self.queue_size = queue_size
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for training data events"""
        self.callbacks.append(callback)
    
    def queue_training_sample(self, 
                             failure_key: FailureKey,
                             patch: PatchConfig, 
                             outcome: OutcomeMetrics,
                             context: Dict[str, Any]) -> None:
        """
        Queue a training sample for SchwaFit
        
        Args:
            failure_key: Original failure
            patch: Applied patch
            outcome: Measured outcome
            context: Additional context data
        """
        sample = {
            'timestamp': time.time(),
            'failure_group': failure_key.group.value,
            'failure_subgroup': failure_key.subgroup.value,
            'failure_severity': failure_key.severity,
            'failure_context': failure_key.ctx,
            'patch_metadata': patch.metadata,
            'patch_priority': patch.priority,
            'patch_persistence': patch.persistence_ticks,
            'outcome_score': outcome.composite_score(),
            'profit_delta': outcome.profit_delta,
            'latency_improvement': outcome.latency_improvement,
            'error_reduction': outcome.error_reduction,
            'stability_score': outcome.stability_score,
            'resource_efficiency': outcome.resource_efficiency,
            'additional_context': context
        }
        
        self.training_queue.append(sample)
        
        # Trim queue if too large
        if len(self.training_queue) > self.queue_size:
            self.training_queue = self.training_queue[-self.queue_size:]
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(sample)
            except Exception as e:
                logger.warning(f"SchwaFit callback error: {e}")
        
        logger.debug(f"Queued training sample: {failure_key.hash} -> score={outcome.composite_score():.3f}")
    
    def get_training_batch(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Get a batch of training samples"""
        if len(self.training_queue) < batch_size:
            return self.training_queue.copy()
        
        # Return most recent samples
        return self.training_queue[-batch_size:].copy()
    
    def clear_queue(self) -> int:
        """Clear training queue and return number of samples cleared"""
        count = len(self.training_queue)
        self.training_queue.clear()
        return count

class SECRWatchdog:
    """Main SECR watchdog system"""
    
    def __init__(self, 
                 failure_logger: FailureLogger,
                 resolver_matrix: ResolverMatrix,
                 config_injector: ConfigInjector,
                 evaluation_window: int = 16,
                 stability_threshold: float = 0.8):
        
        self.failure_logger = failure_logger
        self.resolver_matrix = resolver_matrix
        self.config_injector = config_injector
        self.evaluation_window = evaluation_window
        self.stability_threshold = stability_threshold
        
        # Initialize components
        self.outcome_evaluator = OutcomeEvaluator()
        self.schwafit_interface = SchwaFitInterface()
        
        # Tracking structures
        self.monitored_keys: Dict[str, Tuple[FailureKey, PatchConfig, float, int]] = {}  # hash -> (key, patch, applied_at, ticks_monitored)
        self.evaluation_cache: Dict[str, OutcomeMetrics] = {}
        
        # Statistics
        self.watchdog_stats = {
            'keys_monitored': 0,
            'keys_closed': 0,
            'positive_outcomes': 0,
            'negative_outcomes': 0,
            'avg_outcome_score': 0.0,
            'last_evaluation': None
        }
        
        # Threading
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
    def start_monitoring(self) -> None:
        """Start the watchdog monitoring thread"""
        if self.running:
            logger.warning("Watchdog already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("SECR Watchdog started")
    
    def stop_monitoring(self) -> None:
        """Stop the watchdog monitoring thread"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("SECR Watchdog stopped")
    
    def register_patch_application(self, failure_key: FailureKey, patch: PatchConfig) -> None:
        """Register a patch application for monitoring"""
        with self.lock:
            self.monitored_keys[failure_key.hash] = (failure_key, patch, time.time(), 0)
            self.watchdog_stats['keys_monitored'] += 1
            logger.debug(f"Registered patch for monitoring: {failure_key.hash}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                self._evaluate_monitored_keys()
                self._update_resolver_matrix()
                time.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Error in watchdog monitoring loop: {e}")
                time.sleep(5.0)
    
    def _evaluate_monitored_keys(self) -> None:
        """Evaluate all currently monitored keys"""
        with self.lock:
            keys_to_close = []
            
            for failure_hash, (failure_key, patch, applied_at, ticks_monitored) in self.monitored_keys.items():
                ticks_monitored += 1
                self.monitored_keys[failure_hash] = (failure_key, patch, applied_at, ticks_monitored)
                
                # Check if we should evaluate this key
                should_evaluate = (
                    ticks_monitored >= self.evaluation_window or
                    ticks_monitored >= patch.persistence_ticks or
                    time.time() - applied_at > 120  # Force evaluation after 2 minutes
                )
                
                if should_evaluate:
                    try:
                        outcome = self._evaluate_patch_outcome(failure_key, patch, applied_at)
                        self._close_failure_key(failure_key, patch, outcome)
                        keys_to_close.append(failure_hash)
                        
                    except Exception as e:
                        logger.error(f"Error evaluating key {failure_hash}: {e}")
            
            # Remove closed keys from monitoring
            for failure_hash in keys_to_close:
                del self.monitored_keys[failure_hash]
    
    def _evaluate_patch_outcome(self, 
                               failure_key: FailureKey, 
                               patch: PatchConfig, 
                               applied_at: float) -> OutcomeMetrics:
        """Evaluate the outcome of a patch application"""
        
        # Gather current system metrics
        current_metrics = self._gather_system_metrics()
        
        # Evaluate outcome
        outcome = self.outcome_evaluator.evaluate_outcome(failure_key, patch, current_metrics)
        
        # Cache the evaluation
        self.evaluation_cache[failure_key.hash] = outcome
        
        # Update baseline with current metrics
        self.outcome_evaluator.update_baseline(current_metrics)
        
        return outcome
    
    def _gather_system_metrics(self) -> Dict[str, float]:
        """Gather current system performance metrics"""
        # This would interface with actual system monitoring
        # For now, return simulated metrics
        
        try:
            # Get config injector stats
            injector_stats = self.config_injector.get_injection_stats()
            
            # Get resolver matrix stats  
            resolver_stats = self.resolver_matrix.get_patch_stats()
            
            # Simulated metrics - in real implementation, these would come from
            # actual system monitoring, trading performance, etc.
            metrics = {
                'profit_per_trade': np.random.normal(0.02, 0.01),  # 2% avg with 1% std
                'avg_latency': np.random.normal(50, 10),  # 50ms avg
                'error_rate': max(0, np.random.normal(0.05, 0.02)),  # 5% error rate
                'system_stability': min(1.0, max(0, np.random.normal(0.9, 0.1))),
                'resource_usage': min(1.0, max(0, np.random.normal(0.6, 0.1))),
                'config_success_rate': injector_stats.get('success_rate', 100) / 100,
                'patch_effectiveness': len(resolver_stats.get('resolver_counts', {}))
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error gathering system metrics: {e}")
            return {
                'profit_per_trade': 0.0,
                'avg_latency': 100.0,
                'error_rate': 0.1,
                'system_stability': 0.5,
                'resource_usage': 0.8
            }
    
    def _close_failure_key(self, 
                          failure_key: FailureKey, 
                          patch: PatchConfig, 
                          outcome: OutcomeMetrics) -> None:
        """Close a failure key with outcome evaluation"""
        
        outcome_score = outcome.composite_score()
        
        # Close the key in failure logger
        success = self.failure_logger.close_key(
            failure_key.hash,
            outcome_score,
            outcome_score  # Use composite score as both outcome and score
        )
        
        if success:
            self.watchdog_stats['keys_closed'] += 1
            
            if outcome_score > 0:
                self.watchdog_stats['positive_outcomes'] += 1
            else:
                self.watchdog_stats['negative_outcomes'] += 1
            
            # Update running average
            total_outcomes = self.watchdog_stats['positive_outcomes'] + self.watchdog_stats['negative_outcomes']
            if total_outcomes > 0:
                current_avg = self.watchdog_stats['avg_outcome_score']
                self.watchdog_stats['avg_outcome_score'] = (
                    (current_avg * (total_outcomes - 1) + outcome_score) / total_outcomes
                )
            
            self.watchdog_stats['last_evaluation'] = time.time()
            
            # Queue for SchwaFit training
            self.schwafit_interface.queue_training_sample(
                failure_key,
                patch,
                outcome,
                {
                    'system_state': 'live',
                    'evaluation_window': self.evaluation_window,
                    'patch_persistence': patch.persistence_ticks
                }
            )
            
            logger.info(f"Closed failure key {failure_key.hash} with score {outcome_score:.3f}")
        
        else:
            logger.warning(f"Failed to close failure key {failure_key.hash}")
    
    def _update_resolver_matrix(self) -> None:
        """Update resolver matrix with tick progression"""
        try:
            self.resolver_matrix.tick_update()
        except Exception as e:
            logger.error(f"Error updating resolver matrix: {e}")
    
    def force_evaluate_key(self, failure_hash: str) -> Optional[float]:
        """Force evaluation of a specific failure key"""
        with self.lock:
            if failure_hash not in self.monitored_keys:
                logger.warning(f"Key {failure_hash} not being monitored")
                return None
            
            failure_key, patch, applied_at, _ = self.monitored_keys[failure_hash]
            
            try:
                outcome = self._evaluate_patch_outcome(failure_key, patch, applied_at)
                self._close_failure_key(failure_key, patch, outcome)
                
                # Remove from monitoring
                del self.monitored_keys[failure_hash]
                
                return outcome.composite_score()
                
            except Exception as e:
                logger.error(f"Error force evaluating key {failure_hash}: {e}")
                return None
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        with self.lock:
            active_keys = []
            for failure_hash, (failure_key, patch, applied_at, ticks) in self.monitored_keys.items():
                active_keys.append({
                    'hash': failure_hash,
                    'group': failure_key.group.value,
                    'subgroup': failure_key.subgroup.value,
                    'severity': failure_key.severity,
                    'applied_at': applied_at,
                    'ticks_monitored': ticks,
                    'patch_priority': patch.priority
                })
            
            return {
                'running': self.running,
                'active_monitoring_count': len(self.monitored_keys),
                'active_keys': active_keys,
                'training_queue_size': len(self.schwafit_interface.training_queue),
                'evaluation_cache_size': len(self.evaluation_cache),
                **self.watchdog_stats
            }
    
    def get_schwafit_training_data(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Get training data for SchwaFit"""
        return self.schwafit_interface.get_training_batch(batch_size)
    
    def register_schwafit_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for SchwaFit training events"""
        self.schwafit_interface.register_callback(callback)
    
    def get_outcome_summary(self) -> Dict[str, Any]:
        """Get summary of outcomes"""
        if not self.evaluation_cache:
            return {}
        
        outcomes = list(self.evaluation_cache.values())
        scores = [outcome.composite_score() for outcome in outcomes]
        
        return {
            'total_evaluations': len(outcomes),
            'avg_score': np.mean(scores),
            'median_score': np.median(scores),
            'score_std': np.std(scores),
            'positive_ratio': sum(1 for score in scores if score > 0) / len(scores),
            'score_distribution': {
                'excellent': sum(1 for score in scores if score > 0.5),
                'good': sum(1 for score in scores if 0.1 < score <= 0.5),
                'neutral': sum(1 for score in scores if -0.1 <= score <= 0.1),
                'poor': sum(1 for score in scores if -0.5 <= score < -0.1),
                'terrible': sum(1 for score in scores if score < -0.5)
            }
        } 