#!/usr/bin/env python3
"""
Tesseract Monitoring Script
Monitors tesseract pattern processing and system health.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
from scripts.tesseract_control import TesseractController
from schwabot_math.klein_logic import KleinDecayField

class TesseractMonitor:
    """Monitors tesseract pattern processing and system health."""
    
    def __init__(self, config_path: str):
        self.controller = TesseractController(config_path)
        self.config = self.controller.config
        self._setup_logging()
        self.metrics_history = []
        self.alert_history = []
        self.klein = KleinDecayField()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        if log_config.get('enabled', True):
            logging.basicConfig(
                filename=log_config.get('file', 'logs/tesseract.log'),
                level=getattr(logging, log_config.get('level', 'INFO')),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def monitor_patterns(self, pattern_hash: str) -> Dict:
        """Monitor pattern processing and collect metrics."""
        # Process pattern
        result = self.controller.process_pattern(pattern_hash)
        
        # Klein Drift Integration
        pattern_data = np.asarray(result['pattern']) if 'pattern' in result else np.array([0.0])
        drift_vector = self.klein.compute_decay_vector(pattern_data)
        drift_strength = np.linalg.norm(drift_vector)
        drift_entropy = -np.sum(drift_vector * np.log2(drift_vector + 1e-10))
        
        # Collect metrics
        metrics = {
            'timestamp': time.time(),
            'coherence': result['coherence'],
            'homeostasis': result['homeostasis'],
            'stability': result['metrics']['stability'],
            'entropy': drift_entropy,
            'drift_strength': drift_strength,
            'pattern_variance': np.var(pattern_data) if pattern_data.size > 0 else 0.0
        }
        
        # Update history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.config['processing']['pattern_window']:
            self.metrics_history.pop(0)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: Dict):
        """Check metrics against alert thresholds."""
        alert_config = self.config['monitoring']['alerts']
        
        alerts = []
        if metrics['coherence'] < alert_config['coherence_threshold']:
            alerts.append({
                'type': 'coherence',
                'level': 'warning',
                'message': f'Low coherence detected: {metrics["coherence"]:.2f}',
                'timestamp': time.time()
            })
        
        if metrics['homeostasis'] < alert_config['homeostasis_threshold']:
            alerts.append({
                'type': 'homeostasis',
                'level': 'warning',
                'message': f'Low homeostasis detected: {metrics["homeostasis"]:.2f}',
                'timestamp': time.time()
            })
        
        if metrics['stability'] < alert_config['stability_threshold']:
            alerts.append({
                'type': 'stability',
                'level': 'warning',
                'message': f'Low stability detected: {metrics["stability"]:.2f}',
                'timestamp': time.time()
            })
        
        if alerts:
            self.alert_history.extend(alerts)
            for alert in alerts:
                logging.warning(alert['message'])
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        if not self.metrics_history:
            return {
                'status': 'initializing',
                'metrics': {},
                'alerts': []
            }
        
        # Calculate average metrics
        recent_metrics = self.metrics_history[-5:]
        avg_metrics = {
            'coherence': np.mean([m['coherence'] for m in recent_metrics]),
            'homeostasis': np.mean([m['homeostasis'] for m in recent_metrics]),
            'stability': np.mean([m['stability'] for m in recent_metrics]),
            'entropy': np.mean([m['entropy'] for m in recent_metrics]),
            'pattern_variance': np.mean([m['pattern_variance'] for m in recent_metrics])
        }
        
        # Determine system status
        status = 'healthy'
        if avg_metrics['coherence'] < self.config['processing']['coherence_threshold']:
            status = 'degraded'
        if avg_metrics['homeostasis'] < self.config['processing']['homeostasis_threshold']:
            status = 'unstable'
        
        return {
            'status': status,
            'metrics': avg_metrics,
            'alerts': self.alert_history[-5:],
            'last_update': time.time()
        }
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive system report."""
        status = self.get_system_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': status,
            'metrics_summary': {
                'coherence': {
                    'current': status['metrics']['coherence'],
                    'threshold': self.config['processing']['coherence_threshold'],
                    'trend': self._calculate_trend('coherence')
                },
                'homeostasis': {
                    'current': status['metrics']['homeostasis'],
                    'threshold': self.config['processing']['homeostasis_threshold'],
                    'trend': self._calculate_trend('homeostasis')
                },
                'stability': {
                    'current': status['metrics']['stability'],
                    'threshold': self.config['monitoring']['alerts']['stability_threshold'],
                    'trend': self._calculate_trend('stability')
                }
            },
            'alerts': {
                'total': len(self.alert_history),
                'recent': self.alert_history[-5:],
                'by_type': self._count_alerts_by_type()
            }
        }
    
    def _calculate_trend(self, metric: str) -> str:
        """Calculate trend for a specific metric."""
        if len(self.metrics_history) < 2:
            return 'stable'
        
        values = [m[metric] for m in self.metrics_history[-5:]]
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'degrading'
        return 'stable'
    
    def _count_alerts_by_type(self) -> Dict[str, int]:
        """Count alerts by type."""
        counts = {}
        for alert in self.alert_history:
            counts[alert['type']] = counts.get(alert['type'], 0) + 1
        return counts

def main():
    """Main entry point for the monitoring script."""
    config_path = "config/tesseract-config.json"
    monitor = TesseractMonitor(config_path)
    
    # Example usage
    test_hash = "a" * 64  # Example hash
    metrics = monitor.monitor_patterns(test_hash)
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))
    
    report = monitor.generate_report()
    print("\nSystem Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main() 