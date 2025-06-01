"""
Real-time Risk Monitoring System
Provides real-time monitoring of risk metrics and alerting capabilities.
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import time
import json
from enum import Enum
import numpy as np

from .risk_manager import RiskManager, RiskLevel
from config.risk_config import MONITORING_CONFIG, DEFAULT_RISK_CONFIG

class AlertLevel(Enum):
    INFO = 1
    WARNING = 2
    CRITICAL = 3

@dataclass
class RiskAlert:
    timestamp: datetime
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float

class RiskMonitor:
    """
    Real-time risk monitoring system with alerting capabilities.
    """
    
    def __init__(self,
                 risk_manager: RiskManager,
                 alert_callbacks: Optional[Dict[AlertLevel, List[Callable]]] = None):
        """
        Initialize the risk monitor.
        
        Args:
            risk_manager: Instance of RiskManager
            alert_callbacks: Dictionary of alert level to callback functions
        """
        self.risk_manager = risk_manager
        self.alert_callbacks = alert_callbacks or {}
        self.monitoring_thread = None
        self.running = False
        self.alert_history: List[RiskAlert] = []
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Load monitoring configuration
        self.config = MONITORING_CONFIG
        self.alert_thresholds = DEFAULT_RISK_CONFIG['alert_thresholds']
        
    def start_monitoring(self):
        """Start the risk monitoring thread"""
        if self.running:
            return
            
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop the risk monitoring thread"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Get current risk metrics
                risk_report = self.risk_manager.get_risk_report()
                
                # Check risk metrics against thresholds
                self._check_risk_metrics(risk_report)
                
                # Generate periodic report
                if self._should_generate_report():
                    self._generate_risk_report(risk_report)
                
                # Sleep until next update
                time.sleep(self.config['report_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(1)  # Sleep briefly before retrying
                
    def _check_risk_metrics(self, risk_report: Dict):
        """Check risk metrics against thresholds and generate alerts"""
        # Check drawdown
        if risk_report['current_drawdown'] > self.alert_thresholds['drawdown']:
            self._generate_alert(
                AlertLevel.WARNING,
                "High drawdown detected",
                "current_drawdown",
                risk_report['current_drawdown'],
                self.alert_thresholds['drawdown']
            )
            
        # Check volatility
        if risk_report['portfolio_volatility'] > self.alert_thresholds['volatility']:
            self._generate_alert(
                AlertLevel.WARNING,
                "High portfolio volatility",
                "portfolio_volatility",
                risk_report['portfolio_volatility'],
                self.alert_thresholds['volatility']
            )
            
        # Check exposure
        if risk_report['total_exposure'] > self.alert_thresholds['exposure']:
            self._generate_alert(
                AlertLevel.WARNING,
                "High portfolio exposure",
                "total_exposure",
                risk_report['total_exposure'],
                self.alert_thresholds['exposure']
            )
            
        # Check risk level
        if risk_report['risk_level'] == RiskLevel.EXTREME.name:
            self._generate_alert(
                AlertLevel.CRITICAL,
                "Extreme risk level detected",
                "risk_level",
                1.0,
                0.0
            )
            
    def _generate_alert(self,
                       level: AlertLevel,
                       message: str,
                       metric: str,
                       value: float,
                       threshold: float):
        """Generate and dispatch a risk alert"""
        alert = RiskAlert(
            timestamp=datetime.now(),
            level=level,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold
        )
        
        # Add to alert history
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(f"Risk Alert: {message} (Value: {value:.2f}, Threshold: {threshold:.2f})")
        
        # Call alert callbacks
        if level in self.alert_callbacks:
            for callback in self.alert_callbacks[level]:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")
                    
    def _should_generate_report(self) -> bool:
        """Check if it's time to generate a periodic report"""
        current_time = time.time()
        return (current_time % self.config['report_interval']) < 1.0
        
    def _generate_risk_report(self, risk_report: Dict):
        """Generate and log a periodic risk report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                metric: risk_report.get(metric, None)
                for metric in self.config['metrics_to_track']
            },
            'alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'level': alert.level.name,
                    'message': alert.message,
                    'metric': alert.metric,
                    'value': alert.value,
                    'threshold': alert.threshold
                }
                for alert in self.alert_history[-10:]  # Last 10 alerts
            ]
        }
        
        # Log report
        self.logger.info(f"Risk Report: {json.dumps(report, indent=2)}")
        
    def get_alert_history(self, level: Optional[AlertLevel] = None) -> List[RiskAlert]:
        """Get alert history, optionally filtered by level"""
        if level is None:
            return self.alert_history
        return [alert for alert in self.alert_history if alert.level == level]
        
    def register_alert_callback(self, level: AlertLevel, callback: Callable):
        """Register a callback function for a specific alert level"""
        if level not in self.alert_callbacks:
            self.alert_callbacks[level] = []
        self.alert_callbacks[level].append(callback)
        
    def clear_alert_history(self):
        """Clear the alert history"""
        self.alert_history = [] 