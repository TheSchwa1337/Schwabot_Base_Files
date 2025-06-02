"""
Basket Log Controller
===================

Implements logging and monitoring for Schwabot's recursive trading intelligence.
Manages log levels, metrics tracking, and system state monitoring.
"""

from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime
import os
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class LogEntry:
    """Container for log entries"""
    timestamp: datetime
    level: str
    message: str
    metrics: Dict[str, Any]
    context: Dict[str, Any]

class BasketLogController:
    """Manages logging and monitoring for basket operations"""
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        max_log_size: int = 1000000,  # 1MB
        backup_count: int = 5,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.log_dir = log_dir
        self.max_log_size = max_log_size
        self.backup_count = backup_count
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger("SchwabotBasket")
        self.logger.setLevel(log_level)
        
        # Create file handler
        log_file = os.path.join(log_dir, "basket.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_log_size,
            backupCount=backup_count
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics storage
        self.metrics_history: List[LogEntry] = []
        self.alert_thresholds = alert_thresholds or {
            'thermal_state': 0.8,
            'entropy_rate': 0.9,
            'memory_coherence': 0.2,
            'trust_score': 0.3
        }

    def log_operation(
        self,
        level: str,
        message: str,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        operation_type: Optional[str] = None,
        user_id: Optional[int] = None,
        transaction_id: Optional[int] = None
    ):
        """Log an operation with associated metrics"""
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            metrics=metrics,
            context={
                'operation_type': operation_type,
                'user_id': user_id,
                'transaction_id': transaction_id,
                **context or {}
            }
        )
        
        # Add to history
        self.metrics_history.append(entry)
        
        # Check for alerts
        self._check_alerts(entry)
        
        # Log message
        log_message = f"{message} | Metrics: {json.dumps(metrics)}"
        if context:
            log_message += f" | Context: {json.dumps(context)}"
            
        if level == "DEBUG":
            self.logger.debug(log_message)
        elif level == "INFO":
            self.logger.info(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        elif level == "ERROR":
            self.logger.error(log_message)
        elif level == "CRITICAL":
            self.logger.critical(log_message)

    def _check_alerts(self, entry: LogEntry):
        """Check metrics against alert thresholds"""
        for metric, threshold in self.alert_thresholds.items():
            if metric in entry.metrics:
                value = entry.metrics[metric]
                if isinstance(value, (int, float)):
                    if value > threshold:
                        self.logger.warning(
                            f"Alert: {metric} exceeds threshold "
                            f"({value:.2f} > {threshold:.2f})"
                        )

    def get_metrics_summary(
        self,
        time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get summary of metrics over time window (in seconds)"""
        if not self.metrics_history:
            return {}
            
        # Filter entries by time window
        if time_window:
            cutoff = datetime.now().timestamp() - time_window
            entries = [
                entry for entry in self.metrics_history
                if entry.timestamp.timestamp() > cutoff
            ]
        else:
            entries = self.metrics_history
            
        if not entries:
            return {}
            
        # Calculate statistics for each metric
        summary = {}
        for entry in entries:
            for metric, value in entry.metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in summary:
                        summary[metric] = []
                    summary[metric].append(value)
                    
        # Calculate statistics
        stats = {}
        for metric, values in summary.items():
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
            
        return stats

    def get_recent_entries(
        self,
        count: int = 10,
        level: Optional[str] = None
    ) -> List[Dict]:
        """Get recent log entries"""
        entries = self.metrics_history[-count:]
        if level:
            entries = [e for e in entries if e.level == level]
        return [asdict(entry) for entry in entries]

    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()

    def set_alert_threshold(
        self,
        metric: str,
        threshold: float
    ):
        """Set alert threshold for a metric"""
        if metric not in self.alert_thresholds:
            raise ValueError(f"Metric '{metric}' is not defined.")
        self.alert_thresholds[metric] = threshold

    def get_alert_thresholds(self) -> Dict[str, float]:
        """Get current alert thresholds"""
        return self.alert_thresholds.copy()

# Example usage
if __name__ == "__main__":
    controller = BasketLogController(
        log_level=logging.DEBUG,
        alert_thresholds={
            'thermal_state': 0.85,
            'entropy_rate': 0.95,
            'memory_coherence': 0.25,
            'trust_score': 0.35
        }
    )
    
    # Log some operations
    controller.log_operation(
        "INFO",
        "Basket swap initiated",
        {
            'thermal_state': 0.6,
            'entropy_rate': 0.7,
            'memory_coherence': 0.8,
            'trust_score': 0.9
        },
        operation_type="swap",
        user_id=12345,
        transaction_id=67890
    )
    
    # Get metrics summary
    summary = controller.get_metrics_summary(time_window=3600)
    print("\nMetrics summary:")
    print(json.dumps(summary, indent=2))
    
    # Get recent entries
    recent = controller.get_recent_entries(count=5)
    print("\nRecent entries:")
    print(json.dumps(recent, indent=2)) 