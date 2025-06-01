"""
Phase State for Schwabot System
Tracks and manages all metric states across the system
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import logging
from pathlib import Path
from .event_bus import EventBus, EventState

@dataclass
class MetricState:
    """State container for a single metric"""
    value: float
    timestamp: float
    source: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class PhaseMetrics:
    """Container for all metrics in a phase"""
    entropy: MetricState
    coherence: MetricState
    velocity: MetricState
    liquidity: MetricState
    tpf_state: MetricState
    panic_zone: MetricState
    strategy_state: MetricState

class PhaseState:
    """Tracks and manages all metric states"""
    
    def __init__(self, event_bus: EventBus, log_dir: str = "logs"):
        self.event_bus = event_bus
        self.metrics: Dict[str, MetricState] = {}
        self.phase_history: List[Dict] = []
        self.max_history = 1000
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / "phase_state.log",
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PhaseState')
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Subscribe to event bus updates
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event bus subscriptions"""
        metric_keys = [
            'entropy_score',
            'coherence_score',
            'velocity_class',
            'liquidity_status',
            'tpf_state',
            'panic_zone',
            'strategy_state'
        ]
        
        for key in metric_keys:
            self.event_bus.subscribe(key, lambda state, k=key: self._handle_metric_update(k, state))
    
    def _handle_metric_update(self, key: str, state: EventState) -> None:
        """
        Handle metric updates from event bus
        
        Args:
            key: Metric key
            state: Event state
        """
        with self._lock:
            # Create metric state
            metric_state = MetricState(
                value=state.value,
                timestamp=state.timestamp,
                source=state.source,
                confidence=state.metadata.get('confidence', 1.0),
                metadata=state.metadata
            )
            
            # Update metric
            self.metrics[key] = metric_state
            
            # Log update
            self.logger.info(f"Metric updated: {key}={state.value} (source: {state.source})")
            
            # Record in history
            self.phase_history.append({
                'key': key,
                'value': state.value,
                'timestamp': state.timestamp,
                'source': state.source,
                'confidence': metric_state.confidence,
                'metadata': state.metadata
            })
            
            # Trim history if needed
            if len(self.phase_history) > self.max_history:
                self.phase_history.pop(0)
    
    def get_metric(self, key: str) -> Optional[MetricState]:
        """
        Get current metric state
        
        Args:
            key: Metric key
            
        Returns:
            MetricState object or None
        """
        with self._lock:
            return self.metrics.get(key)
    
    def get_metric_value(self, key: str, default: Any = None) -> Any:
        """
        Get current metric value
        
        Args:
            key: Metric key
            default: Default value if key not found
            
        Returns:
            Current metric value or default
        """
        with self._lock:
            if key in self.metrics:
                return self.metrics[key].value
            return default
    
    def get_phase_metrics(self) -> PhaseMetrics:
        """
        Get all current phase metrics
        
        Returns:
            PhaseMetrics object with all current metrics
        """
        with self._lock:
            return PhaseMetrics(
                entropy=self.metrics.get('entropy_score', MetricState(0.0, 0.0, "system", 0.0, {})),
                coherence=self.metrics.get('coherence_score', MetricState(0.0, 0.0, "system", 0.0, {})),
                velocity=self.metrics.get('velocity_class', MetricState(0.0, 0.0, "system", 0.0, {})),
                liquidity=self.metrics.get('liquidity_status', MetricState(0.0, 0.0, "system", 0.0, {})),
                tpf_state=self.metrics.get('tpf_state', MetricState(0.0, 0.0, "system", 0.0, {})),
                panic_zone=self.metrics.get('panic_zone', MetricState(False, 0.0, "system", 0.0, {})),
                strategy_state=self.metrics.get('strategy_state', MetricState("", 0.0, "system", 0.0, {}))
            )
    
    def get_metric_history(self, key: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get metric history
        
        Args:
            key: Optional key to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of historical metric entries
        """
        with self._lock:
            if key:
                filtered = [h for h in self.phase_history if h['key'] == key]
                return filtered[-limit:]
            return self.phase_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear metric history"""
        with self._lock:
            self.phase_history.clear()
            self.logger.info("History cleared")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all current metric values
        
        Returns:
            Dictionary of all metric values
        """
        with self._lock:
            return {k: v.value for k, v in self.metrics.items()} 