#!/usr/bin/env python3
"""Temporal execution correction layer for trade timing optimization.

This module implements temporal correction logic for execution timing,
handling delays, latency compensation, and temporal drift correction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TemporalExecutionCorrector",
    "correct_execution_timing",
    "calculate_temporal_drift",
]


@dataclass(slots=True)
class TemporalExecutionCorrector:
    """Temporal execution correction with latency compensation."""
    
    base_latency: float = 0.001  # 1ms base latency
    drift_threshold: float = 0.01  # 10ms drift threshold
    correction_factor: float = 1.2  # Correction amplification
    history_window: int = 100  # Historical data window size
    
    def __post_init__(self) -> None:
        """Initialize correction layer state."""
        self.execution_history: List[Dict[str, Any]] = []
        self.drift_history: List[float] = []
        self.last_correction_time: float = 0.0
    
    def correct_execution_timing(
        self,
        intended_time: float,
        actual_time: float,
        execution_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply temporal correction to execution timing.
        
        Parameters
        ----------
        intended_time
            Originally intended execution timestamp
        actual_time
            Actual execution timestamp
        execution_data
            Execution packet data
            
        Returns
        -------
        Dict[str, Any]
            Corrected execution data with temporal adjustments
        """
        current_time = time.time()
        
        # Calculate temporal drift
        drift = self._calculate_drift(intended_time, actual_time)
        
        # Apply correction if drift exceeds threshold
        correction_applied = False
        if abs(drift) > self.drift_threshold:
            corrected_data = self._apply_correction(
                execution_data, drift, current_time
            )
            correction_applied = True
        else:
            corrected_data = execution_data.copy()
        
        # Update history
        self._update_history(drift, execution_data, correction_applied)
        
        # Add correction metadata
        corrected_data.update({
            'temporal_correction': {
                'drift_detected': drift,
                'correction_applied': correction_applied,
                'correction_timestamp': current_time,
                'latency_estimate': self._estimate_latency(),
                'drift_trend': self._analyze_drift_trend(),
            }
        })
        
        return corrected_data
    
    def _calculate_drift(self, intended: float, actual: float) -> float:
        """Calculate temporal drift between intended and actual timing."""
        base_drift = actual - intended
        
        # Account for system latency
        latency_adjusted_drift = base_drift - self.base_latency
        
        return latency_adjusted_drift
    
    def _apply_correction(
        self, 
        data: Dict[str, Any], 
        drift: float, 
        current_time: float
    ) -> Dict[str, Any]:
        """Apply temporal correction to execution data."""
        corrected = data.copy()
        
        # Adjust price based on drift (simple linear model)
        if 'price' in corrected:
            price_adjustment = drift * self.correction_factor * 0.001
            corrected['price'] = float(corrected['price']) + price_adjustment
        
        # Adjust volume based on drift magnitude
        if 'volume' in corrected:
            volume_factor = 1.0 - (abs(drift) * 0.01)  # Reduce volume for high drift
            corrected['volume'] = float(corrected['volume']) * max(0.1, volume_factor)
        
        # Add correction timestamp
        corrected['corrected_at'] = current_time
        corrected['drift_correction'] = -drift * self.correction_factor
        
        self.last_correction_time = current_time
        
        return corrected
    
    def _estimate_latency(self) -> float:
        """Estimate current system latency from execution history."""
        if len(self.execution_history) < 2:
            return self.base_latency
        
        # Calculate average latency from recent executions
        recent_latencies = []
        for record in self.execution_history[-10:]:  # Last 10 executions
            if 'latency' in record:
                recent_latencies.append(record['latency'])
        
        if recent_latencies:
            return float(np.mean(recent_latencies))
        
        return self.base_latency
    
    def _analyze_drift_trend(self) -> str:
        """Analyze drift trend from historical data."""
        if len(self.drift_history) < 3:
            return "insufficient_data"
        
        recent_drifts = self.drift_history[-5:]  # Last 5 measurements
        
        # Simple trend analysis
        if len(recent_drifts) >= 3:
            trend = np.polyfit(range(len(recent_drifts)), recent_drifts, 1)[0]
            
            if trend > 0.001:
                return "increasing"
            elif trend < -0.001:
                return "decreasing"
            else:
                return "stable"
        
        return "stable"
    
    def _update_history(
        self, 
        drift: float, 
        execution_data: Dict[str, Any], 
        correction_applied: bool
    ) -> None:
        """Update execution and drift history."""
        # Update drift history
        self.drift_history.append(drift)
        if len(self.drift_history) > self.history_window:
            self.drift_history.pop(0)
        
        # Update execution history
        history_record = {
            'timestamp': time.time(),
            'drift': drift,
            'correction_applied': correction_applied,
            'execution_size': execution_data.get('volume', 0.0),
            'latency': abs(drift) if abs(drift) > 0 else self.base_latency,
        }
        
        self.execution_history.append(history_record)
        if len(self.execution_history) > self.history_window:
            self.execution_history.pop(0)
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """Get temporal correction statistics."""
        if not self.drift_history:
            return {'status': 'no_data'}
        
        drift_array = np.array(self.drift_history)
        
        return {
            'mean_drift': float(np.mean(drift_array)),
            'drift_std': float(np.std(drift_array)),
            'max_drift': float(np.max(np.abs(drift_array))),
            'correction_rate': sum(
                1 for record in self.execution_history 
                if record.get('correction_applied', False)
            ) / len(self.execution_history) if self.execution_history else 0.0,
            'average_latency': self._estimate_latency(),
            'drift_trend': self._analyze_drift_trend(),
            'total_executions': len(self.execution_history),
        }


def correct_execution_timing(
    intended_time: float,
    actual_time: float,
    execution_data: Dict[str, Any],
    base_latency: float = 0.001,
) -> Dict[str, Any]:
    """Correct execution timing (functional interface).
    
    Parameters
    ----------
    intended_time
        Originally intended execution timestamp
    actual_time
        Actual execution timestamp
    execution_data
        Execution data to correct
    base_latency
        Base system latency estimate
        
    Returns
    -------
    Dict[str, Any]
        Temporally corrected execution data
    """
    corrector = TemporalExecutionCorrector(base_latency=base_latency)
    return corrector.correct_execution_timing(
        intended_time, actual_time, execution_data
    )


def calculate_temporal_drift(
    intended_times: List[float],
    actual_times: List[float],
) -> Tuple[float, float, List[float]]:
    """Calculate temporal drift statistics.
    
    Parameters
    ----------
    intended_times
        List of intended execution times
    actual_times
        List of actual execution times
        
    Returns
    -------
    Tuple[float, float, List[float]]
        Mean drift, drift standard deviation, and individual drifts
    """
    if len(intended_times) != len(actual_times):
        raise ValueError("Intended and actual times must have same length")
    
    if not intended_times:
        return 0.0, 0.0, []
    
    drifts = [actual - intended for intended, actual in zip(intended_times, actual_times)]
    drift_array = np.array(drifts)
    
    mean_drift = float(np.mean(drift_array))
    drift_std = float(np.std(drift_array))
    
    return mean_drift, drift_std, drifts


if __name__ == "__main__":
    # Example usage
    corrector = TemporalExecutionCorrector()
    
    # Simulate execution with drift
    intended_time = time.time()
    actual_time = intended_time + 0.015  # 15ms delay
    
    execution_data = {
        'price': 50000.0,
        'volume': 0.1,
        'symbol': 'BTC/USDT',
        'side': 'buy'
    }
    
    corrected = corrector.correct_execution_timing(
        intended_time, actual_time, execution_data
    )
    
    print(f"Correction applied: {corrected}")
    print(f"Stats: {corrector.get_correction_stats()}") 