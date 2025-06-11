"""
Phase Metrics Engine
=================

Calculates real-time metrics for phase classification and monitoring.
Integrates with GPU acceleration for high-performance computations.
"""

import numpy as np
import cupy as cp
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PhaseMetricsEngine:
    """Calculates and manages phase metrics"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the metrics engine"""
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "schema.yaml"
        self._load_config()
        self._initialize_gpu()
        
    def _load_config(self) -> None:
        """Load metrics configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            metrics_config = config.get("metrics", {})
            self.window_sizes = metrics_config.get("window_sizes", {
                "short": 100,
                "medium": 500,
                "long": 1000
            })
            self.update_interval = metrics_config.get("update_interval_ms", 1000)
            self.use_gpu = metrics_config.get("gpu_acceleration", True)
            self.drift_config = metrics_config.get("drift_detection", {
                "window_size": 1000,
                "threshold": 0.7
            })
            self.entropy_config = metrics_config.get("entropy", {
                "bins": 50,
                "min_probability": 1e-10
            })
            
        except Exception as e:
            logger.error(f"Error loading metrics config: {e}")
            raise
            
    def _initialize_gpu(self) -> None:
        """Initialize GPU resources"""
        if self.use_gpu:
            try:
                # Test GPU availability
                cp.array([1, 2, 3])
                logger.info("GPU acceleration enabled")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                self.use_gpu = False
                
    def compute_metrics(self, price_series: np.ndarray, volume_series: np.ndarray) -> Dict[str, float]:
        """Compute all phase metrics from price and volume data"""
        try:
            # Convert to GPU arrays if enabled
            if self.use_gpu:
                price_series = cp.asarray(price_series)
                volume_series = cp.asarray(volume_series)
                
            metrics = {
                "profit_trend": self._compute_profit_trend(price_series),
                "stability": self._compute_stability(price_series),
                "memory_coherence": self._compute_memory_coherence(price_series),
                "paradox_pressure": self._compute_paradox_pressure(price_series),
                "entropy_rate": self._compute_entropy_rate(price_series),
                "thermal_state": self._compute_thermal_state(price_series, volume_series),
                "bit_depth": self._compute_bit_depth(price_series),
                "trust_score": self._compute_trust_score(price_series, volume_series)
            }
            
            # Convert back to CPU if using GPU
            if self.use_gpu:
                metrics = {k: float(v) for k, v in metrics.items()}
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            raise
            
    def _compute_profit_trend(self, prices: np.ndarray) -> float:
        """Compute profit trend metric"""
        returns = np.diff(prices) / prices[:-1]
        return float(np.mean(returns))
        
    def _compute_stability(self, prices: np.ndarray) -> float:
        """Compute price stability metric"""
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        return float(1.0 / (1.0 + volatility))
        
    def _compute_memory_coherence(self, prices: np.ndarray) -> float:
        """Compute memory coherence metric"""
        # Use autocorrelation of returns
        returns = np.diff(prices) / prices[:-1]
        autocorr = np.correlate(returns, returns, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        return float(np.mean(autocorr[:self.window_sizes["short"]]))
        
    def _compute_paradox_pressure(self, prices: np.ndarray) -> float:
        """Compute paradox pressure metric"""
        # Measure divergence between short and long-term trends
        short_ma = np.mean(prices[-self.window_sizes["short"]:])
        long_ma = np.mean(prices[-self.window_sizes["long"]:])
        return float(abs(short_ma - long_ma) / long_ma)
        
    def _compute_entropy_rate(self, prices: np.ndarray) -> float:
        """Compute entropy rate metric"""
        returns = np.diff(prices) / prices[:-1]
        hist, _ = np.histogram(returns, bins=self.entropy_config["bins"])
        probs = hist / np.sum(hist)
        probs = np.maximum(probs, self.entropy_config["min_probability"])
        return float(-np.sum(probs * np.log2(probs)))
        
    def _compute_thermal_state(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Compute thermal state metric"""
        # Combine price volatility and volume intensity
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        volume_intensity = np.mean(volumes[-self.window_sizes["short"]:]) / np.mean(volumes)
        return float(volatility * volume_intensity)
        
    def _compute_bit_depth(self, prices: np.ndarray) -> int:
        """Compute effective bit depth of price series"""
        # Calculate number of unique price levels
        unique_prices = np.unique(prices)
        return int(np.ceil(np.log2(len(unique_prices))))
        
    def _compute_trust_score(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Compute trust score metric"""
        # Combine multiple factors
        stability = self._compute_stability(prices)
        coherence = self._compute_memory_coherence(prices)
        volume_consistency = np.std(volumes) / np.mean(volumes)
        
        return float(
            0.4 * stability +
            0.4 * coherence +
            0.2 * (1.0 / (1.0 + volume_consistency))
        )
        
    def get_metric_ranges(self) -> Dict[str, List[float]]:
        """Get valid ranges for all metrics"""
        return {
            "profit_trend": [-np.inf, np.inf],
            "stability": [0.0, 1.0],
            "memory_coherence": [0.0, 1.0],
            "paradox_pressure": [0.0, np.inf],
            "entropy_rate": [0.0, np.inf],
            "thermal_state": [0.0, 1.0],
            "bit_depth": [1, 256],
            "trust_score": [0.0, 1.0]
        }
        
    def validate_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Validate metric values against ranges"""
        errors = []
        ranges = self.get_metric_ranges()
        
        for metric, value in metrics.items():
            if metric not in ranges:
                errors.append(f"Unknown metric: {metric}")
                continue
                
            min_val, max_val = ranges[metric]
            if not (min_val <= value <= max_val):
                errors.append(f"Metric {metric}={value} outside range [{min_val}, {max_val}]")
                
        return errors
        
    def get_metric_statistics(self, metrics_history: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Get statistics for each metric"""
        stats = {}
        
        for metric in self.get_metric_ranges().keys():
            values = [m[metric] for m in metrics_history if metric in m]
            if values:
                stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
                
        return stats 