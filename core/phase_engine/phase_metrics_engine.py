"""
Phase Metrics Engine
=================

Calculates real-time metrics for phase classification and monitoring.
Integrates with GPU acceleration for high-performance computations.
"""

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import yaml
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
        """Load metrics configuration from YAML or JSON"""
        try:
            config = None
            
            # Try to load from specified path
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
            
            # Extract metrics config with proper defaults
            if config:
                metrics_config = config.get("metrics", {})
            else:
                metrics_config = {}
                
            self.window_sizes = metrics_config.get("window_sizes", {
                "short": 100,
                "medium": 500,
                "long": 1000
            })
            self.update_interval = metrics_config.get("update_interval_ms", 1000)
            self.use_gpu = metrics_config.get("gpu_acceleration", True) and GPU_AVAILABLE
            self.drift_config = metrics_config.get("drift_detection", {
                "window_size": 1000,
                "threshold": 0.7
            })
            self.entropy_config = metrics_config.get("entropy", {
                "bins": 50,
                "min_probability": 1e-10
            })
            
        except Exception as e:
            logger.warning(f"Error loading metrics config: {e}, using defaults")
            # Set all defaults
            self.window_sizes = {"short": 100, "medium": 500, "long": 1000}
            self.update_interval = 1000
            self.use_gpu = False
            self.drift_config = {"window_size": 1000, "threshold": 0.7}
            self.entropy_config = {"bins": 50, "min_probability": 1e-10}
            
    def _initialize_gpu(self) -> None:
        """Initialize GPU resources"""
        if self.use_gpu and GPU_AVAILABLE:
            try:
                # Test GPU availability
                cp.array([1, 2, 3])
                logger.info("GPU acceleration enabled")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                self.use_gpu = False
        else:
            self.use_gpu = False
                
    def compute_metrics(self, price_series: np.ndarray, volume_series: np.ndarray) -> Dict[str, float]:
        """Compute all phase metrics from price and volume data"""
        try:
            # Convert to appropriate arrays
            if self.use_gpu:
                price_series = cp.asarray(price_series)
                volume_series = cp.asarray(volume_series)
            else:
                price_series = np.asarray(price_series)
                volume_series = np.asarray(volume_series)
                
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
            
            # Convert back to Python floats
            if self.use_gpu:
                metrics = {k: float(cp.asnumpy(v)) if hasattr(v, 'get') else float(v) for k, v in metrics.items()}
            else:
                metrics = {k: float(v) for k, v in metrics.items()}
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            raise
    
    def _compute_profit_trend(self, prices) -> float:
        """Compute profit trend metric"""
        if len(prices) < 2:
            return 0.0
        if self.use_gpu:
            returns = cp.diff(prices) / prices[:-1]
            return float(cp.mean(returns))
        else:
            returns = np.diff(prices) / prices[:-1]
            return float(np.mean(returns))
    
    def _compute_stability(self, prices) -> float:
        """Compute stability metric (inverse of volatility)"""
        if len(prices) < 2:
            return 1.0
        if self.use_gpu:
            returns = cp.diff(prices) / prices[:-1]
            volatility = cp.std(returns)
            return float(1.0 / (1.0 + volatility))
        else:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            return float(1.0 / (1.0 + volatility))
    
    def _compute_memory_coherence(self, prices) -> float:
        """Compute memory coherence metric"""
        if len(prices) < 3:
            return 0.5
        # Simple autocorrelation with lag 1
        if self.use_gpu:
            correlation = cp.corrcoef(prices[:-1], prices[1:])[0, 1]
            return float(cp.abs(correlation))
        else:
            correlation = np.corrcoef(prices[:-1], prices[1:])[0, 1]
            return float(abs(correlation)) if not np.isnan(correlation) else 0.5
    
    def _compute_paradox_pressure(self, prices) -> float:
        """Compute paradox pressure metric"""
        if len(prices) < 3:
            return 0.0
        # Measure of price reversal tendency
        if self.use_gpu:
            returns = cp.diff(prices)
            reversals = cp.sum(returns[:-1] * returns[1:] < 0)
            return float(reversals / len(returns[1:]))
        else:
            returns = np.diff(prices)
            reversals = np.sum(returns[:-1] * returns[1:] < 0)
            return float(reversals / len(returns[1:]))
    
    def _compute_entropy_rate(self, prices) -> float:
        """Compute entropy rate metric"""
        if len(prices) < 2:
            return 0.0
        # Shannon entropy of price returns
        if self.use_gpu:
            returns = cp.diff(prices) / prices[:-1]
            hist, _ = cp.histogram(returns, bins=self.entropy_config["bins"])
            hist = hist.astype(cp.float64) + float(self.entropy_config["min_probability"])  # Avoid log(0)
            prob = hist / cp.sum(hist)
            entropy = -cp.sum(prob * cp.log(prob))
            return float(entropy)
        else:
            returns = np.diff(prices) / prices[:-1]
            hist, _ = np.histogram(returns, bins=self.entropy_config["bins"])
            hist = hist.astype(np.float64) + float(self.entropy_config["min_probability"])  # Avoid log(0)
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log(prob))
            return float(entropy)
    
    def _compute_thermal_state(self, prices, volumes) -> float:
        """Compute thermal state metric"""
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0
        # Volume-weighted price variance
        if self.use_gpu:
            price_changes = cp.diff(prices)
            volume_weights = volumes[1:] / cp.sum(volumes[1:])
            weighted_variance = cp.sum(volume_weights * price_changes**2)
            return float(weighted_variance)
        else:
            price_changes = np.diff(prices)
            volume_weights = volumes[1:] / np.sum(volumes[1:])
            weighted_variance = np.sum(volume_weights * price_changes**2)
            return float(weighted_variance)
    
    def _compute_bit_depth(self, prices) -> float:
        """Compute bit depth metric"""
        if len(prices) < 2:
            return 32.0
        # Measure of price precision needed
        if self.use_gpu:
            price_changes = cp.abs(cp.diff(prices))
            min_change = cp.min(price_changes[price_changes > 0]) if cp.any(price_changes > 0) else 1.0
            depth = cp.log2(cp.max(prices) / min_change)
            return float(cp.clip(depth, 8, 256))
        else:
            price_changes = np.abs(np.diff(prices))
            min_change = np.min(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 1.0
            depth = np.log2(np.max(prices) / min_change)
            return float(np.clip(depth, 8, 256))
    
    def _compute_trust_score(self, prices, volumes) -> float:
        """Compute trust score metric"""
        if len(prices) < 2 or len(volumes) < 2:
            return 0.5
        # Consistency between price moves and volume
        if self.use_gpu:
            price_changes = cp.abs(cp.diff(prices))
            volume_changes = cp.abs(cp.diff(volumes))
            if cp.std(price_changes) > 0 and cp.std(volume_changes) > 0:
                correlation = cp.corrcoef(price_changes, volume_changes)[0, 1]
                return float((1.0 + correlation) / 2.0)
            else:
                return 0.5
        else:
            price_changes = np.abs(np.diff(prices))
            volume_changes = np.abs(np.diff(volumes))
            if np.std(price_changes) > 0 and np.std(volume_changes) > 0:
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                return float((1.0 + correlation) / 2.0) if not np.isnan(correlation) else 0.5
            else:
                return 0.5
        
    def validate_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Validate metric values against expected ranges"""
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
        
    def get_metric_ranges(self) -> Dict[str, tuple]:
        """Get expected ranges for each metric"""
        return {
            "profit_trend": (-1.0, 1.0),
            "stability": (0.0, 1.0),
            "memory_coherence": (0.0, 1.0),
            "paradox_pressure": (0.0, 1.0),
            "entropy_rate": (0.0, 10.0),
            "thermal_state": (0.0, float('inf')),
            "bit_depth": (8.0, 256.0),
            "trust_score": (0.0, 1.0)
        }
        
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