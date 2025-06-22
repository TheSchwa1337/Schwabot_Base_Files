#!/usr/bin/env python3
"""Multi-bit BTC processor for enhanced trading precision.

This module implements multi-bit precision Bitcoin processing logic
for high-frequency trading operations with improved accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "MultiBitBtcProcessor",
    "process_multi_bit_signals",
    "optimize_bit_precision",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MultiBitBtcProcessor:
    """Multi-bit Bitcoin processor with precision optimization."""
    
    bit_precision: int = 64
    processing_threshold: float = 0.001
    optimization_enabled: bool = True
    
    def process_signals(
        self,
        price_data: List[float],
        volume_data: List[float],
        timestamp_data: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Process multi-bit BTC signals with enhanced precision.
        
        Parameters
        ----------
        price_data
            Bitcoin price time series data
        volume_data
            Trading volume time series data
        timestamp_data
            Optional timestamp data for temporal analysis
            
        Returns
        -------
        Dict[str, Any]
            Processed signal results with multi-bit precision
        """
        if len(price_data) != len(volume_data):
            raise ValueError("Price and volume data must have same length")
            
        # Convert to high-precision arrays
        prices = np.array(price_data, dtype=np.float64)
        volumes = np.array(volume_data, dtype=np.float64)
        
        # Apply multi-bit processing
        processed_signals = self._apply_multi_bit_transform(prices, volumes)
        
        # Optimize precision if enabled
        if self.optimization_enabled:
            processed_signals = self._optimize_precision(processed_signals)
            
        return {
            'processed_signals': processed_signals.tolist(),
            'bit_precision': self.bit_precision,
            'signal_strength': float(np.mean(np.abs(processed_signals))),
            'processing_quality': self._assess_quality(processed_signals),
            'status': 'success'
        }
    
    def _apply_multi_bit_transform(
        self, 
        prices: np.ndarray, 
        volumes: np.ndarray
    ) -> np.ndarray:
        """Apply multi-bit transformation to price/volume data."""
        # Normalize inputs
        price_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-10)
        volume_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-10)
        
        # Multi-bit weighted combination
        bit_weights = np.linspace(0.1, 1.0, len(prices))
        transformed = bit_weights * price_norm + (1 - bit_weights) * volume_norm
        
        return transformed
    
    def _optimize_precision(self, signals: np.ndarray) -> np.ndarray:
        """Optimize signal precision using adaptive filtering."""
        # Apply adaptive precision optimization
        optimized = signals.copy()
        
        # Remove noise below threshold
        noise_mask = np.abs(optimized) < self.processing_threshold
        optimized[noise_mask] *= 0.1
        
        # Enhance strong signals
        strong_mask = np.abs(optimized) > (2 * self.processing_threshold)
        optimized[strong_mask] *= 1.2
        
        return optimized
    
    def _assess_quality(self, signals: np.ndarray) -> float:
        """Assess the quality of processed signals."""
        if len(signals) == 0:
            return 0.0
            
        # Signal-to-noise ratio estimation
        signal_power = np.mean(signals**2)
        noise_estimate = np.var(np.diff(signals))
        
        if noise_estimate == 0:
            return 1.0
            
        snr = signal_power / noise_estimate
        quality = min(1.0, snr / 10.0)  # Normalize to [0, 1]
        
        return float(quality)


def process_multi_bit_signals(
    price_data: List[float],
    volume_data: List[float],
    bit_precision: int = 64,
) -> Dict[str, Any]:
    """Process multi-bit BTC signals (functional interface).
    
    Parameters
    ----------
    price_data
        Bitcoin price time series
    volume_data
        Trading volume time series
    bit_precision
        Bit precision for processing (default: 64)
        
    Returns
    -------
    Dict[str, Any]
        Processing results
    """
    processor = MultiBitBtcProcessor(bit_precision=bit_precision)
    return processor.process_signals(price_data, volume_data)


def optimize_bit_precision(
    signals: List[float],
    target_precision: int = 32,
) -> List[float]:
    """Optimize signal bit precision for performance.
    
    Parameters
    ----------
    signals
        Input signal data
    target_precision
        Target bit precision
        
    Returns
    -------
    List[float]
        Precision-optimized signals
    """
    if not signals:
        return []
        
    signal_array = np.array(signals, dtype=np.float64)
    
    # Apply precision optimization
    if target_precision <= 32:
        # Use float32 for lower precision
        optimized = signal_array.astype(np.float32)
    else:
        # Keep float64 for high precision
        optimized = signal_array
        
    return optimized.tolist()


if __name__ == "__main__":
    # Example usage
    processor = MultiBitBtcProcessor()
    
    # Test with sample data
    test_prices = [50000.0, 50100.0, 49950.0, 50200.0, 50050.0]
    test_volumes = [1.5, 2.1, 1.8, 2.3, 1.9]
    
    result = processor.process_signals(test_prices, test_volumes)
    print(f"Processing result: {result}")
