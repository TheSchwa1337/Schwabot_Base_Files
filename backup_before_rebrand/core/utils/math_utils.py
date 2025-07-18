#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Utilities for Schwabot Trading System
=================================================

Provides utility functions for mathematical operations.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class MathUtils:
    """Mathematical utilities for Schwabot."""
    
    def __init__(self):
        """Initialize math utilities."""
        self.logger = logging.getLogger(__name__)
    
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        if not vector:
            return []
        
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return [0.0] * len(vector)
        
        return [x / magnitude for x in vector]
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if not prices or period <= 0:
            return []
        
        alpha = 2.0 / (period + 1)
        ema_values = [prices[0]]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(sma)
        
        return sma_values
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return []
        
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        for i in range(period, len(prices)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
            
            # Update averages
            if i < len(prices) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return rsi_values
    
    def calculate_bollinger_bands(
        self, prices: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return [], [], []
        
        sma_values = self.calculate_sma(prices, period)
        upper_band = []
        lower_band = []
        
        for i in range(len(sma_values)):
            start_idx = i
            end_idx = start_idx + period
            window = prices[start_idx:end_idx]
            
            # Calculate standard deviation
            mean = sma_values[i]
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            std = math.sqrt(variance)
            
            upper_band.append(mean + std_dev * std)
            lower_band.append(mean - std_dev * std)
        
        return sma_values, upper_band, lower_band
    
    def calculate_macd(
        self,
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(prices) < slow_period:
            return [], [], []
        
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
        
        # Calculate signal line
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]
        
        return macd_line, signal_line, histogram

# Global instance
math_utils = MathUtils()

def get_math_utils() -> MathUtils:
    """Get the global math utils instance."""
    return math_utils
