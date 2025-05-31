"""
Core Mathematical Library v0.2x for Quantitative Trading System
Extends the base implementation with advanced multi-signal and risk-aware features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from mathlib import CoreMathLib

class CoreMathLibV2(CoreMathLib):
    """
    Extended mathematical library with v0.2x features
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atr_alpha = 0.1  # ATR smoothing factor
        self.rsi_period = 14  # RSI lookback period
        self.keltner_k = 2.0  # Keltner Channel multiplier
        self.ou_theta = 0.1   # OU mean reversion speed
        self.ou_sigma = 0.1   # OU volatility
        self.memory_lambda = 0.95  # Memory decay factor
        
    def calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Calculate Volume-Weighted Average Price (VWAP)
        """
        cumulative_pv = np.cumsum(prices * volumes)
        cumulative_volume = np.cumsum(volumes)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = np.where(cumulative_volume != 0,
                          cumulative_pv / cumulative_volume,
                          prices)
        return vwap
    
    def calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate True Range (TR)
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(np.maximum(tr1, tr2), tr3)
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR)
        """
        tr = self.calculate_true_range(high, low, close)
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        
        for i in range(1, len(tr)):
            atr[i] = self.atr_alpha * tr[i] + (1 - self.atr_alpha) * atr[i-1]
            
        return atr
    
    def calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI)
        """
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # Initialize first values
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        
        # Calculate smoothed averages
        for i in range(1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (self.rsi_period - 1) + gain[i]) / self.rsi_period
            avg_loss[i] = (avg_loss[i-1] * (self.rsi_period - 1) + loss[i]) / self.rsi_period
        
        # Calculate RS and RSI
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
            rsi = 100 - (100 / (1 + rs))
            
        return rsi
    
    def calculate_kelly_fraction(self, returns: np.ndarray, risk_free_rate: float = 0.01) -> float:
        """
        Calculate Kelly Criterion Position Fraction
        """
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        if variance == 0:
            return 0
            
        return (mean_return - risk_free_rate) / variance
    
    def calculate_covariance(self, returns_x: np.ndarray, returns_y: np.ndarray, 
                           window: int = 20) -> np.ndarray:
        """
        Calculate rolling covariance between two return series
        """
        n = len(returns_x)
        cov = np.zeros(n)
        
        for i in range(window, n):
            x_window = returns_x[i-window:i]
            y_window = returns_y[i-window:i]
            
            x_mean = np.mean(x_window)
            y_mean = np.mean(y_window)
            
            cov[i] = np.mean((x_window - x_mean) * (y_window - y_mean))
            
        return cov
    
    def calculate_correlation(self, returns_x: np.ndarray, returns_y: np.ndarray, 
                            window: int = 20) -> np.ndarray:
        """
        Calculate rolling Pearson correlation
        """
        cov = self.calculate_covariance(returns_x, returns_y, window)
        std_x = self.calculate_rolling_std(returns_x, window)
        std_y = self.calculate_rolling_std(returns_y, window)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = np.where(std_x * std_y != 0,
                                 cov / (std_x * std_y),
                                 0)
        return correlation
    
    def calculate_risk_parity_weights(self, volatilities: np.ndarray) -> np.ndarray:
        """
        Calculate risk-parity weights based on inverse volatility
        """
        inv_vol = 1 / volatilities
        return inv_vol / np.sum(inv_vol)
    
    def calculate_pair_trade_zscore(self, price_x: np.ndarray, price_y: np.ndarray, 
                                  beta: float = 1.0, window: int = 20) -> np.ndarray:
        """
        Calculate pair-trade Z-score
        """
        spread = price_x - beta * price_y
        spread_mean = pd.Series(spread).rolling(window).mean().to_numpy()
        spread_std = pd.Series(spread).rolling(window).std().to_numpy()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            zscore = np.where(spread_std != 0,
                            (spread - spread_mean) / spread_std,
                            0)
        return zscore
    
    def simulate_ornstein_uhlenbeck(self, x0: float, mu: float, n_steps: int) -> np.ndarray:
        """
        Simulate Ornstein-Uhlenbeck process
        """
        dt = self.delta_t
        x = np.zeros(n_steps)
        x[0] = x0
        
        for i in range(1, n_steps):
            drift = self.ou_theta * (mu - x[i-1]) * dt
            diffusion = self.ou_sigma * np.sqrt(dt) * np.random.normal()
            x[i] = x[i-1] + drift + diffusion
            
        return x
    
    def calculate_keltner_channels(self, prices: np.ndarray, atr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Keltner Channels
        """
        ema = self.update_ema(prices)
        upper_channel = ema + self.keltner_k * atr
        lower_channel = ema - self.keltner_k * atr
        
        return upper_channel, lower_channel
    
    def apply_memory_kernel(self, values: np.ndarray) -> np.ndarray:
        """
        Apply exponential memory kernel for time decay
        """
        n = len(values)
        weights = (1 - self.memory_lambda) * self.memory_lambda ** np.arange(n-1, -1, -1)
        weights = weights / np.sum(weights)
        
        return np.convolve(values, weights, mode='valid')
    
    def apply_advanced_strategies_v2(self, prices: np.ndarray, volumes: np.ndarray,
                                   high: Optional[np.ndarray] = None,
                                   low: Optional[np.ndarray] = None) -> Dict:
        """
        Apply extended v0.2x trading strategies
        """
        results = super().apply_advanced_strategies(prices, volumes)
        
        # Calculate VWAP
        results['vwap'] = self.calculate_vwap(prices, volumes)
        
        # Calculate ATR if high/low data is available
        if high is not None and low is not None:
            results['atr'] = self.calculate_atr(high, low, prices)
            results['keltner_upper'], results['keltner_lower'] = self.calculate_keltner_channels(prices, results['atr'])
        
        # Calculate RSI
        results['rsi'] = self.calculate_rsi(prices)
        
        # Calculate Kelly fraction
        returns = results['returns']
        results['kelly_fraction'] = self.calculate_kelly_fraction(returns)
        
        # Calculate pair-trade Z-score (assuming second asset is a benchmark)
        if len(prices) > 1:
            benchmark = np.roll(prices, 1)  # Simple benchmark for demonstration
            results['pair_zscore'] = self.calculate_pair_trade_zscore(prices, benchmark)
        
        # Calculate risk-parity weights
        volatilities = results['std']
        results['risk_parity_weights'] = self.calculate_risk_parity_weights(volatilities)
        
        # Simulate OU process
        results['ou_process'] = self.simulate_ornstein_uhlenbeck(
            x0=prices[0],
            mu=np.mean(prices),
            n_steps=len(prices)
        )
        
        # Apply memory kernel to returns
        results['memory_weighted_returns'] = self.apply_memory_kernel(returns)
        
        return results 