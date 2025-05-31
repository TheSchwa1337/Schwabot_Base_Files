"""
Test file for the Core Mathematical Library v0.2x
Demonstrates the usage of advanced features and multi-signal strategies
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")
from mathlib_v2 import CoreMathLibV2

def generate_test_data(n_samples: int = 1000, seed: int = 42) -> tuple:
    """
    Generate synthetic price, volume, and OHLC data for testing
    """
    np.random.seed(seed)
    
    # Generate trend component
    t = np.linspace(0, 1, n_samples)
    trend = 100 + 20 * t + 5 * np.sin(2 * np.pi * t * 5)
    
    # Generate noise component
    noise = np.random.normal(0, 1, n_samples)
    
    # Generate prices
    close = trend + noise
    
    # Generate high/low prices
    daily_range = np.random.uniform(1, 3, n_samples)
    high = close + daily_range
    low = close - daily_range
    
    # Generate volumes with correlation to price changes
    price_changes = np.diff(close, prepend=close[0])
    volumes = np.abs(price_changes) * 10 + np.random.normal(0, 1, n_samples)
    
    return close, high, low, volumes

def plot_results_v2(prices: np.ndarray, results: dict, volumes: np.ndarray = None, save_path: str = "tests/output/mathlib_v2_results.png") -> None:
    """
    Plot the results of various v0.2x calculations
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
    # Plot 1: Price, VWAP, and Keltner Channels
    axes[0, 0].plot(prices, label='Price', alpha=0.7)
    axes[0, 0].plot(results['vwap'], label='VWAP', alpha=0.7)
    if 'keltner_upper' in results:
        axes[0, 0].plot(results['keltner_upper'], label='Keltner Upper', alpha=0.5)
        axes[0, 0].plot(results['keltner_lower'], label='Keltner Lower', alpha=0.5)
    axes[0, 0].set_title('Price, VWAP, and Keltner Channels')
    axes[0, 0].legend()
    
    # Plot 2: RSI
    axes[0, 1].plot(results['rsi'], label='RSI', alpha=0.7)
    axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Relative Strength Index')
    axes[0, 1].legend()
    
    # Plot 3: ATR and Volatility
    if 'atr' in results:
        axes[1, 0].plot(results['atr'], label='ATR', alpha=0.7)
    axes[1, 0].plot(results['std'], label='Standard Deviation', alpha=0.7)
    axes[1, 0].set_title('Volatility Measures')
    axes[1, 0].legend()
    
    # Plot 4: Kelly Fraction and Risk Parity Weights
    axes[1, 1].plot(results['kelly_fraction'], label='Kelly Fraction', alpha=0.7)
    axes[1, 1].plot(results['risk_parity_weights'], label='Risk Parity Weights', alpha=0.7)
    axes[1, 1].set_title('Position Sizing Metrics')
    axes[1, 1].legend()
    
    # Plot 5: Pair Trade Z-Score
    axes[2, 0].plot(results['pair_zscore'], label='Pair Z-Score', alpha=0.7)
    axes[2, 0].axhline(y=2, color='r', linestyle='--', alpha=0.5)
    axes[2, 0].axhline(y=-2, color='g', linestyle='--', alpha=0.5)
    axes[2, 0].set_title('Pair Trade Z-Score')
    axes[2, 0].legend()
    
    # Plot 6: Ornstein-Uhlenbeck Process
    axes[2, 1].plot(results['ou_process'], label='OU Process', alpha=0.7)
    axes[2, 1].plot(prices, label='Actual Price', alpha=0.3)
    axes[2, 1].set_title('Ornstein-Uhlenbeck Process')
    axes[2, 1].legend()
    
    # Plot 7: Memory Weighted Returns
    axes[3, 0].plot(results['returns'], label='Raw Returns', alpha=0.3)
    axes[3, 0].plot(results['memory_weighted_returns'], label='Memory Weighted', alpha=0.7)
    axes[3, 0].set_title('Memory Weighted Returns')
    axes[3, 0].legend()
    
    # Plot 8: Volume Analysis
    if volumes is not None:
        axes[3, 1].plot(volumes, label='Volume', alpha=0.7)
        axes[3, 1].set_title('Trading Volume')
        axes[3, 1].legend()
    
    plt.tight_layout()
    os.makedirs("tests/output", exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def main() -> None:
    # Initialize the v0.2x mathematical library
    math_lib = CoreMathLibV2(
        base_volume=1.0,
        tick_freq=1.0,
        profit_coef=0.8,
        threshold=0.5
    )
    
    # Generate test data
    prices, high, low, volumes = generate_test_data(n_samples=1000)
    
    # Apply advanced v0.2x strategies
    results = math_lib.apply_advanced_strategies_v2(prices, volumes, high, low)
    
    # Print some statistics
    print("\nTrading Statistics v0.2x:")
    print(f"Kelly Fraction: {results['kelly_fraction']:.4f}")
    print(f"Average RSI: {np.mean(results['rsi']):.2f}")
    if 'atr' in results:
        print(f"Average ATR: {np.mean(results['atr']):.4f}")
    print(f"Risk Parity Weight: {results['risk_parity_weights'][-1]:.4f}")
    
    # Plot results
    plot_results_v2(prices, results, volumes)
    
    # Demonstrate pair trading
    print("\nPair Trading Analysis:")
    zscore = results['pair_zscore'][-1]
    print(f"Current Pair Z-Score: {zscore:.4f}")
    if zscore > 2:
        print("Signal: Short the spread")
    elif zscore < -2:
        print("Signal: Long the spread")
    else:
        print("Signal: No trade")

if __name__ == "__main__":
    main() 