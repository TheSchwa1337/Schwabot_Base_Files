"""
Test file for the Core Mathematical Library
Demonstrates the usage of various mathematical functions and strategies
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")
from mathlib import CoreMathLib

def generate_test_data(n_samples: int = 1000, seed: int = 42) -> tuple:
    """
    Generate synthetic price and volume data for testing
    """
    np.random.seed(seed)
    
    # Generate trend component
    t = np.linspace(0, 1, n_samples)
    trend = 100 + 20 * t + 5 * np.sin(2 * np.pi * t * 5)
    
    # Generate noise component
    noise = np.random.normal(0, 1, n_samples)
    
    # Generate prices
    prices = trend + noise
    
    # Generate volumes with correlation to price changes
    price_changes = np.diff(prices, prepend=prices[0])
    volumes = np.abs(price_changes) * 10 + np.random.normal(0, 1, n_samples)
    
    return prices, volumes

def plot_results(prices: np.ndarray, results: dict, volumes: np.ndarray = None, save_path: str = "tests/output/mathlib_results.png") -> None:
    """
    Plot the results of various calculations
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot 1: Price and Bollinger Bands
    axes[0, 0].plot(prices, label='Price', alpha=0.7)
    axes[0, 0].plot(results['upper_band'], label='Upper Band', alpha=0.5)
    axes[0, 0].plot(results['lower_band'], label='Lower Band', alpha=0.5)
    axes[0, 0].set_title('Price and Bollinger Bands')
    axes[0, 0].legend()
    
    # Plot 2: Returns and Risk-Adjusted Returns
    axes[0, 1].plot(results['returns'], label='Returns', alpha=0.7)
    axes[0, 1].plot(results['risk_adj_returns'], label='Risk-Adjusted Returns', alpha=0.7)
    axes[0, 1].set_title('Returns Analysis')
    axes[0, 1].legend()
    
    # Plot 3: Momentum and Mean Reversion
    axes[1, 0].plot(results['momentum'], label='Momentum', alpha=0.7)
    axes[1, 0].plot(results['mean_reversion'], label='Mean Reversion', alpha=0.7)
    axes[1, 0].set_title('Momentum and Mean Reversion')
    axes[1, 0].legend()
    
    # Plot 4: Z-Scores
    axes[1, 1].plot(results['z_scores'], label='Z-Score', alpha=0.7)
    axes[1, 1].axhline(y=2, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=-2, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Z-Scores')
    axes[1, 1].legend()
    
    # Plot 5: Volume-Price Trend
    axes[2, 0].plot(results['vpt'], label='VPT', alpha=0.7)
    if volumes is not None:
        axes[2, 0].plot(volumes, label='Volume', alpha=0.3)
    axes[2, 0].set_title('Volume-Price Trend')
    axes[2, 0].legend()
    
    # Plot 6: Profit Ratios
    axes[2, 1].plot(results['profit_ratios'], label='Profit Ratios', alpha=0.7)
    axes[2, 1].set_title('Profit Ratios')
    axes[2, 1].legend()
    
    plt.tight_layout()
    os.makedirs("tests/output", exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def main() -> None:
    # Initialize the mathematical library
    math_lib = CoreMathLib()
        base_volume=1.0,
        tick_freq=1.0,
        profit_coef=0.8,
        threshold=0.5
(    )
    
    # Generate test data
    prices, volumes = generate_test_data(n_samples=1000)
    
    # Apply advanced strategies
    results = math_lib.apply_advanced_strategies(prices, volumes)
    
    # Print some statistics
    print("\nTrading, Statistics:")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Cumulative Log Return: {results['log_return']:.4f}")
    print(f"Price Entropy: {results['entropy']:.4f}")
    
    # Plot results
    plot_results(prices, results, volumes)
    
    # Demonstrate hash-based decision making
    print("\nHash-based Decision Making:")
    for i in range(5):
        price = prices[i]
        lower_bound = results['lower_band'][i]
        upper_bound = results['upper_band'][i]
        hash_data = f"price_{price}_tick_{i}"
        
        decision = math_lib.tick_bound_execution(price, lower_bound, upper_bound, hash_data)
        print(f"Tick {i}: Price={price:.2f}, Decision={decision}")
    
    # Demonstrate volume allocation
    print("\nVolume, Allocation:")
    for i in range(5):
        volume = math_lib.allocate_volume(1.0, i, 10, method='sine')
        print(f"Tick {i}: Volume={volume:.4f}")

if __name__ == "__main__":
    main() 