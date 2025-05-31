"""
Test file for DCC (Desync Correction Code) Library
Demonstrates desync detection and correction functionality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dcc_sync import DCCSync
from datetime import datetime
import json

def generate_test_data(n_ticks: int = 1000, seed: int = 42) -> dict:
    """
    Generate synthetic test data
    
    Args:
        n_ticks: Number of ticks to generate
        seed: Random seed
        
    Returns:
        Dictionary of test data
    """
    np.random.seed(seed)
    
    # Generate base data
    ticks = np.arange(n_ticks)
    base_profit = np.cumsum(np.random.normal(0, 0.1, n_ticks))
    
    # Add drift component
    drift = np.linspace(0, 2, n_ticks)
    drifted_profit = base_profit + drift
    
    # Generate hashes
    hashes = [f"{np.random.bytes(32).hex()}" for _ in range(n_ticks)]
    
    # Generate volatility
    sigma = np.abs(np.random.normal(0.1, 0.02, n_ticks))
    
    # Generate book value offset
    deltaB = np.random.normal(0, 0.05, n_ticks)
    
    # Generate ping offset
    ping_offset = np.random.randint(0, 10, n_ticks)
    
    # Generate z-scores
    zscore = np.random.normal(0, 1, n_ticks)
    
    return {
        'ticks': ticks,
        'base_profit': base_profit,
        'drifted_profit': drifted_profit,
        'hashes': hashes,
        'sigma': sigma,
        'deltaB': deltaB,
        'ping_offset': ping_offset,
        'zscore': zscore
    }

def plot_dcc_metrics(results: list):
    """
    Plot DCC metrics over time
    
    Args:
        results: List of DCCResult objects
    """
    # Convert to DataFrame
    df = pd.DataFrame([vars(r) for r in results])
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot syndrome and severity
    ax1.plot(df['syndrome'], label='Syndrome')
    ax1.plot(df['severity'], label='Severity')
    ax1.set_title('DCC Syndrome and Severity')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cost
    ax2.plot(df['cost'], label='Cost')
    ax2.set_title('DCC Cost')
    ax2.legend()
    ax2.grid(True)
    
    # Plot z-score and ping offset
    ax3.plot(df['zscore'], label='Z-Score')
    ax3.plot(df['ping_offset'], label='Ping Offset')
    ax3.set_title('Z-Score and Ping Offset')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_action_distribution(results: list):
    """
    Plot distribution of DCC actions
    
    Args:
        results: List of DCCResult objects
    """
    # Count actions
    actions = [r.action for r in results]
    action_counts = pd.Series(actions).value_counts()
    
    plt.figure(figsize=(10, 6))
    action_counts.plot(kind='bar')
    plt.title('Distribution of DCC Actions')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Initialize DCC sync
    dcc = DCCSync()
    
    # Generate test data
    test_data = generate_test_data()
    
    # Process ticks
    print("Processing ticks...")
    results = []
    for i in range(len(test_data['ticks'])):
        result = dcc.process_tick(
            tick=test_data['ticks'][i],
            H_live=test_data['hashes'][i],
            pi_live=test_data['drifted_profit'][i],
            sigma=test_data['sigma'][i],
            deltaB=test_data['deltaB'][i],
            ping_offset=test_data['ping_offset'][i],
            zscore=test_data['zscore'][i]
        )
        results.append(result)
        
        # Print interesting results
        if result.action != 'MONITOR':
            print(f"\nTick {i}:")
            print(f"Action: {result.action}")
            print(f"Syndrome: {result.syndrome:.4f}")
            print(f"Severity: {result.severity:.4f}")
            print(f"Cost: {result.cost:.4f}")
    
    # Plot results
    plot_dcc_metrics(results)
    plot_action_distribution(results)
    
    # Get history
    history = dcc.get_history()
    if history:
        print("\nDCC History Summary:")
        print(f"Total events: {len(history)}")
        print(f"Last event: {history[-1]}")

if __name__ == "__main__":
    main() 