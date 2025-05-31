"""
Test file for RITTLE-GEMM Ring Value Schema
Demonstrates ring-based matrix operations and strategy triggering
"""

import numpy as np
import pandas as pd
from rittle_gemm import RittleGEMM, RingLayer
from mathlib_v2 import CoreMathLibV2

def generate_test_ticks(n_ticks: int = 1000, seed: int = 42) -> List[dict]:
    """
    Generate synthetic tick data for testing
    """
    np.random.seed(seed)
    ticks = []
    
    # Initialize price and volume
    price = 100.0
    volume = 1.0
    
    for t in range(n_ticks):
        # Generate price movement
        price_change = np.random.normal(0, 0.1)
        price *= (1 + price_change)
        
        # Generate volume
        volume = max(0.1, volume * (1 + np.random.normal(0, 0.05)))
        
        # Calculate return
        return_val = price_change
        
        # Generate hash value
        hash_rec = np.random.random()
        
        # Calculate z-score
        z_score = np.random.normal(0, 1)
        
        # Calculate drift
        drift = np.random.normal(0, 0.01)
        
        # Determine execution
        executed = 1 if np.random.random() > 0.7 else 0
        
        # Calculate profit
        profit = return_val * volume if executed else 0
        
        # Determine rebuy
        rebuy = 1 if np.random.random() > 0.9 else 0
        
        tick_data = {
            'timestamp': t,
            'price': price,
            'volume': volume,
            'return': return_val,
            'hash_rec': hash_rec,
            'z_score': z_score,
            'drift': drift,
            'executed': executed,
            'profit': profit,
            'rebuy': rebuy
        }
        
        ticks.append(tick_data)
    
    return ticks

def plot_ring_values(ring_values: pd.DataFrame):
    """
    Plot ring values over time
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, (ring, values) in enumerate(ring_values.items()):
        if i < len(axes):
            axes[i].plot(values, label=ring)
            axes[i].set_title(f'Ring {ring}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialize RITTLE-GEMM
    rittle = RittleGEMM(ring_size=1000)
    
    # Generate test ticks
    ticks = generate_test_ticks(n_ticks=1000)
    
    # Process ticks and collect ring values
    ring_values = {ring: [] for ring in RingLayer}
    strategy_triggers = []
    
    for tick in ticks:
        # Process tick
        current_rings = rittle.process_tick(tick)
        
        # Store ring values
        for ring, value in current_rings.items():
            ring_values[ring].append(value)
        
        # Check for strategy triggers
        should_trigger, strategy_id = rittle.check_strategy_trigger()
        if should_trigger:
            strategy_triggers.append({
                'timestamp': tick['timestamp'],
                'strategy_id': strategy_id,
                'ring_snapshot': rittle.get_ring_snapshot()
            })
    
    # Convert to DataFrame for analysis
    ring_df = pd.DataFrame(ring_values)
    
    # Print statistics
    print("\nRITTLE-GEMM Ring Statistics:")
    print(ring_df.describe())
    
    print("\nStrategy Triggers:")
    for trigger in strategy_triggers:
        print(f"Time {trigger['timestamp']}: {trigger['strategy_id']}")
        print("Ring Values:", trigger['ring_snapshot'])
        print()
    
    # Plot ring values
    plot_ring_values(ring_df)
    
    # Demonstrate GEMM operations
    print("\nGEMM Operations:")
    A = np.random.randn(10, 10)
    B = np.random.randn(10, 10)
    R_prev = np.zeros((10, 10))
    decay = 0.95
    
    gemm_output = rittle.calculate_gemm_output(A, B, decay, R_prev)
    print("GEMM Output Shape:", gemm_output.shape)
    print("GEMM Output Stats:")
    print("Mean:", np.mean(gemm_output))
    print("Std:", np.std(gemm_output))
    print("Max:", np.max(gemm_output))
    print("Min:", np.min(gemm_output))

if __name__ == "__main__":
    main() 