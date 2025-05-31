"""
Test file for GPU Flash Engine
Demonstrates safe GPU activation and memory swap logic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from gpu_flash_engine import GPUFlashEngine, FlashState

def plot_flash_states(flash_history: list[FlashState]):
    """
    Plot flash states over time
    
    Args:
        flash_history: List of FlashState objects
    """
    # Convert to DataFrame
    df = pd.DataFrame([{
        'timestamp': datetime.fromisoformat(f.timestamp),
        'hash_value': f.hash_value,
        'z_score': f.z_score,
        'gpu_utilization': f.gpu_utilization,
        'gpu_temperature': f.gpu_temperature,
        'zpe_zone': f.zpe_zone,
        'flash_permitted': f.flash_permitted,
        'memory_swap_mode': f.memory_swap_mode
    } for f in flash_history])
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot hash value and z-score
    ax1.plot(df['timestamp'], df['hash_value'], 'b-', label='Hash Value')
    ax1.plot(df['timestamp'], df['z_score'], 'r--', label='Z-Score')
    ax1.set_title('Hash Value and Z-Score Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    ax1.legend()
    
    # Plot GPU metrics
    ax2.plot(df['timestamp'], df['gpu_utilization'], 'g-', label='GPU Utilization')
    ax2.plot(df['timestamp'], df['gpu_temperature'], 'm--', label='GPU Temperature')
    ax2.set_title('GPU Metrics Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    ax2.legend()
    
    # Plot ZPE zones
    zones = df['zpe_zone'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
    for zone, color in zip(zones, colors):
        mask = df['zpe_zone'] == zone
        ax3.scatter(df[mask]['hash_value'], df[mask]['z_score'],
                   c=[color], label=zone, alpha=0.6)
    ax3.set_title('ZPE Zone Classification')
    ax3.set_xlabel('Hash Value')
    ax3.set_ylabel('Z-Score')
    ax3.grid(True)
    ax3.legend()
    
    # Plot memory swap modes
    modes = df['memory_swap_mode'].unique()
    for mode in modes:
        mask = df['memory_swap_mode'] == mode
        ax4.plot(df[mask]['timestamp'], df[mask]['gpu_utilization'],
                label=f'{mode} Mode', alpha=0.6)
    ax4.set_title('Memory Swap Modes')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('GPU Utilization')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main test function"""
    # Initialize engine
    engine = GPUFlashEngine()
    
    # Generate test data
    n_ticks = 100
    hash_values = np.linspace(0, 1, n_ticks)
    z_scores = np.random.normal(0, 1, n_ticks)
    cpu_times = np.random.uniform(10, 30, n_ticks)
    
    # Process ticks
    for i in range(n_ticks):
        state = engine.process_tick(
            hash_value=hash_values[i],
            z_score=z_scores[i],
            cpu_time_us=cpu_times[i]
        )
        
        print(f"\nTick {i+1}:")
        print(f"Hash Value: {state.hash_value:.3f}")
        print(f"Z-Score: {state.z_score:.3f}")
        print(f"ZPE Zone: {state.zpe_zone}")
        print(f"Flash Permitted: {state.flash_permitted}")
        print(f"Memory Swap Mode: {state.memory_swap_mode}")
    
    # Plot results
    plot_flash_states(engine.get_flash_history())

if __name__ == "__main__":
    main() 