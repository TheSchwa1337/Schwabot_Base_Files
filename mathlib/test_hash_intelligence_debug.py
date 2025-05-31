"""
Test file for Hash Intelligence Debug System
Demonstrates safe memory swapping and CPU strain monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from hash_intelligence_debug import HashIntelligenceDebug, HashDebugState

def plot_debug_states(debug_history: list[HashDebugState]):
    """
    Plot debug states over time
    
    Args:
        debug_history: List of HashDebugState objects
    """
    # Convert to DataFrame
    df = pd.DataFrame([{
        'timestamp': datetime.fromisoformat(f.timestamp),
        'cpu_usage': f.cpu_usage,
        'cpu_temp': f.cpu_temp,
        'memory_usage': f.memory_usage,
        'strain_level': f.strain_level,
        'swap_mode': f.swap_mode,
        'hash_rate': f.hash_rate,
        'debug_code': f.debug_code
    } for f in debug_history])
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot CPU metrics
    ax1.plot(df['timestamp'], df['cpu_usage'], 'b-', label='CPU Usage')
    ax1.plot(df['timestamp'], df['cpu_temp']/100, 'r--', label='CPU Temp (normalized)')
    ax1.set_title('CPU Metrics Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    ax1.legend()
    
    # Plot memory usage
    ax2.plot(df['timestamp'], df['memory_usage'], 'g-', label='Memory Usage')
    ax2.set_title('Memory Usage Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Usage')
    ax2.grid(True)
    ax2.legend()
    
    # Plot strain levels
    strain_levels = df['strain_level'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(strain_levels)))
    for level, color in zip(strain_levels, colors):
        mask = df['strain_level'] == level
        ax3.scatter(df[mask]['timestamp'], df[mask]['cpu_usage'],
                   c=[color], label=level, alpha=0.6)
    ax3.set_title('Strain Level Classification')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('CPU Usage')
    ax3.grid(True)
    ax3.legend()
    
    # Plot swap modes
    swap_modes = df['swap_mode'].unique()
    for mode in swap_modes:
        mask = df['swap_mode'] == mode
        ax4.plot(df[mask]['timestamp'], df[mask]['memory_usage'],
                label=f'{mode} Mode', alpha=0.6)
    ax4.set_title('Memory Swap Modes')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Memory Usage')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main test function"""
    # Initialize debug system
    debug = HashIntelligenceDebug()
    
    # Generate test hashes
    n_ticks = 100
    test_hashes = [sha256(str(i).encode()).hexdigest() for i in range(n_ticks)]
    
    # Process hashes
    for i, hash_value in enumerate(test_hashes):
        state = debug.process_hash(hash_value)
        
        print(f"\nTick {i+1}:")
        print(f"Hash: {state.hash_value[:8]}...")
        print(f"CPU Usage: {state.cpu_usage:.1%}")
        print(f"CPU Temp: {state.cpu_temp:.1f}°C")
        print(f"Memory Usage: {state.memory_usage:.1%}")
        print(f"Strain Level: {state.strain_level}")
        print(f"Swap Mode: {state.swap_mode}")
        print(f"Hash Rate: {state.hash_rate}")
        print(f"Debug Code: {state.debug_code}")
    
    # Plot results
    plot_debug_states(debug.get_debug_history())

if __name__ == "__main__":
    main() 