"""
Test file for MEMKEY CPU-SYNC Math Logic
Demonstrates memkey-triggered hash cycles and pattern recognition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from memkey_sync import MemkeySync
from datetime import datetime

def plot_fault_distribution(fault_log: pd.DataFrame):
    """
    Plot distribution of fault types
    
    Args:
        fault_log: DataFrame of fault entries
    """
    # Count fault types
    fault_counts = fault_log['fault_type'].value_counts()
    
    plt.figure(figsize=(10, 6))
    fault_counts.plot(kind='bar')
    plt.title('Distribution of Fault Types')
    plt.xlabel('Fault Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_cpu_timing(fault_log: pd.DataFrame):
    """
    Plot CPU timing distribution
    
    Args:
        fault_log: DataFrame of fault entries
    """
    plt.figure(figsize=(10, 6))
    plt.hist(fault_log['cpu_time'], bins=50)
    plt.title('CPU Timing Distribution')
    plt.xlabel('CPU Time (μs)')
    plt.ylabel('Count')
    plt.axvline(x=10.0, color='r', linestyle='--', label='Threshold (10μs)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_memkey_occurrences(memkey_history: dict):
    """
    Plot memkey occurrence patterns
    
    Args:
        memkey_history: Dictionary of memkey occurrences
    """
    # Count occurrences per memkey
    occurrence_counts = {k: len(v) for k, v in memkey_history.items()}
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(occurrence_counts)), list(occurrence_counts.values()))
    plt.title('Memkey Occurrence Patterns')
    plt.xlabel('Memkey Index')
    plt.ylabel('Occurrence Count')
    plt.tight_layout()
    plt.show()

def main():
    # Initialize MEMKEY sync
    memkey = MemkeySync()
    
    # Process 1000 ticks
    print("Processing 1000 ticks...")
    for i in range(1000):
        result = memkey.process_tick(i)
        
        # Print interesting results
        if result.fault_triggered:
            print(f"\nTick {i}:")
            print(f"Timestamp: {result.timestamp}")
            print(f"Value: {result.value:.2f}")
            print(f"Hash: {result.hash_full}")
            print(f"CPU Time: {result.cpu_time:.2f}μs")
            print(f"Fault Type: {result.fault_type}")
            if result.memkey:
                print(f"Memkey: {result.memkey}")
    
    # Get statistics
    stats = memkey.get_stats()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Get fault log
    fault_log = memkey.get_fault_log()
    if not fault_log.empty:
        print("\nFault Log Summary:")
        print(fault_log.describe())
        
        # Plot fault distribution
        plot_fault_distribution(fault_log)
        
        # Plot CPU timing
        plot_cpu_timing(fault_log)
    
    # Get memkey history
    memkey_history = memkey.get_memkey_history()
    if memkey_history:
        print("\nMemkey History:")
        for memkey, occurrences in memkey_history.items():
            if len(occurrences) > 1:
                print(f"Memkey {memkey}: {len(occurrences)} occurrences at ticks {occurrences}")
        
        # Plot memkey occurrences
        plot_memkey_occurrences(memkey_history)
    
    # Demonstrate memkey reuse check
    print("\nMemkey Reuse Check:")
    for memkey in memkey_history.keys():
        if memkey.check_memkey_reuse(memkey):
            print(f"Memkey {memkey} has been reused")

if __name__ == "__main__":
    main() 