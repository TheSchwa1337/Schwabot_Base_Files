"""
Test file for Matrix Fault Resolver
Demonstrates safe matrix state transitions during faults
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from matrix_fault_resolver import MatrixFaultResolver, FaultState

def plot_fault_transitions(fault_history: list[FaultState]):
    """
    Plot fault transitions over time
    
    Args:
        fault_history: List of FaultState objects
    """
    # Convert to DataFrame
    df = pd.DataFrame([{
        'timestamp': datetime.fromisoformat(f.timestamp),
        'fault_type': f.fault_type,
        'severity': f.severity,
        'current_matrix': f.current_matrix,
        'target_matrix': f.target_matrix,
        'retry_count': f.retry_count
    } for f in fault_history])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot severity over time
    ax1.plot(df['timestamp'], df['severity'], 'b-', label='Severity')
    ax1.set_title('Fault Severity Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Severity')
    ax1.grid(True)
    
    # Plot matrix transitions
    for i, row in df.iterrows():
        ax2.plot([row['timestamp'], row['timestamp']], 
                [row['current_matrix'], row['target_matrix']], 
                'r-', alpha=0.5)
        ax2.scatter(row['timestamp'], row['current_matrix'], 
                   c='blue', marker='o')
        ax2.scatter(row['timestamp'], row['target_matrix'], 
                   c='red', marker='x')
    
    ax2.set_title('Matrix State Transitions')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Matrix State')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main test function"""
    # Initialize resolver
    resolver = MatrixFaultResolver()
    
    # Simulate faults
    fault_types = ['cpu_overload', 'hash_collision', 'memory_leak', 'zpe_risk']
    severities = [0.3, 0.6, 0.8, 0.95]
    
    for fault_type, severity in zip(fault_types, severities):
        # Handle fault
        fault = resolver.handle_fault(fault_type, severity)
        
        # Attempt transition
        success = resolver.transition_matrix(fault)
        
        print(f"\nFault: {fault_type}")
        print(f"Severity: {severity}")
        print(f"Current Matrix: {fault.current_matrix}")
        print(f"Target Matrix: {fault.target_matrix}")
        print(f"Transition Success: {success}")
        print(f"Retry Count: {fault.retry_count}")
    
    # Plot transitions
    plot_fault_transitions(resolver.get_fault_history())
    
    # Print final state
    print("\nFinal Matrix State:", resolver.get_current_matrix())

if __name__ == "__main__":
    main() 