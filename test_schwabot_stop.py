"""
Test file for Schwabot Net Stop Loss Pattern Value Book
Demonstrates pattern detection and state transitions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from schwabot_stop import SchwabotStopBook, StopPatternState
import matplotlib.pyplot as plt

def generate_test_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic test data with various patterns
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with timestamp and value columns
    """
    np.random.seed(seed)
    
    # Generate timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate base values with trend
    trend = np.linspace(0, 0.1, n_samples)
    noise = np.random.normal(0, 0.01, n_samples)
    values = trend + noise
    
    # Add some pattern events
    # Warning pattern
    values[200:250] += np.linspace(0, 0.03, 50)
    
    # Alert pattern
    values[400:450] += np.linspace(0, 0.06, 50)
    
    # Trigger pattern
    values[600:650] += np.linspace(0, 0.09, 50)
    
    # Recovery pattern
    values[800:850] += np.linspace(0.09, 0, 50)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })

def plot_patterns(data: pd.DataFrame, 
                 patterns: dict,
                 transitions: list):
    """
    Plot the data with pattern states and transitions
    
    Args:
        data: DataFrame with timestamp and value columns
        patterns: Dictionary of active patterns
        transitions: List of state transitions
    """
    plt.figure(figsize=(15, 10))
    
    # Plot values
    plt.plot(data['timestamp'], data['value'], label='Value', alpha=0.5)
    
    # Plot pattern states
    for pattern_id, pattern in patterns.items():
        plt.scatter(pattern.timestamp, pattern.value, 
                   label=f'Pattern {pattern_id}: {pattern.state.value}',
                   marker='o')
    
    # Plot state transitions
    for old_state, new_state, timestamp in transitions:
        plt.axvline(x=timestamp, color='gray', linestyle='--', alpha=0.3)
        plt.text(timestamp, plt.ylim()[1], 
                f'{old_state.value}→{new_state.value}',
                rotation=45, ha='right')
    
    plt.title('Stop Loss Pattern Detection')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    # Initialize stop book
    stop_book = SchwabotStopBook(
        warning_threshold=0.02,
        alert_threshold=0.05,
        trigger_threshold=0.08,
        recovery_threshold=0.03
    )
    
    # Generate test data
    data = generate_test_data()
    
    # Process data and track patterns
    pattern_id = 1
    for _, row in data.iterrows():
        # Update pattern
        state = stop_book.update_pattern(
            pattern_id=f'pattern_{pattern_id}',
            value=row['value'],
            timestamp=row['timestamp'],
            metadata={'source': 'test_data'}
        )
        
        # Create new pattern if current one is completed
        if state in [StopPatternState.TRIGGERED, StopPatternState.RESET]:
            pattern_id += 1
    
    # Get results
    active_patterns = stop_book.get_active_patterns()
    pattern_history = stop_book.get_pattern_history()
    state_transitions = stop_book.get_state_transitions()
    
    # Print statistics
    print("\nPattern Statistics:")
    print(f"Active Patterns: {len(active_patterns)}")
    print(f"Completed Patterns: {len(pattern_history)}")
    print(f"State Transitions: {len(state_transitions)}")
    
    print("\nState Transition Summary:")
    transition_counts = {}
    for old_state, new_state, _ in state_transitions:
        key = f"{old_state.value}→{new_state.value}"
        transition_counts[key] = transition_counts.get(key, 0) + 1
    
    for transition, count in transition_counts.items():
        print(f"{transition}: {count}")
    
    # Plot results
    plot_patterns(data, active_patterns, state_transitions)
    
    # Demonstrate threshold recalibration
    print("\nRecalibrating thresholds...")
    stop_book.recalibrate_thresholds(
        new_warning=0.03,
        new_alert=0.06,
        new_trigger=0.09,
        new_recovery=0.04
    )
    
    # Process data again with new thresholds
    pattern_id = 1
    for _, row in data.iterrows():
        state = stop_book.update_pattern(
            pattern_id=f'pattern_{pattern_id}',
            value=row['value'],
            timestamp=row['timestamp'],
            metadata={'source': 'test_data_recalibrated'}
        )
        if state in [StopPatternState.TRIGGERED, StopPatternState.RESET]:
            pattern_id += 1
    
    # Plot results with new thresholds
    active_patterns = stop_book.get_active_patterns()
    state_transitions = stop_book.get_state_transitions()
    plot_patterns(data, active_patterns, state_transitions)

if __name__ == "__main__":
    main() 