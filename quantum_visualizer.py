"""
Quantum visualization tools for DLT Waveform Engine
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

class PanicDriftVisualizer:
    def __init__(self):
        self.time_points = []
        self.entropy_values = []
        self.coherence_values = []
        self.panic_threshold = 4.5
        self.drift_threshold = 0.4
        
    def add_data_point(self, timestamp: float, entropy: float, coherence: float):
        """Add a new data point to the visualization"""
        self.time_points.append(timestamp)
        self.entropy_values.append(entropy)
        self.coherence_values.append(coherence)
        
        # Keep only last 1000 points
        if len(self.time_points) > 1000:
            self.time_points = self.time_points[-1000:]
            self.entropy_values = self.entropy_values[-1000:]
            self.coherence_values = self.coherence_values[-1000:]
            
    def render(self):
        """Render the current state visualization"""
        if not self.time_points:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot entropy
        plt.subplot(2, 1, 1)
        plt.plot(self.time_points, self.entropy_values, 'b-', label='Entropy')
        plt.axhline(y=self.panic_threshold, color='r', linestyle='--', label='Panic Threshold')
        plt.ylabel('Entropy')
        plt.legend()
        
        # Plot coherence
        plt.subplot(2, 1, 2)
        plt.plot(self.time_points, self.coherence_values, 'g-', label='Coherence')
        plt.axhline(y=self.drift_threshold, color='r', linestyle='--', label='Drift Threshold')
        plt.ylabel('Coherence')
        plt.xlabel('Time')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def plot_entropy_waveform(data: List[float]):
    """Plot the entropy waveform of the processed data"""
    if not data:
        return
        
    plt.figure(figsize=(12, 4))
    
    # Plot raw data
    plt.plot(data, 'b-', label='Waveform')
    
    # Add entropy zones
    entropy = np.array([np.std(data[i:i+20]) for i in range(len(data)-20)])
    plt.plot(range(20, len(data)), entropy, 'r--', label='Entropy')
    
    plt.title('Entropy Waveform')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show() 