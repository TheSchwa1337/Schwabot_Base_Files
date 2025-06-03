"""
Example usage of TesseractVisualizer with synthetic data
"""

import numpy as np
import time
from core.tesseract_visualizer import TesseractVisualizer

def generate_synthetic_tensor(dims, t):
    """Generate synthetic tensor data with time-varying patterns"""
    x, y, z, w = np.meshgrid(
        np.linspace(-1, 1, dims[0]),
        np.linspace(-1, 1, dims[1]),
        np.linspace(-1, 1, dims[2]),
        np.linspace(-1, 1, dims[3])
    )
    
    # Create time-varying patterns
    thermal = np.sin(x + t) * np.cos(y + t/2) * np.sin(z + t/3) * np.cos(w + t/4)
    memory = np.cos(x - t) * np.sin(y - t/2) * np.cos(z - t/3) * np.sin(w - t/4)
    profit = np.sin(x * t) * np.cos(y * t/2) * np.sin(z * t/3) * np.cos(w * t/4)
    
    return thermal, memory, profit

def main():
    # Initialize visualizer
    visualizer = TesseractVisualizer(
        tensor_dims=(10, 10, 10, 5),
        update_interval=0.1,
        history_size=1000,
        debug_mode=True
    )
    
    # Start visualization
    visualizer.start_visualization()
    
    try:
        start_time = time.time()
        while True:
            # Calculate current time
            current_time = time.time() - start_time
            
            # Generate synthetic data
            thermal, memory, profit = generate_synthetic_tensor(
                visualizer.tensor_dims,
                current_time
            )
            
            # Update visualization
            visualizer.update_visualization(
                thermal_tensor=thermal,
                memory_tensor=memory,
                profit_tensor=profit,
                current_time=current_time
            )
            
            # Print debug info every second
            if int(current_time) % 1 == 0:
                print("\nDebug Info:")
                print(visualizer.get_debug_info())
                print("\nVisualization Stats:")
                print(visualizer.get_visualization_stats())
            
            # Sleep to maintain update interval
            time.sleep(visualizer.update_interval)
            
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    finally:
        visualizer.stop_visualization()

if __name__ == "__main__":
    main() 