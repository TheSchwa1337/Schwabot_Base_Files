"""
Tesseract Visualizer
==================

Handles visualization of high-dimensional tensor data with advanced debugging capabilities.
Provides real-time visualization of 4D/5D tensor projections and pattern analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import threading
import time

from .tensor_visualization_controller import TensorVisualizationController
from .tesseract_processor import TesseractProcessor

@dataclass
class TesseractMetrics:
    """Container for tesseract-specific metrics"""
    magnitude: float
    centroid_distance: float
    axis_correlation: float
    stability: float
    harmonic_ratio: float
    primary_dominance: float
    dimensional_spread: float
    coherence: float
    homeostasis: float
    entropy: float
    pattern_variance: float

class TesseractVisualizer:
    """Manages visualization of high-dimensional tensor data with debugging capabilities"""
    
    def __init__(
        self,
        tensor_dims: Tuple[int, int, int, int] = (10, 10, 10, 5),
        update_interval: float = 0.1,
        history_size: int = 1000,
        debug_mode: bool = False
    ):
        self.tensor_dims = tensor_dims
        self.update_interval = update_interval
        self.history_size = history_size
        self.debug_mode = debug_mode
        
        # Initialize components
        self.tensor_controller = TensorVisualizationController(
            tensor_dims=tensor_dims[:3],  # 3D for visualization
            update_interval=update_interval
        )
        self.tesseract_processor = TesseractProcessor()
        
        # Initialize state tracking
        self.metrics_history: List[TesseractMetrics] = []
        self.pattern_history: List[np.ndarray] = []
        self.projection_history: List[np.ndarray] = []
        
        # Debug state
        self.debug_data = {
            'projection_errors': [],
            'dimension_mismatches': [],
            'performance_metrics': [],
            'last_update': None
        }
        
        # Initialize logging
        logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Control flags
        self.running = False
        self.update_thread = None
        
    def _project_4d_to_3d(self, tensor_4d: np.ndarray, method: str = 'slice') -> np.ndarray:
        """
        Project 4D tensor to 3D space using specified method.
        
        Args:
            tensor_4d: Input 4D tensor
            method: Projection method ('slice', 'sum', 'max', 'average')
            
        Returns:
            3D tensor projection
        """
        if tensor_4d.shape != self.tensor_dims:
            self.logger.warning(f"Input tensor shape {tensor_4d.shape} doesn't match expected {self.tensor_dims}")
            # Attempt to resize
            try:
                tensor_4d = np.resize(tensor_4d, self.tensor_dims)
            except ValueError as e:
                self.logger.error(f"Failed to resize tensor: {e}")
                return np.zeros(self.tensor_dims[:3])
        
        if method == 'slice':
            # Take middle slice along 4th dimension
            slice_idx = self.tensor_dims[3] // 2
            return tensor_4d[:, :, :, slice_idx]
        elif method == 'sum':
            return np.sum(tensor_4d, axis=3)
        elif method == 'max':
            return np.max(tensor_4d, axis=3)
        elif method == 'average':
            return np.mean(tensor_4d, axis=3)
        else:
            self.logger.warning(f"Unknown projection method: {method}. Using slice.")
            return tensor_4d[:, :, :, 0]
    
    def _calculate_metrics(self, tensor_4d: np.ndarray) -> TesseractMetrics:
        """Calculate comprehensive metrics for the 4D tensor"""
        # Basic geometric properties
        magnitude = np.sqrt(np.sum(tensor_4d**2))
        centroid = np.mean(tensor_4d, axis=(0,1,2,3))
        centroid_distance = np.sqrt(np.sum((tensor_4d - centroid)**2))
        
        # Dimensional analysis
        primary_axis = tensor_4d[:,:,:,0]  # First dimension
        secondary_axis = tensor_4d[:,:,:,-1]  # Last dimension
        axis_correlation = np.corrcoef(primary_axis.flatten(), secondary_axis.flatten())[0,1]
        
        # Pattern stability
        stability = 1.0 / (1.0 + np.var(tensor_4d))
        
        # Harmonic analysis
        fft_result = np.fft.fftn(tensor_4d)
        harmonic_ratio = np.abs(fft_result).mean()
        
        # Entropy calculation
        flat_tensor = tensor_4d.flatten()
        normalized = flat_tensor / (np.sum(flat_tensor) + 1e-10)
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        
        return TesseractMetrics(
            magnitude=float(magnitude),
            centroid_distance=float(centroid_distance),
            axis_correlation=float(axis_correlation),
            stability=float(stability),
            harmonic_ratio=float(harmonic_ratio),
            primary_dominance=float(np.mean(primary_axis)),
            dimensional_spread=float(np.max(tensor_4d) - np.min(tensor_4d)),
            coherence=float(stability),  # Using stability as coherence measure
            homeostasis=float(1.0 / (1.0 + entropy)),  # Inverse entropy as homeostasis
            entropy=float(entropy),
            pattern_variance=float(np.var(tensor_4d))
        )
    
    def update_visualization(
        self,
        thermal_tensor: np.ndarray,
        memory_tensor: np.ndarray,
        profit_tensor: np.ndarray,
        current_time: float
    ):
        """Update visualization with new tensor data"""
        try:
            # Project 4D tensors to 3D
            thermal_3d = self._project_4d_to_3d(thermal_tensor)
            memory_3d = self._project_4d_to_3d(memory_tensor)
            profit_3d = self._project_4d_to_3d(profit_tensor)
            
            # Calculate metrics
            metrics = self._calculate_metrics(profit_tensor)  # Using profit tensor for metrics
            
            # Update history
            self.metrics_history.append(metrics)
            self.pattern_history.append(profit_tensor)
            self.projection_history.append(profit_3d)
            
            # Trim history if needed
            if len(self.metrics_history) > self.history_size:
                self.metrics_history.pop(0)
                self.pattern_history.pop(0)
                self.projection_history.pop(0)
            
            # Update visualization controller
            self.tensor_controller.update_tensors(
                thermal_tensor=thermal_3d,
                memory_tensor=memory_3d,
                profit_tensor=profit_3d,
                t=current_time
            )
            
            # Update debug data
            if self.debug_mode:
                self.debug_data['last_update'] = datetime.now()
                self.debug_data['performance_metrics'].append({
                    'timestamp': current_time,
                    'memory_usage': profit_tensor.nbytes,
                    'projection_time': time.time() - current_time
                })
            
        except Exception as e:
            self.logger.error(f"Error updating visualization: {e}")
            if self.debug_mode:
                self.debug_data['projection_errors'].append({
                    'timestamp': current_time,
                    'error': str(e),
                    'tensor_shapes': {
                        'thermal': thermal_tensor.shape,
                        'memory': memory_tensor.shape,
                        'profit': profit_tensor.shape
                    }
                })
    
    def start_visualization(self):
        """Start real-time visualization"""
        if self.running:
            return
            
        self.running = True
        self.tensor_controller.start_visualization()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_visualization(self):
        """Stop real-time visualization"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        self.tensor_controller.stop_visualization()
    
    def _update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                # Generate test data if no real data is available
                if not self.pattern_history:
                    test_tensor = np.random.rand(*self.tensor_dims)
                    self.update_visualization(
                        thermal_tensor=test_tensor,
                        memory_tensor=test_tensor,
                        profit_tensor=test_tensor,
                        current_time=time.time()
                    )
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the visualization state"""
        if not self.debug_mode:
            return {"debug_mode": False}
            
        return {
            "debug_mode": True,
            "last_update": self.debug_data['last_update'],
            "projection_errors": len(self.debug_data['projection_errors']),
            "dimension_mismatches": len(self.debug_data['dimension_mismatches']),
            "performance_metrics": self.debug_data['performance_metrics'][-10:],  # Last 10 metrics
            "history_size": len(self.metrics_history),
            "current_metrics": self.metrics_history[-1] if self.metrics_history else None
        }
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """Get statistics about the visualization state"""
        return {
            "total_updates": len(self.metrics_history),
            "current_metrics": self.metrics_history[-1] if self.metrics_history else None,
            "pattern_history_size": len(self.pattern_history),
            "projection_history_size": len(self.projection_history),
            "tensor_dimensions": self.tensor_dims,
            "update_interval": self.update_interval,
            "running": self.running
        }

# Example usage
if __name__ == "__main__":
    # Create visualizer with debug mode
    visualizer = TesseractVisualizer(
        tensor_dims=(10, 10, 10, 5),
        update_interval=0.1,
        debug_mode=True
    )
    
    # Start visualization
    visualizer.start_visualization()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            # Print debug info every second
            if visualizer.debug_mode:
                print("\nDebug Info:")
                print(visualizer.get_debug_info())
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    finally:
        visualizer.stop_visualization() 