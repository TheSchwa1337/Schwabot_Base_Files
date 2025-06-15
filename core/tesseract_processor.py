"""
Tesseract Processor
==================

Lightweight processor for 4D tensor operations and tesseract mathematical transformations.
Maintains interface compatibility for the mathematical synthesis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

@dataclass
class TesseractState:
    """State container for tesseract processing"""
    tensor_4d: np.ndarray
    timestamp: float
    magnitude: float
    coherence: float
    stability: float

class TesseractProcessor:
    """Lightweight processor for 4D tensor operations"""
    
    def __init__(self, dimensions: Tuple[int, int, int, int] = (10, 10, 10, 5)):
        self.dimensions = dimensions
        self.state_history: List[TesseractState] = []
        self.max_history = 100
        
        # Initialize transformation matrices
        self.rotation_matrices = self._initialize_rotation_matrices()
        self.projection_matrices = self._initialize_projection_matrices()
        
    def _initialize_rotation_matrices(self) -> Dict[str, np.ndarray]:
        """Initialize 4D rotation matrices"""
        # Simple 4D rotation matrices for tesseract operations
        matrices = {}
        
        # XY plane rotation
        matrices['xy'] = np.eye(4)
        angle = np.pi / 4
        matrices['xy'][0, 0] = np.cos(angle)
        matrices['xy'][0, 1] = -np.sin(angle)
        matrices['xy'][1, 0] = np.sin(angle)
        matrices['xy'][1, 1] = np.cos(angle)
        
        # ZW plane rotation
        matrices['zw'] = np.eye(4)
        matrices['zw'][2, 2] = np.cos(angle)
        matrices['zw'][2, 3] = -np.sin(angle)
        matrices['zw'][3, 2] = np.sin(angle)
        matrices['zw'][3, 3] = np.cos(angle)
        
        return matrices
        
    def _initialize_projection_matrices(self) -> Dict[str, np.ndarray]:
        """Initialize projection matrices for dimensional reduction"""
        matrices = {}
        
        # 4D to 3D projection (orthographic)
        matrices['4d_to_3d'] = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        
        # 4D to 2D projection
        matrices['4d_to_2d'] = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        return matrices
    
    def process_tensor(self, tensor_4d: np.ndarray) -> TesseractState:
        """Process a 4D tensor and return tesseract state"""
        # Ensure tensor has correct dimensions
        if tensor_4d.shape != self.dimensions:
            tensor_4d = np.resize(tensor_4d, self.dimensions)
        
        # Calculate basic metrics
        magnitude = np.sqrt(np.sum(tensor_4d**2))
        coherence = self._calculate_coherence(tensor_4d)
        stability = self._calculate_stability(tensor_4d)
        
        # Create state
        state = TesseractState(
            tensor_4d=tensor_4d,
            timestamp=time.time(),
            magnitude=magnitude,
            coherence=coherence,
            stability=stability
        )
        
        # Store in history
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
            
        return state
    
    def _calculate_coherence(self, tensor_4d: np.ndarray) -> float:
        """Calculate coherence measure for 4D tensor"""
        # Simple coherence based on variance
        return 1.0 / (1.0 + np.var(tensor_4d))
    
    def _calculate_stability(self, tensor_4d: np.ndarray) -> float:
        """Calculate stability measure for 4D tensor"""
        # Stability based on gradient magnitude
        gradients = []
        for axis in range(4):
            grad = np.gradient(tensor_4d, axis=axis)
            gradients.append(np.mean(np.abs(grad)))
        
        avg_gradient = np.mean(gradients)
        return 1.0 / (1.0 + avg_gradient)
    
    def rotate_tesseract(self, tensor_4d: np.ndarray, rotation_type: str = 'xy') -> np.ndarray:
        """Apply 4D rotation to tesseract"""
        if rotation_type not in self.rotation_matrices:
            rotation_type = 'xy'
            
        rotation_matrix = self.rotation_matrices[rotation_type]
        
        # Apply rotation to each point in the tensor
        original_shape = tensor_4d.shape
        flattened = tensor_4d.reshape(-1, 4) if tensor_4d.shape[-1] == 4 else tensor_4d.reshape(-1, 1)
        
        if flattened.shape[1] == 4:
            rotated = np.dot(flattened, rotation_matrix.T)
            return rotated.reshape(original_shape)
        else:
            # If not 4D points, apply rotation to reshaped data
            reshaped = np.resize(tensor_4d, (*original_shape[:-1], 4))
            flattened = reshaped.reshape(-1, 4)
            rotated = np.dot(flattened, rotation_matrix.T)
            return rotated.reshape(reshaped.shape)
    
    def project_to_3d(self, tensor_4d: np.ndarray) -> np.ndarray:
        """Project 4D tensor to 3D space"""
        projection_matrix = self.projection_matrices['4d_to_3d']
        
        # Handle different tensor shapes
        if len(tensor_4d.shape) == 4:
            # Take slice along 4th dimension
            return tensor_4d[:, :, :, 0]
        else:
            # Apply projection matrix if tensor represents 4D points
            original_shape = tensor_4d.shape
            if original_shape[-1] == 4:
                flattened = tensor_4d.reshape(-1, 4)
                projected = np.dot(flattened, projection_matrix.T)
                return projected.reshape((*original_shape[:-1], 3))
            else:
                # Simple dimensional reduction
                return tensor_4d[:, :, :] if len(tensor_4d.shape) >= 3 else tensor_4d
    
    def project_to_2d(self, tensor_4d: np.ndarray) -> np.ndarray:
        """Project 4D tensor to 2D space"""
        projection_matrix = self.projection_matrices['4d_to_2d']
        
        # Handle different tensor shapes
        if len(tensor_4d.shape) == 4:
            # Take slice along 3rd and 4th dimensions
            return tensor_4d[:, :, 0, 0]
        else:
            # Apply projection matrix if tensor represents 4D points
            original_shape = tensor_4d.shape
            if original_shape[-1] == 4:
                flattened = tensor_4d.reshape(-1, 4)
                projected = np.dot(flattened, projection_matrix.T)
                return projected.reshape((*original_shape[:-1], 2))
            else:
                # Simple dimensional reduction
                return tensor_4d[:, :] if len(tensor_4d.shape) >= 2 else tensor_4d
    
    def calculate_tesseract_volume(self, tensor_4d: np.ndarray) -> float:
        """Calculate 4D volume (hypervolume) of tesseract"""
        # Simple hypervolume calculation
        return np.prod(tensor_4d.shape) * np.mean(np.abs(tensor_4d))
    
    def get_tesseract_vertices(self, center: np.ndarray, size: float = 1.0) -> np.ndarray:
        """Generate vertices of a tesseract centered at given point"""
        # Generate 16 vertices of a tesseract (4D cube)
        vertices = []
        for i in range(16):
            vertex = []
            for j in range(4):
                # Binary representation to get all combinations
                vertex.append(((i >> j) & 1) * size - size/2)
            vertices.append(np.array(vertex) + center)
        
        return np.array(vertices)
    
    def get_state_history(self) -> List[TesseractState]:
        """Get processing state history"""
        return self.state_history.copy()
    
    def get_latest_state(self) -> Optional[TesseractState]:
        """Get latest processing state"""
        return self.state_history[-1] if self.state_history else None
    
    def clear_history(self):
        """Clear state history"""
        self.state_history.clear()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current processing metrics"""
        if not self.state_history:
            return {}
            
        latest = self.state_history[-1]
        return {
            'magnitude': latest.magnitude,
            'coherence': latest.coherence,
            'stability': latest.stability,
            'history_size': len(self.state_history),
            'processing_rate': len(self.state_history) / max(1, time.time() - self.state_history[0].timestamp)
        } 