# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Edge Vector Field - Schwabot Mathematical Edge Detection and Vector Analysis
===========================================================================

Provides comprehensive edge detection, vector field analysis, and boundary
condition management for the Schwabot trading system.

Features:
- Mathematical edge detection algorithms
- Vector field analysis and visualization
- Boundary condition management
- Gradient-based signal processing
- Edge strength quantification
- Multi-dimensional vector operations
"""

import logging
from core.unified_math_system import unified_math
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


class EdgeType(Enum):
    """Types of detected edges."""
    PRICE_BREAKOUT = "price_breakout"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_EDGE = "volatility_edge"
    LIQUIDITY_EDGE = "liquidity_edge"
    ENTROPY_EDGE = "entropy_edge"
    FRACTAL_EDGE = "fractal_edge"


class VectorFieldType(Enum):
    """Types of vector fields."""
    GRADIENT = "gradient"
    CURL = "curl"
    DIVERGENCE = "divergence"
    POTENTIAL = "potential"
    STREAM = "stream"


@dataclass
class EdgePoint:
    """Single edge detection point."""
    timestamp: datetime
    edge_type: EdgeType
    strength: float  # 0.0 to 1.0
    position: Tuple[float, float]  # (x, y) coordinates
    direction: Tuple[float, float]  # (dx, dy) direction vector
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorField:
    """Vector field representation."""
    field_type: VectorFieldType
    dimensions: Tuple[int, int]
    vectors: np.ndarray  # Shape: (height, width, 2) for 2D vectors
    magnitude_map: np.ndarray  # Shape: (height, width)
    direction_map: np.ndarray  # Shape: (height, width)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EdgeVectorFieldConfig:
    """Configuration for edge vector field analysis."""
    detection_threshold: float = 0.3
    smoothing_factor: float = 0.1
    max_edge_points: int = 1000
    vector_field_resolution: Tuple[int, int] = (100, 100)
    enable_visualization: bool = False
    edge_confidence_threshold: float = 0.7


class EdgeVectorField:
    """
    Comprehensive edge vector field analysis system.
    
    Provides mathematical edge detection, vector field analysis,
    and boundary condition management for trading signals.
    """
    
    def __init__(self, config: Optional[EdgeVectorFieldConfig] = None):
        """Initialize edge vector field system."""
        self.config = config or EdgeVectorFieldConfig()
        
        # Core data structures
        self.edge_points: List[EdgePoint] = []
        self.vector_fields: Dict[str, VectorField] = {}
        self.edge_history: List[EdgePoint] = []
        
        # Analysis state
        self.current_field: Optional[VectorField] = None
        self.last_analysis: Optional[datetime] = None
        
        # Performance tracking
        self.analysis_count = 0
        self.detection_count = 0
        
        logger.info("Edge Vector Field system initialized")
    
    def detect_edges(self, data_matrix: np.ndarray,
                    data_type: str = "price") -> List[EdgePoint]:
        """Detect edges in a data matrix."""
        try:
            edges = []
            
            # Apply Sobel edge detection
            sobel_edges = self._apply_sobel_detection(data_matrix)
            
            # Apply Canny edge detection
            canny_edges = self._apply_canny_detection(data_matrix)
            
            # Combine edge detections
            combined_edges = self._combine_edge_detections(sobel_edges, canny_edges)
            
            # Extract edge points
            for i, j in np.argwhere(combined_edges > self.config.detection_threshold):
                edge_point = self._create_edge_point(i, j, combined_edges[i, j], data_type)
                if edge_point:
                    edges.append(edge_point)
            
            # Sort by strength and limit count
            edges.sort(key=lambda x: x.strength, reverse=True)
            edges = edges[:self.config.max_edge_points]
            
            # Update state
            self.edge_points = edges
            self.detection_count += 1
            
            logger.info(f"Detected {len(edges)} edges in {data_type} data")
            return edges
            
        except Exception as e:
            logger.error(f"Edge detection failed: {e}")
            return []
    
    def _apply_sobel_detection(self, data_matrix: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection."""
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Apply convolution
        grad_x = self._convolve2d(data_matrix, sobel_x)
        grad_y = self._convolve2d(data_matrix, sobel_y)
        
        # Calculate magnitude
        magnitude = unified_math.unified_math.sqrt(grad_x**2 + grad_y**2)
        
        return magnitude
    
    def _apply_canny_detection(self, data_matrix: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection."""
        # Gaussian smoothing
        smoothed = self._gaussian_smooth(data_matrix, sigma=1.0)
        
        # Gradient calculation
        grad_x = np.gradient(smoothed, axis=1)
        grad_y = np.gradient(smoothed, axis=0)
        
        # Magnitude and direction
        magnitude = unified_math.unified_math.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Non-maximum suppression
        suppressed = self._non_maximum_suppression(magnitude, direction)
        
        # Double thresholding
        edges = self._double_threshold(suppressed, low_threshold=0.1, high_threshold=0.3)
        
        return edges
    
    def _convolve2d(self, data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution implementation."""
        # Simple convolution for edge detection
        h, w = data.shape
        kh, kw = kernel.shape
        
        # Pad the data
        padded = np.pad(data, ((kh//2, kh//2), (kw//2, kw//2)), mode='edge')
        
        # Apply convolution
        result = np.zeros_like(data)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        return result
    
    def _gaussian_smooth(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian smoothing."""
        # Simple Gaussian kernel
        size = int(6 * sigma)
        if size % 2 == 0:
            size += 1
        
        x = np.arange(-size//2, size//2 + 1)
        kernel = unified_math.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        
        # Apply 1D convolution in both directions
        smoothed = self._convolve2d(data, kernel.reshape(1, -1))
        smoothed = self._convolve2d(smoothed, kernel.reshape(-1, 1))
        
        return smoothed
    
    def _non_maximum_suppression(self, magnitude: np.ndarray,
                                direction: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression."""
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        # Convert direction to degrees
        direction_deg = np.degrees(direction) % 180
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Determine gradient direction
                if (0 <= direction_deg[i, j] < 22.5) or (157.5 <= direction_deg[i, j] < 180):
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 22.5 <= direction_deg[i, j] < 67.5:
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif 67.5 <= direction_deg[i, j] < 112.5:
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:  # 112.5 <= direction_deg[i, j] < 157.5
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                
                # Suppress if not maximum
                if magnitude[i, j] >= unified_math.max(neighbors):
                    suppressed[i, j] = magnitude[i, j]
        
        return suppressed
    
    def _double_threshold(self, data: np.ndarray, low_threshold: float,
                         high_threshold: float) -> np.ndarray:
        """Apply double thresholding."""
        # Create binary image
        strong_edges = data > high_threshold
        weak_edges = (data >= low_threshold) & (data <= high_threshold)
        
        # Connect weak edges to strong edges
        result = np.zeros_like(data)
        result[strong_edges] = 1.0
        
        # Simple edge linking
        for i in range(1, data.shape[0]-1):
            for j in range(1, data.shape[1]-1):
                if weak_edges[i, j]:
                    # Check if connected to strong edge
                    neighborhood = strong_edges[i-1:i+2, j-1:j+2]
                    if np.any(neighborhood):
                        result[i, j] = 1.0
        
        return result
    
    def _combine_edge_detections(self, sobel_edges: np.ndarray,
                                canny_edges: np.ndarray) -> np.ndarray:
        """Combine different edge detection results."""
        # Weighted combination
        combined = (0.4 * sobel_edges + 0.6 * canny_edges)
        
        # Normalize
        if unified_math.unified_math.max(combined) > 0:
            combined = combined / unified_math.unified_math.max(combined)
        
        return combined
    
    def _create_edge_point(self, i: int, j: int, strength: float,
                          data_type: str) -> Optional[EdgePoint]:
        """Create an edge point from detection results."""
        try:
            # Determine edge type based on data type
            edge_type_map = {
                "price": EdgeType.PRICE_BREAKOUT,
                "volume": EdgeType.VOLUME_SPIKE,
                "volatility": EdgeType.VOLATILITY_EDGE,
                "liquidity": EdgeType.LIQUIDITY_EDGE,
                "entropy": EdgeType.ENTROPY_EDGE,
                "fractal": EdgeType.FRACTAL_EDGE
            }
            
            edge_type = edge_type_map.get(data_type, EdgeType.PRICE_BREAKOUT)
            
            # Calculate direction (simplified)
            direction = (1.0, 0.0)  # Default direction
            
            # Calculate confidence based on strength
            confidence = unified_math.min(1.0, strength * 1.5)
            
            return EdgePoint(
                timestamp=datetime.now(),
                edge_type=edge_type,
                strength=strength,
                position=(float(i), float(j)),
                direction=direction,
                confidence=confidence,
                metadata={"data_type": data_type}
            )
            
        except Exception as e:
            logger.error(f"Error creating edge point: {e}")
            return None
    
    def generate_vector_field(self, edge_points: List[EdgePoint],
                            field_type: VectorFieldType = VectorFieldType.GRADIENT,
                            dimensions: Optional[Tuple[int, int]] = None) -> VectorField:
        """Generate a vector field from edge points."""
        try:
            if dimensions is None:
                dimensions = self.config.vector_field_resolution
            
            # Initialize field
            vectors = np.zeros((dimensions[0], dimensions[1], 2))
            magnitude_map = np.zeros(dimensions)
            direction_map = np.zeros(dimensions)
            
            # Generate field based on type
            if field_type == VectorFieldType.GRADIENT:
                vectors, magnitude_map, direction_map = self._generate_gradient_field(
                    edge_points, dimensions)
            elif field_type == VectorFieldType.POTENTIAL:
                vectors, magnitude_map, direction_map = self._generate_potential_field(
                    edge_points, dimensions)
            elif field_type == VectorFieldType.STREAM:
                vectors, magnitude_map, direction_map = self._generate_stream_field(
                    edge_points, dimensions)
            
            # Create vector field
            vector_field = VectorField(
                field_type=field_type,
                dimensions=dimensions,
                vectors=vectors,
                magnitude_map=magnitude_map,
                direction_map=direction_map
            )
            
            # Store field
            field_id = f"{field_type.value}_{datetime.now().timestamp()}"
            self.vector_fields[field_id] = vector_field
            self.current_field = vector_field
            
            logger.info(f"Generated {field_type.value} vector field")
            return vector_field
            
        except Exception as e:
            logger.error(f"Vector field generation failed: {e}")
            return VectorField(
                field_type=field_type,
                dimensions=(10, 10),
                vectors=np.zeros((10, 10, 2)),
                magnitude_map=np.zeros((10, 10)),
                direction_map=np.zeros((10, 10))
            )
    
    def _generate_gradient_field(self, edge_points: List[EdgePoint],
                               dimensions: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate gradient vector field."""
        h, w = dimensions
        vectors = np.zeros((h, w, 2))
        magnitude_map = np.zeros((h, w))
        direction_map = np.zeros((h, w))
        
        # Create potential field from edge points
        potential = np.zeros((h, w))
        
        for edge in edge_points:
            x, y = edge.position
            # Scale position to field dimensions
            x_scaled = int(x * w / 100) if w > 0 else 0
            y_scaled = int(y * h / 100) if h > 0 else 0
            
            if 0 <= x_scaled < w and 0 <= y_scaled < h:
                # Add Gaussian potential
                for i in range(unified_math.max(0, y_scaled-5), unified_math.min(h, y_scaled+6)):
                    for j in range(unified_math.max(0, x_scaled-5), unified_math.min(w, x_scaled+6)):
                        dist = unified_math.sqrt((i-y_scaled)**2 + (j-x_scaled)**2)
                        potential[i, j] += edge.strength * unified_math.exp(-dist**2 / 10)
        
        # Calculate gradient
        grad_x = np.gradient(potential, axis=1)
        grad_y = np.gradient(potential, axis=0)
        
        # Set vectors
        vectors[:, :, 0] = -grad_x  # Negative gradient direction
        vectors[:, :, 1] = -grad_y
        
        # Calculate magnitude and direction
        magnitude_map = unified_math.unified_math.sqrt(grad_x**2 + grad_y**2)
        direction_map = np.arctan2(grad_y, grad_x)
        
        return vectors, magnitude_map, direction_map
    
    def _generate_potential_field(self, edge_points: List[EdgePoint],
                                dimensions: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate potential vector field."""
        # Similar to gradient but with different potential function
        return self._generate_gradient_field(edge_points, dimensions)
    
    def _generate_stream_field(self, edge_points: List[EdgePoint],
                             dimensions: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate stream vector field."""
        # Stream field follows edge directions
        h, w = dimensions
        vectors = np.zeros((h, w, 2))
        magnitude_map = np.zeros((h, w))
        direction_map = np.zeros((h, w))
        
        for edge in edge_points:
            x, y = edge.position
            dx, dy = edge.direction
            
            # Scale position
            x_scaled = int(x * w / 100) if w > 0 else 0
            y_scaled = int(y * h / 100) if h > 0 else 0
            
            if 0 <= x_scaled < w and 0 <= y_scaled < h:
                # Set stream direction
                vectors[y_scaled, x_scaled, 0] = dx
                vectors[y_scaled, x_scaled, 1] = dy
                magnitude_map[y_scaled, x_scaled] = edge.strength
                direction_map[y_scaled, x_scaled] = np.arctan2(dy, dx)
        
        return vectors, magnitude_map, direction_map
    
    def analyze_boundary_conditions(self, vector_field: VectorField) -> Dict[str, Any]:
        """Analyze boundary conditions of a vector field."""
        try:
            analysis = {
                "field_type": vector_field.field_type.value,
                "dimensions": vector_field.dimensions,
                "max_magnitude": float(unified_math.unified_math.max(vector_field.magnitude_map)),
                "min_magnitude": float(unified_math.unified_math.min(vector_field.magnitude_map)),
                "mean_magnitude": float(unified_math.unified_math.mean(vector_field.magnitude_map)),
                "std_magnitude": float(unified_math.unified_math.std(vector_field.magnitude_map)),
                "strong_regions": int(np.sum(vector_field.magnitude_map > 0.5)),
                "weak_regions": int(np.sum(vector_field.magnitude_map < 0.1)),
                "boundary_strength": self._calculate_boundary_strength(vector_field),
                "field_coherence": self._calculate_field_coherence(vector_field)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Boundary condition analysis failed: {e}")
            return {}
    
    def _calculate_boundary_strength(self, vector_field: VectorField) -> float:
        """Calculate boundary strength of vector field."""
        h, w = vector_field.dimensions
        
        # Check boundaries
        top_boundary = unified_math.unified_math.mean(vector_field.magnitude_map[0, :])
        bottom_boundary = unified_math.unified_math.mean(vector_field.magnitude_map[-1, :])
        left_boundary = unified_math.unified_math.mean(vector_field.magnitude_map[:, 0])
        right_boundary = unified_math.unified_math.mean(vector_field.magnitude_map[:, -1])
        
        # Average boundary strength
        boundary_strength = (top_boundary + bottom_boundary + left_boundary + right_boundary) / 4.0
        
        return float(boundary_strength)
    
    def _calculate_field_coherence(self, vector_field: VectorField) -> float:
        """Calculate field coherence."""
        # Calculate variance of directions
        direction_variance = unified_math.unified_math.var(vector_field.direction_map)
        
        # Coherence is inverse of variance (normalized)
        coherence = 1.0 / (1.0 + direction_variance)
        
        return float(coherence)
    
    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected edges."""
        if not self.edge_points:
            return {}
        
        edge_types = [edge.edge_type.value for edge in self.edge_points]
        strengths = [edge.strength for edge in self.edge_points]
        confidences = [edge.confidence for edge in self.edge_points]
        
        return {
            "total_edges": len(self.edge_points),
            "edge_type_distribution": {edge_type: edge_types.count(edge_type)
                                     for edge_type in set(edge_types)},
            "average_strength": float(unified_math.unified_math.mean(strengths)),
            "max_strength": float(unified_math.unified_math.max(strengths)),
            "average_confidence": float(unified_math.unified_math.mean(confidences)),
            "strong_edges": len([s for s in strengths if s > 0.7]),
            "weak_edges": len([s for s in strengths if s < 0.3])
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "analysis_count": self.analysis_count,
            "detection_count": self.detection_count,
            "current_edges": len(self.edge_points),
            "vector_fields": len(self.vector_fields),
            "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
            "current_field_type": self.current_field.field_type.value if self.current_field else None
        }


# Global edge vector field instance
edge_vector_field = EdgeVectorField()


def get_edge_vector_field() -> EdgeVectorField:
    """Get global edge vector field instance."""
    return edge_vector_field


def main() -> None:
    """Main function for testing edge vector field."""
    logging.basicConfig(level=logging.INFO)
    
    safe_print("🧪 Testing Edge Vector Field")
    safe_print("=" * 30)
    
    # Create edge vector field
    evf = EdgeVectorField()
    
    # Create test data
    test_data = np.random.rand(50, 50)
    
    # Detect edges
    edges = evf.detect_edges(test_data, "price")
    safe_print(f"✅ Detected {len(edges)} edges")
    
    # Generate vector field
    vector_field = evf.generate_vector_field(edges, VectorFieldType.GRADIENT)
    safe_print(f"✅ Generated {vector_field.field_type.value} vector field")
    
    # Analyze boundary conditions
    analysis = evf.analyze_boundary_conditions(vector_field)
    safe_print(f"📊 Boundary analysis: {analysis['boundary_strength']:.3f} strength")
    
    # Get statistics
    stats = evf.get_edge_statistics()
    safe_print(f"📈 Edge statistics: {stats['total_edges']} total edges")
    
    safe_print("Edge vector field test completed!")


if __name__ == "__main__":
    main() 