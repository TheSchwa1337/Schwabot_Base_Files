# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import numpy as np
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Bitmap Engine - Mathematical Bitmap Processing for Schwabot.

This module provides comprehensive bitmap operations, pattern recognition,
and image processing functions used in Schwabot's trading logic for visual
data analysis and chart pattern detection.

Mathematical Foundation:
- Bitmap convolution: C(x,y) = Σ Σ I(i,j) * K(x-i, y-j)
- Pattern matching: S = Σ Σ |I(i,j) - T(i,j)| / (width * height)
- Edge detection: ∇I = √((∂I/∂x)² + (∂I/∂y)²)
- Histogram analysis: H(k) = Σ δ(I(i,j) - k)
"""

import logging
# from core.unified_math_system import unified_math  # F811: duplicate import
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class BitmapPattern:
    """Bitmap pattern with metadata."""
    pattern: np.ndarray
    name: str
    confidence: float
    location: Tuple[int, int]
    scale: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BitmapAnalysis:
    """Bitmap analysis results."""
    edge_density: float
    pattern_count: int
    histogram_entropy: float
    texture_score: float
    symmetry_score: float
    patterns: List[BitmapPattern] = field(default_factory=list)

class BitmapEngine:
    """Mathematical bitmap processing for trading pattern recognition."""

    def __init__(self):
        self.max_pattern_size = 64
        self.edge_threshold = 0.1
        self.pattern_templates = self._initialize_templates()
        logger.info("BitmapEngine initialized")

    def _initialize_templates(self) -> Dict[str, np.ndarray]:
        """Initialize common trading pattern templates."""
        templates = {}

        # Simple patterns for testing
        # Bull flag pattern
        bull_flag = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ], dtype=float)
        templates['bull_flag'] = bull_flag

        # Bear flag pattern
        bear_flag = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ], dtype=float)
        templates['bear_flag'] = bear_flag

        # Triangle pattern
        triangle = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=float)
        templates['triangle'] = triangle

        return templates

    def load_bitmap(self, data: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Load bitmap data into numpy array.

        Parameters:
        -----------
        data : Union[np.ndarray, List[List[float]]]
            Bitmap data

        Returns:
        --------
        np.ndarray
            Normalized bitmap array
        """
        try:
            if isinstance(data, list):
                bitmap = np.array(data, dtype=float)
            else:
                bitmap = data.astype(float)

            # Normalize to [0, 1] range
            if bitmap.max() > bitmap.min():
                bitmap = (bitmap - bitmap.min()) / (bitmap.max() - bitmap.min())

            return bitmap

        except Exception as e:
            logger.error(f"Error loading bitmap: {e}")
            return np.zeros((10, 10), dtype=float)

    def detect_edges(self, bitmap: np.ndarray) -> np.ndarray:
        """
        Detect edges in bitmap using Sobel operators.

        Mathematical Formula:
        ∇I = √((∂I/∂x)² + (∂I/∂y)²)

        Parameters:
        -----------
        bitmap : np.ndarray
            Input bitmap

        Returns:
        --------
        np.ndarray
            Edge map
        """
        try:
            # Sobel operators
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

            # Convolve with Sobel operators
            grad_x = self._convolve2d(bitmap, sobel_x)
            grad_y = self._convolve2d(bitmap, sobel_y)

            # Calculate gradient magnitude
            edge_map = unified_math.unified_math.sqrt(grad_x**2 + grad_y**2)

            # Normalize
            if edge_map.max() > 0:
                edge_map = edge_map / edge_map.max()

            return edge_map

        except Exception as e:
            logger.error(f"Error detecting edges: {e}")
            return np.zeros_like(bitmap)

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution operation."""
        try:
            # Simple convolution implementation
            h, w = image.shape
            kh, kw = kernel.shape

            # Pad image
            pad_h, pad_w = kh // 2, kw // 2
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

            # Convolve
            result = np.zeros_like(image)
            for i in range(h):
                for j in range(w):
                    result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)

            return result

        except Exception as e:
            logger.error(f"Error in convolution: {e}")
            return np.zeros_like(image)

    def calculate_histogram(self, bitmap: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate histogram of bitmap values.

        Mathematical Formula:
        H(k) = Σ δ(I(i,j) - k)

        Parameters:
        -----------
        bitmap : np.ndarray
            Input bitmap
        bins : int
            Number of histogram bins

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (histogram, bin_edges)
        """
        try:
            histogram, bin_edges = np.histogram(bitmap.flatten(), bins=bins, range=(0, 1))
            return histogram, bin_edges

        except Exception as e:
            logger.error(f"Error calculating histogram: {e}")
            return np.zeros(bins), np.linspace(0, 1, bins + 1)

    def calculate_entropy(self, bitmap: np.ndarray) -> float:
        """
        Calculate Shannon entropy of bitmap.

        Mathematical Formula:
        H = -Σ p_i * log2(p_i)

        Parameters:
        -----------
        bitmap : np.ndarray
            Input bitmap

        Returns:
        --------
        float
            Entropy value
        """
        try:
            histogram, _ = self.calculate_histogram(bitmap, bins=256)

            # Normalize histogram to probabilities
            total_pixels = histogram.sum()
            if total_pixels == 0:
                return 0.0

            probabilities = histogram / total_pixels

            # Calculate entropy
            entropy = 0.0
            for p in probabilities:
                if p > 0:
                    entropy -= p * np.log2(p)

            return entropy

        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0

    def detect_patterns(self, bitmap: np.ndarray,
                       threshold: float = 0.8) -> List[BitmapPattern]:
        """
        Detect patterns in bitmap using template matching.

        Parameters:
        -----------
        bitmap : np.ndarray
            Input bitmap
        threshold : float
            Matching threshold

        Returns:
        --------
        List[BitmapPattern]
            Detected patterns
        """
        try:
            patterns = []

            for pattern_name, template in self.pattern_templates.items():
                # Template matching
                matches = self._template_matching(bitmap, template, threshold)

                for match in matches:
                    pattern = BitmapPattern(
                        pattern=template,
                        name=pattern_name,
                        confidence=match['confidence'],
                        location=match['location'],
                        scale=match['scale'],
                        metadata=match.get('metadata', {})
                    )
                    patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []

    def _template_matching(self, bitmap: np.ndarray, template: np.ndarray,
                          threshold: float) -> List[Dict[str, Any]]:
        """Template matching using normalized cross-correlation."""
        try:
            matches = []
            h, w = bitmap.shape
            th, tw = template.shape

            # Normalize template
            template_norm = template - template.mean()
            template_std = template.std()

            if template_std == 0:
                return matches

            template_norm = template_norm / template_std

            # Slide template over bitmap
            for i in range(h - th + 1):
                for j in range(w - tw + 1):
                    # Extract window
                    window = bitmap[i:i+th, j:j+tw]

                    # Normalize window
                    window_norm = window - window.mean()
                    window_std = window.std()

                    if window_std == 0:
                        continue

                    window_norm = window_norm / window_std

                    # Calculate correlation
                    correlation = np.sum(template_norm * window_norm) / (th * tw)

                    if correlation >= threshold:
                        matches.append({
                            'confidence': correlation,
                            'location': (i, j),
                            'scale': 1.0,
                            'metadata': {'correlation': correlation}
                        })

            return matches

        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            return []

    def analyze_texture(self, bitmap: np.ndarray) -> float:
        """
        Analyze texture complexity of bitmap.

        Parameters:
        -----------
        bitmap : np.ndarray
            Input bitmap

        Returns:
        --------
        float
            Texture complexity score [0, 1]
        """
        try:
            # Calculate edge density
            edge_map = self.detect_edges(bitmap)
            edge_density = unified_math.unified_math.mean(edge_map)

            # Calculate local variance
            kernel = np.ones((3, 3)) / 9
            smoothed = self._convolve2d(bitmap, kernel)
            variance = unified_math.unified_math.var(bitmap - smoothed)

            # Calculate entropy
            entropy = self.calculate_entropy(bitmap)

            # Combine metrics
            texture_score = (edge_density * 0.4 +
                           unified_math.min(1.0, variance * 10) * 0.3 +
                           unified_math.min(1.0, entropy / 8.0) * 0.3)

            return unified_math.max(0.0, unified_math.min(1.0, texture_score))

        except Exception as e:
            logger.error(f"Error analyzing texture: {e}")
            return 0.5

    def calculate_symmetry(self, bitmap: np.ndarray) -> float:
        """
        Calculate symmetry score of bitmap.

        Parameters:
        -----------
        bitmap : np.ndarray
            Input bitmap

        Returns:
        --------
        float
            Symmetry score [0, 1]
        """
        try:
            h, w = bitmap.shape

            # Horizontal symmetry
            if h > 1:
                top_half = bitmap[:h//2, :]
                bottom_half = bitmap[h//2:, :]
                if top_half.shape != bottom_half.shape:
                    bottom_half = bottom_half[:top_half.shape[0], :]
                horizontal_symmetry = 1.0 - unified_math.unified_math.mean(unified_math.unified_math.abs(top_half - bottom_half))
            else:
                horizontal_symmetry = 1.0

            # Vertical symmetry
            if w > 1:
                left_half = bitmap[:, :w//2]
                right_half = bitmap[:, w//2:]
                if left_half.shape != right_half.shape:
                    right_half = right_half[:, :left_half.shape[1]]
                vertical_symmetry = 1.0 - unified_math.unified_math.mean(unified_math.unified_math.abs(left_half - right_half))
            else:
                vertical_symmetry = 1.0

            # Combined symmetry score
            symmetry_score = (horizontal_symmetry + vertical_symmetry) / 2.0

            return unified_math.max(0.0, unified_math.min(1.0, symmetry_score))

        except Exception as e:
            logger.error(f"Error calculating symmetry: {e}")
            return 0.5

    def analyze_bitmap(self, bitmap: np.ndarray) -> BitmapAnalysis:
        """
        Comprehensive bitmap analysis.

        Parameters:
        -----------
        bitmap : np.ndarray
            Input bitmap

        Returns:
        --------
        BitmapAnalysis
            Analysis results
        """
        try:
            # Calculate various metrics
            edge_map = self.detect_edges(bitmap)
            edge_density = unified_math.unified_math.mean(edge_map)

            patterns = self.detect_patterns(bitmap)
            pattern_count = len(patterns)

            histogram_entropy = self.calculate_entropy(bitmap)
            texture_score = self.analyze_texture(bitmap)
            symmetry_score = self.calculate_symmetry(bitmap)

            return BitmapAnalysis(
                edge_density=edge_density,
                pattern_count=pattern_count,
                histogram_entropy=histogram_entropy,
                texture_score=texture_score,
                symmetry_score=symmetry_score,
                patterns=patterns
            )

        except Exception as e:
            logger.error(f"Error analyzing bitmap: {e}")
            return BitmapAnalysis(
                edge_density=0.0,
                pattern_count=0,
                histogram_entropy=0.0,
                texture_score=0.0,
                symmetry_score=0.0,
                patterns=[]
            )

    def create_test_bitmap(self, size: Tuple[int, int] = (32, 32)) -> np.ndarray:
        """Create a test bitmap for demonstration."""
        try:
            h, w = size
            bitmap = np.random.random((h, w))

            # Add some structure
            for i in range(h):
                for j in range(w):
                    # Add gradient
                    bitmap[i, j] += (i + j) / (h + w)

                    # Add some patterns
                    if i % 8 == 0 or j % 8 == 0:
                        bitmap[i, j] += 0.3

            # Normalize
            bitmap = np.clip(bitmap, 0, 1)

            return bitmap

        except Exception as e:
            logger.error(f"Error creating test bitmap: {e}")
            return np.random.random(size)

def main() -> None:
    """Test function for BitmapEngine."""
    safe_print("🧮 Testing Bitmap Engine...")

    engine = BitmapEngine()

    # Create test bitmap
    test_bitmap = engine.create_test_bitmap((64, 64))
    safe_print(f"Test bitmap shape: {test_bitmap.shape}")
    safe_print(f"Bitmap range: [{test_bitmap.min():.3f}, {test_bitmap.max():.3f}]")

    # Analyze bitmap
    analysis = engine.analyze_bitmap(test_bitmap)

    safe_print("\nBitmap Analysis:")
    safe_print(f"  Edge density: {analysis.edge_density:.3f}")
    safe_print(f"  Pattern count: {analysis.pattern_count}")
    safe_print(f"  Histogram entropy: {analysis.histogram_entropy:.3f}")
    safe_print(f"  Texture score: {analysis.texture_score:.3f}")
    safe_print(f"  Symmetry score: {analysis.symmetry_score:.3f}")

    # Test edge detection
    edge_map = engine.detect_edges(test_bitmap)
    safe_print(f"  Edge map range: [{edge_map.min():.3f}, {edge_map.max():.3f}]")

    # Test histogram
    histogram, bin_edges = engine.calculate_histogram(test_bitmap)
    safe_print(f"  Histogram shape: {histogram.shape}")
    safe_print(f"  Histogram sum: {histogram.sum()}")

    # Test pattern detection
    patterns = engine.detect_patterns(test_bitmap, threshold=0.7)
    safe_print(f"  Detected patterns: {len(patterns)}")
    for pattern in patterns[:3]:  # Show first 3 patterns
        safe_print(f"    - {pattern.name}: confidence={pattern.confidence:.3f}")

    return 0

if __name__ == "__main__":
    exit(main())
