from __future__ import annotations

#!/usr/bin/env python3
"""
Tensor Engine - Advanced Tensor Processing and Pattern Analysis
==============================================================

Provides advanced tensor processing, pattern analysis, and dimensionality
reduction capabilities for the Schwabot trading system.

Core Functions:
- create_tensor_space: Create multi-dimensional tensor spaces
- analyze_tensor_patterns: Analyze patterns in tensor data
- compute_tensor_statistics: Compute comprehensive tensor statistics
- tensor_pattern_matching: Perform pattern matching on tensors
- tensor_clustering: Perform clustering on tensor data
- tensor_dimensionality_reduction: Reduce tensor dimensionality
"""


import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Union, Dict, Any
import logging
from scipy.linalg import svd, eig
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


class TensorEngine:
    """
    Advanced Tensor Engine for Schwabot Trading System.

    This engine provides comprehensive tensor processing and analysis
    capabilities for multi-dimensional data.
    """

    def __init__(self):
        """Initialize the tensor engine."""
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.max_dimensions = 10  # Maximum tensor dimensions
        self.default_cluster_count = 3  # Default number of clusters

        logger.info("Tensor Engine initialized")

    def create_tensor_space(self, data: NDArray, dimensions: Tuple[int, ...]) -> NDArray:
        """
        Create multi-dimensional tensor space from data.

        Args:
            data: Input data array
            dimensions: Target tensor dimensions

        Returns:
            Reshaped tensor
        """
        try:
            # Flatten data if needed
            if data.ndim > 1:
                data_flat = data.flatten()
            else:
                data_flat = data

            # Calculate total elements needed
            total_elements = np.prod(dimensions)

            # Pad or truncate data to fit dimensions
            if len(data_flat) < total_elements:
                # Pad with zeros
                padded_data = np.zeros(total_elements)
                padded_data[:len(data_flat)] = data_flat
                data_flat = padded_data
            else:
                # Truncate to fit
                data_flat = data_flat[:total_elements]

            # Reshape to target dimensions
            tensor = data_flat.reshape(dimensions)

            return tensor

        except Exception as e:
            logger.error(f"Tensor space creation failed: {e}")
            return np.zeros(dimensions)

    def analyze_tensor_patterns(self, tensor: NDArray) -> Dict[str, Any]:
        """
        Analyze patterns in tensor data.

        Args:
            tensor: Input tensor

        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            results = {
                'shape': tensor.shape,
                'rank': tensor.ndim,
                'size': tensor.size,
                'sparsity': 0.0,
                'symmetry_score': 0.0,
                'pattern_types': [],
                'dominant_pattern': None
            }

            # Calculate sparsity
            zero_elements = np.sum(tensor == 0)
            results['sparsity'] = float(zero_elements / tensor.size)

            # Check for symmetry (for 2D tensors)
            if tensor.ndim == 2:
                symmetry_score = np.mean(np.abs(tensor - tensor.T))
                results['symmetry_score'] = float(1.0 / (1.0 + symmetry_score))

            # Analyze patterns
            pattern_types = []

            # Check for diagonal patterns
            if tensor.ndim == 2:
                diagonal_strength = np.sum(np.abs(np.diag(tensor)))
                total_strength = np.sum(np.abs(tensor))
                if total_strength > 0:
                    diagonal_ratio = diagonal_strength / total_strength
                    if diagonal_ratio > 0.5:
                        pattern_types.append('diagonal')

            # Check for block patterns
            if tensor.ndim == 2:
                # Simple block detection
                block_score = self._calculate_block_score(tensor)
                if block_score > 0.3:
                    pattern_types.append('block')

            # Check for random patterns
            if results['sparsity'] < 0.1 and len(pattern_types) == 0:
                pattern_types.append('random')

            # Check for sparse patterns
            if results['sparsity'] > 0.7:
                pattern_types.append('sparse')

            results['pattern_types'] = pattern_types
            results['dominant_pattern'] = pattern_types[0] if pattern_types else 'unknown'

            return results

        except Exception as e:
            logger.error(f"Tensor pattern analysis failed: {e}")
            return {
                'shape': tensor.shape,
                'rank': tensor.ndim,
                'size': tensor.size,
                'sparsity': 0.0,
                'symmetry_score': 0.0,
                'pattern_types': [],
                'dominant_pattern': 'unknown'
            }

    def compute_tensor_statistics(self, tensor: NDArray) -> Dict[str, float]:
        """
        Compute comprehensive tensor statistics.

        Args:
            tensor: Input tensor

        Returns:
            Dictionary of tensor statistics
        """
        try:
            stats = {
                'mean': float(np.mean(tensor)),
                'std': float(np.std(tensor)),
                'min': float(np.min(tensor)),
                'max': float(np.max(tensor)),
                'median': float(np.median(tensor)),
                'variance': float(np.var(tensor)),
                'skewness': 0.0,
                'kurtosis': 0.0,
                'condition_number': 0.0,
                'rank': int(np.linalg.matrix_rank(tensor)) if tensor.ndim == 2 else 0
            }

            # Calculate higher moments if possible
            if tensor.size > 2:
                from scipy.stats import skew, kurtosis
                stats['skewness'] = float(skew(tensor.flatten()))
                stats['kurtosis'] = float(kurtosis(tensor.flatten()))

            # Calculate condition number for 2D tensors
            if tensor.ndim == 2:
                try:
                    eigenvals = np.linalg.eigvals(tensor)
                    eigenvals = eigenvals[np.abs(eigenvals) > self.epsilon]
                    if len(eigenvals) > 0:
                        stats['condition_number'] = float(np.max(np.abs(eigenvals)) / np.min(np.abs(eigenvals)))
                except Exception:
                    stats['condition_number'] = 0.0

            return stats

        except Exception as e:
            logger.error(f"Tensor statistics computation failed: {e}")
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'variance': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'condition_number': 0.0,
                'rank': 0
            }

    def tensor_pattern_matching(self, tensor: NDArray, pattern: NDArray,
                              threshold: float = 0.8) -> Dict[str, Any]:
        """
        Perform pattern matching on tensor.

        Args:
            tensor: Input tensor
            pattern: Pattern to match
            threshold: Matching threshold

        Returns:
            Dictionary containing matching results
        """
        try:
            results = {
                'match_score': 0.0,
                'match_locations': [],
                'best_match': None,
                'pattern_found': False
            }

            if tensor.ndim != pattern.ndim:
                return results

            # For 2D tensors, use correlation-based matching
            if tensor.ndim == 2:
                match_score = self._correlation_match(tensor, pattern)
                results['match_score'] = match_score

                if match_score > threshold:
                    results['pattern_found'] = True
                    results['best_match'] = (0, 0)  # Simplified for now

            # For 1D tensors, use sliding window matching
            elif tensor.ndim == 1:
                match_score = self._sliding_window_match(tensor, pattern)
                results['match_score'] = match_score

                if match_score > threshold:
                    results['pattern_found'] = True

            return results

        except Exception as e:
            logger.error(f"Tensor pattern matching failed: {e}")
            return {
                'match_score': 0.0,
                'match_locations': [],
                'best_match': None,
                'pattern_found': False
            }

    def tensor_clustering(self, tensors: List[NDArray],
                         n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform clustering on tensor data.

        Args:
            tensors: List of tensors to cluster
            n_clusters: Number of clusters

        Returns:
            Dictionary containing clustering results
        """
        try:
            if not tensors or len(tensors) < n_clusters:
                return {
                    'cluster_labels': [],
                    'cluster_centers': [],
                    'clusters': [],
                    'inertia': 0.0
                }

            # Flatten tensors for clustering
            flattened_tensors = []
            for tensor in tensors:
                flattened = tensor.flatten()
                flattened_tensors.append(flattened)

            # Ensure all tensors have the same length
            max_length = max(len(t) for t in flattened_tensors)
            padded_tensors = []

            for tensor in flattened_tensors:
                if len(tensor) < max_length:
                    padded = np.zeros(max_length)
                    padded[:len(tensor)] = tensor
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(tensor)

            # Convert to numpy array
            data_matrix = np.array(padded_tensors)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data_matrix)

            # Group tensors by cluster
            clusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(tensors[i])

            # Calculate cluster centers
            cluster_centers = []
            for i in range(n_clusters):
                if clusters[i]:
                    # Calculate mean tensor for this cluster
                    cluster_tensors = np.array(clusters[i])
                    center = np.mean(cluster_tensors, axis=0)
                    cluster_centers.append(center)
                else:
                    cluster_centers.append(np.zeros_like(tensors[0]))

            return {
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': cluster_centers,
                'clusters': clusters,
                'inertia': float(kmeans.inertia_)
            }

        except Exception as e:
            logger.error(f"Tensor clustering failed: {e}")
            return {
                'cluster_labels': [],
                'cluster_centers': [],
                'clusters': [],
                'inertia': 0.0
            }

    def tensor_dimensionality_reduction(self, tensor: NDArray,
                                      target_dimensions: int = 2) -> NDArray:
        """
        Reduce tensor dimensionality using various techniques.

        Args:
            tensor: Input tensor
            target_dimensions: Target number of dimensions

        Returns:
            Reduced tensor
        """
        try:
            if tensor.ndim <= target_dimensions:
                return tensor.copy()

            # Flatten tensor for dimensionality reduction
            flattened = tensor.flatten()

            # Reshape to 2D for PCA
            if len(flattened) > 1:
                # Reshape to have at least 2 samples
                n_samples = max(2, len(flattened) // 10)
                n_features = len(flattened) // n_samples

                if n_features > 0:
                    reshaped = flattened[:n_samples * n_features].reshape(n_samples, n_features)

                    # Apply PCA
                    pca = PCA(n_components=min(target_dimensions, n_features))
                    reduced = pca.fit_transform(reshaped)

                    # Reshape back to target dimensions
                    if target_dimensions == 1:
                        return reduced.flatten()
                    else:
                        return reduced.reshape(target_dimensions)

            # Fallback: simple reshaping
            return tensor.reshape(target_dimensions)

        except Exception as e:
            logger.error(f"Tensor dimensionality reduction failed: {e}")
            return tensor.copy()

    def tensor_similarity(self, tensor1: NDArray, tensor2: NDArray,
                         method: str = 'cosine') -> float:
        """
        Calculate similarity between two tensors.

        Args:
            tensor1: First tensor
            tensor2: Second tensor
            method: Similarity method ('cosine', 'euclidean', 'correlation')

        Returns:
            Similarity score
        """
        try:
            # Flatten tensors
            flat1 = tensor1.flatten()
            flat2 = tensor2.flatten()

            # Ensure same length
            min_length = min(len(flat1), len(flat2))
            flat1 = flat1[:min_length]
            flat2 = flat2[:min_length]

            if method == 'cosine':
                # Cosine similarity
                dot_product = np.dot(flat1, flat2)
                norm1 = np.linalg.norm(flat1)
                norm2 = np.linalg.norm(flat2)

                if norm1 > 0 and norm2 > 0:
                    return float(dot_product / (norm1 * norm2))
                else:
                    return 0.0

            elif method == 'euclidean':
                # Euclidean distance (converted to similarity)
                distance = np.linalg.norm(flat1 - flat2)
                max_distance = np.linalg.norm(flat1) + np.linalg.norm(flat2)
                if max_distance > 0:
                    return float(1.0 - distance / max_distance)
                else:
                    return 0.0

            elif method == 'correlation':
                # Correlation coefficient
                correlation = np.corrcoef(flat1, flat2)[0, 1]
                return float(correlation) if not np.isnan(correlation) else 0.0

            else:
                return 0.0

        except Exception as e:
            logger.error(f"Tensor similarity calculation failed: {e}")
            return 0.0

    def _calculate_block_score(self, tensor: NDArray) -> float:
        """Calculate block pattern score for 2D tensor."""
        try:
            if tensor.ndim != 2:
                return 0.0

            # Simple block detection using variance
            block_size = min(tensor.shape) // 4
            if block_size < 2:
                return 0.0

            block_variances = []

            for i in range(0, tensor.shape[0] - block_size + 1, block_size):
                for j in range(0, tensor.shape[1] - block_size + 1, block_size):
                    block = tensor[i:i+block_size, j:j+block_size]
                    block_variances.append(np.var(block))

            if block_variances:
                # Lower variance indicates more block-like structure
                avg_variance = np.mean(block_variances)
                return float(1.0 / (1.0 + avg_variance))
            else:
                return 0.0

        except Exception:
            return 0.0

    def _correlation_match(self, tensor: NDArray, pattern: NDArray) -> float:
        """Calculate correlation-based match score."""
        try:
            # Normalize both tensors
            tensor_norm = (tensor - np.mean(tensor)) / (np.std(tensor) + self.epsilon)
            pattern_norm = (pattern - np.mean(pattern)) / (np.std(pattern) + self.epsilon)

            # Calculate correlation
            correlation = np.corrcoef(tensor_norm.flatten(), pattern_norm.flatten())[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0

        except Exception:
            return 0.0

    def _sliding_window_match(self, tensor: NDArray, pattern: NDArray) -> float:
        """Calculate sliding window match score."""
        try:
            if len(tensor) < len(pattern):
                return 0.0

            max_correlation = 0.0

            for i in range(len(tensor) - len(pattern) + 1):
                window = tensor[i:i+len(pattern)]
                correlation = np.corrcoef(window, pattern)[0, 1]
                if not np.isnan(correlation):
                    max_correlation = max(max_correlation, correlation)

            return float(max_correlation)

        except Exception:
            return 0.0


# Global instance for convenience
tensor_engine = TensorEngine()

# Convenience functions
def create_tensor_space(data: NDArray, dimensions: Tuple[int, ...]) -> NDArray:
    """Convenience function for tensor space creation."""
    return tensor_engine.create_tensor_space(data, dimensions)


def analyze_tensor_patterns(tensor: NDArray) -> Dict[str, Any]:
    """Convenience function for tensor pattern analysis."""
    return tensor_engine.analyze_tensor_patterns(tensor)


def compute_tensor_statistics(tensor: NDArray) -> Dict[str, float]:
    """Convenience function for tensor statistics computation."""
    return tensor_engine.compute_tensor_statistics(tensor)


def tensor_pattern_matching(tensor: NDArray, pattern: NDArray,
                          threshold: float = 0.8) -> Dict[str, Any]:
    """Convenience function for tensor pattern matching."""
    return tensor_engine.tensor_pattern_matching(tensor, pattern, threshold)


def tensor_clustering(tensors: List[NDArray], n_clusters: int = 3) -> Dict[str, Any]:
    """Convenience function for tensor clustering."""
    return tensor_engine.tensor_clustering(tensors, n_clusters)


def tensor_dimensionality_reduction(tensor: NDArray,
                                  target_dimensions: int = 2) -> NDArray:
    """Convenience function for tensor dimensionality reduction."""
    return tensor_engine.tensor_dimensionality_reduction(tensor, target_dimensions)


if __name__ == "__main__":
    # Test the tensor engine
    import numpy as np

    # Import safe print for Windows compatibility
    try:
        from ...utils.windows_cli_compatibility import safe_print
    except ImportError:
        try:
            from core.utils.windows_cli_compatibility import safe_print
        except ImportError:
            def safe_print(message):
                print(message)

    def main():
        """Main function to test tensor engine and ensure proper initialization."""
        try:
            safe_print("🔢 Testing Tensor Engine")
            safe_print("=" * 40)

            # Create test tensors
            tensor1 = np.random.rand(5, 5)
            tensor2 = np.random.rand(5, 5)
            tensor3 = np.random.rand(10, 10)

            safe_print(f"Tensor 1 Shape: {tensor1.shape}")
            safe_print(f"Tensor 2 Shape: {tensor2.shape}")
            safe_print(f"Tensor 3 Shape: {tensor3.shape}")

            # Test tensor space creation
            safe_print("\n🏗️ Testing Tensor Space Creation:")
            new_tensor = create_tensor_space(tensor1.flatten(), (3, 3))
            safe_print(f"✅ New Tensor Shape: {new_tensor.shape}")

            # Test pattern analysis
            safe_print("\n📊 Testing Pattern Analysis:")
            pattern_results = analyze_tensor_patterns(tensor1)
            safe_print(f"✅ Pattern Types: {pattern_results['pattern_types']}")
            safe_print(f"✅ Dominant Pattern: {pattern_results['dominant_pattern']}")
            safe_print(f"✅ Sparsity: {pattern_results['sparsity']:.4f}")
            safe_print(f"✅ Symmetry Score: {pattern_results['symmetry_score']:.4f}")

            # Test statistics computation
            safe_print("\n📈 Testing Statistics Computation:")
            stats = compute_tensor_statistics(tensor1)
            safe_print(f"✅ Tensor Mean: {stats['mean']:.4f}")
            safe_print(f"✅ Tensor Rank: {stats['rank']}")
            safe_print(f"✅ Tensor Variance: {stats['variance']:.4f}")
            safe_print(f"✅ Tensor Skewness: {stats['skewness']:.4f}")

            # Test pattern matching
            safe_print("\n🎯 Testing Pattern Matching:")
            match_results = tensor_pattern_matching(tensor1, tensor2, threshold=0.5)
            safe_print(f"✅ Match Score: {match_results['match_score']:.4f}")
            safe_print(f"✅ Pattern Found: {match_results['pattern_found']}")

            # Test clustering
            safe_print("\n🎯 Testing Clustering:")
            tensors = [tensor1, tensor2, tensor3]
            clustering_results = tensor_clustering(tensors, n_clusters=2)
            safe_print(f"✅ Cluster Labels: {clustering_results['cluster_labels']}")
            safe_print(f"✅ Inertia: {clustering_results['inertia']:.4f}")

            # Test dimensionality reduction
            safe_print("\n📉 Testing Dimensionality Reduction:")
            reduced = tensor_dimensionality_reduction(tensor3, target_dimensions=2)
            safe_print(f"✅ Reduced Shape: {reduced.shape}")

            # Test advanced tensor engine features
            safe_print("\n🔬 Testing Advanced Features:")

            # Test tensor similarity
            similarity_cosine = tensor_engine.tensor_similarity(tensor1, tensor2, method='cosine')
            similarity_euclidean = tensor_engine.tensor_similarity(tensor1, tensor2, method='euclidean')
            similarity_correlation = tensor_engine.tensor_similarity(tensor1, tensor2, method='correlation')
            safe_print(f"✅ Cosine Similarity: {similarity_cosine:.4f}")
            safe_print(f"✅ Euclidean Similarity: {similarity_euclidean:.4f}")
            safe_print(f"✅ Correlation Similarity: {similarity_correlation:.4f}")

            # Test block score calculation
            block_score = tensor_engine._calculate_block_score(tensor1)
            safe_print(f"✅ Block Score: {block_score:.4f}")

            # Test correlation matching
            correlation_match = tensor_engine._correlation_match(tensor1, tensor2)
            safe_print(f"✅ Correlation Match: {correlation_match:.4f}")

            # Test sliding window matching
            sliding_match = tensor_engine._sliding_window_match(tensor1.flatten(), tensor2.flatten())
            safe_print(f"✅ Sliding Window Match: {sliding_match:.4f}")

            # Test multiple tensor operations
            safe_print("\n🔄 Testing Multiple Tensor Operations:")
            all_tensors = [tensor1, tensor2, tensor3]
            for i, tensor in enumerate(all_tensors):
                pattern_analysis = analyze_tensor_patterns(tensor)
                stats = compute_tensor_statistics(tensor)
                safe_print(f"✅ Tensor {i+1}: Patterns={len(pattern_analysis['pattern_types'])}, Mean={stats['mean']:.4f}, Rank={stats['rank']}")

            safe_print("\n🎉 Tensor Engine tests completed successfully!")
            return True

        except Exception as e:
            safe_print(f"❌ Tensor Engine test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Run main function
    success = main()
    import sys
    sys.exit(0 if success else 1)
