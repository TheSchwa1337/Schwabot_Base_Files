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
"""Cluster Mapper - Mathematical Clustering for Market Data Analysis.

This module provides advanced clustering algorithms for:
- Market data pattern clustering
- Trading signal grouping
- Price movement classification
- Volatility clustering
- Risk pattern identification

Mathematical Foundation:
- K-means clustering with dynamic centroids
- DBSCAN for density-based clustering
- Hierarchical clustering for nested patterns
- Spectral clustering for complex relationships
- Custom mathematical distance metrics
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from core.unified_math_system import unified_math
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

logger = logging.getLogger(__name__)


@dataclass
class ClusterPoint:
    """Represents a data point in clustering space."""
    point_id: str
    coordinates: np.ndarray
    features: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Cluster:
    """Represents a cluster of data points."""
    cluster_id: int
    centroid: np.ndarray
    points: List[ClusterPoint]
    radius: float
    density: float
    confidence: float
    cluster_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusteringResult:
    """Result of clustering analysis."""
    clusters: List[Cluster]
    unassigned_points: List[ClusterPoint]
    algorithm: str
    parameters: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class ClusterMapper:
    """
    Advanced clustering system for market data analysis.
    
    Provides multiple clustering algorithms optimized for:
    - Market pattern recognition
    - Trading signal classification
    - Risk assessment grouping
    - Price movement categorization
    """
    
    def __init__(self):
        """Initialize cluster mapper."""
        self.supported_algorithms = {
            'kmeans': self._kmeans_clustering,
            'dbscan': self._dbscan_clustering,
            'hierarchical': self._hierarchical_clustering,
            'spectral': self._spectral_clustering,
            'custom': self._custom_clustering
        }
        
        self.clustering_history: List[ClusteringResult] = []
        self.max_history = 100
        
        logger.info("ClusterMapper initialized")
    
    def create_cluster_point(
        self,
        point_id: str,
        coordinates: Union[List[float], np.ndarray],
        features: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ClusterPoint:
        """
        Create a cluster point from data.
        
        Parameters:
        -----------
        point_id : str
            Unique identifier for the point
        coordinates : Union[List[float], np.ndarray]
            Numerical coordinates of the point
        features : Optional[Dict[str, float]]
            Additional features for the point
        metadata : Optional[Dict[str, Any]]
            Additional metadata
            
        Returns:
        --------
        ClusterPoint
            Created cluster point
        """
        try:
            coords_array = np.array(coordinates, dtype=float)
            features = features or {}
            metadata = metadata or {}
            
            return ClusterPoint(
                point_id=point_id,
                coordinates=coords_array,
                features=features,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating cluster point: {e}")
            raise
    
    def cluster_market_data(
        self,
        data_points: List[ClusterPoint],
        algorithm: str = 'kmeans',
        n_clusters: int = 5,
        **kwargs
    ) -> ClusteringResult:
        """
        Perform clustering on market data.
        
        Parameters:
        -----------
        data_points : List[ClusterPoint]
            List of data points to cluster
        algorithm : str
            Clustering algorithm to use
        n_clusters : int
            Number of clusters (for applicable algorithms)
        **kwargs
            Additional algorithm-specific parameters
            
        Returns:
        --------
        ClusteringResult
            Clustering analysis result
        """
        start_time = time.time()
        
        try:
            if not data_points:
                raise ValueError("No data points provided for clustering")
            
            if algorithm not in self.supported_algorithms:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Extract coordinates for clustering
            coordinates = np.array([point.coordinates for point in data_points])
            
            # Perform clustering
            cluster_labels = self.supported_algorithms[algorithm](
                coordinates, n_clusters, **kwargs
            )
            
            # Create clusters from labels
            clusters = self._create_clusters_from_labels(
                data_points, cluster_labels, coordinates
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(coordinates, cluster_labels)
            
            processing_time = time.time() - start_time
            
            result = ClusteringResult(
                clusters=clusters,
                unassigned_points=[],  # Will be populated if needed
                algorithm=algorithm,
                parameters={'n_clusters': n_clusters, **kwargs},
                quality_metrics=quality_metrics,
                processing_time=processing_time
            )
            
            # Store in history
            self.clustering_history.append(result)
            if len(self.clustering_history) > self.max_history:
                self.clustering_history.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in market data clustering: {e}")
            return ClusteringResult(
                clusters=[],
                unassigned_points=data_points,
                algorithm=algorithm,
                parameters={'n_clusters': n_clusters, **kwargs},
                quality_metrics={},
                processing_time=time.time() - start_time
            )
    
    def _kmeans_clustering(
        self,
        coordinates: np.ndarray,
        n_clusters: int,
        **kwargs
    ) -> np.ndarray:
        """Perform K-means clustering."""
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                **kwargs
            )
            return kmeans.fit_predict(coordinates)
        except Exception as e:
            logger.error(f"Error in K-means clustering: {e}")
            return np.zeros(len(coordinates), dtype=int)
    
    def _dbscan_clustering(
        self,
        coordinates: np.ndarray,
        n_clusters: int,
        eps: float = 0.5,
        min_samples: int = 5,
        **kwargs
    ) -> np.ndarray:
        """Perform DBSCAN clustering."""
        try:
            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                **kwargs
            )
            return dbscan.fit_predict(coordinates)
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {e}")
            return np.zeros(len(coordinates), dtype=int)
    
    def _hierarchical_clustering(
        self,
        coordinates: np.ndarray,
        n_clusters: int,
        **kwargs
    ) -> np.ndarray:
        """Perform hierarchical clustering."""
        try:
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                **kwargs
            )
            return hierarchical.fit_predict(coordinates)
        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {e}")
            return np.zeros(len(coordinates), dtype=int)
    
    def _spectral_clustering(
        self,
        coordinates: np.ndarray,
        n_clusters: int,
        **kwargs
    ) -> np.ndarray:
        """Perform spectral clustering."""
        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                **kwargs
            )
            return spectral.fit_predict(coordinates)
        except Exception as e:
            logger.error(f"Error in spectral clustering: {e}")
            return np.zeros(len(coordinates), dtype=int)
    
    def _custom_clustering(
        self,
        coordinates: np.ndarray,
        n_clusters: int,
        **kwargs
    ) -> np.ndarray:
        """Perform custom clustering algorithm."""
        try:
            # Custom clustering logic for market data
            # This could implement domain-specific clustering
            
            # For now, use K-means as base
            return self._kmeans_clustering(coordinates, n_clusters, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in custom clustering: {e}")
            return np.zeros(len(coordinates), dtype=int)
    
    def _create_clusters_from_labels(
        self,
        data_points: List[ClusterPoint],
        labels: np.ndarray,
        coordinates: np.ndarray
    ) -> List[Cluster]:
        """Create Cluster objects from clustering labels."""
        clusters = []
        
        try:
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points (DBSCAN)
                    continue
                
                # Get points in this cluster
                cluster_mask = labels == label
                cluster_points = [data_points[i] for i in range(len(data_points)) if cluster_mask[i]]
                cluster_coords = coordinates[cluster_mask]
                
                if len(cluster_points) == 0:
                    continue
                
                # Calculate centroid
                centroid = unified_math.unified_math.mean(cluster_coords, axis=0)
                
                # Calculate radius (max distance from centroid)
                distances = np.linalg.norm(cluster_coords - centroid, axis=1)
                radius = unified_math.unified_math.max(distances) if len(distances) > 0 else 0.0
                
                # Calculate density
                density = len(cluster_points) / (np.pi * radius**2) if radius > 0 else 0.0
                
                # Calculate confidence (based on cluster size and compactness)
                confidence = unified_math.min(1.0, len(cluster_points) / 10.0) * (1.0 - radius / 10.0)
                confidence = unified_math.max(0.0, confidence)
                
                cluster = Cluster(
                    cluster_id=int(label),
                    centroid=centroid,
                    points=cluster_points,
                    radius=radius,
                    density=density,
                    confidence=confidence,
                    cluster_type="market_pattern"
                )
                
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error creating clusters from labels: {e}")
            return []
    
    def _calculate_quality_metrics(
        self,
        coordinates: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        try:
            metrics = {}
            
            # Silhouette score
            if len(np.unique(labels)) > 1:
                try:
                    metrics['silhouette_score'] = silhouette_score(coordinates, labels)
                except:
                    metrics['silhouette_score'] = 0.0
            else:
                metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz score
            if len(np.unique(labels)) > 1:
                try:
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(coordinates, labels)
                except:
                    metrics['calinski_harabasz_score'] = 0.0
            else:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Number of clusters
            metrics['n_clusters'] = len(np.unique(labels[labels != -1]))
            
            # Noise ratio (for DBSCAN)
            noise_ratio = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 0.0
            metrics['noise_ratio'] = noise_ratio
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {}
    
    def analyze_trading_patterns(
        self,
        price_data: List[Dict[str, float]],
        volume_data: List[float],
        volatility_data: List[float]
    ) -> ClusteringResult:
        """
        Analyze trading patterns using clustering.
        
        Parameters:
        -----------
        price_data : List[Dict[str, float]]
            List of price data dictionaries
        volume_data : List[float]
            Volume data
        volatility_data : List[float]
            Volatility data
            
        Returns:
        --------
        ClusteringResult
            Pattern analysis result
        """
        try:
            # Create feature vectors for clustering
            data_points = []
            
            for i in range(len(price_data)):
                if i < len(volume_data) and i < len(volatility_data):
                    # Create feature vector: [price_change, volume, volatility]
                    price_change = price_data[i].get('change', 0.0)
                    volume = volume_data[i]
                    volatility = volatility_data[i]
                    
                    coordinates = [price_change, volume, volatility]
                    
                    point = self.create_cluster_point(
                        point_id=f"pattern_{i}",
                        coordinates=coordinates,
                        features={
                            'price_change': price_change,
                            'volume': volume,
                            'volatility': volatility
                        }
                    )
                    
                    data_points.append(point)
            
            # Perform clustering
            return self.cluster_market_data(data_points, 'kmeans', n_clusters=3)
            
        except Exception as e:
            logger.error(f"Error analyzing trading patterns: {e}")
            return ClusteringResult(
                clusters=[],
                unassigned_points=[],
                algorithm='kmeans',
                parameters={},
                quality_metrics={},
                processing_time=0.0
            )
    
    def get_clustering_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        if not self.clustering_history:
            return {"error": "No clustering history available"}
        
        total_analyses = len(self.clustering_history)
        total_clusters = sum(len(result.clusters) for result in self.clustering_history)
        total_points = sum(
            sum(len(cluster.points) for cluster in result.clusters)
            for result in self.clustering_history
        )
        
        # Algorithm usage statistics
        algorithm_counts = {}
        for result in self.clustering_history:
            algorithm_counts[result.algorithm] = algorithm_counts.get(result.algorithm, 0) + 1
        
        # Average quality metrics
        avg_silhouette = unified_math.mean([
            result.quality_metrics.get('silhouette_score', 0.0)
            for result in self.clustering_history
        ])
        
        avg_processing_time = unified_math.mean([
            result.processing_time for result in self.clustering_history
        ])
        
        return {
            "total_analyses": total_analyses,
            "total_clusters": total_clusters,
            "total_points": total_points,
            "algorithm_usage": algorithm_counts,
            "average_silhouette_score": avg_silhouette,
            "average_processing_time": avg_processing_time,
            "supported_algorithms": list(self.supported_algorithms.keys())
        }


def main() -> None:
    """Test function for ClusterMapper."""
    safe_print("🗺️ Testing Cluster Mapper...")
    
    mapper = ClusterMapper()
    
    # Create sample market data points
    data_points = []
    for i in range(100):
        # Simulate market data: [price_change, volume, volatility]
        price_change = np.random.normal(0, 1)
        volume = np.random.uniform(1000, 10000)
        volatility = np.random.uniform(0.01, 0.1)
        
        point = mapper.create_cluster_point(
            point_id=f"market_point_{i}",
            coordinates=[price_change, volume, volatility],
            features={
                'price_change': price_change,
                'volume': volume,
                'volatility': volatility
            }
        )
        data_points.append(point)
    
    # Test clustering
    result = mapper.cluster_market_data(data_points, 'kmeans', n_clusters=3)
    safe_print(f"✅ Clustering completed:")
    safe_print(f"   Algorithm: {result.algorithm}")
    safe_print(f"   Clusters found: {len(result.clusters)}")
    safe_print(f"   Processing time: {result.processing_time:.4f}s")
    safe_print(f"   Silhouette score: {result.quality_metrics.get('silhouette_score', 0.0):.4f}")
    
    # Test trading pattern analysis
    price_data = [{'change': np.random.normal(0, 1)} for _ in range(50)]
    volume_data = [np.random.uniform(1000, 10000) for _ in range(50)]
    volatility_data = [np.random.uniform(0.01, 0.1) for _ in range(50)]
    
    pattern_result = mapper.analyze_trading_patterns(price_data, volume_data, volatility_data)
    safe_print(f"✅ Pattern analysis completed:")
    safe_print(f"   Patterns found: {len(pattern_result.clusters)}")
    
    # Get statistics
    stats = mapper.get_clustering_statistics()
    safe_print(f"📊 Clustering statistics: {stats}")
    
    return 0

if __name__ == "__main__":
    exit(main())
