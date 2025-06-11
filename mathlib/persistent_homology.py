import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

@dataclass
class PersistenceDiagram:
    """Represents a persistence diagram with birth and death times"""
    birth_times: np.ndarray
    death_times: np.ndarray
    dimensions: np.ndarray  # Homology dimension for each feature
    
    def get_persistence(self) -> np.ndarray:
        """Get persistence (death - birth) for each feature"""
        return self.death_times - self.birth_times
    
    def get_stable_features(self, threshold: float) -> List[int]:
        """Get indices of features with persistence above threshold"""
        persistence = self.get_persistence()
        return np.where(persistence > threshold)[0].tolist()

class PersistentTopologyAnalyzer:
    def __init__(
        self,
        max_dim: int = 2,
        max_filtration_value: float = 1.0,
        num_filtration_steps: int = 100
    ):
        """
        Initialize the persistent homology analyzer
        
        Args:
            max_dim: Maximum homology dimension to compute
            max_filtration_value: Maximum value for the Vietoris-Rips filtration
            num_filtration_steps: Number of steps in the filtration
        """
        self.max_dim = max_dim
        self.max_filtration_value = max_filtration_value
        self.num_filtration_steps = num_filtration_steps
        self.filtration_values = np.linspace(0, max_filtration_value, num_filtration_steps)
        
    def compute_persistence(
        self,
        points: np.ndarray,
        metric: str = 'euclidean'
    ) -> PersistenceDiagram:
        """
        Compute persistent homology features from point cloud data
        
        Args:
            points: Point cloud data (n_samples, n_features)
            metric: Distance metric to use
            
        Returns:
            Persistence diagram with birth and death times
        """
        # Compute pairwise distances
        distances = squareform(pdist(points, metric=metric))
        
        # Initialize arrays to store birth and death times
        birth_times = []
        death_times = []
        dimensions = []
        
        # Compute persistence for each dimension up to max_dim
        for dim in range(self.max_dim + 1):
            dim_birth, dim_death = self._compute_dimension_persistence(
                distances, dim
            )
            birth_times.extend(dim_birth)
            death_times.extend(dim_death)
            dimensions.extend([dim] * len(dim_birth))
            
        return PersistenceDiagram(
            birth_times=np.array(birth_times),
            death_times=np.array(death_times),
            dimensions=np.array(dimensions)
        )
    
    def _compute_dimension_persistence(
        self,
        distances: np.ndarray,
        dimension: int
    ) -> Tuple[List[float], List[float]]:
        """
        Compute persistence for a specific homology dimension
        
        Args:
            distances: Pairwise distance matrix
            dimension: Homology dimension to compute
            
        Returns:
            Tuple of (birth_times, death_times)
        """
        birth_times = []
        death_times = []
        
        # For 0-dimensional homology (connected components)
        if dimension == 0:
            for eps in self.filtration_values:
                # Create adjacency matrix for current epsilon
                adj_matrix = (distances <= eps).astype(int)
                adj_matrix[np.diag_indices_from(adj_matrix)] = 0
                
                # Find connected components
                n_components, labels = connected_components(
                    csr_matrix(adj_matrix)
                )
                
                # Record birth of new components
                if n_components > len(birth_times):
                    birth_times.extend([eps] * (n_components - len(birth_times)))
                    
                # Record death of components that merge
                if n_components < len(death_times):
                    death_times.extend([eps] * (len(death_times) - n_components))
                    
        # For higher dimensions, implement more sophisticated algorithms
        # This is a simplified version - in practice, use a library like Ripser
        else:
            # Placeholder for higher-dimensional persistence
            pass
            
        return birth_times, death_times
    
    def find_stable_patterns(
        self,
        points: np.ndarray,
        persistence_threshold: float = 0.1
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Find stable topological patterns in the data
        
        Args:
            points: Point cloud data
            persistence_threshold: Minimum persistence for a feature to be considered stable
            
        Returns:
            Dictionary mapping dimension to list of stable feature indices
        """
        # Compute persistence diagram
        diagram = self.compute_persistence(points)
        
        # Find stable features for each dimension
        stable_features = {}
        for dim in range(self.max_dim + 1):
            dim_mask = diagram.dimensions == dim
            dim_persistence = diagram.get_persistence()[dim_mask]
            stable_indices = np.where(dim_persistence > persistence_threshold)[0]
            
            if len(stable_indices) > 0:
                stable_features[dim] = [
                    (int(diagram.birth_times[dim_mask][i]),
                     int(diagram.death_times[dim_mask][i]))
                    for i in stable_indices
                ]
                
        return stable_features
    
    def compute_topological_signature(
        self,
        points: np.ndarray,
        num_bins: int = 10
    ) -> Dict[int, np.ndarray]:
        """
        Compute topological signature as persistence histogram
        
        Args:
            points: Point cloud data
            num_bins: Number of bins for persistence histogram
            
        Returns:
            Dictionary mapping dimension to persistence histogram
        """
        # Compute persistence diagram
        diagram = self.compute_persistence(points)
        
        # Compute histograms for each dimension
        signatures = {}
        for dim in range(self.max_dim + 1):
            dim_mask = diagram.dimensions == dim
            persistence = diagram.get_persistence()[dim_mask]
            
            if len(persistence) > 0:
                hist, _ = np.histogram(
                    persistence,
                    bins=num_bins,
                    range=(0, self.max_filtration_value)
                )
                signatures[dim] = hist
            else:
                signatures[dim] = np.zeros(num_bins)
                
        return signatures 