# !/usr/bin/env python3
"""
Matrix Mapper - Hash-to-matrix similarity routing logic with CRWF-CRLF integration

Enhanced with:
- Entropy alignment score for each strategy entry
- CRWF-CRLF fusion capabilities
- Geo-located resonance mapping
- Weather-entropy correlation
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from numpy.linalg import norm
from dataclasses import dataclass, field
from datetime import datetime

from .chrono_resonance_weather_mapper import (
    ChronoResonanceWeatherMapper, GeoLocation, CRWFResponse, create_crwf_mapper
)
from .chrono_recursive_logic_function import (
    ChronoRecursiveLogicFunction, CRLFResponse, create_crlf
)
from .crwf_crlf_integration import CRWFCRLFIntegration, create_crwf_crlf_integration

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMatrixEntry:
    """Enhanced matrix entry with CRWF-CRLF integration."""
    
    # Original matrix data
    hash_vector: np.ndarray
    strategy_weights: Dict[str, float]
    confidence_score: float
    
    # CRWF-CRLF enhanced data
    entropy_alignment_score: float
    weather_resonance_factor: float
    geo_alignment_score: float
    crlf_adjustment_factor: float
    
    # Location data
    location: Optional[GeoLocation] = None
    weather_data: Optional[Dict[str, Any]] = None
    
    # Performance tracking
    last_updated: datetime = field(default_factory=datetime.now)
    performance_history: List[float] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedMatrixMapper:
    """
    Enhanced matrix mapper with CRWF-CRLF integration.
    
    Provides:
    - Entropy alignment scoring
    - Weather-resonance correlation
    - Geo-located strategy optimization
    - Real-time matrix enhancement
    """
    
    def __init__(self, matrix_dir: str, weather_api_key: Optional[str] = None):
        """Initialize enhanced matrix mapper."""
        self.matrix_dir = matrix_dir
        self.matrices: Dict[str, EnhancedMatrixEntry] = {}
        self.crwf_crlf_integration = create_crwf_crlf_integration(weather_api_key)
        
        # Load existing matrices
        self._load_existing_matrices()
        
        logger.info(f"🔧 Enhanced Matrix Mapper initialized with {len(self.matrices)} matrices")
    
    def _load_existing_matrices(self):
        """Load existing matrices and enhance them with CRWF-CRLF data."""
        try:
            for fname in os.listdir(self.matrix_dir):
                if fname.endswith(".json"):
                    filepath = os.path.join(self.matrix_dir, fname)
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    
                    # Create enhanced matrix entry
                    entry = EnhancedMatrixEntry(
                        hash_vector=np.array(data.get('hash_vector', [])),
                        strategy_weights=data.get('strategy_weights', {}),
                        confidence_score=data.get('confidence_score', 0.5),
                        entropy_alignment_score=data.get('entropy_alignment_score', 0.5),
                        weather_resonance_factor=data.get('weather_resonance_factor', 1.0),
                        geo_alignment_score=data.get('geo_alignment_score', 0.5),
                        crlf_adjustment_factor=data.get('crlf_adjustment_factor', 1.0),
                        metadata=data.get('metadata', {})
                    )
                    
                    self.matrices[fname] = entry
            
            logger.info(f"📊 Loaded {len(self.matrices)} existing matrices")
            
        except Exception as e:
            logger.error(f"Error loading existing matrices: {e}")
    
    def enhance_matrix_with_crwf_crlf(
        self,
        matrix_name: str,
        location: GeoLocation,
        weather_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Enhance a matrix entry with CRWF-CRLF analysis.
        
        Args:
            matrix_name: Name of the matrix to enhance
            location: Geographic location for analysis
            weather_data: Optional weather data
            
        Returns:
            True if enhancement successful, False otherwise
        """
        try:
            if matrix_name not in self.matrices:
                logger.warning(f"Matrix {matrix_name} not found")
                return False
            
            entry = self.matrices[matrix_name]
            
            # Get or fetch weather data
            if weather_data is None:
                weather_data = self._get_weather_data_for_location(location)
            
            # Create weather data point
            weather_point = self._create_weather_data_point(location, weather_data)
            
            # Compute CRWF
            crwf_response = self.crwf_crlf_integration.crwf_mapper.compute_crwf(
                weather_point, location
            )
            
            # Create strategy vector from matrix entry
            strategy_vector = self._extract_strategy_vector(entry)
            profit_curve = self._generate_profit_curve(entry)
            
            # Compute CRLF
            crlf_response = self.crwf_crlf_integration.crlf_function.compute_crlf(
                strategy_vector, profit_curve, crwf_response.entropy_score
            )
            
            # Update matrix entry with enhanced data
            entry.entropy_alignment_score = crwf_response.entropy_score
            entry.weather_resonance_factor = crwf_response.resonance_strength
            entry.geo_alignment_score = crwf_response.geo_alignment_score
            entry.crlf_adjustment_factor = crwf_response.crlf_adjustment_factor
            entry.location = location
            entry.weather_data = weather_data
            entry.last_updated = datetime.now()
            
            # Store performance
            entry.performance_history.append(crwf_response.crwf_output)
            if len(entry.performance_history) > 100:
                entry.performance_history = entry.performance_history[-100:]
            
            # Save enhanced matrix
            self._save_enhanced_matrix(matrix_name, entry)
            
            logger.info(f"✅ Enhanced matrix {matrix_name} with CRWF-CRLF data")
            return True
            
        except Exception as e:
            logger.error(f"Error enhancing matrix {matrix_name}: {e}")
            return False
    
    def match_hash_to_enhanced_matrix(
        self,
        hash_vec: np.ndarray,
        location: GeoLocation,
        threshold: float = 0.8
    ) -> Optional[Tuple[str, EnhancedMatrixEntry, float]]:
        """
        Match hash vector to enhanced matrix with location-aware scoring.
        
        Args:
            hash_vec: Hash vector to match
            location: Geographic location for enhanced scoring
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (matrix_name, enhanced_entry, enhanced_score) or None
        """
        try:
            best_score = -1
            best_file = None
            best_entry = None
            
            for fname, entry in self.matrices.items():
                # Base cosine similarity
                base_score = cosine_similarity(hash_vec, entry.hash_vector)
                
                # Enhanced scoring with location and weather factors
                enhanced_score = self._compute_enhanced_similarity(
                    base_score, entry, location
                )
                
                if enhanced_score > best_score and enhanced_score >= threshold:
                    best_score = enhanced_score
                    best_file = fname
                    best_entry = entry
            
            if best_file:
                return (best_file, best_entry, best_score)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error matching hash to enhanced matrix: {e}")
            return None
    
    def get_entropy_aligned_matrices(
        self,
        location: GeoLocation,
        min_entropy_score: float = 0.7
    ) -> List[Tuple[str, EnhancedMatrixEntry]]:
        """
        Get matrices with high entropy alignment for a location.
        
        Args:
            location: Geographic location
            min_entropy_score: Minimum entropy alignment score
            
        Returns:
            List of (matrix_name, entry) tuples with high entropy alignment
        """
        aligned_matrices = []
        
        for fname, entry in self.matrices.items():
            if entry.entropy_alignment_score >= min_entropy_score:
                # Check if location is compatible
                if self._is_location_compatible(entry, location):
                    aligned_matrices.append((fname, entry))
        
        # Sort by entropy alignment score
        aligned_matrices.sort(key=lambda x: x[1].entropy_alignment_score, reverse=True)
        
        return aligned_matrices
    
    def create_geo_optimized_matrix(
        self,
        base_hash_vector: np.ndarray,
        location: GeoLocation,
        strategy_weights: Dict[str, float],
        weather_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a geo-optimized matrix for a specific location.
        
        Args:
            base_hash_vector: Base hash vector
            location: Target location
            strategy_weights: Strategy weights
            weather_data: Optional weather data
            
        Returns:
            Matrix filename if successful, None otherwise
        """
        try:
            # Create enhanced matrix entry
            entry = EnhancedMatrixEntry(
                hash_vector=base_hash_vector,
                strategy_weights=strategy_weights,
                confidence_score=0.5,  # Will be enhanced
                entropy_alignment_score=0.5,  # Will be enhanced
                weather_resonance_factor=1.0,  # Will be enhanced
                geo_alignment_score=0.5,  # Will be enhanced
                crlf_adjustment_factor=1.0,  # Will be enhanced
                location=location,
                weather_data=weather_data
            )
            
            # Enhance with CRWF-CRLF
            matrix_name = f"geo_optimized_{location.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            success = self.enhance_matrix_with_crwf_crlf(
                matrix_name, location, weather_data
            )
            
            if success:
                self.matrices[matrix_name] = entry
                logger.info(f"✅ Created geo-optimized matrix: {matrix_name}")
                return matrix_name
            else:
                logger.error(f"❌ Failed to create geo-optimized matrix")
                return None
                
        except Exception as e:
            logger.error(f"Error creating geo-optimized matrix: {e}")
            return None
    
    def _compute_enhanced_similarity(
        self,
        base_score: float,
        entry: EnhancedMatrixEntry,
        location: GeoLocation
    ) -> float:
        """Compute enhanced similarity score with location and weather factors."""
        # Base similarity weight
        base_weight = 0.6
        
        # Location alignment weight
        location_weight = 0.2
        location_score = 1.0 - abs(entry.geo_alignment_score - location.resonance_factor)
        
        # Weather resonance weight
        weather_weight = 0.1
        weather_score = entry.weather_resonance_factor
        
        # Entropy alignment weight
        entropy_weight = 0.1
        entropy_score = entry.entropy_alignment_score
        
        # Combined enhanced score
        enhanced_score = (
            base_score * base_weight +
            location_score * location_weight +
            weather_score * weather_weight +
            entropy_score * entropy_weight
        )
        
        return float(np.clip(enhanced_score, 0.0, 1.0))
    
    def _is_location_compatible(
        self,
        entry: EnhancedMatrixEntry,
        location: GeoLocation
    ) -> bool:
        """Check if matrix entry is compatible with location."""
        if entry.location is None:
            return True  # No location constraint
        
        # Check geographic proximity (simplified)
        lat_diff = abs(entry.location.latitude - location.latitude)
        lon_diff = abs(entry.location.longitude - location.longitude)
        
        # Within 10 degrees (roughly 1100 km)
        return lat_diff < 10.0 and lon_diff < 10.0
    
    def _get_weather_data_for_location(self, location: GeoLocation) -> Dict[str, Any]:
        """Get weather data for location."""
        # This would integrate with actual weather API
        return {
            'temperature': 20.0,
            'pressure': 1013.25,
            'humidity': 60.0,
            'wind_speed': 5.0,
            'weather_type': 'clear'
        }
    
    def _create_weather_data_point(
        self,
        location: GeoLocation,
        weather_data: Dict[str, Any]
    ):
        """Create weather data point from location and weather data."""
        from .chrono_resonance_weather_mapper import WeatherDataPoint
        
        return WeatherDataPoint(
            timestamp=datetime.now(),
            latitude=location.latitude,
            longitude=location.longitude,
            altitude=location.altitude,
            temperature=weather_data.get('temperature', 20.0),
            pressure=weather_data.get('pressure', 1013.25),
            humidity=weather_data.get('humidity', 60.0),
            wind_speed=weather_data.get('wind_speed', 5.0),
            wind_direction=0.0,
            weather_type=weather_data.get('weather_type', 'unknown')
        )
    
    def _extract_strategy_vector(self, entry: EnhancedMatrixEntry) -> np.ndarray:
        """Extract strategy vector from matrix entry."""
        # Convert strategy weights to vector
        strategies = ['momentum', 'scalping', 'mean_reversion', 'swing']
        vector = []
        
        for strategy in strategies:
            weight = entry.strategy_weights.get(strategy, 0.25)
            vector.append(weight)
        
        return np.array(vector)
    
    def _generate_profit_curve(self, entry: EnhancedMatrixEntry) -> np.ndarray:
        """Generate profit curve from performance history."""
        if entry.performance_history:
            # Use recent performance history
            recent_performance = entry.performance_history[-7:]  # Last 7 entries
            return np.array(recent_performance)
        else:
            # Default profit curve
            return np.array([100, 105, 103, 108, 110, 107, 112])
    
    def _save_enhanced_matrix(self, matrix_name: str, entry: EnhancedMatrixEntry):
        """Save enhanced matrix to file."""
        try:
            data = {
                'hash_vector': entry.hash_vector.tolist(),
                'strategy_weights': entry.strategy_weights,
                'confidence_score': entry.confidence_score,
                'entropy_alignment_score': entry.entropy_alignment_score,
                'weather_resonance_factor': entry.weather_resonance_factor,
                'geo_alignment_score': entry.geo_alignment_score,
                'crlf_adjustment_factor': entry.crlf_adjustment_factor,
                'last_updated': entry.last_updated.isoformat(),
                'performance_history': entry.performance_history,
                'metadata': entry.metadata
            }
            
            filepath = os.path.join(self.matrix_dir, matrix_name)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"💾 Saved enhanced matrix: {matrix_name}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced matrix {matrix_name}: {e}")
    
    def get_matrix_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive matrix performance summary."""
        if not self.matrices:
            return {'error': 'No matrices available'}
        
        entries = list(self.matrices.values())
        
        return {
            'total_matrices': len(self.matrices),
            'average_entropy_alignment': np.mean([e.entropy_alignment_score for e in entries]),
            'average_weather_resonance': np.mean([e.weather_resonance_factor for e in entries]),
            'average_geo_alignment': np.mean([e.geo_alignment_score for e in entries]),
            'average_crlf_adjustment': np.mean([e.crlf_adjustment_factor for e in entries]),
            'recently_updated': len([e for e in entries if (datetime.now() - e.last_updated).days < 1]),
            'location_distribution': self._get_location_distribution(),
            'performance_trends': self._get_performance_trends()
        }
    
    def _get_location_distribution(self) -> Dict[str, int]:
        """Get distribution of matrix locations."""
        distribution = {}
        for entry in self.matrices.values():
            if entry.location and entry.location.name:
                location_name = entry.location.name
                distribution[location_name] = distribution.get(location_name, 0) + 1
        return distribution
    
    def _get_performance_trends(self) -> Dict[str, float]:
        """Get performance trends across matrices."""
        if not self.matrices:
            return {}
        
        entries = list(self.matrices.values())
        
        return {
            'high_entropy_count': len([e for e in entries if e.entropy_alignment_score > 0.8]),
            'high_resonance_count': len([e for e in entries if e.weather_resonance_factor > 1.5]),
            'high_geo_alignment_count': len([e for e in entries if e.geo_alignment_score > 0.8]),
            'average_performance_history_length': np.mean([len(e.performance_history) for e in entries])
        }


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))


def load_matrix_vectors(matrix_dir: str) -> Dict[str, Any]:
    """Load all matrix vectors from JSON files in a directory."""
    matrices = {}
    for fname in os.listdir(matrix_dir):
        if fname.endswith(".json"):
            with open(os.path.join(matrix_dir, fname), "r") as f:
                matrices[fname] = json.load(f)
    return matrices


def load_matrix_from_file(matrix_file) -> np.ndarray:
    """Load a matrix from a file (supports .npy and .json formats)."""
    if str(matrix_file).endswith('.npy'):
        return np.load(matrix_file)
    elif str(matrix_file).endswith('.json'):
        with open(matrix_file, 'r') as f:
            data = json.load(f)
            return np.array(data)
    else:
        raise ValueError(f"Unsupported file format: {matrix_file}")


def match_hash_to_matrix(hash_vec, matrix_dir, threshold=0.8) -> Optional[str]:
    """Match a hash vector to the closest matrix file above threshold."""
    matrices = load_matrix_vectors(matrix_dir)
    best_score = -1
    best_file = None
    for fname, vec in matrices.items():
        score = cosine_similarity(hash_vec, vec)
        if score > best_score and score >= threshold:
            best_score = score
            best_file = fname
    return best_file


def create_enhanced_matrix_mapper(matrix_dir: str, weather_api_key: Optional[str] = None) -> EnhancedMatrixMapper:
    """Factory function to create an enhanced matrix mapper."""
    return EnhancedMatrixMapper(matrix_dir, weather_api_key)


__all__ = [
    "match_hash_to_matrix", "cosine_similarity", "load_matrix_from_file", "load_matrix_vectors",
    "EnhancedMatrixMapper", "EnhancedMatrixEntry", "create_enhanced_matrix_mapper"
]