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
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from numpy.linalg import norm
from dataclasses import dataclass, field
from datetime import datetime

from .crwf_crlf_integration import create_crwf_crlf_integration
from .schwafit_core import SchwafitCore  # <-- NEW: Import SchwafitCore

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
    location: Optional[Any] = None  # GeoLocation type
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
        self.schwafit = SchwafitCore(window=64)  # <-- NEW: Schwafit instance

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
        location: Any,  # GeoLocation type
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
        location: Any,  # GeoLocation type
        threshold: float = 0.8
    ) -> Optional[Tuple[str, EnhancedMatrixEntry, float, Dict[str, Any]]]:  # <-- NEW: Add Schwafit info block
        """
        Match hash vector to enhanced matrix with location-aware scoring and Schwafit fit info.

        Returns:
            Tuple of (matrix_name, enhanced_entry, schwafit_fit_score, schwafit_info_block) or None
        """
        try:
            best_score = -1
            best_file = None
            best_entry = None
            best_schwafit_info = None

            for fname, entry in self.matrices.items():
                # Extract strategy vector and profit curve for Schwafit
                strategy_vector = self._extract_strategy_vector(entry)
                profit_curve = self._generate_profit_curve(entry)
                # Prepare pattern library and profit scores for Schwafit
                pattern_library = [strategy_vector]
                profit_scores = [float(np.mean(profit_curve)) if len(profit_curve) else 0.0]

                # Use Schwafit to compute fit
                schwafit_info = self.schwafit.fit_vector(
                    price_series=hash_vec.tolist(),
                    pattern_library=pattern_library,
                    profit_scores=profit_scores
                )
                schwafit_score = schwafit_info["fit_score"]

                # Optionally, combine with enhanced similarity (location, weather, entropy, etc.)
                enhanced_score = self._compute_enhanced_similarity(
                    schwafit_score, entry, location
                )

                if enhanced_score > best_score and enhanced_score > threshold:
                    best_score = enhanced_score
                    best_file = fname
                    best_entry = entry
                    best_schwafit_info = schwafit_info

            if best_file and best_entry:
                return best_file, best_entry, best_score, best_schwafit_info  # <-- Return Schwafit info block

            return None

        except Exception as e:
            logger.error(f"Error matching hash to enhanced matrix: {e}")
            return None
    
    def get_entropy_aligned_matrices(
        self,
        location: Any,  # GeoLocation type
        min_entropy_score: float = 0.7
    ) -> List[Tuple[str, EnhancedMatrixEntry]]:
        """
        Get matrices with high entropy alignment for a location.
        
        Args:
            location: Geographic location
            min_entropy_score: Minimum entropy alignment score
            
        Returns:
            List of (matrix_name, entry) tuples
        """
        aligned_matrices = []
        
        for fname, entry in self.matrices.items():
            if (entry.entropy_alignment_score >= min_entropy_score and
                self._is_location_compatible(entry, location)):
                aligned_matrices.append((fname, entry))
        
        # Sort by entropy alignment score
        aligned_matrices.sort(key=lambda x: x[1].entropy_alignment_score, reverse=True)
        
        return aligned_matrices
    
    def create_geo_optimized_matrix(
        self,
        base_hash_vector: np.ndarray,
        location: Any,  # GeoLocation type
        strategy_weights: Dict[str, float],
        weather_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a new geo-optimized matrix entry.
        
        Args:
            base_hash_vector: Base hash vector
            location: Geographic location
            strategy_weights: Strategy weight dictionary
            weather_data: Optional weather data
            
        Returns:
            Matrix name if successful, None otherwise
        """
        try:
            # Get weather data if not provided
            if weather_data is None:
                weather_data = self._get_weather_data_for_location(location)
            
            # Create weather data point
            weather_point = self._create_weather_data_point(location, weather_data)
            
            # Compute CRWF for the new location
            crwf_response = self.crwf_crlf_integration.crwf_mapper.compute_crwf(
                weather_point, location
            )
            
            # Create strategy vector
            strategy_vector = np.array(list(strategy_weights.values()))
            profit_curve = np.ones(len(strategy_vector)) * 0.5  # Default profit curve
            
            # Compute CRLF
            crlf_response = self.crwf_crlf_integration.crlf_function.compute_crlf(
                strategy_vector, profit_curve, crwf_response.entropy_score
            )
            
            # Create enhanced matrix entry
            entry = EnhancedMatrixEntry(
                hash_vector=base_hash_vector,
                strategy_weights=strategy_weights,
                confidence_score=0.8,
                entropy_alignment_score=crwf_response.entropy_score,
                weather_resonance_factor=crwf_response.resonance_strength,
                geo_alignment_score=crwf_response.geo_alignment_score,
                crlf_adjustment_factor=crwf_response.crlf_adjustment_factor,
                location=location,
                weather_data=weather_data
            )
            
            # Generate matrix name
            matrix_name = (
                f"geo_optimized_{location.lat:.2f}_{location.lon:.2f}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            # Save matrix
            self._save_enhanced_matrix(matrix_name, entry)
            self.matrices[matrix_name] = entry
            
            logger.info(f"✅ Created geo-optimized matrix: {matrix_name}")
            return matrix_name
            
        except Exception as e:
            logger.error(f"Error creating geo-optimized matrix: {e}")
            return None
    
    def _compute_enhanced_similarity(
        self,
        base_score: float,
        entry: EnhancedMatrixEntry,
        location: Any  # GeoLocation type
    ) -> float:
        """Compute enhanced similarity score with location awareness."""
        try:
            # Location compatibility factor
            location_factor = 1.0
            if entry.location:
                # Calculate distance-based factor
                distance = self._calculate_distance(location, entry.location)
                location_factor = 1.0 / (1.0 + distance / 1000.0)  # Decay over 1000km
            
            # Weather resonance factor
            weather_factor = entry.weather_resonance_factor
            
            # Entropy alignment factor
            entropy_factor = entry.entropy_alignment_score
            
            # Combine factors
            enhanced_score = (
                base_score * 0.4 +
                location_factor * 0.2 +
                weather_factor * 0.2 +
                entropy_factor * 0.2
            )
            
            return float(enhanced_score)
            
        except Exception as e:
            logger.error(f"Error computing enhanced similarity: {e}")
            return base_score
    
    def _is_location_compatible(
        self,
        entry: EnhancedMatrixEntry,
        location: Any  # GeoLocation type
    ) -> bool:
        """Check if matrix entry is compatible with given location."""
        if entry.location is None:
            return True  # No location constraint
        
        # Calculate distance
        distance = self._calculate_distance(location, entry.location)
        
        # Consider compatible if within 500km
        return distance <= 500.0
    
    def _calculate_distance(self, loc1: Any, loc2: Any) -> float:  # GeoLocation types
        """Calculate distance between two locations in kilometers."""
        import math
        
        lat1, lon1 = math.radians(loc1.lat), math.radians(loc1.lon)
        lat2, lon2 = math.radians(loc2.lat), math.radians(loc2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return 6371 * c  # Earth radius in km
    
    def _get_weather_data_for_location(self, location: Any) -> Dict[str, Any]:  # GeoLocation type
        """Get weather data for a location."""
        try:
            # This would typically call a weather API
            # For now, return mock data
            return {
                "temperature": 20.0,
                "humidity": 60.0,
                "pressure": 1013.25,
                "wind_speed": 5.0,
                "conditions": "clear"
            }
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return {}
    
    def _create_weather_data_point(
        self,
        location: Any,  # GeoLocation type
        weather_data: Dict[str, Any]
    ):
        """Create weather data point for CRWF analysis."""
        # This would create the appropriate data structure
        # Implementation depends on the CRWF interface
        return weather_data
    
    def _extract_strategy_vector(self, entry: EnhancedMatrixEntry) -> np.ndarray:
        """Extract strategy vector from matrix entry."""
        return np.array(list(entry.strategy_weights.values()))
    
    def _generate_profit_curve(self, entry: EnhancedMatrixEntry) -> np.ndarray:
        """Generate profit curve from performance history."""
        if entry.performance_history:
            return np.array(entry.performance_history)
        else:
            return np.ones(len(entry.strategy_weights)) * 0.5
    
    def _save_enhanced_matrix(self, matrix_name: str, entry: EnhancedMatrixEntry):
        """Save enhanced matrix to file."""
        try:
            filepath = os.path.join(self.matrix_dir, matrix_name)
            
            data = {
                'hash_vector': entry.hash_vector.tolist(),
                'strategy_weights': entry.strategy_weights,
                'confidence_score': entry.confidence_score,
                'entropy_alignment_score': entry.entropy_alignment_score,
                'weather_resonance_factor': entry.weather_resonance_factor,
                'geo_alignment_score': entry.geo_alignment_score,
                'crlf_adjustment_factor': entry.crlf_adjustment_factor,
                'metadata': entry.metadata
            }
            
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"💾 Saved enhanced matrix: {matrix_name}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced matrix {matrix_name}: {e}")
    
    def get_matrix_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all matrices."""
        try:
            total_matrices = len(self.matrices)
            if total_matrices == 0:
                return {"total_matrices": 0}
            
            # Calculate average scores
            avg_entropy = np.mean([e.entropy_alignment_score for e in self.matrices.values()])
            avg_weather = np.mean([e.weather_resonance_factor for e in self.matrices.values()])
            avg_geo = np.mean([e.geo_alignment_score for e in self.matrices.values()])
            
            # Get location distribution
            location_dist = self._get_location_distribution()
            
            # Get performance trends
            performance_trends = self._get_performance_trends()
            
            return {
                "total_matrices": total_matrices,
                "average_entropy_alignment": float(avg_entropy),
                "average_weather_resonance": float(avg_weather),
                "average_geo_alignment": float(avg_geo),
                "location_distribution": location_dist,
                "performance_trends": performance_trends
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"total_matrices": 0}
    
    def _get_location_distribution(self) -> Dict[str, int]:
        """Get distribution of matrix locations."""
        location_counts = {}
        
        for entry in self.matrices.values():
            if entry.location:
                location_key = f"{entry.location.lat:.1f},{entry.location.lon:.1f}"
                location_counts[location_key] = location_counts.get(location_key, 0) + 1
        
        return location_counts
    
    def _get_performance_trends(self) -> Dict[str, float]:
        """Get performance trends from matrix history."""
        try:
            all_performances = []
            for entry in self.matrices.values():
                all_performances.extend(entry.performance_history)
            
            if not all_performances:
                return {"average_performance": 0.0, "trend": 0.0}
            
            avg_performance = np.mean(all_performances)
            
            # Calculate trend (simple linear regression)
            if len(all_performances) > 1:
                x = np.arange(len(all_performances))
                trend = np.polyfit(x, all_performances, 1)[0]
            else:
                trend = 0.0
            
            return {
                "average_performance": float(avg_performance),
                "trend": float(trend)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance trends: {e}")
            return {"average_performance": 0.0, "trend": 0.0}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def load_matrix_vectors(matrix_dir: str) -> Dict[str, Any]:
    """Load matrix vectors from directory."""
    matrices = {}
    for fname in os.listdir(matrix_dir):
        if fname.endswith(".npy"):
            filepath = os.path.join(matrix_dir, fname)
            matrices[fname] = np.load(filepath)
    return matrices


def load_matrix_from_file(matrix_file) -> np.ndarray:
    """Load matrix from file."""
    if matrix_file.suffix == ".npy":
        return np.load(matrix_file)
    else:
        # Handle other formats as needed
        return np.array([])


def match_hash_to_matrix(hash_vec: np.ndarray, matrix_dir, threshold: float = 0.8) -> Optional[str]:
    """Match hash vector to matrix file."""
    best_score = -1
    best_file = None
    
    for matrix_file in matrix_dir.glob("*.npy"):
        matrix = load_matrix_from_file(matrix_file)
        if len(matrix) > 0:
            score = cosine_similarity(hash_vec, matrix)
            if score > best_score:
                best_score = score
                best_file = matrix_file
    
    if best_score > threshold and best_file:
        return str(best_file)
    
    return None


def create_enhanced_matrix_mapper(matrix_dir: str, weather_api_key: Optional[str] = None) -> EnhancedMatrixMapper:
    """Create enhanced matrix mapper instance."""
    return EnhancedMatrixMapper(matrix_dir, weather_api_key)