"""
Ghost Hash Decoder - SHA256 Interpretability Engine
====================================================

Provides complete decomposition and analysis of Ghost Protocol SHA256 hashes,
enabling interpretability, debugging, and strategic optimization through
mathematical hash dissection and signal layer attribution.

Core Mathematical Principles:
- SHA256 segment mapping with deterministic input vectors
- Hamming distance analysis for similarity scoring
- Signal layer contribution tracking and weighting
- Historical hash pattern recognition and clustering
"""

import hashlib
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

class HashSegment(Enum):
    """SHA256 hash segment mappings"""
    GEOMETRIC = "geometric"      # First 64 bits: Collapse geometry patterns
    SMART_MONEY = "smart_money"  # Next 64 bits: Smart money velocity/walls
    DEPTH = "depth"              # Next 64 bits: Market depth dynamics
    TIMEBAND = "timeband"        # Last 64 bits: Time-based profit correlation

@dataclass
class HashVector:
    """Input vector for hash generation"""
    geometric_vector: np.ndarray    # Δ-reversal sequences, fractal patterns
    smart_money_vector: np.ndarray  # Spoof scores, wall positions, velocity
    depth_vector: np.ndarray        # Synthetic depth curve parameters
    timeband_vector: np.ndarray     # Time slot profit density metrics
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class HashAnalysis:
    """Complete hash analysis result"""
    hash_id: str
    segment_breakdown: Dict[HashSegment, str]  # Individual segment hashes
    vector_inputs: HashVector
    similarity_scores: Dict[str, float]        # Similarity to registry hashes
    layer_contributions: Dict[str, float]      # Contribution weight per layer
    confidence_score: float
    profit_correlation: float
    interpretability_metrics: Dict[str, Any]

class GhostHashDecoder:
    """SHA256 hash decoder for Ghost Protocol interpretability"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.hash_registry: Dict[str, Dict[str, Any]] = {}
        self.segment_mapping = self._initialize_segment_mapping()
        self.similarity_cache: Dict[str, Dict[str, float]] = {}
        self.layer_weights = self._initialize_layer_weights()
        self.interpretability_thresholds = {
            'high_similarity': 0.85,
            'medium_similarity': 0.65,
            'low_similarity': 0.45,
            'noise_threshold': 0.25
        }
        
    def _initialize_segment_mapping(self) -> Dict[HashSegment, Dict[str, int]]:
        """Initialize deterministic SHA256 segment mapping"""
        return {
            HashSegment.GEOMETRIC: {'start_bit': 0, 'end_bit': 64},
            HashSegment.SMART_MONEY: {'start_bit': 64, 'end_bit': 128},
            HashSegment.DEPTH: {'start_bit': 128, 'end_bit': 192},
            HashSegment.TIMEBAND: {'start_bit': 192, 'end_bit': 256}
        }
    
    def _initialize_layer_weights(self) -> Dict[str, float]:
        """Initialize dynamic layer weighting system"""
        return {
            'geometric': 0.25,
            'smart_money': 0.35,
            'depth': 0.20,
            'timeband': 0.20
        }
    
    def generate_ghost_hash(self, vector: HashVector) -> str:
        """Generate SHA256 hash from input vectors with segment determinism"""
        
        # Normalize and quantize vectors for deterministic hashing
        geom_bytes = self._vectorize_geometric_signals(vector.geometric_vector)
        sm_bytes = self._vectorize_smart_money_signals(vector.smart_money_vector)
        depth_bytes = self._vectorize_depth_signals(vector.depth_vector)
        time_bytes = self._vectorize_timeband_signals(vector.timeband_vector)
        
        # Concatenate with segment delimiters for reproducible parsing
        combined_input = (
            f"GEOM:{geom_bytes.hex()}|"
            f"SM:{sm_bytes.hex()}|"
            f"DEPTH:{depth_bytes.hex()}|"
            f"TIME:{time_bytes.hex()}"
        ).encode()
        
        # Generate SHA256 with timestamp stability
        hash_input = combined_input + str(int(vector.timestamp)).encode()
        ghost_hash = hashlib.sha256(hash_input).hexdigest()
        
        return ghost_hash
    
    def decompose_hash(self, ghost_hash: str, vector: Optional[HashVector] = None) -> HashAnalysis:
        """Decompose SHA256 hash into interpretable components"""
        
        # Convert hex to binary for segment extraction
        binary_hash = bin(int(ghost_hash, 16))[2:].zfill(256)
        
        # Extract segments according to mapping
        segment_breakdown = {}
        for segment, mapping in self.segment_mapping.items():
            start, end = mapping['start_bit'], mapping['end_bit']
            segment_bits = binary_hash[start:end]
            segment_hex = hex(int(segment_bits, 2))[2:].zfill(16)
            segment_breakdown[segment] = segment_hex
        
        # Calculate similarity scores to registry
        similarity_scores = self._calculate_registry_similarities(ghost_hash)
        
        # Analyze layer contributions
        layer_contributions = self._analyze_layer_contributions(segment_breakdown, vector)
        
        # Calculate confidence and profit correlation
        confidence_score = self._calculate_confidence_score(similarity_scores, layer_contributions)
        profit_correlation = self._calculate_profit_correlation(ghost_hash, similarity_scores)
        
        # Generate interpretability metrics
        interpretability_metrics = self._generate_interpretability_metrics(
            segment_breakdown, similarity_scores, layer_contributions
        )
        
        return HashAnalysis(
            hash_id=ghost_hash,
            segment_breakdown=segment_breakdown,
            vector_inputs=vector,
            similarity_scores=similarity_scores,
            layer_contributions=layer_contributions,
            confidence_score=confidence_score,
            profit_correlation=profit_correlation,
            interpretability_metrics=interpretability_metrics
        )
    
    def _vectorize_geometric_signals(self, vector: np.ndarray) -> bytes:
        """Convert geometric signals to deterministic byte representation"""
        # Normalize to [0, 1] and quantize to 8-bit precision
        normalized = (vector - vector.min()) / (vector.max() - vector.min() + 1e-10)
        quantized = (normalized * 255).astype(np.uint8)
        return quantized.tobytes()
    
    def _vectorize_smart_money_signals(self, vector: np.ndarray) -> bytes:
        """Convert smart money signals to deterministic byte representation"""
        # Apply sigmoid normalization for bounded values
        sigmoid_normalized = 1 / (1 + np.exp(-vector))
        quantized = (sigmoid_normalized * 255).astype(np.uint8)
        return quantized.tobytes()
    
    def _vectorize_depth_signals(self, vector: np.ndarray) -> bytes:
        """Convert depth signals to deterministic byte representation"""
        # Use log normalization for depth data (often power-law distributed)
        log_normalized = np.log1p(np.abs(vector)) / np.log1p(np.abs(vector).max() + 1e-10)
        quantized = (log_normalized * 255).astype(np.uint8)
        return quantized.tobytes()
    
    def _vectorize_timeband_signals(self, vector: np.ndarray) -> bytes:
        """Convert timeband signals to deterministic byte representation"""
        # Circular normalization for time-based signals
        circular_normalized = (np.sin(vector) + 1) / 2
        quantized = (circular_normalized * 255).astype(np.uint8)
        return quantized.tobytes()
    
    def _calculate_registry_similarities(self, ghost_hash: str) -> Dict[str, float]:
        """Calculate Hamming distance similarities to all registry hashes"""
        similarities = {}
        
        if ghost_hash in self.similarity_cache:
            return self.similarity_cache[ghost_hash]
        
        ghost_binary = bin(int(ghost_hash, 16))[2:].zfill(256)
        
        for registry_hash, registry_data in self.hash_registry.items():
            registry_binary = bin(int(registry_hash, 16))[2:].zfill(256)
            
            # Calculate Hamming distance
            hamming_distance = sum(b1 != b2 for b1, b2 in zip(ghost_binary, registry_binary))
            
            # Convert to similarity score (0-1)
            similarity = 1.0 - (hamming_distance / 256.0)
            similarities[registry_hash] = similarity
        
        # Cache result
        self.similarity_cache[ghost_hash] = similarities
        return similarities
    
    def _analyze_layer_contributions(self, segment_breakdown: Dict[HashSegment, str], 
                                   vector: Optional[HashVector]) -> Dict[str, float]:
        """Analyze contribution weight of each signal layer"""
        contributions = {}
        
        if vector is None:
            # Use default weights if no vector provided
            return dict(self.layer_weights)
        
        # Calculate entropy-based contribution weights
        for segment, segment_hash in segment_breakdown.items():
            # Calculate entropy of segment
            segment_binary = bin(int(segment_hash, 16))[2:].zfill(64)
            bit_counts = [segment_binary.count('0'), segment_binary.count('1')]
            entropy = -sum((count/64) * np.log2(count/64 + 1e-10) for count in bit_counts if count > 0)
            
            # Map segment to layer name
            layer_name = segment.value.lower()
            base_weight = self.layer_weights.get(layer_name, 0.25)
            
            # Adjust weight based on entropy (higher entropy = more information)
            entropy_adjustment = entropy / 1.0  # Max entropy for binary is 1.0
            contributions[layer_name] = base_weight * (1.0 + entropy_adjustment)
        
        # Normalize contributions to sum to 1.0
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {k: v/total_contribution for k, v in contributions.items()}
        
        return contributions
    
    def _calculate_confidence_score(self, similarity_scores: Dict[str, float], 
                                  layer_contributions: Dict[str, float]) -> float:
        """Calculate overall confidence score for ghost hash"""
        
        if not similarity_scores:
            return 0.0
        
        # Weight similarities by layer contributions and registry profit history
        weighted_similarities = []
        for registry_hash, similarity in similarity_scores.items():
            registry_data = self.hash_registry.get(registry_hash, {})
            profit_weight = registry_data.get('profit_correlation', 0.5)
            
            # Combine similarity with profit history and layer contributions
            layer_weight = sum(layer_contributions.values()) / len(layer_contributions)
            weighted_similarity = similarity * profit_weight * layer_weight
            weighted_similarities.append(weighted_similarity)
        
        # Use top 3 similarities for confidence calculation
        top_similarities = sorted(weighted_similarities, reverse=True)[:3]
        confidence = np.mean(top_similarities) if top_similarities else 0.0
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_profit_correlation(self, ghost_hash: str, 
                                    similarity_scores: Dict[str, float]) -> float:
        """Calculate profit correlation based on similar hash performance"""
        
        if not similarity_scores:
            return 0.0
        
        profit_correlations = []
        for registry_hash, similarity in similarity_scores.items():
            if similarity > self.interpretability_thresholds['low_similarity']:
                registry_data = self.hash_registry.get(registry_hash, {})
                profit_history = registry_data.get('profit_history', [])
                
                if profit_history:
                    # Weight profit by similarity
                    avg_profit = np.mean(profit_history)
                    weighted_profit = avg_profit * similarity
                    profit_correlations.append(weighted_profit)
        
        return np.mean(profit_correlations) if profit_correlations else 0.0
    
    def _generate_interpretability_metrics(self, segment_breakdown: Dict[HashSegment, str],
                                         similarity_scores: Dict[str, float],
                                         layer_contributions: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive interpretability metrics"""
        
        # Segment entropy analysis
        segment_entropies = {}
        for segment, segment_hash in segment_breakdown.items():
            segment_binary = bin(int(segment_hash, 16))[2:].zfill(64)
            bit_counts = [segment_binary.count('0'), segment_binary.count('1')]
            entropy = -sum((count/64) * np.log2(count/64 + 1e-10) for count in bit_counts if count > 0)
            segment_entropies[segment.value] = entropy
        
        # Similarity clustering
        high_similarity_count = sum(1 for sim in similarity_scores.values() 
                                  if sim > self.interpretability_thresholds['high_similarity'])
        medium_similarity_count = sum(1 for sim in similarity_scores.values() 
                                    if self.interpretability_thresholds['medium_similarity'] < sim <= self.interpretability_thresholds['high_similarity'])
        
        # Layer dominance analysis
        dominant_layer = max(layer_contributions.items(), key=lambda x: x[1])[0] if layer_contributions else None
        layer_balance = np.std(list(layer_contributions.values())) if layer_contributions else 0.0
        
        return {
            'segment_entropies': segment_entropies,
            'high_similarity_matches': high_similarity_count,
            'medium_similarity_matches': medium_similarity_count,
            'dominant_layer': dominant_layer,
            'layer_balance_std': layer_balance,
            'interpretability_score': self._calculate_interpretability_score(
                segment_entropies, similarity_scores, layer_contributions
            )
        }
    
    def _calculate_interpretability_score(self, segment_entropies: Dict[str, float],
                                        similarity_scores: Dict[str, float],
                                        layer_contributions: Dict[str, float]) -> float:
        """Calculate overall interpretability score (0-1)"""
        
        # Entropy component (higher entropy = more interpretable patterns)
        entropy_score = np.mean(list(segment_entropies.values())) if segment_entropies else 0.0
        
        # Similarity component (some similarities = more interpretable)
        similarity_score = min(1.0, len([s for s in similarity_scores.values() 
                                       if s > self.interpretability_thresholds['low_similarity']]) / 10.0)
        
        # Balance component (balanced layers = more interpretable)
        balance_score = 1.0 - min(1.0, np.std(list(layer_contributions.values())) if layer_contributions else 1.0)
        
        # Combined interpretability score
        interpretability = (entropy_score * 0.4 + similarity_score * 0.3 + balance_score * 0.3)
        return min(1.0, max(0.0, interpretability))
    
    def update_registry(self, ghost_hash: str, profit_result: float, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update hash registry with profit results"""
        
        if ghost_hash not in self.hash_registry:
            self.hash_registry[ghost_hash] = {
                'profit_history': [],
                'trade_count': 0,
                'first_seen': time.time(),
                'metadata': metadata or {}
            }
        
        registry_entry = self.hash_registry[ghost_hash]
        registry_entry['profit_history'].append(profit_result)
        registry_entry['trade_count'] += 1
        registry_entry['last_updated'] = time.time()
        
        # Calculate rolling profit correlation
        profit_history = registry_entry['profit_history']
        registry_entry['profit_correlation'] = np.mean(profit_history) if profit_history else 0.0
        registry_entry['profit_volatility'] = np.std(profit_history) if len(profit_history) > 1 else 0.0
        
        # Clear similarity cache to force recalculation
        self.similarity_cache.clear()
    
    def get_hash_insights(self, ghost_hash: str) -> Dict[str, Any]:
        """Get comprehensive insights for a specific hash"""
        
        analysis = self.decompose_hash(ghost_hash)
        registry_data = self.hash_registry.get(ghost_hash, {})
        
        insights = {
            'hash_analysis': analysis,
            'registry_data': registry_data,
            'recommendations': self._generate_recommendations(analysis, registry_data)
        }
        
        return insights
    
    def _generate_recommendations(self, analysis: HashAnalysis, 
                                registry_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on hash analysis"""
        
        recommendations = []
        
        # Confidence-based recommendations
        if analysis.confidence_score > 0.8:
            recommendations.append("HIGH CONFIDENCE: Strong pattern match - proceed with trade")
        elif analysis.confidence_score > 0.6:
            recommendations.append("MEDIUM CONFIDENCE: Moderate pattern match - use reduced position size")
        else:
            recommendations.append("LOW CONFIDENCE: Weak pattern match - consider skipping trade")
        
        # Layer-based recommendations
        dominant_layer = analysis.interpretability_metrics.get('dominant_layer')
        if dominant_layer == 'smart_money':
            recommendations.append("SMART MONEY DOMINANT: Monitor for spoofing activity")
        elif dominant_layer == 'geometric':
            recommendations.append("GEOMETRIC DOMINANT: Strong technical pattern - verify with additional indicators")
        elif dominant_layer == 'depth':
            recommendations.append("DEPTH DOMINANT: Liquidity-driven signal - check order book stability")
        elif dominant_layer == 'timeband':
            recommendations.append("TIMEBAND DOMINANT: Time-sensitive pattern - execute quickly")
        
        # Similarity-based recommendations
        high_sim_count = analysis.interpretability_metrics.get('high_similarity_matches', 0)
        if high_sim_count > 5:
            recommendations.append("MULTIPLE MATCHES: Well-established pattern with historical precedent")
        elif high_sim_count == 0:
            recommendations.append("NOVEL PATTERN: No close historical matches - proceed with caution")
        
        return recommendations
    
    def save_registry(self, filepath: str) -> None:
        """Save hash registry to file"""
        with open(filepath, 'w') as f:
            json.dump(self.hash_registry, f, indent=2, default=str)
    
    def load_registry(self, filepath: str) -> None:
        """Load hash registry from file"""
        try:
            with open(filepath, 'r') as f:
                self.hash_registry = json.load(f)
            # Clear similarity cache after loading new registry
            self.similarity_cache.clear()
        except FileNotFoundError:
            self.hash_registry = {}

# Utility functions for external integration
def create_hash_vector_from_signals(geometric_signals: np.ndarray,
                                   smart_money_signals: np.ndarray,
                                   depth_signals: np.ndarray,
                                   timeband_signals: np.ndarray) -> HashVector:
    """Create HashVector from individual signal arrays"""
    return HashVector(
        geometric_vector=geometric_signals,
        smart_money_vector=smart_money_signals,
        depth_vector=depth_signals,
        timeband_vector=timeband_signals
    )

def analyze_hash_similarity_cluster(decoder: GhostHashDecoder, 
                                  hash_list: List[str]) -> Dict[str, Any]:
    """Analyze similarity clustering across multiple hashes"""
    
    cluster_analysis = {
        'hash_count': len(hash_list),
        'similarity_matrix': {},
        'cluster_metrics': {},
        'dominant_patterns': []
    }
    
    # Build similarity matrix
    for i, hash1 in enumerate(hash_list):
        cluster_analysis['similarity_matrix'][hash1] = {}
        for j, hash2 in enumerate(hash_list):
            if i != j:
                similarities1 = decoder._calculate_registry_similarities(hash1)
                similarity = similarities1.get(hash2, 0.0)
                cluster_analysis['similarity_matrix'][hash1][hash2] = similarity
    
    return cluster_analysis 