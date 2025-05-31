"""
Aleph Unitizer Library - Nexus Edition
=====================================
Advanced hash-based signal processing system for Schwabot.
Creates tesseract-style pattern mapping from SHA-256 hashes for contextual analysis.

Core Philosophy:
- Transform raw signals into recursive hash signatures
- Generate compact memory keys for fast lookup
- Create 8-dimensional pattern vectors for similarity matching
- Enable batch processing with entropy-based prioritization
"""

import hashlib
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class AlephUnitizer:
    """
    The Aleph Unitizer transforms raw market signals and data into structured
    hash-based representations for contextual pattern matching and batch processing.
    """
    
    def __init__(self, memory_path: str = "aleph_memory"):
        self.session_log = []
        self.pattern_cache = {}
        self.entropy_distribution = {}
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(exist_ok=True)
        
        # Initialize tesseract mapping coefficients
        self.tesseract_weights = [1, 3, 7, 12, 5, 9, 2, 11]  # Prime-like progression
    
    def aleph_seed(self, raw_data: str) -> Dict:
        """
        Create Aleph signature with multiple dimensional representations.
        """
        sha = hashlib.sha256(raw_data.encode()).hexdigest()
        short_tag = sha[:8]
        
        # Entropy calculation with cyclic modulo for categorical binning
        entropy_tag = sum([ord(c) for c in short_tag]) % 144
        
        # Track entropy distribution for analysis
        self.entropy_distribution[entropy_tag] = self.entropy_distribution.get(entropy_tag, 0) + 1
        
        return {
            "full_hash": sha,
            "short_tag": short_tag,
            "entropy_tag": entropy_tag,
            "timestamp": time.time()
        }

    def simplify_sha(self, sha: str) -> List[int]:
        """
        Generate 8-dimensional tesseract pattern vector from SHA-256 hash.
        """
        parts = [sha[i:i+4] for i in range(0, len(sha), 4)]
        pattern = [sum([ord(c) for c in part]) % 16 for part in parts[:8]]
        
        # Apply tesseract weighting for enhanced separation
        weighted_pattern = [(pattern[i] * self.tesseract_weights[i]) % 16 for i in range(8)]
        
        return weighted_pattern

    def calculate_pattern_similarity(self, pattern1: List[int], pattern2: List[int]) -> float:
        """
        Calculate similarity between two tesseract patterns using multiple metrics.
        """
        if len(pattern1) != len(pattern2):
            return 0.0
        
        # Hamming distance (categorical comparison)
        hamming_dist = sum(1 for i in range(8) if pattern1[i] != pattern2[i])
        hamming_sim = 1.0 - (hamming_dist / 8.0)
        
        # Manhattan distance (ordinal comparison)
        manhattan_dist = sum(abs(pattern1[i] - pattern2[i]) for i in range(8))
        manhattan_sim = 1.0 - (manhattan_dist / (8.0 * 15.0))  # Max possible distance
        
        # Cosine similarity (vector angle)
        dot_product = sum(pattern1[i] * pattern2[i] for i in range(8))
        norm1 = sum(x**2 for x in pattern1)**0.5
        norm2 = sum(x**2 for x in pattern2)**0.5
        
        if norm1 == 0 or norm2 == 0:
            cosine_sim = 0.0
        else:
            cosine_sim = dot_product / (norm1 * norm2)
        
        # Weighted combination of similarities
        combined_similarity = (0.4 * hamming_sim + 0.4 * manhattan_sim + 0.2 * cosine_sim)
        return combined_similarity

    def find_contextual_matches(self, target_pattern: List[int], threshold: float = 0.7) -> List[Dict]:
        """
        Find all cached patterns with similarity above threshold.
        """
        matches = []
        
        for hash_key, cached_data in self.pattern_cache.items():
            similarity = self.calculate_pattern_similarity(target_pattern, cached_data['pattern'])
            
            if similarity >= threshold:
                matches.append({
                    'hash': hash_key,
                    'similarity': similarity,
                    'data': cached_data,
                    'timestamp': cached_data.get('timestamp', 0)
                })
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches

    def unitize_message(self, message: str, asset: str = "BTCUSDC", 
                       trigger: str = "", confidence: float = 0.5) -> Tuple[str, int, List[int]]:
        """
        Convert raw signal into Aleph-unitized representation with batch integration.
        """
        # Generate core Aleph representations
        aleph = self.aleph_seed(message)
        tesseract_pattern = self.simplify_sha(aleph['full_hash'])
        
        # Find similar patterns for context
        contextual_matches = self.find_contextual_matches(tesseract_pattern)
        
        # Update pattern cache
        self.pattern_cache[aleph['short_tag']] = {
            'pattern': tesseract_pattern,
            'message': message,
            'asset': asset,
            'timestamp': aleph['timestamp'],
            'entropy': aleph['entropy_tag'],
            'matches_found': len(contextual_matches)
        }
        
        # Create comprehensive log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": message,
            "asset": asset,
            "hash": aleph['full_hash'],
            "short_tag": aleph['short_tag'],
            "entropy": aleph['entropy_tag'],
            "tesseract": tesseract_pattern,
            "trigger": trigger,
            "confidence": confidence,
            "contextual_matches": len(contextual_matches),
            "top_match_similarity": contextual_matches[0]['similarity'] if contextual_matches else 0.0
        }
        
        self.session_log.append(log_entry)
        
        # Integrate with batch processing if available
        try:
            from nano_core.batch_hash_processor import add_hash_to_batch
            add_hash_to_batch(asset, message, aleph['full_hash'], trigger)
        except ImportError:
            print(f"[ALEPH] Batch processor not available, storing locally")
        
        return aleph['short_tag'], aleph['entropy_tag'], tesseract_pattern

    def analyze_entropy_distribution(self) -> Dict:
        """
        Analyze the distribution of entropy tags for balance assessment.
        """
        if not self.entropy_distribution:
            return {}
        
        values = list(self.entropy_distribution.values())
        total_samples = sum(values)
        
        return {
            'total_samples': total_samples,
            'unique_entropies': len(self.entropy_distribution),
            'mean_frequency': np.mean(values),
            'std_frequency': np.std(values),
            'min_entropy': min(self.entropy_distribution.keys()),
            'max_entropy': max(self.entropy_distribution.keys()),
            'distribution': dict(sorted(self.entropy_distribution.items()))
        }

    def export_pattern_map(self, filepath: str = None) -> str:
        """
        Export current pattern cache as JSON for external analysis.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.memory_path / f"aleph_patterns_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'export_time': datetime.utcnow().isoformat(),
                'total_patterns': len(self.pattern_cache),
                'entropy_analysis': self.analyze_entropy_distribution()
            },
            'patterns': self.pattern_cache,
            'session_log': self.session_log[-100:]  # Last 100 entries
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(filepath)

    def load_pattern_map(self, filepath: str) -> bool:
        """
        Load previously exported pattern cache.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.pattern_cache.update(data.get('patterns', {}))
            
            # Rebuild entropy distribution
            for pattern_data in self.pattern_cache.values():
                entropy = pattern_data.get('entropy', 0)
                self.entropy_distribution[entropy] = self.entropy_distribution.get(entropy, 0) + 1
            
            return True
        except Exception as e:
            print(f"[ALEPH] Failed to load pattern map: {e}")
            return False

    def get_contextual_summary(self) -> Dict:
        """
        Generate summary of current Aleph state for monitoring.
        """
        recent_log = self.session_log[-10:] if self.session_log else []
        
        return {
            'total_processed': len(self.session_log),
            'pattern_cache_size': len(self.pattern_cache),
            'entropy_diversity': len(self.entropy_distribution),
            'recent_activity': len(recent_log),
            'avg_contextual_matches': np.mean([entry.get('contextual_matches', 0) 
                                              for entry in recent_log]) if recent_log else 0,
            'system_status': 'active' if self.session_log else 'initialized'
        }

    def clear_cache(self, older_than_hours: int = 24):
        """
        Clean old entries from pattern cache to prevent memory bloat.
        """
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        
        # Filter pattern cache
        self.pattern_cache = {
            k: v for k, v in self.pattern_cache.items() 
            if v.get('timestamp', 0) > cutoff_time
        }
        
        # Rebuild entropy distribution from remaining cache
        self.entropy_distribution = {}
        for pattern_data in self.pattern_cache.values():
            entropy = pattern_data.get('entropy', 0)
            self.entropy_distribution[entropy] = self.entropy_distribution.get(entropy, 0) + 1

# Tesseract Portal Interface
class TesseractPortal:
    """
    Advanced portal for multi-dimensional pattern analysis using Aleph signatures.
    Provides geometric interpretation of hash patterns in 8D tesseract space.
    """
    
    def __init__(self, unitizer: AlephUnitizer):
        self.unitizer = unitizer
        self.dimensional_weights = [0.125] * 8  # Equal weighting initially
    
    def map_to_tesseract_coordinates(self, pattern: List[int]) -> Dict:
        """
        Map 8D pattern to tesseract geometric coordinates with interpretations.
        """
        coords = {
            'sentiment': (pattern[0] - 7.5, pattern[1] - 7.5),      # Centered on 7.5
            'volatility': (pattern[2] - 7.5, pattern[3] - 7.5),
            'momentum': (pattern[4] - 7.5, pattern[5] - 7.5),
            'temporal': (pattern[6] - 7.5, pattern[7] - 7.5),
            'magnitude': sum(p**2 for p in pattern)**0.5,           # Distance from origin
            'phase': np.arctan2(sum(pattern[1::2]), sum(pattern[::2]))  # Angular position
        }
        
        return coords
    
    def find_tesseract_clusters(self, patterns: List[List[int]], max_clusters: int = 8) -> Dict:
        """
        Identify clusters in tesseract space using k-means clustering.
        """
        if len(patterns) < max_clusters:
            return {'clusters': [], 'centroids': []}
        
        from sklearn.cluster import KMeans
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(max_clusters, len(patterns)//2))
        cluster_labels = kmeans.fit_predict(patterns)
        
        # Analyze clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(patterns[i])
        
        return {
            'clusters': clusters,
            'centroids': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_
        }

# Schwabot Integration Module
def initialize_aleph_for_schwabot(memory_path: str = "schwabot_aleph") -> AlephUnitizer:
    """
    Initialize Aleph Unitizer optimized for Schwabot integration.
    """
    unitizer = AlephUnitizer(memory_path)
    
    # Load existing patterns if available
    pattern_files = list(Path(memory_path).glob("aleph_patterns_*.json"))
    if pattern_files:
        latest_file = max(pattern_files, key=lambda p: p.stat().st_mtime)
        unitizer.load_pattern_map(str(latest_file))
        print(f"[ALEPH] Loaded existing patterns from {latest_file}")
    
    return unitizer

# Example Usage and Testing
if __name__ == "__main__":
    # Initialize system
    unitizer = AlephUnitizer()
    portal = TesseractPortal(unitizer)
    
    # Test signal processing
    test_signals = [
        "XRP price at $1.74 with whale compression",
        "BTC showing volume spike at $84,250",
        "ETH breaking resistance near $2,890",
        "XRP whale movement detected - 70M tokens",
        "Market sentiment shifting bullish across altcoins"
    ]
    
    print("=== ALEPH UNITIZER TEST SEQUENCE ===\n")
    
    for i, signal in enumerate(test_signals):
        short_tag, entropy, pattern = unitizer.unitize_message(
            signal, 
            asset=["XRPUSDC", "BTCUSDC", "ETHUSDC", "XRPUSDC", "MARKET"][i],
            trigger=f"test_trigger_{i}",
            confidence=0.5 + (i * 0.1)
        )
        
        coords = portal.map_to_tesseract_coordinates(pattern)
        
        print(f"Signal {i+1}: {signal[:40]}...")
        print(f"  Short Tag: {short_tag}")
        print(f"  Entropy: {entropy}")
        print(f"  Pattern: {pattern}")
        print(f"  Tesseract Magnitude: {coords['magnitude']:.2f}")
        print(f"  Phase: {coords['phase']:.2f}")
        print()
    
    # Display system summary
    summary = unitizer.get_contextual_summary()
    print("=== SYSTEM SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Export patterns
    export_path = unitizer.export_pattern_map()
    print(f"\nPatterns exported to: {export_path}") 