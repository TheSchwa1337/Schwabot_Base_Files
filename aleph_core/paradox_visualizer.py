"""
Paradox Visualizer for high-dimensional fractal state tracking.
Provides comprehensive pattern matching and clustering for TPF systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from sklearn.cluster import DBSCAN, AgglomerativeClustering

try:
    from sklearn.feature_extraction.text import CountVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    CountVectorizer = None
    DBSCAN = None

@dataclass
class ParadoxState:
    """Current state of the paradox visualization."""
    phase: int = 0
    stabilized: bool = False
    paradox_visible: bool = False
    trading_mode: bool = False
    glyph_state: str = "INITIALIZING"
    detonation_protocol: bool = False
    timestamp: float = 0.0

@dataclass
class MarketData:
    """Market data for visualization."""
    price: float = 0.0
    volume: float = 0.0
    rsi: float = 50.0
    drift: float = 0.0
    entropy: float = 0.5

class ParadoxVisualizer:
    """Core paradox visualization engine."""
    
    def __init__(self):
        self.state = ParadoxState()
        self.market_data = MarketData()
        self.trading_signals = []
        self.tpf_fractals = []
        self.last_update = datetime.now()
        
    def update_state(self, new_phase: int, market_data: Optional[Dict] = None) -> ParadoxState:
        """
        Update the paradox visualization state.
        """
        self.state.phase = new_phase % 100
        
        # Update market data if provided
        if market_data:
            self.market_data = MarketData(**market_data)
        
        # Paradox detection at phase 30
        if self.state.phase == 30 and not self.state.paradox_visible:
            self.state.paradox_visible = True
            self.state.glyph_state = "PARADOX DETECTED"
        
        # TPF stabilization at phase 70
        if self.state.phase == 70 and not self.state.stabilized:
            self.state.stabilized = True
            self.state.glyph_state = "TPF STABILIZED"
        
        # Reset cycle at phase 99
        if self.state.phase == 99:
            self.state.paradox_visible = False
            self.state.stabilized = False
            self.state.glyph_state = "INITIALIZING"
        
        self.state.timestamp = datetime.now().timestamp()
        return self.state
    
    def trigger_detonation(self) -> List[Dict]:
        """
        Trigger the detonation protocol and generate trading signals.
        """
        self.state.detonation_protocol = True
        self.state.trading_mode = True
        
        # Generate trading signals based on recursive paradox state
        signals = []
        for i in range(5):
            signal = {
                'id': int(datetime.now().timestamp() * 1000) + i,
                'type': 'SELL' if self.market_data.rsi > 70 else 'BUY' if self.market_data.rsi < 30 else 'HOLD',
                'confidence': 0.95 if self.state.stabilized else 0.4 if self.state.paradox_visible else 0.7,
                'price': self.market_data.price + (np.random.random() - 0.5) * 200,
                'timestamp': datetime.now().timestamp()
            }
            signals.append(signal)
        
        self.trading_signals = signals
        return signals
    
    def calculate_tpf_metrics(self) -> Dict:
        """
        Calculate TPF (Triangle Paradox Fractal) metrics.
        """
        return {
            'magnitude': np.sqrt(sum(x**2 for x in [self.market_data.price, self.market_data.volume, self.market_data.rsi])),
            'phase': np.arctan2(self.market_data.drift, self.market_data.entropy),
            'stability_score': 1.0 if self.state.stabilized else 0.5 if self.state.paradox_visible else 0.0,
            'paradox_intensity': 1.0 if self.state.paradox_visible else 0.0,
            'detonation_ready': self.state.detonation_protocol
        }
    
    def get_visualization_data(self) -> Dict:
        """
        Get complete visualization data for frontend rendering.
        """
        return {
            'state': {
                'phase': self.state.phase,
                'stabilized': self.state.stabilized,
                'paradox_visible': self.state.paradox_visible,
                'trading_mode': self.state.trading_mode,
                'glyph_state': self.state.glyph_state,
                'detonation_protocol': self.state.detonation_protocol
            },
            'market_data': {
                'price': self.market_data.price,
                'volume': self.market_data.volume,
                'rsi': self.market_data.rsi,
                'drift': self.market_data.drift,
                'entropy': self.market_data.entropy
            },
            'trading_signals': self.trading_signals,
            'tpf_metrics': self.calculate_tpf_metrics(),
            'timestamp': self.state.timestamp
        }

    def comprehensive_similarity(self, pattern1: List[int], pattern2: List[int]) -> Dict[str, float]:
        """
        Calculate comprehensive similarity metrics.
        
        Args:
            pattern1 (List[int]): The first pattern.
            pattern2 (List[int]): The second pattern.
            
        Returns:
            Dict[str, float]: A dictionary containing similarity scores.
        """
        # Check for valid input patterns
        if not pattern1 or not pattern2:
            raise ValueError("Both patterns must be non-empty lists.")
        
        if not all(isinstance(x, int) for x in pattern1 + pattern2):
            raise TypeError("Both patterns must contain only integers.")
        
        # Calculate similarity scores using various metrics
        hamming_dist = self.hamming_distance(pattern1, pattern2)
        manhattan_dist = self.manhattan_distance(pattern1, pattern2)
        euclidean_dist = self.euclidean_distance(pattern1, pattern2)
        cosine_sim = self.cosine_similarity(pattern1, pattern2)
        
        # Normalize distances to [0,1] similarity scores
        hamming_sim = 1.0 - (hamming_dist / len(pattern1))
        manhattan_sim = 1.0 - (manhattan_dist / (len(pattern1) * max(1, len(pattern1))))  # Max possible
        euclidean_sim = 1.0 - (euclidean_dist / np.sqrt(len(pattern1) * max(1, len(pattern1))**2))
        
        # Weighted composite similarity
        weights = [0.3, 0.25, 0.25, 0.2]  # Example weights
        if len(weights) != 4:
            raise ValueError("Weights list must contain exactly four elements.")
        
        composite_score = sum(weight * score for weight, score in zip(weights, [
            hamming_sim,
            manhattan_sim,
            euclidean_sim,
            cosine_sim
        ]))
        
        return {
            'hamming_similarity': hamming_sim,
            'manhattan_similarity': manhattan_sim,
            'euclidean_similarity': euclidean_sim,
            'cosine_similarity': cosine_sim,
            'composite_similarity': composite_score
        }

    def find_pattern_clusters(self, patterns: List[List[int]], threshold: float = 0.7) -> List[List[int]]:
        """
        Group similar patterns into clusters using similarity threshold.
        
        Args:
            patterns (List[List[int]]): A list of patterns to cluster.
            threshold (float): The similarity threshold for clustering.
            
        Returns:
            List[List[int]]: A list of clusters, where each cluster is a list of pattern indices.
        """
        if not SKLEARN_AVAILABLE:
            # Simple fallback clustering based on pattern similarity
            clusters = []
            used_indices = set()
            
            for i, pattern1 in enumerate(patterns):
                if i in used_indices:
                    continue
                    
                cluster = [i]
                used_indices.add(i)
                
                for j, pattern2 in enumerate(patterns[i+1:], i+1):
                    if j in used_indices:
                        continue
                        
                    similarity = self.comprehensive_similarity(pattern1, pattern2)
                    if similarity['composite_similarity'] >= threshold:
                        cluster.append(j)
                        used_indices.add(j)
                
                if len(cluster) > 1:  # Only include clusters with more than one pattern
                    clusters.append(cluster)
            
            return clusters
        
        # Convert patterns to string format for CountVectorizer
        pattern_strings = [' '.join(map(str, pattern)) for pattern in patterns]
        
        # Convert patterns to vectors
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(pattern_strings)
        
        # Use DBSCAN for clustering
        dbscan = DBSCAN(eps=1.0-threshold, min_samples=2)  # Convert similarity to distance
        labels = dbscan.fit_predict(X.toarray())
        
        clusters = []
        for label in set(labels):
            if label != -1:  # Ignore noise points
                cluster = [i for i, l in enumerate(labels) if l == label]
                clusters.append(cluster)
        
        return clusters

    def hamming_distance(self, pattern1: List[int], pattern2: List[int]) -> float:
        """Calculate Hamming distance between two patterns."""
        if len(pattern1) != len(pattern2):
            return float('inf')  # Infinite distance for different lengths
        return sum(1 for a, b in zip(pattern1, pattern2) if a != b)
    
    def manhattan_distance(self, pattern1: List[int], pattern2: List[int]) -> float:
        """Calculate Manhattan distance between two patterns."""
        if len(pattern1) != len(pattern2):
            return float('inf')
        return sum(abs(a - b) for a, b in zip(pattern1, pattern2))
    
    def euclidean_distance(self, pattern1: List[int], pattern2: List[int]) -> float:
        """Calculate Euclidean distance between two patterns."""
        if len(pattern1) != len(pattern2):
            return float('inf')
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pattern1, pattern2)))
    
    def cosine_similarity(self, pattern1: List[int], pattern2: List[int]) -> float:
        """Calculate Cosine similarity between two patterns."""
        if len(pattern1) != len(pattern2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(pattern1, pattern2))
        magnitude_a = np.sqrt(sum(a ** 2 for a in pattern1))
        magnitude_b = np.sqrt(sum(b ** 2 for b in pattern2))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)

def test_comprehensive_similarity():
    """Test comprehensive similarity function"""
    pattern_visualizer = ParadoxVisualizer()
    pattern1 = [1, 2, 3]
    pattern2 = [1, 2, 4]
    
    result = pattern_visualizer.comprehensive_similarity(pattern1, pattern2)
    assert 'hamming_similarity' in result
    assert 'manhattan_similarity' in result
    assert 'euclidean_similarity' in result
    assert 'cosine_similarity' in result
    assert 'composite_similarity' in result

def test_find_pattern_clusters():
    """Test pattern clustering function"""
    pattern_visualizer = ParadoxVisualizer()
    patterns = [
        [1, 2, 3],
        [1, 2, 4],
        [2, 3, 4],
        [5, 6, 7]
    ]
    
    clusters = pattern_visualizer.find_pattern_clusters(patterns)
    assert len(clusters) == 2
    assert any([i in cluster for i in range(0, 2)])
    assert any([i in cluster for i in range(2, 4)])

if __name__ == "__main__":
    # Run tests only when module is executed directly
    test_comprehensive_similarity()
    test_find_pattern_clusters()
    print("✅ ParadoxVisualizer tests completed successfully") 