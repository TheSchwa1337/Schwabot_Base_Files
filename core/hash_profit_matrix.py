"""
Hash-to-Profit Matrix Engine
============================

Implements the core mathematical bridge between BTC hash signals and profit prediction.
This is the missing component that converts SHA-256 hash patterns into actionable
profit vectors using the Forever Fractal mathematical framework.

Mathematical Foundation:
H(t) = SHA256(price_t ⊕ time_t ⊕ vault_t ⊕ cycle_t)
P_expected = Σ w_i · ψ_i where ψ_i ∈ SymbolicField(ζ)

Key Components:
- Hash feature extraction (ΔH, ∇H, ζ, TCI)
- Temporal Fractal Similarity Index (TFSI)
- Quantum hash memory with decay
- Profit expectation calculation
- Pattern learning and reinforcement
"""

import numpy as np
import hashlib
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class HashFeatures:
    """Extracted features from BTC hash"""
    hash_echo: float  # ΔH - hash change signal
    hash_curl: float  # ∇H - gradient/momentum
    symbolic_projection: float  # ζ - symbolic field value
    triplet_collapse_index: float  # TCI - collapse probability
    raw_hash: str  # Original hash string
    timestamp: float

@dataclass
class ProfitPattern:
    """Historical profit pattern linked to hash features"""
    pattern_id: str
    hash_features: HashFeatures
    realized_profit: float
    confidence: float
    occurrence_count: int
    last_seen: float
    success_rate: float
    avg_hold_duration: float

@dataclass
class ProfitPrediction:
    """Profit prediction from hash analysis"""
    expected_profit: float
    confidence: float
    pattern_matches: List[str]
    hash_features: HashFeatures
    reasoning: str
    risk_assessment: Dict[str, Any]

class HashProfitMatrix:
    """
    Core engine that maps BTC hash patterns to profit expectations.
    
    Implements the mathematical bridge between hash-derived logic vectors
    and profit-centric trading decisions using fractal memory and symbolic
    field projections.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hash-to-profit mapping engine.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Hash processing parameters
        self.hash_memory_depth = self.config.get('hash_memory_depth', 50)
        self.decay_lambda = self.config.get('decay_lambda', 0.02)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.min_pattern_confidence = self.config.get('min_confidence', 0.6)
        
        # Symbolic field parameters
        self.symbolic_weights = self.config.get('symbolic_weights', {
            'rebirth': 1.5,
            'collapse': -0.8,
            'mirror': 0.3,
            'echo': 0.7,
            'void': -0.2
        })
        
        # Hash memory storage
        self.hash_history: deque = deque(maxlen=self.hash_memory_depth)
        self.profit_patterns: Dict[str, ProfitPattern] = {}
        
        # Quantum memory with temporal decay
        self.quantum_memory: deque = deque(maxlen=100)
        
        # Performance tracking
        self.prediction_accuracy = deque(maxlen=200)
        self.total_predictions = 0
        self.successful_predictions = 0
        
        logger.info("Hash-Profit Matrix initialized with symbolic field mapping")
    
    def generate_btc_hash(self, price: float, timestamp: float, 
                         vault_state: str, cycle_index: int) -> str:
        """
        Generate BTC hash from market state.
        
        Args:
            price: Current BTC price
            timestamp: Current timestamp
            vault_state: Current vault configuration
            cycle_index: Ferris wheel cycle position
            
        Returns:
            SHA-256 hash string
        """
        # Combine inputs for hash generation
        hash_input = f"{price:.8f}_{timestamp:.3f}_{vault_state}_{cycle_index}"
        
        # Generate SHA-256 hash
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        hash_string = hash_bytes.hex()
        
        return hash_string
    
    def extract_hash_features(self, current_hash: str, timestamp: float) -> HashFeatures:
        """
        Extract mathematical features from hash for profit analysis.
        
        Args:
            current_hash: Current BTC hash
            timestamp: Current timestamp
            
        Returns:
            HashFeatures object with extracted components
        """
        # Calculate hash echo (ΔH) - change from previous hash
        hash_echo = self._calculate_hash_echo(current_hash)
        
        # Calculate hash curl (∇H) - gradient/momentum
        hash_curl = self._calculate_hash_curl()
        
        # Calculate symbolic projection (ζ)
        symbolic_projection = self._calculate_symbolic_projection(current_hash)
        
        # Calculate triplet collapse index (TCI)
        triplet_collapse_index = self._calculate_triplet_collapse_index(current_hash)
        
        features = HashFeatures(
            hash_echo=hash_echo,
            hash_curl=hash_curl,
            symbolic_projection=symbolic_projection,
            triplet_collapse_index=triplet_collapse_index,
            raw_hash=current_hash,
            timestamp=timestamp
        )
        
        # Store in hash history
        self.hash_history.append(features)
        
        return features
    
    def predict_profit(self, hash_features: HashFeatures) -> ProfitPrediction:
        """
        Predict profit expectation from hash features.
        
        Args:
            hash_features: Extracted hash features
            
        Returns:
            ProfitPrediction with expected profit and confidence
        """
        # Find similar historical patterns
        pattern_matches = self._find_similar_patterns(hash_features)
        
        # Calculate quantum hash memory influence
        quantum_influence = self._calculate_quantum_memory_influence(hash_features)
        
        # Calculate expected profit using weighted pattern matching
        expected_profit = self._calculate_expected_profit(pattern_matches, quantum_influence)
        
        # Calculate prediction confidence
        confidence = self._calculate_prediction_confidence(pattern_matches, hash_features)
        
        # Generate reasoning
        reasoning = self._generate_prediction_reasoning(pattern_matches, hash_features)
        
        # Assess risk factors
        risk_assessment = self._assess_prediction_risk(hash_features, pattern_matches)
        
        prediction = ProfitPrediction(
            expected_profit=expected_profit,
            confidence=confidence,
            pattern_matches=[p.pattern_id for p in pattern_matches],
            hash_features=hash_features,
            reasoning=reasoning,
            risk_assessment=risk_assessment
        )
        
        self.total_predictions += 1
        
        return prediction
    
    def update_pattern_outcome(self, pattern_id: str, realized_profit: float, 
                              hold_duration: float):
        """
        Update pattern with realized outcome for learning.
        
        Args:
            pattern_id: Pattern identifier
            realized_profit: Actual profit achieved
            hold_duration: How long position was held
        """
        if pattern_id in self.profit_patterns:
            pattern = self.profit_patterns[pattern_id]
            
            # Update pattern statistics
            pattern.occurrence_count += 1
            
            # Update success rate
            if realized_profit > 0:
                pattern.success_rate = (
                    (pattern.success_rate * (pattern.occurrence_count - 1) + 1.0) /
                    pattern.occurrence_count
                )
            else:
                pattern.success_rate = (
                    (pattern.success_rate * (pattern.occurrence_count - 1)) /
                    pattern.occurrence_count
                )
            
            # Update average hold duration
            pattern.avg_hold_duration = (
                (pattern.avg_hold_duration * (pattern.occurrence_count - 1) + hold_duration) /
                pattern.occurrence_count
            )
            
            # Update confidence based on consistency
            profit_error = abs(realized_profit - pattern.realized_profit)
            consistency_factor = np.exp(-profit_error / 100.0)  # Normalize by 100bp
            pattern.confidence = 0.9 * pattern.confidence + 0.1 * consistency_factor
            
            pattern.last_seen = time.time()
            
            logger.info(f"Pattern {pattern_id} updated: profit={realized_profit:.1f}bp, "
                       f"success_rate={pattern.success_rate:.3f}")
    
    def _calculate_hash_echo(self, current_hash: str) -> float:
        """Calculate hash echo (ΔH) - XOR difference from previous hash."""
        if not self.hash_history:
            return 0.0
            
        prev_hash = self.hash_history[-1].raw_hash
        
        # Convert hashes to bytes and XOR
        current_bytes = bytes.fromhex(current_hash)
        prev_bytes = bytes.fromhex(prev_hash)
        
        xor_result = bytes(a ^ b for a, b in zip(current_bytes, prev_bytes))
        
        # Convert to normalized float
        hash_echo = sum(xor_result) / (255.0 * len(xor_result))
        
        return hash_echo
    
    def _calculate_hash_curl(self) -> float:
        """Calculate hash curl (∇H) - gradient over recent history."""
        if len(self.hash_history) < 3:
            return 0.0
            
        # Get recent hash echo values
        recent_echoes = [h.hash_echo for h in list(self.hash_history)[-5:]]
        
        # Calculate gradient (simple finite difference)
        if len(recent_echoes) >= 2:
            gradient = np.gradient(recent_echoes)
            hash_curl = np.mean(gradient)
        else:
            hash_curl = 0.0
            
        return hash_curl
    
    def _calculate_symbolic_projection(self, hash_string: str) -> float:
        """Calculate symbolic projection (ζ) from hash."""
        # Extract symbolic patterns from hash
        hash_int = int(hash_string[:16], 16)  # Use first 16 chars
        
        # Map to symbolic field values
        symbolic_patterns = {
            'rebirth': (hash_int % 1000) / 1000.0,
            'collapse': ((hash_int >> 8) % 1000) / 1000.0,
            'mirror': ((hash_int >> 16) % 1000) / 1000.0,
            'echo': ((hash_int >> 24) % 1000) / 1000.0,
            'void': ((hash_int >> 32) % 1000) / 1000.0
        }
        
        # Calculate weighted symbolic projection
        symbolic_projection = sum(
            self.symbolic_weights.get(pattern, 0.0) * value
            for pattern, value in symbolic_patterns.items()
        )
        
        # Normalize to [-1, 1] range
        symbolic_projection = np.tanh(symbolic_projection)
        
        return symbolic_projection
    
    def _calculate_triplet_collapse_index(self, hash_string: str) -> float:
        """Calculate triplet collapse index (TCI)."""
        if len(self.hash_history) < 2:
            return 0.0
            
        # Get last 3 hashes (including current)
        hash_triplet = [h.raw_hash for h in list(self.hash_history)[-2:]] + [hash_string]
        
        # Convert to integers for calculation
        hash_ints = [int(h[:16], 16) for h in hash_triplet]
        
        # Calculate TCI = |H1·H2 - H3| mod P
        if len(hash_ints) >= 3:
            h1, h2, h3 = hash_ints[-3:]
            tci_raw = abs(h1 * h2 - h3) % 1000000  # Use 1M as modulus
            tci = tci_raw / 1000000.0  # Normalize to [0, 1]
        else:
            tci = 0.0
            
        return tci
    
    def _find_similar_patterns(self, hash_features: HashFeatures) -> List[ProfitPattern]:
        """Find historically similar patterns using TFSI."""
        similar_patterns = []
        
        for pattern in self.profit_patterns.values():
            # Calculate Temporal Fractal Similarity Index (TFSI)
            tfsi = self._calculate_tfsi(hash_features, pattern.hash_features)
            
            if tfsi >= self.similarity_threshold:
                similar_patterns.append(pattern)
        
        # Sort by similarity and confidence
        similar_patterns.sort(
            key=lambda p: p.confidence * self._calculate_tfsi(hash_features, p.hash_features),
            reverse=True
        )
        
        return similar_patterns[:5]  # Return top 5 matches
    
    def _calculate_tfsi(self, features1: HashFeatures, features2: HashFeatures) -> float:
        """Calculate Temporal Fractal Similarity Index between hash features."""
        # Create feature vectors
        vec1 = np.array([
            features1.hash_echo,
            features1.hash_curl,
            features1.symbolic_projection,
            features1.triplet_collapse_index
        ])
        
        vec2 = np.array([
            features2.hash_echo,
            features2.hash_curl,
            features2.symbolic_projection,
            features2.triplet_collapse_index
        ])
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
            
        tfsi = dot_product / norms
        
        # Apply temporal decay
        time_diff = abs(features1.timestamp - features2.timestamp)
        temporal_decay = np.exp(-self.decay_lambda * time_diff / 3600.0)  # Decay over hours
        
        return tfsi * temporal_decay
    
    def _calculate_quantum_memory_influence(self, hash_features: HashFeatures) -> float:
        """Calculate quantum hash memory influence."""
        if not self.quantum_memory:
            return 0.0
            
        # Calculate weighted influence from quantum memory
        total_influence = 0.0
        total_weight = 0.0
        
        current_time = time.time()
        
        for memory_entry in self.quantum_memory:
            # Calculate temporal decay
            time_diff = current_time - memory_entry['timestamp']
            decay_weight = np.exp(-self.decay_lambda * time_diff)
            
            # Calculate feature similarity
            similarity = self._calculate_tfsi(hash_features, memory_entry['features'])
            
            # Weight influence by similarity and decay
            influence = memory_entry['profit'] * similarity * decay_weight
            total_influence += influence
            total_weight += decay_weight
            
        if total_weight > 0:
            return total_influence / total_weight
        else:
            return 0.0
    
    def _calculate_expected_profit(self, pattern_matches: List[ProfitPattern], 
                                 quantum_influence: float) -> float:
        """Calculate expected profit from pattern matches and quantum memory."""
        if not pattern_matches:
            return quantum_influence
            
        # Weight patterns by confidence and recency
        weighted_profit = 0.0
        total_weight = 0.0
        
        for pattern in pattern_matches:
            # Calculate pattern weight
            recency_factor = np.exp(-(time.time() - pattern.last_seen) / 3600.0)
            weight = pattern.confidence * pattern.success_rate * recency_factor
            
            weighted_profit += pattern.realized_profit * weight
            total_weight += weight
        
        if total_weight > 0:
            pattern_profit = weighted_profit / total_weight
        else:
            pattern_profit = 0.0
            
        # Combine with quantum influence
        expected_profit = 0.7 * pattern_profit + 0.3 * quantum_influence
        
        return expected_profit
    
    def _calculate_prediction_confidence(self, pattern_matches: List[ProfitPattern],
                                       hash_features: HashFeatures) -> float:
        """Calculate confidence in profit prediction."""
        if not pattern_matches:
            return 0.1  # Low confidence without patterns
            
        # Base confidence from pattern quality
        avg_confidence = np.mean([p.confidence for p in pattern_matches])
        avg_success_rate = np.mean([p.success_rate for p in pattern_matches])
        
        # Pattern consistency factor
        if len(pattern_matches) > 1:
            profits = [p.realized_profit for p in pattern_matches]
            consistency = 1.0 - (np.std(profits) / (np.mean(np.abs(profits)) + 1e-6))
            consistency = np.clip(consistency, 0.0, 1.0)
        else:
            consistency = 0.5
        
        # Symbolic strength factor
        symbolic_strength = abs(hash_features.symbolic_projection)
        
        # Combined confidence
        confidence = (
            0.4 * avg_confidence +
            0.3 * avg_success_rate +
            0.2 * consistency +
            0.1 * symbolic_strength
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _generate_prediction_reasoning(self, pattern_matches: List[ProfitPattern],
                                     hash_features: HashFeatures) -> str:
        """Generate human-readable reasoning for prediction."""
        if not pattern_matches:
            return f"No historical patterns found. Symbolic projection: {hash_features.symbolic_projection:.3f}"
            
        reasoning_parts = []
        
        # Pattern match information
        reasoning_parts.append(f"Found {len(pattern_matches)} similar patterns")
        
        if pattern_matches:
            best_pattern = pattern_matches[0]
            reasoning_parts.append(
                f"Best match: {best_pattern.success_rate:.1%} success rate, "
                f"{best_pattern.confidence:.3f} confidence"
            )
        
        # Symbolic analysis
        if abs(hash_features.symbolic_projection) > 0.5:
            if hash_features.symbolic_projection > 0:
                reasoning_parts.append("Strong positive symbolic signal")
            else:
                reasoning_parts.append("Strong negative symbolic signal")
        
        # Triplet collapse analysis
        if hash_features.triplet_collapse_index > 0.7:
            reasoning_parts.append("High triplet collapse probability")
        
        return "; ".join(reasoning_parts)
    
    def _assess_prediction_risk(self, hash_features: HashFeatures,
                              pattern_matches: List[ProfitPattern]) -> Dict[str, Any]:
        """Assess risk factors for prediction."""
        risk_factors = {}
        
        # Pattern diversity risk
        if len(pattern_matches) < 2:
            risk_factors['pattern_diversity'] = 'high'
        elif len(pattern_matches) >= 5:
            risk_factors['pattern_diversity'] = 'low'
        else:
            risk_factors['pattern_diversity'] = 'moderate'
        
        # Symbolic volatility risk
        if abs(hash_features.symbolic_projection) > 0.8:
            risk_factors['symbolic_volatility'] = 'high'
        else:
            risk_factors['symbolic_volatility'] = 'low'
        
        # Hash momentum risk
        if abs(hash_features.hash_curl) > 0.5:
            risk_factors['hash_momentum'] = 'high'
        else:
            risk_factors['hash_momentum'] = 'low'
        
        # Overall risk assessment
        high_risk_count = sum(1 for risk in risk_factors.values() if risk == 'high')
        if high_risk_count >= 2:
            risk_factors['overall_risk'] = 'high'
        elif high_risk_count == 1:
            risk_factors['overall_risk'] = 'moderate'
        else:
            risk_factors['overall_risk'] = 'low'
        
        return risk_factors
    
    def create_pattern_from_outcome(self, hash_features: HashFeatures, 
                                  realized_profit: float, hold_duration: float) -> str:
        """Create new profit pattern from trading outcome."""
        # Generate pattern ID
        pattern_id = hashlib.md5(
            f"{hash_features.raw_hash}_{hash_features.timestamp}".encode()
        ).hexdigest()[:12]
        
        # Create pattern
        pattern = ProfitPattern(
            pattern_id=pattern_id,
            hash_features=hash_features,
            realized_profit=realized_profit,
            confidence=0.5,  # Initial confidence
            occurrence_count=1,
            last_seen=time.time(),
            success_rate=1.0 if realized_profit > 0 else 0.0,
            avg_hold_duration=hold_duration
        )
        
        # Store pattern
        self.profit_patterns[pattern_id] = pattern
        
        # Add to quantum memory
        self.quantum_memory.append({
            'timestamp': time.time(),
            'features': hash_features,
            'profit': realized_profit,
            'pattern_id': pattern_id
        })
        
        logger.info(f"New profit pattern created: {pattern_id} "
                   f"(profit: {realized_profit:.1f}bp)")
        
        return pattern_id
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        if self.total_predictions == 0:
            return {"status": "no_predictions"}
            
        return {
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "accuracy_rate": self.successful_predictions / self.total_predictions,
            "total_patterns": len(self.profit_patterns),
            "quantum_memory_size": len(self.quantum_memory),
            "hash_history_size": len(self.hash_history),
            "avg_pattern_confidence": np.mean([p.confidence for p in self.profit_patterns.values()]) if self.profit_patterns else 0.0,
            "avg_pattern_success_rate": np.mean([p.success_rate for p in self.profit_patterns.values()]) if self.profit_patterns else 0.0,
            "recent_accuracy": np.mean(list(self.prediction_accuracy)[-20:]) if len(self.prediction_accuracy) >= 20 else 0.0
        }
    
    def export_patterns(self, filename: str):
        """Export profit patterns to JSON file."""
        export_data = {}
        
        for pattern_id, pattern in self.profit_patterns.items():
            export_data[pattern_id] = {
                "realized_profit": pattern.realized_profit,
                "confidence": pattern.confidence,
                "occurrence_count": pattern.occurrence_count,
                "success_rate": pattern.success_rate,
                "avg_hold_duration": pattern.avg_hold_duration,
                "hash_features": {
                    "hash_echo": pattern.hash_features.hash_echo,
                    "hash_curl": pattern.hash_features.hash_curl,
                    "symbolic_projection": pattern.hash_features.symbolic_projection,
                    "triplet_collapse_index": pattern.hash_features.triplet_collapse_index
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported {len(export_data)} patterns to {filename}")

# Example usage and testing
if __name__ == "__main__":
    # Test hash-profit matrix
    matrix = HashProfitMatrix()
    
    # Generate test hash
    test_hash = matrix.generate_btc_hash(
        price=45000.0,
        timestamp=time.time(),
        vault_state="BTC_USDC",
        cycle_index=7
    )
    
    # Extract features
    features = matrix.extract_hash_features(test_hash, time.time())
    
    print(f"Hash: {test_hash[:16]}...")
    print(f"Features: echo={features.hash_echo:.3f}, curl={features.hash_curl:.3f}")
    print(f"Symbolic: {features.symbolic_projection:.3f}, TCI={features.triplet_collapse_index:.3f}")
    
    # Predict profit
    prediction = matrix.predict_profit(features)
    
    print(f"Predicted profit: {prediction.expected_profit:.1f}bp")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Reasoning: {prediction.reasoning}")
    
    # Simulate outcome and create pattern
    simulated_profit = 75.0  # 75 basis points
    pattern_id = matrix.create_pattern_from_outcome(features, simulated_profit, 25.0)
    
    print(f"Created pattern: {pattern_id}")
    
    # Get system summary
    summary = matrix.get_system_summary()
    print(f"System summary: {summary}") 