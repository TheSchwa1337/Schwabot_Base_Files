"""
Story Parser
===========

Implements symbolic collapse function:
𝑪ₛ(t) = ⨁ 𝑤ᵢ ∈ S(t) :: [POS_tag, sentiment, vector_bias] → action

Converts narrative structures into actionable trading signals.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import yaml
import logging

logger = logging.getLogger(__name__)

class TradeAction(Enum):
    """Trading actions derived from narrative collapse"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    WAIT = "wait"
    ROTATE = "rotate"

@dataclass
class ParsedSignal:
    """Represents a parsed trading signal from a story"""
    action: TradeAction
    confidence: float
    vector_weights: Dict[str, float]
    entropy: float
    narrative_hash: str

class StoryParser:
    """
    Collapses narrative structures into trading actions using
    symbolic-mathematical mapping
    """
    
    def __init__(self, word_data_path: str = "profit_words.yaml"):
        """Initialize parser with word data"""
        self.word_data = self._load_word_data(word_data_path)
        self.pos_weights = self._initialize_pos_weights()
        self.vector_bias_map = self._initialize_vector_bias_map()
        
    def _load_word_data(self, path: str) -> Dict[str, Dict]:
        """Load word data with profit scores and attributes"""
        try:
            with open(path, 'r') as f:
                word_list = yaml.safe_load(f)
            return {w['word']: w for w in word_list}
        except Exception as e:
            logger.error(f"Error loading word data: {e}")
            return {}
    
    def _initialize_pos_weights(self) -> Dict[str, float]:
        """POS tags have different signal strengths"""
        return {
            'verb': 1.0,      # Action words = strong signal
            'noun': 0.7,      # Objects = medium signal
            'adjective': 0.5, # Modifiers = weak signal
            'adverb': 0.3,    # Qualifiers = very weak
        }
    
    def _initialize_vector_bias_map(self) -> Dict[str, TradeAction]:
        """Map vector biases to concrete actions"""
        return {
            'long': TradeAction.BUY,
            'short': TradeAction.SELL,
            'hold': TradeAction.HOLD,
            'warning': TradeAction.WAIT,
            'rotate': TradeAction.ROTATE,
            'neutral': TradeAction.HOLD
        }
    
    def parse(self, sentence: List[str]) -> ParsedSignal:
        """
        Parse a sentence into a trading signal
        
        Implements: 𝑪ₛ(t) = ⨁ 𝑤ᵢ ∈ S(t) :: [POS_tag, sentiment, vector_bias] → action
        """
        # Extract word attributes
        word_attrs = []
        for word in sentence:
            if word in self.word_data:
                attrs = self.word_data[word]
                word_attrs.append({
                    'word': word,
                    'pos': attrs['pos'],
                    'sentiment': attrs['sentiment'],
                    'vector_bias': attrs['vector_bias'],
                    'profit_score': attrs['profit_score'],
                    'entropy': attrs['entropy']
                })
        
        # Calculate vector composition
        vector_composition = self._compose_vectors(word_attrs)
        
        # Determine primary action
        action = self._determine_action(vector_composition)
        
        # Calculate confidence based on profit scores and POS weights
        confidence = self._calculate_confidence(word_attrs)
        
        # Calculate sentence entropy
        entropy = self._calculate_entropy(word_attrs)
        
        # Generate narrative hash for tracking
        narrative_hash = self._generate_narrative_hash(sentence)
        
        return ParsedSignal(
            action=action,
            confidence=confidence,
            vector_weights=vector_composition,
            entropy=entropy,
            narrative_hash=narrative_hash
        )
    
    def _compose_vectors(self, word_attrs: List[Dict]) -> Dict[str, float]:
        """
        Compose word vectors into aggregate signal
        Uses weighted averaging based on POS and profit scores
        """
        vector_counts = {
            'long': 0.0,
            'short': 0.0,
            'hold': 0.0,
            'warning': 0.0,
            'rotate': 0.0,
            'neutral': 0.0
        }
        
        total_weight = 0.0
        
        for attrs in word_attrs:
            pos_weight = self.pos_weights.get(attrs['pos'], 0.1)
            profit_weight = 1.0 + (attrs['profit_score'] / 100.0)  # Normalize profit influence
            
            weight = pos_weight * profit_weight
            vector_counts[attrs['vector_bias']] += weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for key in vector_counts:
                vector_counts[key] /= total_weight
                
        return vector_counts
    
    def _determine_action(self, vector_composition: Dict[str, float]) -> TradeAction:
        """
        Determine primary action from vector composition
        Uses threshold-based decision with tie-breaking
        """
        # Find dominant vector
        dominant_vector = max(vector_composition, key=vector_composition.get)
        dominant_weight = vector_composition[dominant_vector]
        
        # Check if dominant signal is strong enough
        if dominant_weight < 0.3:  # No clear signal
            return TradeAction.WAIT
            
        return self.vector_bias_map.get(dominant_vector, TradeAction.HOLD)
    
    def _calculate_confidence(self, word_attrs: List[Dict]) -> float:
        """
        Calculate confidence based on:
        - Average profit scores
        - POS composition
        - Sentiment alignment
        """
        if not word_attrs:
            return 0.0
            
        # Average profit score component
        avg_profit = np.mean([w['profit_score'] for w in word_attrs])
        profit_component = np.tanh(avg_profit / 10.0)  # Sigmoid-like normalization
        
        # POS strength component
        pos_strengths = [self.pos_weights.get(w['pos'], 0.1) for w in word_attrs]
        pos_component = np.mean(pos_strengths)
        
        # Sentiment alignment component
        sentiments = [w['sentiment'] for w in word_attrs]
        sentiment_counts = {}
        for s in sentiments:
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
        
        max_sentiment_ratio = max(sentiment_counts.values()) / len(sentiments)
        
        # Weighted confidence
        confidence = (
            0.5 * profit_component +
            0.3 * pos_component +
            0.2 * max_sentiment_ratio
        )
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_entropy(self, word_attrs: List[Dict]) -> float:
        """
        Calculate sentence entropy: 𝒟(S) = − ∑ P(𝑤ᵢ|S) · log(P(𝑤ᵢ|S))
        """
        if not word_attrs:
            return 0.0
            
        # Get entropy classes
        entropy_classes = [w['entropy'] for w in word_attrs]
        
        # Map to numeric values
        entropy_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        entropy_values = [entropy_map.get(e, 0.5) for e in entropy_classes]
        
        # Calculate Shannon entropy
        if len(entropy_values) > 1:
            # Normalize to probabilities
            total = sum(entropy_values)
            probs = [e/total for e in entropy_values]
            
            # Calculate entropy
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            return float(entropy)
        else:
            return 0.0
    
    def _generate_narrative_hash(self, sentence: List[str]) -> str:
        """Generate unique hash for narrative tracking"""
        import hashlib
        sentence_str = ' '.join(sentence)
        return hashlib.sha256(sentence_str.encode()).hexdigest()[:12] 