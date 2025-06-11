"""
Lantern Profit Lexicon Engine
===========================

Implements the formal mathematical framework for profit lexicon evolution.
Core mathematical structures and dynamics from the formal specification.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
from datetime import datetime
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class PartOfSpeech(Enum):
    """Part of speech tags"""
    NOUN = "noun"
    VERB = "verb"
    ADJ = "adj"
    ADV = "adv"

class EntropyClass(Enum):
    """Entropy classification levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class VectorBias(Enum):
    """Vector bias directions"""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    WARNING = "warning"
    ROTATE = "rotate"

@dataclass
class WordState:
    """Word state vector W(wᵢ) = [φᵢ, pᵢ, sᵢ, εᵢ, vᵢ, ρᵢ, υᵢ]ᵀ"""
    word: str
    profit_fitness: float  # φᵢ
    pos: PartOfSpeech     # pᵢ
    sentiment: float      # sᵢ ∈ [-1, 1]
    entropy: EntropyClass # εᵢ
    vector_bias: VectorBias # vᵢ
    usage_count: int      # ρᵢ
    volatility_score: float # υᵢ

class LexiconEngine:
    """
    Core lexicon engine implementing the formal mathematical framework.
    Manages the lexicon space L and word state evolution.
    """
    
    def __init__(self, 
                 yaml_path: str = "profit_words.yaml",
                 decay_factor: float = 0.99,
                 learning_rate: float = 0.1,
                 profit_bias: float = 0.7,
                 temperature: float = 1.0):
        """
        Initialize the lexicon engine
        
        Args:
            decay_factor (α): Decay rate for unused words
            learning_rate (β): Learning rate for profit updates
            profit_bias (π): Bias towards profitable words
            temperature (γ): Temperature for neighborhood sampling
        """
        self.yaml_path = Path(yaml_path)
        self.alpha = decay_factor
        self.beta = learning_rate
        self.pi = profit_bias
        self.gamma = temperature
        
        # Load lexicon
        self.lexicon: Dict[str, WordState] = self._load_lexicon()
        self.word_indices = {word: idx for idx, word in enumerate(self.lexicon.keys())}
        
        # Initialize POS weights
        self.pos_weights = {
            PartOfSpeech.VERB: 1.0,
            PartOfSpeech.NOUN: 0.7,
            PartOfSpeech.ADJ: 0.5,
            PartOfSpeech.ADV: 0.3
        }
        
        # Track system state
        self.last_update = datetime.now()
        self.story_history: List[List[str]] = []
        self.cumulative_profit: float = 0.0
        
    def _load_lexicon(self) -> Dict[str, WordState]:
        """Load lexicon from YAML file"""
        try:
            if not self.yaml_path.exists():
                logger.warning(f"Lexicon file not found: {self.yaml_path}")
                return {}
                
            with open(self.yaml_path, 'r') as f:
                word_data = yaml.safe_load(f)
                
            return {
                w['word']: WordState(
                    word=w['word'],
                    profit_fitness=w.get('profit_score', 0.0),
                    pos=PartOfSpeech(w['pos']),
                    sentiment=w.get('sentiment', 0.0),
                    entropy=EntropyClass(w['entropy']),
                    vector_bias=VectorBias(w['vector_bias']),
                    usage_count=w.get('usage_count', 0),
                    volatility_score=w.get('volatility_score', 0.0)
                )
                for w in word_data
            }
        except Exception as e:
            logger.error(f"Error loading lexicon: {e}")
            return {}
            
    def _save_lexicon(self) -> None:
        """Save lexicon to YAML file"""
        try:
            self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
            word_data = [
                {
                    'word': word,
                    'profit_score': state.profit_fitness,
                    'pos': state.pos.value,
                    'sentiment': state.sentiment,
                    'entropy': state.entropy.value,
                    'vector_bias': state.vector_bias.value,
                    'usage_count': state.usage_count,
                    'volatility_score': state.volatility_score
                }
                for word, state in self.lexicon.items()
            ]
            
            with open(self.yaml_path, 'w') as f:
                yaml.dump(word_data, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving lexicon: {e}")
            
    def update_fitness(self, used_words: List[str], profit_delta: float) -> None:
        """
        Update word fitness based on trade outcome
        
        Implements: φᵢ(t+1) = αφᵢ(t) + β·P(t)·𝟙[wᵢ ∈ S(t)]
        """
        # Calculate time-based decay
        time_delta = (datetime.now() - self.last_update).total_seconds()
        decay_factor = self.alpha ** (time_delta / 3600)  # Hourly decay
        
        # Update fitness scores
        for word, state in self.lexicon.items():
            if word in used_words:
                # Reinforcement update
                state.profit_fitness += self.beta * profit_delta
                state.usage_count += 1
            else:
                # Decay update
                state.profit_fitness *= decay_factor
                
        # Update system state
        self.cumulative_profit += profit_delta
        self.last_update = datetime.now()
        self._save_lexicon()
        
    def generate_price_hash(self, price: float, timestamp: datetime) -> str:
        """
        Generate price hash: H(p,t) = SHA256(p||t)
        """
        price_str = f"{price:.2f}_{timestamp.isoformat()}"
        return hashlib.sha256(price_str.encode()).hexdigest()
        
    def hash_to_seed_index(self, price_hash: str) -> int:
        """
        Map hash to lexicon index: Ψ(h) = (∫₀⁶ h[i]·16^(5-i)) mod n
        """
        hash_int = int(price_hash[:6], 16)
        return hash_int % len(self.lexicon)
        
    def get_top_profit_words(self, k: int = 100) -> List[str]:
        """Get top k words by profit fitness"""
        sorted_words = sorted(
            self.lexicon.items(),
            key=lambda x: x[1].profit_fitness,
            reverse=True
        )
        return [word for word, _ in sorted_words[:k]]
        
    def generate_story(self, 
                      seed_index: int,
                      top_words: List[str],
                      length: int) -> List[str]:
        """
        Generate story: G(seed, Θ, ℓ) = [w₁, w₂, ..., wₗ]
        
        Implements word selection probability:
        P(wⱼ = w | w₁,...,wⱼ₋₁) = {
            π·Pₜₒₚ(w) + (1-π)·Pₗₒc(w), if w ∈ Θ
            (1-π)·Pₗₒc(w),              otherwise
        }
        """
        story = []
        current_index = seed_index
        word_list = list(self.lexicon.keys())
        
        # Neighborhood parameters
        neighborhood_size = 50
        
        for _ in range(length):
            if np.random.random() < self.pi and top_words:
                # Choose from top profit words
                word = np.random.choice(top_words)
            else:
                # Choose from neighborhood
                neighborhood_start = max(0, current_index - neighborhood_size // 2)
                neighborhood_end = min(len(word_list), 
                                     current_index + neighborhood_size // 2)
                
                neighborhood = word_list[neighborhood_start:neighborhood_end]
                
                # Calculate local probabilities
                weights = np.array([
                    np.exp(self.gamma * self.lexicon[w].profit_fitness)
                    for w in neighborhood
                ])
                weights = weights / weights.sum()
                
                word = np.random.choice(neighborhood, p=weights)
            
            story.append(word)
            
            # Update current index
            if word in self.word_indices:
                current_index = self.word_indices[word]
                
        self.story_history.append(story)
        return story
        
    def calculate_narrative_entropy(self, story: List[str]) -> float:
        """
        Calculate narrative entropy: H(S) = -∑ᵢ P(wᵢ|S)·log P(wᵢ|S)
        """
        if not story:
            return 0.0
            
        # Get word probabilities
        word_counts = {}
        for word in story:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Calculate probabilities
        total_words = len(story)
        probs = [count/total_words for count in word_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        return float(entropy)
        
    def estimate_story_profitability(self, story: List[str]) -> float:
        """
        Estimate story profitability: P̂(S) = θ₁·(1/|S|)∑ᵢφᵢ + θ₂·(1/(H(S)+ε)) + θ₃·V(S)
        """
        if not story:
            return 0.0
            
        # Parameters
        theta1, theta2, theta3 = 0.5, 0.3, 0.2
        epsilon = 0.1
        
        # Average fitness
        avg_fitness = np.mean([
            self.lexicon[word].profit_fitness
            for word in story
        ])
        
        # Entropy component
        entropy = self.calculate_narrative_entropy(story)
        entropy_component = 1.0 / (entropy + epsilon)
        
        # Volatility alignment
        volatility_score = np.mean([
            self.lexicon[word].volatility_score
            for word in story
        ])
        
        # Calculate estimate
        profitability = (
            theta1 * avg_fitness +
            theta2 * entropy_component +
            theta3 * volatility_score
        )
        
        return float(profitability)
        
    def get_system_state(self) -> Dict:
        """Get current system state vector Ξ(t) = [Φ(t), Σ(t), Π(t)]ᵀ"""
        return {
            'fitness_vector': [state.profit_fitness for state in self.lexicon.values()],
            'story_history': self.story_history,
            'cumulative_profit': self.cumulative_profit,
            'last_update': self.last_update.isoformat()
        }
        
    def get_lexicon_stats(self) -> Dict:
        """Get lexicon statistics"""
        if not self.lexicon:
            return {
                'total_words': 0,
                'avg_fitness': 0.0,
                'total_usage': 0,
                'pos_distribution': {},
                'entropy_distribution': {},
                'vector_bias_distribution': {}
            }
            
        # Calculate statistics
        total_words = len(self.lexicon)
        total_usage = sum(state.usage_count for state in self.lexicon.values())
        avg_fitness = np.mean([state.profit_fitness for state in self.lexicon.values()])
        
        # Distribution counts
        pos_dist = {}
        entropy_dist = {}
        bias_dist = {}
        
        for state in self.lexicon.values():
            pos_dist[state.pos.value] = pos_dist.get(state.pos.value, 0) + 1
            entropy_dist[state.entropy.value] = entropy_dist.get(state.entropy.value, 0) + 1
            bias_dist[state.vector_bias.value] = bias_dist.get(state.vector_bias.value, 0) + 1
            
        # Convert to percentages
        for dist in [pos_dist, entropy_dist, bias_dist]:
            total = sum(dist.values())
            for key in dist:
                dist[key] = dist[key] / total
                
        return {
            'total_words': total_words,
            'avg_fitness': float(avg_fitness),
            'total_usage': total_usage,
            'pos_distribution': pos_dist,
            'entropy_distribution': entropy_dist,
            'vector_bias_distribution': bias_dist
        } 