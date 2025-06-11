"""
Profit Story Engine
=================

Complete pipeline: hash → story → action → profit → word evolution

Implements:
- S(t) = G(Ψ(ℋ(t)), Θ, n)
- Φ(𝑤ᵢ, t) = α · Φ(𝑤ᵢ, t−1) + β · ℘(t) · 𝟙[𝑤ᵢ ∈ S(t)]
- 𝒫̂(S) = λ₁ · avg(Φ(𝑤ᵢ ∈ S)) + λ₂ · (1 / 𝒟(S)) + λ₃ · V(𝑆)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import hashlib
import yaml
from datetime import datetime
import logging
from .story_parser import StoryParser, TradeAction
from .word_fitness_tracker import WordFitnessTracker

logger = logging.getLogger(__name__)

class ProfitStoryEngine:
    """
    Complete pipeline: hash → story → action → profit → word evolution
    """
    
    def __init__(self, 
                 word_data_path: str = "profit_words.yaml",
                 story_length_range: Tuple[int, int] = (5, 12),
                 top_k_words: int = 100):
        """Initialize the profit story engine"""
        self.word_data_path = word_data_path
        self.story_length_range = story_length_range
        self.top_k_words = top_k_words
        
        # Initialize components
        self.fitness_tracker = WordFitnessTracker(word_data_path)
        self.story_parser = StoryParser(word_data_path)
        
        # Load full word list
        self.word_list = self._load_word_list()
        self.word_indices = {w['word']: i for i, w in enumerate(self.word_list)}
        
        # Story history for analysis
        self.story_history = []
        
    def _load_word_list(self) -> List[Dict]:
        """Load complete word list"""
        try:
            with open(self.word_data_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading word list: {e}")
            return []
    
    def generate_story_from_price(self, 
                                 btc_price: float, 
                                 timestamp: Optional[datetime] = None) -> Tuple[List[str], str]:
        """
        Generate story from BTC price hash
        
        Implements: S(t) = G(Ψ(ℋ(t)), Θ, n)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Generate price hash
        price_hash = self._generate_price_hash(btc_price, timestamp)
        
        # Get seed index: Ψ(ℋ(t))
        seed_index = self._hash_to_seed_index(price_hash)
        
        # Get top profit words
        top_words = self._get_top_profit_words()
        
        # Generate story
        story_length = np.random.randint(*self.story_length_range)
        story = self._generate_story(seed_index, top_words, story_length)
        
        return story, price_hash
    
    def _generate_price_hash(self, price: float, timestamp: datetime) -> str:
        """Generate SHA-256 hash from price and timestamp"""
        price_str = f"{price:.2f}_{timestamp.isoformat()}"
        return hashlib.sha256(price_str.encode()).hexdigest()
    
    def _hash_to_seed_index(self, price_hash: str) -> int:
        """
        Implements: Ψ(ℋ(t)) = (int(ℋ(t)[:6], 16)) mod |ℒ|
        """
        hash_int = int(price_hash[:6], 16)
        return hash_int % len(self.word_list)
    
    def _get_top_profit_words(self) -> List[str]:
        """Get top K words by profit score"""
        sorted_words = sorted(
            self.word_list, 
            key=lambda w: w['profit_score'], 
            reverse=True
        )
        return [w['word'] for w in sorted_words[:self.top_k_words]]
    
    def _generate_story(self, 
                       seed_index: int, 
                       top_words: List[str], 
                       length: int) -> List[str]:
        """
        Generate story starting from seed, biased toward profitable words
        """
        story = []
        current_index = seed_index
        
        # Neighborhood exploration parameters
        neighborhood_size = 50
        profit_bias = 0.7  # Probability of choosing from top words
        
        for _ in range(length):
            if np.random.random() < profit_bias and top_words:
                # Choose from top profit words
                word = np.random.choice(top_words)
            else:
                # Choose from neighborhood of current index
                neighborhood_start = max(0, current_index - neighborhood_size // 2)
                neighborhood_end = min(len(self.word_list), 
                                     current_index + neighborhood_size // 2)
                
                neighborhood_words = self.word_list[neighborhood_start:neighborhood_end]
                
                # Weight by profit score
                weights = np.array([w['profit_score'] + 1.0 for w in neighborhood_words])
                weights = weights / weights.sum()
                
                chosen_word = np.random.choice(neighborhood_words, p=weights)
                word = chosen_word['word']
            
            story.append(word)
            
            # Update current index for locality
            if word in self.word_indices:
                current_index = self.word_indices[word]
                
        return story
    
    def parse_story_to_action(self, story: List[str]) -> TradeAction:
        """Parse generated story into trading action"""
        signal = self.story_parser.parse(story)
        
        # Log story with signal
        self.story_history.append({
            'story': story,
            'signal': signal,
            'timestamp': datetime.now()
        })
        
        return signal.action
    
    def update_word_fitness(self, story: List[str], profit_delta: float):
        """
        Update word fitness based on trade outcome
        
        Implements reinforcement update:
        ∀𝑤ᵢ ∈ S(t): Φ(𝑤ᵢ) ← Φ(𝑤ᵢ) + η · ℘(t)
        """
        self.fitness_tracker.update_profit_scores(story, profit_delta)
        
        # Reload word list to get updated scores
        self.word_list = self._load_word_list()
    
    def get_story_profitability_estimate(self, story: List[str]) -> float:
        """
        Estimate story profitability: 𝒫̂(S)
        
        𝒫̂(S) = λ₁ · avg(Φ(𝑤ᵢ ∈ S)) + λ₂ · (1 / 𝒟(S)) + λ₃ · V(S)
        """
        # Lambda weights
        λ1, λ2, λ3 = 0.5, 0.3, 0.2
        
        # Average word fitness
        word_scores = []
        for word in story:
            if word in self.word_indices:
                word_data = self.word_list[self.word_indices[word]]
                word_scores.append(word_data['profit_score'])
        
        avg_fitness = np.mean(word_scores) if word_scores else 0.0
        
        # Parse for entropy
        signal = self.story_parser.parse(story)
        entropy_inverse = 1.0 / (signal.entropy + 0.1)  # Avoid division by zero
        
        # Volatility alignment (simplified)
        volatility_score = signal.confidence  # Use confidence as proxy
        
        # Calculate estimate
        profitability = (
            λ1 * avg_fitness +
            λ2 * entropy_inverse +
            λ3 * volatility_score
        )
        
        return float(profitability)
    
    def generate_batch_stories(self, 
                             price_series: List[float], 
                             n_stories_per_price: int = 5) -> List[Dict]:
        """Generate multiple stories for price series"""
        all_stories = []
        
        for price in price_series:
            for _ in range(n_stories_per_price):
                story, price_hash = self.generate_story_from_price(price)
                signal = self.story_parser.parse(story)
                profitability = self.get_story_profitability_estimate(story)
                
                all_stories.append({
                    'price': price,
                    'hash': price_hash,
                    'story': ' '.join(story),
                    'words': story,
                    'action': signal.action.value,
                    'confidence': signal.confidence,
                    'entropy': signal.entropy,
                    'estimated_profit': profitability
                })
                
        return all_stories 