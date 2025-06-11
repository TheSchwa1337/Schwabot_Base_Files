"""
Word Fitness Tracker
==================

Manages word profit scores and usage tracking.
Implements reinforcement learning for word fitness evolution.
"""

import yaml
from pathlib import Path
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class WordFitnessTracker:
    """Tracks and updates word fitness based on trade outcomes"""
    
    def __init__(self, yaml_path: str = "profit_words.yaml", decay_factor: float = 0.99):
        """Initialize the fitness tracker"""
        self.yaml_path = Path(yaml_path)
        self.decay = decay_factor
        self.words = self._load_words()
        self.last_update = datetime.now()
        
    def _load_words(self) -> List[Dict]:
        """Load word data from YAML"""
        try:
            if not self.yaml_path.exists():
                logger.warning(f"Word data file not found: {self.yaml_path}")
                return []
                
            with open(self.yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading word data: {e}")
            return []
            
    def _save_words(self) -> None:
        """Save word data to YAML"""
        try:
            self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.yaml_path, 'w') as f:
                yaml.dump(self.words, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving word data: {e}")
            
    def update_profit_scores(self, used_words: List[str], delta_profit: float) -> None:
        """
        Update profit scores for words based on trade outcome
        
        Implements: ∀𝑤ᵢ ∈ S(t): Φ(𝑤ᵢ) ← Φ(𝑤ᵢ) + η · ℘(t)
        """
        # Calculate time-based decay
        time_delta = (datetime.now() - self.last_update).total_seconds()
        decay_factor = self.decay ** (time_delta / 3600)  # Hourly decay
        
        # Update scores
        for word_obj in self.words:
            if word_obj["word"] in used_words:
                # Reinforcement update
                word_obj["profit_score"] += delta_profit
                word_obj["usage_count"] += 1
            else:
                # Decay update
                word_obj["profit_score"] *= decay_factor
                
        # Save updates
        self._save_words()
        self.last_update = datetime.now()
        
    def get_top_words(self, top_n: int = 10) -> List[Dict]:
        """Get top N words by profit score"""
        sorted_words = sorted(
            self.words,
            key=lambda x: x["profit_score"],
            reverse=True
        )
        return sorted_words[:top_n]
        
    def get_word_stats(self, word: str) -> Optional[Dict]:
        """Get statistics for a specific word"""
        for word_obj in self.words:
            if word_obj["word"] == word:
                return {
                    "profit_score": word_obj["profit_score"],
                    "usage_count": word_obj["usage_count"],
                    "pos": word_obj["pos"],
                    "sentiment": word_obj["sentiment"],
                    "vector_bias": word_obj["vector_bias"],
                    "entropy": word_obj["entropy"]
                }
        return None
        
    def get_lexicon_stats(self) -> Dict:
        """Get overall lexicon statistics"""
        if not self.words:
            return {
                "total_words": 0,
                "avg_profit_score": 0.0,
                "total_usage": 0,
                "pos_distribution": {},
                "sentiment_distribution": {},
                "vector_bias_distribution": {}
            }
            
        # Calculate statistics
        total_words = len(self.words)
        total_usage = sum(w["usage_count"] for w in self.words)
        avg_profit = sum(w["profit_score"] for w in self.words) / total_words
        
        # Distribution counts
        pos_dist = {}
        sentiment_dist = {}
        bias_dist = {}
        
        for word in self.words:
            pos = word["pos"]
            sentiment = word["sentiment"]
            bias = word["vector_bias"]
            
            pos_dist[pos] = pos_dist.get(pos, 0) + 1
            sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1
            bias_dist[bias] = bias_dist.get(bias, 0) + 1
            
        # Convert to percentages
        for dist in [pos_dist, sentiment_dist, bias_dist]:
            total = sum(dist.values())
            for key in dist:
                dist[key] = dist[key] / total
                
        return {
            "total_words": total_words,
            "avg_profit_score": avg_profit,
            "total_usage": total_usage,
            "pos_distribution": pos_dist,
            "sentiment_distribution": sentiment_dist,
            "vector_bias_distribution": bias_dist
        }
        
    def reset_all_scores(self) -> None:
        """Reset all profit scores and usage counts"""
        for word in self.words:
            word["profit_score"] = 0.0
            word["usage_count"] = 0
        self._save_words()
        
    def add_new_word(self, word: str, pos: str, sentiment: str, 
                    vector_bias: str, entropy: str) -> None:
        """Add a new word to the lexicon"""
        # Check if word exists
        for word_obj in self.words:
            if word_obj["word"] == word:
                logger.warning(f"Word already exists: {word}")
                return
                
        # Add new word
        self.words.append({
            "word": word,
            "pos": pos,
            "sentiment": sentiment,
            "vector_bias": vector_bias,
            "entropy": entropy,
            "profit_score": 0.0,
            "usage_count": 0
        })
        
        self._save_words()
        
    def remove_word(self, word: str) -> bool:
        """Remove a word from the lexicon"""
        initial_length = len(self.words)
        self.words = [w for w in self.words if w["word"] != word]
        
        if len(self.words) < initial_length:
            self._save_words()
            return True
        return False 