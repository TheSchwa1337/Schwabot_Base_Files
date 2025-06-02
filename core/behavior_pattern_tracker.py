"""
Behavior Pattern Tracker
=======================

Implements pattern tracking and analysis for the Forever Fractal system.
Handles temporal frequency decay and n-gram behavior stitching.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import hashlib
from collections import defaultdict
from .spectral_state import SpectralState
import time
import sqlite3
from smart_money_analyzer import SmartMoneyAnalyzer

class BehaviorPatternTracker:
    """Tracks and analyzes behavior patterns with temporal decay"""
    
    def __init__(self):
        self.patterns = {}
        self.decay_rate = 0.1
        self.pattern_frequencies = defaultdict(float)
        self.pattern_timestamps = defaultdict(float)
        self.ngram_sequences = defaultdict(list)
        self.pattern_hashes = set()
        self.last_update = datetime.now().timestamp()
        
    def track_pattern(self, action):
        if action in self.patterns:
            self.patterns[action] += 1
        else:
            self.patterns[action] = 1

    def get_most_frequent_patterns(self, n=5):
        return sorted(self.patterns.items(), key=lambda x: x[1], reverse=True)[:n]

    def set_decay_rate(self, rate):
        self.decay_rate = rate

    def decay_patterns(self):
        for action in list(self.patterns.keys()):
            if self.patterns[action] > 0:
                self.patterns[action] -= self.decay_rate
                if self.patterns[action] <= 0:
                    del self.patterns[action]

    def track_pattern(self, 
                     triplet: str,
                     context: str,
                     action: str,
                     fractal_depth: int,
                     timestamp: Optional[float] = None) -> str:
        """
        Track a new behavior pattern
        
        Args:
            triplet: Triplet sequence
            context: Context information
            action: Action taken
            fractal_depth: Current fractal depth
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            Pattern hash
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        # Calculate pattern hash
        pattern_data = f"{triplet}|{context}|{action}|{fractal_depth}"
        pattern_hash = hashlib.md5(pattern_data.encode()).hexdigest()
        
        # Update pattern frequency with decay
        self._apply_temporal_decay(timestamp)
        self.pattern_frequencies[pattern_hash] += 1
        self.pattern_timestamps[pattern_hash] = timestamp
        
        # Track n-gram sequence
        self.ngram_sequences[pattern_hash].append({
            'triplet': triplet,
            'context': context,
            'action': action,
            'timestamp': timestamp
        })
        
        # Store hash
        self.pattern_hashes.add(pattern_hash)
        
        return pattern_hash
    
    def _apply_temporal_decay(self, current_time: float) -> None:
        """
        Apply temporal decay to all pattern frequencies
        
        Args:
            current_time: Current timestamp
        """
        dt = current_time - self.last_update
        decay_factor = np.exp(-self.decay_rate * dt)
        
        for pattern_hash in self.pattern_frequencies:
            self.pattern_frequencies[pattern_hash] *= decay_factor
            
        self.last_update = current_time
    
    def get_pattern_frequency(self, pattern_hash: str) -> float:
        """
        Get current frequency of a pattern
        
        Args:
            pattern_hash: Pattern hash to look up
            
        Returns:
            Current frequency with decay applied
        """
        self._apply_temporal_decay(datetime.now().timestamp())
        return self.pattern_frequencies.get(pattern_hash, 0.0)
    
    def stitch_ngram_sequence(self, 
                            pattern_hash: str,
                            n: int = 3) -> List[Dict]:
        """
        Get n-gram sequence for a pattern
        
        Args:
            pattern_hash: Pattern hash to look up
            n: Length of n-gram sequence
            
        Returns:
            List of n-gram sequences
        """
        sequence = self.ngram_sequences.get(pattern_hash, [])
        if len(sequence) < n:
            return []
            
        return [sequence[i:i+n] for i in range(len(sequence)-n+1)]
    
    def calculate_pattern_similarity(self, 
                                  hash1: str,
                                  hash2: str) -> float:
        """
        Calculate similarity between two patterns
        
        Args:
            hash1: First pattern hash
            hash2: Second pattern hash
            
        Returns:
            Similarity score [0,1]
        """
        seq1 = self.ngram_sequences.get(hash1, [])
        seq2 = self.ngram_sequences.get(hash2, [])
        
        if not seq1 or not seq2:
            return 0.0
            
        # Calculate Jaccard similarity of n-grams
        ngrams1 = set(tuple(s['triplet'] for s in seq1[i:i+3])
                     for i in range(len(seq1)-2))
        ngrams2 = set(tuple(s['triplet'] for s in seq2[i:i+3])
                     for i in range(len(seq2)-2))
        
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_most_frequent_patterns(self, 
                                 min_frequency: float = 0.1,
                                 max_patterns: int = 10) -> List[Tuple[str, float]]:
        """
        Get most frequent patterns
        
        Args:
            min_frequency: Minimum frequency threshold
            max_patterns: Maximum number of patterns to return
            
        Returns:
            List of (pattern_hash, frequency) tuples
        """
        self._apply_temporal_decay(datetime.now().timestamp())
        
        # Filter and sort patterns
        patterns = [(h, f) for h, f in self.pattern_frequencies.items()
                   if f >= min_frequency]
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        return patterns[:max_patterns]
    
    def update_spectral_state(self, state: SpectralState) -> None:
        """
        Update spectral state with pattern information
        
        Args:
            state: Spectral state to update
        """
        # Get pattern frequency
        frequency = self.get_pattern_frequency(state.pattern_hash)
        
        # Update recursive awareness based on pattern frequency
        state.recursive_awareness = min(1.0, frequency * 2)
        
        # Update memory weight
        state.memory_weight = np.exp(-self.decay_rate * 
                                   (datetime.now().timestamp() - state.timestamp))
    
    def save_state(self, filepath: str) -> None:
        """
        Save tracker state to file
        
        Args:
            filepath: Path to save state
        """
        state = {
            'decay_rate': self.decay_rate,
            'pattern_frequencies': dict(self.pattern_frequencies),
            'pattern_timestamps': dict(self.pattern_timestamps),
            'ngram_sequences': dict(self.ngram_sequences),
            'pattern_hashes': list(self.pattern_hashes),
            'last_update': self.last_update
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, filepath: str) -> 'BehaviorPatternTracker':
        """
        Load tracker state from file
        
        Args:
            filepath: Path to load state from
            
        Returns:
            Loaded BehaviorPatternTracker instance
        """
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        tracker = cls()
        tracker.pattern_frequencies = defaultdict(float, state['pattern_frequencies'])
        tracker.pattern_timestamps = defaultdict(float, state['pattern_timestamps'])
        tracker.ngram_sequences = defaultdict(list, state['ngram_sequences'])
        tracker.pattern_hashes = set(state['pattern_hashes'])
        tracker.last_update = state['last_update']
        
        return tracker 

# Example usage
tracker = BehaviorPatternTracker()

def track_action(triplet, context, action, fractal_depth):
    pattern_hash = tracker.track_pattern(triplet, context, action, fractal_depth)
    print(f"Tracked pattern: {pattern_hash}")

# Simulate event listeners
track_action("triplet1", "context1", "action1", 5)
time.sleep(60)  # Wait for a minute to simulate decay
track_action("triplet2", "context2", "action2", 3)

print(tracker.get_most_frequent_patterns(min_frequency=1)) 

class SpectralState:
    def __init__(self):
        self.spectral_data = {}

    def process_spectral_data(self, data):
        # Example processing logic
        for key, value in data.items():
            if key not in self.spectral_data:
                self.spectral_data[key] = 0
            self.spectral_data[key] += value

    def transition_state(self, spectral_data):
        # Example state transition logic
        self.process_spectral_data(spectral_data)
        # Implement state transitions based on spectral data 

class DeploymentHook:
    def setup_environment(self):
        # Set environment variables
        import os
        os.environ['DB_HOST'] = 'localhost'
        os.environ['DB_PORT'] = '5432'

    def deploy_code(self):
        # Deploy code to a server
        print("Deploying code to server...")

    def configure_services(self):
        # Configure services like database and logging
        self.setup_environment()
        self.deploy_code() 

class DatabaseHook:
    def __init__(self, db_name='tracker.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def create_table(self):
        # Create a table to store patterns
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                action TEXT PRIMARY KEY,
                count INTEGER
            )
        ''')
        self.conn.commit()

    def save_pattern(self, action):
        # Save pattern to the database
        self.cursor.execute('INSERT OR REPLACE INTO patterns (action, count) VALUES (?, 1)', (action,))
        self.conn.commit()

    def retrieve_patterns(self):
        # Retrieve all patterns from the database
        self.cursor.execute('SELECT * FROM patterns')
        return self.cursor.fetchall() 

def main():
    # Initialize both systems
    sma = SmartMoneyAnalyzer()
    bpt = BehaviorPatternTracker()

    # Example data processing
    base_number = sma.read_base_number('cyclicNumbers.txt')
    decimal_expansion = sma.calculate_decimal_expansion(base_number)
    period_length, missing_sequence = sma.find_period_length(decimal_expansion)

    print(f"Base number: {base_number}")
    print(f"Decimal value: {decimal_expansion}")
    print(f"Period length: {period_length}")
    print(f"Missing sequence: {missing_sequence}")

    # Example tracking behavior patterns
    triplet = "triplet1"
    context = "context1"
    action = "action1"
    fractal_depth = 5

    pattern_hash = bpt.track_pattern(triplet, context, action, fractal_depth)
    print(f"Tracked pattern: {pattern_hash}")

    # Example decay management
    current_time = datetime.now().timestamp()
    bpt._apply_temporal_decay(current_time)

if __name__ == "__main__":
    main()