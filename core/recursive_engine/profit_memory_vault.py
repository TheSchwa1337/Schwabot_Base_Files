from __future__ import annotations
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class PatternMemory:
    """Stores a profitable trading pattern."""
    pattern_id: str
    entry_price: float
    exit_price: float
    profit_pct: float
    psi_sequence: List[float]
    entropy_sequence: List[float]
    coherence_sequence: List[float]
    duration: int  # in ticks
    timestamp: datetime
    metadata: Dict[str, Any]

class ProfitMemoryVault:
    """
    Stores and recalls profitable trading patterns based on recursive signals.
    Uses pattern matching to identify potential re-entry opportunities.
    """
    
    def __init__(
        self,
        max_patterns: int = 1000,
        min_profit_threshold: float = 0.02,  # 2% minimum profit
        pattern_similarity_threshold: float = 0.85,
        max_pattern_age: int = 7  # days
    ):
        self.max_patterns = max_patterns
        self.min_profit_threshold = min_profit_threshold
        self.pattern_similarity_threshold = pattern_similarity_threshold
        self.max_pattern_age = max_pattern_age
        
        # Pattern storage
        self.patterns: List[PatternMemory] = []
        self.pattern_counter = 0
        
        logger.info("ProfitMemoryVault initialized with %d max patterns", max_patterns)

    def _calculate_pattern_similarity(
        self,
        current_psi: List[float],
        current_entropy: List[float],
        stored_psi: List[float],
        stored_entropy: List[float]
    ) -> float:
        """Calculate similarity between current and stored patterns."""
        if len(current_psi) != len(stored_psi) or len(current_entropy) != len(stored_entropy):
            return 0.0
            
        # Normalize sequences
        def normalize(seq):
            return (seq - np.mean(seq)) / (np.std(seq) + 1e-8)
            
        current_psi_norm = normalize(np.array(current_psi))
        stored_psi_norm = normalize(np.array(stored_psi))
        current_entropy_norm = normalize(np.array(current_entropy))
        stored_entropy_norm = normalize(np.array(stored_entropy))
        
        # Calculate correlation coefficients
        psi_corr = np.corrcoef(current_psi_norm, stored_psi_norm)[0, 1]
        entropy_corr = np.corrcoef(current_entropy_norm, stored_entropy_norm)[0, 1]
        
        # Combine correlations (equal weight)
        return (psi_corr + entropy_corr) / 2

    def store_pattern(
        self,
        entry_price: float,
        exit_price: float,
        psi_sequence: List[float],
        entropy_sequence: List[float],
        coherence_sequence: List[float],
        duration: int,
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Store a profitable trading pattern if it meets criteria.
        
        Parameters
        ----------
        entry_price : float
            Price at pattern entry
        exit_price : float
            Price at pattern exit
        psi_sequence : List[float]
            Sequence of Ψ(t) values during pattern
        entropy_sequence : List[float]
            Sequence of entropy values
        coherence_sequence : List[float]
            Sequence of coherence values
        duration : int
            Pattern duration in ticks
        metadata : dict
            Additional pattern metadata
            
        Returns
        -------
        Optional[str]
            Pattern ID if stored, None if rejected
        """
        # Calculate profit percentage
        profit_pct = (exit_price - entry_price) / entry_price
        
        # Check if pattern meets minimum profit threshold
        if profit_pct < self.min_profit_threshold:
            return None
            
        # Generate pattern ID
        pattern_id = f"PAT_{self.pattern_counter:06d}"
        self.pattern_counter += 1
        
        # Create pattern memory
        pattern = PatternMemory(
            pattern_id=pattern_id,
            entry_price=entry_price,
            exit_price=exit_price,
            profit_pct=profit_pct,
            psi_sequence=psi_sequence,
            entropy_sequence=entropy_sequence,
            coherence_sequence=coherence_sequence,
            duration=duration,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Add to patterns list
        self.patterns.append(pattern)
        
        # Maintain max patterns limit
        if len(self.patterns) > self.max_patterns:
            # Remove oldest pattern
            self.patterns.sort(key=lambda p: p.timestamp)
            self.patterns.pop(0)
            
        logger.info("Stored pattern %s with %.2f%% profit", pattern_id, profit_pct * 100)
        return pattern_id

    def find_similar_patterns(
        self,
        current_psi: List[float],
        current_entropy: List[float],
        max_results: int = 5
    ) -> List[PatternMemory]:
        """
        Find patterns similar to current market conditions.
        
        Parameters
        ----------
        current_psi : List[float]
            Current Ψ(t) sequence
        current_entropy : List[float]
            Current entropy sequence
        max_results : int
            Maximum number of similar patterns to return
            
        Returns
        -------
        List[PatternMemory]
            List of similar patterns, sorted by similarity
        """
        # Clean old patterns
        cutoff_time = datetime.now() - timedelta(days=self.max_pattern_age)
        self.patterns = [p for p in self.patterns if p.timestamp > cutoff_time]
        
        if not self.patterns:
            return []
            
        # Calculate similarities
        similarities = []
        for pattern in self.patterns:
            similarity = self._calculate_pattern_similarity(
                current_psi,
                current_entropy,
                pattern.psi_sequence,
                pattern.entropy_sequence
            )
            similarities.append((pattern, similarity))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches above threshold
        return [
            pattern for pattern, similarity in similarities[:max_results]
            if similarity >= self.pattern_similarity_threshold
        ]

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about stored patterns."""
        if not self.patterns:
            return {
                'total_patterns': 0,
                'avg_profit': 0.0,
                'max_profit': 0.0,
                'min_profit': 0.0
            }
            
        profits = [p.profit_pct for p in self.patterns]
        return {
            'total_patterns': len(self.patterns),
            'avg_profit': np.mean(profits),
            'max_profit': max(profits),
            'min_profit': min(profits),
            'pattern_age_days': (datetime.now() - min(p.timestamp for p in self.patterns)).days
        } 