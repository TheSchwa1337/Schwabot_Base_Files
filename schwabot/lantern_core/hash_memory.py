"""
Hash Memory Database: Semantic Memory Lattice.
==============================================

Stores and indexes semantic correlations to build a memory lattice
of validated hash-meaning pairs. This creates the pre-built semantic
memory that enables faster and more accurate interpretations.

Memory → Pattern Recognition → Profitable Predictions.
"""

from __future__ import annotations
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    from .lantern_eye import HashBlock
from .semantic_interpreter import LanguagePattern, SemanticCategory


@dataclass
class SemanticCorrelation:
    """A correlation between hash patterns and semantic meanings."""

    hash_signature: str
    semantic_pattern: LanguagePattern
    validation_scores: List[float]
    profit_outcomes: List[float]
    correlation_coefficient: float
    usage_count: int
    last_validated: float
    created_at: float = field(default_factory=time.time)

    def to_dict(self: "SemanticCorrelation") -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "hash_signature": self.hash_signature,
            "semantic_pattern": self.semantic_pattern.to_dict(),
            "validation_scores": self.validation_scores,
            "profit_outcomes": self.profit_outcomes,
            "correlation_coefficient": self.correlation_coefficient,
            "usage_count": self.usage_count,
            "last_validated": self.last_validated,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticCorrelation:
        """Create from dictionary."""
        # Reconstruct LanguagePattern
        pattern_data = data["semantic_pattern"]
        semantic_pattern = LanguagePattern(
            primary_meaning=pattern_data["primary_meaning"],
            category=SemanticCategory(pattern_data["category"]),
            confidence_score=pattern_data["confidence_score"],
            contextual_insights=pattern_data["contextual_insights"],
            profit_potential=pattern_data["profit_potential"],
            risk_assessment=pattern_data["risk_assessment"],
            temporal_relevance=pattern_data["temporal_relevance"],
            entropy_source=pattern_data["entropy_source"],
            pattern_strength=pattern_data["pattern_strength"],
            harmonic_alignment=pattern_data["harmonic_alignment"],
            created_at=pattern_data["created_at"],
        )

        return cls(
            hash_signature=data["hash_signature"],
            semantic_pattern=semantic_pattern,
            validation_scores=data["validation_scores"],
            profit_outcomes=data["profit_outcomes"],
            correlation_coefficient=data["correlation_coefficient"],
            usage_count=data["usage_count"],
            last_validated=data["last_validated"],
            created_at=data["created_at"],
        )


class HashMemoryDatabase:
    """
    Database for storing and retrieving semantic hash correlations.

    Builds a memory lattice of validated hash-meaning pairs that
    strengthens the oracle's pattern recognition capabilities.
    """

    def __init__(
        self: "HashMemoryDatabase",
        max_records: int = 10000,
        persistence_file: Optional[str] = None,
    ) -> None:
        """Initialize the Hash Memory Database."""
        self.max_records = max_records
        self.persistence_file = persistence_file or "lantern_memory_database.json"

        # Core memory storage
        self.correlations: Dict[str, SemanticCorrelation] = {}
        # Hash prefix -> correlation IDs
        self.hash_index: Dict[str, List[str]] = {}
        self.category_index: Dict[
            SemanticCategory, List[str]
        ] = {}  # Category -> correlation IDs
        # Profit range -> correlation IDs
        self.profit_index: Dict[str, List[str]] = {}

        # Performance tracking
        self.total_stored = 0
        self.total_retrieved = 0
        self.cache_hits = 0
        self.correlation_strength_sum = 0.0
        self.created_at = time.time()

        # Load existing data
        self._load_from_file()

        # Initialize category indices
        for category in SemanticCategory:
            if category not in self.category_index:
                self.category_index[category] = []

    def _generate_hash_signature(
        self: "HashMemoryDatabase", hash_block: HashBlock
    ) -> str:
        """Generate a signature for hash pattern matching."""
        # Create signature from key hash characteristics
        signature_components = [
            hash_block.hash_value[:16],  # First 16 chars of hash
            (
                f"E{hash_block.entropy_block.entropy_score:.3f}"
                if hash_block.entropy_block
                else "E0.000"
            ),
            (
                f"S{hash_block.entropy_block.stability_index:.3f}"
                if hash_block.entropy_block
                else "S0.000"
            ),
            (
                f"D{hash_block.entropy_block.recursion_depth}"
                if hash_block.entropy_block
                else "D0"
            ),
        ]

        signature = "_".join(signature_components)
        return hashlib.sha256(signature.encode()).hexdigest()[:32]

    def _index_correlation(
        self: "HashMemoryDatabase",
        correlation_id: str,
        correlation: SemanticCorrelation,
    ) -> None:
        """Add correlation to various indices for fast retrieval."""
        # Hash prefix index
        hash_prefix = correlation.hash_signature[:8]
        if hash_prefix not in self.hash_index:
            self.hash_index[hash_prefix] = []
        self.hash_index[hash_prefix].append(correlation_id)

        # Category index
        category = correlation.semantic_pattern.category
        if category not in self.category_index:
            self.category_index[category] = []
        self.category_index[category].append(correlation_id)

        # Profit index
        profit_range = self._get_profit_range(
            correlation.semantic_pattern.profit_potential
        )
        if profit_range not in self.profit_index:
            self.profit_index[profit_range] = []
        self.profit_index[profit_range].append(correlation_id)

    def _get_profit_range(self: "HashMemoryDatabase", profit_potential: float) -> str:
        """Get profit range category for indexing."""
        if profit_potential >= 0.8:
            return "high"
        elif profit_potential >= 0.5:
            return "medium"
        elif profit_potential >= 0.2:
            return "low"
        else:
            return "minimal"

    def _calculate_correlation_coefficient(
        self: "HashMemoryDatabase",
        validation_scores: List[float],
        profit_outcomes: List[float],
    ) -> float:
        """Calculate correlation coefficient between validation and profit."""
        if len(validation_scores) < 2 or len(profit_outcomes) < 2:
            return 0.5  # Default neutral correlation

        # Ensure equal length
        min_length = min(len(validation_scores), len(profit_outcomes))
        scores = validation_scores[:min_length]
        outcomes = profit_outcomes[:min_length]

        # Calculate Pearson correlation coefficient
        if min_length > 1:
            correlation = np.corrcoef(scores, outcomes)[0, 1]
            # Normalize to [0,1]
            return max(0.0, min(1.0, (correlation + 1.0) / 2.0))
        else:
            return 0.5

    def store_hash_block(
        self: "HashMemoryDatabase", hash_block: HashBlock
    ) -> Optional[SemanticCorrelation]:
        """Store hash block and return correlation if pattern is recognized."""
        if not hash_block.semantic_interpretation:
            return None

        # Generate hash signature
        hash_signature = self._generate_hash_signature(hash_block)

        # Check if we already have this pattern
        existing_correlation = self._find_existing_correlation(
            hash_signature, hash_block.semantic_interpretation
        )

        if existing_correlation:
            # Update existing correlation
            existing_correlation.usage_count += 1
            existing_correlation.last_validated = time.time()

            # Add validation score if available
            if hash_block.truth_score:
                existing_correlation.validation_scores.append(
                    hash_block.truth_score.validation_score
                )
                existing_correlation.profit_outcomes.append(
                    hash_block.truth_score.profit_correlation
                )

                # Recalculate correlation coefficient
                existing_correlation.correlation_coefficient = (
                    self._calculate_correlation_coefficient(
                        existing_correlation.validation_scores,
                        existing_correlation.profit_outcomes,
                    )
                )

            self.cache_hits += 1
            self.total_retrieved += 1
            return existing_correlation

        else:
            # Create new correlation
            validation_scores = []
            profit_outcomes = []

            if hash_block.truth_score:
                validation_scores.append(hash_block.truth_score.validation_score)
                profit_outcomes.append(hash_block.truth_score.profit_correlation)

            correlation = SemanticCorrelation(
                hash_signature=hash_signature,
                semantic_pattern=hash_block.semantic_interpretation,
                validation_scores=validation_scores,
                profit_outcomes=profit_outcomes,
                correlation_coefficient=self._calculate_correlation_coefficient(
                    validation_scores, profit_outcomes
                ),
                usage_count=1,
                last_validated=time.time(),
            )

            # Store correlation
            correlation_id = f"{hash_signature}_{int(time.time())}"
            self.correlations[correlation_id] = correlation

            # Index for fast retrieval
            self._index_correlation(correlation_id, correlation)

            # Update statistics
            self.total_stored += 1
            self.correlation_strength_sum += correlation.correlation_coefficient

            # Cleanup if over limit
            self._cleanup_old_records()

            return correlation

    def _find_existing_correlation(
        self: "HashMemoryDatabase",
        hash_signature: str,
        semantic_pattern: LanguagePattern,
    ) -> Optional[SemanticCorrelation]:
        """Find existing correlation for similar patterns."""
        # Look for exact hash signature match first
        for correlation in self.correlations.values():
            if correlation.hash_signature == hash_signature:
                return correlation

        # Look for similar semantic patterns
        hash_prefix = hash_signature[:8]
        if hash_prefix in self.hash_index:
            for correlation_id in self.hash_index[hash_prefix]:
                if correlation_id in self.correlations:
                    correlation = self.correlations[correlation_id]

                    # Check semantic similarity
                    if (
                        correlation.semantic_pattern.category
                        == semantic_pattern.category
                        and abs(
                            correlation.semantic_pattern.confidence_score
                            - semantic_pattern.confidence_score
                        )
                        < 0.2
                    ):
                        return correlation

        return None

    def find_similar_patterns(
        self: "HashMemoryDatabase", hash_value: str, threshold: float = 0.7
    ) -> List[SemanticCorrelation]:
        """Find patterns similar to the given hash."""
        similar_patterns = []
        hash_prefix = hash_value[:8]

        # Search by hash prefix
        if hash_prefix in self.hash_index:
            for correlation_id in self.hash_index[hash_prefix]:
                if correlation_id in self.correlations:
                    correlation = self.correlations[correlation_id]

                    # Check correlation strength
                    if correlation.correlation_coefficient >= threshold:
                        similar_patterns.append(correlation)

        # Sort by correlation strength and usage
        similar_patterns.sort(
            key=lambda x: (
                x.correlation_coefficient * 0.7 + (x.usage_count / 100.0) * 0.3
            ),
            reverse=True,
        )

        return similar_patterns[:10]  # Return top 10 matches

    def get_patterns_by_category(
        self: "HashMemoryDatabase",
        category: SemanticCategory,
        min_correlation: float = 0.6,
    ) -> List[SemanticCorrelation]:
        """Get patterns by semantic category."""
        patterns = []

        if category in self.category_index:
            for correlation_id in self.category_index[category]:
                if correlation_id in self.correlations:
                    correlation = self.correlations[correlation_id]

                    if correlation.correlation_coefficient >= min_correlation:
                        patterns.append(correlation)

        # Sort by correlation strength
        patterns.sort(key=lambda x: x.correlation_coefficient, reverse=True)
        return patterns

    def get_high_profit_patterns(
        self: "HashMemoryDatabase", min_profit_potential: float = 0.7
    ) -> List[SemanticCorrelation]:
        """Get patterns with high profit potential."""
        high_profit_patterns = []

        for correlation in self.correlations.values():
            if correlation.semantic_pattern.profit_potential >= min_profit_potential:
                if (
                    correlation.correlation_coefficient > 0.6
                ):  # Must have good correlation
                    high_profit_patterns.append(correlation)

        # Sort by profit potential and correlation
        high_profit_patterns.sort(
            key=lambda x: (
                x.semantic_pattern.profit_potential * 0.6
                + x.correlation_coefficient * 0.4
            ),
            reverse=True,
        )

        return high_profit_patterns[:20]  # Return top 20

    def get_recent_correlations(
        self: "HashMemoryDatabase", count: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent correlations for truth scoring."""
        recent_correlations = []

        # Sort correlations by last_validated time
        sorted_correlations = sorted(
            self.correlations.values(), key=lambda x: x.last_validated, reverse=True
        )

        for correlation in sorted_correlations[:count]:
            recent_correlations.append(
                {
                    "hash_signature": correlation.hash_signature,
                    "category": correlation.semantic_pattern.category.value,
                    "confidence_score": correlation.semantic_pattern.confidence_score,
                    "profit_potential": correlation.semantic_pattern.profit_potential,
                    "correlation_coefficient": correlation.correlation_coefficient,
                    "validation_scores": correlation.validation_scores,
                    "profit_outcomes": correlation.profit_outcomes,
                    "usage_count": correlation.usage_count,
                }
            )

        return recent_correlations

    def _cleanup_old_records(self: "HashMemoryDatabase") -> None:
        """Remove old records when over storage limit."""
        if len(self.correlations) <= self.max_records:
            return

        # Sort by last used and correlation strength
        correlations_list = list(self.correlations.items())
        correlations_list.sort(
            key=lambda x: (x[1].last_validated + x[1].correlation_coefficient * 86400),
            # Boost recent and strong correlations
            reverse=True,
        )

        # Keep top records
        records_to_keep = correlations_list[: self.max_records]

        # Clear all data and rebuild
        self.correlations.clear()
        self.hash_index.clear()
        self.category_index.clear()
        self.profit_index.clear()

        # Rebuild with kept records
        for correlation_id, correlation in records_to_keep:
            self.correlations[correlation_id] = correlation
            self._index_correlation(correlation_id, correlation)

    def _save_to_file(self: "HashMemoryDatabase") -> None:
        """Save database to file."""
        try:
            data = {
                "correlations": {k: v.to_dict() for k, v in self.correlations.items()},
                "metadata": {
                    "total_stored": self.total_stored,
                    "total_retrieved": self.total_retrieved,
                    "cache_hits": self.cache_hits,
                    "correlation_strength_sum": self.correlation_strength_sum,
                    "created_at": self.created_at,
                    "last_saved": time.time(),
                },
            }

            with open(self.persistence_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error saving hash memory database: {e}")

    def _load_from_file(self: "HashMemoryDatabase") -> None:
        """Load database from file."""
        try:
            if Path(self.persistence_file).exists():
                with open(self.persistence_file, "r") as f:
                    data = json.load(f)

                # Load correlations
                for correlation_id, correlation_data in data.get(
                    "correlations", {}
                ).items():
                    correlation = SemanticCorrelation.from_dict(correlation_data)
                    self.correlations[correlation_id] = correlation
                    self._index_correlation(correlation_id, correlation)

                # Load metadata
                metadata = data.get("metadata", {})
                self.total_stored = metadata.get("total_stored", 0)
                self.total_retrieved = metadata.get("total_retrieved", 0)
                self.cache_hits = metadata.get("cache_hits", 0)
                self.correlation_strength_sum = metadata.get(
                    "correlation_strength_sum", 0.0
                )
                self.created_at = metadata.get("created_at", time.time())

                print(
                    f"Loaded {len(self.correlations)} correlations from memory database"
                )

        except Exception as e:
            print(f"Error loading hash memory database: {e}")

    def save_database(self: "HashMemoryDatabase") -> None:
        """Manually save database."""
        self._save_to_file()

    def get_database_statistics(self: "HashMemoryDatabase") -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        category_stats = {}
        for category in SemanticCategory:
            patterns = self.get_patterns_by_category(category, min_correlation=0.0)
            category_stats[category.value] = {
                "total_patterns": len(patterns),
                "high_correlation_patterns": len(
                    [p for p in patterns if p.correlation_coefficient > 0.7]
                ),
                "average_correlation": (
                    np.mean([p.correlation_coefficient for p in patterns])
                    if patterns
                    else 0.0
                ),
            }

        profit_ranges = {}
        for profit_range in ["minimal", "low", "medium", "high"]:
            if profit_range in self.profit_index:
                profit_ranges[profit_range] = len(self.profit_index[profit_range])
            else:
                profit_ranges[profit_range] = 0

        return {
            "total_records": len(self.correlations),
            "total_stored_lifetime": self.total_stored,
            "total_retrieved_lifetime": self.total_retrieved,
            "cache_hit_rate": (
                self.cache_hits / self.total_retrieved
                if self.total_retrieved > 0
                else 0.0
            ),
            "average_correlation_strength": (
                self.correlation_strength_sum / len(self.correlations)
                if len(self.correlations) > 0
                else 0.0
            ),
            "category_distribution": category_stats,
            "profit_range_distribution": profit_ranges,
            "correlation_patterns": len(self.hash_index),
            "database_age_hours": (time.time() - self.created_at) / 3600.0,
        }

    def __del__(self: "HashMemoryDatabase") -> None:
        """Save database on destruction."""
        try:
            self._save_to_file()
        except BaseException:
            pass
