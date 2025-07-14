"""Module for Schwabot trading system."""

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# !/usr/bin/env python3
"""
🧠⚛️ FRACTAL MEMORY TRACKER
===========================

Tracks and matches recurring market patterns using cosine similarity.
Enables Schwabot to recognize when similar market conditions reappear
and reuse successful trading strategies from the past.

    Features:
    - Cosine similarity matching for qutrit matrices
    - Pattern recognition across time cycles
    - Fractal replay triggering
    - Memory snapshot management
    """

    logger = logging.getLogger(__name__)


        class FractalMatchType(Enum):
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """Types of fractal matches"""

        EXACT = "exact"  # Perfect match
        HIGH_SIMILARITY = "high"  # Very similar (>0.95)
        MEDIUM_SIMILARITY = "medium"  # Moderately similar (>0.8)
        LOW_SIMILARITY = "low"  # Slightly similar (>0.6)
        NO_MATCH = "none"  # No significant similarity


        @dataclass
            class FractalSnapshot:
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Snapshot of a qutrit matrix at a specific time"""

            q_matrix: np.ndarray
            timestamp: float
            strategy_id: str
            profit_result: Optional[float]
            market_context: Dict[str, Any]
            similarity_score: Optional[float] = None


            @dataclass
                class FractalMatch:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Result of a fractal pattern match"""

                match_type: FractalMatchType
                similarity_score: float
                matched_snapshot: FractalSnapshot
                current_snapshot: FractalSnapshot
                confidence: float
                replay_recommended: bool


                    class FractalMemoryTracker:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """
                    Fractal Memory Tracker

                    Tracks qutrit matrix snapshots and detects recurring patterns
                    using cosine similarity matching.
                    """

                        def __init__(self, max_snapshots: int = 1000, similarity_threshold: float = 0.8) -> None:
                        """
                        Initialize fractal memory tracker

                            Args:
                            max_snapshots: Maximum number of snapshots to store
                            similarity_threshold: Minimum similarity for pattern matching
                            """
                            self.snapshot_stack: List[FractalSnapshot] = []
                            self.max_snapshots = max_snapshots
                            self.similarity_threshold = similarity_threshold
                            self.match_history: List[FractalMatch] = []

                            logger.info(
                            "Fractal Memory Tracker initialized (max: {0}, threshold: {1})".format(max_snapshots, similarity_threshold)
                            )

                            def save_snapshot(
                            self,
                            q_matrix: np.ndarray,
                            strategy_id: str,
                            profit_result: Optional[float] = None,
                            market_context: Optional[Dict[str, Any]] = None,
                                ) -> str:
                                """
                                Save a snapshot of the current qutrit matrix

                                    Args:
                                    q_matrix: Current 3x3 qutrit matrix
                                    strategy_id: Strategy identifier
                                    profit_result: Optional profit result from this snapshot
                                    market_context: Optional market context data

                                        Returns:
                                        Snapshot ID
                                        """
                                            try:
                                            snapshot = FractalSnapshot(
                                            q_matrix=q_matrix.copy(),
                                            timestamp=time.time(),
                                            strategy_id=strategy_id,
                                            profit_result=profit_result,
                                            market_context=market_context or {},
                                            )

                                            # Add to snapshot stack
                                            self.snapshot_stack.append(snapshot)

                                            # Maintain max size
                                                if len(self.snapshot_stack) > self.max_snapshots:
                                                self.snapshot_stack.pop(0)  # Remove oldest

                                                snapshot_id = "snapshot_{0}_{1}".format(len(self.snapshot_stack), int(snapshot.timestamp))
                                                logger.debug("Saved fractal snapshot: {0}".format(snapshot_id))

                                            return snapshot_id

                                                except Exception as e:
                                                logger.error("Error saving fractal snapshot: {0}".format(e))
                                            return ""

                                            def match_fractal(
                                            self,
                                            current_matrix: np.ndarray,
                                            strategy_id: str,
                                            market_context: Optional[Dict[str, Any]] = None,
                                            min_similarity: Optional[float] = None,
                                                ) -> Optional[FractalMatch]:
                                                """
                                                Check if current matrix matches any previous fractal patterns

                                                    Args:
                                                    current_matrix: Current 3x3 qutrit matrix
                                                    strategy_id: Current strategy identifier
                                                    market_context: Current market context
                                                    min_similarity: Minimum similarity threshold (overrides, default)

                                                        Returns:
                                                        FractalMatch if found, None otherwise
                                                        """
                                                            try:
                                                                if not self.snapshot_stack:
                                                            return None

                                                            threshold = min_similarity or self.similarity_threshold
                                                            current_flat = current_matrix.flatten()

                                                            best_match = None
                                                            best_score = 0.0

                                                            # Search through recent snapshots (most recent, first)
                                                                for snapshot in reversed(self.snapshot_stack):
                                                                stored_flat = snapshot.q_matrix.flatten()
                                                                similarity = self._cosine_similarity(current_flat, stored_flat)

                                                                    if similarity > best_score and similarity >= threshold:
                                                                    best_score = similarity
                                                                    best_match = snapshot

                                                                        if best_match:
                                                                        # Create current snapshot for comparison
                                                                        current_snapshot = FractalSnapshot(
                                                                        q_matrix=current_matrix.copy(),
                                                                        timestamp=time.time(),
                                                                        strategy_id=strategy_id,
                                                                        profit_result=None,
                                                                        market_context=market_context or {},
                                                                        )

                                                                        # Determine match type
                                                                        match_type = self._determine_match_type(best_score)

                                                                        # Calculate confidence based on similarity and profit history
                                                                        confidence = self._calculate_match_confidence(best_match, best_score)

                                                                        # Determine if replay is recommended
                                                                        replay_recommended = self._should_replay_pattern(best_match, confidence)

                                                                        fractal_match = FractalMatch(
                                                                        match_type=match_type,
                                                                        similarity_score=best_score,
                                                                        matched_snapshot=best_match,
                                                                        current_snapshot=current_snapshot,
                                                                        confidence=confidence,
                                                                        replay_recommended=replay_recommended,
                                                                        )

                                                                        # Store in match history
                                                                        self.match_history.append(fractal_match)

                                                                        logger.info(
                                                                        "Fractal match found: {0} (similarity: {1}, confidence: {2})".format(
                                                                        match_type.value, best_score, confidence
                                                                        )
                                                                        )
                                                                    return fractal_match

                                                                return None

                                                                    except Exception as e:
                                                                    logger.error("Error matching fractal: {0}".format(e))
                                                                return None

                                                                    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
                                                                    """Calculate cosine similarity between two vectors"""
                                                                        try:
                                                                        dot_product = np.dot(a, b)
                                                                        norm_a = np.linalg.norm(a)
                                                                        norm_b = np.linalg.norm(b)

                                                                            if norm_a == 0 or norm_b == 0:
                                                                        return 0.0

                                                                    return dot_product / (norm_a * norm_b)
                                                                        except Exception:
                                                                    return 0.0

                                                                        def _determine_match_type(self, similarity_score: float) -> FractalMatchType:
                                                                        """Determine the type of fractal match based on similarity score"""
                                                                            if similarity_score >= 0.99:
                                                                        return FractalMatchType.EXACT
                                                                            elif similarity_score >= 0.95:
                                                                        return FractalMatchType.HIGH_SIMILARITY
                                                                            elif similarity_score >= 0.8:
                                                                        return FractalMatchType.MEDIUM_SIMILARITY
                                                                            elif similarity_score >= 0.6:
                                                                        return FractalMatchType.LOW_SIMILARITY
                                                                            else:
                                                                        return FractalMatchType.NO_MATCH

                                                                            def _calculate_match_confidence(self, matched_snapshot: FractalSnapshot, similarity_score: float) -> float:
                                                                            """Calculate confidence score for a fractal match"""
                                                                                try:
                                                                                base_confidence = similarity_score

                                                                                # Boost confidence if the matched pattern was profitable
                                                                                profit_boost = 0.0
                                                                                    if matched_snapshot.profit_result is not None:
                                                                                        if matched_snapshot.profit_result > 0:
                                                                                        profit_boost = min(0.2, matched_snapshot.profit_result * 0.1)
                                                                                            else:
                                                                                            profit_boost = max(-0.1, matched_snapshot.profit_result * 0.5)

                                                                                            # Time decay factor (older patterns get lower confidence)
                                                                                            time_decay = max(0.5, 1.0 - (time.time() - matched_snapshot.timestamp) / 86400)  # 24 hour decay

                                                                                            final_confidence = (base_confidence + profit_boost) * time_decay
                                                                                        return max(0.0, min(1.0, final_confidence))

                                                                                            except Exception:
                                                                                        return similarity_score

                                                                                            def _should_replay_pattern(self, matched_snapshot: FractalSnapshot, confidence: float) -> bool:
                                                                                            """Determine if a pattern should be replayed based on confidence and history"""
                                                                                                try:
                                                                                                # High confidence patterns are more likely to be replayed
                                                                                                    if confidence < 0.6:
                                                                                                return False

                                                                                                # Profitable patterns are more likely to be replayed
                                                                                                    if matched_snapshot.profit_result is not None:
                                                                                                    if matched_snapshot.profit_result > 0.5:  # 5% profit threshold
                                                                                                return True
                                                                                                elif matched_snapshot.profit_result < -0.1:  # 10% loss threshold
                                                                                            return False

                                                                                            # Recent patterns are more likely to be replayed
                                                                                            age_hours = (time.time() - matched_snapshot.timestamp) / 3600
                                                                                            if age_hours < 24:  # Within 24 hours
                                                                                        return confidence > 0.7
                                                                                        elif age_hours < 168:  # Within 1 week
                                                                                    return confidence > 0.8
                                                                                        else:
                                                                                    return confidence > 0.9  # Very high confidence for old patterns

                                                                                        except Exception:
                                                                                    return confidence > 0.8

                                                                                        def find_recent_patterns(self, hours_back: int = 24) -> List[FractalSnapshot]:
                                                                                        """Find patterns from the last N hours"""
                                                                                            try:
                                                                                            cutoff_time = time.time() - (hours_back * 3600)
                                                                                            recent_patterns = []
                                                                                                for snapshot in self.snapshot_stack:
                                                                                                    if snapshot.timestamp >= cutoff_time:
                                                                                                    recent_patterns.append(snapshot)

                                                                                                    logger.debug("Found {0} patterns from last {1} hours".format(len(recent_patterns), hours_back))
                                                                                                return recent_patterns

                                                                                                    except Exception as e:
                                                                                                    logger.error("Error finding recent patterns: {0}".format(e))
                                                                                                return []

                                                                                                    def get_pattern_statistics(self) -> Dict[str, Any]:
                                                                                                    """Get statistics about stored patterns"""
                                                                                                        try:
                                                                                                            if not self.snapshot_stack:
                                                                                                        return {"total_patterns": 0}

                                                                                                        total_patterns = len(self.snapshot_stack)
                                                                                                        profitable_patterns = sum(1 for s in self.snapshot_stack if s.profit_result and s.profit_result > 0)
                                                                                                        loss_patterns = sum(1 for s in self.snapshot_stack if s.profit_result and s.profit_result < 0)

                                                                                                        avg_profit = np.mean([s.profit_result for s in self.snapshot_stack if s.profit_result is not None]) or 0.0
                                                                                                        avg_similarity = np.mean([m.similarity_score for m in self.match_history]) if self.match_history else 0.0

                                                                                                    return {
                                                                                                    "total_patterns": total_patterns,
                                                                                                    "profitable_patterns": profitable_patterns,
                                                                                                    "loss_patterns": loss_patterns,
                                                                                                    "avg_profit": avg_profit,
                                                                                                    "avg_similarity": avg_similarity,
                                                                                                    "match_history_size": len(self.match_history),
                                                                                                    "memory_utilization": total_patterns / self.max_snapshots,
                                                                                                    }

                                                                                                        except Exception as e:
                                                                                                        logger.error("Error getting pattern statistics: {0}".format(e))
                                                                                                    return {}

                                                                                                        def cleanup_old_patterns(self, max_age_hours: int = 168) -> int:
                                                                                                        """Clean up patterns older than specified age"""
                                                                                                            try:
                                                                                                            cutoff_time = time.time() - (max_age_hours * 3600)
                                                                                                            initial_count = len(self.snapshot_stack)

                                                                                                            # Remove old patterns
                                                                                                            self.snapshot_stack = [snapshot for snapshot in self.snapshot_stack if snapshot.timestamp >= cutoff_time]

                                                                                                            removed_count = initial_count - len(self.snapshot_stack)
                                                                                                            logger.info("Cleaned up {0} old patterns".format(removed_count))

                                                                                                        return removed_count

                                                                                                            except Exception as e:
                                                                                                            logger.error("Error cleaning up old patterns: {0}".format(e))
                                                                                                        return 0

                                                                                                            def export_patterns(self, filepath: str) -> bool:
                                                                                                            """Export patterns to JSON file"""
                                                                                                                try:
                                                                                                                export_data = {
                                                                                                                "metadata": {
                                                                                                                "export_time": time.time(),
                                                                                                                "total_patterns": len(self.snapshot_stack),
                                                                                                                "max_snapshots": self.max_snapshots,
                                                                                                                "similarity_threshold": self.similarity_threshold,
                                                                                                                },
                                                                                                                "patterns": [
                                                                                                                {
                                                                                                                "q_matrix": snapshot.q_matrix.tolist(),
                                                                                                                "timestamp": snapshot.timestamp,
                                                                                                                "strategy_id": snapshot.strategy_id,
                                                                                                                "profit_result": snapshot.profit_result,
                                                                                                                "market_context": snapshot.market_context,
                                                                                                                }
                                                                                                                for snapshot in self.snapshot_stack
                                                                                                                ],
                                                                                                                }

                                                                                                                    with open(filepath, "w") as f:
                                                                                                                    json.dump(export_data, f, indent=2)

                                                                                                                    logger.info("Exported {0} patterns to {1}".format(len(self.snapshot_stack), filepath))
                                                                                                                return True

                                                                                                                    except Exception as e:
                                                                                                                    logger.error("Error exporting patterns: {0}".format(e))
                                                                                                                return False

                                                                                                                    def import_patterns(self, filepath: str) -> bool:
                                                                                                                    """Import patterns from JSON file"""
                                                                                                                        try:
                                                                                                                            with open(filepath, "r") as f:
                                                                                                                            import_data = json.load(f)

                                                                                                                            imported_count = 0
                                                                                                                                for pattern_data in import_data.get("patterns", []):
                                                                                                                                snapshot = FractalSnapshot(
                                                                                                                                q_matrix=np.array(pattern_data["q_matrix"]),
                                                                                                                                timestamp=pattern_data["timestamp"],
                                                                                                                                strategy_id=pattern_data["strategy_id"],
                                                                                                                                profit_result=pattern_data.get("profit_result"),
                                                                                                                                market_context=pattern_data.get("market_context", {}),
                                                                                                                                )

                                                                                                                                self.snapshot_stack.append(snapshot)
                                                                                                                                imported_count += 1

                                                                                                                                # Maintain max size
                                                                                                                                    if len(self.snapshot_stack) > self.max_snapshots:
                                                                                                                                    self.snapshot_stack = self.snapshot_stack[-self.max_snapshots :]

                                                                                                                                    logger.info("Imported {0} patterns from {1}".format(imported_count, filepath))
                                                                                                                                return True

                                                                                                                                    except Exception as e:
                                                                                                                                    logger.error("Error importing patterns: {0}".format(e))
                                                                                                                                return False


                                                                                                                                    def create_fractal_memory_tracker() -> FractalMemoryTracker:
                                                                                                                                    """
                                                                                                                                    Factory function to create FractalMemoryTracker

                                                                                                                                        Args:
                                                                                                                                        max_snapshots: Maximum number of snapshots to store
                                                                                                                                        similarity_threshold: Minimum similarity for pattern matching

                                                                                                                                            Returns:
                                                                                                                                            Initialized FractalMemoryTracker instance
                                                                                                                                            """
                                                                                                                                        return FractalMemoryTracker()


                                                                                                                                            def test_fractal_memory_tracker():
                                                                                                                                            """Test function for fractal memory tracker"""
                                                                                                                                            print("🧠⚛️ Testing Fractal Memory Tracker")
                                                                                                                                            print("=" * 50)

                                                                                                                                            # Create tracker
                                                                                                                                            tracker = FractalMemoryTracker(max_snapshots=100, similarity_threshold=0.8)

                                                                                                                                            # Test data
                                                                                                                                            strategy_id = "test_strategy_456"
                                                                                                                                            q_matrix_1 = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
                                                                                                                                            q_matrix_2 = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 1]])  # Slightly different
                                                                                                                                            q_matrix_3 = np.array([[1, 0, 2], [2, 1, 0], [0, 2, 1]])  # Very different

                                                                                                                                            # Test snapshot saving
                                                                                                                                            print("📝 Testing snapshot saving...")
                                                                                                                                            snapshot_id_1 = tracker.save_snapshot(q_matrix_1, strategy_id, profit_result=0.5)
                                                                                                                                            snapshot_id_2 = tracker.save_snapshot(q_matrix_2, strategy_id, profit_result=-0.2)
                                                                                                                                            print("Saved snapshots: {0}, {1}".format(snapshot_id_1, snapshot_id_2))

                                                                                                                                            # Test fractal matching
                                                                                                                                            print("\n🔍 Testing fractal matching...")
                                                                                                                                            match_1 = tracker.match_fractal(q_matrix_1, strategy_id)
                                                                                                                                                if match_1:
                                                                                                                                                print("Exact match found: {0} (similarity: {1})".format(match_1.match_type.value, match_1.similarity_score))

                                                                                                                                                match_2 = tracker.match_fractal(q_matrix_2, strategy_id)
                                                                                                                                                    if match_2:
                                                                                                                                                    print("Similar match found: {0} (similarity: {1})".format(match_2.match_type.value, match_2.similarity_score))

                                                                                                                                                    match_3 = tracker.match_fractal(q_matrix_3, strategy_id)
                                                                                                                                                        if match_3:
                                                                                                                                                        print("Different match found: {0} (similarity: {1})".format(match_3.match_type.value, match_3.similarity_score))
                                                                                                                                                            else:
                                                                                                                                                            print("No match found for very different matrix")

                                                                                                                                                            # Test statistics
                                                                                                                                                            print("\n📊 Testing pattern statistics...")
                                                                                                                                                            stats = tracker.get_pattern_statistics()
                                                                                                                                                            print("Pattern stats: {0}".format(stats))


                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                test_fractal_memory_tracker()
