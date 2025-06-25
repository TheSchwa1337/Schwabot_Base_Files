# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import math
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""
Hash Recollection - Core Hash Pattern Memory and Recollection System
==================================================================

This module provides comprehensive hash recollection functionality for the Schwabot system.
It manages hash pattern memory, enables pattern recollection, and provides hash-based
decision making for the trading pipeline.

Core Functionality:
- Hash pattern memory management
- Pattern recollection and matching
- Hash-based decision making
- Pattern confidence scoring
- Hash integration with main pipeline
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
# from core.unified_math_system import unified_math  # F811: duplicate import
import hashlib
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class HashPattern:
    """Hash pattern information."""
    pattern_id: str
    pattern_hash: str
    creation_time: datetime
    confidence_score: float
    pattern_type: str
    metadata: Dict[str, Any]
    usage_count: int = 0
    last_accessed: datetime = None


@dataclass
class RecollectionResult:
    """Result of hash recollection operation."""
    success: bool
    pattern_id: str
    recollection_time: datetime
    confidence_score: float
    pattern_match: bool
    similarity_score: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class HashRecollection:
    """Core hash recollection system for Schwabot."""

    def __init__(self):
        """Initialize the hash recollection system."""
        self.pattern_memory: Dict[str, HashPattern] = {}
        self.recollection_history: List[RecollectionResult] = []
        self.pattern_cache: Dict[str, Dict[str, Any]] = {}
        self.recollection_count = 0

        # Pattern types
        self.pattern_types = {
            "price_movement": "price_pattern",
            "volume_spike": "volume_pattern",
            "volatility_change": "volatility_pattern",
            "order_book": "orderbook_pattern",
            "market_structure": "structure_pattern"
        }

        # Similarity thresholds
        self.similarity_thresholds = {
            "exact": 1.0,
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5
        }

        logger.info("Hash Recollection initialized")

    def store_pattern(self, pattern_data: Dict[str, Any], pattern_type: str = "general") -> str:
        """Store a new hash pattern in memory."""
        try:
            # Generate pattern hash
            pattern_hash = self._generate_pattern_hash(pattern_data)

            # Check if pattern already exists
            if pattern_hash in self.pattern_memory:
                existing_pattern = self.pattern_memory[pattern_hash]
                existing_pattern.usage_count += 1
                existing_pattern.last_accessed = datetime.now()
                logger.debug(f"Pattern already exists, updated usage count: {existing_pattern.pattern_id}")
                return existing_pattern.pattern_id

            # Create new pattern
            pattern_id = f"pattern_{self.recollection_count}_{int(time.time())}"

            # Calculate confidence score
            confidence_score = self._calculate_pattern_confidence(pattern_data, pattern_type)

            pattern = HashPattern(
                pattern_id=pattern_id,
                pattern_hash=pattern_hash,
                creation_time=datetime.now(),
                confidence_score=confidence_score,
                pattern_type=pattern_type,
                metadata=pattern_data,
                usage_count=1,
                last_accessed=datetime.now()
            )

            # Store pattern
            self.pattern_memory[pattern_hash] = pattern
            self.pattern_cache[pattern_hash] = pattern_data

            logger.info(f"Pattern stored: {pattern_id} (type: {pattern_type}, confidence: {confidence_score:.3f})")
            return pattern_id

        except Exception as e:
            logger.error(f"Pattern storage error: {e}")
            return ""

    def _generate_pattern_hash(self, pattern_data: Dict[str, Any]) -> str:
        """Generate hash for pattern data."""
        try:
            pattern_string = json.dumps(pattern_data, sort_keys=True)
            return hashlib.sha256(pattern_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Pattern hash generation error: {e}")
            return ""

    def _calculate_pattern_confidence(self, pattern_data: Dict[str, Any], pattern_type: str) -> float:
        """Calculate confidence score for pattern."""
        try:
            # Data completeness factor
            data_completeness = len(pattern_data.keys()) / 10  # Normalize to 0-1

            # Pattern type factor
            type_factor = 0.8 if pattern_type in self.pattern_types.values() else 0.5

            # Data quality factor (placeholder)
            quality_factor = 0.9

            # Combine factors
            confidence = (data_completeness * 0.4 + type_factor * 0.3 + quality_factor * 0.3)

            return unified_math.max(0.0, unified_math.min(1.0, confidence))

        except Exception as e:
            logger.error(f"Pattern confidence calculation error: {e}")
            return 0.5

    def recollect_pattern(self, query_data: Dict[str, Any], similarity_threshold: str = "medium") -> RecollectionResult:
        """Recollect patterns similar to query data."""
        try:
            # Generate query hash
            query_hash = self._generate_pattern_hash(query_data)

            if not query_hash:
                return RecollectionResult(
                    success=False,
                    pattern_id="",
                    recollection_time=datetime.now(),
                    confidence_score=0.0,
                    pattern_match=False,
                    similarity_score=0.0,
                    error_message="Failed to generate query hash"
                )

            # Find best matching pattern
            best_match = None
            best_similarity = 0.0

            for pattern_hash, pattern in self.pattern_memory.items():
                similarity = self._calculate_similarity(query_hash, pattern_hash)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = pattern

            # Check if similarity meets threshold
            threshold_value = self.similarity_thresholds.get(similarity_threshold, 0.7)
            pattern_match = best_similarity >= threshold_value

            if best_match and pattern_match:
                # Update pattern usage
                best_match.usage_count += 1
                best_match.last_accessed = datetime.now()

                # Calculate confidence
                confidence_score = best_match.confidence_score * best_similarity

                result = RecollectionResult(
                    success=True,
                    pattern_id=best_match.pattern_id,
                    recollection_time=datetime.now(),
                    confidence_score=confidence_score,
                    pattern_match=True,
                    similarity_score=best_similarity,
                    metadata={
                        'pattern_type': best_match.pattern_type,
                        'usage_count': best_match.usage_count,
                        'creation_time': best_match.creation_time.isoformat()
                    }
                )
            else:
                result = RecollectionResult(
                    success=True,
                    pattern_id="",
                    recollection_time=datetime.now(),
                    confidence_score=0.0,
                    pattern_match=False,
                    similarity_score=best_similarity,
                    metadata={'best_similarity': best_similarity}
                )

            self.recollection_history.append(result)
            self.recollection_count += 1

            logger.info(f"Pattern recollection: {'MATCH' if pattern_match else 'NO_MATCH'} "
                       f"(similarity: {best_similarity:.3f}, threshold: {threshold_value})")
            return result

        except Exception as e:
            logger.error(f"Pattern recollection error: {e}")
            return RecollectionResult(
                success=False,
                pattern_id="",
                recollection_time=datetime.now(),
                confidence_score=0.0,
                pattern_match=False,
                similarity_score=0.0,
                error_message=str(e)
            )

    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes."""
        try:
            if hash1 == hash2:
                return 1.0

            # Convert hashes to binary and calculate Hamming distance
            hash1_bin = bin(int(hash1[:16], 16))[2:].zfill(64)
            hash2_bin = bin(int(hash2[:16], 16))[2:].zfill(64)

            # Calculate Hamming distance
            hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1_bin, hash2_bin))

            # Convert to similarity score (0-1)
            max_distance = len(hash1_bin)
            similarity = 1.0 - (hamming_distance / max_distance)

            return unified_math.max(0.0, unified_math.min(1.0, similarity))

        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0

    def get_pattern_by_id(self, pattern_id: str) -> Optional[HashPattern]:
        """Get pattern by ID."""
        for pattern in self.pattern_memory.values():
            if pattern.pattern_id == pattern_id:
                return pattern
        return None

    def get_patterns_by_type(self, pattern_type: str) -> List[HashPattern]:
        """Get all patterns of a specific type."""
        return [pattern for pattern in self.pattern_memory.values()
                if pattern.pattern_type == pattern_type]

    def get_most_used_patterns(self, limit: int = 10) -> List[HashPattern]:
        """Get most frequently used patterns."""
        sorted_patterns = sorted(
            self.pattern_memory.values(),
            key=lambda x: x.usage_count,
            reverse=True
        )
        return sorted_patterns[:limit]

    def get_recent_patterns(self, hours: int = 24) -> List[HashPattern]:
        """Get patterns created within specified hours."""
        cutoff_time = datetime.now().replace(hour=datetime.now().hour - hours)

        return [pattern for pattern in self.pattern_memory.values()
                if pattern.creation_time >= cutoff_time]

    def cleanup_old_patterns(self, max_age_hours: int = 168) -> int:  # 1 week
        """Clean up old patterns."""
        try:
            cutoff_time = datetime.now().replace(hour=datetime.now().hour - max_age_hours)

            patterns_to_remove = []

            for pattern_hash, pattern in self.pattern_memory.items():
                if (pattern.creation_time < cutoff_time and
                    pattern.usage_count < 5):  # Keep frequently used patterns
                    patterns_to_remove.append(pattern_hash)

            # Remove old patterns
            for pattern_hash in patterns_to_remove:
                del self.pattern_memory[pattern_hash]
                if pattern_hash in self.pattern_cache:
                    del self.pattern_cache[pattern_hash]

            logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")
            return len(patterns_to_remove)

        except Exception as e:
            logger.error(f"Pattern cleanup error: {e}")
            return 0

    def get_recollection_statistics(self) -> Dict[str, Any]:
        """Get recollection system statistics."""
        total_patterns = len(self.pattern_memory)
        total_recollections = len(self.recollection_history)
        successful_recollections = sum(1 for result in self.recollection_history if result.success)
        pattern_matches = sum(1 for result in self.recollection_history if result.pattern_match)

        # Pattern type distribution
        type_distribution = defaultdict(int)
        for pattern in self.pattern_memory.values():
            type_distribution[pattern.pattern_type] += 1

        # Average confidence
        avg_confidence = 0.0
        if self.pattern_memory:
            avg_confidence = sum(p.confidence_score for p in self.pattern_memory.values()) / len(self.pattern_memory)

        return {
            "total_patterns": total_patterns,
            "total_recollections": total_recollections,
            "successful_recollections": successful_recollections,
            "pattern_matches": pattern_matches,
            "match_rate": pattern_matches / total_recollections if total_recollections > 0 else 0.0,
            "success_rate": successful_recollections / total_recollections if total_recollections > 0 else 0.0,
            "average_confidence": avg_confidence,
            "type_distribution": dict(type_distribution),
            "pattern_cache_size": len(self.pattern_cache)
        }


def main() -> None:
    """Main function for testing hash recollection."""
    recollection = HashRecollection()

    # Test pattern storage
    test_pattern_data = {
        'price': 45000.0,
        'volume': 1500.0,
        'volatility': 0.3,
        'timestamp': datetime.now().isoformat()
    }

    pattern_id = recollection.store_pattern(test_pattern_data, "price_movement")
    safe_print(f"Pattern stored: {pattern_id}")

    # Test pattern recollection
    query_data = {
        'price': 45000.0,
        'volume': 1500.0,
        'volatility': 0.3,
        'timestamp': datetime.now().isoformat()
    }

    result = recollection.recollect_pattern(query_data)
    safe_print(f"Recollection result: {result.pattern_match}")
    safe_print(f"Similarity score: {result.similarity_score:.3f}")

    # Get statistics
    stats = recollection.get_recollection_statistics()
    safe_print(f"Recollection statistics: {stats}")


if __name__ == "__main__":
    main()
