"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💰 PROFIT BUCKET REGISTRY - SCHWABOT PROFITABLE PATTERN STORAGE
==============================================================

Advanced profit bucket registry that stores profitable trade patterns as hash buckets
with their associated exit strategies, profit targets, and success rates.

Mathematical Components:
- Hash-based pattern matching with prefix similarity
- Confidence scoring with exponential moving average
- Profit percentage calculations with risk-adjusted returns
- Pattern similarity scoring using Hamming distance
- Time-weighted success rate calculations

Features:
- Persistent JSON storage for pattern survival
- Real-time pattern matching and confidence scoring
- Risk-adjusted profit calculations
- Pattern similarity and clustering
- Integration with hash-based trading system
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.distance import hamming

logger = logging.getLogger(__name__)

# Import dependencies
try:
from core.math_cache import MathResultCache
from core.math_config_manager import MathConfigManager
from core.math_orchestrator import MathOrchestrator
MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
MATH_INFRASTRUCTURE_AVAILABLE = False
logger.warning("Math infrastructure not available")

@dataclass
class ProfitBucket:
"""Class for Schwabot trading functionality."""
"""A profitable trade pattern with its exit strategy and metadata."""

hash_pattern: str
entry_price: float
exit_price: float
profit_pct: float
time_to_exit: int  # seconds
strategy_id: str
canonical_hash: Optional[str] = None  # Reference to canonical trade registry
success_count: int = 1
failure_count: int = 0
last_used: float = field(default_factory=time.time)
confidence: float = 0.5  # 0-1, increases with success
risk_adjusted_return: float = 0.0
max_drawdown: float = 0.0
sharpe_ratio: float = 0.0
pattern_complexity: float = 0.0
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternMatch:
"""Class for Schwabot trading functionality."""
"""Result of pattern matching operation."""
bucket: ProfitBucket
similarity_score: float
confidence: float
expected_profit: float
expected_time: int
risk_level: str


class ProfitBucketRegistry:
"""Class for Schwabot trading functionality."""
"""
💰 Advanced Profit Bucket Registry

Manages profitable trade patterns and their exit strategies with
mathematical rigor and real-time pattern matching capabilities.
"""

def __init__(self, store_path: str = "profit_buckets.json") -> None:
"""
Initialize the profit bucket registry.

Args:
store_path: Path to store profit buckets (default: profit_buckets.json)
"""
self.store_path = store_path
self.buckets: Dict[str, ProfitBucket] = {}
self.pattern_history: List[Dict[str, Any]] = []

# Performance tracking
self.total_patterns = 0
self.successful_matches = 0
self.total_profit = 0.0

# Initialize math infrastructure if available
if MATH_INFRASTRUCTURE_AVAILABLE:
self.math_config = MathConfigManager()
self.math_cache = MathResultCache()
self.math_orchestrator = MathOrchestrator()

self._load()
logger.info(f"💰 ProfitBucketRegistry initialized with {len(self.buckets)} buckets")

def market_hash(self, tick_blob: str) -> str:
"""
Generate hash from market tick data.

Args:
tick_blob: Market tick data string

Returns:
SHA256 hash string
"""
return hashlib.sha256(tick_blob.encode()).hexdigest()

def calculate_pattern_similarity(self, hash1: str, hash2: str) -> float:
"""
Calculate similarity between two hash patterns.

Args:
hash1: First hash string
hash2: Second hash string

Returns:
Similarity score (0-1)
"""
if hash1 == hash2:
return 1.0

# Convert hex strings to binary arrays for Hamming distance
try:
# Convert hex to binary
bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

# Ensure same length
max_len = max(len(bin1), len(bin2))
bin1 = bin1.ljust(max_len, '0')
bin2 = bin2.ljust(max_len, '0')

# Calculate Hamming distance
hamming_dist = hamming([int(b) for b in bin1], [int(b) for b in bin2])
similarity = 1.0 - hamming_dist

return max(0.0, similarity)

except ValueError:
# Fallback: character-based similarity
if len(hash1) != len(hash2):
return 0.0

matching_chars = sum(1 for a, b in zip(hash1, hash2) if a == b)
return matching_chars / len(hash1)

def add_profitable_trade(self, -> None
tick_blob: str,
entry_price: float,
exit_price: float,
time_to_exit: int,
strategy_id: str,
canonical_hash: Optional[str] = None,
risk_metrics: Optional[Dict[str, float]] = None) -> str:
"""
Record a successful trade pattern.

Args:
tick_blob: Market tick data
entry_price: Entry price
exit_price: Exit price
time_to_exit: Time to exit in seconds
strategy_id: Strategy identifier
canonical_hash: Reference to canonical trade registry
risk_metrics: Optional risk metrics (drawdown, volatility, etc.)

Returns:
Hash pattern string
"""
hash_pattern = self.market_hash(tick_blob)
profit_pct = ((exit_price - entry_price) / entry_price) * 100

# Calculate risk-adjusted metrics
risk_adjusted_return = profit_pct
max_drawdown = 0.0
sharpe_ratio = 0.0

if risk_metrics:
risk_adjusted_return = profit_pct / max(risk_metrics.get('volatility', 1.0), 0.01)
max_drawdown = risk_metrics.get('max_drawdown', 0.0)
sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)

# Calculate pattern complexity
pattern_complexity = self._calculate_complexity(tick_blob)

if hash_pattern in self.buckets:
# Update existing bucket
bucket = self.buckets[hash_pattern]
bucket.success_count += 1

# Update confidence using exponential moving average
success_rate = bucket.success_count / (bucket.success_count + bucket.failure_count)
bucket.confidence = 0.9 * bucket.confidence + 0.1 * success_rate

# Update metrics if this trade was more profitable
if profit_pct > bucket.profit_pct:
bucket.exit_price = exit_price
bucket.profit_pct = profit_pct
bucket.time_to_exit = time_to_exit
bucket.risk_adjusted_return = risk_adjusted_return
bucket.max_drawdown = max_drawdown
bucket.sharpe_ratio = sharpe_ratio

bucket.last_used = time.time()
bucket.pattern_complexity = pattern_complexity

# Update canonical hash reference if provided
if canonical_hash:
bucket.canonical_hash = canonical_hash

else:
# Create new bucket
self.buckets[hash_pattern] = ProfitBucket(
hash_pattern=hash_pattern,
entry_price=entry_price,
exit_price=exit_price,
profit_pct=profit_pct,
time_to_exit=time_to_exit,
strategy_id=strategy_id,
canonical_hash=canonical_hash,
last_used=time.time(),
risk_adjusted_return=risk_adjusted_return,
max_drawdown=max_drawdown,
sharpe_ratio=sharpe_ratio,
pattern_complexity=pattern_complexity,
metadata={"created_at": time.time()}
)

# Track pattern history
self.pattern_history.append({
"timestamp": time.time(),
"hash_pattern": hash_pattern,
"profit_pct": profit_pct,
"strategy_id": strategy_id,
"canonical_hash": canonical_hash,
"success": True
})

self.total_patterns += 1
self.total_profit += profit_pct

self._save()
logger.info(f"💰 Added profitable pattern: {hash_pattern[:8]}... | Profit: {profit_pct:.2f}%")

return hash_pattern

def record_trade_failure(self, tick_blob: str, strategy_id: str) -> None:
"""
Record a failed trade pattern.

Args:
tick_blob: Market tick data
strategy_id: Strategy identifier
"""
hash_pattern = self.market_hash(tick_blob)

if hash_pattern in self.buckets:
bucket = self.buckets[hash_pattern]
bucket.failure_count += 1

# Update confidence
success_rate = bucket.success_count / (bucket.success_count + bucket.failure_count)
bucket.confidence = 0.9 * bucket.confidence + 0.1 * success_rate

bucket.last_used = time.time()

# Track pattern history
self.pattern_history.append({
"timestamp": time.time(),
"hash_pattern": hash_pattern,
"profit_pct": 0.0,
"strategy_id": strategy_id,
"success": False
})

self._save()

def find_matching_pattern(self, tick_blob: str, -> None
min_confidence: float = 0.3,
min_similarity: float = 0.6) -> Optional[PatternMatch]:
"""
Find a profitable pattern that matches current market conditions.

Args:
tick_blob: Current market tick data
min_confidence: Minimum confidence threshold
min_similarity: Minimum similarity threshold

Returns:
PatternMatch object if found, None otherwise
"""
current_hash = self.market_hash(tick_blob)

best_match = None
best_score = 0.0

for bucket in self.buckets.values():
if bucket.confidence < min_confidence:
continue

# Calculate similarity
similarity = self.calculate_pattern_similarity(current_hash, bucket.hash_pattern)

if similarity < min_similarity:
continue

# Calculate composite score
score = similarity * bucket.confidence * (1 + bucket.profit_pct / 100)

if score > best_score:
best_score = score
best_match = bucket

if best_match:
# Calculate risk level
risk_level = self._calculate_risk_level(best_match)

return PatternMatch(
bucket=best_match,
similarity_score=best_score,
confidence=best_match.confidence,
expected_profit=best_match.profit_pct,
expected_time=best_match.time_to_exit,
risk_level=risk_level
)

return None

def get_exit_strategy(self, tick_blob: str, -> None
min_confidence: float = 0.3) -> Optional[Tuple[float, int, float]]:
"""
Get exit price, time, and confidence for current market conditions.

Args:
tick_blob: Current market tick data
min_confidence: Minimum confidence threshold

Returns:
Tuple of (exit_price, time_to_exit, confidence) or None
"""
pattern_match = self.find_matching_pattern(tick_blob, min_confidence)
if pattern_match:
return (pattern_match.bucket.exit_price,
pattern_match.bucket.time_to_exit,
pattern_match.confidence)
return None

def get_top_patterns(self, limit: int = 10, -> None
min_confidence: float = 0.5) -> List[ProfitBucket]:
"""
Get top performing patterns sorted by profit and confidence.

Args:
limit: Maximum number of patterns to return
min_confidence: Minimum confidence threshold

Returns:
List of top ProfitBucket objects
"""
qualified_buckets = [
bucket for bucket in self.buckets.values()
if bucket.confidence >= min_confidence
]

# Sort by composite score (profit * confidence * success_rate)
sorted_buckets = sorted(
qualified_buckets,
key=lambda b: b.profit_pct * b.confidence * (b.success_count / max(b.success_count + b.failure_count, 1)),
reverse=True
)

return sorted_buckets[:limit]

def get_bucket(self, hash_pattern: str) -> Optional[ProfitBucket]:
"""
Get a profit bucket by its hash pattern.

Args:
hash_pattern: Hash pattern to look up

Returns:
ProfitBucket if found, None otherwise
"""
return self.buckets.get(hash_pattern)

def get_bucket_stats(self) -> Dict[str, Any]:
"""Get comprehensive statistics about stored buckets."""
if not self.buckets:
return {
"total_buckets": 0,
"avg_confidence": 0.0,
"avg_profit": 0.0,
"total_patterns": 0,
"success_rate": 0.0
}

total_buckets = len(self.buckets)
confidences = [b.confidence for b in self.buckets.values()]
profits = [b.profit_pct for b in self.buckets.values()]

# Calculate success rates
total_successes = sum(b.success_count for b in self.buckets.values())
total_failures = sum(b.failure_count for b in self.buckets.values())
total_trades = total_successes + total_failures

return {
"total_buckets": total_buckets,
"avg_confidence": np.mean(confidences),
"avg_profit": np.mean(profits),
"max_profit": max(profits) if profits else 0.0,
"min_profit": min(profits) if profits else 0.0,
"total_patterns": self.total_patterns,
"total_successes": total_successes,
"total_failures": total_failures,
"success_rate": total_successes / max(total_trades, 1),
"total_profit": self.total_profit,
"avg_risk_adjusted_return": np.mean([b.risk_adjusted_return for b in self.buckets.values()]),
"avg_sharpe_ratio": np.mean([b.sharpe_ratio for b in self.buckets.values()])
}

def cleanup_old_patterns(self, max_age_hours: int = 24) -> int:
"""
Remove old patterns that haven't been used recently.

Args:
max_age_hours: Maximum age in hours before removal

Returns:
Number of patterns removed
"""
cutoff_time = time.time() - (max_age_hours * 3600)
initial_count = len(self.buckets)

self.buckets = {
k: v for k, v in self.buckets.items()
if v.last_used > cutoff_time or v.confidence > 0.7
}

removed_count = initial_count - len(self.buckets)
if removed_count > 0:
logger.info(f"Cleaned up {removed_count} old patterns")
self._save()

return removed_count

def _calculate_complexity(self, tick_blob: str) -> float:
"""Calculate complexity of tick data."""
try:
# Simple complexity based on data length and character diversity
unique_chars = len(set(tick_blob))
total_chars = len(tick_blob)
complexity = unique_chars / max(total_chars, 1)
return min(1.0, complexity)
except:
return 0.5

def _calculate_risk_level(self, bucket: ProfitBucket) -> str:
"""Calculate risk level based on bucket metrics."""
if bucket.max_drawdown > 0.1:  # >10% drawdown
return "high"
elif bucket.max_drawdown > 0.05:  # >5% drawdown
return "medium"
else:
return "low"

def _load(self) -> None:
"""Load buckets from JSON file."""
try:
with open(self.store_path, 'r') as f:
data = json.load(f)
self.buckets = {
k: ProfitBucket(**v) for k, v in data.items()
}
logger.info(f"Loaded {len(self.buckets)} profit buckets from {self.store_path}")
except FileNotFoundError:
self.buckets = {}
logger.info(f"No existing profit buckets found, starting fresh")
except Exception as e:
logger.error(f"Error loading profit buckets: {e}")
self.buckets = {}

def _save(self) -> None:
"""Save buckets to JSON file."""
try:
data = {k: asdict(v) for k, v in self.buckets.items()}
with open(self.store_path, 'w') as f:
json.dump(data, f, indent=2)
except Exception as e:
logger.error(f"Error saving profit buckets: {e}")


# Factory function
def create_profit_bucket_registry(store_path: str = "profit_buckets.json") -> ProfitBucketRegistry:
"""Create a ProfitBucketRegistry instance."""
return ProfitBucketRegistry(store_path)