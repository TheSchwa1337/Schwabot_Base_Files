"""
Profit Bucket Registry
=====================

Stores profitable trade patterns as hash buckets with their associated
exit strategies, profit targets, and success rates. This enables
recursive strategy layering where successful patterns reinforce themselves.

Mathematical Components:
- Hash-based pattern matching with prefix similarity
- Confidence scoring with exponential moving average
- Profit percentage calculations
- Pattern similarity scoring
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ProfitBucket:
    """A profitable trade pattern with its exit strategy."""

    hash_pattern: str
    entry_price: float
    exit_price: float
    profit_pct: float
    time_to_exit: int  # seconds
    strategy_id: str
    success_count: int = 1
    last_used: float = 0.0
    confidence: float = 0.5  # 0-1, increases with success


class ProfitBucketRegistry:
    """Manages profitable trade patterns and their exit strategies."""

    def __init__(self,  store_path: str = "profit_buckets.json") -> None:
        """Initialize the profit bucket registry."""
        self.store_path = store_path
        self.buckets: Dict[str, ProfitBucket] = {}
        self._load()

    def market_hash(self,  tick_blob: str) -> str:
        """Generate hash from market tick data."""
        return hashlib.sha256(tick_blob.encode()).hexdigest()

    def add_profitable_trade(
        self,
        tick_blob: str,
        entry_price: float,
        exit_price: float,
        time_to_exit: int,
        strategy_id: str,
    ) -> None:
        """Record a successful trade pattern."""
        hash_pattern = self.market_hash(tick_blob)
        profit_pct = ((exit_price - entry_price) / entry_price) * 100

        if hash_pattern in self.buckets:
            # Update existing bucket
            bucket = self.buckets[hash_pattern]
            bucket.success_count += 1
            bucket.confidence = min(1.0, bucket.confidence + 0.1)
            bucket.last_used = time.time()
            # Update exit strategy if this one was more profitable
            if profit_pct > bucket.profit_pct:
                bucket.exit_price = exit_price
                bucket.profit_pct = profit_pct
                bucket.time_to_exit = time_to_exit
        else:
            # Create new bucket
            self.buckets[hash_pattern] = ProfitBucket(
                hash_pattern=hash_pattern,
                entry_price=entry_price,
                exit_price=exit_price,
                profit_pct=profit_pct,
                time_to_exit=time_to_exit,
                strategy_id=strategy_id,
                last_used=time.time(),
            )

        self._save()

    def find_matching_pattern(self,  tick_blob: str, min_confidence: float = 0.3) -> Optional[ProfitBucket]:
        """Find a profitable pattern that matches current market conditions."""
        current_hash = self.market_hash(tick_blob)

        best_match = None
        best_score = 0.0

        for bucket in self.buckets.values():
            if bucket.confidence < min_confidence:
                continue

            # Simple prefix matching (can be enhanced with more sophisticated similarity)
            match_length = 0
            for _i, (a, b) in enumerate(zip(current_hash, bucket.hash_pattern)):
                if a == b:
                    match_length += 1
                else:
                    break

            match_score = (match_length / len(current_hash)) * bucket.confidence
            if match_score > best_score:
                best_score = match_score
                best_match = bucket

        return best_match if best_score > 0.6 else None

    def get_exit_strategy(self,  tick_blob: str) -> Optional[Tuple[float, int]]:
        """Get exit price and time for current market conditions."""
        bucket = self.find_matching_pattern(tick_blob)
        if bucket:
            return bucket.exit_price, bucket.time_to_exit
        return None

    def get_bucket_stats(self) -> Dict[str, any]:
        """Get statistics about stored buckets."""
        if not self.buckets:
            return {"total_buckets": 0, "avg_confidence": 0.0, "avg_profit": 0.0}

        total_buckets = len(self.buckets)
        avg_confidence = sum(b.confidence for b in self.buckets.values()) / total_buckets
        avg_profit = sum(b.profit_pct for b in self.buckets.values()) / total_buckets

        return {
            "total_buckets": total_buckets,
            "avg_confidence": avg_confidence,
            "avg_profit": avg_profit,
        }

    def _load(self) -> None:
        """Load buckets from JSON file."""
        try:
            with open(self.store_path, 'r') as f:
                data = json.load(f)
                self.buckets = {k: ProfitBucket(**v) for k, v in data.items()}
        except FileNotFoundError:
            self.buckets = {}

    def _save(self) -> None:
        """Save buckets to JSON file."""
        data = {k: asdict(v) for k, v in self.buckets.items()}
        with open(self.store_path, 'w') as f:
            json.dump(data, f, indent=2)
