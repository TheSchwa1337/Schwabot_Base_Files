import hashlib
import logging
import time
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import BTC_USDC_HASH_MEMORY, BTC_USDC_PRECISION_ANALYSIS

#!/usr/bin/env python3
"""Hash Memory Generator - Historical Pattern Recognition.

Generates SHA256 hash patterns from historical BTC/USDC data for Schwabot's
temporal intelligence and pattern recognition capabilities.

Key Features:
- Multi-decimal hash pattern generation (2, 6, 8 decimals)
- Historical pattern success rate tracking
- Hash entropy calculation for profit scoring
- Integration with QSC-GTS biological immune system
- Real-time hash pattern matching and learning
"""



logger = logging.getLogger(__name__)


class HashMemoryGenerator:
    """Generate hash memory from historical data for pattern recognition."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hash memory generator.

        Args:
            config: Configuration parameters
        """
        self.config = config or self._default_config()

        # Hash memory storage
        self.hash_memory: Optional[pd.DataFrame] = None
        self.precision_analysis: Optional[pd.DataFrame] = None

        # Pattern tracking
        self.hash_patterns: Dict[str, List[float]] = {}  # hash -> profit history
        self.pattern_success_rates: Dict[str, float] = {}
        self.pattern_frequencies: Dict[str, int] = {}

        # Performance metrics
        self.total_patterns_generated = 0
        self.unique_patterns = 0

        logger.info("🔐 Hash Memory Generator initialized")

    def _default_config():-> Dict[str, Any]:
        """Default configuration for hash memory generator."""
        return {
            "hash_memory_window": 1000,  # Hash memory lookback window
            "precision_analysis_window": 500,  # Precision analysis window
            "min_pattern_frequency": 0.01,  # 1% minimum occurrence rate
            "hash_entropy_threshold": 0.3,  # Minimum hash entropy for patterns
            "pattern_strength_threshold": 0.4,  # Minimum pattern strength
            "enable_multi_decimal": True,  # Enable multi-decimal analysis
            "enable_16bit_mapping": True,  # Enable 16-bit tick mapping
            "enable_temporal_analysis": True,  # Enable temporal pattern analysis
            "max_pattern_history": 10000,  # Maximum patterns to track
            "pattern_learning_rate": 0.1,  # Pattern learning rate
            "success_rate_decay": 0.95,  # Success rate decay factor
        }

    def generate_hash_memory():-> bool:
        """Generate hash memory from historical data.

        Args:
            historical_data: Historical BTC/USDC price data

        Returns:
            True if hash memory generated successfully
        """
        try:
            logger.info("🔐 Generating hash memory from historical data...")

            if (
                historical_data is None
                or len(historical_data) < self.config["hash_memory_window"]
            ):
                logger.error(
                    "❌ Insufficient historical data for hash memory generation"
                )
                return False

            # Generate hash memory data
            hash_memory_data = []
            window_size = self.config["hash_memory_window"]

            for i in range(window_size, len(historical_data)):
                # Get window of data
                window_data = historical_data.iloc[i - window_size : i + 1]

                # Generate hash for this window
                hash_data = self._generate_window_hash(window_data)

                # Only include patterns above threshold
                if (
                    hash_data["pattern_strength"]
                    >= self.config["pattern_strength_threshold"]
                ):
                    hash_memory_data.append(
                        {
                            "timestamp": window_data.iloc[-1]["timestamp"],
                            "price": window_data.iloc[-1]["close"],
                            "hash_pattern": hash_data["hash_pattern"],
                            "hash_entropy": hash_data["hash_entropy"],
                            "price_volatility": hash_data["price_volatility"],
                            "volume_pattern": hash_data["volume_pattern"],
                            "trend_direction": hash_data["trend_direction"],
                            "pattern_strength": hash_data["pattern_strength"],
                            "multi_decimal_hashes": hash_data["multi_decimal_hashes"],
                            "tick_16bit": hash_data["tick_16bit"],
                            "temporal_features": hash_data["temporal_features"],
                        }
                    )

            self.hash_memory = pd.DataFrame(hash_memory_data)

            # Generate precision analysis
            if self.config["enable_multi_decimal"]:
                self._generate_precision_analysis(historical_data)

            # Save hash memory
            self._save_hash_memory()

            # Update pattern tracking
            self._update_pattern_tracking()

            self.total_patterns_generated = len(self.hash_memory)
            self.unique_patterns = len(self.hash_memory["hash_pattern"].unique())

            logger.info(
                f"✅ Generated hash memory for {self.total_patterns_generated:,} patterns"
            )
            logger.info(f"📊 Unique hash patterns: {self.unique_patterns:,}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to generate hash memory: {e}")
            return False

    def _generate_window_hash():-> Dict[str, Any]:
        """Generate hash pattern for a window of historical data."""
        # Create hash input string
        price_sequence = window_data["close"].values
        volume_sequence = window_data["volume"].values

        # Multi-decimal price formatting
        latest_price = price_sequence[-1]
        price_2_decimal = self._format_price(latest_price, 2)
        price_6_decimal = self._format_price(latest_price, 6)
        price_8_decimal = self._format_price(latest_price, 8)

        # Create hash input
        hash_input = f"{price_2_decimal}_{price_6_decimal}_{price_8_decimal}_"
        hash_input += f"{price_sequence.mean():.2f}_{volume_sequence.mean():.2f}_"
        hash_input += f"{price_sequence.std():.4f}_{volume_sequence.std():.4f}"

        # Generate hash
        hash_pattern = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # Calculate hash entropy
        hash_bytes = bytes.fromhex(hash_pattern)
        hash_entropy = -sum(
            (b / 255.0) * np.log2((b / 255.0) + 1e-8) for b in hash_bytes
        )
        hash_entropy = min(1.0, hash_entropy / 8.0)

        # Calculate pattern metrics
        price_volatility = price_sequence.std() / price_sequence.mean()
        volume_pattern = volume_sequence.std() / volume_sequence.mean()
        trend_direction = 1 if price_sequence[-1] > price_sequence[0] else -1
        pattern_strength = hash_entropy * (1 + price_volatility)

        # Multi-decimal hashes
        timestamp = window_data["timestamp"].iloc[-1].timestamp()
        multi_decimal_hashes = {
            "macro": self._hash_price(price_2_decimal, timestamp, "macro"),
            "standard": self._hash_price(price_6_decimal, timestamp, "standard"),
            "micro": self._hash_price(price_8_decimal, timestamp, "micro"),
        }

        # 16-bit tick mapping
        tick_16bit = self._map_to_16bit(latest_price)

        # Temporal features
        temporal_features = self._extract_temporal_features(window_data)

        return {
            "hash_pattern": hash_pattern,
            "hash_entropy": hash_entropy,
            "price_volatility": price_volatility,
            "volume_pattern": volume_pattern,
            "trend_direction": trend_direction,
            "pattern_strength": pattern_strength,
            "multi_decimal_hashes": multi_decimal_hashes,
            "tick_16bit": tick_16bit,
            "temporal_features": temporal_features,
        }

    def _generate_precision_analysis():-> None:
        """Generate multi-decimal precision analysis from historical data."""
        logger.info("📊 Generating multi-decimal precision analysis...")

        precision_data = []
        window_size = self.config["precision_analysis_window"]

        for i in range(window_size, len(historical_data)):
            # Get window of data
            window_data = historical_data.iloc[i - window_size : i + 1]

            # Generate precision analysis
            precision_analysis = self._analyze_precision_levels(window_data)

            precision_data.append(
                {
                    "timestamp": window_data.iloc[-1]["timestamp"],
                    "price": window_data.iloc[-1]["close"],
                    **precision_analysis,
                }
            )

        self.precision_analysis = pd.DataFrame(precision_data)

        # Save precision analysis
        self.precision_analysis.to_parquet(BTC_USDC_PRECISION_ANALYSIS)

        logger.info(
            f"✅ Generated precision analysis for {len(self.precision_analysis):,} records"
        )

    def _analyze_precision_levels():-> Dict[str, Any]:
        """Analyze price data at multiple decimal precision levels."""
        latest_price = window_data["close"].iloc[-1]

        # Multi-decimal formatting
        price_2_decimal = self._format_price(latest_price, 2)
        price_6_decimal = self._format_price(latest_price, 6)
        price_8_decimal = self._format_price(latest_price, 8)

        # Generate hashes for each precision level
        timestamp = window_data["timestamp"].iloc[-1].timestamp()

        hash_2_decimal = self._hash_price(price_2_decimal, timestamp, "macro")
        hash_6_decimal = self._hash_price(price_6_decimal, timestamp, "standard")
        hash_8_decimal = self._hash_price(price_8_decimal, timestamp, "micro")

        # Calculate profit scores for each precision level
        macro_profit_score = self._calculate_profit_score(
            hash_2_decimal, "macro", window_data
        )
        standard_profit_score = self._calculate_profit_score(
            hash_6_decimal, "standard", window_data
        )
        micro_profit_score = self._calculate_profit_score(
            hash_8_decimal, "micro", window_data
        )

        # 16-bit tick mapping
        tick_16bit = self._map_to_16bit(latest_price)

        return {
            "price_2_decimal": price_2_decimal,
            "price_6_decimal": price_6_decimal,
            "price_8_decimal": price_8_decimal,
            "hash_2_decimal": hash_2_decimal,
            "hash_6_decimal": hash_6_decimal,
            "hash_8_decimal": hash_8_decimal,
            "macro_profit_score": macro_profit_score,
            "standard_profit_score": standard_profit_score,
            "micro_profit_score": micro_profit_score,
            "tick_16bit": tick_16bit,
            "price_volatility": window_data["close"].std()
            / window_data["close"].mean(),
            "volume_activity": window_data["volume"].mean(),
            "trend_strength": abs(
                window_data["close"].iloc[-1] - window_data["close"].iloc[0]
            )
            / window_data["close"].iloc[0],
        }

    def _format_price():-> str:
        """Format price with specific decimal precision."""
        quant = Decimal("1." + ("0" * decimals))
        d_price = Decimal(str(price)).quantize(quant, rounding=ROUND_DOWN)
        return f"{d_price:.{decimals}f}"

    def _hash_price():-> str:
        """Generate SHA256 hash for price with timestamp and prefix."""
        data = f"{prefix}_{price_str}_{timestamp:.3f}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _map_to_16bit():-> int:
        """Map BTC price to 16-bit integer (0-65535)."""
        min_price, max_price = 10000.0, 100000.0
        clamped_price = max(min_price, min(max_price, price))
        normalized = (clamped_price - min_price) / (max_price - min_price)
        return int(normalized * 65535)

    def _calculate_profit_score():-> float:
        """Calculate profit score based on hash pattern and historical context."""
        # Calculate hash entropy
        hash_bytes = bytes.fromhex(price_hash)
        entropy = -sum((b / 255.0) * np.log2((b / 255.0) + 1e-8) for b in hash_bytes)
        base_score = min(1.0, entropy / 8.0)

        # Apply precision-specific modifiers
        precision_modifiers = {
            "macro": 0.8,  # Conservative macro scoring
            "standard": 1.0,  # Standard scoring
            "micro": 1.2,  # Boosted micro scoring
        }

        # Apply volatility modifier
        volatility = window_data["close"].std() / window_data["close"].mean()
        volatility_modifier = min(1.5, 1.0 + volatility * 10)

        modified_score = (
            base_score * precision_modifiers[precision_level] * volatility_modifier
        )
        return min(1.0, modified_score)

    def _extract_temporal_features():-> Dict[str, float]:
        """Extract temporal features from window data."""
        return {
            "hour_of_day": window_data["timestamp"].iloc[-1].hour,
            "day_of_week": window_data["timestamp"].iloc[-1].dayofweek,
            "month": window_data["timestamp"].iloc[-1].month,
            "price_momentum": (
                window_data["close"].iloc[-1] - window_data["close"].iloc[0]
            )
            / window_data["close"].iloc[0],
            "volume_trend": (
                (window_data["volume"].iloc[-1] - window_data["volume"].iloc[0])
                / window_data["volume"].iloc[0]
                if window_data["volume"].iloc[0] > 0
                else 0
            ),
            "volatility_regime": window_data["close"].std()
            / window_data["close"].mean(),
        }

    def _save_hash_memory():-> None:
        """Save hash memory to file."""
        if self.hash_memory is not None:
            self.hash_memory.to_parquet(BTC_USDC_HASH_MEMORY)
            logger.info(f"✅ Saved hash memory to {BTC_USDC_HASH_MEMORY}")

    def _update_pattern_tracking():-> None:
        """Update pattern tracking statistics."""
        if self.hash_memory is None:
            return

        # Update pattern frequencies
        pattern_counts = self.hash_memory["hash_pattern"].value_counts()
        total_patterns = len(self.hash_memory)

        for pattern, count in pattern_counts.items():
            self.pattern_frequencies[pattern] = count
            count / total_patterns

            # Initialize success rate if not exists
            if pattern not in self.pattern_success_rates:
                self.pattern_success_rates[pattern] = 0.5  # Default 50% success rate

            # Update hash patterns history
            if pattern not in self.hash_patterns:
                self.hash_patterns[pattern] = []

            # Add current timestamp for tracking
            self.hash_patterns[pattern].append(time.time())

            # Keep only recent history
            if len(self.hash_patterns[pattern]) > self.config["max_pattern_history"]:
                self.hash_patterns[pattern] = self.hash_patterns[pattern][
                    -self.config["max_pattern_history"] :
                ]

    def find_similar_patterns():-> List[Dict[str, Any]]:
        """Find similar hash patterns in historical data.

        Args:
            current_hash: Current hash pattern to match
            precision_level: Precision level for matching

        Returns:
            List of similar patterns with metadata
        """
        if self.hash_memory is None:
            return []

        similar_patterns = []

        # Find patterns with similar hash structure
        for _, row in self.hash_memory.iterrows():
            # Check multi-decimal hashes
            if precision_level in row["multi_decimal_hashes"]:
                historical_hash = row["multi_decimal_hashes"][precision_level]

                # Calculate hash similarity (Hamming distance)
                similarity = self._calculate_hash_similarity(
                    current_hash, historical_hash
                )

                if similarity >= 0.7:  # 70% similarity threshold
                    similar_patterns.append(
                        {
                            "timestamp": row["timestamp"],
                            "price": row["price"],
                            "hash_pattern": row["hash_pattern"],
                            "similarity": similarity,
                            "pattern_strength": row["pattern_strength"],
                            "trend_direction": row["trend_direction"],
                            "success_rate": self.pattern_success_rates.get(
                                row["hash_pattern"], 0.5
                            ),
                            "frequency": self.pattern_frequencies.get(
                                row["hash_pattern"], 1
                            ),
                        }
                    )

        # Sort by similarity and strength
        similar_patterns.sort(
            key=lambda x: (x["similarity"], x["pattern_strength"]), reverse=True
        )

        return similar_patterns[:10]  # Return top 10 similar patterns

    def _calculate_hash_similarity():-> float:
        """Calculate similarity between two hash patterns."""
        if len(hash1) != len(hash2):
            return 0.0

        # Convert hex strings to binary and calculate Hamming distance
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

        hamming_distance = sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
        max_distance = len(bin1)

        similarity = 1.0 - (hamming_distance / max_distance)
        return similarity

    def update_pattern_success():-> None:
        """Update pattern success rate based on trading outcome.

        Args:
            hash_pattern: Hash pattern to update
            success: Whether the pattern led to successful trade
        """
        if hash_pattern not in self.pattern_success_rates:
            self.pattern_success_rates[hash_pattern] = 0.5

        current_rate = self.pattern_success_rates[hash_pattern]
        learning_rate = self.config["pattern_learning_rate"]

        if success:
            new_rate = current_rate + learning_rate * (1.0 - current_rate)
        else:
            new_rate = current_rate + learning_rate * (0.0 - current_rate)

        # Apply decay factor
        decay = self.config["success_rate_decay"]
        self.pattern_success_rates[hash_pattern] = new_rate * decay + 0.5 * (1 - decay)

    def get_pattern_statistics():-> Dict[str, Any]:
        """Get comprehensive pattern statistics."""
        if self.hash_memory is None:
            return {}

        total_patterns = len(self.hash_memory)
        unique_patterns = len(self.hash_memory["hash_pattern"].unique())

        # Calculate average pattern strength
        avg_pattern_strength = self.hash_memory["pattern_strength"].mean()
        avg_hash_entropy = self.hash_memory["hash_entropy"].mean()

        # Calculate success rate distribution
        success_rates = list(self.pattern_success_rates.values())
        avg_success_rate = np.mean(success_rates) if success_rates else 0.5

        return {
            "total_patterns": total_patterns,
            "unique_patterns": unique_patterns,
            "pattern_diversity": (
                unique_patterns / total_patterns if total_patterns > 0 else 0
            ),
            "avg_pattern_strength": avg_pattern_strength,
            "avg_hash_entropy": avg_hash_entropy,
            "avg_success_rate": avg_success_rate,
            "high_success_patterns": sum(1 for rate in success_rates if rate > 0.7),
            "low_success_patterns": sum(1 for rate in success_rates if rate < 0.3),
            "pattern_frequencies": len(self.pattern_frequencies),
            "hash_patterns_tracked": len(self.hash_patterns),
        }

    def get_system_status():-> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "hash_memory_loaded": self.hash_memory is not None,
            "precision_analysis_loaded": self.precision_analysis is not None,
            "total_patterns_generated": self.total_patterns_generated,
            "unique_patterns": self.unique_patterns,
            "pattern_statistics": self.get_pattern_statistics(),
            "configuration": self.config,
        }


# Helper function for easy integration
def create_hash_memory_generator():-> HashMemoryGenerator:
    """Create and initialize hash memory generator.

    Args:
        config: Optional configuration parameters

    Returns:
        Initialized hash memory generator
    """
    return HashMemoryGenerator(config)
