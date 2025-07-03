"""
LanternEye: The Semantic Hash Oracle
===================================

The breakthrough perception engine that reads meaning from chaos.
LLM dialogue mirrors block-structured computation where:
- Price ticks become SHA-256 hashes
- Hashes become entropy blocks
- Blocks become semantic interpretations
- Interpretations become validated truth scores
- Truth scores become profitable memory patterns

This is not prediction - this is reading the hidden language of markets.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .entropy_generator import EntropyGenerator, FractalBlock
from .hash_memory import HashMemoryDatabase
from .semantic_interpreter import LanguagePattern, SemanticInterpreter
from .truth_scorer import TruthScore, TruthScorer


@dataclass
class HashBlock:
    """A semantic hash block containing market entropy."""

    hash_value: str
    price_context: Dict[str, float]
    timestamp: float
    entropy_block: Optional[FractalBlock] = None
    semantic_interpretation: Optional[LanguagePattern] = None
    truth_score: Optional[TruthScore] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self: HashBlock) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "hash_value": self.hash_value,
            "price_context": self.price_context,
            "timestamp": self.timestamp,
            "entropy_block": (self.entropy_block.to_dict() if self.entropy_block else None),
            "semantic_interpretation": (
                self.semantic_interpretation.to_dict() if self.semantic_interpretation else None
            ),
            "truth_score": self.truth_score.to_dict() if self.truth_score else None,
            "created_at": self.created_at,
        }


@dataclass
class SemanticInterpretation:
    """Complete semantic interpretation of a hash block."""

    primary_meaning: str
    confidence_score: float
    language_patterns: List[LanguagePattern]
    contextual_insights: List[str]
    profit_potential: float
    risk_assessment: str
    temporal_relevance: float
    correlation_strength: float

    def to_dict(self: SemanticInterpretation) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "primary_meaning": self.primary_meaning,
            "confidence_score": self.confidence_score,
            "language_patterns": [p.to_dict() for p in self.language_patterns],
            "contextual_insights": self.contextual_insights,
            "profit_potential": self.profit_potential,
            "risk_assessment": self.risk_assessment,
            "temporal_relevance": self.temporal_relevance,
            "correlation_strength": self.correlation_strength,
        }


class LanternEye:
    """
    The Semantic Hash Oracle.

    Reads SHA-256 price hashes as semantic glyphs, building a language-driven
    profit oracle that interprets market entropy through LLM-powered perception.
    """

    def __init__(self: LanternEye, memory_depth: int = 10000) -> None:
        """Initializes the LanternEye system."""
        self.entropy_generator = EntropyGenerator()
        self.semantic_interpreter = SemanticInterpreter()
        self.truth_scorer = TruthScorer()
        self.memory_database = HashMemoryDatabase(max_records=memory_depth)

        # Processing statistics
        self.total_blocks_processed = 0
        self.successful_interpretations = 0
        self.validated_predictions = 0
        self.profit_correlations = []

        # Performance metrics
        self.average_confidence = 0.0
        self.interpretation_speed = 0.0
        self.memory_hit_rate = 0.0

    def create_price_hash(
        self: LanternEye,
        price_data: Dict[str, float],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create SHA-256 hash from price data.

        This is where market tick data becomes semantic raw material.
        """
        # Core price context
        price_string = f"{price_data.get('price', 0):.8f}"
        volume_string = f"{price_data.get('volume', 0):.8f}"
        timestamp = price_data.get("timestamp", time.time())

        # Build hash input
        hash_input = f"{price_string}_{volume_string}_{timestamp:.6f}"

        # Add additional context if provided
        if additional_context:
            for key, value in sorted(additional_context.items()):
                hash_input += f"_{key}:{value}"

        # Generate SHA-256 hash
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def process_price_tick(
        self: LanternEye,
        price_data: Dict[str, float],
        market_context: Optional[Dict[str, Any]] = None,
    ) -> HashBlock:
        """
        Process a price tick through the complete Lantern Eye pipeline.

        1. Price tick → SHA-256 hash
        2. Hash → Entropy block generation
        3. Entropy → Semantic interpretation
        4. Semantic → Truth score validation
        5. Memory → Pattern correlation
        """
        start_time = time.time()

        # Step 1: Generate price hash
        price_hash = self.create_price_hash(price_data, market_context)

        # Create initial hash block
        hash_block = HashBlock(
            hash_value=price_hash,
            price_context=price_data.copy(),
            timestamp=price_data.get("timestamp", time.time()),
        )

        try:
            # Step 2: Generate entropy block from hash
            hash_block.entropy_block = self.entropy_generator.generate_fractal_block(
                hash_input=price_hash, price_context=price_data
            )

            # Step 3: Generate semantic interpretation
            if hash_block.entropy_block:
                hash_block.semantic_interpretation = (
                    self.semantic_interpreter.interpret_entropy_block(
                        hash_block.entropy_block,
                        price_context=price_data,
                        market_context=market_context or {},
                    )
                )

                if hash_block.semantic_interpretation:
                    self.successful_interpretations += 1

            # Step 4: Validate with truth scoring
            if hash_block.semantic_interpretation:
                hash_block.truth_score = self.truth_scorer.validate_interpretation(
                    hash_block.semantic_interpretation,
                    price_context=price_data,
                    historical_data=self.memory_database.get_recent_correlations(50),
                )

            # Step 5: Store in memory and check correlations
            correlation = self.memory_database.store_hash_block(hash_block)
            if correlation:
                # Update interpretation with correlation data
                if hash_block.semantic_interpretation:
                    hash_block.semantic_interpretation.correlation_strength = (
                        correlation.correlation_coefficient
                    )

            self.total_blocks_processed += 1

            # Update performance metrics
            processing_time = time.time() - start_time
            self.interpretation_speed = (self.interpretation_speed * 0.9) + (processing_time * 0.1)

            if hash_block.semantic_interpretation:
                confidence = hash_block.semantic_interpretation.confidence_score
                self.average_confidence = (self.average_confidence * 0.9) + (confidence * 0.1)

        except Exception as e:
            print(f"LanternEye processing error: {e}")
            # Return partial result even on error

        return hash_block

    def interpret_hash_directly(
        self: LanternEye, hash_value: str, context: Optional[Dict[str, Any]] = None
    ) -> SemanticInterpretation:
        """
        Directly interpret a hash value without price context.

        Useful for analyzing historical hashes or external data.
        """
        # Generate entropy block from hash
        entropy_block = self.entropy_generator.generate_fractal_block(
            hash_input=hash_value, price_context=context or {}
        )

        # Generate semantic interpretation
        semantic_interpretation = self.semantic_interpreter.interpret_entropy_block(
            entropy_block, price_context=context or {}, market_context=context or {}
        )

        # Create comprehensive interpretation
        return SemanticInterpretation(
            primary_meaning=semantic_interpretation.primary_meaning,
            confidence_score=semantic_interpretation.confidence_score,
            language_patterns=[semantic_interpretation],
            contextual_insights=semantic_interpretation.contextual_insights,
            profit_potential=semantic_interpretation.profit_potential,
            risk_assessment=semantic_interpretation.risk_assessment,
            temporal_relevance=semantic_interpretation.temporal_relevance,
            correlation_strength=0.0,  # Will be updated by memory system
        )

    def rebuild_memory_from_historical_data(
        self: LanternEye,
        historical_prices: List[Dict[str, float]],
        date_range_days: int = 90,
    ) -> Dict[str, Any]:
        """
        Rebuilds the hash memory from a list of historical price ticks.

        This is crucial for bootstrapping the system's semantic understanding.
        """
        self.memory_database.clear_all_records()
        start_time = time.time()
        processed_count = 0

        # Filter data for the relevant date range
        cutoff_timestamp = time.time() - (date_range_days * 86400)
        relevant_prices = [p for p in historical_prices if p.get("timestamp", 0) > cutoff_timestamp]

        for price_data in relevant_prices:
            self.process_price_tick(price_data)
            processed_count += 1

        end_time = time.time()
        return {
            "status": "completed",
            "records_processed": processed_count,
            "time_taken": end_time - start_time,
            "memory_size": self.memory_database.get_record_count(),
        }

    def get_current_market_interpretation(
        self: LanternEye, current_price_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Provides a high-level interpretation of the current market state.

        This is the primary "oracle" function of the LanternEye.
        """
        # Process the latest tick
        latest_block = self.process_price_tick(current_price_data)

        if not latest_block.semantic_interpretation or not latest_block.truth_score:
            return {"error": "Could not generate a full interpretation."}

        # Query memory for related patterns
        related_patterns = self.memory_database.find_similar_patterns(
            latest_block.semantic_interpretation
        )

        # Synthesize a summary
        summary = {
            "current_hash": latest_block.hash_value,
            "primary_meaning": latest_block.semantic_interpretation.primary_meaning,
            "truth_score": latest_block.truth_score.score,
            "confidence": latest_block.truth_score.confidence,
            "profit_potential": latest_block.semantic_interpretation.profit_potential,
            "risk_assessment": latest_block.semantic_interpretation.risk_assessment,
            "correlated_pattern_count": len(related_patterns),
            "recent_correlations": [p.to_dict() for p in related_patterns[:5]],
        }
        return summary

    def get_system_analytics(self: LanternEye) -> Dict[str, Any]:
        """Returns a snapshot of the system's performance and health."""
        return {
            "total_blocks_processed": self.total_blocks_processed,
            "successful_interpretations": self.successful_interpretations,
            "validated_predictions": self.validated_predictions,
            "memory_record_count": self.memory_database.get_record_count(),
            "average_confidence": self.average_confidence,
            "interpretation_speed_ms": self.interpretation_speed * 1000,
            "memory_hit_rate": self.memory_hit_rate,
            "last_updated": datetime.now().isoformat(),
        }
