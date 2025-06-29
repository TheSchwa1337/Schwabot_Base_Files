# -*- coding: utf-8 -*-
"""
Enhanced Lantern Core with Word Library and Glyph Integration
=============================================================

The Enhanced Lantern Core provides introspection, validation, and tagging for
decisions made by the Dualistic Thought Engines. It also includes an advanced
word categorization and bit-gate processing system for nuanced decision routing,
acting as the bridge between raw data and abstract cognitive functions.

This system is the embodiment of the "breathing room" in the architecture,
allowing for detailed logging, historical analysis, and improved validation
of the bot's cognitive processes.

Mathematical Integration:
- SHA-256 based word-to-hash mapping for BTC price correlation
- Bit gate processing for profit tier navigation and decision intensity
- Entropy word generation for glyph routing and state symbolization
- Tagging Logic: T(v) = {tagᵢ | conditionᵢ(v) is true}
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Tuple
import re
from collections import defaultdict

from .thought_types import (
    BitGate,
    BitGateType,
    BrainTag,
    DualisticState,
    ThoughtVector,
)

logger = logging.getLogger(__name__)

# --- Word Libraries from Backup ---
PROFIT_WORDS = [
    "profit",
    "gain",
    "yield",
    "return",
    "growth",
    "increase",
    "rise",
    "bull",
    "moon",
    "rocket",
    "surge",
    "pump",
    "spike",
    "climb",
    "breakout",
    "momentum",
    "uptrend",
    "rally",
    "boom",
    "success",
    "wealth",
    "fortune",
    "treasure",
    "golden",
    "diamond",
    "victory",
]
NAVIGATION_WORDS = [
    "navigate",
    "steer",
    "guide",
    "direct",
    "route",
    "path",
    "journey",
    "compass",
    "beacon",
    "lighthouse",
    "map",
    "chart",
    "coordinate",
    "vector",
    "trajectory",
    "course",
    "heading",
    "waypoint",
    "anchor",
    "harbor",
    "dock",
    "port",
    "bridge",
    "passage",
    "channel",
]
MATHEMATICAL_WORDS = [
    "matrix",
    "vector",
    "tensor",
    "algorithm",
    "equation",
    "formula",
    "calculate",
    "compute",
    "analyze",
    "measure",
    "quantify",
    "derive",
    "integrate",
    "differentiate",
    "optimize",
    "minimize",
    "maximize",
    "probability",
    "statistics",
    "variance",
    "correlation",
    "regression",
]
DUALISTIC_WORDS = [
    "dual",
    "binary",
    "toggle",
    "switch",
    "flip",
    "mirror",
    "reflect",
    "opposite",
    "inverse",
    "complement",
    "parallel",
    "balance",
    "harmony",
    "symmetry",
    "synchronize",
    "phase",
    "oscillate",
    "resonate",
    "align",
    "polar",
    "magnetic",
    "electric",
    "positive",
    "negative",
    "neutral",
]
ENTROPY_WORDS = [
    "chaos",
    "random",
    "disorder",
    "turbulence",
    "volatility",
    "noise",
    "fluctuation",
    "variance",
    "deviation",
    "scatter",
    "dispersion",
    "unpredictable",
    "stochastic",
    "fractal",
    "complex",
    "dynamic",
    "emergence",
    "pattern",
    "structure",
    "order",
    "organization",
]


class EnhancedLanternCore:
    """
    Enhanced Lantern Core with integrated word library and glyph processing.
    """

    def __init__(self):
        """Initialize Enhanced Lantern Core."""
        self.word_categories = {
            "profit_words": PROFIT_WORDS,
            "navigation_words": NAVIGATION_WORDS,
            "mathematical_words": MATHEMATICAL_WORDS,
            "dualistic_words": DUALISTIC_WORDS,
            "entropy_words": ENTROPY_WORDS,
        }

        self.bit_gates = {
            "0": BitGate(BitGateType.NULL_VECTOR, "⚫", 0.8),  # Dampening effect
            "1": BitGate(BitGateType.LOW_TIER, "🟡", 1.0),  # Neutral
            "10": BitGate(BitGateType.MID_TIER, "🟠", 1.15),  # Slight boost
            "11": BitGate(BitGateType.PEAK_TIER, "🔴", 1.3),  # Strong boost
        }

        self.entropy_cache = {}
        self.tagging_thresholds = {
            "high_confidence": 0.8,
            "low_confidence": 0.4,
            "strong_signal": 0.75,
            "high_volatility": 0.6,
            "low_volatility": 0.2,
            "strong_trend": 0.7,
            "historical_confirmation": 0.1,
            "historical_hesitation": -0.1,
        }
        self.dynamic_keywords = defaultdict(list)
        logger.info("Enhanced Lantern Core with Word Library initialized")

    def map_btc_price_to_word(self, btc_price: float) -> Dict[str, Any]:
        """Map BTC price to word entropy for correlation."""
        price_str = f"{btc_price:.2f}"
        price_hash = hashlib.sha256(price_str.encode()).hexdigest()

        hash_int = int(price_hash[:4], 16)
        category_index = hash_int % len(self.word_categories)
        category_name = list(self.word_categories.keys())[category_index]

        words = self.word_categories[category_name]
        word_index = (hash_int // len(self.word_categories)) % len(words)
        selected_word = words[word_index]

        return {
            "btc_price": btc_price,
            "selected_word": selected_word,
            "category": category_name,
            "price_hash": price_hash[:16],
        }

    def process_through_bit_gate(self, word: str, score: float, decision: str) -> Tuple[float, str, BitGate]:
        """Maps a word to a bit pattern and processes a score through the corresponding gate."""
        word_hash = hashlib.sha256(word.encode()).hexdigest()
        hash_int = int(word_hash[:2], 16)

        if hash_int < 64:
            bit_pattern = "0"
        elif hash_int < 128:
            bit_pattern = "1"
        elif hash_int < 192:
            bit_pattern = "10"
        else:
            bit_pattern = "11"

        bit_gate = self.bit_gates[bit_pattern]
        processed_score, processed_decision = bit_gate.process_state(score, decision)

        return processed_score, processed_decision, bit_gate

    def generate_tags(
        self, thought_vector: ThoughtVector, market_data: Dict[str, Any], processing_context: Dict[str, Any] = None
    ) -> List[BrainTag]:
        """
        Analyzes a ThoughtVector and its context to generate a list of BrainTags.
        """
        tags = []
        if processing_context is None:
            processing_context = {}

        # 1. Thermal State Tags
        if thought_vector.thermal_state:
            tags.append(BrainTag[f"THERMAL_{thought_vector.thermal_state.upper()}"])

        # 2. Cognitive State Tags
        if thought_vector.state.value == "alif":
            tags.append(BrainTag.ALIF_DOMINANT)
            tags.append(BrainTag.ALIF_ACTIVATED)

            # Add ALIF decision-specific tags
            if hasattr(thought_vector, "alif_decision") and thought_vector.alif_decision:
                if "enhance" in thought_vector.alif_decision.lower():
                    tags.append(BrainTag.ALIF_ENHANCE_SIGNAL)
                elif "maintain" in thought_vector.alif_decision.lower():
                    tags.append(BrainTag.ALIF_MAINTAIN_SIGNAL)
                elif "attenuate" in thought_vector.alif_decision.lower():
                    tags.append(BrainTag.ALIF_ATTENUATE_SIGNAL)
                elif "halt" in thought_vector.alif_decision.lower():
                    tags.append(BrainTag.ALIF_EMERGENCY_HALT)

        elif thought_vector.logical_score > thought_vector.intuitive_score:
            tags.append(BrainTag.LOGIC_DOMINANT)
        else:
            tags.append(BrainTag.INTUITION_DOMINANT)

        # 3. Confidence Tags
        if thought_vector.confidence > self.tagging_thresholds["high_confidence"]:
            tags.append(BrainTag.HIGH_CONFIDENCE)
        elif thought_vector.confidence < self.tagging_thresholds["low_confidence"]:
            tags.append(BrainTag.LOW_CONFIDENCE)

        # 4. Signal Strength Tags
        if abs(thought_vector.combined_score) > self.tagging_thresholds["strong_signal"]:
            if thought_vector.combined_score > 0:
                tags.append(BrainTag.STRONG_SIGNAL_BUY)
            else:
                tags.append(BrainTag.STRONG_SIGNAL_SELL)
        else:
            tags.append(BrainTag.NEUTRAL_SIGNAL)

        # 5. Historical Consultation Tags
        if thought_vector.historical_adjustment > self.tagging_thresholds["historical_confirmation"]:
            tags.append(BrainTag.HISTORICAL_CONFIRMATION)
        elif thought_vector.historical_adjustment < self.tagging_thresholds["historical_hesitation"]:
            tags.append(BrainTag.HISTORICAL_HESITATION)
        else:
            tags.append(BrainTag.NO_HISTORY_AVAILABLE)

        # 6. Bias Detection and Mitigation Tags
        if thought_vector.bias_mitigated:
            tags.append(BrainTag.BIAS_DETECTED)
            tags.append(BrainTag.BIAS_MITIGATED)

        # 7. Market Condition Tags (from market_data)
        if market_data:
            volatility = market_data.get("volatility", 0.0)
            if volatility > self.tagging_thresholds["high_volatility"]:
                tags.append(BrainTag.HIGH_VOLATILITY)
            elif volatility < self.tagging_thresholds["low_volatility"]:
                tags.append(BrainTag.LOW_VOLATILITY)

            trend_strength = market_data.get("trend_strength", 0.0)
            if abs(trend_strength) > self.tagging_thresholds["strong_trend"]:
                if trend_strength > 0:
                    tags.append(BrainTag.STRONG_UPTREND)
                else:
                    tags.append(BrainTag.STRONG_DOWNTREND)

        # 8. Whale Activity Tags (if available)
        if market_data and "whale_activity" in market_data:
            whale_data = market_data["whale_activity"]
            if whale_data.get("consensus") == "buy":
                tags.append(BrainTag.WHALE_CONSENSUS_BUY)
            elif whale_data.get("consensus") == "sell":
                tags.append(BrainTag.WHALE_CONSENSUS_SELL)
            
            if whale_data.get("activity_level", 0) > 0.7:
                tags.append(BrainTag.WHALE_ACTIVITY_HIGH)

        # 9. Bit Gate Tags (from processing context)
        if processing_context and "bit_gate" in processing_context:
            bit_gate = processing_context["bit_gate"]
            if bit_gate.gate_type == BitGateType.NULL_VECTOR:
                tags.append(BrainTag.BIT_GATE_NULL)
            elif bit_gate.gate_type == BitGateType.LOW_TIER:
                tags.append(BrainTag.BIT_GATE_LOW)
            elif bit_gate.gate_type == BitGateType.MID_TIER:
                tags.append(BrainTag.BIT_GATE_MID)
            elif bit_gate.gate_type == BitGateType.PEAK_TIER:
                tags.append(BrainTag.BIT_GATE_PEAK)

        # 10. Entropy Word Tags (from processing context)
        if processing_context and "entropy_word" in processing_context:
            word = processing_context["entropy_word"]
            if word in PROFIT_WORDS:
                tags.append(BrainTag.ENTROPY_WORD_PROFIT)
            elif word in NAVIGATION_WORDS:
                tags.append(BrainTag.ENTROPY_WORD_NAV)
            elif word in MATHEMATICAL_WORDS:
                tags.append(BrainTag.ENTROPY_WORD_MATH)
            elif word in DUALISTIC_WORDS:
                tags.append(BrainTag.ENTROPY_WORD_DUAL)
            elif word in ENTROPY_WORDS:
                tags.append(BrainTag.ENTROPY_WORD_CHAOS)

        # 11. News-Derived Tags (from processing context)
        if processing_context:
            news_sentiment_type = processing_context.get("news_sentiment_type")
            news_impact_level = processing_context.get("news_impact_level")
            news_linguistic_bit_state = processing_context.get("news_linguistic_bit_state")
            news_linguistic_weight = processing_context.get("news_linguistic_weight")

            if news_sentiment_type:
                if news_sentiment_type == "positive":
                    tags.append(BrainTag.NEWS_POSITIVE)
                    if news_impact_level == "critical":
                        tags.append(BrainTag.NEWS_HIGH_IMPACT_POSITIVE)
                elif news_sentiment_type == "negative":
                    tags.append(BrainTag.NEWS_NEGATIVE)
                    if news_impact_level == "critical":
                        tags.append(BrainTag.NEWS_HIGH_IMPACT_NEGATIVE)
                elif news_sentiment_type == "neutral":
                    tags.append(BrainTag.NEWS_NEUTRAL)
                elif news_sentiment_type == "mixed":
                    tags.append(BrainTag.NEWS_MIXED_SENTIMENT)
            
            if news_impact_level == "critical":
                tags.append(BrainTag.NEWS_CRITICAL_IMPACT)
            elif news_impact_level == "high":
                tags.append(BrainTag.NEWS_HIGH_IMPACT)
            
            if news_linguistic_bit_state is not None:
                if news_linguistic_bit_state == 0b00:
                    tags.append(BrainTag.NEWS_BIT_STATE_NULL)
                elif news_linguistic_bit_state == 0b01:
                    tags.append(BrainTag.NEWS_BIT_STATE_GHOST_ENTRY)
                elif news_linguistic_bit_state == 0b10:
                    tags.append(BrainTag.NEWS_BIT_STATE_MEMORY_LOCK)
                elif news_linguistic_bit_state == 0b11:
                    tags.append(BrainTag.NEWS_BIT_STATE_PROFIT_VECTOR)
            
            if news_linguistic_weight is not None and news_linguistic_weight > 0.7:
                tags.append(BrainTag.NEWS_HIGH_L_WEIGHT)
            elif news_linguistic_weight is not None and news_linguistic_weight < 0.3:
                tags.append(BrainTag.NEWS_LOW_L_WEIGHT)

        return tags

    def validate_thought_vector(self, thought_vector: ThoughtVector) -> Dict[str, Any]:
        """
        Validates a ThoughtVector and returns validation results.
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "confidence_score": 0.0,
        }

        # Check for required fields
        required_fields = [
            "timestamp", "state", "thermal_state", "logical_score",
            "intuitive_score", "combined_score", "decision", "confidence"
        ]

        for field in required_fields:
            if not hasattr(thought_vector, field):
                validation_results["errors"].append(f"Missing required field: {field}")
                validation_results["is_valid"] = False

        # Validate score ranges
        if hasattr(thought_vector, "logical_score"):
            if not -1.0 <= thought_vector.logical_score <= 1.0:
                validation_results["warnings"].append("Logical score out of expected range [-1, 1]")

        if hasattr(thought_vector, "intuitive_score"):
            if not -1.0 <= thought_vector.intuitive_score <= 1.0:
                validation_results["warnings"].append("Intuitive score out of expected range [-1, 1]")

        if hasattr(thought_vector, "confidence"):
            if not 0.0 <= thought_vector.confidence <= 1.0:
                validation_results["warnings"].append("Confidence out of expected range [0, 1]")

        # Calculate confidence score based on validation
        base_confidence = getattr(thought_vector, "confidence", 0.0)
        warning_penalty = len(validation_results["warnings"]) * 0.1
        error_penalty = len(validation_results["errors"]) * 0.3

        validation_results["confidence_score"] = max(0.0, base_confidence - warning_penalty - error_penalty)

        return validation_results

    def get_thought_vector_summary(self, thought_vector: ThoughtVector) -> Dict[str, Any]:
        """
        Generates a comprehensive summary of a ThoughtVector.
        """
        summary = {
            "timestamp": thought_vector.timestamp,
            "state": thought_vector.state.value,
            "thermal_state": thought_vector.thermal_state,
            "scores": {
                "logical": thought_vector.logical_score,
                "intuitive": thought_vector.intuitive_score,
                "combined": thought_vector.combined_score,
                "confidence": thought_vector.confidence,
            },
            "decision": thought_vector.decision,
            "tags": [tag.value for tag in thought_vector.tags],
            "validation": self.validate_thought_vector(thought_vector),
        }

        # Add ALIF-specific information if available
        if hasattr(thought_vector, "alif_score") and thought_vector.alif_score != 0.0:
            summary["alif"] = {
                "score": thought_vector.alif_score,
                "decision": getattr(thought_vector, "alif_decision", None),
                "feedback": getattr(thought_vector, "alif_feedback", None),
            }

        return summary

    def ingest_external_keywords(self, text: str, source_category: str = "news_keywords", impact_score: float = 0.5):
        """
        Ingest and categorize external keywords/phrases from sources like news.
        Dynamically updates word categories based on external linguistic input.
        
        Args:
            text: The text content to extract keywords from.
            source_category: A category for these new keywords (e.g., "news_keywords", "market_alerts").
            impact_score: A score reflecting the importance/impact of the source text (0.0-1.0).
        """
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words and short words
        stop_words = {"the", "a", "an", "is", "and", "of", "to", "in", "for", "with", "on", "at", "by", "from", "be", "are", "as", "it", "do", "he", "she", "they", "we", "you", "that", "this", "will", "can", "have", "has", "had", "not", "but", "or", "if", "then", "than"}
        new_words = [w for w in words if w not in stop_words and len(w) > 2]

        # Add new words to dynamic categories, weighting by impact
        for word in new_words:
            if word not in self.dynamic_keywords[source_category]:
                self.dynamic_keywords[source_category].append(word)
            # Simple dynamic update for existing categories, or create new ones
            if word not in self.word_categories.get(source_category, []):
                if source_category not in self.word_categories:
                    self.word_categories[source_category] = []
                self.word_categories[source_category].append(word)
        
        # Simple mechanism to boost relevant word categories based on impact
        if source_category in self.word_categories:
            # Example: sort or prioritize words by hypothetical impact
            self.word_categories[source_category].sort(key=lambda x: impact_score, reverse=True) # This is a placeholder for actual weighting logic
        
        logger.debug(f"Ingested {len(new_words)} new keywords into {source_category} category with impact {impact_score}")


# Mock class for testing when ThoughtVector is not available
class MockThoughtVector:
    def __init__(self):
        self.timestamp = time.time()
        self.state = DualisticState.LOGICAL
        self.thermal_state = "warm"
        self.logical_score = 0.5
        self.intuitive_score = 0.3
        self.historical_adjustment = 0.1
        self.combined_score = 0.4
        self.decision = "hold"
        self.confidence = 0.7
        self.thought_hash_32bit = "mock_hash"
        self.tags = []
