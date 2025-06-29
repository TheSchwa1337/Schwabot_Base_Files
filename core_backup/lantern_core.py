# -*- coding: utf-8 -*-
"""
Enhanced Lantern Core with integrated word library and glyph processing.

This core module focuses on linguistic and semantic analysis to inform
trading decisions. It includes functionalities for mapping BTC price to
word entropy, processing signals through a 'bit gate' based on strategic
tiers, and generating cognitive 'brain tags' for enhanced decision context.

Integrates with: dualistic_thought_engines.py, unified_math_system.py,
                  linguistic_glyph_engine.py
"""

import hashlib
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
        """
        Process a decision score through a bit gate based on word correlation.

        Args:
            word: The word correlated to BTC price (e.g., 'profit', 'chaos').
            score: The raw decision score (0-1).
            decision: The raw decision string (e.g., 'BUY', 'SELL', 'HOLD').

        Returns:
            A tuple containing the processed score, decision, and the active BitGate.
        """
        # Simple mapping of word to a bit gate type (this can be made more complex)
        if word in PROFIT_WORDS:
            bit_key = "11"  # Strong boost for profit words
        elif word in NAVIGATION_WORDS:
            bit_key = "10"  # Slight boost for navigation words
        elif word in MATHEMATICAL_WORDS:
            bit_key = "1"  # Neutral for mathematical words
        elif word in DUALISTIC_WORDS:
            bit_key = "10"  # Slight boost for dualistic words
        elif word in ENTROPY_WORDS:
            bit_key = "0"  # Dampening for entropy words
        else:
            bit_key = "1"  # Default to neutral

        active_bit_gate = self.bit_gates.get(bit_key, self.bit_gates["1"])  # Default to neutral gate

        processed_score, processed_decision = active_bit_gate.process_state(score, decision)

        logger.debug(
            f"Processed through bit gate: Word='{word}', Raw Score={score:.2f}, "
            f"Processed Score={processed_score:.2f}, Gate Type={active_bit_gate.gate_type.value}"
        )
        return processed_score, processed_decision, active_bit_gate

    def analyze_linguistic_cues(self, text: str) -> float:
        """Analyze linguistic cues from text (e.g., news headlines) to derive a sentiment score."""
        if not text:
            return 0.0

        text_lower = text.lower()
        sentiment_score = 0.0
        total_words = 0

        for category, words in self.word_categories.items():
            for word in words:
                if word in text_lower:
                    total_words += 1
                    if category == "profit_words":
                        sentiment_score += 0.1
                    elif category == "entropy_words":
                        sentiment_score -= 0.1

        if total_words > 0:
            # Normalize sentiment score
            sentiment_score = sentiment_score / total_words

        logger.debug(f"Linguistic cue analysis for '{text[:30]}...': Score={sentiment_score:.2f}")
        return sentiment_score

    def generate_tags(self, thought_vector: ThoughtVector, market_data: Dict[str, Any], processing_context: Dict[str, Any] = None) -> List[BrainTag]:
        """
        Analyzes a ThoughtVector and its context to generate a list of BrainTags.
        """
        tags: List[BrainTag] = []

        if processing_context is None:
            processing_context = {}

        # Thermal State Tags
        if thought_vector.thermal_state == "cool":
            tags.append(BrainTag.THERMAL_COOL)
        elif thought_vector.thermal_state == "warm":
            tags.append(BrainTag.THERMAL_WARM)
        elif thought_vector.thermal_state == "hot":
            tags.append(BrainTag.THERMAL_HOT)
        elif thought_vector.thermal_state == "critical":
            tags.append(BrainTag.THERMAL_CRITICAL)

        # Cognitive State Tags
        if thought_vector.state == DualisticState.LOGICAL:
            tags.append(BrainTag.LOGIC_DOMINANT)
        elif thought_vector.state == DualisticState.INTUITIVE:
            tags.append(BrainTag.INTUITION_DOMINANT)

        if thought_vector.logical_score > 0.7 and thought_vector.intuitive_score > 0.7:
            tags.append(BrainTag.LOGIC_INTUITION_AGREEMENT)
        elif thought_vector.logical_score < 0.3 and thought_vector.intuitive_score > 0.7:
            tags.append(BrainTag.LOGIC_INTUITION_CONFLICT)  # Example conflict scenario

        # Confidence & Score Tags
        if thought_vector.confidence > self.tagging_thresholds["high_confidence"]:
            tags.append(BrainTag.HIGH_CONFIDENCE)
        elif thought_vector.confidence < self.tagging_thresholds["low_confidence"]:
            tags.append(BrainTag.LOW_CONFIDENCE)

        if thought_vector.decision == "BUY" and thought_vector.confidence > self.tagging_thresholds["strong_signal"]:
            tags.append(BrainTag.STRONG_SIGNAL_BUY)
        elif thought_vector.decision == "SELL" and thought_vector.confidence > self.tagging_thresholds["strong_signal"]:
            tags.append(BrainTag.STRONG_SIGNAL_SELL)
        elif thought_vector.decision == "HOLD":
            tags.append(BrainTag.NEUTRAL_SIGNAL)

        # Historical Consultation Tags
        if "historical_confirmation" in thought_vector.historical_consultation:
            if thought_vector.historical_consultation["historical_confirmation"] > self.tagging_thresholds["historical_confirmation"]:
                tags.append(BrainTag.HISTORICAL_CONFIRMATION)
            else:
                tags.append(BrainTag.HISTORICAL_HESITATION) # Or NO_HISTORY_AVAILABLE

        # Bias & Mitigation Tags
        if thought_vector.bias_mitigated:
            tags.append(BrainTag.BIAS_DETECTED)
            tags.append(BrainTag.BIAS_MITIGATED)

        # Market Condition Tags (using market_data)
        current_price = market_data.get("current_price", 0.0)
        previous_price = market_data.get("previous_price", current_price) # Need a previous price from data
        price_change_percent = ((current_price - previous_price) / previous_price) if previous_price != 0 else 0

        if abs(price_change_percent) > self.tagging_thresholds["high_volatility"]:
            tags.append(BrainTag.HIGH_VOLATILITY)
        elif abs(price_change_percent) < self.tagging_thresholds["low_volatility"]:
            tags.append(BrainTag.LOW_VOLATILITY)

        if price_change_percent > self.tagging_thresholds["strong_trend"]:
            tags.append(BrainTag.STRONG_UPTREND)
        elif price_change_percent < -self.tagging_thresholds["strong_trend"]:
            tags.append(BrainTag.STRONG_DOWNTREND)

        # Bit Gate Tags from processing_context
        bit_gate = processing_context.get("bit_gate")
        if bit_gate:
            if bit_gate.gate_type == BitGateType.NULL_VECTOR:
                tags.append(BrainTag.BIT_GATE_NULL)
            elif bit_gate.gate_type == BitGateType.LOW_TIER:
                tags.append(BrainTag.BIT_GATE_LOW)
            elif bit_gate.gate_type == BitGateType.MID_TIER:
                tags.append(BrainTag.BIT_GATE_MID)
            elif bit_gate.gate_type == BitGateType.PEAK_TIER:
                tags.append(BrainTag.BIT_GATE_PEAK)

        # ALIF Tags
        if thought_vector.alif_feedback:
            if thought_vector.alif_feedback.feedback_type == "activate_strategy": # Assuming a mapping
                tags.append(BrainTag.ALIF_ACTIVATED)
            elif thought_vector.alif_feedback.feedback_type == "error_correction_applied":
                tags.append(BrainTag.ALIF_CORRECTION)

        logger.debug(f"Generated {len(tags)} tags for ThoughtVector.")
        return tags


def main():
    """Main function to demonstrate EnhancedLanternCore functionality."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    core = EnhancedLanternCore()

    print("\n--- Enhanced Lantern Core Demo ---")

    # Demo 1: BTC Price to Word Mapping
    btc_price = 52345.67
    word_map = core.map_btc_price_to_word(btc_price)
    print(f"\nBTC Price {btc_price} maps to: ")
    print(f"  Selected Word: {word_map["selected_word"]}")
    print(f"  Category: {word_map["category"]}")
    print(f"  Price Hash: {word_map["price_hash"]}")

    # Demo 2: Process through Bit Gate
    word_for_gate = "profit"
    score_for_gate = 0.75
    decision_for_gate = "BUY"
    processed_score, processed_decision, active_gate = core.process_through_bit_gate(
        word_for_gate, score_for_gate, decision_for_gate
    )
    print(f"\nProcessing '{word_for_gate}' through Bit Gate:")
    print(f"  Raw Score: {score_for_gate:.2f}, Raw Decision: {decision_for_gate}")
    print(f"  Processed Score: {processed_score:.2f}, Processed Decision: {processed_decision}")
    print(f"  Active Bit Gate: {active_gate.gate_type.value} ({active_gate.emoji})")

    # Demo 3: Analyze Linguistic Cues
    news_headline = "Strong market rally today, profit expected!"
    sentiment = core.analyze_linguistic_cues(news_headline)
    print(f"\nSentiment analysis for '{news_headline}': {sentiment:.2f}")

    # Demo 4: Generate Tags for a ThoughtVector (simplified)
    from core.thought_types import ThoughtVector, DualisticState, AlifFeedbackData

    mock_thought_vector = ThoughtVector(
        timestamp=time.time(),
        state=DualisticState.LOGICAL,
        thermal_state="hot",
        logical_score=0.9,
        intuitive_score=0.6,
        historical_adjustment=0.05,
        combined_score=0.85,
        decision="BUY",
        confidence=0.88,
        thought_hash_32bit="abcdef1234567890",
        bias_mitigated=None,
        historical_consultation="historical_confirmation",
        alif_feedback=AlifFeedbackData(feedback_type="activate_strategy", message="ALIF recommends buy"),
        alif_score=0.9,
        alif_decision="ACTIVATE_STRATEGY",
        profit_prediction=0.0,
    )
    mock_market_data = {"current_price": 53000, "previous_price": 52500, "dominant_indicator": "RSI"}
    mock_processing_context = {"bit_gate": active_gate}

    tags = core.generate_tags(mock_thought_vector, mock_market_data, mock_processing_context)
    print("\nGenerated Brain Tags:")
    for tag in tags:
        print(f"- {tag.value}")


if __name__ == "__main__":
    main() 