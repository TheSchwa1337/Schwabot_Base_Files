# -*- coding: utf-8 -*-
"""
Dualistic Thought Engines.

Advanced AI-driven cognitive architectures for enhanced decision-making
within the Schwabot trading system. These engines operate on a dualistic
principle, integrating both logical and intuitive thought processes to
optimize trading strategies, manage risk, and identify complex market patterns.

Mathematical Foundation:
    - Dualistic Logic Gates: (A and B) or (C XOR D)
    - Intuitive Pattern Recognition: f(x) = ∑_{i=1}^{n} w_i * g(x_i)
    - Risk-Adjusted Decision Calculus: R = ∫ (Profit - alpha * Risk) dt
    - Cognitive Bias Mitigation: B_mit = (1 - epsilon) * B_raw
    - 32-bit thought vectorization with thermal state integration
    - Adaptive learning mechanisms based on real-time market feedback and
      historical analysis
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .thought_types import (
    AlifFeedback,
    AlifFeedbackData,
    CognitiveBias,
    DualisticState,
    ThoughtVector,
)
from .lantern_core import EnhancedLanternCore
from .schwafit_core import SchwafitCore
from .unified_math_system import UnifiedMathSystem
from .hash_relay_system import hash_relay_system
from .linguistic_glyph_engine import linguistic_engine, process_linguistic_command
from .lantern_news_intelligence_bridge import LanternNewsIntelligenceBridge, NewsItem

# Initialize logging
logger = logging.getLogger(__name__)

# Thermal state constants for 32-bit operations
COOL = "cool"
WARM = "warm"
HOT = "hot"
CRITICAL = "critical"


class DualisticThoughtEngines:
    """
    Advanced AI-driven cognitive architectures for enhanced decision-making.

    Integrates logical and intuitive thought processes to optimize trading
    strategies, manage risk, and identify complex market patterns with 32-bit
    thought vectorization and thermal awareness.
    """

    def __init__(self: "DualisticThoughtEngines") -> None:
        """Initialize the dualistic thought engines."""
        self.unified_math = UnifiedMathSystem()
        self.schwafit_core = SchwafitCore()
        self.lantern_core = EnhancedLanternCore()
        self.current_state = DualisticState.LOGICAL
        self.thought_history: List[ThoughtVector] = []
        self.bias_detection_threshold = 0.6  # Threshold for detecting bias
        self.adaptation_rate = 0.5  # Rate at which the engine adapts
        self.bias_mitigation_strength = 0.1  # How strongly biases are mitigated
        self.historical_influence_factor = 0.15  # How much history affects decisions

        # ALIF-specific configuration
        self.alif_enabled = True
        self.alif_weights = {
            "volume": 0.4,
            "resonance": 0.3,
            "ai_feedback": 0.2,
            "error_correction": 0.1,
        }
        self.alif_threshold = 0.7  # Threshold for ALIF state activation
        self.alif_memory_size = 1000  # Size of ALIF memory buffer
        self.alif_adaptation_rate = 0.2  # ALIF-specific adaptation rate

        # ALIF memory and feedback storage
        self.alif_memory: List[Dict[str, Any]] = []
        self.alif_feedback_history: List[AlifFeedback] = []
        self.alif_error_log: List[Dict[str, Any]] = []
        self.alif_market_memory: Dict[str, float] = {}

        # Performance metrics
        self.total_decisions = 0
        self.successful_decisions = 0
        self.logical_decisions = 0
        self.intuitive_decisions = 0
        self.alif_decisions = 0
        self.alif_activations = 0
        self.alif_corrections = 0

        logger.info("Dualistic Thought Engines initialized with ALIF support.")

    def process_market_data(
        self: "DualisticThoughtEngines", market_data: Dict[str, Any], thermal_state: str = WARM
    ) -> ThoughtVector:
        """Process real-time market data through dualistic thought engines.

        Mathematical: Decision = BitGate(Word(Price)) * (Logical(data) and
        Intuitive(data)) * f(History) * f(Thermal)

        Args:
            market_data: Dictionary containing various market data points.
            thermal_state: The current thermal state of the system.

        Returns:
            A ThoughtVector representing the engine's decision and analysis.
        """
        self.total_decisions += 1
        start_time = time.time()
        self.current_thermal_state = thermal_state

        try:
            # Step 1: Logical Analysis
            logical_score, logical_decision = self._perform_logical_analysis(market_data)
            self.logical_decisions += 1

            # Step 2: Intuitive Analysis
            intuitive_score, intuitive_decision = self._perform_intuitive_analysis(market_data)
            self.intuitive_decisions += 1

            # Step 3: ALIF Analysis (Adaptive Learning Interference Filter)
            ai_feedback = market_data.get("ai_feedback", [])
            error_logs = market_data.get("error_logs", [])
            alif_score, alif_decision, alif_feedback = self._perform_alif_analysis(market_data, ai_feedback, error_logs)

            # Step 4: Consult Historical Patterns (The "Hesitation" Mechanism)
            historical_adjustment, historical_consultation = self._consult_historical_patterns(market_data)

            # Step 5: Integrate Dualistic Logic with ALIF
            combined_score, pre_processed_decision = self._integrate_dualistic_logic_with_alif(
                logical_score,
                logical_decision,
                intuitive_score,
                intuitive_decision,
                alif_score,
                alif_decision,
                alif_feedback,
                historical_adjustment,
                market_data,
            )

            # Step 6: Route through Lantern Core Bit Gate for final processing and intensity adjustment
            word_map = self.lantern_core.map_btc_price_to_word(market_data.get("current_price", 0))
            final_score, final_decision, bit_gate = self.lantern_core.process_through_bit_gate(
                word_map["selected_word"], combined_score, pre_processed_decision
            )

            processing_context = {"bit_gate": bit_gate, "word_category": word_map["category"]}

            # Step 7: Cognitive Bias Mitigation on the final score
            bias_mitigated = self._mitigate_cognitive_bias(market_data, final_score)

            # Step 8: Adapt and Learn from the final decision
            self._adapt_and_learn(final_score, final_decision, market_data)

            # Generate 32-bit thought vector hash
            thought_hash_32bit = self._generate_input_hash(market_data)

            # Determine final state based on which analysis dominated
            final_state = self._determine_final_state(logical_score, intuitive_score, alif_score)

            thought_vector = ThoughtVector(
                timestamp=time.time(),
                state=final_state,
                thermal_state=self.current_thermal_state,
                logical_score=logical_score,
                intuitive_score=intuitive_score,
                historical_adjustment=historical_adjustment,
                combined_score=final_score,
                decision=final_decision,
                confidence=final_score,
                bias_mitigated=bias_mitigated,
                thought_hash_32bit=thought_hash_32bit,
                historical_consultation=historical_consultation,
                alif_feedback=alif_feedback,
                alif_score=alif_score,
                alif_decision=alif_decision,
            )

            # Step 9: Generate Brain Tags with Lantern Core using the final context
            thought_vector.tags = self.lantern_core.generate_tags(thought_vector, market_data, processing_context)

            self.thought_history.append(thought_vector)
            # Limit history size to prevent memory issues
            if len(self.thought_history) > 5000:
                self.thought_history = self.thought_history[-2500:]

            self.successful_decisions += 1
            logger.info(f"ThoughtVector generated: State={final_state.value}, Confidence={final_score:.2f}")
            return thought_vector

        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return ThoughtVector(
                timestamp=time.time(),
                state=DualisticState.ERROR,
                thermal_state=self.current_thermal_state,
                logical_score=0.0,
                intuitive_score=0.0,
                historical_adjustment=0.0,
                combined_score=0.0,
                decision="ERROR",
                confidence=0.0,
                bias_mitigated=False,
                thought_hash_32bit="",
                historical_consultation="",
                alif_feedback=AlifFeedbackData(feedback_type="error", message=str(e)),
                alif_score=0.0,
                alif_decision="ERROR",
            )

    def _perform_logical_analysis(self, market_data: Dict[str, Any]) -> Tuple[float, str]:
        """Perform logical, rule-based analysis of market data."""
        price = market_data.get("current_price", 0.0)
        volume = market_data.get("current_volume", 0.0)
        news_sentiment = market_data.get("news_sentiment", 0.0)  # -1 to 1
        time_of_day_factor = market_data.get("time_of_day_factor", 0.0)

        # Example logical rules
        score = 0.5
        decision = "HOLD"

        if price > market_data.get("resistance_level", price * 1.01) and volume > 1e6:
            score += 0.2
            decision = "SELL"
        elif price < market_data.get("support_level", price * 0.99) and volume > 1e6:
            score += 0.2
            decision = "BUY"

        # Adjust based on news sentiment
        score += news_sentiment * 0.1

        # Adjust based on time of day (e.g., higher volatility times)
        score += time_of_day_factor * 0.05

        score = np.clip(score, 0, 1.0)  # Ensure score is between 0 and 1
        logger.debug(f"Logical analysis: Score={score:.2f}, Decision={decision}")
        return float(score), decision

    def _perform_intuitive_analysis(self, market_data: Dict[str, Any]) -> Tuple[float, str]:
        """Perform intuitive, pattern-based analysis using Schwafit Core and Lantern Core."""
        # Schwafit Core for fractal pattern recognition
        fractal_pattern_score = self.schwafit_core.analyze_fractal_patterns(market_data)

        # Lantern Core for sentiment and linguistic cues
        linguistic_cue_score = self.lantern_core.analyze_linguistic_cues(market_data.get("recent_news_headlines", ""))

        # Combine intuitively
        intuitive_score = (fractal_pattern_score + linguistic_cue_score) / 2.0
        decision = "HOLD"

        if intuitive_score > 0.7:
            decision = "BUY"
        elif intuitive_score < 0.3:
            decision = "SELL"

        logger.debug(f"Intuitive analysis: Score={intuitive_score:.2f}, Decision={decision}")
        return float(intuitive_score), decision

    def _perform_alif_analysis(
        self, market_data: Dict[str, Any], ai_feedback: List[Dict[str, Any]], error_logs: List[Dict[str, Any]]
    ) -> Tuple[float, str, AlifFeedbackData]:
        """Perform Adaptive Learning Interference Filter (ALIF) analysis."""
        if not self.alif_enabled:
            return 0.5, "N/A", AlifFeedbackData(feedback_type="disabled", message="ALIF disabled")

        alif_score = 0.5
        alif_decision = "HOLD"
        feedback_message = "No specific ALIF feedback"
        feedback_type = "info"

        # Process AI feedback
        if ai_feedback:
            # Simple aggregation of AI sentiment
            total_sentiment = sum(item.get("sentiment", 0) for item in ai_feedback)
            avg_sentiment = total_sentiment / len(ai_feedback)
            alif_score += avg_sentiment * self.alif_weights["ai_feedback"]
            feedback_message = f"Processed {len(ai_feedback)} AI feedback items"
            feedback_type = "ai_feedback_processed"

        # Process error logs for error correction
        if error_logs:
            error_count = len(error_logs)
            alif_score -= error_count * self.alif_weights["error_correction"]
            feedback_message = f"Processed {error_count} error logs"
            feedback_type = "error_correction_applied"

        # Incorporate market memory and resonance
        volume_signal = market_data.get("current_volume", 0.0)
        resonance_signal = market_data.get("resonance_level", 0.0)

        alif_score += (volume_signal / 1e7) * self.alif_weights["volume"]  # Normalize volume
        alif_score += resonance_signal * self.alif_weights["resonance"]

        # Decision based on ALIF score
        if alif_score > self.alif_threshold:
            alif_decision = "ACTIVATE_STRATEGY"
            self.alif_activations += 1
        elif alif_score < (1 - self.alif_threshold):
            alif_decision = "DEACTIVATE_STRATEGY"
        else:
            alif_decision = "ADJUST_STRATEGY"

        # Store current market data in ALIF memory
        self.alif_memory.append(market_data)
        if len(self.alif_memory) > self.alif_memory_size:
            self.alif_memory.pop(0)

        logger.debug(f"ALIF analysis: Score={alif_score:.2f}, Decision={alif_decision}")
        return float(alif_score), alif_decision, AlifFeedbackData(feedback_type=feedback_type, message=feedback_message)

    def _consult_historical_patterns(self, market_data: Dict[str, Any]) -> Tuple[float, str]:
        """Consult historical thought patterns for adjustment (The "Hesitation" Mechanism)."""
        historical_adjustment = 0.0
        consultation_message = "No strong historical pattern detected."

        if not self.thought_history:
            return historical_adjustment, consultation_message

        current_price_vector = np.array(list(market_data.values()))  # Simplified price vector

        # Find most similar historical thought vector
        best_similarity = 0.0
        most_similar_thought: Optional[ThoughtVector] = None

        for thought in self.thought_history:
            if thought.thought_hash_32bit:
                similarity = self.unified_math.hash_similarity_score(
                    self._generate_input_hash(market_data), [thought.thought_hash_32bit]
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    most_similar_thought = thought

        if most_similar_thought and best_similarity > self.historical_influence_factor:
            # Adjust based on historical profit/loss
            historical_adjustment = most_similar_thought.profit_prediction * best_similarity
            consultation_message = (
                f"Similar pattern found with confidence {best_similarity:.2f}. "
                f"Historical prediction: {most_similar_thought.profit_prediction:.2f}"
            )
            logger.debug(f"Historical consultation: {consultation_message}")

        return historical_adjustment, consultation_message

    def _integrate_dualistic_logic_with_alif(
        self,
        logical_score: float,
        logical_decision: str,
        intuitive_score: float,
        intuitive_decision: str,
        alif_score: float,
        alif_decision: str,
        alif_feedback: AlifFeedbackData,
        historical_adjustment: float,
        market_data: Dict[str, Any],
    ) -> Tuple[float, str]:
        """Integrate logical, intuitive, and ALIF analysis into a combined decision."""
        # Weighted average of logical and intuitive scores
        combined_score = (logical_score * 0.6) + (intuitive_score * 0.4)  # Example weights

        # Adjust based on ALIF score and decision
        if alif_decision == "ACTIVATE_STRATEGY":
            combined_score = max(combined_score, alif_score)  # Take higher of the two
        elif alif_decision == "DEACTIVATE_STRATEGY":
            combined_score = min(combined_score, alif_score)  # Take lower of the two

        # Apply historical adjustment
        combined_score += historical_adjustment

        # Determine pre-processed decision
        pre_processed_decision = "HOLD"
        if combined_score > 0.7:
            pre_processed_decision = "BUY"
        elif combined_score < 0.3:
            pre_processed_decision = "SELL"

        logger.debug(f"Integrated decision: Score={combined_score:.2f}, Decision={pre_processed_decision}")
        return combined_score, pre_processed_decision

    def _mitigate_cognitive_bias(self, market_data: Dict[str, Any], raw_score: float) -> bool:
        """Mitigate cognitive biases based on predefined patterns or self-correction."""
        # Example: Simple bias detection (e.g., over-reliance on a single indicator)
        bias_detected = False
        if market_data.get("dominant_indicator", "") == "RSI" and raw_score > 0.8:
            # If RSI is dominant and score is high, potentially over-optimistic bias
            if random.random() < self.bias_detection_threshold:  # Random chance of detection
                raw_score *= (1 - self.bias_mitigation_strength)  # Reduce score
                bias_detected = True
                logger.warning(f"Cognitive bias detected (over-optimism via RSI). Mitigated score: {raw_score:.2f}")

        # Update a bias tracker (simplified)
        self._update_bias_tracker(market_data, bias_detected)

        return bias_detected

    def _update_bias_tracker(self, market_data: Dict[str, Any], bias_detected: bool):
        """Internal tracker for cognitive biases."""
        # This would involve storing bias events, types, and mitigation effectiveness
        pass

    def _adapt_and_learn(self, final_score: float, final_decision: str, market_data: Dict[str, Any]):
        """Adapt the engine's parameters based on the outcome of decisions."""
        # Example: Adjust ALIF weights based on success/failure
        if final_decision != "ERROR":
            if final_score > 0.7:  # Assume a successful decision
                # Increase weights for contributing factors
                for key in self.alif_weights:
                    self.alif_weights[key] = min(1.0, self.alif_weights[key] + self.adaptation_rate * 0.01)
            elif final_score < 0.3:  # Assume a less successful decision
                # Decrease weights
                for key in self.alif_weights:
                    self.alif_weights[key] = max(0.0, self.alif_weights[key] - self.adaptation_rate * 0.005)

        # Store feedback for ALIF
        feedback_type = "success" if final_score > 0.7 else "failure"
        self.alif_feedback_history.append(
            AlifFeedback(
                timestamp=time.time(),
                decision_score=final_score,
                feedback_type=feedback_type,
                metadata=market_data,  # Store relevant market data at time of decision
            )
        )

        # Limit feedback history
        if len(self.alif_feedback_history) > 2000:
            self.alif_feedback_history.pop(0)

        logger.debug(f"Engine adapted. New ALIF weights: {self.alif_weights}")

    def _generate_input_hash(self, market_data: Dict[str, Any]) -> str:
        """Generate a hash from the raw market data input for consistent recall."""
        # Convert market_data to a consistent string representation
        # Sort keys to ensure consistent hash generation
        sorted_items = sorted(market_data.items(), key=lambda x: str(x[0]))
        data_string = json.dumps(sorted_items, separators=(',', ':'))
        return hashlib.sha256(data_string.encode()).hexdigest()

    def _determine_final_state(self, logical_score: float, intuitive_score: float, alif_score: float) -> DualisticState:
        """Determine the final dualistic state based on dominant analysis."""
        scores = {
            DualisticState.LOGICAL: logical_score,
            DualisticState.INTUITIVE: intuitive_score,
            DualisticState.ALIF_ACTIVE: alif_score,  # Consider ALIF as a state when highly influential
        }

        # Determine the state with the highest score
        dominant_state = max(scores, key=scores.get)  # type: ignore

        # If all scores are low, revert to IDLE or a default state
        if all(score < 0.4 for score in scores.values()):
            return DualisticState.IDLE

        return dominant_state

    def get_system_status(self) -> Dict[str, Any]:
        """Get the current operational status and metrics of the Dualistic Thought Engines."""
        return {
            "current_state": self.current_state.value,
            "total_decisions": self.total_decisions,
            "successful_decisions": self.successful_decisions,
            "success_rate": (self.successful_decisions / self.total_decisions) if self.total_decisions > 0 else 0.0,
            "logical_decisions": self.logical_decisions,
            "intuitive_decisions": self.intuitive_decisions,
            "alif_decisions": self.alif_decisions,
            "alif_activations": self.alif_activations,
            "alif_corrections": self.alif_corrections,
            "bias_detection_threshold": self.bias_detection_threshold,
            "adaptation_rate": self.adaptation_rate,
            "bias_mitigation_strength": self.bias_mitigation_strength,
            "historical_influence_factor": self.historical_influence_factor,
            "alif_weights": self.alif_weights,
            "thought_history_size": len(self.thought_history),
            "alif_memory_size": len(self.alif_memory),
            "alif_feedback_history_size": len(self.alif_feedback_history),
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self.total_decisions = 0
        self.successful_decisions = 0
        self.logical_decisions = 0
        self.intuitive_decisions = 0
        self.alif_decisions = 0
        self.alif_activations = 0
        self.alif_corrections = 0
        self.thought_history.clear()
        self.alif_memory.clear()
        self.alif_feedback_history.clear()
        logger.info("Dualistic Thought Engines metrics reset.")


def main() -> None:
    """Main function to demonstrate DualisticThoughtEngines functionality."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize mock dependencies
    class MockUnifiedMathSystem:
        def hash_similarity_score(self, hash1: str, hashes: List[str]) -> float:
            # Simple mock: return high similarity if hash1 is in hashes
            return 0.9 if hash1 in hashes else 0.1

    class MockSchwafitCore:
        def analyze_fractal_patterns(self, market_data: Dict[str, Any]) -> float:
            return 0.7  # Always return 0.7 for demo

    class MockEnhancedLanternCore:
        def map_btc_price_to_word(self, price: float) -> Dict[str, str]:
            return {"selected_word": "mock_word", "category": "mock_category"}

        def process_through_bit_gate(self, word: str, score: float, decision: str) -> Tuple[float, str, str]:
            return score * 1.1, decision, "mock_bit_gate"

        def analyze_linguistic_cues(self, text: str) -> float:
            return 0.6  # Always return 0.6 for demo

        def generate_tags(self, thought_vector: Any, market_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
            return ["mock_tag_1", "mock_tag_2"]

    # Replace actual imports with mocks for demonstration
    import sys

    sys.modules['core.unified_math_system'] = MockUnifiedMathSystem()
    sys.modules['core.schwafit_core'] = MockSchwafitCore()
    sys.modules['core.lantern_core'] = MockEnhancedLanternCore()

    engines = DualisticThoughtEngines()

    print("\n--- Dualistic Thought Engines Demo ---")

    # Simulate market data
    market_data_1 = {
        "current_price": 50000.0,
        "current_volume": 5e6,
        "news_sentiment": 0.8,
        "time_of_day_factor": 0.2,
        "resistance_level": 51000.0,
        "support_level": 49000.0,
        "ai_feedback": [{"sentiment": 0.9}, {"sentiment": 0.7}],
        "error_logs": [],
        "resonance_level": 0.5,
        "recent_news_headlines": "Bullish market sentiment dominates as BTC soars.",
        "dominant_indicator": "RSI",
    }

    market_data_2 = {
        "current_price": 49500.0,
        "current_volume": 1e6,
        "news_sentiment": -0.5,
        "time_of_day_factor": 0.8,
        "resistance_level": 50000.0,
        "support_level": 48000.0,
        "ai_feedback": [{"sentiment": -0.6}],
        "error_logs": [{"code": 101, "message": "Volatility Spike"}],
        "resonance_level": 0.2,
        "recent_news_headlines": "Bearish news on the wires, traders cautious.",
        "dominant_indicator": "MACD",
    }

    # Process data and get decisions
    print("\nProcessing Market Data 1 (Bullish Scenario):")
    thought_1 = engines.process_market_data(market_data_1, thermal_state=HOT)
    print(f"  Final Decision: {thought_1.decision}, Confidence: {thought_1.confidence:.2f}, State: {thought_1.state.value}")
    print(f"  Thought Hash: {thought_1.thought_hash_32bit[:10]}...")
    print(f"  Tags: {thought_1.tags}")

    print("\nProcessing Market Data 2 (Bearish Scenario):")
    thought_2 = engines.process_market_data(market_data_2, thermal_state=COOL)
    print(f"  Final Decision: {thought_2.decision}, Confidence: {thought_2.confidence:.2f}, State: {thought_2.state.value}")
    print(f"  Thought Hash: {thought_2.thought_hash_32bit[:10]}...")
    print(f"  Tags: {thought_2.tags}")

    print("\n--- System Status ---")
    status = engines.get_system_status()
    for key, value in status.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    print("\n--- ALIF Feedback History (Last 2) ---")
    for feedback in engines.alif_feedback_history[-2:]:
        print(f"  Timestamp: {datetime.fromtimestamp(feedback.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Type: {feedback.feedback_type}")
        print(f"  Message: {feedback.message}")
        print(f"  Decision Score: {feedback.decision_score:.2f}")
        print("---")

    # Demonstrate resetting metrics
    engines.reset_metrics()
    print("\nMetrics after reset:")
    print(f"  Total Decisions: {engines.total_decisions}")


if __name__ == "__main__":
    main() 