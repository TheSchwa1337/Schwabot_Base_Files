# -*- coding: utf-8 -*-
"""
Dualistic Thought Engines.

Advanced AI-driven cognitive architectures for enhanced decision-making
within the Schwabot trading system. These engines operate on a dualistic
principle, integrating both logical and intuitive thought processes to
optimize trading strategies, manage risk, and identify complex market patterns.

Mathematical Foundation:
    - Dualistic Logic Gates: (A and B) or (C ⊕ D)
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
        self.adaptation_rate = 0.05  # Rate at which the engine adapts
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
        self.alif_adaptation_rate = 0.02  # ALIF-specific adaptation rate

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

            # --- Hash Relay Integration ---
            # Relay the finalized ThoughtVector (as dict) to the hash relay system
            relay_data = {
                'timestamp': thought_vector.timestamp,
                'state': thought_vector.state.value,
                'thermal_state': thought_vector.thermal_state,
                'logical_score': thought_vector.logical_score,
                'intuitive_score': thought_vector.intuitive_score,
                'historical_adjustment': thought_vector.historical_adjustment,
                'combined_score': thought_vector.combined_score,
                'decision': thought_vector.decision,
                'confidence': thought_vector.confidence,
                'tags': sorted(tag.value for tag in thought_vector.tags),
                'thought_hash_32bit': thought_vector.thought_hash_32bit,
            }
            hash_relay_system.submit(relay_data)
            # --- End Hash Relay Integration ---

            logger.info(
                f"Decision: {final_decision} (Confidence: {final_score:.2f}, "
                f"State: {final_state.value}, ALIF: {alif_score:.3f}, "
                f"Bias: {bias_mitigated.value if bias_mitigated else 'None'}, "
                f"Thermal: {self.current_thermal_state}, "
                f"Tags: {[tag.value for tag in sorted(thought_vector.tags, key=lambda t: t.value)]})")
            return thought_vector

        except Exception as e:
            logger.error(f"Error processing market data: {e}", exc_info=True)
            return self._create_fallback_thought_vector(market_data, start_time)

    def _perform_logical_analysis(
        self, market_data: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Perform logical, rule-based analysis of market data.

        Mathematical: LogicalScore = f(RSI, MACD, Volume) * f(Thermal)
        """
        # Example logical analysis: RSI, MACD, Volume, and Price
        rsi = market_data.get("rsi", 50.0)
        macd_signal = market_data.get("macd_signal", 0.0)
        volume_change = market_data.get("volume_change", 0.0)
        current_price = market_data.get("current_price", 0.0)
        moving_average = market_data.get("moving_average", 0.0)

        score = 0.0
        decision = "hold"

        # Rule 1: RSI indicates oversold and MACD crossover (Buy signal)
        if rsi < 30.0 and macd_signal > 0.0 and current_price > moving_average:
            score += 0.4
            decision = "buy"
        # Rule 2: RSI indicates overbought and MACD crossover (Sell signal)
        elif rsi > 70.0 and macd_signal < 0.0 and current_price < moving_average:
            score += 0.4
            decision = "sell"

        # Rule 3: Significant volume increase in direction of trend
        if (volume_change > 0.2 and decision == "buy") or (volume_change < -0.2 and decision == "sell"):
            score += 0.2

        # Rule 4: Price action confirms decision
        if decision == "buy" and current_price > market_data.get("previous_close", current_price):
            score += 0.1
        elif decision == "sell" and current_price < market_data.get("previous_close", current_price):
            score += 0.1

        # Apply thermal multiplier for enhanced sensitivity in HOT state
        thermal_multiplier = {COOL: 0.9, WARM: 1.0, HOT: 1.15, CRITICAL: 1.3}.get(self.current_thermal_state, 1.0)
        score *= thermal_multiplier

        # Normalize score to 0-1
        score = min(score, 1.0)

        logger.debug(f"Logical analysis: score={score:.2f}, decision={decision}, thermal={self.current_thermal_state}")
        return score, decision

    def _perform_intuitive_analysis(
        self, market_data: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Perform intuitive, pattern-based analysis using SchwafitCore.

        Mathematical: IntuitiveScore = Schwafit(patterns) * f(Thermal)
        """
        # Example intuitive analysis: using SchwafitCore for pattern recognition
        prices = market_data.get("price_history", [])
        volumes = market_data.get("volume_history", [])
        phases = market_data.get("phase_data", [0.0, 0.0, 0.0, 0.0])

        if not prices or not volumes:
            return 0.5, "hold"  # Default if no data

        schwafit_input = {"prices": prices, "volumes": volumes, "phases": phases}
        schwafit_results = self.schwafit_core.comprehensive_mirror_analysis(schwafit_input)

        overall_intuitive_confidence = 0.0
        intuitive_decision = "hold"

        if schwafit_results:
            # Aggregate confidence from Schwafit frameworks
            confidences = [result.confidence for result in schwafit_results.values()]
            if confidences:
                overall_intuitive_confidence = sum(confidences) / len(confidences)

            # Schwafit recommendations for intuitive decision
            recommendations = self.schwafit_core.get_mirror_recommendations(schwafit_results)
            action = recommendations.get("recommended_action", "hold").lower()

            if "buy" in action:
                intuitive_decision = "buy"
            elif "sell" in action:
                intuitive_decision = "sell"
            else:
                intuitive_decision = "hold"

        # Apply thermal multiplier for enhanced sensitivity in HOT state
        thermal_multiplier = {COOL: 0.9, WARM: 1.0, HOT: 1.2, CRITICAL: 1.4}.get(self.current_thermal_state, 1.0)
        overall_intuitive_confidence *= thermal_multiplier
        overall_intuitive_confidence = min(overall_intuitive_confidence, 1.0)

        logger.debug(
            f"Intuitive analysis: score={overall_intuitive_confidence:.2f}, "
            f"decision={intuitive_decision}, thermal={self.current_thermal_state}"
        )
        return overall_intuitive_confidence, intuitive_decision

    def _perform_alif_analysis(
        self,
        market_data: Dict[str, Any],
        ai_feedback: List[Dict[str, Any]] = None,
        error_logs: List[Dict[str, Any]] = None,
    ) -> Tuple[float, str, AlifFeedbackData]:
        """Perform ALIF (Adaptive Learning Interference Filter) analysis.

        Mathematical: F(t) = Σ w_i · ΔV_i + w_j · ΔΨ_j
        Where:
        - ΔV_i: Volume deltas and market memory changes
        - ΔΨ_j: Resonance deltas and AI feedback scores
        - w_i, w_j: Adaptive weights for different feedback layers

        Args:
            market_data: Current market data
            ai_feedback: List of AI model feedback scores
            error_logs: List of recent error corrections

        Returns:
            Tuple of (alif_score, alif_decision, alif_feedback)
        """
        try:
            # Calculate volume deltas (ΔV_i)
            volume_delta = self._calculate_volume_delta(market_data)

            # Calculate resonance deltas (ΔΨ_j)
            resonance_delta = self._calculate_resonance_delta(market_data)

            # Calculate AI feedback score
            ai_feedback_score = self._calculate_ai_feedback_score(ai_feedback or [])

            # Calculate error correction factor
            error_correction = self._calculate_error_correction(error_logs or [])

            # Calculate market memory score
            market_memory_score = self._calculate_market_memory_score(market_data)

            # Compute ALIF feedback score using weighted sum
            alif_score = (
                self.alif_weights["volume"] * volume_delta
                + self.alif_weights["resonance"] * resonance_delta
                + self.alif_weights["ai_feedback"] * ai_feedback_score
                + self.alif_weights["error_correction"] * error_correction
            )

            # Apply market memory adjustment
            alif_score += 0.1 * market_memory_score

            # Normalize score to [0, 1]
            alif_score = max(0.0, min(1.0, alif_score))

            # Determine ALIF decision based on score
            if alif_score > 0.8:
                alif_decision = "ENHANCE_SIGNAL"
                routing_target = "gpu_4bit"
            elif alif_score > 0.6:
                alif_decision = "MAINTAIN_SIGNAL"
                routing_target = "cpu_2bit"
            elif alif_score > 0.4:
                alif_decision = "ATTENUATE_SIGNAL"
                routing_target = "cpu_2bit"
            elif alif_score > 0.2:
                alif_decision = "CORRECT_ERROR"
                routing_target = "coldbase_8bit"
            else:
                alif_decision = "EMERGENCY_HALT"
                routing_target = "coldbase_8bit"

            # Create ALIF feedback structure
            alif_feedback = AlifFeedbackData(
                volume_delta=volume_delta,
                resonance_delta=resonance_delta,
                ai_feedback_score=ai_feedback_score,
                error_correction=error_correction,
                market_memory_score=market_memory_score,
                adaptation_rate=self.alif_adaptation_rate,
                feedback_confidence=alif_score,
                routing_target=routing_target,
            )

            # Store in ALIF memory
            self._store_alif_memory(market_data, alif_feedback)

            # Update performance metrics
            self.alif_decisions += 1
            if alif_score > self.alif_threshold:
                self.alif_activations += 1
            if alif_decision == "CORRECT_ERROR":
                self.alif_corrections += 1

            logger.debug(
                f"ALIF analysis: score={alif_score:.3f}, decision={alif_decision}, "
                f"routing={routing_target}, thermal={self.current_thermal_state}"
            )

            return alif_score, alif_decision, alif_feedback

        except Exception as e:
            logger.error(f"ALIF analysis failed: {e}", exc_info=True)
            # Return fallback values
            fallback_feedback = AlifFeedbackData(feedback_confidence=0.5, routing_target="cpu_2bit")
            return 0.5, "HOLD_SIGNAL", fallback_feedback

    def _calculate_volume_delta(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume delta for ALIF analysis."""
        current_volume = market_data.get("volume", 0.0)
        volume_change = market_data.get("volume_change", 0.0)

        # Normalize volume delta
        if current_volume > 0:
            volume_delta = volume_change / current_volume
        else:
            volume_delta = 0.0

        return volume_delta

    def _calculate_resonance_delta(self, market_data: Dict[str, Any]) -> float:
        """Calculate resonance delta for ALIF analysis."""
        # Resonance based on price momentum and volatility alignment
        price_momentum = market_data.get("price_momentum", 0.0)
        volatility = market_data.get("volatility", 0.5)
        rsi = market_data.get("rsi", 50.0)

        # Normalize RSI to [-1, 1]
        rsi_normalized = (rsi - 50.0) / 50.0

        # Resonance is high when momentum and RSI align
        momentum_rsi_alignment = price_momentum * rsi_normalized

        # Volatility adjustment (higher volatility reduces resonance)
        volatility_factor = 1.0 / (1.0 + volatility)

        resonance_delta = momentum_rsi_alignment * volatility_factor
        return np.clip(resonance_delta, -1.0, 1.0)

    def _calculate_ai_feedback_score(self, ai_feedback: List[Dict[str, Any]]) -> float:
        """Calculate AI feedback score for ALIF analysis."""
        if not ai_feedback:
            return 0.0

        scores = []
        for feedback in ai_feedback:
            score = feedback.get("confidence", 0.0)
            weight = feedback.get("weight", 1.0)
            scores.append(score * weight)

        if not scores:
            return 0.0

        # Return weighted average
        return sum(scores) / len(scores)

    def _calculate_error_correction(self, error_logs: List[Dict[str, Any]]) -> float:
        """Calculate error correction score for ALIF analysis."""
        if not error_logs:
            return 0.0

        # Sum up error severities (negative values)
        total_error_severity = sum(-abs(error.get("severity", 0.0)) for error in error_logs)

        # Normalize by number of errors
        avg_error = total_error_severity / len(error_logs)

        # Apply exponential decay for recent errors
        return np.clip(avg_error, -1.0, 0.0)

    def _calculate_market_memory_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate market memory score for ALIF analysis."""
        current_price = market_data.get("current_price", 0.0)
        if current_price == 0:
            return 0.0

        # Check if current price is in memory
        price_key = f"{current_price:.2f}"
        if price_key in self.alif_market_memory:
            return self.alif_market_memory[price_key]

        # Default to neutral
        return 0.0

    def _store_alif_memory(self, market_data: Dict[str, Any], alif_feedback: AlifFeedbackData):
        """Store ALIF analysis in memory buffer."""
        memory_entry = {
            "timestamp": time.time(),
            "market_data": market_data.copy(),
            "alif_feedback": alif_feedback,
            "thermal_state": self.current_thermal_state,
        }

        self.alif_memory.append(memory_entry)
        self.alif_feedback_history.append(alif_feedback)

        # Maintain memory size
        if len(self.alif_memory) > self.alif_memory_size:
            self.alif_memory = self.alif_memory[-self.alif_memory_size :]
        if len(self.alif_feedback_history) > self.alif_memory_size:
            self.alif_feedback_history = self.alif_feedback_history[-self.alif_memory_size :]

    def _consult_historical_patterns(
        self, market_data: Dict[str, Any]
    ) -> Tuple[float, Dict]:
        """Consult historical thought vectors to find similar past scenarios.

        This provides a "hesitation" and learning mechanism.

        Mathematical: H_adj = sum (similarity(v_current, v_past) * outcome(v_past))
        """
        if not self.thought_history:
            return 0.0, {"status": "no_similar_history"}

        current_hash = self._generate_input_hash(market_data)
        similar_vectors = []

        # Find similar past situations based on hash prefix matching
        for vector in reversed(self.thought_history):
            # Simple similarity: match first 6 hex chars of the hash (24 bits)
            if vector.thought_hash_32bit[:6] == current_hash[:6]:
                similar_vectors.append(vector)
            if len(similar_vectors) >= 5:  # Limit to last 5 similar events
                break

        if not similar_vectors:
            return 0.0, {"status": "no_conclusive_outcomes", "found": len(similar_vectors)}

        # Analyze outcomes of similar past decisions
        # This requires `market_data` to have an `actual_profit` field from a previous cycle
        # For simulation, we can use a placeholder.
        adjustments = []
        for vector in similar_vectors:
            # This logic is illustrative. A real system would need to track trade outcomes.
            past_outcome_profit = market_data.get("actual_profit_from_last_trade", 0.0)

            if vector.decision == "buy" and past_outcome_profit > 0:
                adjustments.append(1)  # Reinforce buy
            elif vector.decision == "buy" and past_outcome_profit < 0:
                adjustments.append(-1)  # Punish buy
            elif vector.decision == "sell" and past_outcome_profit > 0:
                adjustments.append(-1)  # Punish sell (assuming shorting profit)
            elif vector.decision == "sell" and past_outcome_profit < 0:
                adjustments.append(1)  # Reinforce sell

        if not adjustments:
            return 0.0, {"status": "no_conclusive_outcomes", "found": len(similar_vectors)}

        net_adjustment = np.mean(adjustments) * self.historical_influence_factor

        consultation_summary = {
            "status": "consulted",
            "similar_events": len(similar_vectors),
            "net_adjustment": net_adjustment,
        }
        logger.debug(f"Historical consultation: adjustment={net_adjustment:.2f}, similar_events={len(similar_vectors)}")

        return net_adjustment, consultation_summary

    def _integrate_dualistic_logic(
        self: "DualisticThoughtEngines",
        logical_score: float,
        logical_decision: str,
        intuitive_score: float,
        intuitive_decision: str,
        historical_adjustment: float,
        market_data: Dict[str, Any],
    ) -> Tuple[float, str]:
        """Integrate logical and intuitive analysis using dualistic logic gates.

        Factoring in thermal state and historical consultation.

        Mathematical: Combined = ((L and I) or (I ⊕ C)) * (1 + H_adj) * f(Thermal)
        """
        # Implement a weighted average or a more complex logic gate
        # For simplicity, a weighted average is used here
        logical_weight = 0.5
        intuitive_weight = 0.5

        # Adjust weights based on market volatility (example: more intuitive in high volatility)
        volatility = market_data.get("volatility", 0.5)
        # Thermal state can also influence weights
        if self.current_thermal_state == HOT or self.current_thermal_state == CRITICAL:
            logical_weight = 0.4  # Lean more on intuition in hot markets
            intuitive_weight = 0.6
        elif volatility > 0.7:
            logical_weight = 0.4
            intuitive_weight = 0.6
        elif volatility < 0.3:
            logical_weight = 0.6
            intuitive_weight = 0.4

        combined_score = (logical_score * logical_weight) + (intuitive_score * intuitive_weight)

        # Apply historical adjustment
        if historical_adjustment > 0:  # History suggests bullish
            combined_score += historical_adjustment
        elif historical_adjustment < 0:  # History suggests bearish
            combined_score += historical_adjustment

        # Determine final decision based on combined score and agreement
        final_decision = "hold"
        if logical_decision == intuitive_decision:
            final_decision = logical_decision  # Strong agreement
        elif combined_score >= 0.7:
            final_decision = (
                logical_decision if logical_score > intuitive_score else intuitive_decision
            )  # Lean towards stronger signal
        elif combined_score <= 0.3:
            final_decision = (
                logical_decision if logical_score < intuitive_score else intuitive_decision
            )  # Lean away from weaker signal

        # Further refinement based on "consensus" (e.g., external signals)
        consensus_signal = market_data.get("consensus_signal", "neutral")
        if consensus_signal == "buy" and combined_score > 0.5:  # Consensus confirms a buy signal
            final_decision = "buy"
        elif consensus_signal == "sell" and combined_score < 0.5:  # Consensus confirms a sell signal
            final_decision = "sell"

        # Update current state based on which engine dominated
        if logical_score * logical_weight > intuitive_score * intuitive_weight:
            self.current_state = DualisticState.LOGICAL
        else:
            self.current_state = DualisticState.INTUITIVE

        logger.debug(
            f"Dualistic integration: combined_score={combined_score:.2f}, "
            f"final_decision={final_decision}, state={self.current_state.value}"
        )
        # Final normalization
        combined_score = max(0.0, min(1.0, combined_score))
        return combined_score, final_decision

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
        """
        Integrates logical, intuitive, and ALIF analysis using dualistic logic gates,
        factoring in thermal state and historical consultation.

        Mathematical: Combined = ((L and I) or (I ⊕ C)) * (1 + H_adj) * f(Thermal)
        """
        # Implement a weighted average or a more complex logic gate
        # For simplicity, a weighted average is used here
        logical_weight = 0.5
        intuitive_weight = 0.5

        # Adjust weights based on market volatility (example: more intuitive in high volatility)
        volatility = market_data.get("volatility", 0.5)
        # Thermal state can also influence weights
        if self.current_thermal_state == HOT or self.current_thermal_state == CRITICAL:
            logical_weight = 0.4  # Lean more on intuition in hot markets
            intuitive_weight = 0.6
        elif volatility > 0.7:
            logical_weight = 0.4
            intuitive_weight = 0.6
        elif volatility < 0.3:
            logical_weight = 0.6
            intuitive_weight = 0.4

        combined_score = (logical_score * logical_weight) + (intuitive_score * intuitive_weight)

        # Apply historical adjustment
        if historical_adjustment > 0:  # History suggests bullish
            combined_score += historical_adjustment
        elif historical_adjustment < 0:  # History suggests bearish
            combined_score += historical_adjustment

        # Determine final decision based on combined score and agreement
        final_decision = "hold"
        if logical_decision == intuitive_decision:
            final_decision = logical_decision  # Strong agreement
        elif combined_score >= 0.7:
            final_decision = (
                logical_decision if logical_score > intuitive_score else intuitive_decision
            )  # Lean towards stronger signal
        elif combined_score <= 0.3:
            final_decision = (
                logical_decision if logical_score < intuitive_score else intuitive_decision
            )  # Lean away from weaker signal

        # Further refinement based on "consensus" (e.g., external signals)
        consensus_signal = market_data.get("consensus_signal", "neutral")
        if consensus_signal == "buy" and combined_score > 0.5:  # Consensus confirms a buy signal
            final_decision = "buy"
        elif consensus_signal == "sell" and combined_score < 0.5:  # Consensus confirms a sell signal
            final_decision = "sell"

        # Update current state based on which engine dominated
        if logical_score * logical_weight > intuitive_score * intuitive_weight:
            self.current_state = DualisticState.LOGICAL
        else:
            self.current_state = DualisticState.INTUITIVE

        logger.debug(
            f"Dualistic integration with ALIF: combined_score={combined_score:.2f}, "
            f"final_decision={final_decision}, state={self.current_state.value}"
        )
        # Final normalization
        combined_score = max(0.0, min(1.0, combined_score))
        return combined_score, final_decision

    def _mitigate_cognitive_bias(
        self: "DualisticThoughtEngines", market_data: Dict[str, Any], combined_score: float
    ) -> Optional[CognitiveBias]:
        """Identify and mitigate cognitive biases in decision-making.

        Mathematical: B_mit = (1 - epsilon) * B_raw
        """
        # Check for overconfidence bias (recent decisions all the same)
        if (
            len(self.thought_history) >= 10
            and len({tv.decision for tv in self.thought_history[-10:]}) == 1
        ):
            detected_bias = CognitiveBias.OVERCONFIDENCE
        else:
            # Simulate bias detection (in real system, this would be more sophisticated)
            detected_bias = (
                CognitiveBias.HERDING
                if np.random.random() > 0.5
                else CognitiveBias.OVERCONFIDENCE
            )

        if detected_bias:
            logger.warning(f"Detected potential cognitive bias: {detected_bias.value}")
            # Apply mitigation by adjusting combined_score
            combined_score *= 1.0 - self.bias_mitigation_strength
            logger.info(f"Bias mitigated. New combined score: {combined_score:.2f}")

        return detected_bias

    def _adapt_and_learn(
        self: "DualisticThoughtEngines",
        combined_score: float,
        decision: str,
        market_data: Dict[str, Any]
    ) -> None:
        """Adapts engine parameters based on decision outcomes and market feedback.

        Mathematical: P_new = P_old + alpha * Error
        """
        # Example adaptation: adjust weights based on hypothetical outcome
        # (In a real system, this would involve actual trade results)
        is_successful_outcome = market_data.get("actual_profit", 0.0) > 0

        if is_successful_outcome and combined_score > 0.7:
            # Strengthen the dominant engine's influence
            if self.current_state == DualisticState.LOGICAL:
                self.schwafit_core.alif_threshold = min(self.schwafit_core.alif_threshold + self.adaptation_rate, 1.0)
            else:
                self.schwafit_core.mir4x_threshold = min(self.schwafit_core.mir4x_threshold + self.adaptation_rate, 1.0)
            logger.debug("Engine adapted: positive feedback.")
        elif not is_successful_outcome and combined_score < 0.3:
            # Weaken the dominant engine's influence
            if self.current_state == DualisticState.LOGICAL:
                self.schwafit_core.alif_threshold = max(self.schwafit_core.alif_threshold - self.adaptation_rate, 0.0)
            else:
                self.schwafit_core.mir4x_threshold = max(self.schwafit_core.mir4x_threshold - self.adaptation_rate, 0.0)
            logger.debug("Engine adapted: negative feedback.")

    def _generate_input_hash(self: "DualisticThoughtEngines", market_data: Dict[str, Any]) -> str:
        """Generate a 32-bit hash from raw market input for vectorization."""
        # This is a simplified 32-bit hash. For true 32-bit, more complex
        # bit manipulation would be needed. This is more of a unique identifier.
        data_string = json.dumps(market_data, sort_keys=True)
        # Use a portion of SHA256 for a 32-bit like identifier
        full_hash = hashlib.sha256(data_string.encode("utf-8")).hexdigest()
        # Take the first 8 characters (32 bits in hex)
        return full_hash[:8]

    def _create_fallback_thought_vector(
        self: "DualisticThoughtEngines", market_data: Dict[str, Any], start_time: float
    ) -> ThoughtVector:
        """Create a fallback thought vector in case of processing errors."""
        logger.warning("Creating fallback thought vector due to error.")
        return ThoughtVector(
            timestamp=time.time(),
            state=DualisticState.LOGICAL,  # Default to logical for safety
            thermal_state=self.current_thermal_state,
            logical_score=0.5,
            intuitive_score=0.5,
            historical_adjustment=0.0,
            combined_score=0.5,
            decision="hold",  # Default to hold
            confidence=0.5,
            bias_mitigated=None,
            thought_hash_32bit=self._generate_input_hash(market_data),
        )

    def _determine_final_state(
        self: "DualisticThoughtEngines", logical_score: float, intuitive_score: float, alif_score: float
    ) -> DualisticState:
        """Determine the final state based on the scores of the three analyses."""
        if alif_score > 0.7:
            return DualisticState.ALIF
        elif logical_score > intuitive_score:
            return DualisticState.LOGICAL
        else:
            return DualisticState.INTUITIVE

    def get_engine_performance(self: "DualisticThoughtEngines") -> Dict[str, Any]:
        """Return performance metrics for the dualistic thought engines."""
        success_rate = self.successful_decisions / max(self.total_decisions, 1)

        # Calculate ALIF-specific metrics
        alif_activation_rate = self.alif_activations / max(self.alif_decisions, 1)
        alif_correction_rate = self.alif_corrections / max(self.alif_decisions, 1)

        return {
            "total_decisions": self.total_decisions,
            "successful_decisions": self.successful_decisions,
            "success_rate": success_rate,
            "logical_decisions": self.logical_decisions,
            "intuitive_decisions": self.intuitive_decisions,
            "alif_decisions": self.alif_decisions,
            "alif_activations": self.alif_activations,
            "alif_corrections": self.alif_corrections,
            "alif_activation_rate": alif_activation_rate,
            "alif_correction_rate": alif_correction_rate,
            "current_state": self.current_state.value,
            "current_thermal_state": self.current_thermal_state,
            "thought_history_size": len(self.thought_history),
            "alif_memory_size": len(self.alif_memory),
            "alif_feedback_history_size": len(self.alif_feedback_history),
            "alif_enabled": self.alif_enabled,
            "alif_threshold": self.alif_threshold,
            "alif_weights": self.alif_weights.copy(),
        }

    def enable_alif(self: "DualisticThoughtEngines") -> None:
        """Enable ALIF analysis."""
        self.alif_enabled = True
        logger.info("ALIF analysis enabled")

    def disable_alif(self: "DualisticThoughtEngines") -> None:
        """Disable ALIF analysis."""
        self.alif_enabled = False
        logger.info("ALIF analysis disabled")

    def set_alif_threshold(self: "DualisticThoughtEngines", threshold: float) -> None:
        """Set ALIF activation threshold."""
        if 0.0 <= threshold <= 1.0:
            self.alif_threshold = threshold
            logger.info(f"ALIF threshold set to {threshold}")
        else:
            logger.warning(f"Invalid ALIF threshold: {threshold}. Must be between 0.0 and 1.0")

    def set_alif_weights(self: "DualisticThoughtEngines", weights: Dict[str, float]) -> None:
        """Set ALIF weights for different feedback components."""
        required_keys = {"volume", "resonance", "ai_feedback", "error_correction"}
        if set(weights.keys()) == required_keys:
            if all(0.0 <= w <= 1.0 for w in weights.values()):
                self.alif_weights = weights.copy()
                logger.info(f"ALIF weights updated: {weights}")
            else:
                logger.warning("ALIF weights must be between 0.0 and 1.0")
        else:
            logger.warning(f"ALIF weights must contain keys: {required_keys}")

    def get_alif_memory(self: "DualisticThoughtEngines", limit: int = 100) -> List[Dict[str, Any]]:
        """Get ALIF memory entries up to the specified limit."""
        return self.alif_memory[-limit:] if self.alif_memory else []

    def get_alif_feedback_history(self: "DualisticThoughtEngines", limit: int = 100) -> List[AlifFeedback]:
        """Get ALIF feedback history up to the specified limit."""
        return self.alif_feedback_history[-limit:] if self.alif_feedback_history else []

    def clear_alif_memory(self: "DualisticThoughtEngines") -> None:
        """Clear all ALIF memory entries."""
        self.alif_memory.clear()
        self.alif_feedback_history.clear()
        self.alif_error_log.clear()
        self.alif_market_memory.clear()
        logger.info("ALIF memory cleared")

    def add_alif_error(self: "DualisticThoughtEngines", error: Dict[str, Any]) -> None:
        """Add an error to ALIF memory for learning."""
        error["timestamp"] = time.time()
        self.alif_error_log.append(error)

        # Maintain error log size
        if len(self.alif_error_log) > self.alif_memory_size:
            self.alif_error_log = self.alif_error_log[-self.alif_memory_size :]

        logger.debug(f"ALIF error logged: {error.get('message', 'Unknown error')}")

    def add_ai_feedback(self: "DualisticThoughtEngines", feedback: Dict[str, Any]) -> None:
        """Add AI feedback to ALIF memory."""
        feedback["timestamp"] = time.time()
        # Store in market memory for future reference
        self.alif_market_memory[f"ai_feedback_{int(time.time())}"] = feedback.get("confidence", 0.0)
        logger.debug(f"AI feedback added: {feedback.get('model', 'Unknown model')}")

    def force_alif_state(self: "DualisticThoughtEngines", market_data: Dict[str, Any]) -> ThoughtVector:
        """Force ALIF state analysis regardless of normal conditions."""
        if not self.alif_enabled:
            logger.warning("ALIF is disabled. Enabling for forced ALIF state.")
            self.alif_enabled = True

        # Temporarily set current state to ALIF
        original_state = self.current_state
        self.current_state = DualisticState.ALIF

        try:
            # Process with ALIF focus
            thought_vector = self.process_market_data(market_data)
            logger.info("Forced ALIF state processing completed")
            return thought_vector
        finally:
            # Restore original state
            self.current_state = original_state

    def get_alif_statistics(self: "DualisticThoughtEngines") -> Dict[str, Any]:
        """Get comprehensive ALIF statistics and performance metrics."""
        return {
            "enabled": self.alif_enabled,
            "threshold": self.alif_threshold,
            "weights": self.alif_weights.copy(),
            "memory_size": len(self.alif_memory),
            "feedback_history_size": len(self.alif_feedback_history),
            "error_log_size": len(self.alif_error_log),
            "market_memory_size": len(self.alif_market_memory),
            "total_decisions": self.alif_decisions,
            "activations": self.alif_activations,
            "corrections": self.alif_corrections,
            "activation_rate": self.alif_activations / max(self.alif_decisions, 1),
            "correction_rate": self.alif_corrections / max(self.alif_decisions, 1),
            "adaptation_rate": self.alif_adaptation_rate,
        }

    def process_linguistic_trading_command(self, command: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process natural language trading command through linguistic glyph engine.
        
        Args:
            command: Natural language command with optional glyphs
            market_data: Current market data for context
            
        Returns:
            Dictionary with linguistic analysis and trading decision
        """
        try:
            # Process through linguistic glyph engine
            linguistic_result = process_linguistic_command(command)
            
            # Extract BTC price from market data if available
            btc_price = market_data.get('btc_price', 45000.0)
            usdc_balance = market_data.get('usdc_balance', 10000.0)
            
            # Generate trade vector through linguistic engine
            trade_vector = linguistic_engine.process_btc_usdc_waveform(
                command, btc_price, usdc_balance
            )
            
            # Generate thought vector with linguistic context
            thought_vector = self.generate_thought_vector(market_data, {
                'linguistic_command': command,
                'glyph_signature': trade_vector.glyph_signature,
                'bit_state': linguistic_result['bit_state'],
                'entropy_overlay': linguistic_result['entropy_overlay']
            })
            
            # Combine results
            return {
                'linguistic_analysis': linguistic_result,
                'trade_vector': {
                    'entry_hash': trade_vector.entry_hash,
                    'profit_delta': trade_vector.profit_delta,
                    'bit_sequence_length': len(trade_vector.bit_sequence),
                    'glyph_signature': trade_vector.glyph_signature,
                },
                'thought_vector': {
                    'hash': thought_vector.thought_hash_32bit,
                    'decision': thought_vector.decision,
                    'confidence': thought_vector.confidence,
                    'state': thought_vector.state.value,
                },
                'memory_state': linguistic_engine.get_memory_state_summary(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing linguistic command '{command}': {e}")
            return {
                'error': str(e),
                'command': command,
                'success': False
            }

    def process_news_for_decision(self, news_item: NewsItem, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a news item and integrate its sentiment and impact into decision-making.

        Args:
            news_item: The processed NewsItem object from LanternNewsIntelligenceBridge.
            market_data: Current market data for context.

        Returns:
            Dictionary with linguistic analysis, thought vector, and memory state.
        """
        try:
            # Step 1: Convert NewsItem to LinguisticHash
            linguistic_hash = linguistic_engine.process_news_item_for_linguistic_hash(news_item)
            
            # Step 2: Use linguistic hash to influence thought vector generation
            # We'll pass relevant news-derived linguistic data into the context for process_decision
            processing_context = {
                'news_title': news_item.title,
                'news_sentiment_score': news_item.sentiment_score,
                'news_sentiment_type': news_item.sentiment_type.value,
                'news_impact_level': news_item.impact_level.value,
                'news_linguistic_bit_state': linguistic_hash.bit_state,
                'news_linguistic_weight': linguistic_hash.weight,
                'news_linguistic_hash': linguistic_hash.sha_hash,
            }
            
            # Integrate this context into the thought generation
            thought_vector = self.generate_thought_vector(market_data, processing_context)
            
            # Step 3: Relay news-derived thought data
            # The thought_vector will already be relayed by generate_thought_vector if successful
            
            return {
                'news_item': {
                    'id': news_item.news_id,
                    'title': news_item.title,
                    'sentiment': news_item.sentiment_type.value,
                    'impact': news_item.impact_level.value,
                    'confidence': news_item.confidence_score,
                },
                'linguistic_analysis': {
                    'bit_state': linguistic_hash.bit_state,
                    'weight': linguistic_hash.weight,
                    'sha_hash': linguistic_hash.sha_hash,
                },
                'thought_vector': {
                    'hash': thought_vector.thought_hash_32bit,
                    'decision': thought_vector.decision,
                    'confidence': thought_vector.confidence,
                    'state': thought_vector.state.value,
                    'tags': [tag.value for tag in sorted(thought_vector.tags)],
                },
                'memory_state_summary': linguistic_engine.get_memory_state_summary(),
                'success': True,
            }

        except Exception as e:
            logger.error(f"Error processing news item '{news_item.title}' for decision: {e}", exc_info=True)
            return {
                'error': str(e),
                'news_title': news_item.title,
                'success': False,
            }


# Global instance for easy access
dualistic_engines = DualisticThoughtEngines()

# Example usage (for testing purposes)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Dualistic Thought Engines Demonstration (32-bit Thermal Edition with Lantern Core)")
    print("=" * 70)

    # Sample market data
    sample_market_data = {
        "rsi": 25.5,
        "macd_signal": 0.01,
        "volume_change": 0.3,
        "current_price": 62000.0,
        "moving_average": 61500.0,
        "previous_close": 61800.0,
        "price_history": [61000.0, 61500.0, 62000.0, 61800.0, 62200.0, 62500.0, 62300.0],
        "volume_history": [100.0, 120.0, 110.0, 90.0, 130.0, 150.0, 140.0],
        "phase_data": [0.6, 0.4, 0.8, 0.2],
        "volatility": 0.8,  # High volatility to trigger HOT state logic
        "sentiment_score": 0.9,
        "performance_delta": 0.05,
        "actual_profit": 100.0,  # Simulate a successful outcome
        "consensus_signal": "buy",
        "actual_profit_from_last_trade": 50.0,  # For historical consultation demo
    }

    # Process data and get a thought vector in a HOT thermal state
    print("\n--- Processing in HOT Thermal State ---")
    thought = dualistic_engines.process_market_data(sample_market_data, thermal_state=HOT)
    print("\nProcessed Thought Vector:")
    print(f"  Timestamp: {datetime.fromtimestamp(thought.timestamp)}")
    print(f"  State: {thought.state.value}")
    print(f"  Thermal State: {thought.thermal_state}")
    print(f"  Logical Score: {thought.logical_score:.2f}")
    print(f"  Intuitive Score: {thought.intuitive_score:.2f}")
    print(f"  Historical Adjustment: {thought.historical_adjustment:.2f}")
    print(f"  Combined Score: {thought.combined_score:.2f}")
    print(f"  Decision: {thought.decision}")
    print(f"  Confidence: {thought.confidence:.2f}")
    print(f"  Bias Mitigated: {thought.bias_mitigated.value if thought.bias_mitigated else 'None'}")
    print(f"  Input Hash (32-bit): {thought.thought_hash_32bit}")
    print(f"  Historical Consultation: {thought.historical_consultation}")
    print(f"  Brain Tags: {[t.value for t in sorted([tag.value for tag in thought.tags])]}")

    print("\n--- Simulating a second run for historical context ---")
    # Simulate a slightly different market but with the same hash prefix to trigger historical consultation
    sample_market_data_2 = sample_market_data.copy()
    sample_market_data_2["rsi"] = 28.0
    thought_2 = dualistic_engines.process_market_data(sample_market_data_2, thermal_state=HOT)
    print("\nSecond Processed Thought Vector:")
    print(f"  Decision: {thought_2.decision}")
    print(f"  Combined Score: {thought_2.combined_score:.2f}")
    print(f"  Historical Adjustment: {thought_2.historical_adjustment:.2f}")
    print(f"  Historical Consultation: {thought_2.historical_consultation}")
    print(f"  Brain Tags: {[t.value for t in sorted([tag.value for tag in thought_2.tags])]}")

    print("\nEngine Performance:")
    performance = dualistic_engines.get_engine_performance()
    for key, value in performance.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")

    # Simulate an error scenario
    print("\nSimulating error scenario...")
    error_market_data = {"invalid_key": "some_value"}  # Missing required keys
    error_thought = dualistic_engines.process_market_data(error_market_data, thermal_state=COOL)
    print("\nProcessed Error Thought Vector:")
    print(f"  Decision: {error_thought.decision}")
    print(f"  Confidence: {error_thought.confidence:.2f}")

    # End of dualistic_thought_engines.py
