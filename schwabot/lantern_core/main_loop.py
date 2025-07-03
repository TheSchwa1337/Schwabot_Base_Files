"""
Lantern Main Loop: Runtime Integration Hub.
==========================================

Coordinates all LanternCore components to create a unified runtime
that continuously processes market data through the semantic oracle.

The Complete Loop:
1. Price tick → SHA-256 hash
2. Hash → Entropy block generation
3. Entropy → Semantic interpretation
4. Semantic → Truth score validation
5. Memory → Strategy adjustment
6. Recurse.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from gatekeeper.recursive_gate_stack import RecursiveGateStack

from .hash_memory import HashMemoryDatabase
from .lantern_eye import HashBlock, LanternEye, SemanticInterpretation
from .nexus_thought_core import NexusThoughtCore, ZalgoLockState
from .semantic_interpreter import SemanticInterpreter
from .truth_scorer import TruthScorer


@dataclass
class LanternProcessingResult:
    """Complete processing result from Lantern loop."""

    hash_block: HashBlock
    semantic_interpretation: Optional[SemanticInterpretation]
    market_signals: List[str]
    profit_recommendations: List[str]
    risk_warnings: List[str]
    confidence_score: float
    processing_time: float
    timestamp: float
    truth_score: Optional[Dict] = None
    gate_validation_result: bool = False
    nexus_core_result: Optional[Dict] = None

    def to_dict(self: LanternProcessingResult) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            "hash_value": self.hash_block.hash_value,
            "semantic_interpretation": (
                self.semantic_interpretation.to_dict()
                if self.semantic_interpretation
                else None
            ),
            "market_signals": self.market_signals,
            "profit_recommendations": self.profit_recommendations,
            "risk_warnings": self.risk_warnings,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp,
            "truth_score": self.truth_score,
            "gate_validation_result": self.gate_validation_result,
            "nexus_core_result": self.nexus_core_result,
        }


class LanternMainLoop:
    """
    Runtime integration hub for the complete Lantern Eye system.

    Coordinates all components to create a unified semantic oracle
    that continuously reads market language and builds profitable memory.
    """

    def __init__(
        self: LanternMainLoop,
        processing_interval: float = 1.0,
        memory_save_interval: float = 300.0,
        max_processing_history: int = 1000,
    ) -> None:
        """Initialize the Lantern Main Loop with all components."""
        # Core components
        self.lantern_eye = LanternEye()
        self.nexus_core = NexusThoughtCore(seed=33, scale=0.01)
        self.semantic_interpreter = SemanticInterpreter()
        self.truth_scorer = TruthScorer()
        self.hash_memory = HashMemoryDatabase()

        # Runtime configuration
        self.processing_interval = processing_interval
        self.memory_save_interval = memory_save_interval
        self.max_processing_history = max_processing_history

        # Runtime state
        self.is_running = False
        self.processing_history: List[LanternProcessingResult] = []
        self.last_memory_save = time.time()
        self.loop_iterations = 0
        self.total_processing_time = 0.0

        # Market data callback
        self.market_data_callback: Optional[Callable[[], Dict[str, float]]] = None
        self.interpretation_callback: Optional[
            Callable[[LanternProcessingResult], None]
        ] = None

        # Performance tracking
        self.successful_interpretations = 0
        self.failed_interpretations = 0
        self.average_confidence = 0.0

        # Gatekeeper system
        self.gate_stack: Optional[RecursiveGateStack] = None

    def set_market_data_source(
        self: LanternMainLoop, callback: Callable[[], Dict[str, float]]
    ) -> None:
        """Set callback function to get current market data."""
        self.market_data_callback = callback

    def set_interpretation_handler(
        self: LanternMainLoop, callback: Callable[[LanternProcessingResult], None]
    ) -> None:
        """Set callback function to handle interpretation results."""
        self.interpretation_callback = callback

    def process_single_tick(
        self: LanternMainLoop,
        market_data: Dict[str, float],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> LanternProcessingResult:
        """
        Process a single market tick through the complete Lantern pipeline.
        """
        start_time = time.time()

        # Step 1: Process through Lantern Eye
        hash_block = self.lantern_eye.process_price_tick(
            market_data, additional_context
        )

        # Step 2: Process through Nexus Thought Core
        price_input = market_data.get("price", 0.0)
        nexus_result = self.nexus_core.nexus_omega_exec(
            price_input, hash_block.hash_value
        )

        # Step 3: Create semantic interpretation
        semantic_interpretation = None
        if hash_block.semantic_interpretation:
            semantic_interpretation = SemanticInterpretation(
                primary_meaning=hash_block.semantic_interpretation.primary_meaning,
                confidence_score=hash_block.semantic_interpretation.confidence_score,
                language_patterns=[hash_block.semantic_interpretation],
                contextual_insights=hash_block.semantic_interpretation.contextual_insights,
                profit_potential=hash_block.semantic_interpretation.profit_potential,
                risk_assessment=hash_block.semantic_interpretation.risk_assessment,
                temporal_relevance=hash_block.semantic_interpretation.temporal_relevance,
                correlation_strength=hash_block.semantic_interpretation.correlation_strength,
            )

        # Step 4: Validate through Truth Scorer
        truth_score = None
        if semantic_interpretation:
            truth_score = self.truth_scorer.validate_semantic_interpretation(
                semantic_interpretation.to_dict(), price_input, additional_context
            )

        # Step 5: Gatekeeper validation
        gate_validation_result = self._validate_through_gates(
            nexus_result, hash_block, market_data
        )

        # Step 6: Generate actionable outputs
        market_signals = self._generate_market_signals(hash_block)
        profit_recommendations = self._generate_profit_recommendations(hash_block)
        risk_warnings = self._generate_risk_warnings(hash_block)

        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(
            hash_block, truth_score
        )

        processing_time = time.time() - start_time

        # Create processing result
        result = LanternProcessingResult(
            hash_block=hash_block,
            semantic_interpretation=semantic_interpretation,
            market_signals=market_signals,
            profit_recommendations=profit_recommendations,
            risk_warnings=risk_warnings,
            confidence_score=confidence_score,
            processing_time=processing_time,
            timestamp=time.time(),
            truth_score=truth_score,
            gate_validation_result=gate_validation_result,
            nexus_core_result=nexus_result,
        )

        # Update performance tracking
        if semantic_interpretation:
            self.successful_interpretations += 1
            self.average_confidence = (
                self.average_confidence * (self.successful_interpretations - 1)
                + confidence_score
            ) / self.successful_interpretations
        else:
            self.failed_interpretations += 1

        # Store in processing history
        self.processing_history.append(result)
        if len(self.processing_history) > self.max_processing_history:
            self.processing_history.pop(0)

        # Update runtime metrics
        self.loop_iterations += 1
        self.total_processing_time += processing_time

        return result

    def _validate_through_gates(
        self: LanternMainLoop,
        nexus_result: Dict,
        hash_block: HashBlock,
        market_data: Dict[str, float],
    ) -> bool:
        """Validate processing result through gatekeeper system."""
        if not nexus_result or "zalgo_lock" not in nexus_result:
            return False

        # Extract ZALGO lock state
        zalgo_data = nexus_result["zalgo_lock"]
        zalgo_lock = ZalgoLockState(
            fractal_containment=zalgo_data["fractal_containment"],
            drift_suppression=zalgo_data["drift_suppression"],
            collapse_stability=zalgo_data["collapse_stability"],
            recursive_bound=zalgo_data["recursive_bound"],
            sigmoid_collapse=zalgo_data["sigmoid_collapse"],
            qutrit_state=zalgo_data["qutrit_state"],
            locked=zalgo_data["locked"],
        )

        # Create profit band data
        profit_band = {
            "score": (
                hash_block.semantic_interpretation.profit_potential
                if hash_block.semantic_interpretation
                else 0.5
            ),
            "zone": 1,  # Default zone
            "confidence": (
                hash_block.semantic_interpretation.confidence_score
                if hash_block.semantic_interpretation
                else 0.5
            ),
        }

        # Gatekeeper stack processing
        if self.gate_stack:
            return self.gate_stack.process(market_data, zalgo_lock, profit_band)

        return zalgo_lock.locked  # Fallback to simple lock state

    def _generate_market_signals(
        self: LanternMainLoop, hash_block: HashBlock
    ) -> List[str]:
        """Generate market signals from semantic interpretation."""
        if not hash_block.semantic_interpretation:
            return []

        signals = []
        interpretation = hash_block.semantic_interpretation
        if interpretation.confidence_score > 0.7:
            signals.append(f"HIGH_CONFIDENCE:{interpretation.primary_meaning}")
        if interpretation.profit_potential > 0.6:
            signals.append("STRONG_PROFIT_POTENTIAL")
        if interpretation.risk_assessment in ["LOW", "MINIMAL"]:
            signals.append("LOW_RISK_OPPORTUNITY")

        return signals

    def _generate_profit_recommendations(
        self: LanternMainLoop, hash_block: HashBlock
    ) -> List[str]:
        """Generate profit recommendations based on insights."""
        if not hash_block.semantic_interpretation:
            return []

        recommendations = []
        insights = hash_block.semantic_interpretation.contextual_insights
        for insight in insights:
            if "potential" in insight.lower():
                recommendations.append(f"ACTIONABLE_INSIGHT:{insight}")

        return recommendations

    def _generate_risk_warnings(
        self: LanternMainLoop, hash_block: HashBlock
    ) -> List[str]:
        """Generate risk warnings from risk assessment."""
        if (
            not hash_block.semantic_interpretation
            or not hash_block.semantic_interpretation.risk_assessment
        ):
            return []

        risk = hash_block.semantic_interpretation.risk_assessment
        if risk in ["HIGH", "CRITICAL", "EXTREME"]:
            return [f"RISK_ALERT:{risk}"]
        return []

    def _calculate_overall_confidence(
        self: LanternMainLoop,
        hash_block: HashBlock,
        truth_score: Optional[Dict],
    ) -> float:
        """Calculate a blended confidence score from multiple factors."""
        if not hash_block.semantic_interpretation:
            return 0.0

        sem_conf = hash_block.semantic_interpretation.confidence_score
        
        truth_score_val = 0.0
        truth_conf = 0.0
        if truth_score:
            truth_score_val = truth_score.get("score", 0.0)
            truth_conf = truth_score.get("confidence", 0.0)

        # Weighted average
        weights = {"sem_conf": 0.5, "truth_score": 0.3, "truth_conf": 0.2}
        confidence = (
            sem_conf * weights["sem_conf"]
            + truth_score_val * weights["truth_score"]
            + truth_conf * weights["truth_conf"]
        )
        return confidence

    async def run_continuous_loop(self: LanternMainLoop) -> None:
        """Run the main processing loop continuously."""
        print("Lantern Main Loop starting...")
        self.is_running = True

        while self.is_running:
            start_time = time.time()

                if self.market_data_callback:
                    market_data = self.market_data_callback()
                if market_data:
                result = self.process_single_tick(market_data)

                if self.interpretation_callback:
                    self.interpretation_callback(result)
                else:
                    self._default_interpretation_handler(result)

                # Save memory periodically
            if time.time() - self.last_memory_save > self.memory_save_interval:
                self.hash_memory.save_to_disk()
                self.last_memory_save = time.time()

            # Maintain processing interval
            elapsed = time.time() - start_time
            await asyncio.sleep(max(0, self.processing_interval - elapsed))

        print("Lantern Main Loop stopped.")

    def stop_loop(self: LanternMainLoop) -> None:
        """Stop the continuous processing loop."""
        self.is_running = False

    def _generate_mock_market_data(self: LanternMainLoop) -> Dict[str, float]:
        """Generate mock market data for testing."""
        price = 100 + random.uniform(-1.5, 1.5)
        volume = 10 + random.uniform(-2, 2)
        return {"price": price, "volume": volume, "timestamp": time.time()}

    def _default_interpretation_handler(
        self: LanternMainLoop, result: LanternProcessingResult
    ) -> None:
        """Print interpretation results to the console."""
        if result.semantic_interpretation:
            print(
                f"[{datetime.now().isoformat()}] "
                f"Hash: {result.hash_block.hash_value[:10]}... "
                f"Meaning: {result.semantic_interpretation.primary_meaning} "
                f"(Conf: {result.confidence_score:.2f})"
            )

    def get_recent_interpretations(
        self: LanternMainLoop, count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the most recent processing results."""
        return [r.to_dict() for r in self.processing_history[-count:]]

    def get_performance_analytics(self: LanternMainLoop) -> Dict[str, Any]:
        """Get performance analytics for the main loop."""
        avg_processing_time = (
            self.total_processing_time / self.loop_iterations
            if self.loop_iterations > 0
            else 0
        )
        return {
            "is_running": self.is_running,
            "loop_iterations": self.loop_iterations,
            "successful_interpretations": self.successful_interpretations,
            "failed_interpretations": self.failed_interpretations,
            "success_rate": (
                self.successful_interpretations / self.loop_iterations
                if self.loop_iterations > 0
                else 0
            ),
            "average_confidence": self.average_confidence,
            "average_processing_time_ms": avg_processing_time * 1000,
            "last_memory_save": datetime.fromtimestamp(
                self.last_memory_save
            ).isoformat(),
        }

    def export_processing_history(
        self: LanternMainLoop, filename: Optional[str] = None
    ) -> str:
        """Export the full processing history to a JSON file."""
        if filename is None:
            filename = f"lantern_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        history_dicts = [r.to_dict() for r in self.processing_history]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(history_dicts, f, indent=2)

        print(f"Processing history exported to {filename}")
        return filename


def demo_lantern_main_loop() -> Dict[str, Any]:
    """Demonstrate the Lantern Main Loop functionality."""
    print("--- Lantern Main Loop Demonstration ---")
    loop = LanternMainLoop(processing_interval=0.1)

    # Set up mock data source
    loop.set_market_data_source(loop._generate_mock_market_data)

    async def run_demo():
        # Run loop for a short period
        loop_task = asyncio.create_task(loop.run_continuous_loop())
        await asyncio.sleep(5)
        loop.stop_loop()
        await loop_task

    # Run the asynchronous demo
    asyncio.run(run_demo())

    # Get analytics
    analytics = loop.get_performance_analytics()
    print("\n--- Performance Analytics ---")
    for key, value in analytics.items():
        print(f"{key}: {value}")

    # Export history
    loop.export_processing_history()

    print("\n--- Lantern Main Loop Demo Complete ---")
    return analytics


if __name__ == "__main__":
    demo_lantern_main_loop()