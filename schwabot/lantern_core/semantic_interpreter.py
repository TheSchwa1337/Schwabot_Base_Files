"""
Semantic Interpreter: LLM-Powered Meaning Extraction
===================================================

Converts fractal entropy blocks into semantic language patterns.
This is where mathematical entropy becomes readable meaning through
pattern recognition and language-driven interpretation.

Entropy Block → Semantic Patterns → Market Language
"""

from __future__ import annotations
import numpy as np
import time
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

from .entropy_generator import FractalBlock


class SemanticCategory(Enum):
    """Categories of semantic interpretation"""

    BULLISH_MOMENTUM = "bullish_momentum"
    BEARISH_PRESSURE = "bearish_pressure"
    CONSOLIDATION = "consolidation"
    VOLATILITY_SPIKE = "volatility_spike"
    TREND_REVERSAL = "trend_reversal"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    UNCERTAINTY = "uncertainty"
    BREAKOUT_POTENTIAL = "breakout_potential"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class LanguagePattern:
    """A semantic language pattern extracted from entropy"""

    primary_meaning: str
    category: SemanticCategory
    confidence_score: float
    contextual_insights: List[str]
    profit_potential: float
    risk_assessment: str
    temporal_relevance: float
    entropy_source: str
    pattern_strength: float
    harmonic_alignment: float
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "primary_meaning": self.primary_meaning,
            "category": self.category.value,
            "confidence_score": self.confidence_score,
            "contextual_insights": self.contextual_insights,
            "profit_potential": self.profit_potential,
            "risk_assessment": self.risk_assessment,
            "temporal_relevance": self.temporal_relevance,
            "entropy_source": self.entropy_source,
            "pattern_strength": self.pattern_strength,
            "harmonic_alignment": self.harmonic_alignment,
            "created_at": self.created_at,
        }


class SemanticInterpreter:
    """
    Interprets fractal entropy blocks as semantic language patterns

    Uses pattern recognition algorithms inspired by LLM processing
    to convert mathematical entropy into readable market language.
    """

    def __init__(self):
        # Semantic pattern libraries
        self.pattern_templates = self._initialize_pattern_templates()
        self.meaning_vocabulary = self._initialize_vocabulary()
        self.context_modifiers = self._initialize_context_modifiers()

        # Performance metrics
        self.interpretations_performed = 0
        self.average_confidence = 0.0
        self.pattern_hit_rate = 0.0

    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize semantic pattern templates for interpretation"""
        return {
            "high_entropy_rising": {
                "base_meaning": "Volatile upward momentum building",
                "category": SemanticCategory.BULLISH_MOMENTUM,
                "confidence_base": 0.7,
                "profit_indicator": 0.8,
                "risk_level": "moderate",
            },
            "low_entropy_stable": {
                "base_meaning": "Consolidation phase with low volatility",
                "category": SemanticCategory.CONSOLIDATION,
                "confidence_base": 0.8,
                "profit_indicator": 0.3,
                "risk_level": "low",
            },
            "fractal_convergence": {
                "base_meaning": "Pattern convergence suggesting trend continuation",
                "category": SemanticCategory.TREND_REVERSAL,
                "confidence_base": 0.6,
                "profit_indicator": 0.7,
                "risk_level": "high",
            },
            "harmonic_resonance": {
                "base_meaning": "Strong harmonic alignment indicating momentum",
                "category": SemanticCategory.BREAKOUT_POTENTIAL,
                "confidence_base": 0.75,
                "profit_indicator": 0.85,
                "risk_level": "moderate",
            },
            "phase_disruption": {
                "base_meaning": "Phase patterns suggesting market uncertainty",
                "category": SemanticCategory.UNCERTAINTY,
                "confidence_base": 0.5,
                "profit_indicator": 0.2,
                "risk_level": "high",
            },
            "stability_breakdown": {
                "base_meaning": "Stability index decline indicating volatility increase",
                "category": SemanticCategory.VOLATILITY_SPIKE,
                "confidence_base": 0.65,
                "profit_indicator": 0.6,
                "risk_level": "very_high",
            },
        }

    def _initialize_vocabulary(self) -> Dict[str, List[str]]:
        """Initialize semantic vocabulary for meaning construction"""
        return {
            "momentum_words": [
                "surge",
                "momentum",
                "acceleration",
                "thrust",
                "drive",
                "push",
                "force",
                "velocity",
                "impulse",
                "pressure",
                "energy",
                "power",
                "strength",
            ],
            "stability_words": [
                "stable",
                "steady",
                "consistent",
                "balanced",
                "calm",
                "quiet",
                "peaceful",
                "consolidated",
                "anchored",
                "grounded",
                "solid",
                "firm",
                "secure",
            ],
            "volatility_words": [
                "volatile",
                "chaotic",
                "turbulent",
                "erratic",
                "unpredictable",
                "wild",
                "frantic",
                "unstable",
                "fluctuating",
                "oscillating",
                "swinging",
                "dancing",
            ],
            "trend_words": [
                "trending",
                "directional",
                "flowing",
                "streaming",
                "channeling",
                "moving",
                "progressing",
                "advancing",
                "evolving",
                "developing",
                "unfolding",
                "emerging",
            ],
            "reversal_words": [
                "reversing",
                "turning",
                "shifting",
                "pivoting",
                "changing",
                "flipping",
                "rotating",
                "inverting",
                "switching",
                "transitioning",
            ],
        }

    def _initialize_context_modifiers(self) -> Dict[str, List[str]]:
        """Initialize contextual modifiers for meaning enhancement"""
        return {
            "intensity_modifiers": [
                "strongly",
                "intensely",
                "significantly",
                "dramatically",
                "moderately",
                "subtly",
                "gradually",
                "rapidly",
                "slowly",
                "aggressively",
            ],
            "temporal_modifiers": [
                "currently",
                "presently",
                "immediately",
                "gradually",
                "eventually",
                "temporarily",
                "persistently",
                "briefly",
                "continuously",
                "intermittently",
            ],
            "probability_modifiers": [
                "likely",
                "probably",
                "possibly",
                "potentially",
                "certainly",
                "definitely",
                "maybe",
                "perhaps",
                "presumably",
                "apparently",
            ],
        }

    def _analyze_entropy_characteristics(
        self, fractal_block: FractalBlock
    ) -> Dict[str, float]:
        """Analyze key characteristics of the entropy block"""
        characteristics = {}

        # Entropy level analysis
        characteristics["entropy_level"] = fractal_block.entropy_score
        characteristics["stability_level"] = fractal_block.stability_index
        # Normalize
        characteristics["recursion_depth"] = fractal_block.recursion_depth / 100.0

        # Convergence pattern analysis
        if fractal_block.convergence_patterns:
            convergence_trend = np.mean(np.diff(fractal_block.convergence_patterns))
            characteristics["convergence_trend"] = convergence_trend
            characteristics["convergence_volatility"] = np.std(
                fractal_block.convergence_patterns
            )
        else:
            characteristics["convergence_trend"] = 0.0
            characteristics["convergence_volatility"] = 0.0

        # Harmonic analysis
        if fractal_block.harmonic_frequencies:
            characteristics["harmonic_strength"] = np.mean(
                fractal_block.harmonic_frequencies
            )
            characteristics["harmonic_variance"] = np.var(
                fractal_block.harmonic_frequencies
            )
        else:
            characteristics["harmonic_strength"] = 0.0
            characteristics["harmonic_variance"] = 0.0

        # Phase relationship analysis
        if fractal_block.phase_relationships:
            characteristics["phase_coherence"] = 1.0 / (
                1.0 + np.std(fractal_block.phase_relationships)
            )
            characteristics["phase_alignment"] = np.mean(
                np.cos(fractal_block.phase_relationships)
            )
        else:
            characteristics["phase_coherence"] = 0.5
            characteristics["phase_alignment"] = 0.0

        # Temporal signature analysis
        characteristics["temporal_complexity"] = (
            len(fractal_block.temporal_signature) / 50.0
        )  # Normalize

        return characteristics

    def _select_semantic_pattern(self, characteristics: Dict[str, float]) -> str:
        """Select the most appropriate semantic pattern based on characteristics"""
        pattern_scores = {}

        # Score each pattern template
        for pattern_name, template in self.pattern_templates.items():
            score = self._calculate_pattern_score(pattern_name, characteristics)
            pattern_scores[pattern_name] = score

        # Return pattern with highest score
        best_pattern = max(pattern_scores, key=pattern_scores.get)
        if pattern_scores[best_pattern] > 0.1:  # Minimum threshold
            return best_pattern
        else:
            return "phase_disruption"  # Default uncertainty pattern

    def _calculate_pattern_score(
        self, pattern_name: str, characteristics: Dict[str, float]
    ) -> float:
        """Calculate how well characteristics match a pattern."""
        # Pattern-specific scoring logic
        if pattern_name == "high_entropy_rising":
            score = (
                characteristics["entropy_level"] * 0.3
                + max(0, characteristics["convergence_trend"]) * 0.4
                + characteristics["entropy_variance"] * 0.3
            )
        elif pattern_name == "low_entropy_stable":
            score = (
                (1.0 - characteristics["entropy_level"]) * 0.4
                + (1.0 - characteristics["convergence_volatility"]) * 0.3
                + characteristics["recursion_depth"] * 0.3
            )
        elif pattern_name == "fractal_convergence":
            score = (
                characteristics["convergence_volatility"] * 0.4
                + characteristics["phase_coherence"] * 0.3
                + (1.0 - characteristics["entropy_variance"]) * 0.3
            )
        elif pattern_name == "harmonic_resonance":
            score = (
                characteristics["harmonic_strength"] * 0.5
                + characteristics["phase_coherence"] * 0.3
                + characteristics["harmonic_variance"] * 0.2
            )
        elif pattern_name == "phase_disruption":
            score = (
                (1.0 - characteristics["phase_coherence"]) * 0.4
                + (1.0 - characteristics["phase_alignment"]) * 0.3
                + characteristics["entropy_variance"] * 0.3
            )
        elif pattern_name == "stability_breakdown":
            score = (
                characteristics["entropy_variance"] * 0.4
                + abs(characteristics["convergence_trend"]) * 0.3
                + (1.0 - characteristics["recursion_depth"]) * 0.3
            )
        else:
            # Default scoring
            score = (
                characteristics["entropy_level"] * 0.5
                + characteristics["entropy_variance"] * 0.5
            )

        return max(0.0, min(1.0, score))

    def _construct_semantic_meaning(
        self,
        pattern_name: str,
        characteristics: Dict[str, float],
        price_context: Dict[str, float],
    ) -> str:
        """Construct semantic meaning using vocabulary and context"""
        template = self.pattern_templates[pattern_name]
        base_meaning = template["base_meaning"]

        # Select contextual words based on characteristics
        if characteristics["entropy_level"] > 0.6:
            vocab_category = "volatility_words"
        elif characteristics["stability_level"] > 0.7:
            vocab_category = "stability_words"
        elif characteristics["convergence_trend"] > 0.1:
            vocab_category = "momentum_words"
        elif abs(characteristics["convergence_trend"]) > 0.05:
            vocab_category = "trend_words"
        else:
            vocab_category = "reversal_words"

        # Select intensity modifier
        intensity_level = int(characteristics["entropy_level"] * 4)
        intensity_modifier = self.context_modifiers["intensity_modifiers"][
            min(intensity_level, len(self.context_modifiers["intensity_modifiers"]) - 1)
        ]

        # Select temporal modifier based on recursion depth
        temporal_level = int(characteristics["recursion_depth"] * 4)
        temporal_modifier = self.context_modifiers["temporal_modifiers"][
            min(temporal_level, len(self.context_modifiers["temporal_modifiers"]) - 1)
        ]

        # Select vocabulary word
        vocab_words = self.meaning_vocabulary[vocab_category]
        word_index = int(characteristics["harmonic_strength"] * len(vocab_words)) % len(
            vocab_words
        )
        vocab_word = vocab_words[word_index]

        # Construct enhanced meaning
        if price_context.get("price", 0) > 0:
            price_context_str = f"at ${price_context['price']:.2f}"
        else:
            price_context_str = "in current market"

        enhanced_meaning = (
            f"{intensity_modifier.capitalize()} {vocab_word} {base_meaning.lower()} "
            f"with {temporal_modifier} implications {price_context_str}"
        )

        return enhanced_meaning

    def _generate_contextual_insights(
        self, characteristics: Dict[str, float], pattern_name: str
    ) -> List[str]:
        """Generate contextual insights based on entropy characteristics"""
        insights = []

        # Entropy-based insights
        if characteristics["entropy_level"] > 0.8:
            insights.append(
                "High entropy suggests significant market uncertainty and potential for large moves"
            )
        elif characteristics["entropy_level"] < 0.2:
            insights.append(
                "Low entropy indicates market stability and predictable price action"
            )

        # Stability-based insights
        if characteristics["stability_level"] > 0.8:
            insights.append("Strong stability index supports trend continuation")
        elif characteristics["stability_level"] < 0.3:
            insights.append("Low stability suggests increased volatility ahead")

        # Convergence-based insights
        if characteristics["convergence_trend"] > 0.1:
            insights.append("Convergence patterns indicate building momentum")
        elif characteristics["convergence_trend"] < -0.1:
            insights.append("Divergence patterns suggest potential reversal")

        # Harmonic-based insights
        if characteristics["harmonic_strength"] > 0.7:
            insights.append("Strong harmonic resonance supports directional movement")
        elif characteristics["harmonic_variance"] > 0.6:
            insights.append("Harmonic discord indicates conflicting market forces")

        # Phase-based insights
        if characteristics["phase_coherence"] > 0.7:
            insights.append("Phase coherence suggests aligned market dynamics")
        elif characteristics["phase_coherence"] < 0.3:
            insights.append("Phase disruption indicates potential volatility spikes")

        # Pattern-specific insights
        template = self.pattern_templates[pattern_name]
        if template["category"] == SemanticCategory.BULLISH_MOMENTUM:
            insights.append("Bullish momentum patterns favor long positions")
        elif template["category"] == SemanticCategory.BEARISH_PRESSURE:
            insights.append("Bearish pressure patterns favor short positions")
        elif template["category"] == SemanticCategory.UNCERTAINTY:
            insights.append("Uncertainty patterns recommend reduced position sizes")

        return insights[:5]  # Limit to top 5 insights

    def _calculate_profit_potential(
        self, characteristics: Dict[str, float], template: Dict[str, Any]
    ) -> float:
        """Calculate profit potential based on pattern characteristics"""
        base_potential = template["profit_indicator"]

        # Adjust based on characteristics
        # Higher entropy = higher potential
        entropy_factor = characteristics["entropy_level"]
        # Higher stability = more reliable
        stability_factor = characteristics["stability_level"]
        # Higher harmonic = stronger signal
        harmonic_factor = characteristics["harmonic_strength"]

        # Combined profit potential calculation
        profit_potential = (
            base_potential * 0.4
            + entropy_factor * 0.2
            + stability_factor * 0.2
            + harmonic_factor * 0.2
        )

        return min(profit_potential, 1.0)  # Cap at 1.0

    def _assess_risk_level(
        self, characteristics: Dict[str, float], template: Dict[str, Any]
    ) -> str:
        """Assess risk level based on pattern characteristics"""
        base_risk = template["risk_level"]

        # Risk factors
        volatility_risk = characteristics["convergence_volatility"]
        stability_risk = 1.0 - characteristics["stability_level"]
        phase_risk = 1.0 - characteristics["phase_coherence"]

        # Calculate overall risk score
        risk_score = (volatility_risk + stability_risk + phase_risk) / 3.0

        # Adjust base risk level
        if risk_score > 0.7:
            if base_risk in ["low", "minimal"]:
                return "moderate"
            elif base_risk == "moderate":
                return "high"
            else:
                return "very_high"
        elif risk_score < 0.3:
            if base_risk in ["high", "very_high"]:
                return "moderate"
            elif base_risk == "moderate":
                return "low"
            else:
                return "minimal"
        else:
            return base_risk

    def _calculate_temporal_relevance(self, characteristics: Dict[str, float]) -> float:
        """Calculate temporal relevance of the interpretation"""
        # Based on recursion depth and convergence stability
        # Deeper = longer term relevance
        depth_factor = characteristics["recursion_depth"]
        # More stable = longer relevance
        stability_factor = characteristics["stability_level"]
        # Lower entropy = longer relevance
        entropy_factor = 1.0 - characteristics["entropy_level"]

        temporal_relevance = (
            depth_factor * 0.4 + stability_factor * 0.3 + entropy_factor * 0.3
        )

        return min(temporal_relevance, 1.0)

    def interpret_entropy_block(
        self,
        fractal_block: FractalBlock,
        price_context: Dict[str, float],
        market_context: Dict[str, Any],
    ) -> LanguagePattern:
        """
        Convert fractal entropy block into semantic language pattern

        This is the core transformation: Entropy Block → Semantic Meaning
        """
        time.time()

        # Analyze entropy characteristics
        characteristics = self._analyze_entropy_characteristics(fractal_block)

        # Select semantic pattern
        pattern_name = self._select_semantic_pattern(characteristics)
        template = self.pattern_templates[pattern_name]

        # Construct semantic meaning
        primary_meaning = self._construct_semantic_meaning(
            pattern_name, characteristics, price_context
        )

        # Generate contextual insights
        contextual_insights = self._generate_contextual_insights(
            characteristics, pattern_name
        )

        # Calculate metrics
        confidence_score = template["confidence_base"] * characteristics.get(
            "phase_coherence", 0.5
        )
        profit_potential = self._calculate_profit_potential(characteristics, template)
        risk_assessment = self._assess_risk_level(characteristics, template)
        temporal_relevance = self._calculate_temporal_relevance(characteristics)

        # Calculate pattern strength
        pattern_strength = (
            characteristics["entropy_level"] * 0.3
            + characteristics["stability_level"] * 0.3
            + characteristics["harmonic_strength"] * 0.4
        )

        # Calculate harmonic alignment
        harmonic_alignment = characteristics.get("phase_coherence", 0.5)

        # Create language pattern
        language_pattern = LanguagePattern(
            primary_meaning=primary_meaning,
            category=template["category"],
            confidence_score=confidence_score,
            contextual_insights=contextual_insights,
            profit_potential=profit_potential,
            risk_assessment=risk_assessment,
            temporal_relevance=temporal_relevance,
            entropy_source=fractal_block.source_hash[:16],
            pattern_strength=pattern_strength,
            harmonic_alignment=harmonic_alignment,
        )

        # Update performance metrics
        self.interpretations_performed += 1
        self.average_confidence = (
            self.average_confidence * (self.interpretations_performed - 1)
            + confidence_score
        ) / self.interpretations_performed

        return language_pattern

    def get_interpretation_statistics(self) -> Dict[str, Any]:
        """Get semantic interpretation performance statistics"""
        return {
            "total_interpretations": self.interpretations_performed,
            "average_confidence_score": self.average_confidence,
            "pattern_template_count": len(self.pattern_templates),
            "vocabulary_size": sum(
                len(words) for words in self.meaning_vocabulary.values()
            ),
            "semantic_categories": [cat.value for cat in SemanticCategory],
        }
