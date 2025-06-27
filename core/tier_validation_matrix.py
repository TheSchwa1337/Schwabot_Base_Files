# -*- coding: utf - 8 -*-
""""""
""""""
# -*- coding: utf - 8 -*-
from __future__ import annotations

""""""
""""""
# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-


Tier Validation Matrix - Profit Tier Cross - Validation System

Validates profit tier combinations, cross - references tier compatibility,
and provides mathematical validation for tier transition sequences.
""""""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Import core mathematical modules
from core.unified_math_system import unified_math
from core.bit_phase_sequencer import BitPhase, BitSequence
from core.symbolic_profit_router import ProfitTier, FlipBias, SymbolicState
from core.dual_error_handler import PhaseState, SickType, SickState
from core.profit_tier_sequencer import TierAction, SymbolZone


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"
    EMERGENCY = "emergency"


class TierCompatibility(Enum):
    """Tier compatibility classifications."""
    COMPATIBLE = "compatible"
    RISKY = "risky"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass
class ValidationRule:
    """Validation rule for tier transitions."""
    from_tier: ProfitTier
    to_tier: ProfitTier
    compatibility: TierCompatibility
    confidence_threshold: float
    phase_requirements: List[PhaseState]
    risk_factor: float
    description: str


@dataclass
class ValidationResult:
    """Result of tier validation process."""
    is_valid: bool
    compatibility: TierCompatibility
    confidence_score: float
    risk_assessment: float
    warnings: List[str]
    recommendations: List[str]
    fallback_tier: Optional[ProfitTier]


@dataclass
class TierMatrix:
    """Matrix representation of tier relationships."""
    tier_combinations: Dict[Tuple[ProfitTier, ProfitTier], TierCompatibility]
    confidence_matrix: np.ndarray
    risk_matrix: np.ndarray
    phase_requirements: Dict[Tuple[ProfitTier, ProfitTier], List[PhaseState]]


class TierValidationMatrix:
    """Profit tier cross - validation system with mathematical validation."""

    def __init__(self):
        """Initialize tier validation matrix with compatibility rules."""
# Initialize bit sequencer for phase logic
        self.bit_sequencer = BitSequence(
            phase = BitPhase.BIT_4,
            short_term_logic = True,
            mid_term_logic = True,
            long_term_logic = True
        )

# Validation rules registry
        self.validation_rules: List[ValidationRule] = []

# Tier matrix for fast lookups
        self.tier_matrix: Optional[TierMatrix] = None

# Validation level setting
        self.validation_level = ValidationLevel.MODERATE

# Initialize validation rules and matrix
        self._initialize_validation_rules()
        self._build_tier_matrix()

    def _initialize_validation_rules(self):
        """Initialize standard validation rules for tier transitions."""
        rules = [
# TIER_1 transitions
            ValidationRule(
                from_tier = ProfitTier.TIER_1,
                to_tier = ProfitTier.TIER_2,
                compatibility = TierCompatibility.COMPATIBLE,
                confidence_threshold = 0.7,
                phase_requirements=[PhaseState.BIT_2, PhaseState.BIT_4],
                risk_factor = 0.2,
                description="Safe progression from basic to intermediate tier"
            ),
            ValidationRule(
                from_tier = ProfitTier.TIER_1,
                to_tier = ProfitTier.TIER_3,
                compatibility = TierCompatibility.RISKY,
                confidence_threshold = 0.8,
                phase_requirements=[PhaseState.BIT_4, PhaseState.BIT_8],
                risk_factor = 0.6,
                description="Aggressive jump from basic to advanced tier"
            ),
            ValidationRule(
                from_tier = ProfitTier.TIER_1,
                to_tier = ProfitTier.TIER_4,
                compatibility = TierCompatibility.INCOMPATIBLE,
                confidence_threshold = 0.95,
                phase_requirements=[PhaseState.BIT_8, PhaseState.BIT_42, PhaseState.BIT_256],
                risk_factor = 0.9,
                description="Extreme jump from basic to maximum tier - high risk"
            ),

# TIER_2 transitions
            ValidationRule(
                from_tier = ProfitTier.TIER_2,
                to_tier = ProfitTier.TIER_1,
                compatibility = TierCompatibility.COMPATIBLE,
                confidence_threshold = 0.5,
                phase_requirements=[PhaseState.BIT_2],
                risk_factor = 0.1,
                description="Safe downgrade from intermediate to basic tier"
            ),
            ValidationRule(
                from_tier = ProfitTier.TIER_2,
                to_tier = ProfitTier.TIER_3,
                compatibility = TierCompatibility.COMPATIBLE,
                confidence_threshold = 0.75,
                phase_requirements=[PhaseState.BIT_4, PhaseState.BIT_8],
                risk_factor = 0.3,
                description="Standard progression from intermediate to advanced tier"
            ),
            ValidationRule(
                from_tier = ProfitTier.TIER_2,
                to_tier = ProfitTier.TIER_4,
                compatibility = TierCompatibility.RISKY,
                confidence_threshold = 0.85,
                phase_requirements=[PhaseState.BIT_8, PhaseState.BIT_42],
                risk_factor = 0.7,
                description="Aggressive jump from intermediate to maximum tier"
            ),

# TIER_3 transitions
            ValidationRule(
                from_tier = ProfitTier.TIER_3,
                to_tier = ProfitTier.TIER_1,
                compatibility = TierCompatibility.RISKY,
                confidence_threshold = 0.6,
                phase_requirements=[PhaseState.BIT_2, PhaseState.BIT_4],
                risk_factor = 0.5,
                description="Major downgrade from advanced to basic tier"
            ),
            ValidationRule(
                from_tier = ProfitTier.TIER_3,
                to_tier = ProfitTier.TIER_2,
                compatibility = TierCompatibility.COMPATIBLE,
                confidence_threshold = 0.6,
                phase_requirements=[PhaseState.BIT_4],
                risk_factor = 0.2,
                description="Safe downgrade from advanced to intermediate tier"
            ),
            ValidationRule(
                from_tier = ProfitTier.TIER_3,
                to_tier = ProfitTier.TIER_4,
                compatibility = TierCompatibility.COMPATIBLE,
                confidence_threshold = 0.8,
                phase_requirements=[PhaseState.BIT_8, PhaseState.BIT_42],
                risk_factor = 0.4,
                description="Standard progression from advanced to maximum tier"
            ),

# TIER_4 transitions
            ValidationRule(
                from_tier = ProfitTier.TIER_4,
                to_tier = ProfitTier.TIER_1,
                compatibility = TierCompatibility.INCOMPATIBLE,
                confidence_threshold = 0.9,
                phase_requirements=[PhaseState.BIT_2, PhaseState.BIT_4, PhaseState.BIT_8],
                risk_factor = 0.8,
                description="Extreme downgrade from maximum to basic tier - high risk"
            ),
            ValidationRule(
                from_tier = ProfitTier.TIER_4,
                to_tier = ProfitTier.TIER_2,
                compatibility = TierCompatibility.RISKY,
                confidence_threshold = 0.7,
                phase_requirements=[PhaseState.BIT_4, PhaseState.BIT_8],
                risk_factor = 0.6,
                description="Major downgrade from maximum to intermediate tier"
            ),
            ValidationRule(
                from_tier = ProfitTier.TIER_4,
                to_tier = ProfitTier.TIER_3,
                compatibility = TierCompatibility.COMPATIBLE,
                confidence_threshold = 0.7,
                phase_requirements=[PhaseState.BIT_8],
                risk_factor = 0.3,
                description="Safe downgrade from maximum to advanced tier"
            )
        ]

        self.validation_rules = rules

    def _build_tier_matrix(self):
        """Build tier compatibility matrix for fast lookups."""
        tiers = list(ProfitTier)
        tier_count = len(tiers)

# Initialize matrices
        confidence_matrix = np.zeros((tier_count, tier_count))
        risk_matrix = np.zeros((tier_count, tier_count))
        tier_combinations = {}
        phase_requirements = {}

# Populate matrices from validation rules
        for rule in self.validation_rules:
            from_idx = tiers.index(rule.from_tier)
            to_idx = tiers.index(rule.to_tier)

# Store in matrices
            confidence_matrix[from_idx, to_idx] = rule.confidence_threshold
            risk_matrix[from_idx, to_idx] = rule.risk_factor

# Store in dictionaries
            tier_key = (rule.from_tier, rule.to_tier)
            tier_combinations[tier_key] = rule.compatibility
            phase_requirements[tier_key] = rule.phase_requirements

# Fill diagonal (same tier to same tier)
        for i in range(tier_count):
            confidence_matrix[i, i] = 1.0
            risk_matrix[i, i] = 0.0
            tier_key = (tiers[i], tiers[i])
            tier_combinations[tier_key] = TierCompatibility.COMPATIBLE
            phase_requirements[tier_key] = [PhaseState.BIT_2]

        self.tier_matrix = TierMatrix(
            tier_combinations = tier_combinations,
            confidence_matrix = confidence_matrix,
            risk_matrix = risk_matrix,
            phase_requirements = phase_requirements
        )

    def validate_tier_transition(self,
                                    from_tier: ProfitTier,
                                    to_tier: ProfitTier,
                                    current_phase: PhaseState,
                                    confidence_score: float = 0.0) -> ValidationResult:
        """"""
        Validate tier transition with comprehensive analysis.

        Args:
            from_tier: Source profit tier
            to_tier: Target profit tier
            current_phase: Current phase state
            confidence_score: Current confidence score

        Returns:
            Comprehensive validation result
        """"""
        tier_key = (from_tier, to_tier)

# Get compatibility from matrix
        compatibility = self.tier_matrix.tier_combinations.get(
            tier_key, TierCompatibility.UNKNOWN
        )

# Get required confidence threshold
        tiers = list(ProfitTier)
        from_idx = tiers.index(from_tier)
        to_idx = tiers.index(to_tier)

        required_confidence = self.tier_matrix.confidence_matrix[from_idx, to_idx]
        risk_factor = self.tier_matrix.risk_matrix[from_idx, to_idx]

# Check phase requirements
        required_phases = self.tier_matrix.phase_requirements.get(tier_key, [])
        phase_satisfied = current_phase in required_phases or not required_phases

# Calculate overall validation
        is_valid = self._calculate_validation_result(
            compatibility, confidence_score, required_confidence, phase_satisfied
        )

# Generate warnings and recommendations
        warnings = self._generate_warnings(
            compatibility, confidence_score, required_confidence, phase_satisfied, risk_factor
        )
        recommendations = self._generate_recommendations(
            from_tier, to_tier, compatibility, phase_satisfied
        )

# Determine fallback tier
        fallback_tier = self._determine_fallback_tier(from_tier, to_tier, compatibility)

        return ValidationResult(
            is_valid = is_valid,
            compatibility = compatibility,
            confidence_score = max(confidence_score, required_confidence),
            risk_assessment = risk_factor,
            warnings = warnings,
            recommendations = recommendations,
            fallback_tier = fallback_tier
        )

    def _calculate_validation_result(self,
                                        compatibility: TierCompatibility,
                                        confidence_score: float,
                                        required_confidence: float,
                                        phase_satisfied: bool) -> bool:
        """Calculate overall validation result."""
        if self.validation_level == ValidationLevel.EMERGENCY:
            return True  # Emergency mode allows all transitions

        if not phase_satisfied:
            return False

        if compatibility == TierCompatibility.INCOMPATIBLE:
            if self.validation_level == ValidationLevel.STRICT:
                return False
            elif self.validation_level == ValidationLevel.MODERATE:
                return confidence_score >= required_confidence * 1.2  # Higher threshold
            else:  # PERMISSIVE
                return confidence_score >= required_confidence

        if compatibility == TierCompatibility.RISKY:
            if self.validation_level == ValidationLevel.STRICT:
                return confidence_score >= required_confidence * 1.1  # Higher threshold
            else:
                return confidence_score >= required_confidence

# COMPATIBLE
        return confidence_score >= required_confidence

    def _generate_warnings(self,
                            compatibility: TierCompatibility,
                            confidence_score: float,
                            required_confidence: float,
                            phase_satisfied: bool,
                            risk_factor: float) -> List[str]:
        """Generate validation warnings."""
        warnings = []

        if not phase_satisfied:
            warnings.append("Phase requirements not satisfied for this transition")

        if confidence_score < required_confidence:
            warnings.append(
                f"Confidence score {
                    confidence_score:.2f} below required {
                    required_confidence:.2f}")

        if compatibility == TierCompatibility.RISKY:
            warnings.append(f"Risky transition with risk factor {risk_factor:.2f}")

        if compatibility == TierCompatibility.INCOMPATIBLE:
            warnings.append(f"Incompatible transition with high risk factor {risk_factor:.2f}")

        if risk_factor > 0.7:
            warnings.append("High risk transition - consider fallback options")

        return warnings

    def _generate_recommendations(self,
                                    from_tier: ProfitTier,
                                    to_tier: ProfitTier,
                                    compatibility: TierCompatibility,
                                    phase_satisfied: bool) -> List[str]:
        """Generate validation recommendations."""
        recommendations = []

        if not phase_satisfied:
            recommendations.append("Advance to required phase state before attempting transition")

        if compatibility == TierCompatibility.RISKY:
            recommendations.append("Consider intermediate tier steps for safer transition")

        if compatibility == TierCompatibility.INCOMPATIBLE:
            recommendations.append("Use fallback tier or staged transition approach")

# Suggest intermediate steps for large tier jumps
        tier_distance = abs(list(ProfitTier).index(to_tier) - list(ProfitTier).index(from_tier))
        if tier_distance > 1:
            recommendations.append("Consider staged transition through intermediate tiers")

        return recommendations

    def _determine_fallback_tier(self,
                                    from_tier: ProfitTier,
                                    to_tier: ProfitTier,
                                    compatibility: TierCompatibility) -> Optional[ProfitTier]:
        """Determine appropriate fallback tier."""
        if compatibility == TierCompatibility.COMPATIBLE:
            return None  # No fallback needed

        tiers = list(ProfitTier)
        from_idx = tiers.index(from_tier)
        to_idx = tiers.index(to_tier)

# Suggest intermediate tier
        if to_idx > from_idx:  # Moving up
            intermediate_idx = from_idx + 1
            if intermediate_idx < to_idx and intermediate_idx < len(tiers):
                return tiers[intermediate_idx]
        else:  # Moving down
            intermediate_idx = from_idx - 1
            if intermediate_idx > to_idx and intermediate_idx >= 0:
                return tiers[intermediate_idx]

        return from_tier  # Stay at current tier

    def validate_tier_sequence(self, tier_sequence: List[ProfitTier]) -> Dict[str, Any]:
        """"""
        Validate complete sequence of tier transitions.

        Args:
            tier_sequence: List of profit tiers in transition order

        Returns:
            Sequence validation result
        """"""
        if len(tier_sequence) < 2:
            return {
                'status': 'success',
                'sequence_valid': True,
                'total_risk': 0.0,
                'transitions': [],
                'warnings': [],
                'recommendations': []
            }

        transitions = []
        total_risk = 0.0
        all_warnings = []
        all_recommendations = []
        sequence_valid = True

        for i in range(len(tier_sequence) - 1):
            from_tier = tier_sequence[i]
            to_tier = tier_sequence[i + 1]

# Validate individual transition
            validation_result = self.validate_tier_transition(
                from_tier, to_tier, PhaseState.BIT_4, 0.8  # Default values
            )

            transitions.append({
                'from': from_tier.value,
                'to': to_tier.value,
                'valid': validation_result.is_valid,
                'compatibility': validation_result.compatibility.value,
                'risk': validation_result.risk_assessment
            })

            total_risk += validation_result.risk_assessment
            all_warnings.extend(validation_result.warnings)
            all_recommendations.extend(validation_result.recommendations)

            if not validation_result.is_valid:
                sequence_valid = False

        return {
            'status': 'success' if sequence_valid else 'error',
            'sequence_valid': sequence_valid,
            'total_risk': total_risk,
            'average_risk': total_risk / (len(tier_sequence) - 1),
            'transitions': transitions,
            'warnings': list(set(all_warnings)),  # Remove duplicates
            'recommendations': list(set(all_recommendations))
        }

    def get_optimal_tier_path(self,
                                from_tier: ProfitTier,
                                to_tier: ProfitTier) -> List[ProfitTier]:
        """"""
        Calculate optimal path between two tiers.

        Args:
            from_tier: Starting tier
            to_tier: Target tier

        Returns:
            Optimal tier transition path
        """"""
        if from_tier == to_tier:
            return [from_tier]

        tiers = list(ProfitTier)
        from_idx = tiers.index(from_tier)
        to_idx = tiers.index(to_tier)

# Direct path for adjacent tiers
        if abs(to_idx - from_idx) == 1:
            return [from_tier, to_tier]

# Calculate staged path for non - adjacent tiers
        path = [from_tier]
        current_idx = from_idx

        while current_idx != to_idx:
            if to_idx > current_idx:
                current_idx += 1
            else:
                current_idx -= 1
            path.append(tiers[current_idx])

        return path

    def set_validation_level(self, level: ValidationLevel):
        """Set validation strictness level."""
        self.validation_level = level

    def get_tier_compatibility_matrix(self) -> np.ndarray:
        """Get tier compatibility matrix for analysis."""
        return self.tier_matrix.confidence_matrix

    def get_tier_risk_matrix(self) -> np.ndarray:
        """Get tier risk matrix for analysis."""
        return self.tier_matrix.risk_matrix


# Global instance for system - wide access
tier_validation_matrix = TierValidationMatrix()


def validate_profit_tier_transition(from_tier: ProfitTier,
                                    to_tier: ProfitTier,
                                    current_phase: PhaseState = PhaseState.BIT_4,
                                    confidence_score: float = 0.8) -> ValidationResult:
    """"""
    Global function for tier transition validation.

    Args:
        from_tier: Source profit tier
        to_tier: Target profit tier
        current_phase: Current phase state
        confidence_score: Current confidence score

    Returns:
        Validation result
    """"""
    return tier_validation_matrix.validate_tier_transition(
        from_tier, to_tier, current_phase, confidence_score
    )


def get_optimal_profit_tier_path(from_tier: ProfitTier, to_tier: ProfitTier) -> List[ProfitTier]:
    """"""
    Global function for optimal tier path calculation.

    Args:
        from_tier: Starting tier
        to_tier: Target tier

    Returns:
        Optimal tier transition path
    """"""
    return tier_validation_matrix.get_optimal_tier_path(from_tier, to_tier)


""""""
Tier Validation Matrix Module

This module implements profit tier cross - validation system with mathematical validation
for tier transition sequences and compatibility analysis.

Key features:
- Comprehensive tier compatibility rules
- Risk assessment and confidence thresholds
- Phase requirement validation
- Optimal path calculation between tiers
- Staged transition recommendations
- Multiple validation strictness levels
""""""



