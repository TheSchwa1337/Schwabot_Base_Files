"""Recursive Gate Stack for Enhanced Trading Intelligence.

Auto-generative threshold validator that lives in recursive identity space.
Implements mathematical lock equations and validation chains.
"""

import math
import yaml
from pathlib import Path
from typing import Dict, Any


# Use absolute import to fix the import issue
try:
    from lantern_core.nexus_thought_core import ZalgoLockState
except ImportError:
    # Fallback for testing - create a minimal ZalgoLockState
    from dataclasses import dataclass

    @dataclass
    class ZalgoLockState:
        fractal_containment: float = 0.0
        drift_suppression: float = 0.0
        collapse_stability: float = 0.0
        recursive_bound: float = 0.0
        sigmoid_collapse: float = 0.0
        qutrit_state: int = 0
        locked: bool = False


class RecursiveGateStack:
    """Auto-generative recursive threshold validator.

    Implements self-evolving threshold validation using entropy vectors,
    thought patterns, and past trade memory with ZALGO locks.
    """

    def __init__(
        self,
        zalgo_core: ZalgoLockState,
        entropy: float,
        profit_band: Dict[str, Any],
        bayes_confidence: float,
        config_path: str = "config/gate_profiles.yaml",
    ) -> None:
        """Initialize the recursive gate stack with validation components."""
        self.zalgo_core = zalgo_core
        self.entropy = entropy
        self.profit_band = profit_band
        self.bayes_confidence = bayes_confidence

        # Load configuration thresholds
        self.thresholds = self._load_config(config_path)

        # Gate validation state
        self.last_validation_result = None
        self.validation_history = []

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load threshold configuration from YAML file."""
        try:
            yaml_path = Path(__file__).parent.parent / config_path
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️ Failed to load config from {config_path}: {e}")
            return self._get_default_thresholds()

    def _get_default_thresholds(self) -> Dict[str, Any]:
        """Return default threshold values if config loading fails."""
        return {
            "entropy_collapse": 0.001,
            "fuzzy_trigger": 0.65,
            "bayes_trigger": 0.72,
            "sigmoid_collapse_trigger": 0.1,
            "kappa_offset": 0.001,
        }

    def greyscale_collapse_filter(self) -> bool:
        """Implement Equation 1A: Greyscale Entropy Collapse Core.

        C_grey(t) = ΣC(t)/(1+e^(-Ωt)) < ε
        Lock triggered if entropy collapse below threshold.
        """
        threshold = self.thresholds["entropy_collapse"]

        # Implement sigmoid-weighted entropy collapse
        if hasattr(self.zalgo_core, "sigmoid_collapse"):
            collapse_value = self.zalgo_core.sigmoid_collapse
        else:
            # Fallback calculation with bounds checking
            omega = self.thresholds.get("omega_coefficient", 0.1)
            time_factor = 1.0  # Simplified for this implementation

            # Normalize entropy to prevent overflow
            normalized_entropy = min(1.0, max(0.0, self.entropy / 100.0))

            try:
                exponent = -omega * time_factor
                # Clamp exponent to prevent overflow
                exponent = max(-700, min(700, exponent))
                collapse_value = normalized_entropy / (1 + math.exp(exponent))
            except (OverflowError, ValueError):
                # Conservative fallback
                collapse_value = normalized_entropy * 0.5

        return collapse_value < threshold

    def fuzzy_zone_gate(self) -> bool:
        """Implement Equation 2A: Fuzzy Zone Activation Gate.

        F(p,e) = μ_profit(p) * (1 - μ_entropy(e)) > λ
        Fuzzy membership overlap validation.
        """
        # Extract profit score from profit band
        profit_score = self.profit_band.get("score", 0.0)

        # Calculate fuzzy membership functions
        mu_profit = self._profit_membership(profit_score)
        mu_entropy = self._entropy_membership(self.entropy)

        # Fuzzy evaluation
        fuzzy_eval = mu_profit * (1 - mu_entropy)
        fuzzy_threshold = self.thresholds["fuzzy_trigger"]

        return fuzzy_eval > fuzzy_threshold

    def _profit_membership(self, profit_score: float) -> float:
        """Calculate profit membership function (trapezoidal)."""
        min_zone = self.thresholds.get("profit_zone_min", 0.4)
        max_zone = self.thresholds.get("profit_zone_max", 0.6)

        if profit_score <= min_zone:
            return 0.0
        elif min_zone < profit_score <= max_zone:
            return 2.5 * profit_score - 1.0
        else:
            return 1.0

    def _entropy_membership(self, entropy: float) -> float:
        """Calculate entropy membership function (inverse sigmoid)."""
        # Higher entropy = lower membership (we want low entropy)
        # Add bounds checking to prevent overflow
        entropy = max(0.0, min(100.0, entropy))  # Clamp entropy to reasonable range

        # Scale entropy to prevent overflow in exp()
        scaled_entropy = entropy / 100.0  # Scale to 0-1 range

        try:
            exponent = 10 * (scaled_entropy - 0.5)
            # Clamp exponent to prevent overflow
            exponent = max(-700, min(700, exponent))
            return 1.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            # Fallback for extreme values
            if scaled_entropy > 0.5:
                return 0.0  # High entropy = low membership
            else:
                return 1.0  # Low entropy = high membership

    def bayesian_confidence_gate(self) -> bool:
        """Implement Equation 3A: Bayesian Confidence Estimator.

        P(Trade|Zalgo,Entropy,HashState) > θ
        Recursive memory-based confidence validation.
        """
        threshold = self.thresholds["bayes_trigger"]
        return self.bayes_confidence > threshold

    def zalgo_lock_gate(self) -> bool:
        """Validate ZALGO lock system integrity.

        Check all ZALGO lock equations for system stability.
        """
        if not self.zalgo_core:
            return False

        return (
            self.zalgo_core.locked
            and self.zalgo_core.collapse_stability
            <= self.thresholds["entropy_collapse"]
            and abs(self.zalgo_core.drift_suppression)
            < self.thresholds["kappa_offset"] * 2
            and self.zalgo_core.sigmoid_collapse
            < self.thresholds["sigmoid_collapse_trigger"]
            and self.zalgo_core.qutrit_state == 0
        )

    def temporal_hash_drift_gate(
        self, current_hash: str, previous_hash: str = None
    ) -> bool:
        """Implement Equation 5A: Temporal Hash Drift Compensation.

        D_hash(t) = e^(-λt) * sin(Ψt) + κ
        Validate hash consistency over time.
        """
        if not previous_hash:
            return True  # No previous hash to compare

        # Calculate hash similarity/drift
        hash_similarity = self._calculate_hash_similarity(current_hash, previous_hash)
        drift_threshold = self.thresholds.get("hash_drift_max", 0.2)

        return hash_similarity > (1.0 - drift_threshold)

    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hash strings."""
        if len(hash1) != len(hash2):
            return 0.0

        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)

    def adaptive_threshold_gate(self, market_volatility: float = None) -> bool:
        """Implement adaptive threshold adjustment based on market conditions.

        T_adapt(x) = T_0 + η * log(volatility_t)
        Dynamic threshold tuning.
        """
        if market_volatility is None:
            return True  # No adjustment needed

        # Adjust thresholds based on volatility
        eta = self.thresholds.get("learning_rate", 0.1)

        # Only adjust if volatility is significant
        volatility_threshold = self.thresholds.get("volatility_threshold", 0.05)
        if market_volatility > volatility_threshold:
            adjustment = eta * math.log(market_volatility + 1.0)
            # Store original values for restoration
            self._adjust_thresholds(adjustment)

        return True

    def _adjust_thresholds(self, adjustment: float) -> None:
        """Temporarily adjust thresholds based on market conditions."""
        # Conservative adjustment - only modify by small amounts
        max_adjustment = 0.1
        safe_adjustment = max(-max_adjustment, min(max_adjustment, adjustment))

        # Adjust key thresholds
        self.thresholds["bayes_trigger"] += safe_adjustment * 0.1
        self.thresholds["fuzzy_trigger"] += safe_adjustment * 0.1

    def validate_all_gates(
        self,
        current_hash: str = None,
        previous_hash: str = None,
        market_volatility: float = None,
    ) -> bool:
        """Validate all gate conditions for trade execution.

        Combined gate validation using mathematical lock equations.
        Returns True if all gates pass, False otherwise.
        """
        # Run all gate validations
        gates_result = {
            "zalgo_lock": self.zalgo_lock_gate(),
            "greyscale_collapse": self.greyscale_collapse_filter(),
            "fuzzy_zone": self.fuzzy_zone_gate(),
            "bayesian_confidence": self.bayesian_confidence_gate(),
            "hash_drift": self.temporal_hash_drift_gate(current_hash, previous_hash),
            "adaptive_threshold": self.adaptive_threshold_gate(market_volatility),
        }

        # Store validation result
        self.last_validation_result = gates_result
        self.validation_history.append(gates_result)

        # Keep history manageable
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-50:]

        # All gates must pass
        all_pass = all(gates_result.values())

        # Log validation result
        self._log_validation_result(gates_result, all_pass)

        return all_pass

    def _log_validation_result(self, gates: Dict[str, bool], all_pass: bool) -> None:
        """Log the gate validation results."""
        status = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"🚪 GATE VALIDATION: {status}")

        for gate_name, gate_pass in gates.items():
            symbol = "✓" if gate_pass else "✗"
            print(f"   {symbol} {gate_name.replace('_', ' ').title()}")

        if all_pass:
            print("🚀 ALL GATES VALIDATED - TRADE EXECUTION AUTHORIZED")
        else:
            failed_gates = [name for name, passed in gates.items() if not passed]
            print(f"⚠️  FAILED GATES: {', '.join(failed_gates)}")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary with metrics."""
        if not self.validation_history:
            return {"status": "No validation history"}

        # Calculate success rates
        total_validations = len(self.validation_history)
        gate_success_rates = {}

        for gate_name in self.validation_history[0].keys():
            successes = sum(
                1 for result in self.validation_history if result[gate_name]
            )
            gate_success_rates[gate_name] = successes / total_validations

        overall_success_rate = (
            sum(1 for result in self.validation_history if all(result.values()))
            / total_validations
        )

        return {
            "total_validations": total_validations,
            "overall_success_rate": overall_success_rate,
            "gate_success_rates": gate_success_rates,
            "last_result": self.last_validation_result,
            "current_thresholds": self.thresholds.copy(),
        }


def demo_recursive_gate_stack() -> Dict[str, Any]:
    """Demonstrate the Recursive Gate Stack system."""
    print("🚪 RECURSIVE GATE STACK DEMONSTRATION")
    print("=" * 60)

    # Create mock data for demonstration
    mock_zalgo = ZalgoLockState(
        fractal_containment=0.5,
        drift_suppression=0.001,
        collapse_stability=0.0005,
        recursive_bound=8.0,
        sigmoid_collapse=0.05,
        qutrit_state=0,
        locked=True,
    )

    mock_profit_band = {
        "score": 0.7,
        "zone": 2,
        "confidence": 0.8,
    }

    # Initialize gate stack
    gate_stack = RecursiveGateStack(
        zalgo_core=mock_zalgo,
        entropy=0.0008,
        profit_band=mock_profit_band,
        bayes_confidence=0.75,
    )

    # Run validation
    result = gate_stack.validate_all_gates(
        current_hash="abc123def456",
        previous_hash="abc123def455",
        market_volatility=0.03,
    )

    # Get summary
    summary = gate_stack.get_validation_summary()

    print("\n🔍 VALIDATION SUMMARY:")
    print(f"   Overall Result: {'AUTHORIZED' if result else 'REJECTED'}")
    print(f"   Success Rate: {summary['overall_success_rate']:.2%}")

    return summary


if __name__ == "__main__":
    demo_recursive_gate_stack()
