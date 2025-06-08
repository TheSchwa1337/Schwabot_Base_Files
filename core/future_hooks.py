"""
Future Hooks (HookRegistry)
==========================

Evaluates rebind/preserve logic:
- Hook(S_t) = rebind if ℋ(S_t)>θ ∨ Δ_sym>δ ∨ GAN_anom>τ else preserve

Invariants:
- Hook monotonicity: Increasing entropy or Δ_sym cannot reduce rebind probability.

See docs/math/hooks.md for details.
"""
import logging
from typing import Any, Dict
import numpy as np

logger = logging.getLogger(__name__)

class HookRegistry:
    """
    Evaluates whether to rebind or preserve a shell state.
    Rebind is triggered if:
        - Entropy exceeds threshold
        - Symbolic delta exceeds threshold
        - GAN anomaly score exceeds threshold
    """
    def __init__(self, ledger: Any, entropy_engine: Any, gan_filter: Any,
                 entropy_threshold: float = 0.75,
                 delta_sym_threshold: float = 0.3,
                 anomaly_threshold: float = 0.85):
        self.ledger = ledger
        self.entropy_engine = entropy_engine
        self.gan_filter = gan_filter
        self.entropy_threshold = entropy_threshold
        self.delta_sym_threshold = delta_sym_threshold
        self.anomaly_threshold = anomaly_threshold

    def evaluate(self, state: Dict[str, Any]) -> str:
        """
        Evaluate hook decision for a shell state.
        Returns 'rebind' or 'preserve'.
        """
        logger.info("Evaluating hook decision.")

        shell_vector = state["vector"]         # np.ndarray or list[float]
        symbol_token = state["symbol"]         # str or int or hash
        prior_token = self.ledger.get("last_symbol")

        # --- Compute entropy ---
        entropy = self.entropy_engine.compute_entropy(np.array(shell_vector))
        logger.debug(f"Shell entropy: {entropy:.4f}")

        # --- Compute symbol delta ---
        delta_sym = self._symbolic_delta(prior_token, symbol_token)
        logger.debug(f"Symbolic delta: {delta_sym:.4f}")

        # --- Compute GAN anomaly score ---
        gan_metrics = self.gan_filter.detect(np.array(shell_vector))
        anomaly_score = gan_metrics.anomaly_score
        logger.debug(f"GAN anomaly score: {anomaly_score:.4f}")

        # --- Decision logic ---
        if (
            entropy > self.entropy_threshold or
            delta_sym > self.delta_sym_threshold or
            anomaly_score > self.anomaly_threshold
        ):
            logger.info("Hook decision: REBIND")
            return "rebind"
        else:
            logger.info("Hook decision: PRESERVE")
            return "preserve"

    def _symbolic_delta(self, prev: Any, curr: Any) -> float:
        """
        Symbolic difference heuristic between two shell state tokens.
        Can be simple hash distance or string diff ratio.
        """
        if prev is None:
            return 1.0  # No prior symbol → full delta

        if isinstance(prev, str) and isinstance(curr, str):
            return float(sum(a != b for a, b in zip(prev, curr))) / max(len(prev), 1)
        if isinstance(prev, int) and isinstance(curr, int):
            return abs(prev - curr) / max(abs(prev), 1)
        return 0.5  # Unknown types → conservative delta 