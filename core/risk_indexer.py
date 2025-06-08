from collections import defaultdict
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class RiskIndexer:
    """
    Keeps a base-snapshot of every metric and returns a rolling index
    normalised so that base == 1.0.
    """
    def __init__(self):
        self.memory = defaultdict(lambda: defaultdict(float))

    def update(self, basket_id: str, metrics: Dict[str, float]) -> Dict[str, float]:
        indexed = {}
        baseline = self.memory[basket_id]

        for key, value in metrics.items():
            base = baseline.get(key, 1e-6)
            indexed[key] = value / base if base else 1.0
            # Adaptive baseline update with decay
            self.memory[basket_id][key] = 0.9 * base + 0.1 * value

        return indexed

    def get_last(self, basket: str) -> Dict[str, float]:
        """Return the most recent indexed snapshot for a basket."""
        return self.memory.get(basket, {})

    def maybe_reset_baseline(self, basket: str, step: int, flip: int = 10):
        """Reset baseline every <flip> updates to keep index responsive, using the latest raw values."""
        if step % flip == 0 and basket in self.memory:
            # Reset the base for this basket using the current raw values
            new_base = {}
            for m, v_raw in self.memory[basket].items():
                if not isinstance(v_raw, (int, float)):
                    continue
                new_base[m] = abs(v_raw) if abs(v_raw) > 1e-12 else 1.0
            self.memory[basket] = new_base
            logger.info(f"Baseline reset for basket '{basket}' at step {step}. New base: {self.memory[basket]}") 