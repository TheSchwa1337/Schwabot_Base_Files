# core/symbolic_ledger.py

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SymbolicLedger:
    """"""
    Manages the storage, comparison, and recall of symbolic anchors for
    recursive alignment and strategy rebinding.
    Mathematical Form: L_k = { t_k, A_k, vec_k, H_k, rho_k, gamma_k }
    Where:
        t_k: timestamp
        A_k: symbolic anchor string
        vec_k: vector hash (conceptual, could be actual hash of state vector)
        H_k: entropy
        rho_k: phase angle
        gamma_k: drift resonance
    """"""

    def __init__(self, max_history_size: int = 1000, default_rebind_threshold: int = 5):
        """"""
        Initializes the SymbolicLedger.

        Args:
            max_history_size (int): Maximum number of symbolic states to keep in history.
            default_rebind_threshold (int): Default Hamming distance threshold to trigger a rebind.
        """"""
        self.ledger_history: List[Dict[str, Any]] = []
        self.max_history_size = max_history_size
        self.default_rebind_threshold = default_rebind_threshold
        self.current_symbolic_anchor: Optional[str] = None
        logger.info("SymbolicLedger initialized.")

    def add_symbolic_state()
        self, anchor: str, vector_hash: str, entropy: float, phase_angle: float, drift_resonance: float
    ):
        """"""
        Adds a new symbolic state entry to the ledger history.

        Args:
            anchor (str): The symbolic anchor string for the current shell state.
            vector_hash (str): A hash or representation of the shell state vector.
            entropy (float): The entropy of the shell state.
            phase_angle (float): The phase angle of the shell state.
            drift_resonance (float): The drift resonance of the shell state.
        """"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "anchor": anchor,
            "vector_hash": vector_hash,
            "entropy": entropy,
            "phase_angle": phase_angle,
            "drift_resonance": drift_resonance,
}
}
        self.ledger_history.append(entry)
        self.current_symbolic_anchor = anchor

        # Maintain history size
        if len(self.ledger_history) > self.max_history_size:
            self.ledger_history.pop(0)  # Remove oldest entry

        logger.debug(f"Added symbolic state: {anchor}. History size: {len(self.ledger_history)}")

    def hamming_distance(self, anchor1: str, anchor2: str) -> int:
        """"""
        Calculates the Hamming distance between two symbolic anchor strings.
        This metric is used to determine symbolic divergence.

        Args:
            anchor1 (str): First symbolic anchor string.
            anchor2 (str): Second symbolic anchor string.

        Returns:
            int: The Hamming distance. Returns -1 if strings have different lengths.
        """"""
        if len(anchor1) != len(anchor2):
            logger.warning()
                f"Cannot compute Hamming distance for strings of different lengths: {len(anchor1)} vs {len(anchor2)}"
            )
            return -1  # Or raise an error

        distance = 0
        for char1, char2 in zip(anchor1, anchor2):
            if char1 != char2:
                distance += 1
        logger.debug(f"Hamming distance between '{anchor1}' and '{anchor2}': {distance}")
        return distance

    def check_rebind_trigger(self, new_anchor: str, threshold: Optional[int] = None) -> Tuple[bool, Optional[int]]:
        """"""
        Checks if a rebind trigger condition is met based on symbolic divergence
        from the current (or last) symbolic anchor in the ledger.
        Mathematical Form: d_sym(A_i, A_j) > delta_threshold

        Args:
            new_anchor (str): The new symbolic anchor to compare against.
            threshold (Optional[int]): Custom Hamming distance threshold. If None, uses default.

        Returns:
            Tuple[bool, Optional[int]]: (True if rebind triggered, Hamming distance).
                                        Returns (False, None) if no current anchor to compare against.
        """"""
        if self.current_symbolic_anchor is None:
            logger.info("No current symbolic anchor in ledger to compare for rebind trigger.")
            return False, None

        rebind_threshold = threshold if threshold is not None else self.default_rebind_threshold

        distance = self.hamming_distance(self.current_symbolic_anchor, new_anchor)

        if distance == -1:
            logger.warning("Rebind check skipped due to anchor length mismatch.")
            return False, None

        if distance > rebind_threshold:
            logger.info(f"Rebind triggered: Symbolic divergence ({distance}) > threshold ({rebind_threshold})")
            return True, distance
        else:
            logger.debug(f"No rebind triggered: Symbolic divergence ({distance}) <= threshold ({rebind_threshold})")
            return False, distance

    def get_ledger_history(self) -> List[Dict[str, Any]]:
        """"""
        Returns the full symbolic ledger history.
        """"""
        return list(self.ledger_history)

    def get_last_anchor(self) -> Optional[str]:
        """"""
        Returns the most recently added symbolic anchor.
        """"""
        return self.current_symbolic_anchor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ledger = SymbolicLedger(max_history_size=5, default_rebind_threshold=2)

    print("\n--- Testing SymbolicLedger --- ")

    # Add initial state
    ledger.add_symbolic_state("AABBCC", "hash123", 0.5, 0.1, 0.5)
    print(f"Current anchor: {ledger.get_last_anchor()}")

    # Test no rebind
    is_rebind, dist = ledger.check_rebind_trigger("AABBCC", threshold=0)
    print(f"Rebind check for AABBCC (threshold 0): {is_rebind}, Distance: {dist}")

    is_rebind, dist = ledger.check_rebind_trigger("AABBCD", threshold=1)
    print(f"Rebind check for AABBCD (threshold 1): {is_rebind}, Distance: {dist}")

    # Test rebind triggered
    is_rebind, dist = ledger.check_rebind_trigger("AATTDD")  # Default threshold is 2
    print(f"Rebind check for AATTDD (default threshold 2): {is_rebind}, Distance: {dist}")

    # Add more states to fill history
    ledger.add_symbolic_state("AATTDD", "hash456", 0.6, 0.2, 0.1)
    ledger.add_symbolic_state("XXYYZZ", "hash789", 0.7, 0.3, 0.15)
    ledger.add_symbolic_state("123456", "hashabc", 0.8, 0.4, 0.2)
    ledger.add_symbolic_state("ABCDEF", "hashdef", 0.9, 0.5, 0.25)

    print(f"History size: {len(ledger.get_ledger_history())}")
    ledger.add_symbolic_state("GHIJKL", "hashghi", 0.95, 0.6, 0.3)  # This should push out the oldest
    print(f"History size after exceeding max: {len(ledger.get_ledger_history())}")
    print(f"Oldest entry in history: {ledger.get_ledger_history()[0]['anchor']}")

    # Test different length anchors
    is_rebind, dist = ledger.check_rebind_trigger("SHORT", threshold=1)
    print(f"Rebind check for SHORT (diff length): {is_rebind}, Distance: {dist}")
