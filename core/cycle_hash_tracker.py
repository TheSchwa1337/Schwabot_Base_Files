# core/cycle_hash_tracker.py

import hashlib
import json
import logging
import time
from collections import deque
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class CycleHashTracker:
    """
    Tracks and analyzes cyclical hash patterns to identify recurring sequences
    and potentially predict future hash behaviors.
    """

    def __init__(self, history_size: int = 1000, cycle_window_min: int = 5, cycle_window_max: int = 50):
        self.hash_history = deque(maxlen=history_size)
        self.cycle_window_min = cycle_window_min
        self.cycle_window_max = cycle_window_max
        self.detected_cycles: Dict[str, Any] = {}
        logger.info("CycleHashTracker initialized.")

    def update_hash_cycle(self, new_hash: str):
        """
        Adds a new hash to the historical record and attempts to detect cyclical patterns.

        Args:
            new_hash (str): The latest hash to add to the history.
        """
        self.hash_history.append(new_hash)
        self._detect_cycles()
        logger.debug(f"New hash added: {new_hash[:8]}.... Current history size: {len(self.hash_history)}")

    def _detect_cycles(self):
        """
        Internal method to detect cyclical patterns within the hash history.
        This is a simplified conceptual pattern recognition.
        Mathematical Logic: Pattern Recognition (conceptual string matching / sequence analysis)
        """
        current_history = list(self.hash_history)
        if len(current_history) < self.cycle_window_min * 2:  # Need at least two potential cycles to compare
            return

        # Iterate through possible cycle lengths
        for cycle_len in range(self.cycle_window_min, min(self.cycle_window_max, len(current_history) // 2) + 1):
            # Check if the most recent 'cycle_len' hashes match a previous sequence
            current_segment = tuple(current_history[-cycle_len:])

            # Search for this segment earlier in the history
            for i in range(len(current_history) - cycle_len - 1, cycle_len - 1, -1):
                previous_segment = tuple(current_history[i - cycle_len + 1 : i + 1])

                if current_segment == previous_segment:
                    cycle_key = f"cycle_{cycle_len}_{current_segment[0][:4]}"
                    if cycle_key not in self.detected_cycles:
                        self.detected_cycles[cycle_key] = {
                            "length": cycle_len,
                            "pattern": list(current_segment),
                            "occurrences": 1,
                            "last_detected_idx": len(current_history) - 1,
                        }
                        logger.info(f"Detected new cycle: {cycle_key} (Length: {cycle_len})")
                    else:
                        # Update occurrence count and last detected index
                        self.detected_cycles[cycle_key]["occurrences"] += 1
                        self.detected_cycles[cycle_key]["last_detected_idx"] = len(current_history) - 1
                        logger.debug(
                            f"Updated cycle {cycle_key}. Occurrences: {
                                self.detected_cycles[cycle_key]['occurrences']}"
                        )
                    break  # Found a match, move to next cycle length

    def predict_next_cycle(self, basis_hash: str = None) -> Dict[str, Any]:
        """
        Attempts to predict the next hash cycle based on detected patterns.
        Mathematical Logic: Time Series Prediction (conceptual extrapolation of detected patterns)

        Args:
            basis_hash (str, optional): A specific hash to use as a basis for prediction.
                                        If None, uses the most recently detected cycle.

        Returns:
            Dict[str, Any]: A dictionary containing prediction details or an empty dict if no prediction.
        """
        if not self.detected_cycles:
            logger.warning("No cycles detected to make a prediction.")
            return {"predicted_next_hash": None, "confidence": 0.0, "cycle_info": None}

        target_cycle = None
        if basis_hash:
            # Try to find a cycle that ends with or contains the basis_hash
            for cycle_info in self.detected_cycles.values():
                if basis_hash in cycle_info["pattern"]:
                    target_cycle = cycle_info
                    break
        else:
            # Use the most frequently occurring cycle, or the most recent one
            if self.detected_cycles:
                # Sort by occurrences and then by recency (last_detected_idx)
                sorted_cycles = sorted(
                    self.detected_cycles.values(),
                    key=lambda x: (x["occurrences"], x["last_detected_idx"]),
                    reverse=True,
                )
                target_cycle = sorted_cycles[0]

        if target_cycle:
            pattern = target_cycle["pattern"]
            length = target_cycle["length"]
            occurrences = target_cycle["occurrences"]

            if length > 0:  # Ensure non-zero length
                # Simple prediction: the next hash in the sequence is the first hash of the pattern
                predicted_next_hash = pattern[0]
                confidence = min(1.0, occurrences / 5.0)  # Conceptual confidence based on occurrences

                logger.info(
                    f"Predicted next hash: {predicted_next_hash[:8]}... with confidence: {confidence:.2f} (based on cycle length {length}, {occurrences} occurrences)"
                )
                return {
                    "predicted_next_hash": predicted_next_hash,
                    "confidence": confidence,
                    "cycle_info": target_cycle,
                }

        logger.warning("Could not make a confident prediction based on available cycles.")
        return {"predicted_next_hash": None, "confidence": 0.0, "cycle_info": None}

    def get_detected_cycles(self) -> Dict[str, Any]:
        """Returns the currently detected cyclical hash patterns."""
        return self.detected_cycles

    def get_hash_history(self) -> List[str]:
        """Returns the full hash history."""
        return list(self.hash_history)


if __name__ == "__main__":
    tracker = CycleHashTracker(history_size=100, cycle_window_min=3, cycle_window_max=10)

    # Simulate hash stream with a repeating pattern "A", "B", "C"
    hash_stream_part1 = [hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in range(10)]
    hash_stream_part2 = [hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in range(3, 6)]  # Pattern: A, B, C
    hash_stream_part3 = [hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in range(15, 20)]
    hash_stream_part4 = [hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in range(3, 6)]  # Repeat pattern

    full_hash_stream = hash_stream_part1 + hash_stream_part2 + hash_stream_part3 + hash_stream_part4

    print("\n--- Simulating Hash Stream and Detecting Cycles ---")
    for i, h in enumerate(full_hash_stream):
        tracker.update_hash_cycle(h)
        if i % 10 == 0 and i > 0:
            print(f"After {i + 1} hashes, detected cycles: {tracker.get_detected_cycles()}")

    print("\n--- Final Detected Cycles ---")
    for key, cycle_info in tracker.get_detected_cycles().items():
        print(
            f"  {key}: Length={cycle_info['length']}, Occurrences={cycle_info['occurrences']}, Pattern=[{cycle_info['pattern'][0][:8]}..., {cycle_info['pattern'][1][:8]}..., ...]"
        )

    print("\n--- Predicting Next Cycle ---")
    prediction = tracker.predict_next_cycle()
    print("Prediction based on most frequent/recent cycle:", prediction)

    # Test prediction with a specific basis hash (e.g., the last hash of the detected pattern)
    if "cycle_3_4a7b" in tracker.get_detected_cycles():
        specific_basis_hash = tracker.get_detected_cycles()["cycle_3_4a7b"]["pattern"][-1]
        prediction_specific = tracker.predict_next_cycle(basis_hash=specific_basis_hash)
        print(f"Prediction based on basis hash {specific_basis_hash[:8]}...:", prediction_specific)
    else:
        print("No 'cycle_3_4a7b' detected for specific basis hash prediction.")

    print("\n--- Full Hash History (last 10) ---")
    print(tracker.get_hash_history()[-10:])
