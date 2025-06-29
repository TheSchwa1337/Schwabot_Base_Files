# core/phase_certainty_filter.py

import logging
from collections import deque
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

class PhaseCertaintyFilter:
""""""
Filters tick rhythms and assesses the certainty of a detected phase.
""""""

    def __init__(self, history_size: int = 50, consistency_threshold: float = 0.8):
    self.phase_history = deque(maxlen = history_size)
    self.consistency_threshold = consistency_threshold
    logger.info("PhaseCertaintyFilter initialized.")

    def update_phase_history(self, new_phase_value: float):
    """"""
    Adds a new phase value to the history.
        
    Args:
        new_phase_value (float): The latest phase value detected.
    """"""
    self.phase_history.append(new_phase_value)
    logger.debug(f"New phase value added: {new_phase_value:.4f}. History size: {len(self.phase_history)}")

    def assess_phase_certainty(self) -> Dict[str, float]:
    """"""
    Assesses the certainty of the detected phase based on historical consistency and deviation.
    Mathematical Logic: Certainty = f(Consistency, Deviation)
        
    Returns:
        Dict[str, float]: A dictionary containing the certainty score and related metrics.
    """"""
        if len(self.phase_history) < 2:
        logger.warning("Not enough history to assess phase certainty. Returning 0.0.")
        return {"certainty_score": 0.0, "consistency": 0.0, "deviation": 0.0}

    current_history_array = np.array(self.phase_history)
        
    # 1. Calculate Consistency
    # One way to conceptualize consistency is how tightly grouped the phases are
    # Could use standard deviation or a measure of clustering
    deviation = np.std(current_history_array)
    mean_phase = np.mean(current_history_array)

        # A simple inverse relationship for consistency: lower deviation means higher consistency
        # We'll normalize it to a 0-1 scale. Max deviation could be amplitude * 2 for sine wave'
        # Assuming phases are normalized or bounded (e.g., -1 to 1 for a sine wave output)
    max_possible_deviation = 2.0 # Assuming phase values typically range -1 to 1 (amplitude of 1)
    consistency = max(0.0, 1.0 - (deviation / max_possible_deviation))
    consistency = min(1.0, consistency) # Clamp between 0 and 1
        
    # 2. Combine Consistency and Deviation into Certainty Score
    # This formula is conceptual. Higher consistency and lower deviation lead to higher certainty.
    certainty_score = (consistency * 0.7) + ((1 - (deviation / max_possible_deviation)) * 0.3) # Simple weighted average
    certainty_score = max(0.0, min(1.0, certainty_score)) # Clamp between 0 and 1

    is_certain_enough = certainty_score >= self.consistency_threshold

    logger.info(f"Phase Certainty Assessment: Score={certainty_score:.4f}, Consistency={consistency:.4f}, Deviation={deviation:.4f}. Certain Enough: {is_certain_enough}")

    return {)
        "certainty_score": certainty_score,
        "consistency": consistency,
        "deviation": deviation,
        "is_certain_enough": is_certain_enough
    }

    if __name__ == "__main__":
filter_obj = PhaseCertaintyFilter(history_size=20, consistency_threshold=0.75)

print("--- Simulating Phase History Updates ---")
# Simulate consistent phase data
consistent_phases = [0.1, 0.12, 0.11, 0.13, 0.1, 0.12, 0.11, 0.13, 0.1, 0.12]
        for p in consistent_phases:
    filter_obj.update_phase_history(p)
    assessment = filter_obj.assess_phase_certainty()
    print(f"Phase: {p:.4f}, Assessment: {assessment['certainty_score']:.4f}, Certain: {assessment['is_certain_enough']}")

    print("\n--- Introducing Volatility/Inconsistency ---")
    # Introduce some noisy/inconsistent phase data
    inconsistent_phases = [0.5, 0.2, 0.8, 0.1, 0.9, 0.3, 0.6, 0.0]
            for p in inconsistent_phases:
        filter_obj.update_phase_history(p)
        assessment = filter_obj.assess_phase_certainty()
        print(f"Phase: {p:.4f}, Assessment: {assessment['certainty_score']:.4f}, Certain: {assessment['is_certain_enough']}")

        print("\n--- Final Assessment ---")
        final_assessment = filter_obj.assess_phase_certainty()
        print("Final Assessment:", final_assessment)