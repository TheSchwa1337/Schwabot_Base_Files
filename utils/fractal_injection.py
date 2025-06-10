from core.fractal_core import FractalState
from schwabot_math.klein_logic import KleinDecayField
import numpy as np
import time
from typing import List

def inject_fractal_state(data: List[float], phase: float = 0.0) -> FractalState:
    """
    Creates and injects a FractalState based on Klein Decay Field
    for a given data series.
    """
    klein = KleinDecayField()
    decay = klein.compute_decay_vector(np.asarray(data))
    if decay.size == 0 or np.all(decay == 0):
        entropy_val = 0.0
    else:
        normalized_decay = np.abs(decay) / np.sum(np.abs(decay))
        entropy_val = -np.sum(normalized_decay * np.log2(normalized_decay + 1e-10))
    return FractalState(
        vector=decay,
        phase=phase,
        entropy=entropy_val,
        timestamp=time.time()
    ) 