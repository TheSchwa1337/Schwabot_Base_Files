# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from __future__ import annotations

# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from dataclasses import dataclass
from dual_unicore_handler import DualUnicoreHandler
from typing import Sequence

import numpy as np

from utils.math_utils import calculate_entropy, moving_average


# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf - 8 -*-\n"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"DriftWeightReport",
    "DriftPhaseWeighter",


@dataclass(slots = True)
class Placeholder:
    pass  # Emergency placeholder

"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Function implementation pending."""
# return {}"""
        "drift_weight": self.drift_weight,
        "lambda": self.lambda_,


class Placeholder:
    pass  # Emergency placeholder

"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
        raise ValueError("lambda_ must be in (0, 1"))
        self.lambda_ = lambda_

# ------------------------------------------------------------------
def calculate_drift_weight():
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
raise ValueError("need at least 3 price points")

delta = np.abs(np.diff(prices_arr))
# exponential smoothing: s[t] = lambda * x[t] + (1 - lambda)*s[t - 1]
        smoothed = np.empty_like(delta)
        smoothed[0] = delta[0]
        for i in range(1, delta.size):
        smoothed[i] = self.lambda_ * delta[i]
        + (1.0 - self.lambda_) * smoothed[i - 1]

# use last 5 points (or all if shorter) for current drift weight
        tail = smoothed[-5:]
        drift_weight=float(np.mean(tail))

#         return DriftWeightReport(drift_weight, smoothed, self.lambda_)

# ------------------------------------------------------------------
@ staticmethod
def gradient_entropy_score():
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
samples and the preceding `window` samples."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
raise ValueError("price history too short for entropy gradient")
        ent_now = calculate_entropy(arr[-window:])
        ent_prev = calculate_entropy(arr[-(2 * window): -window])
#         return float(ent_now - ent_prev)


"""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""