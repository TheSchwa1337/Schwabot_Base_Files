# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from __future__ import annotations

# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from dataclasses import dataclass
from dual_unicore_handler import DualUnicoreHandler
from pathlib import Path
from typing import Dict, Sequence
import json

import numpy as np

from utils.math_utils import cosine_similarity


# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf - 8 -*-
"""core.overlay.aleph_overlay_mapper"""
""""""
""""""
Aleph Overlay Mapper
== == == == == == == == == ==

Matches live price signature against a stored * Aleph * memory bank and returns
the best overlay together with a confidence score derived from cosine
similarity.
""""""
""""""
""""""


__all__ = []
    "OverlayMatch",
    "AlephOverlayMapper",
    "map_aleph_overlay",


@dataclass(slots=True)
class Placeholder:

    """[BRAIN] Placeholder class for recursive profit mapping"""


""""""
""""""
    pass
    overlay_id: str
    similarity: float  # in [-1, 1]
    overlay_vector: np.ndarray

    def as_dict(self) -> Dict[str, str | float]:

        return {}
            "overlay_id": self.overlay_id,
            "similarity": self.similarity,


class Placeholder:

    """[BRAIN] Placeholder class for recursive profit mapping"""


""""""
""""""
    pass
    """Load overlay memory and perform similarity search."""
""""""
""""""

    def __init__(self, memory_json: str | Path) -> None:

    self.memory_path = Path(memory_json)
        if not self.memory_path.exists():
            raise FileNotFoundError(self.memory_path)
        self._load_memory()

# ------------------------------------------------------------------
    def _load_memory(self) -> None:

        data = json.loads(self.memory_path.read_text())
# expect {"overlay_id": [float, float, ...], ...}
    self._memory: Dict[str, np.ndarray] = {}
        key: np.asarray(vec, dtype=float) for key, vec in data.items()

        if not self._memory:
            raise ValueError("Aleph memory is empty")

# ------------------------------------------------------------------
    def map_overlay(self, live_vector: Sequence[float]) -> OverlayMatch:
        """Return best matching overlay for *live_vector*."""
""""""
""""""
        live = np.asarray(live_vector, dtype=float)
        best_id = None
        best_sim = -2.0  # less than minimum possible
        best_vec: np.ndarray | None = None
        for overlay_id, vec in self._memory.items():
# pad / truncate to match length
            if vec.size != live.size:
                min_len = min(vec.size, live.size)
                sim = cosine_similarity(vec[:min_len], live[:min_len])
            else:
                sim = cosine_similarity(vec, live)
            if sim > best_sim:
                best_sim = sim
                best_id = overlay_id
                best_vec = vec
        if best_id is None or best_vec is None:
            raise RuntimeError("no overlays found, memory may be empty")
#         return OverlayMatch(best_id, best_sim, best_vec)

# ------------------------------------------------------------------
    def overlay_confidence(self, sim: float) -> float:

        """Convert similarity to 0 - 1 confidence."""
""""""
""""""
#         return (sim + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Stand - alone functional API requested in integration docs
# ---------------------------------------------------------------------------

def map_aleph_overlay():

    live_price: float,
    memory_prices: Sequence[float],
    omega: Sequence[float]
    -> float:
    """Return weighted sum of *omega* for memory prices close to *live_price*."""
""""""
""""""

    The helper mirrors the mathematical definition::

        Psi_ALEPH(t) = Sum [ A(t) * delta(P(t) - P_mem) * Omega(t) ]

    with a practical interpretation - if the price difference is below a very
    small epsilon (1e-4) we treat it as a *match* and accumulate the
    associated *omega* weight. This is a lightweight convenience wrapper useful
    when the full :class:`AlephOverlayMapper` class overhead is not required.
    """"""
""""""
""""""
    if len(memory_prices) != len(omega):
        raise ValueError("memory_prices and omega must have equal length")
    import numpy as np

    mem_arr = np.asarray(memory_prices, dtype = float)
    omega_arr = np.asarray(omega, dtype = float)

    diff = np.abs(mem_arr - live_price) < 1e-4
#     return float(np.sum(omega_arr[diff]))



""""""
""""""
""""""
""""""
