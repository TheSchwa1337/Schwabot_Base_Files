# -*- coding: utf-8 -*-
""""""
State Connectivity and Algebraic Measurement
===========================================

Loads YAML connectivity/state definitions and provides functions to measure,
    render, compute, and connect all state forms (small, large, mid, compact, gated,)
tall, looped, oblong) for the mathematical relay system.

Integrates with SciPy for advanced/differential math and exposes a clean API.
""""""

import hashlib
import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
from scipy import stats

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "connectivity_hooks.yaml")


class StateConnectivity:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.state_forms = {s["name"]: s for s in self.config["state_forms"]}
        self.algebraic_connectivity = {a["name"]: a for a in self.config["algebraic_connectivity"]}
        self.handoff_definitions = {h["name"]: h for h in self.config["handoff_definitions"]}

    # --- State Form Utilities ---
    def get_state_form(self, name: str) -> Dict[str, Any]:
        return self.state_forms.get(name)

    def list_state_forms(self) -> List[str]:
        return list(self.state_forms.keys())

    # --- Algebraic Connectivity Functions ---
    def sha256_state(self, data: Union[str, bytes, List[float]]) -> str:
        if isinstance(data, list):
            data = ",".join(map(str, data))
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def mean(self, values: List[float]) -> float:
        return float(np.mean(values))

    def median(self, values: List[float]) -> float:
        return float(np.median(values))

    def deviation(self, values: List[float]) -> float:
        return float(np.std(values))

    def frequency_ratio(self, values: List[float]) -> float:
        # Ratio of max to min frequency (or 1 if not enough data)
        if len(values) < 2:
            return 1.0
        freqs = np.fft.rfftfreq(len(values))
        spectrum = np.abs(np.fft.rfft(values))
        if np.all(spectrum == 0):
            return 1.0
        max_freq = freqs[np.argmax(spectrum)]
        min_freq = freqs[np.argmin(spectrum)] if np.any(spectrum) else 1.0
        return float(max_freq / (min_freq or 1.0))

    def pool_valuation(self, values: List[float], method: str = "mean") -> float:
        # For missing/hanging states, use mean/median or fallback
        if not values:
            return 0.0
        if method == "median":
            return self.median(values)
        return self.mean(values)

    def apr(self, portfolio_history: List[float], periods_per_year: int = 365) -> float:
        # Calculate APR from portfolio value history
        if len(portfolio_history) < 2:
            return 0.0
        start = portfolio_history[0]
        end = portfolio_history[-1]
        n_periods = len(portfolio_history) - 1
        if start == 0:
            return 0.0
        rate = (end / start) ** (periods_per_year / n_periods) - 1
        return float(rate * 100)

    # --- State Measurement, Rendering, and Computation ---
    def measure_state(self, values: List[float], form: str = "midform") -> Dict[str, Any]:
        return {}
            "form": form,
                "mean": self.mean(values),
                    "median": self.median(values),
                    "deviation": self.deviation(values),
                    "sha256": self.sha256_state(values),
                    "frequency_ratio": self.frequency_ratio(values),
}
    def render_state(self, state: Dict[str, Any]) -> str:
        # Simple string rendering for now
        return f"State[{state.get('form')}] mean={state.get('mean'):.4f} median={state.get('median'):.4f} dev={state.get('deviation'):.4f} sha256={state.get('sha256')[:8]}... freq_ratio={state.get('frequency_ratio'):.4f}"

    def compute_state(self, values: List[float], form: str = "midform") -> Dict[str, Any]:
        # Alias for measure_state (could add more computation later)
        return self.measure_state(values, form=form)

    # --- State Switching, Adapting, Connecting ---
    def switch_state_form(self, values: List[float], from_form: str, to_form: str) -> Dict[str, Any]:
        # For now, just recompute with new form
        return self.measure_state(values, form=to_form)

    def adapt_state(self, values: List[float], target: float) -> List[float]:
        # Adapt values to target mean
        current_mean = self.mean(values)
        if current_mean == 0:
            return values
        factor = target / current_mean
        return [v * factor for v in values]

    def connect_state(self, state: Dict[str, Any], api_endpoint: Optional[str] = None) -> bool:
        # Placeholder for API/mathematical plumbing connection
        # In real use, would POST to API or hand off to another module
        return True


# Global instance
state_connectivity = StateConnectivity()
