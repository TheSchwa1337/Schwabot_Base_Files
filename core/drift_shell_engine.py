"""
drift_shell_engine.py
---------------------
Core engine for calculating Drift Shell Cluster Variance (ΔΨᵢ) in the NCCO-SFSSS pipeline.

ΔΨᵢ = ∇ᵗ[Hₙ ⊕ S(τᵢ)] · Λᵢ(t) → Π(χₙ)
"""

from scipy.stats import entropy
import numpy as np
from typing import List, Dict

class DriftShellEngine:
    def __init__(self, baseline_entropy=0.0):
        """
        Initialize the drift shell engine.
        :param baseline_entropy: Reference entropy for baseline deviation.
        """
        self.baseline_entropy = baseline_entropy

    def hash_entropy(self, hash_stream: List[str]) -> float:
        """
        Calculate entropy Hₙ from a stream of hashes.
        :param hash_stream: List of hash strings (e.g., SHA-256 hex).
        :return: Entropy value (float).
        """
        if not hash_stream:
            return 0.0
        unique, counts = np.unique(hash_stream, return_counts=True)
        probs = counts / counts.sum()
        return float(entropy(probs, base=2))

    def strategy_signal(self, cluster_features: Dict) -> float:
        """
        Compute S(τᵢ): Strategy signal for a cluster.
        :param cluster_features: Dict of cluster properties (e.g., tier, momentum).
        :return: Signal value (float).
        """
        tier = cluster_features.get("tier", 0)
        momentum = cluster_features.get("momentum", 1.0)
        volatility = cluster_features.get("volatility", 1.0)
        return (tier + 1) * momentum * np.log1p(volatility)

    def time_gradient(self, tick_times: List[float]) -> float:
        """
        Compute ∇ᵗ: Time-weighted gradient.
        :param tick_times: List of timestamps (float or int).
        :return: Gradient value (float).
        """
        if len(tick_times) < 2:
            return 1.0
        diffs = np.diff(sorted(tick_times))
        return np.mean(diffs) if np.mean(diffs) > 0 else 1.0

    def logic_selector(self, cluster_meta: Dict) -> float:
        """
        Λᵢ(t): Time-encoded logic selector (e.g., activation weight).
        :param cluster_meta: Dict of meta properties (e.g., known family, recency).
        :return: Selector value (float).
        """
        return 1.0 if cluster_meta.get("is_known_family", False) else 0.5

    def project_outcome(self, drift_shell: float) -> float:
        return 1 / (1 + np.exp(-drift_shell))

    def drift_variance(self, hash_stream, cluster_features, tick_times, cluster_meta):
        """
        Calculate ΔΨᵢ for a cluster.
        :return: Drift shell variance (float).
        """
        Hn = self.hash_entropy(hash_stream)
        S_tau = self.strategy_signal(cluster_features)
        grad_t = self.time_gradient(tick_times)
        Lambda = self.logic_selector(cluster_meta)

        entropy_signal = Hn + S_tau
        drift_shell = grad_t * entropy_signal * Lambda
        projected = self.project_outcome(drift_shell)

        return projected

# --- Example Usage / Test ---
if __name__ == "__main__":
    engine = DriftShellEngine(baseline_entropy=0.2)
    hashes = ["a1b2", "a1b2", "c3d4", "e5f6", "f7g8"]
    features = {"tier": 2, "momentum": 1.3, "volatility": 0.8}
    times = [100, 102, 105, 110, 115]
    meta = {"is_known_family": True}
    result = engine.drift_variance(hashes, features, times, meta)
    print(f"ΔΨᵢ (Drift Shell Variance): {result:.6f}") 