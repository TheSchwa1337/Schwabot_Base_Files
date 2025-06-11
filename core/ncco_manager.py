"""
NCCO Manager
============

Manages cluster recognition, entropy validation, memory formation, and signal generation for the Schwabot system.
Integrates with DriftShellEngine for drift shell variance calculation.
"""

from .drift_shell_engine import DriftShellEngine
import numpy as np

class NCCOManager:
    def __init__(self, baseline_entropy=0.0):
        self.drift_engine = DriftShellEngine(baseline_entropy=baseline_entropy)
        self.cluster_memory = {}  # {cluster_id: {hashes, features, meta}}
        self.drift_history = []

    def recognize_cluster(self, hash_stream):
        """
        Recognize clusters from a stream of hashes. Returns cluster_id.
        """
        # Simple: cluster by first 4 chars of hash
        cluster_id = hash_stream[0][:4] if hash_stream else None
        if cluster_id and cluster_id not in self.cluster_memory:
            self.cluster_memory[cluster_id] = {'hashes': [], 'features': {}, 'meta': {}}
        if cluster_id:
            self.cluster_memory[cluster_id]['hashes'].extend(hash_stream)
        return cluster_id

    def assess_cluster(self, cluster_id, features, tick_times, meta):
        """
        Assess a cluster using drift shell variance and return drift value.
        """
        if cluster_id not in self.cluster_memory:
            return None
        hashes = self.cluster_memory[cluster_id]['hashes']
        drift = self.drift_engine.drift_variance(hashes, features, tick_times, meta)
        self.drift_history.append(drift)
        return drift

    def get_signal(self, cluster_id, features, tick_times, meta, drift_threshold=0.5):
        """
        Returns True if drift variance exceeds threshold, else False.
        """
        drift = self.assess_cluster(cluster_id, features, tick_times, meta)
        return drift is not None and drift > drift_threshold 