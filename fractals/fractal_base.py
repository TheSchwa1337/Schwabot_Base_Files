import numpy as np


class FractalBase:
    def __init__(self):
        self.memory_shell = 1.0
        self.entropy_anchor = 0.0
        self.coherence = 1.0
        self.status = "active"  # can be: "active", "collapsed", "mirrored"

    def compute_entropy(self, data):
        if len(data) < 2:
            return 0.0
        hist, _ = np.histogram(data, bins=min(len(data), 50), density=True)
        hist = hist[hist > 0] / hist[hist > 0].sum()
        if len(hist) == 0:
            return 0.0
        return -np.sum(hist * np.log2(hist))

    def compute_coherence(self, data):
        # A simple placeholder for coherence. This would be expanded later.
        # For now, it could be inverse of variance or a measure of stability.
        if len(data) < 2:
            return 1.0
        return 1.0 - np.std(data) / (np.mean(data) + 1e-9) if np.mean(data) > 0 else 0.0
