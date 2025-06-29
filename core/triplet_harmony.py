import collections
import hashlib
import math

import numpy as np


class TripletHarmony:
    def __init__(self, coherence_threshold=0.85):
        self.coherence_threshold = coherence_threshold

    def _normalize_vector(self, vec):
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)  # Add epsilon to prevent division by zero

    def calculate_coherence(self, triplet: collections.deque) -> float:
        """
        Calculates the coherence score for a triplet of normalized vectors.
        A higher score (closer to 1) indicates greater harmony.
        """
        if len(triplet) != 3:
            raise ValueError("Triplet must contain exactly 3 vectors.")

        v1, v2, v3 = [self._normalize_vector(v) for v in triplet]

        # Calculate the standard deviation of the triplet vectors
        std_dev_matrix = np.std(np.array([v1, v2, v3]), axis=0)
        std_dev_magnitude = np.linalg.norm(std_dev_matrix)

        # Calculate the mean of the triplet vectors
        mean_vector = np.mean(np.array([v1, v2, v3]), axis=0)
        mean_magnitude = np.linalg.norm(mean_vector)

        # Coherence(T) = 1 - ||STD([v1, v2, v3])|| / (||µ([v1, v2, v3])|| + epsilon)
        coherence = 1 - (std_dev_magnitude / (mean_magnitude + 1e-9))
        return max(0.0, min(1.0, coherence))  # Clamp score between 0 and 1

    def get_triplet_hash(self, central_vector: np.ndarray) -> str:
        """
        Generates a SHA256 hash for the central vector of a triplet.
        Used for unique identification and recall.
        """
        return hashlib.sha256(central_vector.tobytes()).hexdigest()

    def check_harmony(self, triplet_data: collections.deque) -> tuple[bool, float, str]:
        """
        Checks the harmony of a triplet and returns its status, coherence, and hash.
        """
        coherence_score = self.calculate_coherence(triplet_data)
        central_vector = triplet_data[1]  # Assuming the middle vector is the anchor
        triplet_hash = self.get_triplet_hash(central_vector)
        is_harmonic = coherence_score >= self.coherence_threshold
        return is_harmonic, coherence_score, triplet_hash


# Example Usage (for testing/demonstration, will be removed in final integration)
if __name__ == "__main__":
    harmony_checker = TripletHarmony(coherence_threshold=0.88)

    # Example vectors (e.g., normalized price, volume, hashrate features)
    v_t1 = np.array([0.1, 0.2, 0.7, 0.3])
    v_t2 = np.array([0.12, 0.21, 0.72, 0.31])
    v_t3 = np.array([0.11, 0.23, 0.70, 0.32])

    v_noisy1 = np.array([0.5, 0.1, 0.9, 0.2])
    v_noisy2 = np.array([0.1, 0.8, 0.2, 0.7])
    v_noisy3 = np.array([0.9, 0.2, 0.1, 0.5])

    triplet_harmonic = collections.deque([v_t1, v_t2, v_t3])
    triplet_noisy = collections.deque([v_noisy1, v_noisy2, v_noisy3])

    is_harmonic, score, t_hash = harmony_checker.check_harmony(triplet_harmonic)
    print(f"Harmonic Triplet - Is Harmonic: {is_harmonic}, Score: {score:.4f}, Hash: {t_hash}")

    is_harmonic_noisy, score_noisy, t_hash_noisy = harmony_checker.check_harmony(triplet_noisy)
    print(f"Noisy Triplet - Is Harmonic: {is_harmonic_noisy}, Score: {score_noisy:.4f}, Hash: {t_hash_noisy}")
