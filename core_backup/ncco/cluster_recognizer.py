import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from core.entropy_engine import UnifiedEntropyEngine  # Assuming this is available
from core.ncco.cluster_memory import ClusterMemoryManager, ProfitCluster

logger = logging.getLogger(__name__)


class ClusterRecognizer:
    """"""
    Detects and manages market clusters based on SHA-256 hash patterns.
    It identifies new clusters, updates existing ones, and calculates
    hash similarity.
    """"""

    def __init__()
        self,
            cluster_memory_manager: ClusterMemoryManager,
                entropy_engine: UnifiedEntropyEngine,
                similarity_threshold: float = 0.85,
                min_hash_length_for_similarity: int = 64,
                ):
        """"""
        Initializes the ClusterRecognizer.

        Args:
            cluster_memory_manager (ClusterMemoryManager): An instance of the ClusterMemoryManager.
            entropy_engine (UnifiedEntropyEngine): An instance of the UnifiedEntropyEngine.
            similarity_threshold (float): The minimum similarity score to consider two hashes part of the same cluster.
            min_hash_length_for_similarity (int): Minimum length of hash to perform similarity calculation on.
        """"""
        self.memory_manager = cluster_memory_manager
        self.entropy_engine = entropy_engine
        self.similarity_threshold = similarity_threshold
        self.min_hash_length_for_similarity = min_hash_length_for_similarity
        logger.info("ClusterRecognizer initialized.")

    def calculate_sha_similarity(self, hash1: str, hash2: str) -> float:
        """"""
        Calculates a similarity score between two SHA-256 hashes.
        Ranges from 0.0 (completely different) to 1.0 (identical).

        Mathematical Form (Bitwise Proximity Metric): Sᵢⱼ = sum (bit(Hᵢ[k] ⊕ Hⱼ[k]) == 0) / 256
        (Conceptual: integrates information-theoretic distance for future)

        Args:
            hash1 (str): First SHA-256 hash string.
            hash2 (str): Second SHA-256 hash string.

        Returns:
            float: The similarity score.
        """"""
        if not hash1 or not hash2 or len(hash1) != len(hash2) or len(hash1) < self.min_hash_length_for_similarity:
            logger.warning("Invalid hashes for similarity calculation or hashes too short.")
            return 0.0

        # Convert hex strings to integer arrays for bitwise comparison
        # This assumes SHA256 hex strings (64 chars) converted to 256 bits
        bits1 = bin(int(hash1, 16))[2:].zfill(256)  # Ensure 256-bit representation
        bits2 = bin(int(hash2, 16))[2:].zfill(256)

        if len(bits1) != 256 or len(bits2) != 256:
            logger.error("Bit string conversion failed to yield 256 bits.")
            return 0.0

        matching_bits = sum(b1 == b2 for b1, b2 in zip(bits1, bits2))
        similarity = matching_bits / 256.0
        logger.debug(f"SHA Similarity between {hash1[:8]}... and {hash2[:8]}... : {similarity:.4f}")

        # Future: Integrate Information-Theoretic Distance and Adaptive Weighting
        # This would involve analyzing statistical properties of hash slices.
        # E.g., D_KL(P(H_i) || P(H_j)) and W_ij = exp(-alpha * D_KL) + beta * S_ij

        return float(similarity)

    def recognize_and_store_cluster()
        self,
            new_hash_block: str,
                tick: int,
                market_data_for_entropy: np.ndarray,
                strategy_trigger: Optional[str] = None,
                cluster_group: Optional[str] = None,
                ) -> ProfitCluster:
        """"""
        Recognizes a new or existing cluster from a hash block and stores/updates it.

        Args:
            new_hash_block (str): The new SHA-256 hash block.
            tick (int): The current tick number.
            market_data_for_entropy (np.ndarray): Numerical data for entropy calculation (e.g., price series).
            strategy_trigger (Optional[str]): The strategy triggered by this hash (if any).
            cluster_group (Optional[str]): Categorization of the cluster (e.g., 'bullish_trend').

        Returns:
            ProfitCluster: The recognized or updated ProfitCluster object.
        """"""
        # Calculate entropy for the new hash block context
        current_entropy = 0.0
        if market_data_for_entropy.size > 0:
            try:
                current_entropy = self.entropy_engine.compute_entropy(market_data_for_entropy, method="shannon")
            except Exception as e:
                logger.error(f"Error computing entropy for cluster: {e}")

        # Look for similar existing clusters
        matched_cluster_id: Optional[str] = None
        best_similarity = 0.0

        for existing_cluster in self.memory_manager.get_all_clusters():
            similarity = self.calculate_sha_similarity(new_hash_block, existing_cluster.hash_block)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                matched_cluster_id = existing_cluster.cluster_id

        if matched_cluster_id:
            # Update existing cluster
            cluster = self.memory_manager.get_cluster(matched_cluster_id)
            if cluster:  # MyPy check
                cluster.hash_block = new_hash_block  # Update with latest hash
                cluster.tick = tick
                cluster.entropy = current_entropy
                cluster.usage_count += 1
                cluster.last_updated = datetime.now().isoformat()
                if strategy_trigger and not cluster.strategy_trigger:
                    cluster.strategy_trigger = strategy_trigger
                if cluster_group and not cluster.cluster_group:
                    cluster.cluster_group = cluster_group
                # More sophisticated update logic for profit_tier, stability_score etc. would go here
                self.memory_manager.add_cluster(cluster)  # Add will update
                logger.info()
                    f"Updated existing cluster {cluster.cluster_id} with new hash. Similarity: {best_similarity:.4f}"
                )
                return cluster
            else:
                # Should not happen if matched_cluster_id came from get_all_clusters
                logger.error(f"Internal error: Matched cluster {matched_cluster_id} not found in memory.")

        # Create a new cluster if no match or error in retrieving
        new_cluster_id = f"CLUSTER_{hashlib.sha256(new_hash_block.encode()).hexdigest()[:10]}"
        new_cluster = ProfitCluster()
            cluster_id=new_cluster_id,
                hash_block=new_hash_block,
                    tick=tick,
                    entropy=current_entropy,
                    strategy_trigger=strategy_trigger,
                    cluster_group=cluster_group,
                    usage_count=1,  # First usage
        )
        self.memory_manager.add_cluster(new_cluster)
        logger.info(f"Created new cluster: {new_cluster_id}")
        return new_cluster


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Mock dependencies for testing
    class MockClusterMemoryManager(ClusterMemoryManager):
        def __init__(self):
            self.clusters = {}
            self.counter = 0

        def add_cluster(self, cluster: ProfitCluster) -> None:
            # Simulate persistence by not actually deleting
            self.clusters[cluster.cluster_id] = cluster
            if cluster.usage_count == 1:  # Only increment on first add
                self.counter += 1
            logger.debug(f"Mock: Added/Updated cluster: {cluster.cluster_id}")

        def get_all_clusters(self) -> List[ProfitCluster]:
            return list(self.clusters.values())

        def get_cluster(self, cluster_id: str) -> Optional[ProfitCluster]:
            return self.clusters.get(cluster_id)

    class MockUnifiedEntropyEngine:
        def compute_entropy(self, data: np.ndarray, method: str) -> float:
            # Simple mock entropy: sum of data points / 100
            return np.sum(data) / 100.0 if data.size > 0 else 0.0

    mock_memory_manager = MockClusterMemoryManager()
    mock_entropy_engine = MockUnifiedEntropyEngine()

    recognizer = ClusterRecognizer()
        cluster_memory_manager=mock_memory_manager,
            entropy_engine=mock_entropy_engine,
                similarity_threshold=0.9,  # High threshold for clear matches
    )

    print("\n--- Testing ClusterRecognizer ---")

    # Simulate some initial hashes and market data
    hash_a = hashlib.sha256(b"market_state_alpha_1").hexdigest()
    hash_b = hashlib.sha256(b"market_state_beta_2").hexdigest()
    # Create a slightly similar hash to 'hash_a'
    hash_a_similar = hash_a[:-2] + "1"  # Change last two hex digits
    hash_a_very_similar = hash_a[:-1] + "F"  # Change last hex digit

    market_data_1 = np.array([100.0, 101.5, 99.8])
    market_data_2 = np.array([50.0, 52.1, 49.5])

    # 1. Recognize a brand new cluster (hash_a)
    print("\nScenario 1: New Cluster Recognition")
    cluster_obj_1 = recognizer.recognize_and_store_cluster()
        new_hash_block=hash_a, tick=1, market_data_for_entropy=market_data_1, cluster_group="bullish_trend"
    )
    print(f"  Initial Cluster ID: {cluster_obj_1.cluster_id}, Usage: {cluster_obj_1.usage_count}")
    print(f"  Memory Manager has {mock_memory_manager.get_cluster_count()} clusters.")

    # 2. Recognize a very similar hash (should update the existing cluster)
    print("\nScenario 2: Update Existing Cluster (Very Similar Hash)")
    updated_cluster_obj = recognizer.recognize_and_store_cluster()
        new_hash_block=hash_a_very_similar,
            tick=2,
                market_data_for_entropy=market_data_1,  # Same data for simplicity
        cluster_group="bullish_trend",
            )
    print(f"  Updated Cluster ID: {updated_cluster_obj.cluster_id}, Usage: {updated_cluster_obj.usage_count}")
    print(f"  Memory Manager has {mock_memory_manager.get_cluster_count()} clusters.")  # Should still be 1
    print(f"  Similarity to original: {recognizer.calculate_sha_similarity(hash_a, hash_a_very_similar):.4f}")

    # 3. Recognize a new, distinct cluster (hash_b)
    print("\nScenario 3: New Distinct Cluster Recognition")
    cluster_obj_2 = recognizer.recognize_and_store_cluster()
        new_hash_block=hash_b, tick=3, market_data_for_entropy=market_data_2, cluster_group="bearish_reversal"
    )
    print(f"  New Distinct Cluster ID: {cluster_obj_2.cluster_id}, Usage: {cluster_obj_2.usage_count}")
    print(f"  Memory Manager has {mock_memory_manager.get_cluster_count()} clusters.")  # Should be 2

    # 4. Recognize a slightly similar hash (below threshold, so new cluster)
    print("\nScenario 4: Slightly Similar Hash (Below Threshold) -> New Cluster")
    cluster_obj_3 = recognizer.recognize_and_store_cluster()
        new_hash_block=hash_a_similar,
            tick=4,
                market_data_for_entropy=market_data_1,
                cluster_group="neutral_consolidation",
                )
    print(f"  New Cluster ID: {cluster_obj_3.cluster_id}, Usage: {cluster_obj_3.usage_count}")
    print(f"  Similarity to original: {recognizer.calculate_sha_similarity(hash_a, hash_a_similar):.4f}")
    print(f"  Memory Manager has {mock_memory_manager.get_cluster_count()} clusters.")  # Should be 3

    print("\n--- All Clusters in Memory ---")
    for cluster in mock_memory_manager.get_all_clusters():
        print()
            f"  ID: {cluster.cluster_id}, Hash: {cluster.hash_block[:8]}..., Usage: {cluster.usage_count}, Entropy: {cluster.entropy:.4f}"
        )
