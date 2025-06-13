"""
ClusterMapper
=============

Forms clusters, links by entropy, and manages echo family scoring for NCCO/SFSSS integration.
"""
import uuid
from typing import Dict, List, Optional, Union
import numpy as np
import json
from pathlib import Path

class ClusterMapper:
    def __init__(self):
        self.clusters: Dict[str, Dict] = {}  # cluster_id → cluster node
        self.echo_families: Dict[str, List[str]] = {}  # echo_id → list of cluster_ids

    def form_cluster(self, drift_score: float, strategy_signal: float, timestamp: float) -> Dict:
        """
        Create a cluster node with entropy derived from drift and signal.
        
        Parameters:
            drift_score (float): The drift score for the cluster.
            strategy_signal (float): The strategy signal for the cluster.
            timestamp (float): The timestamp of when the cluster was formed.
            
        Returns:
            Dict: A dictionary representing the cluster node.
        """
        if not isinstance(drift_score, (int, float)) or not isinstance(strategy_signal, (int, float)):
            raise ValueError("Drift score and strategy signal must be numbers.")
        
        cluster_id = str(uuid.uuid4())[:8]
        entropy = abs(drift_score - strategy_signal)
        node = {
            'id': cluster_id,
            'drift': drift_score,
            'signal': strategy_signal,
            'timestamp': timestamp,
            'entropy': entropy,
            'echo_id': None
        }
        self.clusters[cluster_id] = node
        return node

    def link_clusters_by_entropy(self, cluster_node: Dict, threshold: float = 0.05) -> str:
        """
        Assign the cluster to an echo family with similar entropy,
        or create a new echo family if no match is found.
        
        Parameters:
            cluster_node (Dict): The cluster node to link.
            threshold (float): The similarity threshold for entropy matching.
            
        Returns:
            str: The echo_id of the assigned echo family.
        """
        if not isinstance(cluster_node, dict) or 'entropy' not in cluster_node:
            raise ValueError("Cluster node must be a dictionary with an 'entropy' key.")
        
        entropy = cluster_node['entropy']
        for echo_id, members in self.echo_families.items():
            if any(abs(self.clusters[mid]['entropy'] - entropy) < threshold for mid in members):
                members.append(cluster_node['id'])
                cluster_node['echo_id'] = echo_id
                return echo_id

        echo_id = str(uuid.uuid4())[:6]
        self.echo_families[echo_id] = [cluster_node['id']]
        cluster_node['echo_id'] = echo_id
        return echo_id

    def get_echo_score(self, cluster_node: Dict, decay: float = 1.0) -> float:
        """
        Calculate the echo score as drift sum of all echo family clusters,
        weighted by inverse age (time decay).
        
        Parameters:
            cluster_node (Dict): The cluster node to calculate the score for.
            decay (float): The time decay factor for weighting cluster scores.
            
        Returns:
            float: The calculated echo score.
        """
        if not isinstance(cluster_node, dict) or 'echo_id' not in cluster_node:
            raise ValueError("Cluster node must be a dictionary with an 'echo_id' key.")
        
        now = cluster_node['timestamp']
        members = self.echo_families.get(cluster_node['echo_id'], [])
        total_score = 0.0

        for cid in members:
            member = self.clusters[cid]
            age = max(decay, now - member['timestamp'])
            total_score += member['drift'] / age

        return total_score

    def get_family_clusters(self, echo_id: str) -> List[Dict]:
        """
        Retrieve all clusters belonging to a specific echo family.
        
        Parameters:
            echo_id (str): The echo_id of the family to retrieve clusters for.
            
        Returns:
            List[Dict]: A list of cluster nodes belonging to the specified echo family.
        """
        if not isinstance(echo_id, str):
            raise ValueError("Echo ID must be a string.")
        
        return [self.clusters[cid] for cid in self.echo_families.get(echo_id, [])]

    def recent_entropy_window(self, window: float, now: float) -> List[Dict]:
        """
        Return clusters within a recent entropy time window.
        
        Parameters:
            window (float): The time window to consider for recent clusters.
            now (float): The current timestamp.
            
        Returns:
            List[Dict]: A list of cluster nodes within the specified time window.
        """
        if not isinstance(window, (int, float)) or not isinstance(now, (int, float)):
            raise ValueError("Window and now must be numbers.")
        
        return [c for c in self.clusters.values() if (now - c['timestamp']) <= window]

    def summarize(self) -> Dict[str, int]:
        """
        Summary of current cluster state.
        
        Returns:
            Dict[str, int]: A dictionary with the total number of clusters and families.
        """
        return {
            "total_clusters": len(self.clusters),
            "total_families": len(self.echo_families),
        }

    def __len__(self) -> int:
        """
        Return the number of clusters in the mapper.
        
        Returns:
            int: The total number of clusters.
        """
        return len(self.clusters)

    def get_cluster_by_id(self, cluster_id: str) -> Optional[Dict]:
        """
        Retrieve a cluster node by its UUID.
        
        Parameters:
            cluster_id (str): The UUID of the cluster to retrieve.
            
        Returns:
            Optional[Dict]: The cluster node if found, otherwise None.
        """
        return self.clusters.get(cluster_id)

    def validate_echo_integrity(self) -> bool:
        """
        Check for consistency in echo family membership.
        
        Returns:
            bool: True if all clusters are members of their assigned echo families, False otherwise.
        """
        for echo_id, members in self.echo_families.items():
            if not all(mid in self.clusters for mid in members):
                return False
        return True

    def to_json(self, path: Path):
        """
        Serialize the cluster mapper state to a JSON file.
        
        Parameters:
            path (Path): The path to the output JSON file.
        """
        data = {'clusters': self.clusters, 'echo_families': self.echo_families}
        with path.open('w') as f:
            json.dump(data, f, indent=2)

    def from_json(self, path: Path):
        """
        Deserialize the cluster mapper state from a JSON file.
        
        Parameters:
            path (Path): The path to the input JSON file.
        """
        with path.open('r') as f:
            data = json.load(f)
        self.clusters = data.get('clusters', {})
        self.echo_families = data.get('echo_families', {})

    def __str__(self) -> str:
        """
        String representation of the ClusterMapper instance.
        
        Returns:
            str: A string summary of the cluster mapper's state.
        """
        return (
            f"ClusterMapper with {len(self.clusters)} clusters and "
            f"{len(self.echo_families)} echo families."
        )

# Example usage
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    mapper = ClusterMapper()
    try:
        cluster1 = mapper.form_cluster(0.5, 0.3, 1634829600)
        cluster2 = mapper.form_cluster(0.7, 0.4, 1634829660)
        echo_id = mapper.link_clusters_by_entropy(cluster1)
        echo_id = mapper.link_clusters_by_entropy(cluster2)

        score = mapper.get_echo_score(cluster1)
        logger.info(f"Echo score for cluster {cluster1['id']}: {score}")

        family_clusters = mapper.get_family_clusters(echo_id)
        logger.info("Clusters in echo family:")
        for c in family_clusters:
            logger.info(c)

        recent_clusters = mapper.recent_entropy_window(60, 1634829720)
        logger.info("Recent clusters within the last 60 seconds:")
        for c in recent_clusters:
            logger.info(c)

        summary = mapper.summarize()
        logger.info(f"Cluster summary: {summary}")
    except Exception as e:
        logger.error(f"An error occurred: {e}") 