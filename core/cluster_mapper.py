"""
ClusterMapper
=============

Forms clusters, links by entropy, and manages echo family scoring for NCCO/SFSSS integration.
"""
import uuid
import numpy as np

class ClusterMapper:
    def __init__(self):
        self.clusters = {}  # cluster_id: cluster_node
        self.echo_families = {}  # echo_id: [cluster_ids]

    def form_cluster(self, drift_score, strategy_signal, timestamp):
        cluster_id = str(uuid.uuid4())[:8]
        cluster_node = {
            'id': cluster_id,
            'drift': drift_score,
            'signal': strategy_signal,
            'timestamp': timestamp,
            'entropy': abs(drift_score - strategy_signal),
            'echo_id': None
        }
        self.clusters[cluster_id] = cluster_node
        return cluster_node

    def link_clusters_by_entropy(self, cluster_node):
        # Simple: group clusters with similar entropy into an echo family
        entropy = cluster_node['entropy']
        for echo_id, members in self.echo_families.items():
            if any(abs(self.clusters[mid]['entropy'] - entropy) < 0.05 for mid in members):
                members.append(cluster_node['id'])
                cluster_node['echo_id'] = echo_id
                return echo_id
        # New echo family
        echo_id = str(uuid.uuid4())[:6]
        self.echo_families[echo_id] = [cluster_node['id']]
        cluster_node['echo_id'] = echo_id
        return echo_id

    def get_echo_score(self, cluster_node):
        # Echo score: sum drift of all family members weighted by recency
        echo_id = cluster_node['echo_id']
        if not echo_id or echo_id not in self.echo_families:
            return 0.0
        members = self.echo_families[echo_id]
        now = cluster_node['timestamp']
        score = 0.0
        for cid in members:
            c = self.clusters[cid]
            age = max(1, now - c['timestamp'])
            score += c['drift'] / age
        return score 