"""
UFSEchoLogger
=============

Logs cluster memory and echo reinforcement for NCCO/SFSSS.
"""
import json
from datetime import datetime

class UFSEchoLogger:
    def __init__(self, log_path='ufs_echo_log.jsonl'):
        self.log_path = log_path

    def log_cluster_memory(self, cluster_id, strategy_id, entropy_signature):
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'cluster_id': cluster_id,
            'strategy_id': strategy_id,
            'entropy_signature': entropy_signature
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        print(f"[UFS Echo] Logged: {entry}") 