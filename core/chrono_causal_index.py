# -*- coding: utf-8 -*-

""""
Chrono Causal Index (CCI)
=========================

This module implements the Chrono Causal Index (CCI), the core of Schwabot's'
**Unified Chrono-Causal Layer**. The CCI is responsible for maintaining and
querying the **Dual-Index Recall Matrix**, enabling seamless cross-indexing
between macro-level market "weather" (from CRWM) and micro-level causal
event pathways (from CRTPM).

Key functionalities include:
- Storing and retrieving CRWM weather signatures by hash.
- Storing and retrieving CRTPM causal pathways by hash.
- Cross-indexing pathways with relevant CRWM weather hashes (start and end).
- Facilitating multi-timescale retrieval for trade decisions and patch proposals.
- Implementing decay and pruning logic for memory optimization.

CCI is central to Schwabot's ability to recall historical market contexts'
and their causal implications, ensuring that every decision is informed by
a comprehensive understanding of both macro temporal resonance and micro
cause-effect sequences.
""""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import hashlib
import pandas as pd # Required for pd.Timestamp in pruning logic

# Assuming these are available from other core modules
from .chrono_resonance_mapper import ChronoResonanceMapper
from .causal_path_tracker import CausalPathway, TickEvent # Import necessary for type hints/mocking

logger = logging.getLogger(__name__)

class ChronoCausalIndex:
    """Manages the Dual-Index Recall Matrix for CRWM and CRTPM data."""

    def __init__(self):
    logger.info("CCI: Initializing Chrono Causal Index...")
    self.crwm_index: Dict[str, Dict[str, Any]] = {} # {weather_hash: weather_signature}
    self.crtpm_index: Dict[str, CausalPathway] = {} # {path_hash: CausalPathway}
    # Cross-index: {weather_hash: {path_id: score_or_relevance}}
    self.weather_to_path_map: Dict[str, Dict[str, float]] = {}
    # Reverse cross-index: {path_id: {weather_hash: score_or_relevance}}
    self.path_to_weather_map: Dict[str, Dict[str, float]] = {}
    logger.info("CCI: Chrono Causal Index initialized.")

    def add_crwm_signature(self, weather_signature: Dict[str, Any]) -> str:
    """Adds a CRWM weather signature to the index."
        # Generate a stable hash for the weather signature
        weather_str = str(sorted(weather_signature.items())) # Ensure consistent string for hashing
    weather_hash = hashlib.sha256(weather_str.encode()).hexdigest()
    self.crwm_index[weather_hash] = weather_signature
        logger.debug(f"CCI: Added CRWM signature with hash: {weather_hash[:8]}...")
    return weather_hash

    def add_crtpm_pathway(self, pathway: CausalPathway) -> str:
    """Adds a CRTPM causal pathway to the index."
        # Generate a stable hash for the pathway content (e.g., its ID or a more comprehensive hash)
    path_hash = pathway.path_id # Assuming pathway object has a unique path_id
    self.crtpm_index[path_hash] = pathway
        logger.debug(f"CCI: Added CRTPM pathway with hash: {path_hash[:8]}...")
    return path_hash

    def cross_index_pathway_with_weather(self, path_id: str, crwm_start_hash: str, crwm_end_hash: str, relevance_score: float = 1.0):
        """Links a CRTPM pathway with its associated CRWM weather hashes."
        if path_id not in self.crtpm_index:
        logger.warning(f"CCI: Pathway {path_id} not found in CRTPM index. Cannot cross-index.")
        return

    # Link start weather hash to pathway
        if crwm_start_hash:
            if crwm_start_hash not in self.weather_to_path_map:
            self.weather_to_path_map[crwm_start_hash] = {}
        self.weather_to_path_map[crwm_start_hash][path_id] = relevance_score
            
            if path_id not in self.path_to_weather_map:
            self.path_to_weather_map[path_id] = {}
        self.path_to_weather_map[path_id][crwm_start_hash] = relevance_score

    # Link end weather hash to pathway
        if crwm_end_hash and crwm_start_hash != crwm_end_hash: # Avoid duplicating if start and end are same:
            if crwm_end_hash not in self.weather_to_path_map:
            self.weather_to_path_map[crwm_end_hash] = {}
        self.weather_to_path_map[crwm_end_hash][path_id] = relevance_score

            if path_id not in self.path_to_weather_map:
            self.path_to_weather_map[path_id] = {}
        self.path_to_weather_map[path_id][crwm_end_hash] = relevance_score

        logger.debug(f"CCI: Cross-indexed pathway {path_id[:8]}... with CRWM hashes {crwm_start_hash[:8]}... and {crwm_end_hash[:8]}...")

    def retrieve_paths_by_weather_hash(self, weather_hash: str, top_n: Optional[int] = None) -> List[CausalPathway]:
        """Retrieves relevant CRTPM pathways for a given CRWM weather hash."
    linked_paths = self.weather_to_path_map.get(weather_hash, {})
        
    # Sort by relevance_score and retrieve actual pathway objects
    sorted_paths_ids = sorted(linked_paths.items(), key = lambda item: item[1], reverse = True)
        
    results = []
        for path_id, _ in sorted_paths_ids:
            if path_id in self.crtpm_index:
            results.append(self.crtpm_index[path_id])
            if top_n and len(results) >= top_n:
            break
        
        logger.debug(f"CCI: Retrieved {len(results)} pathways for weather hash {weather_hash[:8]}...")
    return results

    def retrieve_weather_by_path_id(self, path_id: str) -> List[Dict[str, Any]]:
        """Retrieves associated CRWM weather signatures for a given CRTPM pathway ID."
    linked_weather_hashes = self.path_to_weather_map.get(path_id, {})
        
    results = []
        for weather_hash, _ in linked_weather_hashes.items():
            if weather_hash in self.crwm_index:
            results.append(self.crwm_index[weather_hash])
        
        logger.debug(f"CCI: Retrieved {len(results)} weather signatures for pathway {path_id[:8]}...")
    return results

    def prune_old_entries(self, current_timestamp: float, retention_period_days: float = 30):
    """Prunes old CRWM and CRTPM entries based on a retention period."
    # This is a simplified pruning logic. In a real system, you'd consider'
    # usage frequency, impact, and other factors as discussed.
        
    # Prune CRWM signatures
    crwm_hashes_to_prune = []
        for w_hash, signature in list(self.crwm_index.items()):
        # Assuming 'timestamp' is present in CRWM signature and is sortable
        sig_timestamp_str = signature.get("timestamp")
            if sig_timestamp_str:
            sig_timestamp = pd.Timestamp(sig_timestamp_str).timestamp()
                if (current_timestamp - sig_timestamp) > (retention_period_days * 24 * 3600):
                crwm_hashes_to_prune.append(w_hash)
        
        for w_hash in crwm_hashes_to_prune:
        self.crwm_index.pop(w_hash, None)
        self.weather_to_path_map.pop(w_hash, None) # Remove related cross-index entries
        logger.debug(f"CCI: Pruned old CRWM signature {w_hash[:8]}...")

    # Prune CRTPM pathways (more complex, requires pathway end_timestamp)
    # For this example, we'll assume pathways have an `end_timestamp` attribute'
    crtpm_paths_to_prune = []
        for p_hash, pathway in list(self.crtpm_index.items()):
            if hasattr(pathway, 'end_timestamp') and (current_timestamp - pathway.end_timestamp) > (retention_period_days * 24 * 3600):
            crtpm_paths_to_prune.append(p_hash)
        
        for p_hash in crtpm_paths_to_prune:
        self.crtpm_index.pop(p_hash, None)
        self.path_to_weather_map.pop(p_hash, None) # Remove related cross-index entries
            # Also need to remove path_id from weather_to_path_map for each linked weather_hash
            for w_hash in list(self.weather_to_path_map.keys()):
                if p_hash in self.weather_to_path_map[w_hash]:
                self.weather_to_path_map[w_hash].pop(p_hash)
                    if not self.weather_to_path_map[w_hash]: # Remove weather_hash if no paths left:
                    self.weather_to_path_map.pop(w_hash)
        logger.debug(f"CCI: Pruned old CRTPM pathway {p_hash[:8]}...")

    logger.info(f"CCI: Pruning complete. CRWM entries: {len(self.crwm_index)}, CRTPM entries: {len(self.crtpm_index)}")

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
cci = ChronoCausalIndex()

# Simulate CRWM signatures
dummy_weather1 = {"timeframe": "1h", "price_gradient": 0.1, "fourier_amplitude": 5.0, "timestamp": "2024-01-01T10:00:00"}
dummy_weather2 = {"timeframe": "4h", "price_gradient": -0.05, "fourier_amplitude": 12.0, "timestamp": "2024-01-01T11:00:00"}
dummy_weather3_old = {"timeframe": "1d", "price_gradient": 0.2, "fourier_amplitude": 8.0, "timestamp": "2023-01-01T12:00:00"}

hash_w1 = cci.add_crwm_signature(dummy_weather1)
hash_w2 = cci.add_crwm_signature(dummy_weather2)
hash_w3_old = cci.add_crwm_signature(dummy_weather3_old)

# Simulate CRTPM pathways (using simplified objects matching expected attributes)
    class MockCausalPathway:
        def __init__(self, path_id: str, psi_score: float, end_timestamp: float):
        self.path_id = path_id
        self.psi_score = psi_score
        self.end_timestamp = end_timestamp

dummy_path1 = MockCausalPathway("path_abc_123", 0.15, pd.Timestamp("2024-01-01T10:05:00").timestamp())
dummy_path2 = MockCausalPathway("path_xyz_456", -0.02, pd.Timestamp("2024-01-01T11:10:00").timestamp())
dummy_path3_old = MockCausalPathway("path_old_789", 0.08, pd.Timestamp("2023-01-01T12:15:00").timestamp())

hash_p1 = cci.add_crtpm_pathway(dummy_path1)
hash_p2 = cci.add_crtpm_pathway(dummy_path2)
hash_p3_old = cci.add_crtpm_pathway(dummy_path3_old)

# Cross-index them
cci.cross_index_pathway_with_weather(hash_p1, hash_w1, hash_w1, relevance_score=0.9)
cci.cross_index_pathway_with_weather(hash_p2, hash_w2, hash_w2, relevance_score=0.7)
cci.cross_index_pathway_with_weather(hash_p3_old, hash_w3_old, hash_w3_old, relevance_score=0.5)
cci.cross_index_pathway_with_weather(hash_p1, hash_w2, "", relevance_score=0.3) # Path1 also linked to weather2

logger.info(f"Current CRWM index size: {len(cci.crwm_index)}")
logger.info(f"Current CRTPM index size: {len(cci.crtpm_index)}")

# Retrieve paths by weather hash
retrieved_paths_w1 = cci.retrieve_paths_by_weather_hash(hash_w1)
    logger.info(f"Paths linked to weather {hash_w1[:8]}...: {[p.path_id for p in retrieved_paths_w1]}")

# Retrieve weather by path ID
retrieved_weather_p1 = cci.retrieve_weather_by_path_id(hash_p1)
    logger.info(f"Weather linked to path {hash_p1[:8]}...: {[w.get('timeframe') for w in retrieved_weather_p1]}")

# Prune old entries
current_time = pd.Timestamp.now().timestamp()
logger.info(f"Pruning old entries (current timestamp: {current_time})")
    cci.prune_old_entries(current_time, retention_period_days = pd.Timedelta(days=1).total_seconds() / (24 * 3600 / 2)) # Shorter period for testing

logger.info(f"After pruning, CRWM index size: {len(cci.crwm_index)}")
logger.info(f"After pruning, CRTPM index size: {len(cci.crtpm_index)}")