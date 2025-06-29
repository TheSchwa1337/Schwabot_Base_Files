# -*- coding: utf-8 -*-

""""
Causal Retentive Tick Pathway Memory (CRTPM)
============================================

This module implements the Causal Retentive Tick Pathway Memory (CRTPM) system,
responsible for tracking and retaining **event-wise, hash-to-pathway microstructure**
of market data. CRTPM focuses on the actual tick-by-tick memory, actions, and
cause-effect chains to build a causally-aware historical memory for Schwabot.

Key functionalities include:
- Tracking individual tick-sequences and their associated events.
- Assigning weights to events based on price impact, entropy, and anomaly.
- Pruning less impactful pathways to manage data bloat effectively.
- Computation of Psi-scores (\\(\\psi_k\\)) for each pathway.
- Integration with the Unified Chrono-Causal Layer for cross-indexing.

Mathematical Foundation:
- Pathway Tagging: \\(T_i = \\{event_j, w_j, p_j\\}_{j=0}^i\\)
- Retention Function: \\(\text{Retain}(T_i) = \begin{cases} 1 & \\sum_j w_j > \\kappa \\land \text{causal chain triggers profit anomaly} \\ 0 & \text{otherwise} \\end{cases}\\)
- Causal Impact-Weighted Memory: \\(\text{Retention}(t_{0:n}) = \\sum_{k \\in \text{Seq}(t_{0:n})} I[\\Delta\\psi_k > \\lambda] \\cdot \\log \frac{P(\text{profit} \\mid k)}{P(\text{profit} \\mid \neg k)}\\)

CRTPM ensures that Schwabot "remembers" rare but critical market events
and their causal implications, enabling deeper learning and adaptation.
""""

import logging
import hashlib
from typing import Any, Dict, List, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TickEvent:
timestamp: float
price: float
event_type: str  # e.g., "trade_fill", "order_book_update", "regime_switch"
impact_weight: float = 1.0 # Weight based on price change, entropy, anomaly
data: Dict[str, Any] = field(default_factory = dict)

@dataclass
class CausalPathway:
path_id: str
events: Deque[TickEvent]
start_timestamp: float
end_timestamp: float
psi_score: float = 0.0
    # Fields for cross-indexing with CRWM
crwm_start_hash: Optional[str] = None
crwm_end_hash: Optional[str] = None

class CausalPathTracker:
"""Manages and weights causal event pathways (CRTPM)."""

    def __init__(self, max_path_length: int = 1000, retention_threshold: float = 0.01):
    logger.info("CRTPM: Initializing Causal Path Tracker...")
    self.active_pathways: Dict[str, CausalPathway] = {}
    self.retained_pathways: Dict[str, CausalPathway] = {}
    self.max_path_length = max_path_length
    self.retention_threshold = retention_threshold
    logger.info("CRTPM: Causal Path Tracker initialized.")

    def start_new_pathway(self, initial_event: TickEvent) -> str:
        """Starts a new causal pathway with an initial event."
    path_id = hashlib.sha256(str(initial_event.timestamp).encode()).hexdigest()
    new_path = CausalPathway()
        path_id = path_id,
        events = deque([initial_event]),
        start_timestamp = initial_event.timestamp,
        end_timestamp = initial_event.timestamp
    )
    self.active_pathways[path_id] = new_path
        logger.debug(f"CRTPM: Started new pathway {path_id} with event {initial_event.event_type}.")
    return path_id

    def add_event_to_pathway(self, path_id: str, event: TickEvent):
    """Adds an event to an existing active pathway."
    path = self.active_pathways.get(path_id)
        if path:
        path.events.append(event)
        path.end_timestamp = event.timestamp
            # Trim path if it exceeds max_path_length
            while len(path.events) > self.max_path_length:
            path.events.popleft()
        logger.debug(f"CRTPM: Added event {event.event_type} to pathway {path_id}. Path length: {len(path.events)}.")
        else:
        logger.warning(f"CRTPM: Pathway {path_id} not found. Event {event.event_type} not added.")

    def calculate_psi_score(self, pathway: CausalPathway) -> float:
        """Calculates the Psi-score for a given pathway."
        # Simplified Psi-score calculation for demonstration
    # In a real scenario, this would involve more complex logic based on
    # profit/loss, event impact, and possibly a learned model.
    psi_score = 0.0
        if not pathway.events:
        return psi_score

        # Sum of (impact_weight * price_change) for each event
    events_list = list(pathway.events)
        for i in range(1, len(events_list)):
        event = events_list[i]
        prev_event = events_list[i-1]
        price_change = event.price - prev_event.price
        psi_score += event.impact_weight * price_change

    return psi_score

    def evaluate_and_retain_pathway(self, path_id: str, causal_impact_trigger: bool = False):
        """Evaluates a pathway's Psi-score and retains it if above threshold."'
    path = self.active_pathways.pop(path_id, None)
        if path:
        path.psi_score = self.calculate_psi_score(path)
        # Retention logic based on Psi-score and causal impact trigger
            if path.psi_score > self.retention_threshold or causal_impact_trigger:
            self.retained_pathways[path.path_id] = path
                logger.info(f"CRTPM: Retained pathway {path.path_id} with Psi-score: {path.psi_score:.4f}.")
            else:
            logger.debug(f"CRTPM: Discarded pathway {path.path_id} (Psi-score: {path.psi_score:.4f}).")
        else:
        logger.warning(f"CRTPM: Attempted to evaluate non-existent or already moved pathway {path_id}.")

    def get_retained_pathway(self, path_id: str) -> Optional[CausalPathway]:
    """Retrieves a retained pathway by its ID."
    return self.retained_pathways.get(path_id)

    def get_all_retained_pathways(self) -> Dict[str, CausalPathway]:
    """Returns all currently retained pathways."""
    return self.retained_pathways

    def link_with_crwm(self, path_id: str, crwm_start_hash: str, crwm_end_hash: str):
        """Links a pathway with CRWM weather hashes for cross-indexing."
    path = self.retained_pathways.get(path_id)
        if path:
        path.crwm_start_hash = crwm_start_hash
        path.crwm_end_hash = crwm_end_hash
            logger.debug(f"CRTPM: Linked pathway {path_id} with CRWM hashes.")
        else:
            logger.warning(f"CRTPM: Pathway {path_id} not found for CRWM linking.")

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
path_tracker = CausalPathTracker(max_path_length=10, retention_threshold=0.05)

# Simulate events
event1 = TickEvent(timestamp=1678886400.0, price=100.0, event_type="trade_entry", impact_weight=1.0)
event2 = TickEvent(timestamp=1678886405.0, price=100.5, event_type="ob_update", impact_weight=0.2)
event3 = TickEvent(timestamp=1678886410.0, price=101.2, event_type="regime_shift", impact_weight=1.5)
event4 = TickEvent(timestamp=1678886415.0, price=100.9, event_type="ob_update", impact_weight=0.1)
event5 = TickEvent(timestamp=1678886420.0, price=102.5, event_type="trade_exit", impact_weight=2.0)

# Start a pathway
path_id1 = path_tracker.start_new_pathway(event1)
path_tracker.add_event_to_pathway(path_id1, event2)
path_tracker.add_event_to_pathway(path_id1, event3)
path_tracker.add_event_to_pathway(path_id1, event4)
path_tracker.add_event_to_pathway(path_id1, event5)

# Evaluate and retain
path_tracker.evaluate_and_retain_pathway(path_id1, causal_impact_trigger = True)

# Get retained pathways
retained = path_tracker.get_retained_pathway(path_id1)
    if retained:
        logger.info(f"Main: Successfully retained pathway {retained.path_id} with Psi-score: {retained.psi_score:.4f}")
    logger.info(f"Main: Pathway events count: {len(retained.events)}")

# Simulate another pathway that might be discarded
event6 = TickEvent(timestamp=1678887000.0, price=200.0, event_type="trade_entry", impact_weight=1.0)
event7 = TickEvent(timestamp=1678887005.0, price=199.5, event_type="ob_update", impact_weight=0.1)
path_id2 = path_tracker.start_new_pathway(event6)
path_tracker.add_event_to_pathway(path_id2, event7)
path_tracker.evaluate_and_retain_pathway(path_id2, causal_impact_trigger = False)

    # Check if the second pathway was retained
discarded = path_tracker.get_retained_pathway(path_id2)
    if not discarded:
    logger.info(f"Main: Pathway {path_id2} was correctly discarded due to low Psi-score.")

    # Link a pathway with dummy CRWM hashes
path_tracker.link_with_crwm(path_id1, "crwm_hash_abc", "crwm_hash_xyz")
linked_path = path_tracker.get_retained_pathway(path_id1)
    if linked_path:
    logger.info(f"Main: Linked pathway CRWM hashes: Start={linked_path.crwm_start_hash}, End={linked_path.crwm_end_hash}")