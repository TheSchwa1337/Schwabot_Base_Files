"""
registry_vote_matrix.py
----------------------
Aggregates trade proposals coming from multiple AI agents and decides whether
there is enough weighted consensus to pass the proposal on to execution.

Each agent has an associated performance score (0.0-1.0) that represents its
historical profitability.  Votes are weighted by this score so that agents with
better track records have a stronger say.
"""

from __future__ import annotations

from typing import Dict, Mapping


class VoteRegistry:
    """Stores agent performance metrics and evaluates weighted vote outcomes."""

    def __init__(self,  performance_db: Mapping[str, float] | None = None, approval_threshold: float = 0.6) -> None:
        # default every agent weight = 0.5 (neutral) if not provided
        self.performance_db: Dict[str, float] = dict(performance_db or {})
        self.approval_threshold = approval_threshold

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def update_performance(self,  agent_id: str, new_score: float) -> None:
        """Persist a new performance score for *agent_id* (clamped 0-1)."""
        self.performance_db[agent_id] = max(0.0, min(1.0, new_score))

    def evaluate(self,  votes: Mapping[str, bool]) -> bool:
        """Return *True* if the weighted approval meets *approval_threshold*.

        *votes* is a mapping of *agent_id* → *bool* (True for approve,
        False for reject).
        """
        if not votes:
            return False

        # Convert boolean vote to signed weight (+weight or -weight)
        weighted_sum = 0.0
        max_weight_sum = 0.0
        for agent_id, approve in votes.items():
            weight = self.performance_db.get(agent_id, 0.5)  # default neutral
            signed = weight if approve else -weight
            weighted_sum += signed
            max_weight_sum += weight  # used to normalise 0-1

        if max_weight_sum == 0:
            return False

        # Normalised approval ratio in range 0-1
        approval_ratio = (weighted_sum + max_weight_sum) / (2 * max_weight_sum)
        return approval_ratio >= self.approval_threshold

    # ------------------------------------------------------------------
    # Representation helpers for logging / debugging
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"VoteRegistry(agents={len(self.performance_db)}, " f"threshold={self.approval_threshold})"
