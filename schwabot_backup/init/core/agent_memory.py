"""
Agent Memory System
==================

Persistent scorekeeper for AI agent voting performance.

Scores are stored in a simple JSON file so they survive between Schwabot
sessions. Each agent starts with a neutral 0.5 score unless defined
otherwise. Scores are clamped to the range 0-1 and updated via exponential
moving average.

Mathematical Components:
- Exponential moving average: new_score = (decay * current) + ((1-decay) * (current + reward))
- Score clamping to [0, 1] range
- Decay factor of 0.9 for historical influence
"""

import json
import pathlib
from typing import Dict

_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent / "agent_scores.json"
_DECAY = 0.9  # how much past performance influences the new score


class AgentMemory:
    """Tracks and persists agent performance scores."""

    def __init__(self,  store_path: str | pathlib.Path | None = None) -> None:
        """Initialize agent memory with optional custom store path."""
        self.path = pathlib.Path(store_path) if store_path else _DEFAULT_PATH
        self._scores: Dict[str, float] = {}
        self._load()

    def get_performance_db(self) -> Dict[str, float]:
        """Return a copy of the agent→score mapping."""
        return dict(self._scores)

    def update_score(self,  agent_id: str, reward: float) -> None:
        """Update agent_id score with reward in [-1, 1].

        Positive reward increases trust; negative decreases.
        Uses exponential moving average for smooth updates.
        """
        cur = self._scores.get(agent_id, 0.5)
        # Exponential moving average
        new_score = (_DECAY * cur) + ((1 - _DECAY) * (cur + reward))
        self._scores[agent_id] = max(0.0, min(1.0, new_score))
        self._save()

    def get_score(self,  agent_id: str) -> float:
        """Get current score for an agent."""
        return self._scores.get(agent_id, 0.5)

    def reset_score(self,  agent_id: str) -> None:
        """Reset an agent's score to neutral (0.5)."""
        self._scores[agent_id] = 0.5
        self._save()

    def get_top_agents(self,  limit: int = 5) -> list[tuple[str, float]]:
        """Get top performing agents sorted by score."""
        sorted_agents = sorted(self._scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_agents[:limit]

    def get_agent_stats(self) -> Dict[str, any]:
        """Get statistics about agent performance."""
        if not self._scores:
            return {"total_agents": 0, "avg_score": 0.5, "best_score": 0.5}

        total_agents = len(self._scores)
        avg_score = sum(self._scores.values()) / total_agents
        best_score = max(self._scores.values())

        return {"total_agents": total_agents, "avg_score": avg_score, "best_score": best_score}

    def _load(self) -> None:
        """Load scores from JSON file."""
        if self.path.exists():
            try:
                self._scores = json.loads(self.path.read_text())
            except Exception:
                self._scores = {}
        else:
            self._scores = {}

    def _save(self) -> None:
        """Save scores to JSON file."""
        try:
            self.path.write_text(json.dumps(self._scores, indent=2))
        except Exception as exc:
            print(f"[AgentMemory] Failed to save scores: {exc}")
