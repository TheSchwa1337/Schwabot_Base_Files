"""
agent_memory.py
---------------
Persistent scorekeeper for AI agent voting performance.

Scores are stored in a simple JSON file so they survive between Schwabot
sessions.  Each agent starts with a neutral 0.5 score unless defined
otherwise.  Scores are clamped to the range 0‒1 and updated via a simple moving
average rule.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict

_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent / "agent_scores.json"

_DECAY = 0.9  # how much past performance influences the new score


class AgentMemory:
    """Tracks and persists agent performance scores."""

    def __init__(self, store_path: str | pathlib.Path | None = None) -> None:
        self.path = pathlib.Path(store_path) if store_path else _DEFAULT_PATH
        self._scores: Dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_performance_db(self) -> Dict[str, float]:
        """Return a *copy* of the agent→score mapping."""
        return dict(self._scores)

    def update_score(self, agent_id: str, reward: float) -> None:
        """Update *agent_id* score with *reward* in [-1, 1].

        Positive reward increases trust; negative decreases.
        """
        cur = self._scores.get(agent_id, 0.5)
        # Simple exponential moving average
        new_score = (_DECAY * cur) + ((1 - _DECAY) * (cur + reward))
        self._scores[agent_id] = max(0.0, min(1.0, new_score))
        self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                self._scores = json.loads(self.path.read_text())
            except Exception:
                self._scores = {}
        else:
            self._scores = {}

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._scores, indent=2))
        except Exception as exc:  # pragma: no cover
            print(f"[AgentMemory] Failed to save scores: {exc}") 