"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 AGENT MEMORY SYSTEM - SCHWABOT AI AGENT PERFORMANCE TRACKER
==============================================================

Advanced persistent scorekeeper for AI agent voting performance with mathematical rigor.

Mathematical Components:
- Exponential moving average: new_score = (decay * current) + ((1-decay) * (current + reward))
- Score clamping to [0, 1] range with sigmoid normalization
- Decay factor of 0.9 for historical influence
- Confidence intervals based on performance variance
- Agent consensus building with weighted voting

Features:
- Persistent JSON storage for session survival
- Real-time performance tracking
- Agent consensus and voting mechanisms
- Confidence scoring and uncertainty quantification
- Integration with hash-based trading system
"""

import json
import logging
import math
import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Import dependencies
try:
from core.math_cache import MathResultCache
from core.math_config_manager import MathConfigManager
from core.math_orchestrator import MathOrchestrator
MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
MATH_INFRASTRUCTURE_AVAILABLE = False
logger.warning("Math infrastructure not available")

@dataclass
class AgentScore:
"""Class for Schwabot trading functionality."""
"""Individual agent score with confidence and metadata."""
score: float = 0.5
confidence: float = 0.5
total_votes: int = 0
correct_votes: int = 0
last_update: float = field(default_factory=time.time)
performance_history: List[float] = field(default_factory=list)
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
"""Class for Schwabot trading functionality."""
"""Result of agent consensus voting."""
consensus_decision: bool
confidence: float
participating_agents: List[str]
vote_distribution: Dict[str, float]
uncertainty: float


class AgentMemory:
"""Class for Schwabot trading functionality."""
"""
🧠 Advanced Agent Memory System

Tracks and persists agent performance scores with mathematical rigor.
Provides consensus building and confidence scoring for AI agent voting.
"""

def __init__(self, store_path: Optional[str | pathlib.Path] = None) -> None:
"""
Initialize agent memory with optional custom store path.

Args:
store_path: Path to store agent scores (default: core/agent_scores.json)
"""
self.path = pathlib.Path(store_path) if store_path else pathlib.Path(__file__).parent / "agent_scores.json"
self._scores: Dict[str, AgentScore] = {}
self._decay = 0.9  # Exponential moving average decay factor
self._min_confidence = 0.1
self._max_history = 1000  # Maximum history entries per agent

# Initialize math infrastructure if available
if MATH_INFRASTRUCTURE_AVAILABLE:
self.math_config = MathConfigManager()
self.math_cache = MathResultCache()
self.math_orchestrator = MathOrchestrator()

self._load()
logger.info(f"🧠 AgentMemory initialized with {len(self._scores)} agents")

def get_performance_db(self) -> Dict[str, float]:
"""Return a copy of the agent→score mapping."""
return {agent_id: score.score for agent_id, score in self._scores.items()}

def update_score(self, agent_id: str, reward: float, confidence: float = 0.5) -> None:
"""
Update agent_id score with reward in [-1, 1].

Args:
agent_id: Unique identifier for the agent
reward: Performance reward in range [-1, 1]
confidence: Confidence in the reward (0-1)
"""
if agent_id not in self._scores:
self._scores[agent_id] = AgentScore()

score_obj = self._scores[agent_id]
current_score = score_obj.score

# Exponential moving average with confidence weighting
weighted_reward = reward * confidence
new_score = (self._decay * current_score) + ((1 - self._decay) * (current_score + weighted_reward))

# Clamp to [0, 1] range
score_obj.score = max(0.0, min(1.0, new_score))

# Update confidence based on performance consistency
score_obj.confidence = self._calculate_confidence(agent_id, confidence)

# Update metadata
score_obj.last_update = time.time()
score_obj.performance_history.append(reward)

# Limit history size
if len(score_obj.performance_history) > self._max_history:
score_obj.performance_history = score_obj.performance_history[-self._max_history:]

self._save()
logger.debug(f"Updated agent {agent_id}: score={score_obj.score:.3f}, confidence={score_obj.confidence:.3f}")

def record_vote(self, agent_id: str, vote: bool, was_correct: bool) -> None:
"""
Record an agent's vote and whether it was correct.

Args:
agent_id: Agent identifier
vote: Agent's vote (True/False)
was_correct: Whether the vote was correct
"""
if agent_id not in self._scores:
self._scores[agent_id] = AgentScore()

score_obj = self._scores[agent_id]
score_obj.total_votes += 1

if was_correct:
score_obj.correct_votes += 1
reward = 0.1  # Small positive reward for correct vote
else:
reward = -0.1  # Small negative reward for incorrect vote

# Update score based on vote accuracy
self.update_score(agent_id, reward, confidence=0.5)

def get_score(self, agent_id: str) -> float:
"""Get current score for an agent."""
return self._scores.get(agent_id, AgentScore()).score

def get_confidence(self, agent_id: str) -> float:
"""Get current confidence for an agent."""
return self._scores.get(agent_id, AgentScore()).confidence

def reset_score(self, agent_id: str) -> None:
"""Reset an agent's score to neutral (0.5)."""
self._scores[agent_id] = AgentScore()
self._save()
logger.info(f"Reset agent {agent_id} score to neutral")

def get_top_agents(self, limit: int = 5, min_confidence: float = 0.3) -> List[Tuple[str, float, float]]:
"""
Get top performing agents sorted by score.

Args:
limit: Maximum number of agents to return
min_confidence: Minimum confidence threshold

Returns:
List of (agent_id, score, confidence) tuples
"""
qualified_agents = [
(agent_id, score.score, score.confidence)
for agent_id, score in self._scores.items()
if score.confidence >= min_confidence
]

sorted_agents = sorted(qualified_agents, key=lambda x: x[1], reverse=True)
return sorted_agents[:limit]

def build_consensus(self, agent_votes: Dict[str, bool], min_participation: float = 0.5) -> ConsensusResult:
"""
Build consensus from multiple agent votes.

Args:
agent_votes: Dictionary of agent_id -> vote
min_participation: Minimum fraction of agents that must participate

Returns:
ConsensusResult with decision and confidence
"""
if not agent_votes:
return ConsensusResult(
consensus_decision=False,
confidence=0.0,
participating_agents=[],
vote_distribution={},
uncertainty=1.0
)

# Calculate participation rate
total_agents = len(self._scores)
participating_agents = list(agent_votes.keys())
participation_rate = len(participating_agents) / max(total_agents, 1)

if participation_rate < min_participation:
return ConsensusResult(
consensus_decision=False,
confidence=0.0,
participating_agents=participating_agents,
vote_distribution={},
uncertainty=1.0
)

# Weight votes by agent confidence and score
weighted_votes = {}
total_weight = 0.0

for agent_id, vote in agent_votes.items():
if agent_id in self._scores:
score_obj = self._scores[agent_id]
weight = score_obj.score * score_obj.confidence
weighted_votes[agent_id] = weight if vote else -weight
total_weight += abs(weight)

if total_weight == 0:
return ConsensusResult(
consensus_decision=False,
confidence=0.0,
participating_agents=participating_agents,
vote_distribution=weighted_votes,
uncertainty=1.0
)

# Calculate consensus
consensus_value = sum(weighted_votes.values()) / total_weight
consensus_decision = consensus_value > 0

# Calculate confidence and uncertainty
confidence = abs(consensus_value)
uncertainty = 1.0 - confidence

return ConsensusResult(
consensus_decision=consensus_decision,
confidence=confidence,
participating_agents=participating_agents,
vote_distribution=weighted_votes,
uncertainty=uncertainty
)

def get_agent_stats(self) -> Dict[str, Any]:
"""Get comprehensive statistics about agent performance."""
if not self._scores:
return {
"total_agents": 0,
"avg_score": 0.5,
"best_score": 0.5,
"avg_confidence": 0.5,
"total_votes": 0,
"participation_rate": 0.0
}

total_agents = len(self._scores)
scores = [score.score for score in self._scores.values()]
confidences = [score.confidence for score in self._scores.values()]
total_votes = sum(score.total_votes for score in self._scores.values())

return {
"total_agents": total_agents,
"avg_score": np.mean(scores),
"best_score": max(scores),
"worst_score": min(scores),
"score_std": np.std(scores),
"avg_confidence": np.mean(confidences),
"total_votes": total_votes,
"participation_rate": total_votes / max(total_agents, 1)
}

def _calculate_confidence(self, agent_id: str, new_confidence: float) -> float:
"""Calculate updated confidence based on performance consistency."""
score_obj = self._scores[agent_id]

if len(score_obj.performance_history) < 5:
return new_confidence

# Calculate performance variance
recent_performance = score_obj.performance_history[-20:]  # Last 20 entries
variance = np.var(recent_performance) if len(recent_performance) > 1 else 1.0

# Lower variance = higher confidence
consistency_factor = max(0.1, 1.0 - variance)

# Combine with new confidence
updated_confidence = (0.7 * score_obj.confidence + 0.3 * new_confidence) * consistency_factor

return max(self._min_confidence, min(1.0, updated_confidence))

def _load(self) -> None:
"""Load scores from JSON file."""
if self.path.exists():
try:
data = json.loads(self.path.read_text())
# Convert legacy format to new format
for agent_id, score_data in data.items():
if isinstance(score_data, (int, float)):
# Legacy format: just a score
self._scores[agent_id] = AgentScore(score=float(score_data))
else:
# New format: AgentScore object
self._scores[agent_id] = AgentScore(**score_data)
logger.info(f"Loaded {len(self._scores)} agents from {self.path}")
except Exception as e:
logger.error(f"Failed to load agent scores: {e}")
self._scores = {}
else:
self._scores = {}

def _save(self) -> None:
"""Save scores to JSON file."""
try:
# Convert AgentScore objects to dictionaries
data = {}
for agent_id, score_obj in self._scores.items():
data[agent_id] = {
"score": score_obj.score,
"confidence": score_obj.confidence,
"total_votes": score_obj.total_votes,
"correct_votes": score_obj.correct_votes,
"last_update": score_obj.last_update,
"performance_history": score_obj.performance_history[-100:],  # Keep last 100
"metadata": score_obj.metadata
}

self.path.write_text(json.dumps(data, indent=2))
except Exception as e:
logger.error(f"Failed to save agent scores: {e}")


# Factory function for compatibility
def create_agent_memory(store_path: Optional[str | pathlib.Path] = None) -> AgentMemory:
"""Create an AgentMemory instance."""
return AgentMemory(store_path)