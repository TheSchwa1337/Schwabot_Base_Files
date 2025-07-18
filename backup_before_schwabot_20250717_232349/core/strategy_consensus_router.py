"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧭 STRATEGY CONSENSUS ROUTER - SCHWABOT LIVE STRATEGY CONSENSUS ROUTER
=====================================================================

Advanced strategy consensus router system for the Schwabot trading system that
manages live strategy consensus and routing decisions.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import logging
import time


logger = logging.getLogger(__name__)


class ConsensusMode(Enum):
"""Class for Schwabot trading functionality."""
"""Consensus modes for strategy routing."""
MAJORITY = "majority"
WEIGHTED = "weighted"
UNANIMOUS = "unanimous"
ADAPTIVE = "adaptive"

class RouteSelectionMode(Enum):
"""Class for Schwabot trading functionality."""
"""Route selection modes."""
BEST_PERFORMANCE = "best_performance"
LOWEST_RISK = "lowest_risk"
HIGHEST_CONFIDENCE = "highest_confidence"
BALANCED = "balanced"

@dataclass
class StrategyVote:
"""Class for Schwabot trading functionality."""
"""Strategy vote with metadata."""
strategy_id: str
vote: str  # "BUY", "SELL", "HOLD"
confidence: float
performance_score: float
risk_score: float
timestamp: float = field(default_factory=time.time)
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsensusResult:
"""Class for Schwabot trading functionality."""
"""Result of consensus calculation."""
final_decision: str
consensus_score: float
participating_strategies: int
vote_distribution: Dict[str, int]
confidence: float
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RouteDecision:
"""Class for Schwabot trading functionality."""
"""Route decision with execution details."""
route_id: str
selected_strategy: str
action: str
confidence: float
execution_priority: int
timestamp: float = field(default_factory=time.time)
metadata: Dict[str, Any] = field(default_factory=dict)

class StrategyConsensusRouter:
"""Class for Schwabot trading functionality."""
"""🧭 Strategy Consensus Router System"""

def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
self.config = config or self._default_config()
self.logger = logging.getLogger(__name__)
self.active_votes: List[StrategyVote] = []
self.consensus_history: List[ConsensusResult] = []
self.route_history: List[RouteDecision] = []
self.total_consensus_checks = 0
self.successful_routes = 0
self._initialize_system()

def _default_config(self) -> Dict[str, Any]:
return {
'enabled': True,
'consensus_threshold': 0.6,
'min_participants': 3,
'vote_timeout': 30.0,
'max_votes': 1000,
}

def _initialize_system(self) -> None:
try:
self.logger.info(f"🧭 Initializing {self.__class__.__name__}")
self.initialized = True
self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
except Exception as e:
self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
self.initialized = False

def submit_vote(self, strategy_vote: StrategyVote) -> bool:
"""Submit a strategy vote for consensus."""
try:
# Clean old votes
current_time = time.time()
timeout = self.config.get('vote_timeout', 30.0)
self.active_votes = [vote for vote in self.active_votes
if current_time - vote.timestamp < timeout]

# Add new vote
self.active_votes.append(strategy_vote)

# Limit vote storage
max_votes = self.config.get('max_votes', 1000)
if len(self.active_votes) > max_votes:
self.active_votes = self.active_votes[-max_votes:]

self.logger.info(f"🧭 Vote submitted by {strategy_vote.strategy_id}: {strategy_vote.vote}")
return True

except Exception as e:
self.logger.error(f"❌ Error submitting vote: {e}")
return False

def calculate_consensus(self, consensus_mode: ConsensusMode = ConsensusMode.WEIGHTED) -> ConsensusResult:
"""Calculate consensus from active votes."""
try:
self.total_consensus_checks += 1

if len(self.active_votes) < self.config.get('min_participants', 3):
return ConsensusResult(
final_decision="HOLD",
consensus_score=0.0,
participating_strategies=len(self.active_votes),
vote_distribution={},
confidence=0.0,
metadata={"error": "Insufficient participants"}
)

# Calculate vote distribution
vote_distribution = {}
for vote in self.active_votes:
vote_distribution[vote.vote] = vote_distribution.get(vote.vote, 0) + 1

# Calculate consensus based on mode
if consensus_mode == ConsensusMode.MAJORITY:
final_decision, consensus_score = self._majority_consensus(vote_distribution)
elif consensus_mode == ConsensusMode.WEIGHTED:
final_decision, consensus_score = self._weighted_consensus()
elif consensus_mode == ConsensusMode.UNANIMOUS:
final_decision, consensus_score = self._unanimous_consensus(vote_distribution)
else:  # ADAPTIVE
final_decision, consensus_score = self._adaptive_consensus()

# Calculate overall confidence
confidence = np.mean([vote.confidence for vote in self.active_votes])

result = ConsensusResult(
final_decision=final_decision,
consensus_score=consensus_score,
participating_strategies=len(self.active_votes),
vote_distribution=vote_distribution,
confidence=confidence
)

self.consensus_history.append(result)

self.logger.info(f"🧭 Consensus: {final_decision} (score: {consensus_score:.3f}, "
f"participants: {len(self.active_votes)})")

return result

except Exception as e:
self.logger.error(f"❌ Error calculating consensus: {e}")
return ConsensusResult(
final_decision="HOLD",
consensus_score=0.0,
participating_strategies=0,
vote_distribution={},
confidence=0.0,
metadata={"error": str(e)}
)


def select_route(self, consensus_result: ConsensusResult, -> None
selection_mode: RouteSelectionMode = RouteSelectionMode.BALANCED) -> RouteDecision:
"""Select execution route based on consensus."""
try:
if not self.active_votes:
return RouteDecision(
route_id="no_route",
selected_strategy="none",
action="HOLD",
confidence=0.0,
execution_priority=0
)

# Select strategy based on mode
if selection_mode == RouteSelectionMode.BEST_PERFORMANCE:
selected_vote = max(self.active_votes, key=lambda v: v.performance_score)
elif selection_mode == RouteSelectionMode.LOWEST_RISK:
selected_vote = min(self.active_votes, key=lambda v: v.risk_score)
elif selection_mode == RouteSelectionMode.HIGHEST_CONFIDENCE:
selected_vote = max(self.active_votes, key=lambda v: v.confidence)
else:  # BALANCED
selected_vote = self._balanced_selection()

# Calculate execution priority
priority = self._calculate_execution_priority(selected_vote, consensus_result)

route_decision = RouteDecision(
route_id=f"route_{int(time.time() * 1000)}",
selected_strategy=selected_vote.strategy_id,
action=consensus_result.final_decision,
confidence=consensus_result.confidence,
execution_priority=priority
)

self.route_history.append(route_decision)
self.successful_routes += 1

self.logger.info(f"🧭 Route selected: {selected_vote.strategy_id} "
f"-> {consensus_result.final_decision} (priority: {priority})")

return route_decision

except Exception as e:
self.logger.error(f"❌ Error selecting route: {e}")
return RouteDecision(
route_id="error",
selected_strategy="none",
action="HOLD",
confidence=0.0,
execution_priority=0,
metadata={"error": str(e)}
)

def _majority_consensus(self, vote_distribution: Dict[str, int]) -> Tuple[str, float]:
"""Calculate majority consensus."""
total_votes = sum(vote_distribution.values())
if total_votes == 0:
return "HOLD", 0.0

# Find majority vote
majority_vote = max(vote_distribution.items(), key=lambda x: x[1])
consensus_score = majority_vote[1] / total_votes

return majority_vote[0], consensus_score

def _weighted_consensus(self) -> Tuple[str, float]:
"""Calculate weighted consensus based on confidence and performance."""
if not self.active_votes:
return "HOLD", 0.0

# Calculate weighted scores for each vote
vote_scores = {}
total_weight = 0.0

for vote in self.active_votes:
weight = vote.confidence * vote.performance_score
vote_scores[vote.vote] = vote_scores.get(vote.vote, 0.0) + weight
total_weight += weight

if total_weight == 0:
return "HOLD", 0.0

# Find highest weighted vote
best_vote = max(vote_scores.items(), key=lambda x: x[1])
consensus_score = best_vote[1] / total_weight

return best_vote[0], consensus_score

def _unanimous_consensus(self, vote_distribution: Dict[str, int]) -> Tuple[str, float]:
"""Calculate unanimous consensus."""
if len(vote_distribution) == 1:
vote = list(vote_distribution.keys())[0]
return vote, 1.0
else:
return "HOLD", 0.0

def _adaptive_consensus(self) -> Tuple[str, float]:
"""Calculate adaptive consensus based on market conditions."""
# Simplified adaptive logic
if len(self.active_votes) >= 5:
return self._weighted_consensus()
else:
return self._majority_consensus({vote.vote: 1 for vote in self.active_votes})

def _balanced_selection(self) -> StrategyVote:
"""Select strategy using balanced criteria."""
if not self.active_votes:
return StrategyVote("none", "HOLD", 0.0, 0.0, 1.0)

# Calculate balanced score for each vote
best_vote = None
best_score = -1.0

for vote in self.active_votes:
# Balanced score: confidence * performance / risk
balanced_score = vote.confidence * vote.performance_score / max(vote.risk_score, 0.1)

if balanced_score > best_score:
best_score = balanced_score
best_vote = vote

return best_vote or self.active_votes[0]


def _calculate_execution_priority(self, selected_vote: StrategyVote, -> None
consensus_result: ConsensusResult) -> int:
"""Calculate execution priority (1-10, higher is more urgent)."""
# Base priority on consensus score and confidence
base_priority = int(consensus_result.consensus_score * 5 + consensus_result.confidence * 5)

# Adjust based on vote characteristics
if selected_vote.vote in ["BUY", "SELL"]:
base_priority += 2

# Clamp to valid range
return max(1, min(10, base_priority))

def start_consensus_system(self) -> bool:
"""Start the consensus system."""
if not self.initialized:
self.logger.error("Consensus system not initialized")
return False

try:
self.logger.info("🧭 Starting Strategy Consensus Router system")
return True
except Exception as e:
self.logger.error(f"❌ Error starting consensus system: {e}")
return False

def get_consensus_stats(self) -> Dict[str, Any]:
"""Get consensus system statistics."""
return {
"total_consensus_checks": self.total_consensus_checks,
"successful_routes": self.successful_routes,
"active_votes": len(self.active_votes),
"consensus_history": len(self.consensus_history),
"route_history": len(self.route_history)
}

# Factory function

def create_strategy_consensus_router(config: Optional[Dict[str, Any]] = None) -> StrategyConsensusRouter:
"""Create a StrategyConsensusRouter instance."""
return StrategyConsensusRouter(config)
