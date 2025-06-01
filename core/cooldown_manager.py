"""
Enhanced Cooldown Management System with Profit Protection
Implements sophisticated cooldown rules and profit protection mechanisms
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any
import time
import numpy as np
from datetime import datetime

class CooldownScope(Enum):
    """Scope of cooldown application"""
    GLOBAL = auto()         # affects entire system
    ASSET = auto()          # specific asset
    STRATEGY = auto()       # specific strategy
    BASKET = auto()         # specific basket
    PROFIT_LEVEL = auto()   # profit-based cooldown

@dataclass
class CooldownRule:
    """Enhanced cooldown rule with profit protection"""
    rule_id: str
    trigger_events: List[str]
    conditions: Optional[Callable[[Dict[str, Any]], bool]] = None
    cooldown_ticks: int = 0
    cooldown_seconds: float = 0.0
    scope: CooldownScope = CooldownScope.GLOBAL
    target_id: Optional[str] = None
    actions: List[str] = field(default_factory=lambda: ["block_new_entries"])
    priority: int = 0
    profit_threshold: Optional[float] = None
    profit_decay_rate: float = 0.95  # How quickly profit threshold decays

    # Runtime state
    _active: bool = field(init=False, default=False)
    _activated_tick: int = field(init=False, default=0)
    _activated_time: float = field(init=False, default=0.0)
    _current_profit_threshold: Optional[float] = field(init=False, default=None)

    def maybe_activate(self, event: str, payload: Dict[str, Any], tick: int) -> bool:
        """Activate rule if conditions are met"""
        if self._active:
            return False
        if event not in self.trigger_events:
            return False
        if self.conditions and not self.conditions(payload):
            return False

        # Set profit threshold if applicable
        if self.profit_threshold is not None:
            self._current_profit_threshold = self.profit_threshold

        # Activate rule
        self._active = True
        self._activated_tick = tick
        self._activated_time = time.time()
        return True

    def still_active(self, tick: int, current_profit: Optional[float] = None) -> bool:
        """Check if rule is still active, considering profit thresholds"""
        if not self._active:
            return False

        # Check time-based conditions
        tick_ok = (self.cooldown_ticks == 0 or tick < self._activated_tick + self.cooldown_ticks)
        time_ok = (self.cooldown_seconds == 0 or time.time() < self._activated_time + self.cooldown_seconds)

        # Check profit threshold if applicable
        profit_ok = True
        if self._current_profit_threshold is not None and current_profit is not None:
            profit_ok = current_profit >= self._current_profit_threshold
            # Decay profit threshold
            self._current_profit_threshold *= self.profit_decay_rate

        if tick_ok and time_ok and profit_ok:
            return True

        # Deactivate if any condition fails
        self._active = False
        return False

class CooldownManager:
    """Enhanced cooldown manager with profit protection"""
    
    def __init__(self, rules: List[CooldownRule]):
        self._rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self._tick_counter = 0
        self._profit_history: Dict[str, List[float]] = {}
        self._last_profit_update: Dict[str, float] = {}

    def register_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Register a trading event"""
        self._tick_counter += 1
        for rule in self._rules:
            rule.maybe_activate(event, payload, self._tick_counter)

    def update_profit(self, target_id: str, current_profit: float) -> None:
        """Update profit history for a target"""
        if target_id not in self._profit_history:
            self._profit_history[target_id] = []
        self._profit_history[target_id].append(current_profit)
        self._last_profit_update[target_id] = time.time()

    def can_proceed(self, scope: CooldownScope, target_id: Optional[str] = None) -> bool:
        """Check if action can proceed"""
        current_profit = None
        if target_id and target_id in self._profit_history:
            current_profit = self._profit_history[target_id][-1]

        for rule in self._rules:
            if not rule.still_active(self._tick_counter, current_profit):
                continue

            if rule.scope == CooldownScope.GLOBAL:
                return False
            if rule.scope == scope and (rule.target_id is None or rule.target_id == target_id):
                return False
        return True

    def get_active_actions(self, scope: CooldownScope, target_id: Optional[str] = None) -> List[str]:
        """Get active actions for scope/target"""
        actions = []
        current_profit = None
        if target_id and target_id in self._profit_history:
            current_profit = self._profit_history[target_id][-1]

        for rule in self._rules:
            if rule.still_active(self._tick_counter, current_profit):
                if rule.scope == CooldownScope.GLOBAL or (rule.scope == scope and (rule.target_id is None or rule.target_id == target_id)):
                    actions.extend(rule.actions)
        return list(dict.fromkeys(actions))  # Remove duplicates while preserving order

def create_default_rules() -> List[CooldownRule]:
    """Create default cooldown rules with profit protection"""
    return [
        # Profit protection rule
        CooldownRule(
            rule_id="profit_protection",
            trigger_events=["take_profit"],
            conditions=lambda d: d.get("profit_ratio", 0) > 0.02,
            cooldown_seconds=300,
            scope=CooldownScope.ASSET,
            target_id=lambda d: d.get("symbol"),
            actions=["reduce_position_size", "tighten_stops"],
            priority=100,
            profit_threshold=0.02,
            profit_decay_rate=0.95
        ),
        
        # Market instability rule
        CooldownRule(
            rule_id="market_instability",
            trigger_events=["paradox_detected", "stability_breach"],
            conditions=lambda d: d.get("stability_score", 1) < 0.7,
            cooldown_seconds=600,
            scope=CooldownScope.GLOBAL,
            actions=["block_new_entries", "reduce_leverage"],
            priority=90
        ),
        
        # Pattern break rule
        CooldownRule(
            rule_id="pattern_break",
            trigger_events=["memory_coherence_breach"],
            conditions=lambda d: d.get("memory_score", 1) < 0.8,
            cooldown_seconds=180,
            scope=CooldownScope.ASSET,
            target_id=lambda d: d.get("symbol"),
            actions=["monitor_only"],
            priority=80
        ),
        
        # Rapid profit rule
        CooldownRule(
            rule_id="rapid_profit",
            trigger_events=["take_profit"],
            conditions=lambda d: d.get("profit_ratio", 0) > 0.01 and d.get("time_in_trade", 0) < 300,
            cooldown_seconds=120,
            scope=CooldownScope.ASSET,
            target_id=lambda d: d.get("symbol"),
            actions=["reduce_position_size_50"],
            priority=85,
            profit_threshold=0.01
        ),
        
        # Basket swap rule
        CooldownRule(
            rule_id="basket_swap",
            trigger_events=["basket_swap"],
            cooldown_seconds=60,
            scope=CooldownScope.GLOBAL,
            actions=["block_new_entries"],
            priority=70
        )
    ] 