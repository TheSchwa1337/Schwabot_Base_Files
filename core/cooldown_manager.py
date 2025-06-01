"""
Enhanced Cooldown Management System with Forever Fractal Integration
Implements sophisticated cooldown rules with fractal state tracking and triplet matching
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Tuple
import time
import numpy as np
from datetime import datetime
from .fractal_core import ForeverFractalCore, FractalState
from .triplet_matcher import TripletMatcher, TripletMatch

class CooldownScope(Enum):
    """Scope of cooldown application"""
    GLOBAL = auto()         # affects entire system
    ASSET = auto()          # specific asset
    STRATEGY = auto()       # specific strategy
    BASKET = auto()         # specific basket
    PROFIT_LEVEL = auto()   # profit-based cooldown
    FRACTAL = auto()        # fractal-based cooldown

@dataclass
class FractalCooldownState:
    """Represents fractal state for cooldown management"""
    vector: List[float]
    phase: float
    entropy: float
    timestamp: float
    coherence_score: float = 0.0
    is_mirror: bool = False

@dataclass
class CooldownRule:
    """Enhanced cooldown rule with fractal integration"""
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
    profit_decay_rate: float = 0.95
    fractal_threshold: float = 0.7  # Minimum coherence for fractal matching
    fractal_decay: float = 0.8      # Decay rate for fractal influence

    # Runtime state
    _active: bool = field(init=False, default=False)
    _activated_tick: int = field(init=False, default=0)
    _activated_time: float = field(init=False, default=0.0)
    _current_profit_threshold: Optional[float] = field(init=False, default=None)
    _fractal_states: List[FractalCooldownState] = field(init=False, default_factory=list)

    def maybe_activate(self, event: str, payload: Dict[str, Any], tick: int) -> bool:
        """Activate rule if conditions are met, including fractal state"""
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

    def still_active(self, tick: int, current_profit: Optional[float] = None,
                    fractal_state: Optional[FractalCooldownState] = None) -> bool:
        """Check if rule is still active, considering profit and fractal thresholds"""
        if not self._active:
            return False

        # Check time-based conditions
        tick_ok = (self.cooldown_ticks == 0 or tick < self._activated_tick + self.cooldown_ticks)
        time_ok = (self.cooldown_seconds == 0 or time.time() < self._activated_time + self.cooldown_seconds)

        # Check profit threshold if applicable
        profit_ok = True
        if self._current_profit_threshold is not None and current_profit is not None:
            profit_ok = current_profit >= self._current_profit_threshold
            self._current_profit_threshold *= self.profit_decay_rate

        # Check fractal state if applicable
        fractal_ok = True
        if fractal_state and self.scope == CooldownScope.FRACTAL:
            fractal_ok = fractal_state.coherence_score >= self.fractal_threshold
            # Apply fractal decay
            self.fractal_threshold *= self.fractal_decay

        if tick_ok and time_ok and profit_ok and fractal_ok:
            return True

        # Deactivate if any condition fails
        self._active = False
        return False

class CooldownManager:
    """Enhanced cooldown manager with fractal integration"""
    
    def __init__(self, rules: List[CooldownRule]):
        self._rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self._tick_counter = 0
        self._profit_history: Dict[str, List[float]] = {}
        self._last_profit_update: Dict[str, float] = {}
        
        # Initialize fractal components
        self.fractal_core = ForeverFractalCore(
            decay_power=2.0,
            terms=50,
            dimension=3
        )
        self.triplet_matcher = TripletMatcher(
            fractal_core=self.fractal_core,
            epsilon=0.1,
            min_coherence=0.7
        )
        self._fractal_states: Dict[str, List[FractalCooldownState]] = {}

    def register_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Register a trading event with fractal state tracking"""
        self._tick_counter += 1
        
        # Generate fractal state
        fractal_vector = self.fractal_core.generate_fractal_vector(
            t=time.time(),
            phase_shift=payload.get('phase_angle', 0.0)
        )
        
        # Create fractal state
        fractal_state = FractalCooldownState(
            vector=fractal_vector,
            phase=payload.get('phase_angle', 0.0),
            entropy=payload.get('entropy', 0.0),
            timestamp=time.time()
        )
        
        # Store fractal state
        target_id = payload.get('target_id')
        if target_id:
            if target_id not in self._fractal_states:
                self._fractal_states[target_id] = []
            self._fractal_states[target_id].append(fractal_state)
            
            # Check for triplet matches
            if len(self._fractal_states[target_id]) >= 3:
                recent_states = self._fractal_states[target_id][-3:]
                match = self.triplet_matcher.find_matching_triplet(recent_states)
                if match:
                    fractal_state.coherence_score = match.coherence
                    fractal_state.is_mirror = match.is_mirror
                    
                    # Apply fractal correction if needed
                    if match.coherence > 0.9:
                        self._apply_fractal_correction(target_id, match)
        
        # Activate rules
        for rule in self._rules:
            rule.maybe_activate(event, payload, self._tick_counter)

    def _apply_fractal_correction(self, target_id: str, match: TripletMatch) -> None:
        """Apply fractal-based correction to cooldown state"""
        # Get correction matrix
        correction = self.fractal_core.compute_correction_vector(match.states)
        
        # Update fractal states with correction
        if target_id in self._fractal_states:
            self._fractal_states[target_id][-1].vector = correction

    def update_profit(self, target_id: str, current_profit: float) -> None:
        """Update profit history for a target"""
        if target_id not in self._profit_history:
            self._profit_history[target_id] = []
        self._profit_history[target_id].append(current_profit)
        self._last_profit_update[target_id] = time.time()

    def can_proceed(self, scope: CooldownScope, target_id: Optional[str] = None) -> bool:
        """Check if action can proceed, considering fractal state"""
        current_profit = None
        fractal_state = None
        
        if target_id:
            if target_id in self._profit_history:
                current_profit = self._profit_history[target_id][-1]
            if target_id in self._fractal_states:
                fractal_state = self._fractal_states[target_id][-1]

        for rule in self._rules:
            if not rule.still_active(self._tick_counter, current_profit, fractal_state):
                continue

            if rule.scope == CooldownScope.GLOBAL:
                return False
            if rule.scope == scope and (rule.target_id is None or rule.target_id == target_id):
                return False
        return True

    def get_active_actions(self, scope: CooldownScope, target_id: Optional[str] = None) -> List[str]:
        """Get active actions for scope/target, including fractal-based actions"""
        actions = []
        current_profit = None
        fractal_state = None
        
        if target_id:
            if target_id in self._profit_history:
                current_profit = self._profit_history[target_id][-1]
            if target_id in self._fractal_states:
                fractal_state = self._fractal_states[target_id][-1]

        for rule in self._rules:
            if rule.still_active(self._tick_counter, current_profit, fractal_state):
                if rule.scope == CooldownScope.GLOBAL or (rule.scope == scope and (rule.target_id is None or rule.target_id == target_id)):
                    actions.extend(rule.actions)
                    
                    # Add fractal-specific actions
                    if fractal_state and fractal_state.coherence_score > 0.9:
                        actions.extend(["apply_fractal_correction", "monitor_mirror_pattern"])
                    
        return list(dict.fromkeys(actions))  # Remove duplicates while preserving order

    def get_fractal_metrics(self, target_id: str) -> Dict[str, float]:
        """Get fractal metrics for a target"""
        if target_id not in self._fractal_states:
            return {}
            
        states = self._fractal_states[target_id]
        if not states:
            return {}
            
        return {
            'coherence_score': states[-1].coherence_score,
            'is_mirror': states[-1].is_mirror,
            'phase': states[-1].phase,
            'entropy': states[-1].entropy,
            'time_since_update': time.time() - states[-1].timestamp
        }

def create_default_rules() -> List[CooldownRule]:
    """Create default cooldown rules with fractal integration"""
    return [
        # Fractal-based rule
        CooldownRule(
            rule_id="fractal_coherence",
            trigger_events=["fractal_match", "mirror_detected"],
            conditions=lambda d: d.get("coherence_score", 0) > 0.9,
            cooldown_seconds=180,
            scope=CooldownScope.FRACTAL,
            target_id=lambda d: d.get("target_id"),
            actions=["apply_fractal_correction", "monitor_mirror_pattern"],
            priority=100,
            fractal_threshold=0.9,
            fractal_decay=0.95
        ),
        
        # Profit protection rule
        CooldownRule(
            rule_id="profit_protection",
            trigger_events=["take_profit"],
            conditions=lambda d: d.get("profit_ratio", 0) > 0.02,
            cooldown_seconds=300,
            scope=CooldownScope.ASSET,
            target_id=lambda d: d.get("symbol"),
            actions=["reduce_position_size", "tighten_stops"],
            priority=90,
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
            priority=80
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
            priority=70
        )
    ] 