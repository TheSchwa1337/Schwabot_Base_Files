"""
Profit Protection System
Implements sophisticated profit protection mechanisms with dynamic thresholds
and adaptive position sizing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from .cooldown_manager import CooldownManager, CooldownScope, CooldownRule
import json
import hashlib

@dataclass
class ProfitThreshold:
    """Dynamic profit threshold with decay"""
    initial_value: float
    decay_rate: float
    min_value: float
    current_value: float = field(init=False)
    
    def __post_init__(self):
        self.current_value = self.initial_value
        
    def update(self) -> float:
        """Update threshold value with decay"""
        self.current_value = max(
            self.min_value,
            self.current_value * self.decay_rate
        )
        return self.current_value

@dataclass
class ProfitLevel:
    """Profit level with associated actions"""
    level: float
    actions: List[str]
    cooldown_seconds: float
    position_size_multiplier: float

class ProfitProtectionSystem:
    """Manages profit protection with dynamic thresholds and adaptive sizing"""
    
    def __init__(self):
        self.registered_assets: Dict[str, float] = {}
        self.profit_thresholds: Dict[str, ProfitThreshold] = {}
        self.position_sizes: Dict[str, float] = {}
        self.position_sizes = {}
        self.last_profit_updates: Dict[str, datetime] = {}
        self.cooldown_manager = CooldownManager(rules=create_profit_protection_rules())
        self.profit_levels: List[ProfitLevel] = [
            ProfitLevel(ProfitThreshold(decay_rate=0.95)),  # Example use
            ProfitLevel(ProfitThreshold(decay_rate=0.90))
        ]
        
    def can_proceed(self, basket_id: str) -> bool:
        """Check if a basket can proceed based on profit protection rules"""
        return True  # Stub implementation
        
    def register_asset(self, asset_id: str, decay_rate: float = 0.97):
        self.registered_assets[asset_id] = 1.0
        self.profit_thresholds[asset_id] = ProfitThreshold(decay_rate=decay_rate)
        self.position_sizes[asset_id] = 1.0
        self.last_profit_updates[asset_id] = datetime.now()
        
    def log_profit_snapshot(self, asset_id: str, profit: float):
        data = {
            "asset_id": asset_id,
            "profit": profit,
            "threshold": self.profit_thresholds[asset_id].current_value,
            "position_size": self.position_sizes[asset_id],
            "timestamp": datetime.now().isoformat()
        }
        encoded = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def update_profit(self, asset_id: str, profit: float):
        snapshot_hash = self.log_profit_snapshot(asset_id, profit)

        # Threshold decay
        if asset_id in self.profit_thresholds:
            self.profit_thresholds[asset_id].current_value *= self.profit_thresholds[asset_id].decay_rate

        # Evaluate profit levels
        for level in self.profit_levels:
            if profit >= level.level:
                self.position_sizes[asset_id] *= level.position_size_multiplier
                break  # Only apply first matching level

        self.last_profit_updates[asset_id] = datetime.now()
        return snapshot_hash
        
    def get_position_size(self, asset_id: str) -> float:
        """Get current position size for asset"""
        return self.position_sizes.get(asset_id, 1.0)
        
    def can_increase_position(self, asset_id: str, current_profit: float) -> bool:
        """Check if position can be increased based on profit levels"""
        if asset_id not in self.profit_thresholds:
            return True
            
        threshold = self.profit_thresholds[asset_id]
        return current_profit >= threshold.current_value
        
    def get_profit_metrics(self, asset_id: str) -> Dict[str, float]:
        """Get current profit metrics for asset"""
        if asset_id not in self.profit_thresholds:
            return {}
            
        return {
            "current_threshold": self.profit_thresholds[asset_id].current_value,
            "position_size": self.position_sizes[asset_id],
            "time_since_update": (datetime.now() - self.last_profit_updates.get(asset_id, datetime.now())).total_seconds()
        }

    def adjust_threshold_by_volatility(self, asset_id: str, volatility_score: float):
        """Adjust decay rate based on volatility"""
        if asset_id in self.profit_thresholds:
            threshold = self.profit_thresholds[asset_id]
            # Example: higher volatility means slower decay
            decay_adjustment = max(0.85, min(1.0, 1.1 - volatility_score))
            threshold.decay_rate *= decay_adjustment

    def apply_trend_sniffer_adjustments(self, asset_id: str, trend_strength: float, direction: str):
        if direction == "up" and trend_strength > 0.7:
            self.position_sizes[asset_id] *= 1.1
        elif direction == "down" and trend_strength > 0.5:
            self.position_sizes[asset_id] *= 0.85

def create_profit_protection_rules() -> List[CooldownRule]:
    """Create profit protection cooldown rules"""
    return [
        # Profit level reached rule
        CooldownRule(
            rule_id="profit_level_reached",
            trigger_events=["profit_level_reached"],
            conditions=lambda d: d.get("profit_level", 0) > 0,
            cooldown_seconds=lambda d: d.get("cooldown_seconds", 300),
            scope=CooldownScope.ASSET,
            target_id=lambda d: d.get("asset_id"),
            actions=lambda d: d.get("actions", ["reduce_position_size"]),
            priority=100
        ),
        
        # Profit decay rule
        CooldownRule(
            rule_id="profit_decay",
            trigger_events=["profit_update"],
            conditions=lambda d: d.get("profit_ratio", 0) < d.get("threshold", 0),
            cooldown_seconds=60,
            scope=CooldownScope.ASSET,
            target_id=lambda d: d.get("asset_id"),
            actions=["reduce_position_size_25"],
            priority=90
        ),
        
        # Profit acceleration rule
        CooldownRule(
            rule_id="profit_acceleration",
            trigger_events=["profit_update"],
            conditions=lambda d: d.get("profit_ratio", 0) > d.get("threshold", 0) * 1.5,
            cooldown_seconds=30,
            scope=CooldownScope.ASSET,
            target_id=lambda d: d.get("asset_id"),
            actions=["increase_position_size_25"],
            priority=85
        )
    ] 

system = ProfitProtectionSystem()
system.register_asset("BTC", decay_rate=0.96)
system.update_profit("BTC", profit=0.18)
system.adjust_threshold_by_volatility("BTC", volatility_score=0.91)
system.apply_trend_sniffer_adjustments("BTC", trend_strength=0.8, direction="up") 