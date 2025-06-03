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
        self.profit_thresholds: Dict[str, float] = {}
        
    def can_proceed(self, basket_id: str) -> bool:
        """Check if a basket can proceed based on profit protection rules"""
        return True  # Stub implementation
        
    def register_asset(self, asset: str, position_size: float) -> None:
        """Register an asset with its position size for profit protection"""
        self.registered_assets[asset] = position_size
        
    def update_profit(self, asset_id: str, current_profit: float) -> List[str]:
        """Update profit state and return required actions"""
        if asset_id not in self.profit_thresholds:
            return []
            
        # Update profit threshold
        threshold = self.profit_thresholds[asset_id]
        threshold.update()
        
        # Update cooldown manager
        self.cooldown_manager.update_profit(asset_id, current_profit)
        
        # Determine required actions
        actions = []
        for level in sorted(self.profit_levels, key=lambda x: x.level, reverse=True):
            if current_profit >= level.level:
                # Apply cooldown
                self.cooldown_manager.register_event(
                    "profit_level_reached",
                    {
                        "asset_id": asset_id,
                        "profit_level": level.level,
                        "current_profit": current_profit
                    }
                )
                
                # Update position size
                self.position_sizes[asset_id] *= level.position_size_multiplier
                
                # Add actions
                actions.extend(level.actions)
                break
                
        return actions
        
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