"""
Basket Swapping System
Implements sophisticated basket swapping logic with profit protection
and cooldown integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from datetime import datetime
from .cooldown_manager import CooldownManager, CooldownScope, CooldownRule
from .profit_protection import ProfitProtectionSystem

@dataclass
class BasketState:
    """Current state of a trading basket"""
    basket_id: str
    assets: Set[str]
    total_profit: float
    stability_score: float
    paradox_score: float
    memory_coherence: float
    last_update: datetime
    position_sizes: Dict[str, float]
    profit_levels: Dict[str, float]

@dataclass
class SwapCriteria:
    """Criteria for basket swapping"""
    min_profit_threshold: float
    min_stability_score: float
    max_paradox_score: float
    min_memory_coherence: float
    max_position_size: float
    min_time_between_swaps: float

class BasketSwapper:
    """Manages basket swapping with profit protection and cooldown integration"""
    
    def __init__(self, 
                 cooldown_manager: CooldownManager,
                 profit_protection: ProfitProtectionSystem,
                 swap_criteria: Optional[SwapCriteria] = None):
        self.cooldown_manager = cooldown_manager
        self.profit_protection = profit_protection
        self.swap_criteria = swap_criteria or SwapCriteria(
            min_profit_threshold=0.02,    # 2% minimum profit
            min_stability_score=0.7,      # 70% stability
            max_paradox_score=0.3,        # 30% paradox
            min_memory_coherence=0.8,     # 80% memory coherence
            max_position_size=1.0,        # 100% position size
            min_time_between_swaps=300.0  # 5 minutes
        )
        
        self.active_baskets: Dict[str, BasketState] = {}
        self.last_swap_time: Dict[str, datetime] = {}
        
    def register_basket(self, basket_id: str, assets: Set[str]) -> None:
        """Register a new trading basket"""
        self.active_baskets[basket_id] = BasketState(
            basket_id=basket_id,
            assets=assets,
            total_profit=0.0,
            stability_score=1.0,
            paradox_score=0.0,
            memory_coherence=1.0,
            last_update=datetime.now(),
            position_sizes={asset: 1.0 for asset in assets},
            profit_levels={asset: 0.0 for asset in assets}
        )
        
    def update_basket_state(self, 
                          basket_id: str,
                          total_profit: float,
                          stability_score: float,
                          paradox_score: float,
                          memory_coherence: float,
                          asset_profits: Dict[str, float]) -> List[str]:
        """Update basket state and return required actions"""
        if basket_id not in self.active_baskets:
            return []
            
        basket = self.active_baskets[basket_id]
        basket.total_profit = total_profit
        basket.stability_score = stability_score
        basket.paradox_score = paradox_score
        basket.memory_coherence = memory_coherence
        basket.last_update = datetime.now()
        
        # Update profit levels
        for asset, profit in asset_profits.items():
            if asset in basket.profit_levels:
                basket.profit_levels[asset] = profit
                
        # Check if swap is needed
        actions = []
        if self._should_swap_basket(basket):
            actions.extend(self._execute_basket_swap(basket))
            
        return actions
        
    def _should_swap_basket(self, basket: BasketState) -> bool:
        """Determine if basket should be swapped"""
        # Check cooldown
        if not self.cooldown_manager.can_proceed(CooldownScope.BASKET, basket.basket_id):
            return False
            
        # Check time since last swap
        last_swap = self.last_swap_time.get(basket.basket_id)
        if last_swap and (datetime.now() - last_swap).total_seconds() < self.swap_criteria.min_time_between_swaps:
            return False
            
        # Check profit threshold
        if basket.total_profit < self.swap_criteria.min_profit_threshold:
            return False
            
        # Check stability
        if basket.stability_score < self.swap_criteria.min_stability_score:
            return False
            
        # Check paradox
        if basket.paradox_score > self.swap_criteria.max_paradox_score:
            return False
            
        # Check memory coherence
        if basket.memory_coherence < self.swap_criteria.min_memory_coherence:
            return False
            
        return True
        
    def _execute_basket_swap(self, basket: BasketState) -> List[str]:
        """Execute basket swap and return required actions"""
        # Register swap event
        self.cooldown_manager.register_event(
            "basket_swap",
            {
                "basket_id": basket.basket_id,
                "total_profit": basket.total_profit,
                "stability_score": basket.stability_score,
                "paradox_score": basket.paradox_score,
                "memory_coherence": basket.memory_coherence
            }
        )
        
        # Update last swap time
        self.last_swap_time[basket.basket_id] = datetime.now()
        
        # Reset position sizes
        for asset in basket.assets:
            basket.position_sizes[asset] = 1.0
            self.profit_protection.register_asset(asset, 1.0)
            
        return [
            "reset_position_sizes",
            "block_new_entries",
            "update_profit_thresholds"
        ]
        
    def get_basket_metrics(self, basket_id: str) -> Dict[str, float]:
        """Get current metrics for basket"""
        if basket_id not in self.active_baskets:
            return {}
            
        basket = self.active_baskets[basket_id]
        return {
            "total_profit": basket.total_profit,
            "stability_score": basket.stability_score,
            "paradox_score": basket.paradox_score,
            "memory_coherence": basket.memory_coherence,
            "time_since_update": (datetime.now() - basket.last_update).total_seconds(),
            "time_since_swap": (datetime.now() - self.last_swap_time.get(basket_id, datetime.now())).total_seconds()
        }

def create_basket_swap_rules() -> List[CooldownRule]:
    """Create basket swap cooldown rules"""
    return [
        # Basket swap rule
        CooldownRule(
            rule_id="basket_swap",
            trigger_events=["basket_swap"],
            cooldown_seconds=300,
            scope=CooldownScope.GLOBAL,
            actions=["block_new_entries", "reset_position_sizes"],
            priority=100
        ),
        
        # Basket stability rule
        CooldownRule(
            rule_id="basket_stability",
            trigger_events=["stability_breach"],
            conditions=lambda d: d.get("stability_score", 1) < 0.7,
            cooldown_seconds=600,
            scope=CooldownScope.BASKET,
            target_id=lambda d: d.get("basket_id"),
            actions=["reduce_position_sizes", "block_new_entries"],
            priority=90
        ),
        
        # Basket paradox rule
        CooldownRule(
            rule_id="basket_paradox",
            trigger_events=["paradox_detected"],
            conditions=lambda d: d.get("paradox_score", 0) > 0.3,
            cooldown_seconds=180,
            scope=CooldownScope.BASKET,
            target_id=lambda d: d.get("basket_id"),
            actions=["monitor_only"],
            priority=85
        )
    ] 