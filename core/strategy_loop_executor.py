"""
Strategy Loop Executor
====================

Orchestrates the trading loop with phase-aware strategy execution,
basket rotation, and state persistence.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from .phase_engine import BasketPhaseMap, DataProvider
from .phase_engine.phase_logger import PhaseLogger
from .strategy_logic import StrategyLogic
from .gpu_metrics import GPUMetrics
from .config.unifier import ConfigUnifier

logger = logging.getLogger(__name__)

class StrategyLoopExecutor:
    """Executes the trading loop with phase-aware strategy management"""
    
    def __init__(
        self,
        data_provider: DataProvider,
        config_root: Optional[Path] = None
    ):
        """Initialize the strategy loop executor"""
        self.data_provider = data_provider
        
        # Initialize config unifier
        self.config = ConfigUnifier(config_root)
        self.config.ensure_all()
        
        # Get configs
        self.strategy_config = self.config.get("strategy")
        self.phase_config = self.config.get("phase")
        self.meta_config = self.config.get("meta")
        
        # Initialize components
        self.phase_map = BasketPhaseMap()
        self.strategy_logic = StrategyLogic(
            phase_map=self.phase_map,
            data_provider=data_provider
        )
        self.phase_logger = PhaseLogger()
        self.gpu_metrics = GPUMetrics()
        
        # Load state if exists
        self._load_state()
        
    def _load_state(self) -> None:
        """Load persisted state"""
        state_path = Path(self.meta_config["logging"]["file"]).parent / "state.json"
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    
                # Restore phase map state
                if 'phase_map' in state:
                    self.phase_map.from_dict(state['phase_map'])
                    
                # Restore phase logger state
                if 'phase_transitions' in state:
                    self.phase_logger.load_transitions()
                    
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                
    def _save_state(self) -> None:
        """Save current state"""
        state_path = Path(self.meta_config["logging"]["file"]).parent / "state.json"
        try:
            state = {
                'phase_map': self.phase_map.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'config_hashes': {
                    key: self.config.get_state(key).hash
                    for key in ['strategy', 'phase', 'meta']
                }
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            
    def step(self, basket_id: str) -> Dict[str, Any]:
        """Execute a single step of the trading loop"""
        try:
            # Get current price data
            price_data = self.data_provider.get_price(basket_id)
            
            # Update GPU metrics
            self.gpu_metrics.update(
                price=price_data['price'],
                volume=price_data['volume'],
                bit_depth=price_data.get('bit_depth', 64)
            )
            
            # Get current metrics
            metrics = self.gpu_metrics.get_current_metrics()
            
            # Update phase map
            old_phase = self.phase_map.get_current_phase(basket_id)
            self.phase_map.update_phase_entry(
                basket_id=basket_id,
                metrics=metrics
            )
            new_phase = self.phase_map.get_current_phase(basket_id)
            
            # Log phase transition if occurred
            if old_phase != new_phase:
                self.phase_logger.log_transition(
                    basket_id=basket_id,
                    from_phase=old_phase,
                    to_phase=new_phase,
                    urgency=self.phase_map.get_swap_urgency(basket_id),
                    metrics=metrics
                )
                
            # Check phase conditions
            if self.phase_map.check_basket_swap_condition(basket_id):
                self._rotate_strategy(basket_id)
                
            # Update strategy state
            self.strategy_logic.update_strategy_state(
                basket_id=basket_id,
                metrics=metrics
            )
            
            # Persist state
            self._save_state()
            
            return {
                'basket_id': basket_id,
                'phase': new_phase,
                'metrics': metrics,
                'strategy_state': self.strategy_logic.get_strategy_state(basket_id),
                'config_hashes': {
                    key: self.config.get_state(key).hash
                    for key in ['strategy', 'phase', 'meta']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in trading loop step: {e}")
            raise
            
    def _rotate_strategy(self, basket_id: str) -> None:
        """Rotate strategy based on phase conditions"""
        current_phase = self.phase_map.get_current_phase(basket_id)
        urgency = self.phase_map.get_swap_urgency(basket_id)
        
        # Get fallback pairs from config
        fallback_pairs = self.strategy_config['baskets'][basket_id]['fallback_pairs']
        
        # Select new strategy based on phase and urgency
        if current_phase == 'UNSTABLE':
            # Use most stable fallback pair
            new_basket = self._select_most_stable_basket(fallback_pairs)
            self.strategy_logic.trigger_fallback(basket_id, new_basket)
            
        elif current_phase == 'SMART_MONEY':
            # Use smart money strategy
            self.strategy_logic.activate_smart_money_strategy(basket_id)
            
        elif current_phase == 'OVERLOADED':
            # Use overloaded strategy
            self.strategy_logic.activate_overloaded_strategy(basket_id)
            
        else:  # STABLE
            # Use default strategy
            self.strategy_logic.activate_default_strategy(basket_id)
            
    def _select_most_stable_basket(self, basket_ids: List[str]) -> str:
        """Select the most stable basket from a list"""
        stabilities = {
            basket_id: self.phase_map.get_stability_score(basket_id)
            for basket_id in basket_ids
        }
        return max(stabilities.items(), key=lambda x: x[1])[0]
        
    def get_phase_statistics(self) -> Dict[str, Any]:
        """Get phase transition statistics"""
        return self.phase_logger.get_phase_statistics()
        
    def plot_phase_transitions(
        self,
        basket_id: Optional[str] = None,
        metric: str = 'urgency',
        save_path: Optional[Path] = None
    ) -> None:
        """Plot phase transitions"""
        self.phase_logger.plot_phase_transitions(
            basket_id=basket_id,
            metric=metric,
            save_path=save_path
        )
        
    def plot_phase_heatmap(
        self,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot phase transition heatmap"""
        self.phase_logger.plot_phase_heatmap(save_path=save_path) 