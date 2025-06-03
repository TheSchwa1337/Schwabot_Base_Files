"""
Basket Swapper with SP Core Integration
Implements sophisticated basket swapping system with SP Core integration.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging
import yaml
from pathlib import Path

from core.basket_swap_logic import BasketSwapLogic
from .cooldown_manager import CooldownManager
from .profit_protection import ProfitProtection
from .time_entropy import TimeEntropyEdgeCase
from .bitmap_engine import BitmapEngine
from .profit_tensor import ProfitTensorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SwapCriteria:
    """Criteria for basket swaps"""
    min_valid_signal_threshold: float = 0.3
    min_profit_threshold: float = 0.02
    max_risk_threshold: float = 0.1
    stability_threshold: float = 0.8

class BasketSwapper:
    """Manages basket swapping with profit protection and cooldown integration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.cooldown_manager = CooldownManager()
        self.profit_protection = ProfitProtection()
        self.swap_criteria = SwapCriteria()
        self.time_entropy = TimeEntropyEdgeCase()
        self.bitmap_engine = BitmapEngine()
        self.profit_tensor = ProfitTensorStore()
        self.basket_swap_logic = BasketSwapLogic(config_path)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        if 'basket_swap' not in data:
            raise ValueError("Missing 'basket_swap' root key in config.")
        config = data['basket_swap']
        self._validate_config_schema(config)
        return config
        
    def _validate_config_schema(self, cfg):
        required_keys = ['signal_weights', 'thresholds']
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required key: {key}")
        # ... further checks as in your example ...
            
    def register_basket(self, basket_id: str, assets: List[str], 
                       initial_weights: Optional[Dict[str, float]] = None):
        """Register a new basket for swapping"""
        try:
            # Initialize basket in cooldown manager
            self.cooldown_manager.register_basket(basket_id, assets)
            
            # Set initial weights if provided
            if initial_weights:
                self.profit_protection.set_weights(basket_id, initial_weights)
                
            logger.info(f"Registered basket {basket_id} with {len(assets)} assets")
            
        except Exception as e:
            logger.error(f"Failed to register basket {basket_id}: {str(e)}")
            
    def update_asset_state(self, basket_id: str, asset: str, 
                          state: Dict[str, Any]):
        """Update state for an asset in a basket"""
        try:
            # Update profit protection
            self.profit_protection.update_asset_state(basket_id, asset, state)
            
            # Update time entropy
            self.time_entropy.update_asset_state(asset, state)
            
            # Update bitmap engine
            self.bitmap_engine.update_asset_state(asset, state)
            
            # Update profit tensor
            self.profit_tensor.update_asset_state(asset, state)
            
        except Exception as e:
            logger.error(f"Failed to update asset state for {asset} in {basket_id}: {str(e)}")
            
    def evaluate_swap(self, basket_id: str, pkt: Dict[str, Any], 
                     rings: Dict[str, Any]) -> bool:
        """Evaluate whether a swap should be executed"""
        try:
            # Check cooldown status
            if self.cooldown_manager.is_in_cooldown(basket_id):
                logger.info(f"Basket {basket_id} is in cooldown")
                return False
                
            # Get current weights
            current_weights = self.profit_protection.get_weights(basket_id)
            
            # Calculate signal strength
            signal_strength = self.basket_swap_logic._calculate_signal_strength(
                trust_score=pkt.get('trust_score', 0.0),
                entropy_score=pkt.get('entropy_score', 0.0),
                phase_depth=pkt.get('phase_depth', 0.0)
            )
            
            # Check if signal is strong enough
            if signal_strength < self.swap_criteria.min_valid_signal_threshold:
                logger.info(f"Signal strength {signal_strength:.3f} below threshold")
                return False
                
            # Check profit protection
            if not self.profit_protection.should_allow_swap(basket_id):
                logger.info(f"Profit protection blocked swap for {basket_id}")
                return False
                
            # Check for override conditions
            if self.basket_swap_logic._should_override(basket_id, pkt, rings):
                logger.warning(f"Override triggered for basket {basket_id}")
                return True
                
            # Calculate new weights
            new_weights = self._calculate_new_weights(
                basket_id, current_weights, pkt, rings
            )
            
            # Validate new weights
            if not self._validate_weights(new_weights):
                logger.warning(f"Invalid weights calculated for {basket_id}")
                return False
                
            # Apply swap
            return self.basket_swap_logic.apply_swap(
                trigger_type="evaluation",
                asset_adjustments=new_weights,
                pkt=pkt,
                rings=rings
            )
            
        except Exception as e:
            logger.error(f"Swap evaluation failed for {basket_id}: {str(e)}")
            return False
            
    def _calculate_new_weights(self, basket_id: str, 
                             current_weights: Dict[str, float],
                             pkt: Dict[str, Any],
                             rings: Dict[str, Any]) -> Dict[str, float]:
        """Calculate new weights for basket assets"""
        try:
            # Get time entropy adjustments
            entropy_adjustments = self.time_entropy.calculate_adjustments(
                basket_id, current_weights
            )
            
            # Get bitmap adjustments
            bitmap_adjustments = self.bitmap_engine.calculate_adjustments(
                basket_id, current_weights
            )
            
            # Get profit tensor adjustments
            tensor_adjustments = self.profit_tensor.calculate_adjustments(
                basket_id, current_weights
            )
            
            # Combine adjustments
            new_weights = {}
            for asset in current_weights:
                base_weight = current_weights[asset]
                entropy_adj = entropy_adjustments.get(asset, 0.0)
                bitmap_adj = bitmap_adjustments.get(asset, 0.0)
                tensor_adj = tensor_adjustments.get(asset, 0.0)
                
                # Apply weighted adjustments
                new_weights[asset] = base_weight * (
                    1.0 + 
                    0.4 * entropy_adj +
                    0.3 * bitmap_adj +
                    0.3 * tensor_adj
                )
                
            # Normalize weights
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {k: v/total for k, v in new_weights.items()}
                
            return new_weights
            
        except Exception as e:
            logger.error(f"Weight calculation failed for {basket_id}: {str(e)}")
            return current_weights
            
    def _validate_weights(self, weights: Dict[str, float]) -> bool:
        """Validate calculated weights"""
        try:
            # Check for negative weights
            if any(w < 0 for w in weights.values()):
                return False
                
            # Check sum is close to 1.0
            total = sum(weights.values())
            if not 0.99 <= total <= 1.01:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Weight validation failed: {str(e)}")
            return False
            
    def get_swap_history(self, basket_id: Optional[str] = None) -> List[Any]:
        """Get swap history, optionally filtered by basket"""
        return self.basket_swap_logic.get_swap_history(basket_id)
        
    def clear_history(self):
        """Clear swap history"""
        self.basket_swap_logic.clear_history() 

    def _should_override(self, basket_id: str, pkt: Dict[str, Any], rings: Dict[str, Any]) -> bool:
        try:
            # Check for drift exit
            if pkt.get("force_drift_exit", False):
                logger.warning(f"[SP OVERRIDE] Drift exit triggered for basket {basket_id}")
                return True
            
            # Check for ZPE override
            if pkt.get("force_zpe_override", False):
                logger.warning(f"[SP OVERRIDE] ZPE override triggered for basket {basket_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Override check failed: {str(e)}")
            return False 