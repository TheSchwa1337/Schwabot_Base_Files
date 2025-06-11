"""
Basket Swapper with SP Core Integration
Implements sophisticated basket swapping system with SP Core integration.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging
from pathlib import Path
from core.config import ConfigLoader, ConfigError

from core.basket_swap_logic import BasketSwapLogic
from .cooldown_manager import CooldownManager
from .profit_protection import ProfitProtection
from .time_entropy import TimeEntropyEdgeCase
from .bitmap_engine import BitmapEngine
from .profit_tensor import ProfitTensorStore

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
        """Initialize the basket swapper.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Initialize components
        self.cooldown_manager = CooldownManager()
        self.profit_protection = ProfitProtection()
        self.swap_criteria = SwapCriteria()
        self.time_entropy = TimeEntropyEdgeCase()
        self.bitmap_engine = BitmapEngine()
        self.profit_tensor = ProfitTensorStore()
        
        # Initialize configuration
        self.config_loader = ConfigLoader()
        try:
            if config_path:
                self.config = self.config_loader.load_yaml(config_path)
            else:
                self.config = self.config_loader.load_yaml("basket_swapper_config.yaml")
        except ConfigError as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            self.config = self.config_loader.load_yaml("defaults.yaml")
            
        # Initialize swap logic with config
        self.basket_swap_logic = BasketSwapLogic(config_path)
        
        logger.info("BasketSwapper initialized with configuration")
        
    def process_swap_request(self, request: Dict[str, Any]) -> bool:
        """Process a basket swap request.
        
        Args:
            request: Dictionary containing swap request data
            
        Returns:
            bool: True if swap was successful
        """
        try:
            # Validate request
            if not self._validate_request(request):
                logger.warning(f"Invalid swap request received: {request}")
                return False
                
            # Check cooldown
            if self.cooldown_manager.is_in_cooldown(request.get("basket_id")):
                logger.info(f"Basket {request.get('basket_id')} is in cooldown")
                return False
                
            # Check profit protection
            if not self.profit_protection.validate_swap(request):
                logger.info("Swap rejected by profit protection")
                return False
                
            # Process swap signal
            swap_action = self.basket_swap_logic.process_swap_signal(request)
            if not swap_action:
                return False
                
            # Update cooldown
            self.cooldown_manager.record_swap(request.get("basket_id"))
            
            logger.info(f"Successfully processed swap request: {request.get('basket_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing swap request: {e}")
            return False
            
    def _validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate a swap request.
        
        Args:
            request: Dictionary containing swap request data
            
        Returns:
            bool: True if request is valid
        """
        required_fields = self.config.get("required_fields", ["basket_id", "assets"])
        return all(field in request for field in required_fields)
        
    def get_swap_history(self) -> List[Dict[str, Any]]:
        """Get the swap history.
        
        Returns:
            List[Dict[str, Any]]: List of swap actions
        """
        return self.basket_swap_logic.get_swap_history()
        
    def clear_history(self) -> None:
        """Clear the swap history."""
        self.basket_swap_logic.clear_history()
        logger.info("Swap history cleared")

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