"""
Basket Swap Logic with SP Core Integration
Implements core swap logic for basket trading operations with signal sanitization
and override handling.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging
from pathlib import Path
from core.config import ConfigLoader, ConfigError

logger = logging.getLogger(__name__)

@dataclass
class SwapAction:
    """Represents a swap action with metadata"""
    trigger_type: str
    asset_adjustments: Dict[str, float]
    fallback: Optional[str] = None
    timestamp: datetime = datetime.now()
    signal_strength: float = 0.0
    override_triggered: bool = False

class BasketSwapLogic:
    """Manages basket swap operations with SP Core integration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the basket swap logic.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.history: List[SwapAction] = []
        self.last_swap_time: Dict[str, datetime] = {}
        
        # Initialize configuration
        self.config_loader = ConfigLoader()
        try:
            if config_path:
                self.config = self.config_loader.load_yaml(config_path)
            else:
                self.config = self.config_loader.load_yaml("basket_swap_config.yaml")
        except ConfigError as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            self.config = self.config_loader.load_yaml("defaults.yaml")
            
        logger.info("BasketSwapLogic initialized with configuration")
        
    def process_swap_signal(self, signal: Dict[str, Any]) -> Optional[SwapAction]:
        """Process a swap signal and generate appropriate action.
        
        Args:
            signal: Dictionary containing swap signal data
            
        Returns:
            Optional[SwapAction]: Generated swap action if signal is valid
        """
        try:
            if not self._validate_signal(signal):
                logger.warning(f"Invalid swap signal received: {signal}")
                return None
                
            action = self._create_swap_action(signal)
            self.history.append(action)
            self.last_swap_time[action.trigger_type] = action.timestamp
            
            logger.info(f"Processed swap signal: {action.trigger_type}")
            return action
            
        except Exception as e:
            logger.error(f"Error processing swap signal: {e}")
            return None
            
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate a swap signal.
        
        Args:
            signal: Dictionary containing swap signal data
            
        Returns:
            bool: True if signal is valid
        """
        required_fields = self.config.get("required_fields", ["trigger_type", "asset_adjustments"])
        return all(field in signal for field in required_fields)
        
    def _create_swap_action(self, signal: Dict[str, Any]) -> SwapAction:
        """Create a swap action from a validated signal.
        
        Args:
            signal: Dictionary containing swap signal data
            
        Returns:
            SwapAction: Created swap action
        """
        return SwapAction(
            trigger_type=signal["trigger_type"],
            asset_adjustments=signal["asset_adjustments"],
            fallback=signal.get("fallback"),
            signal_strength=signal.get("signal_strength", 0.0),
            override_triggered=signal.get("override_triggered", False)
        )
        
    def get_swap_history(self) -> List[SwapAction]:
        """Get the swap action history.
        
        Returns:
            List[SwapAction]: List of swap actions
        """
        return self.history
        
    def clear_history(self) -> None:
        """Clear the swap action history."""
        self.history.clear()
        self.last_swap_time.clear()
        logger.info("Swap history cleared") 