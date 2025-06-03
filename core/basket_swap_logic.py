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
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        self.history: List[SwapAction] = []
        self.last_swap_time: Dict[str, datetime] = {}
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "basket_swap_config.yaml"
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)['basket_swap']
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return {}
            
    def _sanitize_score(self, score: Optional[float], fallback: float = 0.0) -> float:
        """Sanitize a score value to prevent NaN/inf leakage"""
        if score is None or not isinstance(score, (float, int)):
            return fallback
        if np.isnan(score) or np.isinf(score):
            return fallback
        return float(score)
        
    def _calculate_signal_strength(self, trust_score: float, entropy_score: float, 
                                 phase_depth: float) -> float:
        """Calculate signal strength with sanitization"""
        try:
            # Sanitize inputs
            trust_score = self._sanitize_score(trust_score)
            entropy_score = self._sanitize_score(entropy_score)
            phase_depth = self._sanitize_score(phase_depth)
            
            # Calculate entropy component safely
            entropy_component = 1.0 / entropy_score if entropy_score > 0 else 0.0
            
            # Calculate weighted signal strength
            weights = self.config.get('signal_weights', {
                'trust_score': 0.4,
                'entropy_component': 0.3,
                'phase_depth': 0.3
            })
            
            signal_strength = (
                trust_score * weights['trust_score'] +
                entropy_component * weights['entropy_component'] +
                phase_depth * weights['phase_depth']
            )
            
            return self._sanitize_score(signal_strength)
            
        except Exception as e:
            logger.warning(f"Signal strength calculation failed: {str(e)}")
            return 0.0
            
    def _should_override(self, basket_id: str, pkt: Dict[str, Any], 
                        rings: Dict[str, Any]) -> bool:
        """Check if swap should be overridden"""
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
        
    def apply_swap(self, trigger_type: str, asset_adjustments: Dict[str, float], 
                  fallback: Optional[str] = None, pkt: Optional[Dict[str, Any]] = None,
                  rings: Optional[Dict[str, Any]] = None) -> bool:
        """
        Apply a swap operation with enhanced signal handling.
        
        Args:
            trigger_type: Type of trigger that initiated the swap
            asset_adjustments: Dictionary of asset symbols to position size adjustments
            fallback: Optional fallback strategy if primary swap fails
            pkt: Optional packet data for override checks
            rings: Optional rings data for override checks
            
        Returns:
            bool: True if swap was successful
        """
        try:
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(
                trust_score=asset_adjustments.get('trust_score', 0.0),
                entropy_score=asset_adjustments.get('entropy_score', 0.0),
                phase_depth=asset_adjustments.get('phase_depth', 0.0)
            )
            
            # Check for override
            override_triggered = False
            if pkt and rings:
                override_triggered = self._should_override(
                    basket_id=asset_adjustments.get('basket_id', 'unknown'),
                    pkt=pkt,
                    rings=rings
                )
            
            # Create swap action with enhanced metadata
            swap_action = SwapAction(
                trigger_type=trigger_type,
                asset_adjustments=asset_adjustments,
                fallback=fallback,
                signal_strength=signal_strength,
                override_triggered=override_triggered
            )
            
            # Record in history
            self.history.append(swap_action)
            
            # Update last swap time for affected assets
            for asset in asset_adjustments.keys():
                self.last_swap_time[asset] = datetime.now()
                
            # Log swap details
            logger.info(
                f"Swap applied: type={trigger_type}, "
                f"signal_strength={signal_strength:.3f}, "
                f"override={override_triggered}"
            )
                
            return True
            
        except Exception as e:
            logger.error(f"Swap application failed: {str(e)}")
            return False
            
    def get_swap_history(self, asset: Optional[str] = None) -> List[SwapAction]:
        """Get swap history, optionally filtered by asset"""
        if asset is None:
            return self.history
            
        return [
            action for action in self.history
            if asset in action.asset_adjustments
        ]
        
    def get_last_swap_time(self, asset: str) -> Optional[datetime]:
        """Get the last swap time for an asset"""
        return self.last_swap_time.get(asset)
        
    def clear_history(self):
        """Clear swap history"""
        self.history.clear()
        self.last_swap_time.clear() 