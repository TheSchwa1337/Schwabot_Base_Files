"""
Swap Controller
=============

Handles phase transitions and executes swap actions based on urgency levels.
Manages position sizing and risk control during phase changes.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

from .phase_loader import PhaseConfigLoader

logger = logging.getLogger(__name__)

class SwapController:
    """Controls position swaps and phase transitions"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the swap controller"""
        self.phase_loader = PhaseConfigLoader(config_path)
        self.swap_history: List[Dict[str, Any]] = []
        self._load_swap_config()
        
    def _load_swap_config(self) -> None:
        """Load swap control configuration"""
        config = self.phase_loader.get_phase_region("STABLE")  # Use any phase to get config
        if not config:
            raise ValueError("Failed to load phase configuration")
            
        self.urgency_thresholds = {
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2
        }
        
        self.phase_actions = {
            "UNSTABLE": {
                "high": "liquidate",
                "medium": "reduce_position",
                "low": "monitor"
            },
            "OVERLOADED": {
                "high": "emergency_exit",
                "medium": "liquidate",
                "low": "reduce_position"
            },
            "SMART_MONEY": {
                "high": "scale_up",
                "medium": "maintain",
                "low": "monitor"
            },
            "STABLE": {
                "high": "maintain",
                "medium": "maintain",
                "low": "monitor"
            }
        }
        
    def execute_swap(self, basket_id: str, phase: str, urgency: float) -> Dict[str, Any]:
        """Execute swap action based on phase and urgency"""
        # Validate phase
        if not self.phase_loader.get_phase_region(phase):
            raise ValueError(f"Invalid phase: {phase}")
            
        # Get urgency level
        urgency_level = self._get_urgency_level(urgency)
        
        # Get action for phase and urgency
        action = self.phase_actions[phase][urgency_level]
        
        # Execute action
        result = self._execute_action(basket_id, action, urgency)
        
        # Log swap
        self._log_swap(basket_id, phase, urgency, action, result)
        
        return result
        
    def _get_urgency_level(self, urgency: float) -> str:
        """Get urgency level from value"""
        if urgency >= self.urgency_thresholds["high"]:
            return "high"
        elif urgency >= self.urgency_thresholds["medium"]:
            return "medium"
        else:
            return "low"
            
    def _execute_action(self, basket_id: str, action: str, urgency: float) -> Dict[str, Any]:
        """Execute a swap action"""
        result = {
            "basket_id": basket_id,
            "action": action,
            "urgency": urgency,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "details": {}
        }
        
        try:
            if action == "liquidate":
                result["details"] = self._liquidate_position(basket_id)
            elif action == "reduce_position":
                result["details"] = self._reduce_position(basket_id, urgency)
            elif action == "emergency_exit":
                result["details"] = self._emergency_exit(basket_id)
            elif action == "scale_up":
                result["details"] = self._scale_up_position(basket_id, urgency)
            elif action == "maintain":
                result["details"] = self._maintain_position(basket_id)
            elif action == "monitor":
                result["details"] = self._monitor_position(basket_id)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error executing {action} for {basket_id}: {e}")
            result["success"] = False
            result["error"] = str(e)
            
        return result
        
    def _liquidate_position(self, basket_id: str) -> Dict[str, Any]:
        """Liquidate entire position"""
        # TODO: Implement actual liquidation logic
        return {
            "action": "liquidate",
            "amount": "100%",
            "reason": "High urgency liquidation"
        }
        
    def _reduce_position(self, basket_id: str, urgency: float) -> Dict[str, Any]:
        """Reduce position size based on urgency"""
        reduction = min(0.5 + urgency, 0.9)  # Reduce 50-90% based on urgency
        return {
            "action": "reduce",
            "amount": f"{reduction:.1%}",
            "reason": "Medium urgency reduction"
        }
        
    def _emergency_exit(self, basket_id: str) -> Dict[str, Any]:
        """Emergency position exit"""
        return {
            "action": "emergency_exit",
            "amount": "100%",
            "reason": "Critical phase transition"
        }
        
    def _scale_up_position(self, basket_id: str, urgency: float) -> Dict[str, Any]:
        """Scale up position based on urgency"""
        scale = min(0.2 + urgency, 0.5)  # Scale up 20-50% based on urgency
        return {
            "action": "scale_up",
            "amount": f"{scale:.1%}",
            "reason": "High confidence scaling"
        }
        
    def _maintain_position(self, basket_id: str) -> Dict[str, Any]:
        """Maintain current position"""
        return {
            "action": "maintain",
            "amount": "0%",
            "reason": "Stable phase maintenance"
        }
        
    def _monitor_position(self, basket_id: str) -> Dict[str, Any]:
        """Monitor position without changes"""
        return {
            "action": "monitor",
            "amount": "0%",
            "reason": "Low urgency monitoring"
        }
        
    def _log_swap(self, basket_id: str, phase: str, urgency: float, action: str, result: Dict[str, Any]) -> None:
        """Log swap event"""
        swap_event = {
            "basket_id": basket_id,
            "phase": phase,
            "urgency": urgency,
            "action": action,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.swap_history.append(swap_event)
        
        # Save to file
        self._save_swap_history()
        
    def _save_swap_history(self) -> None:
        """Save swap history to file"""
        try:
            history_path = Path("logs/swap_events.json")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_path, 'w') as f:
                json.dump(self.swap_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving swap history: {e}")
            
    def get_swap_history(self) -> List[Dict[str, Any]]:
        """Get swap history"""
        return self.swap_history.copy()
        
    def get_phase_statistics(self) -> Dict[str, Any]:
        """Get statistics about phase transitions and swaps"""
        stats = {
            "total_swaps": len(self.swap_history),
            "phase_counts": {},
            "action_counts": {},
            "success_rate": 0.0,
            "average_urgency": 0.0
        }
        
        if not self.swap_history:
            return stats
            
        # Count phases and actions
        for swap in self.swap_history:
            phase = swap["phase"]
            action = swap["action"]
            result = swap["result"]
            
            stats["phase_counts"][phase] = stats["phase_counts"].get(phase, 0) + 1
            stats["action_counts"][action] = stats["action_counts"].get(action, 0) + 1
            
            if result["success"]:
                stats["success_rate"] += 1
                
            stats["average_urgency"] += swap["urgency"]
            
        # Calculate averages
        stats["success_rate"] /= len(self.swap_history)
        stats["average_urgency"] /= len(self.swap_history)
        
        return stats 