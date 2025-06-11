"""
Swap Decision Engine
=================

Handles strategy activation based on phase transitions.
Manages swap intents and strategy mapping.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json

from ..phase_engine.phase_map import PhaseMap
from ..phase_engine.sha_mapper import SHAMapper
from ..config.manager import ConfigManager

logger = logging.getLogger(__name__)

class SwapIntent:
    """Represents a swap decision intent"""
    
    def __init__(self, strategy: str, urgency: float, memory_coherence: float,
                 hash_id: str, metrics: Dict[str, float]):
        """Initialize swap intent"""
        self.strategy = strategy
        self.urgency = urgency
        self.memory_coherence = memory_coherence
        self.hash_id = hash_id
        self.metrics = metrics
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert intent to dictionary"""
        return {
            "strategy": self.strategy,
            "urgency": self.urgency,
            "memory_coherence": self.memory_coherence,
            "hash_id": self.hash_id,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwapIntent':
        """Create intent from dictionary"""
        return cls(
            strategy=data["strategy"],
            urgency=data["urgency"],
            memory_coherence=data["memory_coherence"],
            hash_id=data["hash_id"],
            metrics=data["metrics"]
        )

class StrategyMapper:
    """Maps phases to strategies"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize strategy mapper"""
        self.config = ConfigManager(config_path)
        self.strategy_config = self.config.get_config("strategy_config.yaml")
        
    def activate(self, phase: str) -> Dict[str, Any]:
        """Activate strategy for a phase"""
        strategies = self.strategy_config.get("active_strategies", [])
        if not strategies:
            raise ValueError("No active strategies configured")
            
        # Get strategy for phase
        strategy = self._get_strategy_for_phase(phase)
        if not strategy:
            # Use default strategy
            strategy = self.strategy_config.get("default_strategy", {})
            
        return {
            "type": strategy.get("type", "phase_aware"),
            "parameters": strategy.get("parameters", {}),
            "phase": phase
        }
        
    def _get_strategy_for_phase(self, phase: str) -> Optional[Dict[str, Any]]:
        """Get strategy configuration for a phase"""
        strategies = self.strategy_config.get("strategies", {})
        return strategies.get(phase)

class SwapDecisionEngine:
    """Handles swap decisions and strategy activation"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize swap decision engine"""
        self.phase_map = PhaseMap(config_path)
        self.sha_mapper = SHAMapper()
        self.strategy_mapper = StrategyMapper(config_path)
        self.intent_history: List[SwapIntent] = []
        self._load_intent_history()
        
    def _load_intent_history(self) -> None:
        """Load swap intent history"""
        try:
            history_path = Path("logs/swap_intents.json")
            if history_path.exists():
                with open(history_path, 'r') as f:
                    data = json.load(f)
                    self.intent_history = [
                        SwapIntent.from_dict(entry)
                        for entry in data.get("intents", [])
                    ]
                logger.info(f"Loaded {len(self.intent_history)} swap intents")
                
        except Exception as e:
            logger.error(f"Error loading swap intent history: {e}")
            
    def _save_intent_history(self) -> None:
        """Save swap intent history"""
        try:
            history_path = Path("logs/swap_intents.json")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "intents": [intent.to_dict() for intent in self.intent_history],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(history_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving swap intent history: {e}")
            
    def check_and_activate(self, basket_id: str) -> Optional[SwapIntent]:
        """Check conditions and activate strategy if needed"""
        # Get current state
        state = self.phase_map.latest()
        if not state:
            return None
            
        # Get SHA key for basket
        hash_id = self.sha_mapper.get_sha_key(basket_id)
        
        # Check phase conditions
        if state.phase == "SMART_MONEY" and state.urgency > 0.7:
            # Activate strategy
            strategy = self.strategy_mapper.activate(state.phase)
            
            # Create swap intent
            intent = SwapIntent(
                strategy=strategy["type"],
                urgency=state.urgency,
                memory_coherence=state.memory_coherence,
                hash_id=hash_id,
                metrics=state.metrics
            )
            
            # Add to history
            self.intent_history.append(intent)
            self._save_intent_history()
            
            return intent
            
        return None
        
    def get_intent_history(self) -> List[SwapIntent]:
        """Get swap intent history"""
        return self.intent_history.copy()
        
    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get statistics about swap intents"""
        stats = {
            "total_intents": len(self.intent_history),
            "strategy_counts": {},
            "average_urgency": 0.0,
            "average_coherence": 0.0,
            "last_intent": None
        }
        
        if not self.intent_history:
            return stats
            
        # Count strategies
        for intent in self.intent_history:
            stats["strategy_counts"][intent.strategy] = \
                stats["strategy_counts"].get(intent.strategy, 0) + 1
            stats["average_urgency"] += intent.urgency
            stats["average_coherence"] += intent.memory_coherence
            
        # Calculate averages
        stats["average_urgency"] /= len(self.intent_history)
        stats["average_coherence"] /= len(self.intent_history)
        
        # Get last intent
        last = self.intent_history[-1]
        stats["last_intent"] = {
            "strategy": last.strategy,
            "urgency": last.urgency,
            "coherence": last.memory_coherence,
            "timestamp": last.timestamp.isoformat()
        }
        
        return stats 