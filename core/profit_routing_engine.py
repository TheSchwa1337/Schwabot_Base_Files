# -*- coding: utf-8 -*-
"""
Profit Routing Engine - Schwabot UROS v1.0
=========================================

Core profit routing engine that integrates with the symbolic profit router
to implement 2-bit phase mapping and profit tier navigation logic.

This module provides the main interface for profit tier routing decisions
and integrates with the dualistic system architecture.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from dual_unicore_handler import DualUnicoreHandler
from .symbolic_profit_router import (
    SymbolicProfitRouter, ProfitTier, ProfitVaultAction, 
    ProfitTrigger, TriggerType, BitPhase, route_profit_phase,
    hash_to_strategy, fold_hash_to_2bit
)

# Initialize Unicode handler
unicore = DualUnicoreHandler()

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Routing decisions for profit tier navigation."""
    EXECUTE = "execute"
    HOLD = "hold"
    SKIP = "skip"
    OVERRIDE = "override"
    ESCALATE = "escalate"
    COLLAPSE = "collapse"


class VaultLevel(Enum):
    """Vault levels for profit tier execution."""
    SHORT = "short"
    MID = "mid"
    LONG = "long"
    OVERRIDE = "override"


@dataclass
class ProfitRoutingConfig:
    """Configuration for profit routing engine."""
    enable_2bit_mapping: bool = True
    enable_hash_triggers: bool = True
    enable_recursive_learning: bool = True
    confidence_threshold: float = 0.75
    max_allocation: float = 0.7
    min_expected_return: float = 0.042
    enable_temporal_correction: bool = True
    enable_failure_recovery: bool = True
    log_level: str = "INFO"


@dataclass
class ProfitRoutingResult:
    """Result of profit routing decision."""
    decision: RoutingDecision
    tier: ProfitTier
    allocation: float
    confidence: float
    trigger_type: TriggerType
    execution_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProfitRoutingEngine:
    """
    Main profit routing engine that coordinates profit tier navigation.
    
    Integrates with the symbolic profit router to implement the dualistic
    2-bit mapping system for profit tier navigation.
    """

    def __init__(self, config: Optional[ProfitRoutingConfig] = None):
        """Initialize the profit routing engine."""
        self.config = config or ProfitRoutingConfig()
        self.symbolic_router = SymbolicProfitRouter()
        self.routing_history: List[ProfitRoutingResult] = []
        self.vault_states: Dict[str, Dict[str, Any]] = {}
        self.failure_recovery_state: Dict[str, Any] = {}
        
        # Initialize vault states
        self._initialize_vault_states()

        logger.info("Profit Routing Engine initialized")

    def _initialize_vault_states(self) -> None:
        """Initialize vault states for all tiers."""
        for tier in ProfitTier:
            self.vault_states[tier.value] = {
                "active": False,
                "current_allocation": 0.0,
                "last_execution": None,
                "success_count": 0,
                "failure_count": 0,
                "total_return": 0.0
            }

    def route_profit(self, payload: Dict[str, Any]) -> ProfitRoutingResult:
        """
        Main profit routing function that processes a payload and returns a routing decision.
        
        Args:
            payload: Dictionary containing routing parameters
            
        Returns:
            ProfitRoutingResult with routing decision
        """
        try:
            # Extract parameters from payload
            phase = payload.get("phase", "2-bit")
            flip_bias = payload.get("flip_bias", "neutral")
            asset = payload.get("asset", "BTC")
            expected_return = payload.get("expected_return", 0.0)
            hash_input = payload.get("hash_input", "")
            
            # Generate hash bits if not provided
            if "hash_bits" not in payload:
                if hash_input:
                    hash_bits = fold_hash_to_2bit(hash_input)
                else:
                    # Generate default hash from parameters
                    default_hash = f"vault_trigger::{asset}::{flip_bias}::{expected_return}"
                    hash_bits = fold_hash_to_2bit(default_hash)
            else:
                hash_bits = payload["hash_bits"]
            
            # Apply temporal correction if enabled
            if self.config.enable_temporal_correction:
                if not self._is_temporally_correct(asset):
                    return self._create_hold_result("Temporal correction applied")
            
            # Check failure recovery state
            if self.config.enable_failure_recovery:
                if self._is_in_recovery_mode(asset):
                    return self._handle_recovery_routing(asset, payload)
            
            # Route through symbolic router
            vault_action = self.symbolic_router.route_profit_phase(
                phase, flip_bias, hash_bits, asset, expected_return
            )
            
            # Convert to routing result
            routing_result = self._convert_vault_action_to_result(vault_action)
            
            # Log the routing decision
            self._log_routing_decision(routing_result)
            
            # Update vault states
            self._update_vault_state(routing_result)
            
            return routing_result
            
        except Exception as e:
            logger.error(f"Error in profit routing: {e}")
            return self._create_error_result(str(e))

    def activate_profit_vault(self, level: str, trigger: str, 
                            asset: str = "BTC", **kwargs) -> ProfitRoutingResult:
        """
        Activate a profit vault with specified level and trigger.
        
        Args:
            level: Vault level (short/mid/long/override)
            trigger: Trigger type (emoji_hash_match, etc.)
            asset: Asset symbol
            **kwargs: Additional parameters
            
        Returns:
            ProfitRoutingResult with activation decision
        """
        try:
            # Map level to tier
            tier_mapping = {
                "short": ProfitTier.SHORT,
                "mid": ProfitTier.MID,
                "long": ProfitTier.LONG,
                "override": ProfitTier.OVERRIDE
            }
            
            tier = tier_mapping.get(level, ProfitTier.MID)
            
            # Create payload for routing
            payload = {
                "phase": "2-bit",
                "flip_bias": "up" if trigger == "emoji_hash_match" else "neutral",
                "asset": asset,
                "expected_return": kwargs.get("expected_return", 0.11),
                "trigger_type": trigger,
                **kwargs
            }
            
            # Route the profit
            result = self.route_profit(payload)
            
            # Activate the vault
            self._activate_vault(tier, result)
            
            logger.info(f"Profit vault activated: {level} tier for {asset} with {trigger} trigger")
            
            return result
            
        except Exception as e:
            logger.error(f"Error activating profit vault: {e}")
            return self._create_error_result(str(e))

    def _is_temporally_correct(self, asset: str) -> bool:
        """
        Check if the current time is appropriate for trading the asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            True if temporally correct, False otherwise
        """
        try:
            # Simple temporal check - could be enhanced with market hours, etc.
            current_hour = datetime.now().hour
            
            # Avoid trading during low liquidity hours (2-6 AM UTC)
            if 2 <= current_hour <= 6:
                logger.warning(f"Temporal correction: Low liquidity hours for {asset}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in temporal correction: {e}")
            return True  # Default to allow trading

    def _is_in_recovery_mode(self, asset: str) -> bool:
        """
        Check if the asset is in failure recovery mode.
        
        Args:
            asset: Asset symbol
            
        Returns:
            True if in recovery mode, False otherwise
        """
        return asset in self.failure_recovery_state

    def _handle_recovery_routing(self, asset: str, payload: Dict[str, Any]) -> ProfitRoutingResult:
        """
        Handle routing when asset is in recovery mode.
        
        Args:
            asset: Asset symbol
            payload: Original routing payload
            
        Returns:
            ProfitRoutingResult with recovery decision
        """
        recovery_info = self.failure_recovery_state[asset]
        
        # Check if recovery period has expired
        recovery_time = recovery_info.get("recovery_time")
        if recovery_time and datetime.now() > recovery_time:
            # Remove from recovery state
            del self.failure_recovery_state[asset]
            logger.info(f"Recovery period expired for {asset}")
            return self.route_profit(payload)
        
        # Apply conservative routing during recovery
        payload["expected_return"] *= 0.5  # Reduce expected return
        payload["confidence"] = min(payload.get("confidence", 0.5), 0.6)  # Cap confidence
        
        result = self.route_profit(payload)
        result.metadata["recovery_mode"] = True
        
        return result

    def _convert_vault_action_to_result(self, vault_action: ProfitVaultAction) -> ProfitRoutingResult:
        """
        Convert vault action to routing result.
        
        Args:
            vault_action: Vault action from symbolic router
            
        Returns:
            ProfitRoutingResult
        """
        # Map action to decision
        action_to_decision = {
            "execute": RoutingDecision.EXECUTE,
            "monitor": RoutingDecision.HOLD,
            "skip": RoutingDecision.SKIP,
            "override": RoutingDecision.OVERRIDE
        }
        
        decision = action_to_decision.get(vault_action.action, RoutingDecision.HOLD)
        
        return ProfitRoutingResult(
            decision=decision,
            tier=vault_action.tier,
            allocation=vault_action.allocation,
            confidence=vault_action.trigger.confidence,
            trigger_type=vault_action.trigger.trigger_type,
            execution_time=vault_action.execution_time,
            metadata=vault_action.metadata
        )

    def _create_hold_result(self, reason: str) -> ProfitRoutingResult:
        """Create a hold result with specified reason."""
        return ProfitRoutingResult(
            decision=RoutingDecision.HOLD,
            tier=ProfitTier.SHORT,
            allocation=0.0,
            confidence=0.5,
            trigger_type=TriggerType.EMOJI_HASH_MATCH,
            execution_time=datetime.now(),
            metadata={"hold_reason": reason}
        )

    def _create_error_result(self, error: str) -> ProfitRoutingResult:
        """Create an error result with specified error."""
        return ProfitRoutingResult(
            decision=RoutingDecision.SKIP,
            tier=ProfitTier.SHORT,
            allocation=0.0,
            confidence=0.0,
            trigger_type=TriggerType.EMOJI_HASH_MATCH,
            execution_time=datetime.now(),
            metadata={"error": error}
        )

    def _activate_vault(self, tier: ProfitTier, result: ProfitRoutingResult) -> None:
        """
        Activate a vault for the specified tier.
        
        Args:
            tier: Profit tier to activate
            result: Routing result
        """
        vault_state = self.vault_states[tier.value]
        vault_state["active"] = True
        vault_state["current_allocation"] = result.allocation
        vault_state["last_execution"] = datetime.now()
        
        logger.info(f"Vault activated: {tier.value} tier with {result.allocation:.2%} allocation")

    def _log_routing_decision(self, result: ProfitRoutingResult) -> None:
        """
        Log routing decision for audit trail.
        
        Args:
            result: Routing result to log
        """
        self.routing_history.append(result)
        
        logger.info(f"Routing decision: {result.decision.value} for {result.tier.value} tier, "
                   f"confidence: {result.confidence:.2f}, allocation: {result.allocation:.2%}")

    def _update_vault_state(self, result: ProfitRoutingResult) -> None:
        """
        Update vault state based on routing result.
        
        Args:
            result: Routing result
        """
        vault_state = self.vault_states[result.tier.value]
        
        if result.decision == RoutingDecision.EXECUTE:
            vault_state["success_count"] += 1
        elif result.decision in [RoutingDecision.SKIP, RoutingDecision.COLLAPSE]:
            vault_state["failure_count"] += 1

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        if not self.routing_history:
            return {"total_decisions": 0, "success_rate": 0.0}
        
        total_decisions = len(self.routing_history)
        successful_decisions = len([r for r in self.routing_history 
                                  if r.decision == RoutingDecision.EXECUTE])
        success_rate = successful_decisions / total_decisions if total_decisions > 0 else 0.0
        
        # Decision distribution
        decision_distribution = {}
        for result in self.routing_history:
            decision = result.decision.value
            decision_distribution[decision] = decision_distribution.get(decision, 0) + 1
        
        # Tier distribution
        tier_distribution = {}
        for result in self.routing_history:
            tier = result.tier.value
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "success_rate": success_rate,
            "decision_distribution": decision_distribution,
            "tier_distribution": tier_distribution,
            "vault_states": self.vault_states,
            "failure_recovery_count": len(self.failure_recovery_state),
            "symbolic_router_stats": self.symbolic_router.get_routing_stats()
        }

    def set_failure_recovery(self, asset: str, duration_hours: int = 24) -> None:
        """
        Set an asset into failure recovery mode.
        
        Args:
            asset: Asset symbol
            duration_hours: Recovery duration in hours
        """
        recovery_time = datetime.now() + timedelta(hours=duration_hours)
        self.failure_recovery_state[asset] = {
            "recovery_time": recovery_time,
            "duration_hours": duration_hours
        }
        
        logger.warning(f"Asset {asset} set to failure recovery mode for {duration_hours} hours")

    def clear_failure_recovery(self, asset: str) -> None:
        """
        Clear failure recovery mode for an asset.
        
        Args:
            asset: Asset symbol
        """
        if asset in self.failure_recovery_state:
            del self.failure_recovery_state[asset]
            logger.info(f"Failure recovery cleared for {asset}")

    def export_routing_log(self, filepath: str) -> None:
        """Export routing log to file."""
        self.symbolic_router.export_log(filepath)

    def clear_routing_log(self) -> None:
        """Clear routing log."""
        self.routing_history.clear()
        self.symbolic_router.clear_log()
        logger.info("Routing log cleared")


# Convenience functions for external use
def route_profit(payload: Dict[str, Any]) -> ProfitRoutingResult:
    """Convenience function to route profit."""
    engine = ProfitRoutingEngine()
    return engine.route_profit(payload)


def activate_profit_vault(level: str, trigger: str, asset: str = "BTC", **kwargs) -> ProfitRoutingResult:
    """Convenience function to activate profit vault."""
    engine = ProfitRoutingEngine()
    return engine.activate_profit_vault(level, trigger, asset, **kwargs)


if __name__ == "__main__":
    # Test the profit routing engine
    engine = ProfitRoutingEngine()

    # Test basic routing
    payload = {
        "phase": "2-bit",
        "flip_bias": "up",
        "asset": "BTC",
        "expected_return": 0.15,
        "hash_input": "vault_trigger::BTC::mid::24hr"
    }
    
    result = engine.route_profit(payload)
    print(f"Routing result: {result.decision.value} for {result.tier.value} tier")
    
    # Test vault activation
    vault_result = engine.activate_profit_vault("mid", "emoji_hash_match", "BTC", expected_return=0.12)
    print(f"Vault activation: {vault_result.decision.value} with {vault_result.allocation:.2%} allocation")
    
    # Get stats
    stats = engine.get_routing_stats()
    print(f"Engine stats: {stats}")


