# -*- coding: utf-8 -*-
"""
Symbolic Profit Router - Schwabot UROS v1.0
==========================================

Implements dualistic 2-bit mapping system for profit tier navigation.
Combines symbolic execution paths with hash-driven triggers for recursive
profit tier sequencing via strategic fallback navigation.

Core Architecture:
- Side A: Symbolic Execution Path (human-readable logic)
- Side B: Raw Bytecode Reflection (via hash, machine-decoded via 2-bit segmentation)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from dual_unicore_handler import DualUnicoreHandler

# Initialize Unicode handler
unicore = DualUnicoreHandler()

logger = logging.getLogger(__name__)


class BitPhase(Enum):
    """2-bit phase mapping for profit tier navigation."""
    IDLE = "00"      # Idle or default wait-state
    SOFT_TRIGGER = "01"  # Phase prediction or non-financial shift
    HARD_ENTRY = "10"    # Triggers a vault/strategy execution
    OVERRIDE = "11"      # Recursively modifies strategy stack


class ProfitTier(Enum):
    """Profit tier levels for strategy execution."""
    SHORT = "short"
    MID = "mid"
    LONG = "long"
    OVERRIDE = "override"


class FlipBias(Enum):
    """Flip bias enumeration for directional strategy preference."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


class SymbolicState(Enum):
    """Symbolic state enumeration for strategy execution state tracking."""
    GENESIS = "genesis"
    ACTIVE = "active"
    DORMANT = "dormant"
    TRIGGERED = "triggered"
    EXECUTED = "executed"
    COMPLETED = "completed"
    FAILED = "failed"
    RESET = "reset"


class TriggerType(Enum):
    """Types of triggers that can activate profit tiers."""
    EMOJI_HASH_MATCH = "emoji_hash_match"
    MOMENTUM_SHIFT = "momentum_shift"
    VOLUME_SPIKE = "volume_spike"
    PRICE_BREAKOUT = "price_breakout"
    SYMBOLIC_OVERRIDE = "symbolic_override"


@dataclass
class ProfitTrigger:
    """Container for profit trigger data."""
    phase: str
    flip_bias: str
    hash_bits: str
    asset: str
    expected_return: float
    confidence: float
    timestamp: datetime
    trigger_type: TriggerType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfitVaultAction:
    """Container for profit vault action data."""
    tier: ProfitTier
    action: str
    allocation: float
    trigger: ProfitTrigger
    execution_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SymbolicProfitRouter:
    """
    Dualistic profit router that maps symbolic triggers to 2-bit phase codes.
    
    Implements the recursive profit tier navigation system with hash-driven
    triggers and symbolic execution paths.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the symbolic profit router."""
        self.config = config or self._get_default_config()
        self.log: List[Dict[str, Any]] = []
        self.vault_history: List[ProfitVaultAction] = []
        self.hash_registry: Dict[str, Dict[str, Any]] = {}
        self.tier_weights = {
            ProfitTier.SHORT: 0.4,
            ProfitTier.MID: 0.6,
            ProfitTier.LONG: 0.85,
            ProfitTier.OVERRIDE: 1.0
        }
        
        # Initialize bit phase mapping
        self.bit_phase_map = {
            BitPhase.IDLE: {"action": "skip", "tier": None},
            BitPhase.SOFT_TRIGGER: {"action": "monitor", "tier": ProfitTier.SHORT},
            BitPhase.HARD_ENTRY: {"action": "execute", "tier": ProfitTier.MID},
            BitPhase.OVERRIDE: {"action": "override", "tier": ProfitTier.OVERRIDE}
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the router."""
        return {
            "hash_digest_size": 64,
            "confidence_threshold": 0.75,
            "max_allocation": 0.7,
            "min_expected_return": 0.042,
            "enable_recursive_learning": True,
            "log_level": "INFO"
        }

    def fold_hash_to_2bit(self, hash_string: str) -> str:
        """
        Fold a hash string into a 2-bit sequence.
        
        Args:
            hash_string: The hash string to fold
            
        Returns:
            2-bit sequence as string
        """
        try:
            # Convert hash to binary
            hash_int = int(hash_string[:16], 16)  # Use first 16 chars for consistency
            binary = format(hash_int, '064b')  # 64-bit binary
            
            # Fold into 2-bit segments and take the most common pattern
            segments = [binary[i:i+2] for i in range(0, len(binary), 2)]
            
            # Count occurrences and return most common
            from collections import Counter
            counter = Counter(segments)
            most_common = counter.most_common(1)[0][0]
            
            return most_common
            
        except Exception as e:
            logger.error(f"Error folding hash to 2-bit: {e}")
            return BitPhase.IDLE.value

    def hash_to_strategy(self, hash_input: str) -> Dict[str, Any]:
        """
        Decode hash/UTF to a strategy dictionary.
        
        Args:
            hash_input: Hash string or UTF trigger
            
        Returns:
            Strategy dictionary with decoded parameters
        """
        try:
            # Generate hash if input is not already a hash
            if len(hash_input) < 64:
                hash_string = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
            else:
                hash_string = hash_input
            
            # Fold to 2-bit
            bit_sequence = self.fold_hash_to_2bit(hash_string)
            
            # Parse hash input for symbolic data
            parts = hash_input.split("::")
            asset = parts[1] if len(parts) > 1 else "BTC"
            tier = parts[2] if len(parts) > 2 else "mid"
            horizon = parts[3] if len(parts) > 3 else "24hr"
            
            # Calculate confidence based on hash entropy
            confidence = self._calculate_hash_confidence(hash_string)
            
            return {
                "asset": asset,
                "tier": tier,
                "expected_horizon": horizon,
                "confidence": confidence,
                "bit_sequence": bit_sequence,
                "hash": hash_string
            }
            
        except Exception as e:
            logger.error(f"Error decoding hash to strategy: {e}")
            return {
                "asset": "BTC",
                "tier": "mid",
                "expected_horizon": "24hr",
                "confidence": 0.5,
                "bit_sequence": BitPhase.IDLE.value,
                "hash": hash_input
            }

    def _calculate_hash_confidence(self, hash_string: str) -> float:
        """
        Calculate confidence score based on hash entropy.
        
        Args:
            hash_string: The hash string to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Convert hash to binary and calculate entropy
            hash_int = int(hash_string[:16], 16)
            binary = format(hash_int, '064b')
            
            # Count 1s and 0s
            ones = binary.count('1')
            zeros = binary.count('0')
            total = len(binary)
            
            # Calculate entropy-based confidence
            p1 = ones / total
            p0 = zeros / total
            
            if p1 == 0 or p0 == 0:
                return 0.5  # Neutral confidence
            
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
            max_entropy = 1.0
            
            # Normalize to confidence (higher entropy = higher confidence)
            confidence = entropy / max_entropy
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating hash confidence: {e}")
            return 0.5

    def route_profit_phase(self, phase: str, flip_bias: str, hash_bits: str, 
                          asset: str = "BTC", expected_return: float = 0.0) -> ProfitVaultAction:
        """
        Route profit phase based on 2-bit mapping and symbolic triggers.
        
        Args:
            phase: The phase (e.g., "2-bit")
            flip_bias: The flip bias ("up", "down", "neutral")
            hash_bits: 2-bit hash sequence
            asset: Asset symbol
            expected_return: Expected return percentage
            
        Returns:
            ProfitVaultAction with routing decision
        """
        try:
            # Create trigger
            trigger = ProfitTrigger(
                phase=phase,
                flip_bias=flip_bias,
                hash_bits=hash_bits,
                asset=asset,
                expected_return=expected_return,
                confidence=self._calculate_trigger_confidence(phase, flip_bias, hash_bits),
                timestamp=datetime.now(),
                trigger_type=self._determine_trigger_type(phase, flip_bias, hash_bits)
            )
            
            # Map to bit phase
            bit_phase = self._map_hash_bits_to_phase(hash_bits)
            phase_info = self.bit_phase_map[bit_phase]
            
            # Determine tier based on phase and expected return
            tier = self._determine_tier(phase_info["tier"], expected_return, trigger.confidence)
            
            # Calculate allocation
            allocation = self._calculate_allocation(tier, expected_return, trigger.confidence)
            
            # Create vault action
            vault_action = ProfitVaultAction(
                tier=tier,
                action=phase_info["action"],
                allocation=allocation,
                trigger=trigger,
                execution_time=datetime.now(),
                metadata={
                    "bit_phase": bit_phase.value,
                    "phase_info": phase_info,
                    "hash_bits": hash_bits
                }
            )
            
            # Log the action
            self._log_vault_action(vault_action)
            
            # Update hash registry for recursive learning
            if self.config["enable_recursive_learning"]:
                self._update_hash_registry(trigger, vault_action)
            
            return vault_action
            
        except Exception as e:
            logger.error(f"Error routing profit phase: {e}")
            # Return safe default action
            return ProfitVaultAction(
                tier=ProfitTier.SHORT,
                action="skip",
                allocation=0.0,
                trigger=trigger,
                execution_time=datetime.now(),
                metadata={"error": str(e)}
            )

    def _map_hash_bits_to_phase(self, hash_bits: str) -> BitPhase:
        """Map hash bits to bit phase."""
        try:
            return BitPhase(hash_bits)
        except ValueError:
            logger.warning(f"Invalid hash bits: {hash_bits}, defaulting to IDLE")
            return BitPhase.IDLE

    def _determine_trigger_type(self, phase: str, flip_bias: str, hash_bits: str) -> TriggerType:
        """Determine trigger type based on input parameters."""
        if phase == "2-bit" and flip_bias == "up":
            if hash_bits == "10":
                return TriggerType.EMOJI_HASH_MATCH
            elif hash_bits == "11":
                return TriggerType.SYMBOLIC_OVERRIDE
        elif flip_bias == "up":
            return TriggerType.MOMENTUM_SHIFT
        elif flip_bias == "down":
            return TriggerType.PRICE_BREAKOUT
        
        return TriggerType.EMOJI_HASH_MATCH

    def _calculate_trigger_confidence(self, phase: str, flip_bias: str, hash_bits: str) -> float:
        """Calculate confidence for the trigger."""
        base_confidence = 0.5
        
        # Adjust based on phase
        if phase == "2-bit":
            base_confidence += 0.2
        
        # Adjust based on flip bias
        if flip_bias in ["up", "down"]:
            base_confidence += 0.1
        
        # Adjust based on hash bits
        if hash_bits in ["10", "11"]:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)

    def _determine_tier(self, suggested_tier: Optional[ProfitTier], 
                       expected_return: float, confidence: float) -> ProfitTier:
        """Determine the appropriate tier based on return and confidence."""
        if suggested_tier is None:
            suggested_tier = ProfitTier.SHORT
        
        # Override based on expected return
        if expected_return >= 0.25 and confidence >= 0.8:
            return ProfitTier.LONG
        elif expected_return >= 0.11 and confidence >= 0.6:
            return ProfitTier.MID
        elif expected_return >= 0.042:
            return ProfitTier.SHORT
        else:
            return ProfitTier.SHORT

    def _calculate_allocation(self, tier: ProfitTier, expected_return: float, 
                            confidence: float) -> float:
        """Calculate allocation percentage based on tier and confidence."""
        base_allocation = self.tier_weights.get(tier, 0.4)
        
        # Adjust based on expected return and confidence
        adjustment = (expected_return * confidence) / 0.25  # Normalize to 25% return
        
        allocation = base_allocation * adjustment
        
        # Cap at maximum allocation
        return min(allocation, self.config["max_allocation"])

    def _log_vault_action(self, vault_action: ProfitVaultAction) -> None:
        """Log vault action for audit trail."""
        log_entry = {
            "timestamp": vault_action.execution_time.isoformat(),
            "tier": vault_action.tier.value,
            "action": vault_action.action,
            "allocation": vault_action.allocation,
            "asset": vault_action.trigger.asset,
            "expected_return": vault_action.trigger.expected_return,
            "confidence": vault_action.trigger.confidence,
            "trigger_type": vault_action.trigger.trigger_type.value,
            "hash_bits": vault_action.trigger.hash_bits,
            "metadata": vault_action.metadata
        }
        
        self.log.append(log_entry)
        self.vault_history.append(vault_action)
        
        logger.info(f"Vault action logged: {vault_action.tier.value} tier, "
                   f"{vault_action.action} action, {vault_action.allocation:.2%} allocation")

    def _update_hash_registry(self, trigger: ProfitTrigger, vault_action: ProfitVaultAction) -> None:
        """Update hash registry for recursive learning."""
        hash_key = f"{trigger.asset}_{trigger.hash_bits}_{trigger.flip_bias}"
        
        if hash_key not in self.hash_registry:
            self.hash_registry[hash_key] = {
                "count": 0,
                "success_rate": 0.0,
                "avg_return": 0.0,
                "last_used": None,
                "tier_preferences": {}
            }
        
        registry = self.hash_registry[hash_key]
        registry["count"] += 1
        registry["last_used"] = datetime.now().isoformat()
        
        # Update tier preferences
        tier = vault_action.tier.value
        if tier not in registry["tier_preferences"]:
            registry["tier_preferences"][tier] = 0
        registry["tier_preferences"][tier] += 1

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics."""
        if not self.vault_history:
            return {"total_actions": 0, "success_rate": 0.0}
        
        total_actions = len(self.vault_history)
        successful_actions = len([a for a in self.vault_history if a.action == "execute"])
        success_rate = successful_actions / total_actions if total_actions > 0 else 0.0
        
        tier_distribution = {}
        for action in self.vault_history:
            tier = action.tier.value
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        
        return {
            "total_actions": total_actions,
            "success_rate": success_rate,
            "tier_distribution": tier_distribution,
            "hash_registry_size": len(self.hash_registry),
            "log_entries": len(self.log)
        }

    def export_log(self, filepath: str) -> None:
        """Export routing log to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.log, f, indent=2, default=str)
            logger.info(f"Routing log exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting log: {e}")

    def clear_log(self) -> None:
        """Clear the routing log."""
        self.log.clear()
        self.vault_history.clear()
        logger.info("Routing log cleared")


# Convenience functions for external use
def route_profit_phase(phase: str, flip_bias: str, hash_bits: str, 
                      asset: str = "BTC", expected_return: float = 0.0) -> ProfitVaultAction:
    """Convenience function to route profit phase."""
    router = SymbolicProfitRouter()
    return router.route_profit_phase(phase, flip_bias, hash_bits, asset, expected_return)


def hash_to_strategy(hash_input: str) -> Dict[str, Any]:
    """Convenience function to decode hash to strategy."""
    router = SymbolicProfitRouter()
    return router.hash_to_strategy(hash_input)


def fold_hash_to_2bit(hash_string: str) -> str:
    """Convenience function to fold hash to 2-bit."""
    router = SymbolicProfitRouter()
    return router.fold_hash_to_2bit(hash_string)


if __name__ == "__main__":
    # Test the symbolic profit router
    router = SymbolicProfitRouter()
    
    # Test hash folding
    test_hash = hashlib.sha256(b"vault_trigger:BTC:mid:32hr").hexdigest()
    bit_sequence = router.fold_hash_to_2bit(test_hash)
    print(f"Hash: {test_hash[:16]}... -> 2-bit: {bit_sequence}")
    
    # Test strategy decoding
    strategy = router.hash_to_strategy("vault_trigger::BTC::long::32hr")
    print(f"Strategy: {strategy}")
    
    # Test profit phase routing
    vault_action = router.route_profit_phase("2-bit", "up", "10", "BTC", 0.15)
    print(f"Vault Action: {vault_action.tier.value} tier, {vault_action.action} action")
    
    # Get stats
    stats = router.get_routing_stats()
    print(f"Stats: {stats}")
