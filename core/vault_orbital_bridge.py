#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vault-to-Orbital Strategy Bridge for Schwabot
=============================================
Maps vault logic into orbital logic for emergent trade cycles:
• Vault state ↔ Orbital state mapping
• Liquidity phase transitions
• Entropy-driven strategy routing
• Cross-layer strategy coordination
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class VaultState(Enum):
    """Vault states for liquidity management."""
    EMPTY = "empty"
    LOW = "low"
    STABLE = "stable"
    HIGH = "high"
    OVERFLOW = "overflow"
    PHANTOM = "phantom"
    LOCKED = "locked"

class OrbitalState(Enum):
    """Orbital states for quantum-inspired classification."""
    S = "s"      # Sharp/stable
    P = "p"      # Phase/periodic
    D = "d"      # Drift/diffuse
    F = "f"      # Fast/volatile
    NULL = "null" # No orbital

@dataclass
class VaultOrbitalMapping:
    """Mapping between vault and orbital states."""
    vault_state: VaultState
    orbital_state: OrbitalState
    transition_probability: float
    strategy_trigger: str
    liquidity_threshold: float
    entropy_threshold: float

@dataclass
class BridgeResult:
    """Result of vault-orbital bridge operation."""
    vault_state: VaultState
    orbital_state: OrbitalState
    recommended_strategy: str
    confidence: float
    transition_triggered: bool
    liquidity_level: float
    entropy_level: float

class VaultOrbitalBridge:
    """Bridge between vault and orbital logic systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize vault-orbital bridge."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # State mappings
        self.vault_orbital_mappings = self._build_mappings()
        
        # Transition history
        self.transition_history: List[Tuple[VaultState, OrbitalState, float]] = []
        
        # Current state tracking
        self.current_vault_state = VaultState.STABLE
        self.current_orbital_state = OrbitalState.S
        
        self.logger.info("✅ Vault-Orbital Bridge initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'enable_automatic_transitions': True,
            'transition_threshold': 0.7,
            'liquidity_thresholds': {
                'empty': 0.0,
                'low': 0.2,
                'stable': 0.5,
                'high': 0.8,
                'overflow': 1.0
            },
            'entropy_thresholds': {
                's': 0.1,
                'p': 0.3,
                'd': 0.5,
                'f': 0.8,
                'null': 1.0
            },
            'strategy_mappings': {
                'overflow_f': 'full_sweep_exit',
                'high_d': 'partial_profit_take',
                'stable_s': 'hold_maintain',
                'low_p': 'accumulate_dca',
                'empty_null': 'emergency_recovery'
            }
        }
    
    def _build_mappings(self) -> List[VaultOrbitalMapping]:
        """Build vault-to-orbital state mappings."""
        mappings = [
            # Overflow states
            VaultOrbitalMapping(
                vault_state=VaultState.OVERFLOW,
                orbital_state=OrbitalState.F,
                transition_probability=0.9,
                strategy_trigger="full_sweep_exit",
                liquidity_threshold=0.95,
                entropy_threshold=0.8
            ),
            
            # High liquidity states
            VaultOrbitalMapping(
                vault_state=VaultState.HIGH,
                orbital_state=OrbitalState.D,
                transition_probability=0.7,
                strategy_trigger="partial_profit_take",
                liquidity_threshold=0.8,
                entropy_threshold=0.5
            ),
            
            # Stable states
            VaultOrbitalMapping(
                vault_state=VaultState.STABLE,
                orbital_state=OrbitalState.S,
                transition_probability=0.8,
                strategy_trigger="hold_maintain",
                liquidity_threshold=0.5,
                entropy_threshold=0.1
            ),
            
            # Low liquidity states
            VaultOrbitalMapping(
                vault_state=VaultState.LOW,
                orbital_state=OrbitalState.P,
                transition_probability=0.6,
                strategy_trigger="accumulate_dca",
                liquidity_threshold=0.2,
                entropy_threshold=0.3
            ),
            
            # Empty states
            VaultOrbitalMapping(
                vault_state=VaultState.EMPTY,
                orbital_state=OrbitalState.NULL,
                transition_probability=0.5,
                strategy_trigger="emergency_recovery",
                liquidity_threshold=0.0,
                entropy_threshold=1.0
            ),
            
            # Phantom states
            VaultOrbitalMapping(
                vault_state=VaultState.PHANTOM,
                orbital_state=OrbitalState.F,
                transition_probability=0.8,
                strategy_trigger="phantom_hold",
                liquidity_threshold=0.3,
                entropy_threshold=0.9
            ),
            
            # Locked states
            VaultOrbitalMapping(
                vault_state=VaultState.LOCKED,
                orbital_state=OrbitalState.S,
                transition_probability=0.1,
                strategy_trigger="wait_unlock",
                liquidity_threshold=0.0,
                entropy_threshold=0.0
            )
        ]
        
        return mappings
    
    def determine_vault_state(self, liquidity_level: float, 
                            entropy_level: float, 
                            phantom_detected: bool = False) -> VaultState:
        """Determine vault state based on liquidity and entropy."""
        try:
            if phantom_detected:
                return VaultState.PHANTOM
            
            # Check liquidity thresholds
            thresholds = self.config['liquidity_thresholds']
            
            if liquidity_level <= thresholds['empty']:
                return VaultState.EMPTY
            elif liquidity_level <= thresholds['low']:
                return VaultState.LOW
            elif liquidity_level <= thresholds['stable']:
                return VaultState.STABLE
            elif liquidity_level <= thresholds['high']:
                return VaultState.HIGH
            elif liquidity_level >= thresholds['overflow']:
                return VaultState.OVERFLOW
            else:
                return VaultState.STABLE
                
        except Exception as e:
            self.logger.error(f"Error determining vault state: {e}")
            return VaultState.STABLE
    
    def determine_orbital_state(self, entropy_level: float, 
                              volatility: float, 
                              phase_consistency: float) -> OrbitalState:
        """Determine orbital state based on entropy and market conditions."""
        try:
            # Combine entropy, volatility, and phase consistency
            combined_score = (entropy_level * 0.4 + 
                            volatility * 0.4 + 
                            (1 - phase_consistency) * 0.2)
            
            # Map to orbital states
            if combined_score <= 0.2:
                return OrbitalState.S  # Sharp/stable
            elif combined_score <= 0.4:
                return OrbitalState.P  # Phase/periodic
            elif combined_score <= 0.6:
                return OrbitalState.D  # Drift/diffuse
            elif combined_score <= 0.8:
                return OrbitalState.F  # Fast/volatile
            else:
                return OrbitalState.NULL  # No orbital
                
        except Exception as e:
            self.logger.error(f"Error determining orbital state: {e}")
            return OrbitalState.S
    
    def bridge_states(self, liquidity_level: float, 
                     entropy_level: float, 
                     volatility: float = 0.0,
                     phase_consistency: float = 1.0,
                     phantom_detected: bool = False) -> BridgeResult:
        """Bridge vault and orbital states to determine strategy."""
        try:
            # Determine states
            vault_state = self.determine_vault_state(liquidity_level, entropy_level, phantom_detected)
            orbital_state = self.determine_orbital_state(entropy_level, volatility, phase_consistency)
            
            # Find matching mapping
            mapping = self._find_mapping(vault_state, orbital_state)
            
            # Calculate confidence based on how well states match
            confidence = self._calculate_confidence(vault_state, orbital_state, mapping)
            
            # Determine if transition should be triggered
            transition_triggered = self._should_trigger_transition(
                vault_state, orbital_state, mapping, confidence
            )
            
            # Get recommended strategy
            recommended_strategy = self._get_recommended_strategy(
                vault_state, orbital_state, mapping, transition_triggered
            )
            
            # Update current states
            if transition_triggered:
                self.current_vault_state = vault_state
                self.current_orbital_state = orbital_state
                self.transition_history.append((vault_state, orbital_state, entropy_level))
            
            return BridgeResult(
                vault_state=vault_state,
                orbital_state=orbital_state,
                recommended_strategy=recommended_strategy,
                confidence=confidence,
                transition_triggered=transition_triggered,
                liquidity_level=liquidity_level,
                entropy_level=entropy_level
            )
            
        except Exception as e:
            self.logger.error(f"Error bridging states: {e}")
            return BridgeResult(
                vault_state=VaultState.STABLE,
                orbital_state=OrbitalState.S,
                recommended_strategy="error_fallback",
                confidence=0.0,
                transition_triggered=False,
                liquidity_level=liquidity_level,
                entropy_level=entropy_level
            )
    
    def _find_mapping(self, vault_state: VaultState, 
                     orbital_state: OrbitalState) -> Optional[VaultOrbitalMapping]:
        """Find mapping for vault-orbital state combination."""
        for mapping in self.vault_orbital_mappings:
            if mapping.vault_state == vault_state and mapping.orbital_state == orbital_state:
                return mapping
        return None
    
    def _calculate_confidence(self, vault_state: VaultState, 
                            orbital_state: OrbitalState, 
                            mapping: Optional[VaultOrbitalMapping]) -> float:
        """Calculate confidence in state mapping."""
        if not mapping:
            return 0.0
        
        # Base confidence from mapping probability
        confidence = mapping.transition_probability
        
        # Adjust based on state consistency
        if vault_state == self.current_vault_state:
            confidence *= 1.1  # Boost for state consistency
        
        if orbital_state == self.current_orbital_state:
            confidence *= 1.1  # Boost for orbital consistency
        
        # Penalize for extreme states
        if vault_state in [VaultState.EMPTY, VaultState.OVERFLOW]:
            confidence *= 0.9
        
        if orbital_state == OrbitalState.NULL:
            confidence *= 0.8
        
        return min(1.0, confidence)
    
    def _should_trigger_transition(self, vault_state: VaultState, 
                                 orbital_state: OrbitalState, 
                                 mapping: Optional[VaultOrbitalMapping],
                                 confidence: float) -> bool:
        """Determine if state transition should be triggered."""
        if not self.config['enable_automatic_transitions']:
            return False
        
        if not mapping:
            return False
        
        # Check if states have changed
        state_changed = (vault_state != self.current_vault_state or 
                        orbital_state != self.current_orbital_state)
        
        # Check confidence threshold
        confidence_sufficient = confidence >= self.config['transition_threshold']
        
        return state_changed and confidence_sufficient
    
    def _get_recommended_strategy(self, vault_state: VaultState, 
                                orbital_state: OrbitalState, 
                                mapping: Optional[VaultOrbitalMapping],
                                transition_triggered: bool) -> str:
        """Get recommended strategy based on state mapping."""
        if mapping:
            return mapping.strategy_trigger
        
        # Fallback strategies based on individual states
        if vault_state == VaultState.OVERFLOW:
            return "emergency_exit"
        elif vault_state == VaultState.EMPTY:
            return "emergency_recovery"
        elif orbital_state == OrbitalState.F:
            return "volatility_management"
        elif orbital_state == OrbitalState.NULL:
            return "wait_for_signal"
        else:
            return "hold_maintain"
    
    def get_transition_statistics(self) -> Dict[str, Any]:
        """Get statistics about state transitions."""
        try:
            if not self.transition_history:
                return {
                    'total_transitions': 0,
                    'vault_state_distribution': {},
                    'orbital_state_distribution': {},
                    'average_entropy': 0.0
                }
            
            vault_counts = {}
            orbital_counts = {}
            entropies = []
            
            for vault_state, orbital_state, entropy in self.transition_history:
                vault_counts[vault_state.value] = vault_counts.get(vault_state.value, 0) + 1
                orbital_counts[orbital_state.value] = orbital_counts.get(orbital_state.value, 0) + 1
                entropies.append(entropy)
            
            return {
                'total_transitions': len(self.transition_history),
                'vault_state_distribution': vault_counts,
                'orbital_state_distribution': orbital_counts,
                'average_entropy': np.mean(entropies) if entropies else 0.0,
                'entropy_std': np.std(entropies) if entropies else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting transition statistics: {e}")
            return {}
    
    def get_current_state_summary(self) -> Dict[str, Any]:
        """Get current state summary."""
        return {
            'vault_state': self.current_vault_state.value,
            'orbital_state': self.current_orbital_state.value,
            'transition_count': len(self.transition_history),
            'last_transition': self.transition_history[-1] if self.transition_history else None
        }
    
    def reset_states(self) -> None:
        """Reset current states to defaults."""
        self.current_vault_state = VaultState.STABLE
        self.current_orbital_state = OrbitalState.S
        self.transition_history.clear()
        self.logger.info("States reset to defaults")

# Factory function
def create_vault_orbital_bridge(config: Optional[Dict[str, Any]] = None) -> VaultOrbitalBridge:
    """Create a vault-orbital bridge instance."""
    return VaultOrbitalBridge(config) 