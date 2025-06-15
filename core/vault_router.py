"""
Enhanced Vault Router with Dynamic Volume Allocation
===================================================

Advanced vault routing system that dynamically allocates trade volume based on:
- Collapse confidence scores
- Profit momentum (dP/dt)
- Fractal alignment strength
- Risk management constraints

Mathematical Foundation:
V = V₀ · confidence · (1 + dP/dt) · risk_factor

Where:
- V₀ = base volume allocation
- confidence = collapse confidence score [0, 1]
- dP/dt = profit momentum/velocity
- risk_factor = volatility and drawdown adjustment
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from enum import Enum

from collapse_confidence import CollapseState, CollapseConfidenceEngine

logger = logging.getLogger(__name__)

class VaultStatus(Enum):
    """Vault operational status"""
    CLOSED = "closed"
    OPEN = "open"
    RESTRICTED = "restricted"
    EMERGENCY_LOCK = "emergency_lock"

@dataclass
class VaultAllocation:
    """Volume allocation decision from vault router"""
    timestamp: float
    vault_status: VaultStatus
    base_volume: float
    allocated_volume: float
    volume_multiplier: float
    confidence_factor: float
    momentum_factor: float
    risk_factor: float
    reasoning: str
    allocation_components: Dict[str, float] = field(default_factory=dict)

@dataclass
class VaultMetrics:
    """Vault performance and allocation metrics"""
    total_allocations: int = 0
    avg_volume_multiplier: float = 1.0
    high_confidence_allocations: int = 0
    emergency_locks: int = 0
    total_volume_allocated: float = 0.0
    allocation_history: List[float] = field(default_factory=list)

class EnhancedVaultRouter:
    """
    Advanced vault routing system with dynamic volume allocation.
    
    Integrates collapse confidence scoring with risk management to
    determine optimal position sizing for each trading opportunity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced vault router.
        
        Args:
            config: Configuration parameters for vault routing
        """
        self.config = config or {}
        
        # Volume allocation parameters
        self.base_volume = self.config.get('base_volume', 1000.0)  # Base position size
        self.min_volume_multiplier = self.config.get('min_multiplier', 0.1)
        self.max_volume_multiplier = self.config.get('max_multiplier', 2.0)
        self.confidence_weight = self.config.get('confidence_weight', 0.6)
        self.momentum_weight = self.config.get('momentum_weight', 0.3)
        self.risk_weight = self.config.get('risk_weight', 0.1)
        
        # Risk management parameters
        self.max_drawdown_threshold = self.config.get('max_drawdown', 0.15)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.8)
        self.emergency_lock_threshold = self.config.get('emergency_threshold', 0.25)
        
        # Historical tracking
        self.profit_history: deque = deque(maxlen=50)
        self.volume_history: deque = deque(maxlen=100)
        self.allocation_history: deque = deque(maxlen=200)
        
        # Current state
        self.vault_status = VaultStatus.OPEN
        self.current_drawdown = 0.0
        self.total_allocated_volume = 0.0
        
        # Metrics tracking
        self.metrics = VaultMetrics()
        
        # Initialize confidence engine
        self.confidence_engine = CollapseConfidenceEngine()
        
        logger.info("Enhanced Vault Router initialized with dynamic volume allocation")
    
    def calculate_volume_allocation(self, collapse_state: CollapseState, 
                                  current_profit: float, recent_profits: List[float],
                                  market_volatility: float) -> VaultAllocation:
        """
        Calculate dynamic volume allocation based on collapse confidence and market conditions.
        
        Args:
            collapse_state: Collapse confidence state
            current_profit: Current profit/loss in basis points
            recent_profits: Recent profit history for momentum calculation
            market_volatility: Current market volatility
            
        Returns:
            VaultAllocation with calculated volume and reasoning
        """
        # Update profit history
        self.profit_history.append(current_profit)
        
        # Check vault status and emergency conditions
        vault_status = self._update_vault_status(market_volatility, current_profit)
        
        if vault_status == VaultStatus.EMERGENCY_LOCK:
            return self._create_emergency_allocation(collapse_state)
        
        # Calculate profit momentum (dP/dt)
        profit_momentum = self._calculate_profit_momentum(recent_profits)
        
        # Calculate risk factor
        risk_factor = self._calculate_risk_factor(market_volatility, current_profit)
        
        # Calculate volume multiplier components
        confidence_factor = self._calculate_confidence_factor(collapse_state)
        momentum_factor = self._calculate_momentum_factor(profit_momentum)
        
        # Combined volume multiplier
        volume_multiplier = self._calculate_volume_multiplier(
            confidence_factor, momentum_factor, risk_factor
        )
        
        # Calculate final allocated volume
        allocated_volume = self.base_volume * volume_multiplier
        
        # Create allocation components breakdown
        components = {
            "base_volume": self.base_volume,
            "confidence_contribution": confidence_factor * self.confidence_weight,
            "momentum_contribution": momentum_factor * self.momentum_weight,
            "risk_adjustment": risk_factor * self.risk_weight,
            "final_multiplier": volume_multiplier,
            "profit_momentum": profit_momentum
        }
        
        # Generate reasoning
        reasoning = self._generate_allocation_reasoning(
            collapse_state, confidence_factor, momentum_factor, risk_factor
        )
        
        # Create allocation object
        allocation = VaultAllocation(
            timestamp=time.time(),
            vault_status=vault_status,
            base_volume=self.base_volume,
            allocated_volume=allocated_volume,
            volume_multiplier=volume_multiplier,
            confidence_factor=confidence_factor,
            momentum_factor=momentum_factor,
            risk_factor=risk_factor,
            reasoning=reasoning,
            allocation_components=components
        )
        
        # Update tracking and metrics
        self._update_allocation_tracking(allocation)
        
        logger.info(f"Volume allocation: {allocated_volume:.0f} "
                   f"(multiplier: {volume_multiplier:.2f}, confidence: {collapse_state.confidence:.3f})")
        
        return allocation
    
    def _update_vault_status(self, market_volatility: float, current_profit: float) -> VaultStatus:
        """Update vault operational status based on market conditions."""
        # Calculate current drawdown
        if self.profit_history:
            peak_profit = max(self.profit_history)
            self.current_drawdown = (peak_profit - current_profit) / max(abs(peak_profit), 1.0)
        
        # Check emergency conditions
        if (self.current_drawdown > self.max_drawdown_threshold or 
            market_volatility > self.volatility_threshold):
            self.vault_status = VaultStatus.EMERGENCY_LOCK
            self.metrics.emergency_locks += 1
            logger.warning(f"Vault emergency lock triggered: drawdown={self.current_drawdown:.3f}, "
                          f"volatility={market_volatility:.3f}")
        elif market_volatility > 0.6:
            self.vault_status = VaultStatus.RESTRICTED
        else:
            self.vault_status = VaultStatus.OPEN
            
        return self.vault_status
    
    def _calculate_profit_momentum(self, recent_profits: List[float]) -> float:
        """Calculate profit momentum (dP/dt)."""
        if len(recent_profits) < 3:
            return 0.0
            
        # Calculate linear trend in recent profits
        x = np.arange(len(recent_profits))
        y = np.array(recent_profits)
        
        # Linear regression to get slope (momentum)
        if len(x) > 1:
            momentum = np.polyfit(x, y, 1)[0]
            # Normalize momentum to reasonable range
            normalized_momentum = np.tanh(momentum / 50.0)  # Scale by 50bp
            return normalized_momentum
        
        return 0.0
    
    def _calculate_risk_factor(self, market_volatility: float, current_profit: float) -> float:
        """Calculate risk adjustment factor."""
        # Volatility risk component
        volatility_risk = 1.0 - np.clip(market_volatility, 0.0, 1.0)
        
        # Drawdown risk component
        drawdown_risk = 1.0 - np.clip(self.current_drawdown, 0.0, 1.0)
        
        # Profit stability component
        if len(self.profit_history) > 5:
            profit_stability = 1.0 - (np.std(list(self.profit_history)) / 100.0)
            profit_stability = np.clip(profit_stability, 0.0, 1.0)
        else:
            profit_stability = 0.7
        
        # Combined risk factor
        risk_factor = 0.4 * volatility_risk + 0.4 * drawdown_risk + 0.2 * profit_stability
        
        return np.clip(risk_factor, 0.1, 1.0)
    
    def _calculate_confidence_factor(self, collapse_state: CollapseState) -> float:
        """Calculate confidence contribution to volume multiplier."""
        # Base confidence contribution
        confidence_contrib = collapse_state.confidence
        
        # Boost for very high confidence
        if collapse_state.confidence > 0.8:
            confidence_contrib *= 1.2
        
        # Penalty for low confidence
        elif collapse_state.confidence < 0.3:
            confidence_contrib *= 0.5
            
        return np.clip(confidence_contrib, 0.1, 1.5)
    
    def _calculate_momentum_factor(self, profit_momentum: float) -> float:
        """Calculate momentum contribution to volume multiplier."""
        # Positive momentum increases allocation
        if profit_momentum > 0:
            momentum_factor = 1.0 + profit_momentum * 0.5
        else:
            # Negative momentum decreases allocation
            momentum_factor = 1.0 + profit_momentum * 0.3
            
        return np.clip(momentum_factor, 0.2, 1.8)
    
    def _calculate_volume_multiplier(self, confidence_factor: float, 
                                   momentum_factor: float, risk_factor: float) -> float:
        """Calculate final volume multiplier using weighted components."""
        # Weighted combination of factors
        multiplier = (
            self.confidence_weight * confidence_factor +
            self.momentum_weight * momentum_factor +
            self.risk_weight * risk_factor
        )
        
        # Apply vault status restrictions
        if self.vault_status == VaultStatus.RESTRICTED:
            multiplier *= 0.5
        elif self.vault_status == VaultStatus.EMERGENCY_LOCK:
            multiplier = 0.0
            
        return np.clip(multiplier, self.min_volume_multiplier, self.max_volume_multiplier)
    
    def _generate_allocation_reasoning(self, collapse_state: CollapseState,
                                     confidence_factor: float, momentum_factor: float,
                                     risk_factor: float) -> str:
        """Generate human-readable reasoning for allocation decision."""
        confidence_grade = self.confidence_engine.get_confidence_grade(collapse_state.confidence)
        
        reasoning_parts = []
        
        # Confidence reasoning
        if collapse_state.confidence > 0.7:
            reasoning_parts.append(f"High confidence ({confidence_grade})")
        elif collapse_state.confidence < 0.4:
            reasoning_parts.append(f"Low confidence ({confidence_grade})")
        else:
            reasoning_parts.append(f"Moderate confidence ({confidence_grade})")
        
        # Momentum reasoning
        if momentum_factor > 1.2:
            reasoning_parts.append("strong positive momentum")
        elif momentum_factor < 0.8:
            reasoning_parts.append("negative momentum")
        
        # Risk reasoning
        if risk_factor < 0.5:
            reasoning_parts.append("high risk environment")
        elif risk_factor > 0.8:
            reasoning_parts.append("favorable risk conditions")
        
        # Vault status
        if self.vault_status == VaultStatus.RESTRICTED:
            reasoning_parts.append("vault restricted")
        elif self.vault_status == VaultStatus.EMERGENCY_LOCK:
            reasoning_parts.append("emergency lock active")
        
        return "; ".join(reasoning_parts)
    
    def _create_emergency_allocation(self, collapse_state: CollapseState) -> VaultAllocation:
        """Create emergency allocation with zero volume."""
        return VaultAllocation(
            timestamp=time.time(),
            vault_status=VaultStatus.EMERGENCY_LOCK,
            base_volume=self.base_volume,
            allocated_volume=0.0,
            volume_multiplier=0.0,
            confidence_factor=0.0,
            momentum_factor=0.0,
            risk_factor=0.0,
            reasoning="Emergency lock - no allocation",
            allocation_components={"emergency_lock": True}
        )
    
    def _update_allocation_tracking(self, allocation: VaultAllocation):
        """Update allocation tracking and metrics."""
        # Update volume history
        self.volume_history.append(allocation.allocated_volume)
        self.allocation_history.append(allocation)
        
        # Update total allocated volume
        self.total_allocated_volume += allocation.allocated_volume
        
        # Update metrics
        self.metrics.total_allocations += 1
        self.metrics.total_volume_allocated += allocation.allocated_volume
        
        # Update running averages
        alpha = 0.1
        self.metrics.avg_volume_multiplier = (
            alpha * allocation.volume_multiplier + 
            (1 - alpha) * self.metrics.avg_volume_multiplier
        )
        
        # Track high confidence allocations
        if allocation.confidence_factor > 0.7:
            self.metrics.high_confidence_allocations += 1
        
        # Update allocation history
        self.metrics.allocation_history.append(allocation.volume_multiplier)
        if len(self.metrics.allocation_history) > 100:
            self.metrics.allocation_history.pop(0)
    
    def get_vault_summary(self) -> Dict[str, Any]:
        """Get comprehensive vault status and metrics summary."""
        if self.metrics.total_allocations == 0:
            return {"status": "no_allocations"}
            
        return {
            "vault_status": self.vault_status.value,
            "current_drawdown": self.current_drawdown,
            "total_allocations": self.metrics.total_allocations,
            "avg_volume_multiplier": self.metrics.avg_volume_multiplier,
            "total_volume_allocated": self.metrics.total_volume_allocated,
            "high_confidence_rate": self.metrics.high_confidence_allocations / self.metrics.total_allocations,
            "emergency_lock_count": self.metrics.emergency_locks,
            "allocation_stability": 1.0 - np.var(self.metrics.allocation_history) if len(self.metrics.allocation_history) > 5 else 0.5,
            "recent_allocations": list(self.volume_history)[-10:] if self.volume_history else []
        }
    
    def update_vault_trigger(self, current_tick: float, braid_signal: float, 
                           meta_state: str) -> str:
        """
        Enhanced vault trigger logic based on braid signal and meta state.
        
        Args:
            current_tick: Current market tick
            braid_signal: Current braid fractal signal
            meta_state: Current meta state of the system
            
        Returns:
            Vault trigger decision: "vault_open", "vault_hold", "vault_close"
        """
        if self.vault_status == VaultStatus.EMERGENCY_LOCK:
            return "vault_close"
            
        # High confidence braid signal with rebound sync
        if braid_signal > 0.8 and meta_state == "rebound_sync":
            return "vault_open"
        
        # Moderate signal with stable conditions
        elif braid_signal > 0.6 and self.vault_status == VaultStatus.OPEN:
            return "vault_open"
        
        # Low signal or restricted conditions
        elif braid_signal < 0.3 or self.vault_status == VaultStatus.RESTRICTED:
            return "vault_hold"
        
        return "vault_hold"
    
    def reset_vault_state(self):
        """Reset vault state and metrics."""
        self.vault_status = VaultStatus.OPEN
        self.current_drawdown = 0.0
        self.total_allocated_volume = 0.0
        self.metrics = VaultMetrics()
        self.profit_history.clear()
        self.volume_history.clear()
        self.allocation_history.clear()
        logger.info("Vault router state reset")

# Example usage and testing
if __name__ == "__main__":
    # Test enhanced vault router
    router = EnhancedVaultRouter()
    
    # Create test collapse state
    from collapse_confidence import CollapseConfidenceEngine
    
    confidence_engine = CollapseConfidenceEngine()
    collapse_state = confidence_engine.calculate_collapse_confidence(
        profit_delta=85.0,
        braid_signal=0.75,
        paradox_signal=0.65,
        recent_volatility=[0.2, 0.3, 0.25]
    )
    
    # Test volume allocation
    allocation = router.calculate_volume_allocation(
        collapse_state=collapse_state,
        current_profit=50.0,
        recent_profits=[30.0, 40.0, 45.0, 50.0],
        market_volatility=0.3
    )
    
    print(f"Volume Allocation: {allocation.allocated_volume:.0f}")
    print(f"Volume Multiplier: {allocation.volume_multiplier:.2f}")
    print(f"Reasoning: {allocation.reasoning}")
    print(f"Components: {allocation.allocation_components}")
    
    # Test vault trigger
    trigger = router.update_vault_trigger(
        current_tick=time.time(),
        braid_signal=0.8,
        meta_state="rebound_sync"
    )
    print(f"Vault Trigger: {trigger}")
    
    # Get vault summary
    summary = router.get_vault_summary()
    print(f"Vault Summary: {summary}") 