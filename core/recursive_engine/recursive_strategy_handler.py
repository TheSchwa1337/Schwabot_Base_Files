from __future__ import annotations
import logging
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradeSignal:
    action: TradeAction
    confidence: float
    metadata: Dict[str, Any]

class RecursiveStrategyHandler:
    """
    Processes recursive signals and makes trading decisions based on:
    - Ψ(t) integration values
    - Entropy and coherence metrics
    - Historical pattern matching
    - Current market state
    """
    
    def __init__(
        self,
        min_confidence: float = 0.75,
        max_position_size: float = 1.0,
        entropy_threshold: float = 0.7,
        coherence_threshold: float = 0.6
    ):
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size
        self.entropy_threshold = entropy_threshold
        self.coherence_threshold = coherence_threshold
        
        # State tracking
        self.current_position: float = 0.0  # -1 to 1, negative = short
        self.last_trade_price: Optional[float] = None
        self.psi_history: list[float] = []
        self.entropy_history: list[float] = []
        
        logger.info("RecursiveStrategyHandler initialized with confidence threshold %.2f", min_confidence)

    def _calculate_position_size(self, confidence: float, current_price: float) -> float:
        """Calculate position size based on confidence and current price."""
        base_size = self.max_position_size * confidence
        
        # Adjust for current position
        if self.current_position != 0:
            # Reduce size if we already have a position
            base_size *= (1 - abs(self.current_position))
            
        return base_size

    def _evaluate_psi_slope(self) -> float:
        """Calculate the rate of change of Ψ(t)."""
        if len(self.psi_history) < 2:
            return 0.0
            
        recent_psi = self.psi_history[-10:]  # Look at last 10 points
        if len(recent_psi) < 2:
            return 0.0
            
        return np.polyfit(range(len(recent_psi)), recent_psi, 1)[0]

    def _evaluate_entropy_trend(self) -> float:
        """Calculate the trend in entropy."""
        if len(self.entropy_history) < 2:
            return 0.0
            
        recent_entropy = self.entropy_history[-10:]
        if len(recent_entropy) < 2:
            return 0.0
            
        return np.polyfit(range(len(recent_entropy)), recent_entropy, 1)[0]

    def process_signal(
        self,
        psi: float,
        entropy: float,
        coherence: float,
        current_price: float,
        metadata: Dict[str, Any]
    ) -> TradeSignal:
        """
        Process recursive signals and generate trading decisions.
        
        Parameters
        ----------
        psi : float
            Current Ψ(t) integration value
        entropy : float
            Current entropy measure
        coherence : float
            Current coherence score
        current_price : float
            Current BTC/USDC price
        metadata : dict
            Additional market data and state information
            
        Returns
        -------
        TradeSignal
            Trading decision with confidence and metadata
        """
        # Update history
        self.psi_history.append(psi)
        self.entropy_history.append(entropy)
        
        # Calculate trends
        psi_slope = self._evaluate_psi_slope()
        entropy_trend = self._evaluate_entropy_trend()
        
        # Initialize decision components
        action = TradeAction.HOLD
        confidence = 0.0
        
        # Decision logic based on recursive state
        if metadata.get('recursive_state') == 'TPF':
            # Paradox Fractal state - high volatility, potential reversal
            if psi_slope > 0 and entropy > self.entropy_threshold:
                action = TradeAction.BUY if self.current_position <= 0 else TradeAction.HOLD
                confidence = min(0.8, entropy)
            elif psi_slope < 0 and entropy > self.entropy_threshold:
                action = TradeAction.SELL if self.current_position >= 0 else TradeAction.HOLD
                confidence = min(0.8, entropy)
                
        elif metadata.get('recursive_state') == 'TEF':
            # Echo Fractal state - pattern replay
            if coherence > self.coherence_threshold:
                # Strong pattern recognition
                if psi > np.mean(self.psi_history[-20:]):
                    action = TradeAction.BUY if self.current_position <= 0 else TradeAction.HOLD
                    confidence = coherence
                else:
                    action = TradeAction.SELL if self.current_position >= 0 else TradeAction.HOLD
                    confidence = coherence
                    
        else:  # TFF - Forever Fractal state
            # Stable state, use Ψ slope for direction
            if abs(psi_slope) > 0.1:  # Significant trend
                if psi_slope > 0:
                    action = TradeAction.BUY if self.current_position <= 0 else TradeAction.HOLD
                    confidence = min(0.7, abs(psi_slope))
                else:
                    action = TradeAction.SELL if self.current_position >= 0 else TradeAction.HOLD
                    confidence = min(0.7, abs(psi_slope))

        # Calculate position size
        position_size = self._calculate_position_size(confidence, current_price)
        
        # Update current position
        if action == TradeAction.BUY:
            self.current_position = position_size
        elif action == TradeAction.SELL:
            self.current_position = -position_size
            
        self.last_trade_price = current_price
        
        # Prepare metadata
        signal_metadata = {
            'psi_slope': psi_slope,
            'entropy_trend': entropy_trend,
            'position_size': position_size,
            'current_position': self.current_position,
            'recursive_state': metadata.get('recursive_state'),
            'coherence': coherence
        }
        
        return TradeSignal(
            action=action,
            confidence=confidence,
            metadata=signal_metadata
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current handler state."""
        return {
            'current_position': self.current_position,
            'last_trade_price': self.last_trade_price,
            'psi_history_length': len(self.psi_history),
            'entropy_history_length': len(self.entropy_history)
        } 