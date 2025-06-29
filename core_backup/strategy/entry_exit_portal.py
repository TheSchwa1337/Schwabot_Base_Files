# -*- coding: utf-8 -*-
"""
Entry/Exit Portal for Glyph Strategy Integration
-----------------------------------------------
Handles trade entry/exit signals from glyph strategy core and integrates
with Schwabot's trading execution system.

Provides signal processing, position sizing, and execution coordination
for both live and simulated trading modes.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Import glyph strategy core
try:
    from core.strategy.glyph_strategy_core import GlyphStrategyCore, GlyphStrategyResult
except ImportError:
    GlyphStrategyCore = None
    GlyphStrategyResult = None

# Import existing Schwabot components
try:
    from core.strategy_logic import StrategyLogic, SignalType, SignalStrength
    from core.trade_executor import TradeExecutor
    from core.risk_manager import RiskManager
    from core.portfolio_tracker import PortfolioTracker
except ImportError:
    StrategyLogic = None
    SignalType = None
    SignalStrength = None
    TradeExecutor = None
    RiskManager = None
    PortfolioTracker = None

logger = logging.getLogger(__name__)

class SignalDirection(Enum):
    """Signal direction enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

@dataclass
class TradeSignal:
    """Trade signal container."""
    glyph: str
    strategy_id: int
    direction: SignalDirection
    asset: str
    price: float
    volume: float
    confidence: float
    fractal_hash: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class PositionSizing:
    """Position sizing parameters."""
    base_size: float
    confidence_multiplier: float
    risk_adjusted_size: float
    max_position_size: float
    min_position_size: float

class EntryExitPortal:
    """
    Entry/Exit Portal for glyph strategy integration.
    
    Processes glyph strategy signals and coordinates trade execution
    with risk management and portfolio tracking.
    """
    
    def __init__(self,
                 glyph_core: Optional[GlyphStrategyCore] = None,
                 enable_risk_management: bool = True,
                 enable_portfolio_tracking: bool = True,
                 max_position_size: float = 0.1,
                 min_confidence_threshold: float = 0.6):
        """
        Initialize the entry/exit portal.
        
        Args:
            glyph_core: Glyph strategy core instance
            enable_risk_management: Enable risk management integration
            enable_portfolio_tracking: Enable portfolio tracking
            max_position_size: Maximum position size as fraction of portfolio
            min_confidence_threshold: Minimum confidence for trade execution
        """
        self.glyph_core = glyph_core or GlyphStrategyCore()
        self.enable_risk_management = enable_risk_management
        self.enable_portfolio_tracking = enable_portfolio_tracking
        self.max_position_size = max_position_size
        self.min_confidence_threshold = min_confidence_threshold
        
        # Initialize components
        self.strategy_logic = StrategyLogic() if StrategyLogic else None
        self.trade_executor = TradeExecutor() if TradeExecutor else None
        self.risk_manager = RiskManager() if RiskManager and enable_risk_management else None
        self.portfolio_tracker = PortfolioTracker() if PortfolioTracker and enable_portfolio_tracking else None
        
        # Signal processing state
        self.active_signals: List[TradeSignal] = []
        self.signal_history: List[TradeSignal] = []
        self.max_signal_history = 1000
        
        # Performance tracking
        self.stats = {
            "total_signals": 0,
            "executed_trades": 0,
            "rejected_signals": 0,
            "avg_processing_time": 0.0
        }
        
        logger.info(f"EntryExitPortal initialized: "
                   f"risk_mgmt={enable_risk_management}, "
                   f"portfolio_tracking={enable_portfolio_tracking}")
    
    def process_glyph_signal(self, glyph: str, volume_signal: float,
                           asset: str = "BTC/USD", current_price: float = 0.0,
                           confidence_boost: float = 0.0) -> Optional[TradeSignal]:
        """
        Process glyph signal and generate trade signal.
        
        Args:
            glyph: Input glyph
            volume_signal: Market volume signal
            asset: Trading asset pair
            current_price: Current asset price
            confidence_boost: Additional confidence boost
            
        Returns:
            TradeSignal if valid, None otherwise
        """
        start_time = time.time()
        
        try:
            # Get strategy selection from glyph core
            strategy_result = self.glyph_core.select_strategy(
                glyph, volume_signal, confidence_boost
            )
            
            # Check confidence threshold
            if strategy_result.confidence < self.min_confidence_threshold:
                logger.debug(f"Signal rejected: confidence {strategy_result.confidence:.3f} "
                           f"below threshold {self.min_confidence_threshold}")
                self.stats["rejected_signals"] += 1
                return None
            
            # Determine signal direction based on strategy and market conditions
            direction = self._determine_signal_direction(
                strategy_result, volume_signal, current_price
            )
            
            # Create trade signal
            signal = TradeSignal(
                glyph=glyph,
                strategy_id=strategy_result.strategy_id,
                direction=direction,
                asset=asset,
                price=current_price,
                volume=volume_signal,
                confidence=strategy_result.confidence,
                fractal_hash=strategy_result.fractal_hash,
                metadata={
                    "gear_state": strategy_result.gear_state,
                    "processing_time": time.time() - start_time
                }
            )
            
            # Store signal
            self.active_signals.append(signal)
            self.signal_history.append(signal)
            
            # Maintain history size
            if len(self.signal_history) > self.max_signal_history:
                self.signal_history.pop(0)
            
            self.stats["total_signals"] += 1
            
            logger.info(f"Trade signal generated: {glyph} -> {direction.value} "
                       f"{asset} (confidence: {strategy_result.confidence:.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal processing failed: {e}")
            return None
    
    def _determine_signal_direction(self, strategy_result: GlyphStrategyResult,
                                  volume_signal: float, current_price: float) -> SignalDirection:
        """
        Determine signal direction based on strategy and market conditions.
        
        Args:
            strategy_result: Glyph strategy result
            volume_signal: Market volume signal
            current_price: Current asset price
            
        Returns:
            Signal direction
        """
        # Simple heuristic based on strategy ID and volume
        strategy_id = strategy_result.strategy_id
        gear_state = strategy_result.gear_state
        
        # Use strategy ID to determine bias
        if strategy_id % 2 == 0:  # Even strategies tend to be bullish
            base_direction = SignalDirection.BUY
        else:  # Odd strategies tend to be bearish
            base_direction = SignalDirection.SELL
        
        # Adjust based on volume and gear state
        if volume_signal > 5e6 and gear_state >= 8:  # High volume, high gear
            if base_direction == SignalDirection.BUY:
                return SignalDirection.BUY
            else:
                return SignalDirection.SELL
        elif volume_signal < 1e6:  # Low volume
            return SignalDirection.HOLD
        else:  # Medium volume
            return base_direction
    
    def calculate_position_size(self, signal: TradeSignal,
                              portfolio_value: float = 10000.0) -> PositionSizing:
        """
        Calculate position size based on signal and risk parameters.
        
        Args:
            signal: Trade signal
            portfolio_value: Current portfolio value
            
        Returns:
            Position sizing parameters
        """
        # Base position size
        base_size = portfolio_value * self.max_position_size
        
        # Adjust for confidence
        confidence_multiplier = signal.confidence
        
        # Risk-adjusted size
        risk_adjusted_size = base_size * confidence_multiplier
        
        # Apply risk management if available
        if self.risk_manager:
            # Get risk adjustment from risk manager
            risk_multiplier = self.risk_manager.get_position_risk_multiplier(
                signal.asset, signal.direction.value
            )
            risk_adjusted_size *= risk_multiplier
        
        # Ensure within bounds
        max_size = portfolio_value * self.max_position_size
        min_size = portfolio_value * 0.01  # 1% minimum
        
        final_size = max(min_size, min(max_size, risk_adjusted_size))
        
        return PositionSizing(
            base_size=base_size,
            confidence_multiplier=confidence_multiplier,
            risk_adjusted_size=risk_adjusted_size,
            max_position_size=max_size,
            min_position_size=min_size
        )
    
    def execute_signal(self, signal: TradeSignal,
                      portfolio_value: float = 10000.0,
                      dry_run: bool = True) -> Dict[str, any]:
        """
        Execute trade signal.
        
        Args:
            signal: Trade signal to execute
            portfolio_value: Current portfolio value
            dry_run: If True, simulate execution without actual trades
            
        Returns:
            Execution result dictionary
        """
        start_time = time.time()
        
        try:
            # Calculate position size
            position_sizing = self.calculate_position_size(signal, portfolio_value)
            
            # Prepare execution parameters
            execution_params = {
                "asset": signal.asset,
                "direction": signal.direction.value,
                "size": position_sizing.risk_adjusted_size,
                "price": signal.price,
                "confidence": signal.confidence,
                "strategy_id": signal.strategy_id,
                "fractal_hash": signal.fractal_hash,
                "dry_run": dry_run
            }
            
            # Execute trade if trade executor is available
            if self.trade_executor and not dry_run:
                execution_result = self.trade_executor.execute_trade(execution_params)
            else:
                # Simulate execution
                execution_result = {
                    "status": "simulated",
                    "order_id": f"sim_{int(time.time())}",
                    "executed_price": signal.price,
                    "executed_size": position_sizing.risk_adjusted_size,
                    "fees": position_sizing.risk_adjusted_size * 0.001,  # 0.1% fee
                    "timestamp": time.time()
                }
            
            # Update portfolio if tracking is enabled
            if self.portfolio_tracker and not dry_run:
                self.portfolio_tracker.update_position(
                    signal.asset, signal.direction.value,
                    position_sizing.risk_adjusted_size, signal.price
                )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["executed_trades"] += 1
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (self.stats["executed_trades"] - 1) + 
                 processing_time) / self.stats["executed_trades"]
            )
            
            # Remove from active signals
            if signal in self.active_signals:
                self.active_signals.remove(signal)
            
            result = {
                "signal": signal,
                "position_sizing": position_sizing,
                "execution_result": execution_result,
                "processing_time": processing_time
            }
            
            logger.info(f"Signal executed: {signal.glyph} -> {signal.direction.value} "
                       f"{signal.asset} (size: {position_sizing.risk_adjusted_size:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Signal execution failed: {e}")
            return {
                "error": str(e),
                "signal": signal,
                "processing_time": time.time() - start_time
            }
    
    def get_active_signals(self) -> List[TradeSignal]:
        """Get currently active signals."""
        return self.active_signals.copy()
    
    def get_signal_history(self, limit: int = 100) -> List[TradeSignal]:
        """Get signal history."""
        return self.signal_history[-limit:]
    
    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics."""
        return {
            **self.stats,
            "active_signals": len(self.active_signals),
            "signal_history_size": len(self.signal_history)
        }
    
    def clear_signals(self):
        """Clear all active signals."""
        self.active_signals.clear()
        logger.info("All active signals cleared")

# Convenience function for quick signal processing
def process_glyph_trade_signal(glyph: str, volume: float, asset: str = "BTC/USD",
                              price: float = 50000.0, dry_run: bool = True) -> Dict[str, any]:
    """
    Convenience function for quick glyph trade signal processing.
    
    Args:
        glyph: Input glyph
        volume: Market volume signal
        asset: Trading asset
        price: Current asset price
        dry_run: If True, simulate execution
        
    Returns:
        Processing result dictionary
    """
    portal = EntryExitPortal()
    
    # Process signal
    signal = portal.process_glyph_signal(glyph, volume, asset, price)
    
    if signal is None:
        return {"error": "Signal rejected"}
    
    # Execute signal
    result = portal.execute_signal(signal, dry_run=dry_run)
    
    return {
        "signal": {
            "glyph": signal.glyph,
            "direction": signal.direction.value,
            "asset": signal.asset,
            "confidence": signal.confidence,
            "strategy_id": signal.strategy_id
        },
        "execution": result.get("execution_result", {}),
        "stats": portal.get_performance_stats()
    }

if __name__ == "__main__":
    # Test the entry/exit portal
    portal = EntryExitPortal()
    
    test_glyphs = ['🧠', '💀', '🔥', '⏳', '🌪️']
    test_volumes = [1e6, 3e6, 6e6]
    
    print("=== Entry/Exit Portal Test ===")
    
    for glyph in test_glyphs:
        for volume in test_volumes:
            result = process_glyph_trade_signal(glyph, volume, dry_run=True)
            if "error" not in result:
                signal = result["signal"]
                print(f"Glyph: {signal['glyph']}, Direction: {signal['direction']}, "
                      f"Confidence: {signal['confidence']:.3f}")
    
    print(f"\nPortal Stats: {portal.get_performance_stats()}") 