#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phantom Band Navigator Strategy
==============================

Advanced trading strategy implementing Phantom Math for pre-candle entry detection.
Uses entropy-driven pattern recognition to identify profitable trading opportunities
before traditional indicators trigger.

Core Features:
- Phantom Zone detection and navigation
- Entropy-driven entry/exit logic
- Pattern similarity matching
- Risk-adjusted position sizing
- Multi-timeframe analysis
- Live execution integration
"""

import asyncio
import logging
import os

# Import Phantom Math components
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.phantom_detector import PhantomDetector, PhantomZone
from core.phantom_logger import PhantomLogger

logger = logging.getLogger(__name__)

@dataclass
class PhantomSignal:
    """Phantom trading signal with full metadata."""
    symbol: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    entry_price: float
    exit_price: float
    confidence: float
    phantom_zone: PhantomZone
    timestamp: float
    risk_level: str
    position_size: float
    stop_loss: float
    take_profit: float
    market_condition: str
    strategy_metadata: Dict[str, Any]

class PhantomBandNavigator:
    """Advanced Phantom Band Navigator trading strategy."""
    
    def __init__(self, 
                 symbols: List[str] = None,
                 base_position_size: float = 0.01,
                 max_risk_per_trade: float = 0.02,
                 phantom_threshold: float = 0.7,
                 similarity_threshold: float = 0.8):
        
        self.symbols = symbols or ["BTC", "ETH", "ADA", "SOL", "XRP"]
        self.base_position_size = base_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.phantom_threshold = phantom_threshold
        self.similarity_threshold = similarity_threshold
        
        # Initialize Phantom Math components
        self.detector = PhantomDetector()
        self.logger = PhantomLogger()
        
        # Strategy state
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.phantom_history: List[PhantomZone] = []
        self.signal_history: List[PhantomSignal] = []
        
        # Performance tracking
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        
        # Market condition tracking
        self.market_conditions = {
            "bull": 0,
            "bear": 0,
            "sideways": 0,
            "volatile": 0
        }
        
        logger.info("Phantom Band Navigator initialized")
    
    def analyze_market_condition(self, tick_prices: List[float]) -> str:
        """Analyze current market condition based on price action."""
        if len(tick_prices) < 20:
            return "unknown"
        
        # Calculate price momentum
        recent_prices = tick_prices[-20:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        volatility = np.std(np.diff(recent_prices)) / np.mean(recent_prices)
        
        # Determine market condition
        if abs(price_change) < 0.01 and volatility < 0.005:
            return "sideways"
        elif volatility > 0.02:
            return "volatile"
        elif price_change > 0.02:
            return "bull"
        elif price_change < -0.02:
            return "bear"
        else:
            return "sideways"
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              confidence: float, available_balance: float) -> float:
        """Calculate position size based on risk management and confidence."""
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0.0
        
        # Base position size
        base_size = self.base_position_size * available_balance
        
        # Adjust for confidence
        confidence_multiplier = min(confidence / self.phantom_threshold, 2.0)
        
        # Adjust for risk
        risk_multiplier = min(self.max_risk_per_trade / (risk_per_share / entry_price), 1.0)
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * risk_multiplier
        
        # Ensure it doesn't exceed available balance
        position_size = min(position_size, available_balance * 0.1)  # Max 10% of balance
        
        return max(0.0, position_size)
    
    def calculate_stop_loss_take_profit(self, entry_price: float, 
                                      confidence: float, market_condition: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        # Base percentages
        base_stop_loss_pct = 0.02  # 2%
        base_take_profit_pct = 0.04  # 4%
        
        # Adjust based on confidence
        confidence_multiplier = confidence / self.phantom_threshold
        
        # Adjust based on market condition
        condition_multipliers = {
            "bull": 1.2,
            "bear": 0.8,
            "sideways": 1.0,
            "volatile": 1.5
        }
        condition_multiplier = condition_multipliers.get(market_condition, 1.0)
        
        # Calculate final levels
        stop_loss_pct = base_stop_loss_pct * condition_multiplier / confidence_multiplier
        take_profit_pct = base_take_profit_pct * condition_multiplier * confidence_multiplier
        
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
        
        return stop_loss, take_profit
    
    def phantom_band_navigator(self, symbol: str, tick_window: List[float], 
                             available_balance: float = 1000.0) -> Optional[PhantomSignal]:
        """
        Main Phantom Band Navigator strategy.
        
        Args:
            symbol: Trading symbol
            tick_window: Recent tick prices
            available_balance: Available balance for trading
            
        Returns:
            PhantomSignal if strategy triggers, None otherwise
        """
        try:
            # Analyze market condition
            market_condition = self.analyze_market_condition(tick_window)
            self.market_conditions[market_condition] += 1
            
            # Detect Phantom Zone
            if not self.detector.detect(tick_window, symbol):
                return None
            
            # Get full Phantom Zone analysis
            phantom_zone = self.detector.detect_phantom_zone(tick_window, symbol)
            if not phantom_zone:
                return None
            
            # Check confidence threshold
            if phantom_zone.confidence < self.phantom_threshold:
                return None
            
            # Calculate entry price
            entry_price = tick_window[-1]
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                entry_price, phantom_zone.confidence, market_condition
            )
            
            # Calculate position size
            position_size = self.calculate_position_size(
                entry_price, stop_loss, phantom_zone.confidence, available_balance
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(phantom_zone.confidence)
            
            # Create signal
            signal = PhantomSignal(
                symbol=symbol,
                signal_type="BUY",  # Default to BUY for Phantom strategy
                entry_price=entry_price,
                exit_price=take_profit,
                confidence=phantom_zone.confidence,
                phantom_zone=phantom_zone,
                timestamp=time.time(),
                risk_level=risk_level,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                market_condition=market_condition,
                strategy_metadata={
                    "phantom_zone_id": phantom_zone.zone_id,
                    "similarity_score": phantom_zone.similarity_score,
                    "entropy_value": phantom_zone.entropy_value
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in phantom band navigator: {e}")
            return None
    
    def _determine_risk_level(self, confidence: float) -> str:
        """Determine risk level based on confidence."""
        if confidence > 0.9:
            return "low"
        elif confidence > 0.7:
            return "medium"
        else:
            return "high"
    
    def execute_signal(self, signal: PhantomSignal, current_price: float) -> Dict[str, Any]:
        """Execute a trading signal."""
        try:
            # Check existing positions
            if signal.symbol in self.active_positions:
                position = self.active_positions[signal.symbol]
                
                # Check stop loss
                if current_price <= position['stop_loss']:
                    return self._exit_position(signal.symbol, current_price, "stop_loss")
                
                # Check take profit
                if current_price >= position['take_profit']:
                    return self._exit_position(signal.symbol, current_price, "take_profit")
                
                # Check Phantom exit conditions
                if self._should_exit_phantom(position, current_price):
                    return self._exit_position(signal.symbol, current_price, "phantom_exit")
            
            # Enter new position
            if signal.signal_type == "BUY":
                return self._enter_position(signal, current_price)
            
            return {"action": "none", "reason": "no_action_taken"}
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {"action": "error", "reason": str(e)}
    
    def _enter_position(self, signal: PhantomSignal, current_price: float) -> Dict[str, Any]:
        """Enter a new trading position."""
        try:
            position = {
                "entry_price": signal.entry_price,
                "entry_time": signal.timestamp,
                "position_size": signal.position_size,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "phantom_zone": signal.phantom_zone,
                "signal": signal
            }
            
            self.active_positions[signal.symbol] = position
            self.signal_history.append(signal)
            
            logger.info(f"Entered position for {signal.symbol} at ${current_price:.2f}")
            
            return {
                "action": "enter",
                "symbol": signal.symbol,
                "price": current_price,
                "size": signal.position_size,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit
            }
            
        except Exception as e:
            logger.error(f"Error entering position: {e}")
            return {"action": "error", "reason": str(e)}
    
    def _exit_position(self, symbol: str, current_price: float, exit_reason: str) -> Dict[str, Any]:
        """Exit an existing trading position."""
        try:
            if symbol not in self.active_positions:
                return {"action": "none", "reason": "no_position"}
            
            position = self.active_positions[symbol]
            entry_price = position["entry_price"]
            
            # Calculate profit/loss
            if position["signal"].signal_type == "BUY":
                profit = (current_price - entry_price) * position["position_size"]
            else:
                profit = (entry_price - current_price) * position["position_size"]
            
            # Update statistics
            self.total_trades += 1
            if profit > 0:
                self.profitable_trades += 1
            self.total_profit += profit
            
            # Update max drawdown
            if profit < 0:
                self.max_drawdown = min(self.max_drawdown, profit)
            
            # Update Phantom Zone with results
            phantom_zone = position["phantom_zone"]
            phantom_zone.exit_tick = current_price
            phantom_zone.profit_actual = profit
            self.detector.update_phantom_zone(phantom_zone, current_price, profit)
            
            # Log the Phantom Zone
            self.logger.log_zone(
                phantom_zone, 
                profit_actual=profit,
                market_condition=position["signal"].market_condition,
                strategy_used="phantom_band_navigator"
            )
            
            # Remove from active positions
            del self.active_positions[symbol]
            
            logger.info(f"Exited position for {symbol} at ${current_price:.2f}")
            logger.info(f"  Profit: ${profit:.4f}, Reason: {exit_reason}")
            
            return {
                "action": "exit",
                "symbol": symbol,
                "price": current_price,
                "profit": profit,
                "reason": exit_reason
            }
            
        except Exception as e:
            logger.error(f"Error exiting position: {e}")
            return {"action": "error", "reason": str(e)}
    
    def _should_exit_phantom(self, position: Dict[str, Any], current_price: float) -> bool:
        """Check if we should exit based on Phantom conditions."""
        try:
            phantom_zone = position["phantom_zone"]
            entry_time = position["entry_time"]
            current_time = time.time()
            
            # Exit if Phantom duration exceeded
            max_phantom_duration = 300  # 5 minutes
            if current_time - entry_time > max_phantom_duration:
                return True
            
            # Exit if price moved significantly against us
            entry_price = position["entry_price"]
            price_change = abs(current_price - entry_price) / entry_price
            
            if price_change > 0.05:  # 5% move
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking Phantom exit: {e}")
            return False
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics."""
        stats = {
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "success_rate": self.profitable_trades / self.total_trades if self.total_trades > 0 else 0.0,
            "total_profit": self.total_profit,
            "max_drawdown": self.max_drawdown,
            "active_positions": len(self.active_positions),
            "market_conditions": self.market_conditions,
            "phantom_statistics": self.detector.get_phantom_statistics()
        }
        
        return stats
    
    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current active positions."""
        return self.active_positions.copy()
    
    def reset_strategy(self):
        """Reset strategy state."""
        self.active_positions.clear()
        self.signal_history.clear()
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        
        for condition in self.market_conditions:
            self.market_conditions[condition] = 0
        
        logger.info("Strategy state reset")

def main():
    """Test the Phantom Band Navigator strategy."""
    import matplotlib.pyplot as plt

    # Initialize strategy
    navigator = PhantomBandNavigator()
    
    # Create test data
    np.random.seed(42)
    base_price = 50000.0
    ticks = [base_price]
    
    for i in range(200):
        # Simulate price movement with Phantom-like patterns
        if i % 30 == 0:  # Create Phantom-like flatness
            change = np.random.normal(0, 0.1)
        else:
            change = np.random.normal(0, 2.0)
        
        new_price = ticks[-1] + change
        ticks.append(new_price)
    
    # Test strategy
    print("Testing Phantom Band Navigator")
    print("=" * 50)
    
    available_balance = 10000.0
    
    for i in range(20, len(ticks)):
        window = ticks[i-20:i]
        current_price = ticks[i]
        
        # Generate signal
        signal = navigator.phantom_band_navigator("BTC", window, available_balance)
        
        if signal:
            print(f"Signal generated: {signal.signal_type} at ${current_price:.2f}")
            print(f"Confidence: {signal.confidence:.3f}")
            print(f"Position size: {signal.position_size:.4f}")
            print("-" * 30)

if __name__ == "__main__":
    main() 