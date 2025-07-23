#!/usr/bin/env python3
"""
🎯 Mode Integration System - Real Trading Logic Integration
==========================================================

This system integrates Default, Ghost, and Hybrid modes into the actual trading logic.
It applies mode-specific configurations to buy/sell decisions, position sizing, 
risk management, and entry/exit strategies for real capital trading.
"""

import yaml
import logging
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading mode enumeration."""
    DEFAULT = "default"
    GHOST = "ghost"
    HYBRID = "hybrid"

class TradeAction(Enum):
    """Trade action enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBUY = "rebuy"

@dataclass
class ModeConfig:
    """Mode-specific configuration."""
    mode: TradingMode
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    max_exposure_pct: float
    confidence_threshold: float
    ai_priority: float
    update_interval: float
    supported_symbols: List[str]
    orbital_shells: List[int]
    profit_target_usd: float
    max_daily_loss_pct: float
    win_rate_target: float

@dataclass
class TradingDecision:
    """Trading decision with mode-specific parameters."""
    action: TradeAction
    symbol: str
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    mode: TradingMode
    reasoning: str
    timestamp: float

class ModeIntegrationSystem:
    """Integrates trading modes into actual trading logic."""
    
    def __init__(self):
        self.current_mode = TradingMode.DEFAULT
        self.mode_configs = self._load_mode_configurations()
        self.portfolio_state = {
            'balance': 10000.0,
            'positions': {},
            'total_exposure': 0.0,
            'daily_profit': 0.0,
            'daily_loss': 0.0,
            'trades_today': 0
        }
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0
        }
        
        logger.info("🎯 Mode Integration System initialized")
    
    def _load_mode_configurations(self) -> Dict[TradingMode, ModeConfig]:
        """Load mode-specific configurations."""
        configs = {}
        
        # Default Mode Configuration
        configs[TradingMode.DEFAULT] = ModeConfig(
            mode=TradingMode.DEFAULT,
            position_size_pct=10.0,      # 10% position size
            stop_loss_pct=2.0,           # 2% stop loss
            take_profit_pct=3.0,         # 3% take profit
            max_exposure_pct=30.0,       # 30% max exposure
            confidence_threshold=0.7,    # 70% confidence required
            ai_priority=0.5,             # 50% AI priority
            update_interval=1.0,         # 1 second updates
            supported_symbols=["BTC/USDC", "USDC/BTC", "ETH/USDC", "USDC/ETH", "XRP/USDC", "USDC/XRP", "SOL/USDC", "USDC/SOL"],
            orbital_shells=[1, 2, 3, 4, 5, 6, 7, 8],  # All shells
            profit_target_usd=30.0,      # $30 per trade
            max_daily_loss_pct=1.0,      # 1% max daily loss
            win_rate_target=0.75         # 75% win rate target
        )
        
        # Ghost Mode Configuration
        configs[TradingMode.GHOST] = ModeConfig(
            mode=TradingMode.GHOST,
            position_size_pct=15.0,      # 15% position size
            stop_loss_pct=2.5,           # 2.5% stop loss
            take_profit_pct=4.0,         # 4% take profit
            max_exposure_pct=40.0,       # 40% max exposure
            confidence_threshold=0.65,   # 65% confidence required
            ai_priority=0.8,             # 80% AI priority
            update_interval=0.5,         # 0.5 second updates
            supported_symbols=["BTC/USDC", "USDC/BTC"],  # BTC/USDC only
            orbital_shells=[2, 6, 8],    # Medium-risk orbitals
            profit_target_usd=75.0,      # $75 per trade
            max_daily_loss_pct=2.0,      # 2% max daily loss
            win_rate_target=0.70         # 70% win rate target
        )
        
        # Hybrid Mode Configuration
        configs[TradingMode.HYBRID] = ModeConfig(
            mode=TradingMode.HYBRID,
            position_size_pct=30.5,      # 30.5% position size (12% base × 1.73 quantum × 1.47 consciousness)
            stop_loss_pct=2.33,          # 2.33% quantum stop loss
            take_profit_pct=4.47,        # 4.47% hybrid take profit
            max_exposure_pct=85.0,       # 85% max exposure (quantum consciousness)
            confidence_threshold=0.73,   # 73% quantum confidence required
            ai_priority=0.81,            # 81% quantum AI priority
            update_interval=0.33,        # 0.33 second quantum updates
            supported_symbols=["BTC/USDC", "USDC/BTC", "ETH/USDC", "USDC/ETH", "XRP/USDC", "USDC/XRP", "SOL/USDC", "USDC/SOL", "BTC/ETH", "ETH/BTC"],
            orbital_shells=[1, 3, 5, 7, 9],  # Hybrid orbitals
            profit_target_usd=147.7,     # $147.7 per hybrid trade
            max_daily_loss_pct=2.23,     # 2.23% max hybrid daily loss
            win_rate_target=0.81         # 81% hybrid win rate target
        )
        
        return configs
    
    def set_mode(self, mode: TradingMode) -> bool:
        """Set the current trading mode."""
        try:
            if mode not in self.mode_configs:
                logger.error(f"❌ Invalid trading mode: {mode}")
                return False
            
            self.current_mode = mode
            config = self.mode_configs[mode]
            
            logger.info(f"🎯 Trading mode set to: {mode.value.upper()}")
            logger.info(f"   Position Size: {config.position_size_pct}%")
            logger.info(f"   Stop Loss: {config.stop_loss_pct}%")
            logger.info(f"   Take Profit: {config.take_profit_pct}%")
            logger.info(f"   Max Exposure: {config.max_exposure_pct}%")
            logger.info(f"   AI Priority: {config.ai_priority:.1%}")
            logger.info(f"   Update Interval: {config.update_interval}s")
            logger.info(f"   Supported Symbols: {len(config.supported_symbols)} pairs")
            logger.info(f"   Orbital Shells: {config.orbital_shells}")
            logger.info(f"   Profit Target: ${config.profit_target_usd}")
            logger.info(f"   Win Rate Target: {config.win_rate_target:.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set trading mode: {e}")
            return False
    
    def get_current_config(self) -> ModeConfig:
        """Get current mode configuration."""
        return self.mode_configs[self.current_mode]
    
    def generate_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradingDecision]:
        """Generate trading decision based on current mode."""
        try:
            config = self.get_current_config()
            
            # Check if symbol is supported in current mode
            symbol = market_data.get('symbol', 'BTC/USDC')
            if symbol not in config.supported_symbols:
                logger.debug(f"Symbol {symbol} not supported in {self.current_mode.value} mode")
                return None
            
            # Apply mode-specific analysis
            if self.current_mode == TradingMode.DEFAULT:
                return self._generate_default_decision(market_data, config)
            elif self.current_mode == TradingMode.GHOST:
                return self._generate_ghost_decision(market_data, config)
            elif self.current_mode == TradingMode.HYBRID:
                return self._generate_hybrid_decision(market_data, config)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating trading decision: {e}")
            return None
    
    def _generate_default_decision(self, market_data: Dict[str, Any], config: ModeConfig) -> Optional[TradingDecision]:
        """Generate Default Mode trading decision."""
        try:
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            sentiment = market_data.get('sentiment', 0.5)
            
            # Default Mode Logic: Conservative, balanced approach
            confidence = 0.5
            action = TradeAction.HOLD
            reasoning = "Default mode: Waiting for clear signals"
            
            # Buy conditions (conservative)
            if (rsi < 30 and macd > 0 and sentiment > 0.6 and 
                self._can_open_position(config)):
                action = TradeAction.BUY
                confidence = 0.75
                reasoning = "Default mode: Oversold conditions with positive momentum"
            
            # Sell conditions (conservative)
            elif (rsi > 70 and macd < 0 and sentiment < 0.4 and 
                  self._has_position(market_data.get('symbol', 'BTC/USDC'))):
                action = TradeAction.SELL
                confidence = 0.70
                reasoning = "Default mode: Overbought conditions with negative momentum"
            
            # Check confidence threshold
            if confidence < config.confidence_threshold:
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(price, config)
            
            # Calculate entry/exit points
            entry_price = price
            stop_loss = entry_price * (1 - config.stop_loss_pct / 100)
            take_profit = entry_price * (1 + config.take_profit_pct / 100)
            
            return TradingDecision(
                action=action,
                symbol=market_data.get('symbol', 'BTC/USDC'),
                entry_price=entry_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                mode=self.current_mode,
                reasoning=reasoning,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Error in Default Mode decision: {e}")
            return None
    
    def _generate_ghost_decision(self, market_data: Dict[str, Any], config: ModeConfig) -> Optional[TradingDecision]:
        """Generate Ghost Mode trading decision."""
        try:
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            sentiment = market_data.get('sentiment', 0.5)
            
            # Ghost Mode Logic: BTC/USDC focused, medium risk
            confidence = 0.5
            action = TradeAction.HOLD
            reasoning = "Ghost mode: Monitoring BTC/USDC for opportunities"
            
            # Ghost Mode Buy conditions (medium risk)
            if (rsi < 35 and macd > 0 and sentiment > 0.55 and 
                self._can_open_position(config)):
                action = TradeAction.BUY
                confidence = 0.70
                reasoning = "Ghost mode: BTC/USDC oversold with momentum"
            
            # Ghost Mode Sell conditions (medium risk)
            elif (rsi > 65 and macd < 0 and sentiment < 0.45 and 
                  self._has_position(market_data.get('symbol', 'BTC/USDC'))):
                action = TradeAction.SELL
                confidence = 0.65
                reasoning = "Ghost mode: BTC/USDC overbought with reversal"
            
            # Ghost Mode Rebuy conditions (aggressive)
            elif (rsi < 40 and macd > 0 and sentiment > 0.6 and 
                  self._has_position(market_data.get('symbol', 'BTC/USDC'))):
                action = TradeAction.REBUY
                confidence = 0.60
                reasoning = "Ghost mode: BTC/USDC re-entry opportunity"
            
            # Check confidence threshold
            if confidence < config.confidence_threshold:
                return None
            
            # Calculate position size (larger for Ghost Mode)
            position_size = self._calculate_position_size(price, config)
            
            # Calculate entry/exit points
            entry_price = price
            stop_loss = entry_price * (1 - config.stop_loss_pct / 100)
            take_profit = entry_price * (1 + config.take_profit_pct / 100)
            
            return TradingDecision(
                action=action,
                symbol=market_data.get('symbol', 'BTC/USDC'),
                entry_price=entry_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                mode=self.current_mode,
                reasoning=reasoning,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Error in Ghost Mode decision: {e}")
            return None
    
    def _generate_hybrid_decision(self, market_data: Dict[str, Any], config: ModeConfig) -> Optional[TradingDecision]:
        """Generate Hybrid Mode trading decision."""
        try:
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            sentiment = market_data.get('sentiment', 0.5)
            
            # Hybrid Mode Logic: Quantum consciousness, multi-dimensional
            confidence = 0.5
            action = TradeAction.HOLD
            reasoning = "Hybrid mode: Quantum consciousness analyzing market"
            
            # Apply quantum consciousness boost
            quantum_boost = 1.47  # 47% consciousness boost
            dimensional_boost = 1.33  # 33% dimensional boost
            
            # Hybrid Mode Buy conditions (quantum consciousness)
            if (rsi < 40 and macd > 0 and sentiment > 0.5 and 
                self._can_open_position(config)):
                action = TradeAction.BUY
                confidence = 0.73 * quantum_boost  # Apply quantum boost
                reasoning = "Hybrid mode: Quantum consciousness detects buying opportunity"
            
            # Hybrid Mode Sell conditions (dimensional analysis)
            elif (rsi > 60 and macd < 0 and sentiment < 0.5 and 
                  self._has_position(market_data.get('symbol', 'BTC/USDC'))):
                action = TradeAction.SELL
                confidence = 0.68 * dimensional_boost  # Apply dimensional boost
                reasoning = "Hybrid mode: Dimensional analysis suggests selling"
            
            # Hybrid Mode Rebuy conditions (parallel universe)
            elif (rsi < 45 and macd > 0 and sentiment > 0.55 and 
                  self._has_position(market_data.get('symbol', 'BTC/USDC'))):
                action = TradeAction.REBUY
                confidence = 0.65 * quantum_boost * dimensional_boost  # Apply both boosts
                reasoning = "Hybrid mode: Parallel universe analysis suggests re-entry"
            
            # Check quantum confidence threshold
            if confidence < config.confidence_threshold:
                return None
            
            # Calculate quantum position size
            position_size = self._calculate_position_size(price, config)
            
            # Calculate quantum entry/exit points
            entry_price = price
            stop_loss = entry_price * (1 - config.stop_loss_pct / 100)
            take_profit = entry_price * (1 + config.take_profit_pct / 100)
            
            return TradingDecision(
                action=action,
                symbol=market_data.get('symbol', 'BTC/USDC'),
                entry_price=entry_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                mode=self.current_mode,
                reasoning=reasoning,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Error in Hybrid Mode decision: {e}")
            return None
    
    def _calculate_position_size(self, price: float, config: ModeConfig) -> float:
        """Calculate position size based on mode configuration."""
        try:
            # Base position size from portfolio
            base_amount = self.portfolio_state['balance'] * (config.position_size_pct / 100)
            
            # Apply mode-specific adjustments
            if self.current_mode == TradingMode.DEFAULT:
                # Default: Conservative position sizing
                position_size = base_amount / price
            elif self.current_mode == TradingMode.GHOST:
                # Ghost: Medium risk position sizing
                position_size = base_amount * 1.5 / price  # 50% larger
            elif self.current_mode == TradingMode.HYBRID:
                # Hybrid: Quantum position sizing
                quantum_multiplier = 1.73  # 73% quantum boost
                consciousness_multiplier = 1.47  # 47% consciousness boost
                position_size = base_amount * quantum_multiplier * consciousness_multiplier / price
            
            # Ensure minimum position size
            min_position = 0.001  # Minimum 0.001 units
            return max(position_size, min_position)
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.001  # Return minimum position size
    
    def _can_open_position(self, config: ModeConfig) -> bool:
        """Check if we can open a new position."""
        try:
            # Check total exposure
            if self.portfolio_state['total_exposure'] >= config.max_exposure_pct:
                return False
            
            # Check daily loss limit
            if self.portfolio_state['daily_loss'] >= config.max_daily_loss_pct:
                return False
            
            # Check balance
            if self.portfolio_state['balance'] <= 100:  # Minimum $100 balance
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking position availability: {e}")
            return False
    
    def _has_position(self, symbol: str) -> bool:
        """Check if we have an existing position for the symbol."""
        return symbol in self.portfolio_state['positions'] and self.portfolio_state['positions'][symbol] > 0
    
    def execute_trade(self, decision: TradingDecision) -> bool:
        """Execute a trading decision."""
        try:
            if not decision:
                return False
            
            config = self.get_current_config()
            symbol = decision.symbol
            action = decision.action
            
            logger.info(f"💰 Executing {self.current_mode.value.upper()} trade: {action.value} {symbol}")
            logger.info(f"   Entry Price: ${decision.entry_price:.4f}")
            logger.info(f"   Position Size: {decision.position_size:.6f}")
            logger.info(f"   Stop Loss: ${decision.stop_loss:.4f}")
            logger.info(f"   Take Profit: ${decision.take_profit:.4f}")
            logger.info(f"   Confidence: {decision.confidence:.1%}")
            logger.info(f"   Reasoning: {decision.reasoning}")
            
            # Simulate trade execution (replace with actual API call)
            if action == TradeAction.BUY:
                cost = decision.entry_price * decision.position_size
                if cost <= self.portfolio_state['balance']:
                    self.portfolio_state['balance'] -= cost
                    self.portfolio_state['positions'][symbol] = decision.position_size
                    self.portfolio_state['total_exposure'] += config.position_size_pct
                    self.performance_metrics['total_trades'] += 1
                    self.portfolio_state['trades_today'] += 1
                    logger.info(f"✅ {self.current_mode.value.upper()} BUY executed successfully")
                    return True
                else:
                    logger.warning(f"⚠️ Insufficient balance for {self.current_mode.value} BUY")
                    return False
            
            elif action == TradeAction.SELL:
                if self._has_position(symbol):
                    position_size = self.portfolio_state['positions'][symbol]
                    revenue = decision.entry_price * position_size
                    profit = revenue - (decision.entry_price * position_size)  # Simplified profit calculation
                    
                    self.portfolio_state['balance'] += revenue
                    self.portfolio_state['positions'][symbol] = 0
                    self.portfolio_state['total_exposure'] -= config.position_size_pct
                    self.portfolio_state['daily_profit'] += profit
                    
                    if profit > 0:
                        self.performance_metrics['winning_trades'] += 1
                    else:
                        self.performance_metrics['losing_trades'] += 1
                        self.portfolio_state['daily_loss'] += abs(profit)
                    
                    self.performance_metrics['total_trades'] += 1
                    self.portfolio_state['trades_today'] += 1
                    self.performance_metrics['total_profit'] += profit
                    
                    logger.info(f"✅ {self.current_mode.value.upper()} SELL executed successfully")
                    logger.info(f"   Profit: ${profit:.2f}")
                    return True
                else:
                    logger.warning(f"⚠️ No position to sell for {symbol}")
                    return False
            
            elif action == TradeAction.REBUY:
                # Rebuy logic (add to existing position)
                cost = decision.entry_price * decision.position_size
                if cost <= self.portfolio_state['balance']:
                    self.portfolio_state['balance'] -= cost
                    self.portfolio_state['positions'][symbol] += decision.position_size
                    self.performance_metrics['total_trades'] += 1
                    self.portfolio_state['trades_today'] += 1
                    logger.info(f"✅ {self.current_mode.value.upper()} REBUY executed successfully")
                    return True
                else:
                    logger.warning(f"⚠️ Insufficient balance for {self.current_mode.value} REBUY")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for current mode."""
        try:
            config = self.get_current_config()
            
            total_trades = self.performance_metrics['total_trades']
            winning_trades = self.performance_metrics['winning_trades']
            losing_trades = self.performance_metrics['losing_trades']
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_profit_per_trade = self.performance_metrics['total_profit'] / total_trades if total_trades > 0 else 0
            
            return {
                "mode": self.current_mode.value,
                "portfolio_balance": self.portfolio_state['balance'],
                "total_exposure": self.portfolio_state['total_exposure'],
                "daily_profit": self.portfolio_state['daily_profit'],
                "daily_loss": self.portfolio_state['daily_loss'],
                "trades_today": self.portfolio_state['trades_today'],
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_profit": self.performance_metrics['total_profit'],
                "avg_profit_per_trade": avg_profit_per_trade,
                "max_drawdown": self.performance_metrics['max_drawdown'],
                "target_profit_per_trade": config.profit_target_usd,
                "target_win_rate": config.win_rate_target,
                "max_daily_loss_pct": config.max_daily_loss_pct
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {}

# Global Mode Integration System instance
mode_integration_system = ModeIntegrationSystem() 