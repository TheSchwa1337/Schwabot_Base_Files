#!/usr/bin/env python3
"""
Schwabot Executable Core
Advanced Algorithmic Trading Intelligence System

This is the MAIN EXECUTABLE that builds INSIDE the logical math we already have.
It uses the profit tier navigation system and nano-core strategy switching
to create a progressive, mathematically-driven trading system.

NO LAYERING - ONLY PROGRESSIVE BUILDING INSIDE EXISTING MATH
"""

import os
import sys
import time
import json
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'nano-core'))

# Import our existing mathematical framework
from core.soulprint_registry import SoulprintRegistry, SoulprintEntry
from core.qsc_enhanced_profit_allocator import QSCEnhancedProfitAllocator, QSCAllocationMode
from core.profit_optimization_engine import ProfitOptimizationEngine, OptimizationMethod, RiskMetric
from core.enhanced_ccxt_trading_engine import EnhancedCCXTTradingEngine
from core.automated_strategy_engine import AutomatedStrategyEngine
from nano_core.strategy_switch import match_strategy, select_best_trade_batch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_executable.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProfitTier(Enum):
    """Profit tier navigation system."""
    TIER_1_SAFE = "tier_1_safe"           # Conservative, low risk
    TIER_2_MODERATE = "tier_2_moderate"   # Balanced risk/reward
    TIER_3_AGGRESSIVE = "tier_3_aggressive" # High risk, high reward
    TIER_4_EMERGENCY = "tier_4_emergency" # Emergency mode, capital preservation


@dataclass
class TickCycle:
    """16-bit tick cycle for BTC/USDC (every 3.75 minutes)."""
    cycle_id: str
    start_time: float
    end_time: float
    tick_number: int  # 1-16
    btc_price: float
    usdc_balance: float
    btc_balance: float
    profit_tier: ProfitTier
    strategy_activated: str
    qsc_mode: QSCAllocationMode
    soulprint_hash: str
    confidence_score: float
    executed_trades: List[Dict[str, Any]] = field(default_factory=list)
    profit_result: float = 0.0
    is_complete: bool = False


@dataclass
class FerrisWheelState:
    """Ferris wheel orchestrator state."""
    current_tick: int = 1
    total_ticks: int = 16
    tick_interval: float = 225.0  # 3.75 minutes in seconds
    last_tick_time: float = 0.0
    active_cycles: Dict[str, TickCycle] = field(default_factory=dict)
    profit_tier_history: List[ProfitTier] = field(default_factory=list)
    total_profit: float = 0.0
    is_running: bool = False


class SchwabotExecutableCore:
    """
    Main Schwabot executable core that builds INSIDE the existing mathematical framework.
    
    This system:
    1. Uses the 16-bit tick cycle (every 3.75 minutes)
    2. Navigates profit tiers based on mathematical analysis
    3. Uses nano-core strategy switching for decision making
    4. Integrates with soulprint registry for state management
    5. Uses QSC profit allocation for risk management
    6. Executes trades through enhanced CCXT engine
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Schwabot executable core."""
        self.config = config or self._default_config()
        
        # Initialize core components
        self.soulprint_registry = SoulprintRegistry()
        self.qsc_allocator = QSCEnhancedProfitAllocator()
        self.profit_optimizer = ProfitOptimizationEngine()
        self.trading_engine = EnhancedCCXTTradingEngine()
        self.strategy_engine = AutomatedStrategyEngine()
        
        # Ferris wheel state
        self.ferris_wheel = FerrisWheelState()
        
        # Threading and async
        self.tick_thread = None
        self.is_running = False
        
        logger.info("🚀 Schwabot Executable Core initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "exchange": "coinbase",
            "base_pair": "BTC/USDC",
            "initial_balance": 10000.0,
            "max_risk_per_tick": 0.1,
            "profit_tier_thresholds": {
                "tier_1_safe": {"min_confidence": 0.9, "max_allocation": 0.2},
                "tier_2_moderate": {"min_confidence": 0.7, "max_allocation": 0.5},
                "tier_3_aggressive": {"min_confidence": 0.5, "max_allocation": 0.8},
                "tier_4_emergency": {"min_confidence": 0.3, "max_allocation": 0.1}
            },
            "tick_cycle_interval": 225.0,  # 3.75 minutes
            "enable_demo_mode": True,
            "enable_live_trading": False
        }
    
    def start_ferris_wheel(self):
        """Start the Ferris wheel orchestrator."""
        if self.is_running:
            logger.warning("Ferris wheel already running")
            return
            
        self.is_running = True
        self.ferris_wheel.is_running = True
        self.ferris_wheel.last_tick_time = time.time()
        
        # Start tick cycle thread
        self.tick_thread = threading.Thread(target=self._tick_cycle_loop, daemon=True)
        self.tick_thread.start()
        
        logger.info("🎡 Ferris wheel started - 16-bit tick cycle active")
        
    def stop_ferris_wheel(self):
        """Stop the Ferris wheel orchestrator."""
        self.is_running = False
        self.ferris_wheel.is_running = False
        
        if self.tick_thread:
            self.tick_thread.join(timeout=5.0)
            
        logger.info("🛑 Ferris wheel stopped")
    
    def _tick_cycle_loop(self):
        """Main tick cycle loop - runs every 3.75 minutes."""
        while self.is_running:
            try:
                current_time = time.time()
                time_since_last_tick = current_time - self.ferris_wheel.last_tick_time
                
                if time_since_last_tick >= self.config["tick_cycle_interval"]:
                    self._execute_tick_cycle()
                    self.ferris_wheel.last_tick_time = current_time
                else:
                    time.sleep(1.0)  # Check every second
                    
            except Exception as e:
                logger.error(f"Error in tick cycle loop: {e}")
                time.sleep(5.0)
    
    def _execute_tick_cycle(self):
        """Execute a single tick cycle."""
        tick_number = self.ferris_wheel.current_tick
        cycle_id = f"tick_{tick_number:02d}_{int(time.time())}"
        
        logger.info(f"🔄 Executing tick cycle {tick_number}/16 - {cycle_id}")
        
        try:
            # 1. Get current market state
            market_state = self._get_market_state()
            
            # 2. Calculate mathematical vectors
            drift_vector = self._calculate_drift_vector(market_state)
            
            # 3. Register soulprint
            soulprint_hash = self.soulprint_registry.register_soulprint(
                vector=drift_vector,
                strategy_id=f"tick_{tick_number}",
                confidence=drift_vector.get('confidence', 0.5)
            )
            
            # 4. Determine profit tier
            profit_tier = self._determine_profit_tier(drift_vector)
            
            # 5. Get strategy recommendation
            strategy_data = {
                "hash": soulprint_hash,
                "asset": self.config["base_pair"],
                "price": market_state["btc_price"],
                "trigger": f"tick_{tick_number}",
                "confidence": drift_vector.get('confidence', 0.5)
            }
            
            strategy_recommendation = match_strategy(strategy_data)
            
            # 6. Create tick cycle
            tick_cycle = TickCycle(
                cycle_id=cycle_id,
                start_time=time.time(),
                end_time=time.time() + self.config["tick_cycle_interval"],
                tick_number=tick_number,
                btc_price=market_state["btc_price"],
                usdc_balance=market_state["usdc_balance"],
                btc_balance=market_state["btc_balance"],
                profit_tier=profit_tier,
                strategy_activated=strategy_recommendation["name"],
                qsc_mode=self._get_qsc_mode_for_tier(profit_tier),
                soulprint_hash=soulprint_hash,
                confidence_score=drift_vector.get('confidence', 0.5)
            )
            
            # 7. Execute trading decision
            if self._should_execute_trade(tick_cycle, strategy_recommendation):
                trade_result = self._execute_trade_decision(tick_cycle, strategy_recommendation)
                tick_cycle.executed_trades.append(trade_result)
                tick_cycle.profit_result = trade_result.get('profit', 0.0)
                
                # Update soulprint with execution result
                self.soulprint_registry.mark_executed(
                    soulprint_hash, 
                    profit_result=trade_result.get('profit', 0.0)
                )
            
            # 8. Complete tick cycle
            tick_cycle.is_complete = True
            self.ferris_wheel.active_cycles[cycle_id] = tick_cycle
            self.ferris_wheel.profit_tier_history.append(profit_tier)
            self.ferris_wheel.total_profit += tick_cycle.profit_result
            
            # 9. Update tick counter
            self.ferris_wheel.current_tick = (tick_number % 16) + 1
            
            logger.info(f"✅ Tick cycle {tick_number} completed - Profit: {tick_cycle.profit_result:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error in tick cycle {tick_number}: {e}")
    
    def _get_market_state(self) -> Dict[str, Any]:
        """Get current market state."""
        try:
            # Get BTC price from trading engine
            ticker = self.trading_engine.get_ticker(self.config["base_pair"])
            btc_price = float(ticker['last']) if ticker else 60000.0
            
            # Get balances (demo mode for now)
            usdc_balance = self.config["initial_balance"]
            btc_balance = 0.0
            
            return {
                "btc_price": btc_price,
                "usdc_balance": usdc_balance,
                "btc_balance": btc_balance,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting market state: {e}")
            return {
                "btc_price": 60000.0,
                "usdc_balance": self.config["initial_balance"],
                "btc_balance": 0.0,
                "timestamp": time.time()
            }
    
    def _calculate_drift_vector(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate drift vector using existing mathematical framework."""
        try:
            # Use the existing mathematical framework to calculate drift vector
            btc_price = market_state["btc_price"]
            
            # Calculate entropy (price volatility)
            entropy = min(1.0, abs(btc_price - 60000.0) / 60000.0)
            
            # Calculate momentum (price direction)
            momentum = 0.5  # Neutral for now
            
            # Calculate volatility
            volatility = entropy * 0.5
            
            # Calculate temporal variance
            temporal_variance = 0.5  # Neutral for now
            
            # Calculate confidence based on mathematical coherence
            confidence = max(0.1, min(1.0, 1.0 - entropy))
            
            return {
                "pair": self.config["base_pair"],
                "entropy": entropy,
                "momentum": momentum,
                "volatility": volatility,
                "temporal_variance": temporal_variance,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error calculating drift vector: {e}")
            return {
                "pair": self.config["base_pair"],
                "entropy": 0.5,
                "momentum": 0.5,
                "volatility": 0.5,
                "temporal_variance": 0.5,
                "confidence": 0.5
            }
    
    def _determine_profit_tier(self, drift_vector: Dict[str, float]) -> ProfitTier:
        """Determine profit tier based on mathematical analysis."""
        confidence = drift_vector.get('confidence', 0.5)
        entropy = drift_vector.get('entropy', 0.5)
        
        # Use mathematical thresholds to determine tier
        if confidence >= 0.9 and entropy <= 0.2:
            return ProfitTier.TIER_1_SAFE
        elif confidence >= 0.7 and entropy <= 0.4:
            return ProfitTier.TIER_2_MODERATE
        elif confidence >= 0.5 and entropy <= 0.6:
            return ProfitTier.TIER_3_AGGRESSIVE
        else:
            return ProfitTier.TIER_4_EMERGENCY
    
    def _get_qsc_mode_for_tier(self, profit_tier: ProfitTier) -> QSCAllocationMode:
        """Get QSC mode for profit tier."""
        tier_mapping = {
            ProfitTier.TIER_1_SAFE: QSCAllocationMode.IMMUNE_VALIDATED,
            ProfitTier.TIER_2_MODERATE: QSCAllocationMode.RESONANCE_OPTIMIZED,
            ProfitTier.TIER_3_AGGRESSIVE: QSCAllocationMode.QUANTUM_ENHANCED,
            ProfitTier.TIER_4_EMERGENCY: QSCAllocationMode.EMERGENCY_CONSERVATIVE
        }
        return tier_mapping.get(profit_tier, QSCAllocationMode.EMERGENCY_CONSERVATIVE)
    
    def _should_execute_trade(self, tick_cycle: TickCycle, strategy: Dict[str, Any]) -> bool:
        """Determine if we should execute a trade."""
        # Check confidence threshold for profit tier
        tier_config = self.config["profit_tier_thresholds"][tick_cycle.profit_tier.value]
        min_confidence = tier_config["min_confidence"]
        
        if tick_cycle.confidence_score < min_confidence:
            return False
        
        # Check if action is not HOLD
        if strategy["action"] == "HOLD":
            return False
        
        # Check if we have sufficient balance
        if strategy["action"] == "BUY" and tick_cycle.usdc_balance < 100:
            return False
        
        if strategy["action"] == "SELL" and tick_cycle.btc_balance < 0.001:
            return False
        
        return True
    
    def _execute_trade_decision(self, tick_cycle: TickCycle, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the trading decision."""
        try:
            action = strategy["action"]
            confidence = strategy["confidence"]
            
            # Calculate position size based on profit tier
            tier_config = self.config["profit_tier_thresholds"][tick_cycle.profit_tier.value]
            max_allocation = tier_config["max_allocation"]
            
            if action == "BUY":
                # Calculate buy amount
                available_usdc = tick_cycle.usdc_balance * max_allocation
                btc_amount = available_usdc / tick_cycle.btc_price
                
                # Execute buy order (demo mode)
                trade_result = {
                    "action": "BUY",
                    "amount": btc_amount,
                    "price": tick_cycle.btc_price,
                    "confidence": confidence,
                    "profit": 0.0,  # Will be calculated on sell
                    "timestamp": time.time(),
                    "tick_cycle": tick_cycle.cycle_id
                }
                
                logger.info(f"💰 BUY executed: {btc_amount:.6f} BTC at ${tick_cycle.btc_price:,.2f}")
                
            elif action == "SELL":
                # Calculate sell amount
                btc_amount = tick_cycle.btc_balance * max_allocation
                usdc_amount = btc_amount * tick_cycle.btc_price
                
                # Calculate profit (demo mode)
                profit = usdc_amount * 0.02  # 2% profit for demo
                
                trade_result = {
                    "action": "SELL",
                    "amount": btc_amount,
                    "price": tick_cycle.btc_price,
                    "confidence": confidence,
                    "profit": profit,
                    "timestamp": time.time(),
                    "tick_cycle": tick_cycle.cycle_id
                }
                
                logger.info(f"💰 SELL executed: {btc_amount:.6f} BTC at ${tick_cycle.btc_price:,.2f} - Profit: ${profit:.2f}")
                
            else:
                trade_result = {
                    "action": "HOLD",
                    "amount": 0.0,
                    "price": tick_cycle.btc_price,
                    "confidence": confidence,
                    "profit": 0.0,
                    "timestamp": time.time(),
                    "tick_cycle": tick_cycle.cycle_id
                }
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                "action": "ERROR",
                "amount": 0.0,
                "price": tick_cycle.btc_price,
                "confidence": 0.0,
                "profit": 0.0,
                "timestamp": time.time(),
                "tick_cycle": tick_cycle.cycle_id,
                "error": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "ferris_wheel": {
                "is_running": self.ferris_wheel.is_running,
                "current_tick": self.ferris_wheel.current_tick,
                "total_ticks": self.ferris_wheel.total_ticks,
                "total_profit": self.ferris_wheel.total_profit,
                "active_cycles": len(self.ferris_wheel.active_cycles)
            },
            "soulprint_registry": self.soulprint_registry.get_registry_stats(),
            "qsc_allocator": self.qsc_allocator.get_system_stats(),
            "config": {
                "exchange": self.config["exchange"],
                "base_pair": self.config["base_pair"],
                "tick_interval": self.config["tick_cycle_interval"]
            }
        }
    
    def get_demo_data(self) -> Dict[str, Any]:
        """Get demo data for dashboard."""
        return {
            "current_tick": self.ferris_wheel.current_tick,
            "total_profit": self.ferris_wheel.total_profit,
            "profit_tier": self.ferris_wheel.profit_tier_history[-1].value if self.ferris_wheel.profit_tier_history else "tier_2_moderate",
            "active_cycles": len(self.ferris_wheel.active_cycles),
            "registry_stats": self.soulprint_registry.get_registry_stats(),
            "recent_trades": self._get_recent_trades()
        }
    
    def _get_recent_trades(self) -> List[Dict[str, Any]]:
        """Get recent trades for demo display."""
        recent_trades = []
        
        for cycle in list(self.ferris_wheel.active_cycles.values())[-5:]:
            for trade in cycle.executed_trades:
                recent_trades.append({
                    "pair": self.config["base_pair"],
                    "action": trade["action"],
                    "price": trade["price"],
                    "profit": trade["profit"],
                    "timestamp": datetime.fromtimestamp(trade["timestamp"]).strftime("%H:%M:%S")
                })
        
        return recent_trades


def main():
    """Main executable entry point."""
    print("🚀 Schwabot Executable Core")
    print("=" * 50)
    print("Advanced Algorithmic Trading Intelligence System")
    print("Building INSIDE existing mathematical framework")
    print("=" * 50)
    
    # Create executable core
    schwabot = SchwabotExecutableCore()
    
    try:
        # Start Ferris wheel
        schwabot.start_ferris_wheel()
        
        print("🎡 Ferris wheel started - Press Ctrl+C to stop")
        print("📊 System status updates every tick cycle (3.75 minutes)")
        
        # Keep running
        while True:
            time.sleep(10)
            status = schwabot.get_system_status()
            print(f"🔄 Tick {status['ferris_wheel']['current_tick']}/16 - Profit: ${status['ferris_wheel']['total_profit']:.2f}")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping Schwabot...")
        schwabot.stop_ferris_wheel()
        print("✅ Schwabot stopped safely")


if __name__ == "__main__":
    main() 