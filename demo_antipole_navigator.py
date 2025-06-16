#!/usr/bin/env python3
"""
Anti-Pole Profit Navigator Demo
===============================

Demonstrates the complete Anti-Pole Theory system with simulated BTC market data.
Shows real-time profit navigation, thermal monitoring, and 4D visualization.

Usage:
    python demo_antipole_navigator.py
"""

import asyncio
import logging
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from core.profit_navigator import AntiPoleProfitNavigator
from core.antipole import AntiPoleConfig

class BTCMarketSimulator:
    """Simulates realistic BTC market data for demonstration"""
    
    def __init__(self, initial_price: float = 45000.0):
        self.current_price = initial_price
        self.base_volume = 1000000  # Base volume
        self.trend = 0.0  # Current trend direction
        self.volatility = 0.02  # Price volatility
        self.tick_count = 0
        
        # Market regime simulation
        self.regime = "neutral"  # "bull", "bear", "neutral", "volatile"
        self.regime_timer = 0
        
    def next_tick(self) -> Dict[str, float]:
        """Generate next market tick"""
        self.tick_count += 1
        
        # Update regime periodically
        if self.tick_count % 100 == 0:
            self._update_regime()
        
        # Generate price movement
        price_change = self._generate_price_change()
        self.current_price += price_change
        
        # Ensure price doesn't go negative
        self.current_price = max(self.current_price, 1000.0)
        
        # Generate volume
        volume = self._generate_volume()
        
        return {
            'price': self.current_price,
            'volume': volume,
            'regime': self.regime,
            'trend': self.trend
        }
    
    def _update_regime(self):
        """Update market regime"""
        regimes = ["bull", "bear", "neutral", "volatile"]
        weights = [0.25, 0.20, 0.40, 0.15]  # Neutral is most common
        
        self.regime = np.random.choice(regimes, p=weights)
        
        # Set regime parameters
        if self.regime == "bull":
            self.trend = 0.001
            self.volatility = 0.015
        elif self.regime == "bear":
            self.trend = -0.001
            self.volatility = 0.020
        elif self.regime == "volatile":
            self.trend = 0.0
            self.volatility = 0.035
        else:  # neutral
            self.trend = 0.0
            self.volatility = 0.012
        
        logger.info(f"📊 Market regime changed to: {self.regime.upper()}")
    
    def _generate_price_change(self) -> float:
        """Generate realistic price change"""
        # Base trend movement
        trend_component = self.trend * self.current_price
        
        # Random walk component
        random_component = np.random.normal(0, self.volatility * self.current_price)
        
        # Momentum component (sometimes prices continue in same direction)
        momentum = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        momentum_component = momentum * 0.0005 * self.current_price
        
        return trend_component + random_component + momentum_component
    
    def _generate_volume(self) -> float:
        """Generate realistic volume"""
        # Base volume with random variation
        volume_multiplier = np.random.lognormal(0, 0.3)  # Log-normal distribution
        
        # Higher volume during volatile periods
        if self.regime == "volatile":
            volume_multiplier *= 1.5
        elif self.regime in ["bull", "bear"]:
            volume_multiplier *= 1.2
        
        return self.base_volume * volume_multiplier

class DemoController:
    """Controls the demonstration flow"""
    
    def __init__(self):
        # Create Anti-Pole configuration
        config = AntiPoleConfig(
            mu_c=0.020,           # Slightly higher threshold for demo
            sigma_c=0.008,
            tau_icap=0.60,        # Lower threshold for more activity
            epsilon=1e-9,
            hash_window=128,      # Smaller window for faster response
            thermal_decay=0.90,
            profit_amplification=1.3,
            recursion_depth=6
        )
        
        # Initialize systems
        self.navigator = AntiPoleProfitNavigator(
            initial_balance=50000.0,  # Start with $50K
            max_position_size=0.20,   # Max 20% position size
            config=config
        )
        
        self.market_sim = BTCMarketSimulator(initial_price=43500.0)
        
        # Demo state
        self.running = False
        self.tick_count = 0
        self.demo_duration = 300  # 5 minutes demo
        self.last_status_report = time.time()
        
    async def run_demo(self):
        """Run the complete demonstration"""
        logger.info("🚀 Starting Anti-Pole Profit Navigator Demo")
        logger.info("="*60)
        
        # Start the visualization server
        await self.navigator.start_visualization_server()
        logger.info("🖥️  Tesseract visualization available at ws://localhost:8765")
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running and (time.time() - start_time) < self.demo_duration:
                # Generate market tick
                market_data = self.market_sim.next_tick()
                
                # Process through Anti-Pole Navigator
                tick_report = await self.navigator.process_market_tick(
                    btc_price=market_data['price'],
                    volume=market_data['volume']
                )
                
                # Log interesting events
                self._log_demo_events(tick_report, market_data)
                
                # Periodic status report
                if time.time() - self.last_status_report > 30:  # Every 30 seconds
                    self._print_status_report(tick_report, market_data)
                    self.last_status_report = time.time()
                
                self.tick_count += 1
                
                # Sleep for realistic tick rate (1 second)
                await asyncio.sleep(1.0)
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            self.running = False
            await self.navigator.stop_visualization_server()
            self._print_final_report()
    
    def _log_demo_events(self, tick_report: Dict[str, Any], market_data: Dict[str, float]):
        """Log interesting events during demo"""
        antipole = tick_report['antipole_state']
        thermal = tick_report['thermal_state']
        
        # Log Anti-Pole ready states
        if antipole['is_ready'] and antipole['profit_tier']:
            logger.info(f"🎯 PROFIT OPPORTUNITY: {antipole['profit_tier']} tier detected! "
                       f"ICAP: {antipole['icap_probability']:.3f} at ${market_data['price']:,.2f}")
        
        # Log thermal events
        if thermal['cooldown_active']:
            logger.warning(f"🌡️ THERMAL COOLDOWN: System cooling down ({thermal['state']})")
        
        # Log trades
        for trade in tick_report['executed_trades']:
            if trade['type'] == 'BUY_EXECUTED':
                logger.info(f"💰 BOUGHT: {trade['size']:.6f} BTC @ ${trade['price']:,.2f} "
                           f"(Confidence: {trade['confidence']:.2f})")
            elif trade['type'] == 'SELL_EXECUTED':
                pnl_symbol = "📈" if trade['trade_pnl'] > 0 else "📉"
                logger.info(f"{pnl_symbol} SOLD: {trade['size']:.6f} BTC @ ${trade['price']:,.2f} "
                           f"P&L: ${trade['trade_pnl']:,.2f}")
        
        # Log high-value opportunities
        for opp in tick_report['opportunities']:
            if opp['profit_tier'] in ['PLATINUM', 'GOLD'] and opp['confidence'] > 0.8:
                logger.info(f"✨ HIGH-VALUE OPPORTUNITY: {opp['opportunity_type']} {opp['profit_tier']} "
                           f"({opp['confidence']*100:.0f}% confidence)")
    
    def _print_status_report(self, tick_report: Dict[str, Any], market_data: Dict[str, float]):
        """Print periodic status report"""
        antipole = tick_report['antipole_state']
        thermal = tick_report['thermal_state']
        portfolio = tick_report['portfolio_state']
        
        print("\n" + "="*80)
        print(f"📊 DEMO STATUS REPORT - Tick #{self.tick_count}")
        print("="*80)
        
        # Market Data
        print(f"🏪 MARKET DATA:")
        print(f"   BTC Price: ${market_data['price']:,.2f}")
        print(f"   Volume: {market_data['volume']:,.0f}")
        print(f"   Regime: {market_data['regime'].upper()}")
        
        # Anti-Pole State
        ready_status = "🔥 READY" if antipole['is_ready'] else "⏳ Waiting"
        phase_status = "🔒 Locked" if antipole['phase_lock'] else "🔄 Floating"
        print(f"\n🔺 ANTI-POLE STATE:")
        print(f"   Status: {ready_status}")
        print(f"   ICAP: {antipole['icap_probability']:.3f}")
        print(f"   Profit Tier: {antipole['profit_tier'] or 'None'}")
        print(f"   Phase Lock: {phase_status}")
        print(f"   Hash Entropy: {antipole['hash_entropy']:.3f}")
        print(f"   Drift: {antipole['delta_psi_bar']:.6f}")
        
        # Thermal State
        thermal_emoji = {"COLD": "❄️", "COOL": "🧊", "WARM": "🌡️", 
                        "HOT": "🔥", "CRITICAL": "🚨", "EMERGENCY": "💥"}.get(thermal['state'], "🌡️")
        print(f"\n🌡️ THERMAL STATE:")
        print(f"   State: {thermal_emoji} {thermal['state']}")
        print(f"   Load: {thermal['thermal_load']*100:.1f}%")
        print(f"   Safe to Trade: {'✅' if thermal['safe_to_trade'] else '❌'}")
        
        # Portfolio
        total_return = ((portfolio['total_value'] - 50000) / 50000) * 100
        print(f"\n💰 PORTFOLIO:")
        print(f"   Total Value: ${portfolio['total_value']:,.2f}")
        print(f"   Cash: ${portfolio['cash_balance']:,.2f}")
        print(f"   BTC Position: {portfolio['btc_position']:.6f}")
        print(f"   Unrealized P&L: ${portfolio['unrealized_pnl']:,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Win Rate: {portfolio['win_rate']*100:.1f}%")
        
        # Opportunities
        if tick_report['opportunities']:
            print(f"\n🎯 ACTIVE OPPORTUNITIES:")
            for opp in tick_report['opportunities'][:3]:  # Show top 3
                print(f"   {opp['opportunity_type']} {opp['profit_tier']} "
                     f"({opp['confidence']*100:.0f}% confidence)")
        
        print("="*80)
    
    def _print_final_report(self):
        """Print final demo results"""
        status = self.navigator.get_comprehensive_status()
        
        print("\n" + "🏁"*40)
        print("DEMO COMPLETE - FINAL RESULTS")
        print("🏁"*40)
        
        portfolio = status['portfolio']
        initial_value = 50000.0
        final_value = portfolio['total_value']
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        print(f"\n💰 PORTFOLIO PERFORMANCE:")
        print(f"   Initial Balance: ${initial_value:,.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Total Trades: {portfolio['total_trades']}")
        print(f"   Win Rate: {portfolio['win_rate']*100:.1f}%")
        print(f"   Total Profit: ${portfolio['total_profit']:,.2f}")
        
        print(f"\n🔺 ANTI-POLE STATISTICS:")
        antipole_stats = status['antipole']
        print(f"   Buffer Fill: {antipole_stats.get('buffer_fill', 0)}")
        print(f"   Ready Rate: {antipole_stats.get('ready_rate', 0)*100:.1f}%")
        print(f"   Avg ICAP: {antipole_stats.get('icap_mean', 0):.3f}")
        
        print(f"\n🌡️ THERMAL STATISTICS:")
        thermal_stats = status['thermal']
        print(f"   Avg Thermal Load: {thermal_stats.get('avg_thermal_load', 0)*100:.1f}%")
        print(f"   Emergency Stops: {thermal_stats.get('emergency_stops', 0)}")
        
        print(f"\n🖥️ VISUALIZATION:")
        viz_stats = status['visualization']
        print(f"   Frames Rendered: {viz_stats.get('frames_rendered', 0)}")
        print(f"   Active Connections: {viz_stats.get('active_connections', 0)}")
        
        print(f"\n📊 DEMO STATISTICS:")
        print(f"   Total Ticks Processed: {self.tick_count}")
        print(f"   Market Regimes Seen: Bull, Bear, Neutral, Volatile")
        print(f"   Tesseract Glyphs Generated: {viz_stats.get('frames_rendered', 0) * 5}")  # Estimate
        
        print("\n🏁" + "="*78 + "🏁")
        print("Thank you for trying the Anti-Pole Profit Navigator!")
        print("🔺 The future of quantum profit navigation is here 🔻")

async def main():
    """Main demo entry point"""
    print("""
🔺🔻 ANTI-POLE PROFIT NAVIGATOR DEMO 🔻🔺
==========================================

This demo showcases the complete Anti-Pole Theory system:

• Real-time BTC market simulation
• Anti-Pole vector calculations and ICAP probability
• Thermal protection and cooldown system  
• 4D Tesseract visualization
• Automated profit navigation and trading
• Portfolio management and risk control

The demo will run for 5 minutes with 1-second ticks.
Open your browser to view the Tesseract visualization!

Press Ctrl+C to stop early.
    """)
    
    # Wait for user to be ready
    input("Press Enter to start the demo...")
    
    demo = DemoController()
    await demo.run_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user. Goodbye!")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise 