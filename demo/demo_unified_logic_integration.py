import random
import time
from datetime import datetime
from typing import Any, Dict

import yaml

from core.unified_logic.btc_tick_backlog import BTCTick, save_btc_tick
from core.unified_logic.entry_logic import entry_score
from core.unified_logic.float_valuation import float_valuation
from core.unified_logic.ghost_conditionals import ghost_conditional
from core.unified_logic.phase_math import phase_adjust
from core.unified_logic.unicode_emoji_asic import label_state

# -*- coding: utf-8 -*-
"""Demo script for unified logic integration with live phase simulation and backlogging."""


# Import unified logic modules

class UnifiedLogicDemo:"""
"""Demo class for testing and demonstrating unified logic integration."""
    
def __init__(self):
        self.tick_count = 0
        self.phase_history = []
        self.decision_history = []
        self.backlog_data = []
        """


def simulate_btc_tick():-> Dict[str, Any]:
        """Simulate a BTC tick with hash rate and price data."""
# Simulate realistic BTC data
        
base_price = 50000.0
        base_hash_rate = 200.0  # EH/s
        
# Add some realistic variation
price_variation = random.uniform(-0.02, 0.02)  # +/-2%
        hash_variation = random.uniform(-0.01, 0.01)   # +/-1%
        
price = base_price * (1 + price_variation)
        hash_rate = base_hash_rate * (1 + hash_variation)
        
# Calculate float valuation"""
law_factor = 1.0 if phase == "mid" else (0.9 if phase == "low" else 1.1)
        float_val = float_valuation(price, hash_rate, law_factor)
        
# Generate Unicode label
unicode_label = label_state("BTC", phase, "ASIC001")
        
# Save tick to backlog
tick = save_btc_tick(hash_rate, price, float_val, phase, unicode_label)
        self.backlog_data.append(tick)
        
return {
            "tick": tick,
            "price": price,
            "hash_rate": hash_rate,
            "float_valuation": float_val,
            "unicode_label": unicode_label
    
def process_ghost_conditional():-> Dict[str, Any]:
        """Process ghost conditional logic for routing decisions."""
# Simulate coefficients based on tick data"""
psi = 0.5 + (tick_data["price"] - 50000) / 10000  # Path coefficient
        xi_sent = 0.5 + (tick_data["hash_rate"] - 200) / 100  # Sentiment
        phi_drift = 0.5 + (tick_data["float_valuation"] - 10000000) / 2000000  # Drift
        
# Clamp values to [0, 1]
        psi = max(0, min(1, psi))
        xi_sent = max(0, min(1, xi_sent))
        phi_drift = max(0, min(1, phi_drift))
        
# Process ghost conditional
decision = ghost_conditional(
            psi=psi,
            xi_sent=xi_sent,
            phi_drift=phi_drift,
            phase=tick_data["tick"].phase,
            unicode_label=tick_data["unicode_label"]
        )
        
self.decision_history.append(decision)
        return decision
    
def process_entry_logic():-> Dict[str, Any]:
        """Process entry logic for trading signals."""
# Calculate normalized price change and volatility"""
dp_norm = (tick_data["price"] - 50000) / 50000  # Normalized price change
        sigma_vol = abs(dp_norm) * 0.1  # Simple volatility measure
        
# Process entry score
entry = entry_score(
            dp_norm=dp_norm,
            sigma_vol=sigma_vol,
            w_btc=1.2,
            w_usdc=0.8,
            phase=tick_data["tick"].phase
        )
        
return {
            "entry_score": entry,
            "dp_norm": dp_norm,
            "sigma_vol": sigma_vol,
            "phase": tick_data["tick"].phase
    
def run_live_demo(self, duration_minutes: int = 5, tick_interval_seconds: float = 16):
        """Run a live demo simulation with real-time backlogging.""""""
print(f"🚀 Starting Unified Logic Live Demo for {duration_minutes} minutes")
        print(f"📊 Tick interval: {tick_interval_seconds} seconds (simulating 3.75/min)")
        print("=" * 60)
        
start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
phases = ["low", "mid", "high"]
        phase_index = 1  # Start with "mid"
        
while time.time() < end_time:
            current_time = datetime.now()
            phase = phases[phase_index]
            
print(f"\n⏰ {current_time.strftime('%H:%M:%S')} | Phase: {phase.upper()}")
            print("-" * 40)
            
# Simulate BTC tick
tick_data = self.simulate_btc_tick(phase)
            print(f"📈 BTC Price: ${tick_data['price']:,.2f}")
            print(f"⚡ Hash Rate: {tick_data['hash_rate']:.1f} EH/s")
            print(f"💰 Float Valuation: {tick_data['float_valuation']:,.0f}")
            print(f"🏷️  Label: {tick_data['unicode_label']}")
            
# Process ghost conditional
ghost_result = self.process_ghost_conditional(tick_data)
            print(f"👻 Ghost Decision: {ghost_result['decision']} (Score: {ghost_result['score']:.3f})")
            
# Process entry logic
entry_result = self.process_entry_logic(tick_data)
            print(f"🎯 Entry Score: {entry_result['entry_score']:.3f}")
            
# Phase adjustment example
adjusted_price = phase_adjust(tick_data['price'], phase)
            print(f"🔄 Phase Adjusted Price: ${adjusted_price:,.2f}")
            
self.tick_count += 1
            
# Rotate phases every few ticks
if self.tick_count % 3 == 0:
                phase_index = (phase_index + 1) % len(phases)
            
# Wait for next tick
time.sleep(tick_interval_seconds)
        
self.print_demo_summary()
    
def print_demo_summary(self):
        """Print a summary of the demo results.""""""
print("\n" + "=" * 60)
        print("📊 DEMO SUMMARY")
        print("=" * 60)
        print(f"Total Ticks Processed: {self.tick_count}")
        print(f"Backlog Entries: {len(self.backlog_data)}")
        print(f"Decisions Made: {len(self.decision_history)}")
        
# Calculate some statistics
if self.decision_history:
            true_decisions = sum(1 for d in self.decision_history if d['decision'])
            decision_rate = true_decisions / len(self.decision_history)
            print(f"Decision Rate: {decision_rate:.1%}")
        
if self.backlog_data:
            avg_price = sum(t.price for t in self.backlog_data) / len(self.backlog_data)
            avg_hash_rate = sum(t.hash_rate for t in self.backlog_data) / len(self.backlog_data)
            print(f"Average Price: ${avg_price:,.2f}")
            print(f"Average Hash Rate: {avg_hash_rate:.1f} EH/s")
    
def generate_yaml_config():-> str:
        """Generate YAML configuration for API integration."""
config = {"""
            "unified_logic": {
                "tick_interval_seconds": 16.0,  # 3.75 per minute
                "phases": ["low", "mid", "high"],
                "default_phase": "mid",
                "backlog_enabled": True,
                "ghost_conditional": {
                    "enabled": True,
                    "thresholds": {
                        "low": 0.3,
                        "mid": 0.5,
                        "high": 0.7
},
                "entry_logic": {
                    "enabled": True,
                    "weights": {
                        "btc": 1.2,
                        "usdc": 0.8
},
                "unicode_labels": {
                    "enabled": True,
                    "tier_emojis": {
                        "low": "🟢",
                        "mid": "🟡",
                        "high": "🔴"
        
return yaml.dump(config, default_flow_style=False, indent=2)

def main():
    """Main function to run the unified logic demo."""
demo = UnifiedLogicDemo()
    
# Run live demo
demo.run_live_demo(duration_minutes=2, tick_interval_seconds=5)  # Faster for demo
    
# Generate YAML config"""
print("\n" + "=" * 60)
    print("⚙️  YAML CONFIGURATION FOR API INTEGRATION")
    print("=" * 60)
    yaml_config = demo.generate_yaml_config()
    print(yaml_config)
    
# Save config to file
with open("demo/unified_logic_config.yaml", "w") as f:
        f.write(yaml_config)
    
print("\n✅ Demo completed! Configuration saved to 'demo/unified_logic_config.yaml'")

if __name__ == "__main__":
    main() 