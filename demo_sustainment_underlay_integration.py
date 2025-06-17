#!/usr/bin/env python3
"""
Sustainment Underlay Integration Demo
====================================

Demonstrates the Law of Sustainment as a mathematical underlay that continuously
monitors and corrects all Schwabot controllers to maintain the 8-principle framework.

This shows how the underlay acts as a mathematical synthesis layer that ensures
all controllers (thermal, cooldown, profit, fractal) operate within sustainable bounds.
"""

import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# Import existing controllers (with fallbacks for demo)
try:
    from core.thermal_zone_manager import ThermalZoneManager
    from core.cooldown_manager import CooldownManager
    from core.profit_navigator import AntiPoleProfitNavigator
    from core.fractal_core import FractalCore
    from core.sustainment_underlay_controller import (
        SustainmentUnderlayController, 
        SustainmentPrinciple,
        create_sustainment_underlay
    )
    print("✅ Successfully imported all controllers")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("📝 Running in mock mode")

class MockController:
    """Mock controller for demonstration"""
    def __init__(self, name: str):
        self.name = name
        self.state = {}
        self.corrections_applied = 0
    
    def get_current_state(self):
        return self.state
    
    def apply_correction(self, action: str, magnitude: float):
        self.corrections_applied += 1
        logger.info(f"🔧 {self.name} applying correction: {action} (magnitude: {magnitude:.2f})")

def create_mock_controllers():
    """Create mock controllers for demonstration"""
    thermal_mock = MockController("ThermalManager")
    thermal_mock.state = {
        'zone': type('Zone', (), {'value': 'normal'})(),
        'load_cpu': 0.4,
        'load_gpu': 0.6,
        'timestamp': datetime.now()
    }
    
    cooldown_mock = MockController("CooldownManager")
    cooldown_mock.register_event = lambda event, payload: logger.info(f"Cooldown event: {event}")
    
    profit_mock = MockController("ProfitNavigator")
    profit_mock.get_comprehensive_status = lambda: {
        'return_rate': np.random.uniform(-0.1, 0.15),
        'drawdown': np.random.uniform(0.0, 0.2),
        'volatility_response': np.random.uniform(0.3, 0.8)
    }
    
    fractal_mock = MockController("FractalCore")
    fractal_mock.get_current_state = lambda: type('FractalState', (), {
        'coherence': np.random.uniform(0.4, 0.9),
        'entropy': np.random.uniform(0.2, 0.8),
        'phase': np.random.uniform(0, 2*np.pi)
    })()
    
    return thermal_mock, cooldown_mock, profit_mock, fractal_mock

def simulate_system_drift(controllers, step: int):
    """Simulate natural system drift that requires correction"""
    thermal_mock, cooldown_mock, profit_mock, fractal_mock = controllers
    
    # Simulate various drift scenarios
    if step % 10 == 0:
        # Thermal spike
        thermal_mock.state['load_gpu'] = min(1.0, thermal_mock.state['load_gpu'] + 0.3)
        thermal_mock.state['zone'].value = 'hot' if thermal_mock.state['load_gpu'] > 0.8 else 'normal'
        logger.warning("🌡️ Thermal spike detected")
    
    elif step % 7 == 0:
        # Profit efficiency drop
        current_status = profit_mock.get_comprehensive_status()
        current_status['return_rate'] = max(-0.2, current_status['return_rate'] - 0.1)
        logger.warning("📉 Profit efficiency dropped")
    
    elif step % 13 == 0:
        # Fractal coherence loss
        fractal_state = fractal_mock.get_current_state()
        fractal_state.coherence = max(0.2, fractal_state.coherence - 0.3)
        fractal_state.entropy = min(1.0, fractal_state.entropy + 0.2)
        logger.warning("🌀 Fractal coherence degraded")
    
    # Add some random noise to all systems
    thermal_mock.state['load_cpu'] = np.clip(
        thermal_mock.state['load_cpu'] + np.random.uniform(-0.05, 0.05), 0.0, 1.0
    )

def print_sustainment_status(underlay: SustainmentUnderlayController, step: int):
    """Print current sustainment status"""
    status = underlay.get_sustainment_status()
    
    if status.get('status') == 'initializing':
        print(f"Step {step:03d}: 🔄 Initializing sustainment framework...")
        return
    
    si = status['sustainment_index']
    health = status['system_health_score']
    
    # Status emoji
    if si >= underlay.s_crit:
        status_emoji = "✅"
        status_text = "SUSTAINABLE"
    else:
        status_emoji = "⚠️" if si > underlay.s_crit * 0.8 else "❌"
        status_text = "CORRECTING" if si > underlay.s_crit * 0.8 else "CRITICAL"
    
    print(f"\n{'='*60}")
    print(f"Step {step:03d}: {status_emoji} {status_text}")
    print(f"{'='*60}")
    print(f"Sustainment Index: {si:.3f} (threshold: {underlay.s_crit:.3f})")
    print(f"System Health: {health:.1%}")
    print(f"Total Corrections: {status['total_corrections']}")
    
    # Print principle breakdown
    print(f"\n📊 Principle Breakdown:")
    vector = status['current_vector']
    for principle, value in vector.items():
        bar = "█" * int(value * 20)
        bar += "░" * (20 - int(value * 20))
        status_icon = "✅" if value >= 0.5 else "⚠️" if value >= 0.3 else "❌"
        print(f"  {status_icon} {principle.capitalize():<15}: {value:.3f} |{bar}|")
    
    # Show violations if any
    violations = status['principle_violations']
    if violations:
        print(f"\n⚠️ Principle Violations:")
        for principle, count in violations.items():
            if count > 0:
                print(f"  • {principle.value}: {count} violations")

def run_sustainment_demo():
    """Run the complete sustainment underlay demonstration"""
    print("🚀 Starting Sustainment Underlay Integration Demo")
    print("="*60)
    print("This demonstrates the Law of Sustainment as a mathematical underlay")
    print("that continuously corrects system drift across all controllers.")
    print("="*60)
    
    # Create controllers (mock for demo)
    print("\n🔧 Initializing controllers...")
    thermal_manager, cooldown_manager, profit_navigator, fractal_core = create_mock_controllers()
    
    # Create sustainment underlay
    print("🧮 Creating sustainment underlay mathematical synthesis...")
    underlay = SustainmentUnderlayController(
        thermal_manager=thermal_manager,
        cooldown_manager=cooldown_manager,
        profit_navigator=profit_navigator,
        fractal_core=fractal_core
    )
    
    # Start continuous synthesis
    print("⚡ Starting continuous mathematical synthesis...")
    underlay.start_continuous_synthesis(interval=2.0)  # Fast for demo
    
    # Demo parameters
    max_steps = 30
    controllers = (thermal_manager, cooldown_manager, profit_navigator, fractal_core)
    
    print(f"\n🔄 Running {max_steps} simulation steps...")
    print("The system will naturally drift and the underlay will correct it.")
    
    try:
        for step in range(1, max_steps + 1):
            # Simulate system drift
            simulate_system_drift(controllers, step)
            
            # Wait for synthesis cycle
            time.sleep(2.5)
            
            # Print status
            print_sustainment_status(underlay, step)
            
            # Check for corrections
            if len(underlay.correction_history) > 0:
                recent_corrections = list(underlay.correction_history)[-3:]
                if recent_corrections:
                    print(f"\n🔧 Recent Corrections Applied:")
                    for correction in recent_corrections:
                        print(f"  • {correction.principle.value} -> {correction.target_controller}")
                        print(f"    Action: {correction.action_type} (magnitude: {correction.magnitude:.2f})")
            
            # Pause between steps
            if step < max_steps:
                time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    
    finally:
        # Stop synthesis
        print("\n🔄 Stopping sustainment synthesis...")
        underlay.stop_continuous_synthesis()
    
    # Final summary
    print("\n" + "="*60)
    print("📈 FINAL SUSTAINMENT SUMMARY")
    print("="*60)
    
    final_status = underlay.get_sustainment_status()
    if final_status.get('status') != 'initializing':
        print(f"Final Sustainment Index: {final_status['sustainment_index']:.3f}")
        print(f"System Health Score: {final_status['system_health_score']:.1%}")
        print(f"Total Corrections Applied: {final_status['total_corrections']}")
        print(f"History Length: {final_status['history_length']} samples")
        
        # Show correction effectiveness
        if final_status['total_corrections'] > 0:
            effectiveness = final_status['sustainment_index'] / underlay.s_crit
            print(f"Correction Effectiveness: {effectiveness:.1%}")
            
            if effectiveness >= 1.0:
                print("✅ System successfully maintained sustainability")
            elif effectiveness >= 0.8:
                print("⚠️ System partially sustainable (within tolerance)")
            else:
                print("❌ System requires additional intervention")
    
    print("\n🎯 Key Takeaways:")
    print("• The sustainment underlay continuously monitors all controllers")
    print("• Mathematical synthesis translates controller states into 8 principles")
    print("• Automatic corrections guide the system back to sustainable states")
    print("• The system is profit-centric while maintaining operational integrity")
    print("• This is an underlay system - existing controllers remain unchanged")
    
    print("\n🔗 Integration Points Demonstrated:")
    print("• Thermal management -> Survivability & Economy principles")
    print("• Profit navigation -> Economy & Anticipation principles")
    print("• Fractal processing -> Integration & Continuity principles")
    print("• Cooldown management -> Responsiveness & Simplicity principles")
    
    print("\n✨ Mathematical Foundation:")
    print("• SI(t) = F(A, I, R, S, E, Sv, C, Im) > S_crit")
    print("• Continuous correction when SI(t) < threshold")
    print("• Self-correcting mathematical synthesis")
    print("• Profit-centric optimization within sustainable bounds")

def demonstrate_mathematical_formulation():
    """Demonstrate the mathematical formulation of the Law of Sustainment"""
    print("\n" + "="*60)
    print("🧮 MATHEMATICAL FORMULATION DEMONSTRATION")
    print("="*60)
    
    print("The Law of Sustainment implements:")
    print("SI(t) = Σ wᵢ × Pᵢ(t) > S_crit")
    print("\nWhere:")
    print("• SI(t) = Sustainment Index at time t")
    print("• wᵢ = Weight for principle i")
    print("• Pᵢ(t) = Value of principle i at time t")
    print("• S_crit = Critical sustainment threshold")
    
    print("\nThe 8 Principles (Mathematical Mapping):")
    principles = [
        ("A(t)", "Anticipation", "Predictive modeling: A(t+1) = f(Xₜ, Ẋₜ, Ẍₜ, ...)"),
        ("I(t)", "Integration", "System coherence: I = Σ wᵢSᵢ"),
        ("R(t)", "Responsiveness", "Adaptation rate: R = ΔS/Δt"),
        ("S(t)", "Simplicity", "Complexity minimization: min Complexity(M)"),
        ("E(t)", "Economy", "Resource efficiency: E = Output/Input"),
        ("Sv(t)", "Survivability", "Risk management: Sv = P(Uptime > T)"),
        ("C(t)", "Continuity", "Persistence: C = ∫ S(t)dt"),
        ("Im(t)", "Improvisation", "Creative adaptation: Im = argmax U(a|Xₜ)")
    ]
    
    for symbol, name, formula in principles:
        print(f"  {symbol:<6} {name:<15}: {formula}")
    
    print("\nContinuous Correction Algorithm:")
    print("1. Monitor: S(t) = synthesize_current_state()")
    print("2. Evaluate: SI(t) = S(t) · w")
    print("3. Detect: if SI(t) < S_crit then generate_corrections()")
    print("4. Apply: apply_corrections(controllers)")
    print("5. Adapt: update_weights(performance)")
    print("6. Repeat: continuous_synthesis_loop()")

if __name__ == "__main__":
    print("Sustainment Underlay Mathematical Controller")
    print("Schwabot Law of Sustainment Demo v1.0")
    print("="*60)
    
    try:
        # Demonstrate mathematical formulation
        demonstrate_mathematical_formulation()
        
        # Run main demo
        run_sustainment_demo()
        
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Thank you for exploring the Law of Sustainment!")
    print("Mathematical synthesis ensures continuous operational sustainability.") 