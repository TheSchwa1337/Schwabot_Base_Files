#!/usr/bin/env python3
"""
Test core UROS v1.0 functionality.
"""

import sys
import traceback
from datetime import datetime

def test_core_components():
    """Test core UROS v1.0 components."""
    print("Testing UROS v1.0 Core Components...")
    print("=" * 50)
    
    # Test 1: Basic imports
    try:
        from core.gpt_command_layer import AIAgentType, CommandDomain, CommandPriority, AICommand, CommandResponse
        print("✅ GPT Command Layer classes imported")
    except Exception as e:
        print(f"❌ GPT Command Layer import failed: {e}")
        return False
    
    # Test 2: Create AI Command
    try:
        command = AICommand(
            command_id="test_cmd_001",
            agent_type=AIAgentType.GPT,
            domain=CommandDomain.STRATEGY,
            priority=CommandPriority.MEDIUM,
            hash_signature="test_hash_123",
            timestamp=datetime.now(),
            payload={"test": True},
            context={"test": True}
        )
        print("✅ AI Command created successfully")
    except Exception as e:
        print(f"❌ AI Command creation failed: {e}")
        return False
    
    # Test 3: Strategy Mapper
    try:
        from core.strategy_mapper import StrategyMapper
        mapper = StrategyMapper()
        print("✅ Strategy Mapper initialized")
    except Exception as e:
        print(f"❌ Strategy Mapper failed: {e}")
        return False
    
    # Test 4: DLT Waveform Engine
    try:
        from core.dlt_waveform_engine import DLTWaveformEngine
        engine = DLTWaveformEngine()
        print("✅ DLT Waveform Engine initialized")
    except Exception as e:
        print(f"❌ DLT Waveform Engine failed: {e}")
        return False
    
    # Test 5: Windows CLI Compatibility
    try:
        from core.utils.windows_cli_compatibility import safe_print, safe_format_error
        safe_print("✅ Windows CLI Compatibility working")
    except Exception as e:
        print(f"❌ Windows CLI Compatibility failed: {e}")
        return False
    
    print("=" * 50)
    print("🎉 Core functionality test passed!")
    return True

def test_mathematical_calculations():
    """Test mathematical calculations."""
    print("\nTesting Mathematical Calculations...")
    print("=" * 50)
    
    try:
        import numpy as np
        
        # Test DLT Waveform Engine calculations
        from core.dlt_waveform_engine import DLTWaveformEngine
        engine = DLTWaveformEngine()
        
        # Add some test data
        for i in range(50):
            price = 50000.0 + (i * 10) + (i % 3 - 1) * 100
            engine.update_tick_data(price)
        
        # Analyze waveform
        analysis = engine.analyze_current_waveform()
        
        if "mathematical_measures" in analysis:
            math_measures = analysis["mathematical_measures"]
            print(f"✅ ρ coefficient: {math_measures.get('rho_coefficient', 0.0):.4f}")
            print(f"✅ Resonance strength: {math_measures.get('resonance_strength', 0.0):.4f}")
            print(f"✅ Entropy complexity: {math_measures.get('entropy_complexity', 0.0):.4f}")
            print(f"✅ Current acceleration: {math_measures.get('current_acceleration', 0.0):.4f}")
            print(f"✅ Current velocity: {math_measures.get('current_velocity', 0.0):.4f}")
        else:
            print("⚠️ Mathematical measures not found in analysis")
        
        print("✅ Mathematical calculations completed")
        return True
        
    except Exception as e:
        print(f"❌ Mathematical calculations failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 UROS v1.0 Core Functionality Test")
    print("=" * 60)
    
    # Test core components
    core_success = test_core_components()
    
    # Test mathematical calculations
    math_success = test_mathematical_calculations()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Core Components: {'✅ PASSED' if core_success else '❌ FAILED'}")
    print(f"Math Calculations: {'✅ PASSED' if math_success else '❌ FAILED'}")
    
    overall_success = core_success and math_success
    print(f"\nOverall Result: {'🎉 ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 