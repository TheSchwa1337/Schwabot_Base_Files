#!/usr/bin/env python3
"""
Test script for restored mathematical components.
"""

import numpy as np
from mathlib_v4 import MathLibV4


def test_mathlib_v4():
    """Test MathLib V4 functionality."""
    print("🧮 Testing MathLib V4...")
    
    # Initialize MathLib
    ml4 = MathLibV4()
    print(f"✅ MathLib V4 initialized: v{ml4.version}")
    
    # Test data
    test_data = np.array([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
    print(f"📊 Test data: {test_data}")
    
    # Test DLT analysis
    print("\n🔍 Testing DLT Analysis...")
    result = ml4.analyze_dlt_waveform(test_data)
    
    if "error" in result:
        print(f"❌ DLT Analysis failed: {result['error']}")
        return False
    
    print(f"✅ DLT Analysis successful:")
    print(f"   Pattern Hash: {result['pattern_hash'][:10]}...")
    print(f"   Triplet Lock: {result['triplet_lock']}")
    print(f"   Mean Delta: {result['mean_delta']:.3f}")
    print(f"   Std Dev: {result['std_dev']:.3f}")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    # Test fractal creation
    print("\n🌀 Testing Fractal Creation...")
    deltas = ml4.calculate_deltas(test_data)
    fractal = ml4.create_forever_fractal(deltas)
    
    print(f"✅ Fractal created:")
    print(f"   Pattern Hash: {fractal.pattern_hash[:10]}...")
    print(f"   Length: {fractal.length}")
    print(f"   Mean Delta: {fractal.mean_delta:.3f}")
    print(f"   Std Dev: {fractal.std_dev:.3f}")
    
    # Test confidence calculation
    print("\n🎯 Testing Confidence Calculation...")
    confidence = ml4.calculate_greyscale_confidence(0.85, drift_velocity=0.1)
    print(f"✅ Confidence calculated: {confidence:.3f}")
    
    # Test warp drift correction
    print("\n⏰ Testing Warp Drift Correction...")
    correction = ml4.calculate_warp_drift_correction(0.2, 0.4)
    print(f"✅ Warp correction: {correction:.3f}")
    
    print("\n🎉 All MathLib V4 tests passed!")
    return True


def test_api_gateway():
    """Test API Gateway functionality."""
    print("\n🌐 Testing API Gateway...")
    
    try:
        from api_gateway import SchwabotAPIGateway
        gateway = SchwabotAPIGateway(host="127.0.0.1", port=8001)
        print(f"✅ API Gateway initialized on {gateway.host}:{gateway.port}")
        
        if gateway.app:
            print("✅ FastAPI application available")
        else:
            print("⚠️  FastAPI not available")
            
        return True
        
    except Exception as e:
        print(f"❌ API Gateway test failed: {e}")
        return False


def test_mathematical_consciousness_bridge():
    """Test Mathematical Consciousness Bridge."""
    print("\n🧠 Testing Mathematical Consciousness Bridge...")
    
    try:
        from mathematical_consciousness_bridge import MathematicalConsciousnessBridge
        import asyncio
        
        async def test_bridge():
            bridge = MathematicalConsciousnessBridge()
            status = await bridge.get_bridge_status()
            
            print(f"✅ Bridge initialized: v{status['bridge_version']}")
            print("🔧 Components available:")
            for component, available in status['components_available'].items():
                print(f"   - {component}: {'✅' if available else '❌'}")
            
            await bridge.cleanup()
            return True
        
        return asyncio.run(test_bridge())
        
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Restored Mathematical Components")
    print("=" * 50)
    
    tests = [
        ("MathLib V4", test_mathlib_v4),
        ("API Gateway", test_api_gateway),
        ("Mathematical Consciousness Bridge", test_mathematical_consciousness_bridge)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Mathematical components restored successfully.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main() 