"""
Simple Core Systems Integration Test
===================================

A simplified test to validate core system functionality and integration.
"""

import sys
import os
import traceback
import numpy as np

# Add paths for all core systems
sys.path.extend([
    './core',
    './ncco_core', 
    './aleph_core'
])

def test_mathlib_systems():
    """Test all mathlib versions"""
    print("=" * 60)
    print("TESTING MATHEMATICAL LIBRARY SYSTEMS")
    print("=" * 60)
    
    results = {}
    
    # Test MathLib v1
    try:
        from core.mathlib import CoreMathLib, GradedProfitVector
        mathlib_v1 = CoreMathLib()
        
        # Basic vector operations
        test_a = np.array([1.0, 2.0, 3.0])
        test_b = np.array([4.0, 5.0, 6.0])
        
        similarity = mathlib_v1.cosine_similarity(test_a, test_b)
        distance = mathlib_v1.euclidean_distance(test_a, test_b)
        
        # Test graded profit vector
        trade_data = {
            'profit': 100.0,
            'volume_allocated': 1000.0,
            'time_held': 3600.0,
            'signal_strength': 0.8,
            'smart_money_score': 0.75
        }
        graded_vector = mathlib_v1.grading_vector(trade_data)
        
        print("✅ MathLib v1: PASSED")
        print(f"   - Cosine similarity: {similarity:.4f}")
        print(f"   - Euclidean distance: {distance:.4f}")
        print(f"   - Graded vector profit: {graded_vector.profit}")
        results['mathlib_v1'] = True
        
    except Exception as e:
        print(f"❌ MathLib v1: FAILED - {e}")
        results['mathlib_v1'] = False
    
    # Test MathLib v2
    try:
        from core.mathlib_v2 import CoreMathLibV2, SmartStop
        mathlib_v2 = CoreMathLibV2()
        
        # Test advanced features
        test_prices = np.array([100.0, 102.0, 98.0, 105.0, 97.0])
        test_volumes = np.array([1000.0, 1500.0, 800.0, 2000.0, 1200.0])
        
        vwap = mathlib_v2.calculate_vwap(test_prices, test_volumes)
        rsi = mathlib_v2.calculate_rsi(test_prices)
        
        # Test smart stop
        smart_stop = SmartStop()
        stop_result = smart_stop.update(105.0, 100.0)
        
        print("✅ MathLib v2: PASSED")
        print(f"   - VWAP calculated: {len(vwap)} values")
        print(f"   - RSI calculated: {len(rsi)} values")
        print(f"   - Smart stop profit: {stop_result['profit_pct']:.4f}")
        results['mathlib_v2'] = True
        
    except Exception as e:
        print(f"❌ MathLib v2: FAILED - {e}")
        results['mathlib_v2'] = False
    
    # Test MathLib v3
    try:
        from core.mathlib_v3 import SustainmentMathLib, SustainmentVector, create_test_context
        mathlib_v3 = SustainmentMathLib()
        
        # Test sustainment framework
        context = create_test_context()
        sustainment_vector = mathlib_v3.calculate_sustainment_vector(context)
        
        si_value = sustainment_vector.sustainment_index()
        is_sustainable = sustainment_vector.is_sustainable()
        
        print("✅ MathLib v3: PASSED")
        print(f"   - Sustainment Index: {si_value:.4f}")
        print(f"   - Is Sustainable: {is_sustainable}")
        print(f"   - Principles count: {len(sustainment_vector.principles)}")
        results['mathlib_v3'] = True
        
    except Exception as e:
        print(f"❌ MathLib v3: FAILED - {e}")
        traceback.print_exc()
        results['mathlib_v3'] = False
    
    return results

def test_ncco_system():
    """Test NCCO core system"""
    print("\n" + "=" * 60)
    print("TESTING NCCO CORE SYSTEM")
    print("=" * 60)
    
    results = {}
    
    try:
        from ncco_core import NCCO, generate_nccos, score_nccos
        
        # Test NCCO creation
        ncco = NCCO(
            id=1,
            price_delta=5.0,
            base_price=100.0,
            bit_mode=8,
            score=0.75
        )
        
        # Test NCCO generation
        price_deltas = [5.0, -3.0, 2.5]  # List of price deltas
        generated_nccos = generate_nccos(price_deltas, base_price=100.0, bit_mode=8)
        scores = score_nccos(generated_nccos)
        
        print("✅ NCCO Core: PASSED")
        print(f"   - NCCO created with ID: {ncco.id}")
        print(f"   - Generated NCCOs: {len(generated_nccos)}")
        print(f"   - Scores calculated: {len(scores)}")
        results['ncco_core'] = True
        
    except Exception as e:
        print(f"❌ NCCO Core: FAILED - {e}")
        traceback.print_exc()
        results['ncco_core'] = False
    
    return results

def test_aleph_system():
    """Test Aleph core system"""
    print("\n" + "=" * 60)
    print("TESTING ALEPH CORE SYSTEM")
    print("=" * 60)
    
    results = {}
    
    try:
        from aleph_core import AlephUnitizer, TesseractPortal, PatternMatcher
        
        # Test AlephUnitizer
        unitizer = AlephUnitizer()
        test_data = [
            {'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0},
            {'feature1': 0.5, 'feature2': 1.5, 'feature3': 2.5},
        ]
        
        validation_result = unitizer.validate_units(test_data)
        
        # Test TesseractPortal
        tesseract = TesseractPortal()
        test_hash = "e3b0c44298fc1c149afbf4c8996fb924" * 2
        pattern, entropy = tesseract.extract_tesseract_pattern(test_hash)
        metrics = tesseract.calculate_tesseract_metrics(pattern, entropy)
        
        # Test PatternMatcher
        pattern_matcher = PatternMatcher()
        
        print("✅ Aleph Core: PASSED")
        print(f"   - Unitizer validation: {validation_result['dormant_states']} states")
        print(f"   - Tesseract pattern: {len(pattern)} dimensions")
        print(f"   - Pattern metrics magnitude: {metrics.magnitude:.4f}")
        results['aleph_core'] = True
        
    except Exception as e:
        print(f"❌ Aleph Core: FAILED - {e}")
        traceback.print_exc()
        results['aleph_core'] = False
    
    return results

def test_integration():
    """Test cross-system integration"""
    print("\n" + "=" * 60)
    print("TESTING CROSS-SYSTEM INTEGRATION")
    print("=" * 60)
    
    results = {}
    
    try:
        # Import from all systems
        from core.mathlib_v3 import SustainmentMathLib, create_test_context
        from ncco_core import NCCO, generate_nccos
        from aleph_core import AlephUnitizer
        
        # Initialize systems
        sustainment_lib = SustainmentMathLib()
        context = create_test_context()
        price_deltas = [5.0, -3.0, 2.5]  # List of price deltas
        nccos = generate_nccos(price_deltas, base_price=100.0, bit_mode=8)
        unitizer = AlephUnitizer()
        
        # Create integrated test data
        test_data = []
        for i, ncco in enumerate(nccos):
            test_data.append({
                'feature1': ncco.price_delta,
                'feature2': ncco.base_price / 100.0,
                'feature3': ncco.score
            })
        
        # Process through all systems
        aleph_result = unitizer.validate_units(test_data)
        sustainment_vector = sustainment_lib.calculate_sustainment_vector(context)
        
        # Test mathematical consistency
        test_prices = np.array([ncco.base_price for ncco in nccos])
        test_volumes = np.array([abs(ncco.price_delta) * 100 for ncco in nccos])
        
        trading_decision = sustainment_lib.sustainment_aware_trading_decision(
            test_prices, test_volumes, context
        )
        
        print("✅ Integration: PASSED")
        print(f"   - Processed {len(nccos)} NCCOs through all systems")
        print(f"   - Aleph validation: {aleph_result['dormant_states']} states")
        print(f"   - Sustainment index: {sustainment_vector.sustainment_index():.4f}")
        print(f"   - Trading analysis: {trading_decision.get('sustainment_index', 'N/A'):.4f}")
        print(f"   - System sustainable: {trading_decision.get('is_sustainable', False)}")
        results['integration'] = True
        
    except Exception as e:
        print(f"❌ Integration: FAILED - {e}")
        traceback.print_exc()
        results['integration'] = False
    
    return results

def generate_comprehensive_report(all_results):
    """Generate final test report"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE CORE SYSTEMS TEST REPORT")
    print("=" * 80)
    
    total_tests = sum(len(results) for results in all_results.values())
    passed_tests = sum(sum(results.values()) for results in all_results.values())
    
    for category, results in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * 40)
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name}: {status}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("🎉 EXCELLENT: All core systems are functioning optimally!")
        print("   - Mathematical libraries are working correctly")
        print("   - NCCO and Aleph cores are operational")
        print("   - Cross-system integration is successful")
    elif success_rate >= 80:
        print("✅ GOOD: Core systems are mostly functional with minor issues")
        print("   - Most components are working correctly")
        print("   - Minor fixes may be needed for full optimization")
    elif success_rate >= 60:
        print("⚠️  WARNING: Some core systems have significant issues")
        print("   - Major components need attention")
        print("   - Integration may be compromised")
    else:
        print("❌ CRITICAL: Major issues detected in core systems")
        print("   - Core functionality is severely impaired")
        print("   - Immediate fixes required")
    
    print("=" * 80)
    
    # Specific recommendations
    print("\nRECOMMENDations:")
    print("-" * 40)
    
    if all_results.get('mathlib', {}).get('mathlib_v1', True):
        print("✅ MathLib v1: Core mathematical operations are stable")
    else:
        print("❌ MathLib v1: Fix required for basic mathematical operations")
    
    if all_results.get('mathlib', {}).get('mathlib_v2', True):
        print("✅ MathLib v2: Advanced trading algorithms are functional")
    else:
        print("❌ MathLib v2: Issues with advanced trading features")
    
    if all_results.get('mathlib', {}).get('mathlib_v3', True):
        print("✅ MathLib v3: Sustainment framework is operational")
    else:
        print("❌ MathLib v3: Sustainment framework needs attention")
    
    if all_results.get('ncco', {}).get('ncco_core', True):
        print("✅ NCCO Core: Neural coordination system is working")
    else:
        print("❌ NCCO Core: Neural coordination system needs fixes")
    
    if all_results.get('aleph', {}).get('aleph_core', True):
        print("✅ Aleph Core: Pattern analysis system is operational")
    else:
        print("❌ Aleph Core: Pattern analysis system requires debugging")
    
    if all_results.get('integration', {}).get('integration', True):
        print("✅ Integration: All systems work together seamlessly")
    else:
        print("❌ Integration: Cross-system communication needs improvement")

def main():
    """Run comprehensive core systems test"""
    print("🚀 Starting Comprehensive Core Systems Integration Test")
    print("Testing all mathematical libraries, NCCO core, and Aleph core systems")
    
    all_results = {}
    
    # Test each system
    all_results['mathlib'] = test_mathlib_systems()
    all_results['ncco'] = test_ncco_system()
    all_results['aleph'] = test_aleph_system()
    all_results['integration'] = test_integration()
    
    # Generate final report
    generate_comprehensive_report(all_results)

if __name__ == "__main__":
    main() 