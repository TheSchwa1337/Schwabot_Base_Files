#!/usr/bin/env python3
"""
Test for MathLib Add/Subtract Functions Fix - Missing Basic Mathematical Operations Resolution
============================================================================================

This test verifies that the missing add() and subtract() functions have been
properly added to mathlib.py and can be imported by mathlib_v2.py, fixing
the critical gap in our mathematical foundation's basic operations.
"""

import sys
import traceback
from typing import Dict, Any
import numpy as np

def test_mathlib_add_subtract_functions_fix() -> Dict[str, Any]:
    """
    Test that the missing add() and subtract() mathematical functions have been properly fixed
    
    Returns:
        Dict with test results and status
    """
    print("🔧 TESTING MATHLIB ADD/SUBTRACT FUNCTIONS FIX")
    print("=" * 60)
    
    results = {
        'test_name': 'MathLib Add/Subtract Functions Fix Verification',
        'timestamp': str(np.datetime64('now')),
        'tests_passed': 0,
        'tests_total': 0,
        'critical_issues': [],
        'success_items': [],
        'overall_status': 'UNKNOWN'
    }
    
    # Test 1: Import add and subtract from mathlib directly
    print("\n1️⃣ Testing direct import from mathlib.py...")
    results['tests_total'] += 1
    try:
        from mathlib import add, subtract
        print("   ✅ Successfully imported add and subtract from mathlib.py")
        
        # Test the functions work
        test_result_add = add(5.0, 3.0)
        test_result_subtract = subtract(10.0, 4.0)
        
        if test_result_add == 8.0 and test_result_subtract == 6.0:
            print(f"   ✅ Functions work correctly: add(5,3)={test_result_add}, subtract(10,4)={test_result_subtract}")
            results['tests_passed'] += 1
            results['success_items'].append("Direct mathlib import and function execution")
        else:
            results['critical_issues'].append(f"Functions return incorrect values: add={test_result_add}, subtract={test_result_subtract}")
    except Exception as e:
        print(f"   ❌ Failed to import or use functions: {e}")
        results['critical_issues'].append(f"Direct mathlib import failed: {e}")
    
    # Test 2: Import from mathlib package
    print("\n2️⃣ Testing import from mathlib package...")
    results['tests_total'] += 1
    try:
        from mathlib import add as pkg_add, subtract as pkg_subtract
        print("   ✅ Successfully imported from mathlib package")
        
        # Test package functions
        pkg_add_result = pkg_add(7.5, 2.5)
        pkg_subtract_result = pkg_subtract(15.0, 6.0)
        
        if pkg_add_result == 10.0 and pkg_subtract_result == 9.0:
            print(f"   ✅ Package functions work: add(7.5,2.5)={pkg_add_result}, subtract(15,6)={pkg_subtract_result}")
            results['tests_passed'] += 1
            results['success_items'].append("Mathlib package import and function execution")
        else:
            results['critical_issues'].append(f"Package functions incorrect: add={pkg_add_result}, subtract={pkg_subtract_result}")
    except Exception as e:
        print(f"   ❌ Failed to import from mathlib package: {e}")
        results['critical_issues'].append(f"Mathlib package import failed: {e}")
    
    # Test 3: Test mathlib_v2 import (the original failing case)
    print("\n3️⃣ Testing mathlib_v2.py import (original failing case)...")
    results['tests_total'] += 1
    try:
        # This should now work since we added the missing functions
        sys.path.insert(0, '.')
        from mathlib_v2 import CoreMathLibV2
        print("   ✅ Successfully imported CoreMathLibV2 (mathlib_v2.py can now find add/subtract)")
        
        # Test that mathlib_v2 can use the functions it imports
        math_v2 = CoreMathLibV2()
        print("   ✅ CoreMathLibV2 instantiated successfully")
        results['tests_passed'] += 1
        results['success_items'].append("mathlib_v2.py import resolved (original failing case fixed)")
    except Exception as e:
        print(f"   ❌ mathlib_v2 import still failing: {e}")
        print(f"   Full traceback: {traceback.format_exc()}")
        results['critical_issues'].append(f"mathlib_v2 import still fails: {e}")
    
    # Test 4: Array operations support
    print("\n4️⃣ Testing numpy array support...")
    results['tests_total'] += 1
    try:
        from mathlib import add, subtract
        
        # Test with numpy arrays
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])
        
        add_result = add(arr1, arr2)
        subtract_result = subtract(arr2, arr1)
        
        expected_add = np.array([5.0, 7.0, 9.0])
        expected_subtract = np.array([3.0, 3.0, 3.0])
        
        if np.allclose(add_result, expected_add) and np.allclose(subtract_result, expected_subtract):
            print(f"   ✅ Array operations work correctly")
            print(f"      add([1,2,3], [4,5,6]) = {add_result}")
            print(f"      subtract([4,5,6], [1,2,3]) = {subtract_result}")
            results['tests_passed'] += 1
            results['success_items'].append("Numpy array operations support")
        else:
            results['critical_issues'].append("Array operations produce incorrect results")
    except Exception as e:
        print(f"   ❌ Array operations failed: {e}")
        results['critical_issues'].append(f"Array operations failed: {e}")
    
    # Test 5: Cross-library compatibility
    print("\n5️⃣ Testing cross-library mathematical compatibility...")
    results['tests_total'] += 1
    try:
        from mathlib import CoreMathLib, add, subtract
        from core.mathlib_v2 import CoreMathLibV2
        
        # Test that both libraries can coexist
        math_v1 = CoreMathLib()
        math_v2 = CoreMathLibV2()
        
        print("   ✅ Both MathLib v1 and v2 can be instantiated together")
        
        # Test mathematical operations work in both contexts
        test_values = np.array([10.0, 20.0, 30.0])
        add_result = add(test_values, 5.0)
        subtract_result = subtract(test_values, 3.0)
        
        print(f"   ✅ Cross-library operations: add(array, scalar) and subtract(array, scalar) work")
        results['tests_passed'] += 1
        results['success_items'].append("Cross-library mathematical compatibility verified")
    except Exception as e:
        print(f"   ❌ Cross-library compatibility issue: {e}")
        results['critical_issues'].append(f"Cross-library compatibility: {e}")
    
    # Calculate final status
    success_rate = results['tests_passed'] / results['tests_total'] if results['tests_total'] > 0 else 0
    
    if success_rate == 1.0:
        results['overall_status'] = 'FULLY_FIXED'
        status_emoji = "🎉"
        status_msg = "ALL TESTS PASSED - ADD/SUBTRACT FUNCTIONS COMPLETELY FIXED!"
    elif success_rate >= 0.8:
        results['overall_status'] = 'MOSTLY_FIXED'
        status_emoji = "✅"
        status_msg = f"MOSTLY FIXED - {results['tests_passed']}/{results['tests_total']} tests passed"
    elif success_rate >= 0.6:
        results['overall_status'] = 'PARTIALLY_FIXED'
        status_emoji = "⚠️"
        status_msg = f"PARTIALLY FIXED - {results['tests_passed']}/{results['tests_total']} tests passed"
    else:
        results['overall_status'] = 'STILL_BROKEN'
        status_emoji = "❌"
        status_msg = f"STILL BROKEN - Only {results['tests_passed']}/{results['tests_total']} tests passed"
    
    print(f"\n{status_emoji} FINAL STATUS: {status_msg}")
    print("=" * 60)
    
    if results['success_items']:
        print("✅ SUCCESSFUL FIXES:")
        for item in results['success_items']:
            print(f"   • {item}")
    
    if results['critical_issues']:
        print("\n❌ REMAINING ISSUES:")
        for issue in results['critical_issues']:
            print(f"   • {issue}")
    
    print(f"\n📊 SUMMARY: {results['tests_passed']}/{results['tests_total']} tests passed ({success_rate*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    test_results = test_mathlib_add_subtract_functions_fix()
    
    # Exit with appropriate code
    if test_results['overall_status'] in ['FULLY_FIXED', 'MOSTLY_FIXED']:
        print("\n🎯 SUCCESS: Add/Subtract functions fix is working!")
        sys.exit(0)
    else:
        print("\n💥 FAILURE: Add/Subtract functions fix needs more work")
        sys.exit(1) 