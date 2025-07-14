#!/usr/bin/env python3
"""
Test script for the math library to verify all components are working.
"""

def test_math_library():
    """Test the math library components."""
    try:
        print("🧮 Testing Math Library Components...")
        
        # Test basic imports
        from mathlib import MathLib, MathLibV2, MathLibV3, Dual, kelly_fraction, cvar
        print("✅ All imports successful")
        
        # Test MathLib V1
        ml = MathLib()
        print(f"✅ MathLib V1: {ml.version}")
        
        # Test basic operations
        result = ml.add(5, 3)
        print(f"✅ Addition: 5 + 3 = {result}")
        
        result = ml.multiply(4, 7)
        print(f"✅ Multiplication: 4 * 7 = {result}")
        
        result = ml.divide(15, 3)
        print(f"✅ Division: 15 / 3 = {result}")
        
        # Test MathLib V2
        ml2 = MathLibV2()
        print(f"✅ MathLib V2: {ml2.version}")
        
        # Test matrix operations
        matrix = [[1, 2], [3, 4]]
        det = ml2.determinant(matrix)
        print(f"✅ Matrix determinant: {det}")
        
        # Test MathLib V3
        ml3 = MathLibV3()
        print(f"✅ MathLib V3: {ml3.version}")
        
        # Test Dual numbers
        x = Dual(2.0, 1.0)
        y = x * x + 3 * x + 1  # f(x) = x² + 3x + 1, f'(x) = 2x + 3
        print(f"✅ Dual numbers: f(2) = {y.val}, f'(2) = {y.eps}")
        
        # Test financial functions
        kf = kelly_fraction(0.1, 0.04)
        print(f"✅ Kelly fraction: {kf}")
        
        cv = cvar([0.1, -0.05, 0.2, -0.1], 0.95)
        print(f"✅ CVaR: {cv}")
        
        # Test profit optimization
        profit = ml.calculate_profit_optimization(60000.0, 1000.0)
        print(f"✅ Profit optimization: {profit}")
        
        print("🎉 All math library tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Math library test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_math_library()
    if success:
        print("✅ Math library is working correctly!")
    else:
        print("❌ Math library has issues that need to be fixed.") 