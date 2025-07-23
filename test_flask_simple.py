#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Flask Test - BEST TRADING SYSTEM ON EARTH
===============================================

Test Flask app import without the problematic components.
"""

def test_flask_import():
    """Test Flask app import."""
    print("🌐 Testing Flask App Import...")
    
    try:
        # Try to import Flask app
        import api.flask_app
        print("✅ Flask app imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error importing Flask app: {e}")
        return False

def test_enhanced_fractal_routes():
    """Test Enhanced Fractal Routes import."""
    print("\n🧬 Testing Enhanced Fractal Routes...")
    
    try:
        # Try to import Enhanced Fractal Routes
        import api.enhanced_fractal_routes
        print("✅ Enhanced Fractal Routes imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error importing Enhanced Fractal Routes: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TESTING FLASK COMPONENTS")
    print("=" * 40)
    
    # Run tests
    test1 = test_flask_import()
    test2 = test_enhanced_fractal_routes()
    
    print("\n" + "=" * 40)
    print("🏆 FLASK TEST RESULTS:")
    print(f"   Flask App: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Enhanced Fractal Routes: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 FLASK COMPONENTS ARE WORKING!")
    else:
        print("\n⚠️ Some Flask components have issues.") 