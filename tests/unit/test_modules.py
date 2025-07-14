#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to check module availability
"""

import sys
import os

def test_module(module_name, class_name=None):
    """Test if a module can be imported."""
    try:
        if class_name:
            exec(f"from {module_name} import {class_name}")
            print(f"✅ {module_name} - {class_name} available")
            return True
        else:
            exec(f"import {module_name}")
            print(f"✅ {module_name} available")
            return True
    except ImportError as e:
        print(f"❌ {module_name} - ImportError: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name} - Error: {e}")
        return False

def main():
    print("🔍 Testing module availability...")
    print("=" * 50)
    
    # Test core modules
    modules_to_test = [
        ("core.enhanced_entropy_randomization_system", "EnhancedEntropyRandomizationSystem"),
        ("core.self_generating_strategy_system", "SelfGeneratingStrategySystem"),
        ("core.unified_memory_registry_system", "UnifiedMemoryRegistrySystem"),
        ("core.unified_mathematical_bridge", "UnifiedMathematicalBridge"),
        ("core.unified_mathematical_integration_methods", "UnifiedMathematicalIntegrationMethods"),
        ("core.unified_mathematical_performance_monitor", "UnifiedMathematicalPerformanceMonitor"),
    ]
    
    available_count = 0
    total_count = len(modules_to_test)
    
    for module_name, class_name in modules_to_test:
        if test_module(module_name, class_name):
            available_count += 1
    
    print("=" * 50)
    print(f"📊 Results: {available_count}/{total_count} modules available")
    
    if available_count < total_count:
        print("\n🔧 Missing modules detected. Let's check what's available:")
        
        # Check if files exist
        print("\n📁 Checking file existence:")
        for module_name, class_name in modules_to_test:
            file_path = module_name.replace('.', '/') + '.py'
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
            else:
                print(f"❌ {file_path} missing")
    
    # Test some dependencies
    print("\n🔍 Testing dependencies:")
    dependencies = [
        "numpy",
        "asyncio",
        "logging",
        "time",
        "json",
        "hashlib"
    ]
    
    for dep in dependencies:
        test_module(dep)

if __name__ == "__main__":
    main() 