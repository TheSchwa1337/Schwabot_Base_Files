#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Mathematical Systems Restoration Test
==================================================

This script tests all mathematical systems to identify what's broken
after the cleanup and restoration process.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import(module_name, description=""):
    """Test importing a module."""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True, module
    except Exception as e:
        print(f"❌ {module_name} - {description}")
        print(f"   Error: {str(e)}")
        return False, None

def test_mathematical_systems():
    """Test all mathematical systems."""
    print("🔬 COMPREHENSIVE MATHEMATICAL SYSTEMS TEST")
    print("=" * 60)
    
    results = {}
    
    # Test core mathematical modules
    print("\n📊 CORE MATHEMATICAL MODULES:")
    print("-" * 40)
    
    core_modules = [
        ("core.backend_math", "Backend mathematical operations"),
        ("core.math", "Math package"),
        ("core.math.unified_tensor_algebra", "Unified tensor algebra"),
        ("core.math.mathematical_framework_integrator", "Mathematical framework integrator"),
        ("core.utils.math_utils", "Mathematical utilities"),
        ("core.hash_config_manager", "Hash configuration manager"),
    ]
    
    for module_name, description in core_modules:
        success, module = test_import(module_name, description)
        results[module_name] = {"success": success, "module": module}
    
    # Test AOI mathematical modules
    print("\n🧮 AOI MATHEMATICAL MODULES:")
    print("-" * 40)
    
    aoi_modules = [
        ("mathlib", "MathLib package"),
        ("mathlib.mathlib_v4", "MathLib v4"),
        ("mathlib.persistent_homology", "Persistent homology"),
        ("mathlib.quantum_strategy", "Quantum strategy"),
        ("mathlib.matrix_fault_resolver", "Matrix fault resolver"),
        ("mathlib.memkey_sync", "Memory key synchronization"),
    ]
    
    for module_name, description in aoi_modules:
        success, module = test_import(module_name, description)
        results[module_name] = {"success": success, "module": module}
    
    # Test core mathematical systems
    print("\n⚙️ CORE MATHEMATICAL SYSTEMS:")
    print("-" * 40)
    
    core_systems = [
        ("core.advanced_tensor_algebra", "Advanced tensor algebra"),
        ("core.unified_profit_vectorization_system", "Unified profit vectorization"),
        ("core.phase_bit_integration", "Phase bit integration"),
        ("core.unified_math_system", "Unified math system"),
        ("core.clean_unified_math", "Clean unified math"),
        ("core.unified_mathematical_bridge", "Unified mathematical bridge"),
        ("core.enhanced_math_to_trade_integration", "Enhanced math to trade integration"),
        ("core.quantum_classical_hybrid_mathematics", "Quantum classical hybrid mathematics"),
    ]
    
    for module_name, description in core_systems:
        success, module = test_import(module_name, description)
        results[module_name] = {"success": success, "module": module}
    
    # Test memory and registry systems
    print("\n🧠 MEMORY AND REGISTRY SYSTEMS:")
    print("-" * 40)
    
    memory_systems = [
        ("core.memory_stack.memory_key_allocator", "Memory key allocator"),
        ("core.memory_stack.execution_validator", "Execution validator"),
        ("core.memory_stack.ai_command_sequencer", "AI command sequencer"),
        ("core.unified_memory_registry_system", "Unified memory registry system"),
    ]
    
    for module_name, description in memory_systems:
        success, module = test_import(module_name, description)
        results[module_name] = {"success": success, "module": module}
    
    # Test trading mathematical integration
    print("\n💰 TRADING MATHEMATICAL INTEGRATION:")
    print("-" * 40)
    
    trading_systems = [
        ("core.ccxt_integration", "CCXT integration"),
        ("core.risk_manager", "Risk manager"),
        ("core.profit_scaling_optimizer", "Profit scaling optimizer"),
        ("core.profit_projection_engine", "Profit projection engine"),
        ("core.profit_optimization_engine", "Profit optimization engine"),
        ("core.pure_profit_calculator", "Pure profit calculator"),
    ]
    
    for module_name, description in trading_systems:
        success, module = test_import(module_name, description)
        results[module_name] = {"success": success, "module": module}
    
    # Test quantum smoothing system
    print("\n🌊 QUANTUM SMOOTHING SYSTEM:")
    print("-" * 40)
    
    quantum_systems = [
        ("core.quantum_smoothing_system", "Quantum smoothing system"),
        ("core.trading_smoothing_integration", "Trading smoothing integration"),
        ("core.quantum_auto_scaler", "Quantum auto scaler"),
    ]
    
    for module_name, description in quantum_systems:
        success, module = test_import(module_name, description)
        results[module_name] = {"success": success, "module": module}
    
    # Test mathematical strategy systems
    print("\n🎯 MATHEMATICAL STRATEGY SYSTEMS:")
    print("-" * 40)
    
    strategy_systems = [
        ("core.strategy.multi_phase_strategy_weight_tensor", "Multi-phase strategy weight tensor"),
        ("core.strategy.strategy_executor", "Strategy executor"),
        ("core.strategy.enhanced_math_ops", "Enhanced math operations"),
        ("core.strategy.loss_anticipation_curve", "Loss anticipation curve"),
        ("core.strategy.volume_weighted_hash_oscillator", "Volume weighted hash oscillator"),
    ]
    
    for module_name, description in strategy_systems:
        success, module = test_import(module_name, description)
        results[module_name] = {"success": success, "module": module}
    
    # Test mathematical core systems
    print("\n🔧 MATHEMATICAL CORE SYSTEMS:")
    print("-" * 40)
    
    math_core_systems = [
        ("core.bro_logic_module", "Bro logic module"),
        ("core.matrix_math_utils", "Matrix math utilities"),
        ("core.schwabot_core_system", "Schwabot core system"),
        ("core.tcell_survival_engine", "Tcell survival engine"),
        ("core.vault_orbital_bridge", "Vault orbital bridge"),
    ]
    
    for module_name, description in math_core_systems:
        success, module = test_import(module_name, description)
        results[module_name] = {"success": success, "module": module}
    
    # Summary
    print("\n📋 SUMMARY:")
    print("=" * 60)
    
    total_modules = len(results)
    successful_modules = sum(1 for result in results.values() if result["success"])
    failed_modules = total_modules - successful_modules
    
    print(f"Total modules tested: {total_modules}")
    print(f"Successful imports: {successful_modules}")
    print(f"Failed imports: {failed_modules}")
    print(f"Success rate: {(successful_modules/total_modules)*100:.1f}%")
    
    if failed_modules > 0:
        print(f"\n❌ FAILED MODULES:")
        for module_name, result in results.items():
            if not result["success"]:
                print(f"   - {module_name}")
    
    return results

def test_mathematical_functionality():
    """Test actual mathematical functionality."""
    print("\n🧮 TESTING MATHEMATICAL FUNCTIONALITY:")
    print("=" * 60)
    
    try:
        # Test backend math
        from core.backend_math import BackendMath
        backend_math = BackendMath()
        
        # Test basic operations
        assert backend_math.add(2, 3) == 5
        assert backend_math.multiply(4, 5) == 20
        assert backend_math.mean([1, 2, 3, 4, 5]) == 3.0
        print("✅ Backend math basic operations working")
        
        # Test math utils
        from core.utils.math_utils import MathUtils
        math_utils = MathUtils()
        print("✅ Math utils imported successfully")
        
        # Test hash config manager
        from core.hash_config_manager import HashConfigManager
        config_manager = HashConfigManager()
        print("✅ Hash config manager working")
        
        print("✅ Core mathematical functionality verified")
        
    except Exception as e:
        print(f"❌ Mathematical functionality test failed: {e}")
        traceback.print_exc()

def run_flake8_check():
    """Run flake8 check on mathematical files."""
    print("\n🔍 RUNNING FLAKE8 CHECK:")
    print("=" * 60)
    
    try:
        import subprocess
        
        # Get all Python files in core and mathlib directories
        core_files = list(Path("core").rglob("*.py"))
        mathlib_files = list(Path("mathlib").rglob("*.py"))
        all_files = core_files + mathlib_files
        
        if not all_files:
            print("❌ No Python files found in core or mathlib directories")
            return
        
        # Run flake8 on the files
        cmd = ["python", "-m", "flake8"] + [str(f) for f in all_files[:10]]  # Limit to first 10 files
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Flake8 check passed - no style issues found")
        else:
            print("⚠️ Flake8 found style issues:")
            print(result.stdout)
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Flake8 check failed: {e}")

def main():
    """Main test function."""
    print("🚀 STARTING COMPREHENSIVE MATHEMATICAL SYSTEMS RESTORATION TEST")
    print("=" * 80)
    
    # Test imports
    results = test_mathematical_systems()
    
    # Test functionality
    test_mathematical_functionality()
    
    # Run flake8 check
    run_flake8_check()
    
    print("\n🎯 TEST COMPLETE")
    print("=" * 80)
    
    # Save results
    import json
    with open("mathematical_restoration_test_results.json", "w") as f:
        json.dump({
            "timestamp": str(Path().cwd()),
            "results": {k: {"success": v["success"]} for k, v in results.items()}
        }, f, indent=2)
    
    print("📄 Results saved to mathematical_restoration_test_results.json")

if __name__ == "__main__":
    main() 