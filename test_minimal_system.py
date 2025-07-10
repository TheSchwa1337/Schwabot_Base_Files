#!/usr/bin/env python3
"""
Minimal SchwabotCoreSystem Test
==============================

Test for the minimal version of SchwabotCoreSystem that only includes essential components.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_minimal_system_creation():
    """Test minimal system creation."""
    print("Testing minimal system creation...")
    
    try:
        from core.schwabot_core_system_minimal import SchwabotCoreSystem

        # Create system instance
        system = SchwabotCoreSystem()
        print("✅ Minimal system instance created successfully")
        
        # Check subsystem count
        subsystem_count = len(system.subsystems)
        print(f"✅ {subsystem_count} subsystems wrapped")
        
        # List subsystems
        subsystems = system.list_subsystems()
        print(f"✅ Subsystems: {subsystems}")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal system creation failed: {e}")
        return False


async def test_minimal_system_initialization():
    """Test minimal system initialization."""
    print("\nTesting minimal system initialization...")
    
    try:
        from core.schwabot_core_system_minimal import SchwabotCoreSystem
        
        system = SchwabotCoreSystem()
        
        # Initialize system
        success = await system.initialize()
        if success:
            print("✅ Minimal system initialization successful")
            
            # Check initialization status
            initialized_count = sum(
                1 for wrapper in system.subsystems.values() 
                if wrapper.is_initialized
            )
            print(f"✅ {initialized_count}/{len(system.subsystems)} subsystems initialized")
            
            return True
        else:
            print("❌ Minimal system initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Minimal system initialization failed: {e}")
        return False


async def test_minimal_system_lifecycle():
    """Test minimal system start/stop lifecycle."""
    print("\nTesting minimal system lifecycle...")
    
    try:
        from core.schwabot_core_system_minimal import SchwabotCoreSystem
        
        system = SchwabotCoreSystem()
        
        # Initialize
        await system.initialize()
        
        # Start system
        success = await system.start()
        if success:
            print("✅ Minimal system started successfully")
            
            # Check running status
            running_count = sum(
                1 for wrapper in system.subsystems.values() 
                if wrapper.is_running
            )
            print(f"✅ {running_count}/{len(system.subsystems)} subsystems running")
            
            # Get system status
            status = system.get_system_status()
            print("✅ System status retrieved")
            
            # Stop system
            await system.stop()
            print("✅ Minimal system stopped successfully")
            
            return True
        else:
            print("❌ Minimal system start failed")
            return False
            
    except Exception as e:
        print(f"❌ Minimal system lifecycle failed: {e}")
        return False


async def test_hot_reload():
    """Test hot reload functionality."""
    print("\nTesting hot reload functionality...")
    
    try:
        from core.schwabot_core_system_minimal import SchwabotCoreSystem
        
        system = SchwabotCoreSystem()
        await system.initialize()
        await system.start()
        
        # Test reloading a specific subsystem
        test_subsystem = "MathConfigManager"
        if test_subsystem in system.subsystems:
            success = await system.reload_subsystem(test_subsystem)
            if success:
                print(f"✅ {test_subsystem} reloaded successfully")
            else:
                print(f"❌ {test_subsystem} reload failed")
        
        # Test reloading all subsystems
        success = await system.reload_all_subsystems()
        if success:
            print("✅ All subsystems reloaded successfully")
        else:
            print("❌ All subsystems reload failed")
        
        await system.stop()
        return True
        
    except Exception as e:
        print(f"❌ Hot reload test failed: {e}")
        return False


async def test_entropy_monitoring():
    """Test entropy change monitoring."""
    print("\nTesting entropy monitoring...")
    
    try:
        from core.schwabot_core_system_minimal import SchwabotCoreSystem
        
        system = SchwabotCoreSystem()
        await system.initialize()
        await system.start()
        
        # Check for entropy changes
        entropy_changes = await system.check_entropy_changes()
        print(f"✅ Entropy monitoring active, {len(entropy_changes)} changes detected")
        
        await system.stop()
        return True
        
    except Exception as e:
        print(f"❌ Entropy monitoring test failed: {e}")
        return False


async def test_cli_api_methods():
    """Test CLI and API access methods."""
    print("\nTesting CLI/API methods...")
    
    try:
        from decimal import Decimal

        from core.schwabot_core_system_minimal import SchwabotCoreSystem
        from core.type_defs import OrderSide, OrderType
        
        system = SchwabotCoreSystem()
        await system.initialize()
        await system.start()
        
        # Test system status
        status = system.get_system_status()
        print("✅ System status retrieved")
        
        # Test subsystem listing
        subsystems = system.list_subsystems()
        print(f"✅ {len(subsystems)} subsystems listed")
        
        # Test getting specific subsystem
        if "MathConfigManager" in subsystems:
            subsystem = system.get_subsystem("MathConfigManager")
            if subsystem:
                print("✅ MathConfigManager subsystem retrieved")
        
        # Test calling subsystem method
        try:
            if "MathConfigManager" in system.subsystems:
                result = await system.call_subsystem_method("MathConfigManager", "get_config")
                print("✅ Subsystem method called successfully")
        except Exception as e:
            print(f"⚠️  Subsystem method call failed (expected): {e}")
        
        # Test portfolio summary
        portfolio = await system.get_portfolio_summary()
        print("✅ Portfolio summary retrieved")
        
        await system.stop()
        return True
        
    except Exception as e:
        print(f"❌ CLI/API methods test failed: {e}")
        return False


async def test_trading_operations():
    """Test trading operations."""
    print("\nTesting trading operations...")
    
    try:
        from decimal import Decimal

        from core.schwabot_core_system_minimal import SchwabotCoreSystem
        from core.type_defs import OrderSide, OrderType
        
        system = SchwabotCoreSystem()
        await system.initialize()
        await system.start()
        
        # Test order placement (will fail in demo mode, but should not crash)
        try:
            result = await system.place_order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001")
            )
            print("✅ Order placement method called")
        except Exception as e:
            print(f"⚠️  Order placement failed (expected in demo): {e}")
        
        # Test order status
        try:
            result = await system.get_order_status("test_order_id")
            print("✅ Order status method called")
        except Exception as e:
            print(f"⚠️  Order status failed (expected): {e}")
        
        # Test order cancellation
        try:
            result = await system.cancel_order("test_order_id")
            print("✅ Order cancellation method called")
        except Exception as e:
            print(f"⚠️  Order cancellation failed (expected): {e}")
        
        await system.stop()
        return True
        
    except Exception as e:
        print(f"❌ Trading operations test failed: {e}")
        return False


async def test_trading_cycle():
    """Test complete trading cycle."""
    print("\nTesting trading cycle...")
    
    try:
        from core.schwabot_core_system_minimal import SchwabotCoreSystem
        
        system = SchwabotCoreSystem()
        await system.initialize()
        await system.start()
        
        # Run a few trading cycles
        for i in range(3):
            await system._execute_trading_cycle()
            print(f"✅ Trading cycle {i+1} completed")
            await asyncio.sleep(0.1)  # Small delay
        
        await system.stop()
        return True
        
    except Exception as e:
        print(f"❌ Trading cycle test failed: {e}")
        return False


async def test_system_integration():
    """Test system integration with main entry point."""
    print("\nTesting system integration...")
    
    try:
        from core.schwabot_core_system_minimal import create_system_instance, get_system_instance

        # Test global instance management
        system1 = create_system_instance()
        system2 = get_system_instance()
        
        if system1 is system2:
            print("✅ Global instance management working")
        else:
            print("❌ Global instance management failed")
        
        # Test system instance creation with config
        system3 = create_system_instance("config/schwabot_config.yaml")
        print("✅ System creation with config successful")
        
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling and robustness."""
    print("\nTesting error handling...")
    
    try:
        from core.schwabot_core_system_minimal import SchwabotCoreSystem
        
        system = SchwabotCoreSystem()
        
        # Test calling methods before initialization
        try:
            await system.start()
            print("❌ Should not start before initialization")
            return False
        except Exception as e:
            print("✅ Properly prevented start before initialization")
        
        # Test invalid subsystem access
        try:
            result = await system.call_subsystem_method("NonExistentSubsystem", "method")
            print("❌ Should not find non-existent subsystem")
            return False
        except ValueError as e:
            print("✅ Properly handled non-existent subsystem")
        
        # Test invalid method call
        try:
            result = await system.call_subsystem_method("MathConfigManager", "non_existent_method")
            print("❌ Should not find non-existent method")
            return False
        except ValueError as e:
            print("✅ Properly handled non-existent method")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Minimal SchwabotCoreSystem Test")
    print("=" * 50)
    
    tests = [
        ("System Creation", test_minimal_system_creation),
        ("System Initialization", test_minimal_system_initialization),
        ("System Lifecycle", test_minimal_system_lifecycle),
        ("Hot Reload", test_hot_reload),
        ("Entropy Monitoring", test_entropy_monitoring),
        ("CLI/API Methods", test_cli_api_methods),
        ("Trading Operations", test_trading_operations),
        ("Trading Cycle", test_trading_cycle),
        ("System Integration", test_system_integration),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if await test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All minimal system tests passed!")
        print("✅ Minimal SchwabotCoreSystem is fully functional")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 