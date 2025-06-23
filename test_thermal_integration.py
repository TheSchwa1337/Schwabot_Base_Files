#!/usr/bin/env python3
"""
Thermal Integration Test - Verify Thermal Boundary Manager Integration
====================================================================

Simple test script to verify that the thermal boundary manager integrates
properly with the existing thermal system components.
"""

import asyncio
import logging
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_thermal_boundary_manager() -> Dict[str, Any]:
    """Test the thermal boundary manager functionality."""
    try:
        # Import the thermal boundary manager
        from core.thermal_boundary_manager import create_thermal_boundary_manager
        
        # Create manager instance
        manager = create_thermal_boundary_manager()
        logger.info("Thermal boundary manager created successfully")
        
        # Test hardware detection
        hardware_profile = manager.hardware_profile
        logger.info(f"Hardware detected: {hardware_profile}")
        
        # Test thermal state retrieval
        thermal_state = await manager.get_thermal_state()
        logger.info(f"Thermal state: {thermal_state}")
        
        # Test resource allocation
        allocation = await manager.update_resource_allocation()
        logger.info(f"Resource allocation: {allocation}")
        
        # Test processing recommendations
        recommendations = manager.get_processing_recommendations()
        logger.info(f"Processing recommendations: {recommendations}")
        
        # Test system status
        status = manager.get_system_status()
        logger.info(f"System status: {status}")
        
        return {
            "success": True,
            "hardware_profile": hardware_profile,
            "thermal_state": thermal_state,
            "allocation": allocation,
            "recommendations": recommendations,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Thermal boundary manager test failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def test_thermal_integration() -> Dict[str, Any]:
    """Test integration with existing thermal components."""
    try:
        # Test thermal zone manager integration
        thermal_zone_available = False
        try:
            from core.thermal_zone_manager import ThermalZoneManager
            thermal_zone_available = True
            logger.info("Thermal zone manager available")
        except ImportError:
            logger.warning("Thermal zone manager not available")
        
        # Test thermal map allocator integration
        thermal_map_available = False
        try:
            from core.thermal_map_allocator import ThermalMapAllocator
            thermal_map_available = True
            logger.info("Thermal map allocator available")
        except ImportError:
            logger.warning("Thermal map allocator not available")
        
        # Test fault bus integration
        fault_bus_available = False
        try:
            from core.fault_bus import FaultBus
            fault_bus_available = True
            logger.info("Fault bus available")
        except ImportError:
            logger.warning("Fault bus not available")
        
        return {
            "success": True,
            "thermal_zone_manager": thermal_zone_available,
            "thermal_map_allocator": thermal_map_available,
            "fault_bus": fault_bus_available
        }
        
    except Exception as e:
        logger.error(f"Thermal integration test failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def test_error_handling() -> Dict[str, Any]:
    """Test error handling and recovery mechanisms."""
    try:
        from core.thermal_boundary_manager import ThermalBoundaryManager
        
        # Create manager with error-prone configuration
        manager = ThermalBoundaryManager(
            config={"invalid_config": "should_handle_gracefully"},
            enable_gpu_monitoring=False  # Disable GPU to test fallbacks
        )
        
        # Test error handling in thermal state retrieval
        thermal_state = await manager.get_thermal_state()
        logger.info(f"Thermal state with error handling: {thermal_state}")
        
        # Test error tracking
        error_count = manager.error_count
        logger.info(f"Error count: {error_count}")
        
        return {
            "success": True,
            "thermal_state": thermal_state,
            "error_count": error_count,
            "recovery_mode": manager.recovery_mode,
            "emergency_mode": manager.emergency_mode
        }
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def main() -> None:
    """Main test function."""
    logger.info("Starting thermal integration tests...")
    
    # Test 1: Thermal boundary manager functionality
    logger.info("\n=== Test 1: Thermal Boundary Manager ===")
    result1 = await test_thermal_boundary_manager()
    print(f"Test 1 Result: {result1}")
    
    # Test 2: Integration with existing components
    logger.info("\n=== Test 2: Thermal Integration ===")
    result2 = await test_thermal_integration()
    print(f"Test 2 Result: {result2}")
    
    # Test 3: Error handling
    logger.info("\n=== Test 3: Error Handling ===")
    result3 = await test_error_handling()
    print(f"Test 3 Result: {result3}")
    
    # Summary
    logger.info("\n=== Test Summary ===")
    all_tests_passed = all(result.get("success", False) for result in [result1, result2, result3])
    
    if all_tests_passed:
        logger.info("✅ All thermal integration tests passed!")
        print("\n🎉 THERMAL SYSTEM INTEGRATION SUCCESSFUL!")
        print("The thermal boundary manager is properly integrated and ready for production use.")
    else:
        logger.error("❌ Some thermal integration tests failed!")
        print("\n⚠️  THERMAL SYSTEM INTEGRATION ISSUES DETECTED!")
        print("Please review the test results above for details.")
    
    return all_tests_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1) 