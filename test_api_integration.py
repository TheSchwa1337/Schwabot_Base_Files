#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test API Integration
===================

Test script to verify Glassnode and Whale Alert API integration with ZPE/ZBE thermal system.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_api_integration():
    """Test the API integration system."""
    logger.info("🧪 Starting API Integration Test")
    
    try:
        # Import the integration manager
        from core.api_integration_manager import APIIntegrationManager
        
        # Create integration manager
        config = {
            "enabled": True,
            "integration_interval": 60,  # 1 minute for testing
            "glassnode_api_key": "demo-key",
            "whale_alert_api_key": "demo-key",
            "thermal_integration": True,
            "profit_scheduler_integration": True,
            "debug": True,
        }
        
        integration_manager = APIIntegrationManager(config)
        
        # Test 1: Check initialization
        logger.info("📋 Test 1: Checking initialization...")
        status = integration_manager.get_integration_status()
        logger.info(f"✅ Integration status: {status}")
        
        # Test 2: Start integration
        logger.info("📋 Test 2: Starting integration...")
        success = await integration_manager.start_integration()
        if success:
            logger.info("✅ Integration started successfully")
        else:
            logger.error("❌ Failed to start integration")
            return False
        
        # Test 3: Wait for first cycle
        logger.info("📋 Test 3: Waiting for first integration cycle...")
        await asyncio.sleep(70)  # Wait for first cycle to complete
        
        # Test 4: Check results
        logger.info("📋 Test 4: Checking integration results...")
        signals = integration_manager.get_latest_signals()
        if signals:
            logger.info("✅ Integration signals generated:")
            logger.info(f"   - Glassnode signals: {signals.get('glassnode_signals', {})}")
            logger.info(f"   - Whale Alert signals: {signals.get('whale_alert_signals', {})}")
            logger.info(f"   - Combined signals: {signals.get('combined_signals', {})}")
        else:
            logger.warning("⚠️ No integration signals available yet")
        
        # Test 5: Stop integration
        logger.info("📋 Test 5: Stopping integration...")
        await integration_manager.stop_integration()
        logger.info("✅ Integration stopped successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def test_individual_handlers():
    """Test individual API handlers."""
    logger.info("🧪 Testing Individual API Handlers")
    
    try:
        # Test Glassnode Handler
        logger.info("📋 Testing Glassnode Handler...")
        from core.api.handlers.glassnode import GlassnodeHandler
        
        glassnode = GlassnodeHandler(api_key="demo-key")
        
        # Test thermal integration data
        thermal_data = glassnode.get_thermal_integration_data()
        logger.info(f"✅ Glassnode thermal data: {thermal_data}")
        
        # Test profit scheduler data
        scheduler_data = glassnode.get_profit_scheduler_data()
        logger.info(f"✅ Glassnode scheduler data: {scheduler_data}")
        
        # Test Whale Alert Handler
        logger.info("📋 Testing Whale Alert Handler...")
        from core.api.handlers.whale_alert import WhaleAlertHandler
        
        whale_alert = WhaleAlertHandler(api_key="demo-key")
        
        # Test thermal integration data
        thermal_data = whale_alert.get_thermal_integration_data()
        logger.info(f"✅ Whale Alert thermal data: {thermal_data}")
        
        # Test profit scheduler data
        scheduler_data = whale_alert.get_profit_scheduler_data()
        logger.info(f"✅ Whale Alert scheduler data: {scheduler_data}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Handler test failed: {e}")
        return False


async def test_thermal_integration():
    """Test thermal system integration."""
    logger.info("🧪 Testing Thermal System Integration")
    
    try:
        # Test thermal strategy router
        logger.info("📋 Testing Thermal Strategy Router...")
        from core.thermal_strategy_router import ThermalStrategyRouter
        
        thermal_router = ThermalStrategyRouter()
        mode = thermal_router.determine_mode()
        logger.info(f"✅ Thermal mode determined: {mode}")
        
        # Test heartbeat integration manager
        logger.info("📋 Testing Heartbeat Integration Manager...")
        from core.heartbeat_integration_manager import HeartbeatIntegrationManager
        
        heartbeat_manager = HeartbeatIntegrationManager()
        stats = heartbeat_manager.get_integration_stats()
        logger.info(f"✅ Heartbeat stats: {stats}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Thermal integration test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting Comprehensive API Integration Tests")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Individual handlers
    logger.info("\n🔍 Test Suite 1: Individual API Handlers")
    result1 = await test_individual_handlers()
    test_results.append(("Individual Handlers", result1))
    
    # Test 2: Thermal integration
    logger.info("\n🔍 Test Suite 2: Thermal System Integration")
    result2 = await test_thermal_integration()
    test_results.append(("Thermal Integration", result2))
    
    # Test 3: Full integration
    logger.info("\n🔍 Test Suite 3: Full API Integration")
    result3 = await test_api_integration()
    test_results.append(("Full Integration", result3))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! API integration is working correctly.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Check the logs for details.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        sys.exit(1) 