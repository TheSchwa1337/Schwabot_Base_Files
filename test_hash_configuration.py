#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hash Configuration System Test
=============================
Test script to verify the centralized hash configuration system works correctly
across different hardware scenarios and CLI options.
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import hash configuration manager
from core.hash_config_manager import (
    hash_config_manager, 
    get_hash_config, 
    get_hash_settings, 
    generate_hash, 
    generate_hash_from_string,
    HardwareTier
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_hardware_detection():
    """Test hardware detection and tier assignment."""
    logger.info("🔍 Testing Hardware Detection")
    logger.info("=" * 50)
    
    try:
        # Test hardware tier detection
        hardware_tier = hash_config_manager._detect_hardware_tier()
        logger.info(f"✅ Detected hardware tier: {hardware_tier.value}")
        
        # Test Raspberry Pi detection
        is_pi = hash_config_manager._is_raspberry_pi()
        logger.info(f"✅ Raspberry Pi detection: {is_pi}")
        
        # Test Pi Zero/Pico detection
        is_pi_zero = hash_config_manager._is_pi_zero_or_pico()
        logger.info(f"✅ Pi Zero/Pico detection: {is_pi_zero}")
        
        # Test Pi 3 detection
        is_pi_3 = hash_config_manager._is_pi_3()
        logger.info(f"✅ Pi 3 detection: {is_pi_3}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hardware detection test failed: {e}")
        return False

def test_hash_configuration():
    """Test hash configuration initialization."""
    logger.info("🔧 Testing Hash Configuration")
    logger.info("=" * 50)
    
    try:
        # Test default initialization
        hash_config_manager.initialize()
        config = get_hash_config()
        settings = get_hash_settings()
        
        logger.info(f"✅ Default config - truncated: {config.truncated_hash}, length: {config.hash_length}")
        logger.info(f"✅ Hardware tier: {config.hardware_tier.value}")
        logger.info(f"✅ Auto-detected: {config.auto_detected}")
        logger.info(f"✅ CLI override: {config.cli_override}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hash configuration test failed: {e}")
        return False

def test_cli_override():
    """Test CLI override functionality."""
    logger.info("🎛️ Testing CLI Override")
    logger.info("=" * 50)
    
    try:
        # Test truncated hash override
        hash_config_manager.initialize(cli_truncated_hash=True, cli_hash_length=12)
        config = get_hash_config()
        
        logger.info(f"✅ CLI override - truncated: {config.truncated_hash}, length: {config.hash_length}")
        logger.info(f"✅ CLI override flag: {config.cli_override}")
        
        # Test custom hash length override
        hash_config_manager.initialize(cli_truncated_hash=False, cli_hash_length=32)
        config = get_hash_config()
        
        logger.info(f"✅ Custom length override - truncated: {config.truncated_hash}, length: {config.hash_length}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CLI override test failed: {e}")
        return False

def test_hash_generation():
    """Test hash generation with different configurations."""
    logger.info("🔐 Testing Hash Generation")
    logger.info("=" * 50)
    
    try:
        test_data = b"Hello, Schwabot Hash Configuration System!"
        test_string = "Test string for hash generation"
        
        # Test with default configuration
        hash_config_manager.initialize()
        default_hash = generate_hash(test_data)
        default_string_hash = generate_hash_from_string(test_string)
        
        logger.info(f"✅ Default hash (data): {default_hash[:16]}...")
        logger.info(f"✅ Default hash (string): {default_string_hash[:16]}...")
        
        # Test with truncated configuration
        hash_config_manager.initialize(cli_truncated_hash=True, cli_hash_length=8)
        truncated_hash = generate_hash(test_data)
        truncated_string_hash = generate_hash_from_string(test_string)
        
        logger.info(f"✅ Truncated hash (data): {truncated_hash}")
        logger.info(f"✅ Truncated hash (string): {truncated_string_hash}")
        
        # Verify truncation
        assert len(truncated_hash) == 8, f"Expected 8 characters, got {len(truncated_hash)}"
        assert len(truncated_string_hash) == 8, f"Expected 8 characters, got {len(truncated_string_hash)}"
        
        logger.info("✅ Hash truncation verified")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hash generation test failed: {e}")
        return False

def test_hardware_tier_configurations():
    """Test different hardware tier configurations."""
    logger.info("🏗️ Testing Hardware Tier Configurations")
    logger.info("=" * 50)
    
    try:
        test_data = b"Hardware tier test data"
        
        # Test Pi Zero/Pico configuration
        hash_config_manager.config.hardware_tier = HardwareTier.PI_ZERO_PICO
        hash_config_manager.config.truncated_hash = True
        hash_config_manager.config.hash_length = 8
        
        pi_zero_hash = generate_hash(test_data)
        logger.info(f"✅ Pi Zero/Pico hash: {pi_zero_hash}")
        
        # Test Pi 3 configuration
        hash_config_manager.config.hardware_tier = HardwareTier.PI_3
        hash_config_manager.config.hash_length = 12
        
        pi_3_hash = generate_hash(test_data)
        logger.info(f"✅ Pi 3 hash: {pi_3_hash}")
        
        # Test Pi 4/Mobile configuration
        hash_config_manager.config.hardware_tier = HardwareTier.PI_4_MOBILE
        hash_config_manager.config.hash_length = 16
        
        pi_4_hash = generate_hash(test_data)
        logger.info(f"✅ Pi 4/Mobile hash: {pi_4_hash}")
        
        # Test Desktop/Server configuration
        hash_config_manager.config.hardware_tier = HardwareTier.DESKTOP_SERVER
        hash_config_manager.config.truncated_hash = False
        hash_config_manager.config.hash_length = 64
        
        desktop_hash = generate_hash(test_data)
        logger.info(f"✅ Desktop/Server hash: {desktop_hash[:16]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hardware tier configuration test failed: {e}")
        return False

def test_system_status():
    """Test system status reporting."""
    logger.info("📊 Testing System Status")
    logger.info("=" * 50)
    
    try:
        status = hash_config_manager.get_status()
        
        logger.info("✅ System Status:")
        logger.info(f"   Config: {status['config']}")
        logger.info(f"   Platform: {status['hardware_info']['platform']}")
        logger.info(f"   Architecture: {status['hardware_info']['architecture']}")
        logger.info(f"   CPU Count: {status['hardware_info']['cpu_count']}")
        logger.info(f"   Memory (GB): {status['hardware_info']['memory_gb']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System status test failed: {e}")
        return False

def main():
    """Run all hash configuration tests."""
    logger.info("🚀 Starting Hash Configuration System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Hardware Detection", test_hardware_detection),
        ("Hash Configuration", test_hash_configuration),
        ("CLI Override", test_cli_override),
        ("Hash Generation", test_hash_generation),
        ("Hardware Tier Configurations", test_hardware_tier_configurations),
        ("System Status", test_system_status),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} Test")
        logger.info("-" * 40)
        
        try:
            if test_func():
                logger.info(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} test FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
            failed += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Tests Passed: {passed}")
    logger.info(f"❌ Tests Failed: {failed}")
    logger.info(f"📊 Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Hash configuration system is working correctly.")
        return 0
    else:
        logger.error("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 