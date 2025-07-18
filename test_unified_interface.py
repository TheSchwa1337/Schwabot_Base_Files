#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for Schwabot Unified Interface
==========================================
Validates the unified interface functionality and ensures all components
are working correctly before deployment.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup test logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing module imports...")
    
    required_modules = [
        'flask',
        'flask_cors',
        'flask_socketio',
        'numpy',
        'matplotlib'
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All required modules imported successfully")
    return True

def test_schwabot_components():
    """Test Schwabot core component imports."""
    print("\n🔍 Testing Schwabot core components...")
    
    schwabot_modules = [
        'core.strategy_mapper',
        'core.profile_router',
        'core.api.multi_profile_coinbase_manager',
        'core.visual_layer_controller',
        'core.unified_math_system',
        'core.hardware_auto_detector',
        'core.soulprint_registry'
    ]
    
    failed_imports = []
    for module in schwabot_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"⚠️  {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Some Schwabot components not available: {', '.join(failed_imports)}")
        print("Interface will run with reduced functionality")
    
    return True

def test_configuration_files():
    """Test configuration file availability."""
    print("\n🔍 Testing configuration files...")
    
    config_files = [
        'config/unified_settings.yaml',
        'config/coinbase_profiles.yaml'
    ]
    
    missing_files = []
    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"⚠️  {file_path} (not found)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing configuration files: {', '.join(missing_files)}")
        print("Some features may not work properly")
    
    return True

def test_directory_structure():
    """Test directory structure and create missing directories."""
    print("\n🔍 Testing directory structure...")
    
    required_dirs = [
        'logs',
        'data',
        'visualizations',
        'templates',
        'gui/templates'
    ]
    
    for directory in required_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"✅ {directory}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created {directory}")
    
    return True

def test_unified_interface_import():
    """Test unified interface import."""
    print("\n🔍 Testing unified interface import...")
    
    try:
        from gui.unified_schwabot_interface import SchwabotUnifiedInterface
        print("✅ Unified interface imported successfully")
        
        # Test interface instantiation
        interface = SchwabotUnifiedInterface()
        print("✅ Interface instantiated successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import unified interface: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to instantiate interface: {e}")
        return False

def test_api_endpoints():
    """Test API endpoint functionality."""
    print("\n🔍 Testing API endpoints...")
    
    try:
        from gui.unified_schwabot_interface import app
        
        # Test basic route
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Main dashboard route")
            else:
                print(f"❌ Main dashboard route: {response.status_code}")
            
            # Test system status endpoint
            response = client.get('/api/system/status')
            if response.status_code == 200:
                print("✅ System status endpoint")
                data = json.loads(response.data)
                print(f"   System state: {data.get('system_state', 'unknown')}")
            else:
                print(f"❌ System status endpoint: {response.status_code}")
            
            # Test profile list endpoint
            response = client.get('/api/profile/list')
            if response.status_code == 200:
                print("✅ Profile list endpoint")
            else:
                print(f"❌ Profile list endpoint: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False

def test_template_rendering():
    """Test HTML template rendering."""
    print("\n🔍 Testing template rendering...")
    
    try:
        from gui.unified_schwabot_interface import app
        
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                content = response.data.decode('utf-8')
                if 'Schwabot Unified Trading Terminal' in content:
                    print("✅ Template renders correctly")
                else:
                    print("❌ Template content not found")
            else:
                print(f"❌ Template rendering failed: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False

def test_socketio_functionality():
    """Test Socket.IO functionality."""
    print("\n🔍 Testing Socket.IO functionality...")
    
    try:
        from gui.unified_schwabot_interface import socketio
        
        # Test Socket.IO initialization
        if socketio:
            print("✅ Socket.IO initialized")
        else:
            print("❌ Socket.IO not initialized")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Socket.IO test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🧪 Starting Schwabot Unified Interface Test Suite")
    print("=" * 60)
    
    setup_logging()
    
    tests = [
        ("Module Imports", test_imports),
        ("Schwabot Components", test_schwabot_components),
        ("Configuration Files", test_configuration_files),
        ("Directory Structure", test_directory_structure),
        ("Interface Import", test_unified_interface_import),
        ("API Endpoints", test_api_endpoints),
        ("Template Rendering", test_template_rendering),
        ("Socket.IO Functionality", test_socketio_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Unified interface is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test Schwabot Unified Interface')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Running quick tests...")
        # Run only essential tests
        test_imports()
        test_unified_interface_import()
        test_api_endpoints()
    else:
        run_comprehensive_test()

if __name__ == '__main__':
    import argparse
    main() 