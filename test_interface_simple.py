#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test to isolate the Dict import issue in the unified interface.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test basic imports."""
    try:
        from typing import Dict, Any, List, Optional
        print("✅ Basic typing imports successful")
        
        from dataclasses import dataclass, field
        print("✅ Dataclass imports successful")
        
        from enum import Enum
        print("✅ Enum imports successful")
        
        from datetime import datetime
        print("✅ Datetime imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_flask_imports():
    """Test Flask imports."""
    try:
        from flask import Flask, jsonify, render_template, request, session, Response
        print("✅ Flask imports successful")
        
        from flask_cors import CORS
        print("✅ Flask-CORS imports successful")
        
        from flask_socketio import SocketIO, emit, join_room, leave_room
        print("✅ Flask-SocketIO imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Flask import error: {e}")
        return False

def test_unified_interface_import():
    """Test unified interface import."""
    try:
        print("🔄 Attempting to import unified interface...")
        from gui.unified_schwabot_interface import SchwabotUnifiedInterface
        print("✅ Unified interface import successful")
        
        print("🔄 Attempting to instantiate interface...")
        interface = SchwabotUnifiedInterface()
        print("✅ Interface instantiation successful")
        
        return True
    except Exception as e:
        print(f"❌ Unified interface error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Simple Interface Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Flask Imports", test_flask_imports),
        ("Unified Interface", test_unified_interface_import),
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")

if __name__ == "__main__":
    main() 