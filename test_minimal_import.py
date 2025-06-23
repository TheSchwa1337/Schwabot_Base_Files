#!/usr/bin/env python3
"""
Minimal import test.
"""

print("Starting minimal import test...")

try:
    print("Importing basic modules...")
    import sys
    import os
    print("✅ Basic imports successful")
    
    print("Testing core.utils import...")
    from core.utils.windows_cli_compatibility import safe_print
    print("✅ Windows CLI compatibility imported")
    
    print("Testing gpt_command_layer_simple import...")
    from core.gpt_command_layer_simple import AIAgentType
    print("✅ AIAgentType imported")
    
    print("🎉 All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc() 