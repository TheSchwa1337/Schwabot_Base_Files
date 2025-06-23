#!/usr/bin/env python3
"""
Schwabot Unified Interface Launcher
===================================

This script launches the Schwabot Unified Dual-Interface System,
which provides both practical monitoring and unified configuration interfaces.

Usage:
    python launch_unified_interface.py
"""

import sys
import os

# Add the core directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

def main():
    """Launch the unified interface system"""
    try:
        print("🚀 Starting Schwabot Unified Dual-Interface System...")
        print("=" * 60)
        
        # Import and run the unified interface
        from core.schwabot_unified_interface_system import main as run_interface
        run_interface()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all required dependencies are installed:")
        print("  pip install tkinter matplotlib numpy")
        return 1
        
    except Exception as e:
        print(f"❌ Error starting interface: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 