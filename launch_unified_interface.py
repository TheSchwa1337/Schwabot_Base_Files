#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Unified Interface Launcher
===================================
Launcher script for the Schwabot Unified Trading Interface.
Provides easy startup, configuration, and access to the trading terminal.

Usage:
    python launch_unified_interface.py [--host HOST] [--port PORT] [--debug]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/unified_interface.log')
        ]
    )

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'flask',
        'flask_cors',
        'flask_socketio',
        'numpy',
        'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_configuration():
    """Check if required configuration files exist."""
    required_files = [
        'config/unified_settings.yaml',
        'config/coinbase_profiles.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing configuration files: {', '.join(missing_files)}")
        print("Some features may not work properly.")
        return False
    
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'logs',
        'data',
        'visualizations',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def display_startup_banner():
    """Display Schwabot startup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🌀 SCHWABOT v0.5 🌀                      ║
    ║                                                              ║
    ║              Unified Trading Interface Launcher              ║
    ║                                                              ║
    ║              Advanced AI-Powered Trading System              ║
    ║                                                              ║
    ║              Multi-Profile • GPU Accelerated                 ║
    ║              Real-Time • Strategy Execution                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='Launch Schwabot Unified Trading Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-banner', action='store_true', help='Skip startup banner')
    
    args = parser.parse_args()
    
    # Display banner
    if not args.no_banner:
        display_startup_banner()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Schwabot Unified Interface Launcher...")
    
    # Check dependencies
    logger.info("🔍 Checking dependencies...")
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        sys.exit(1)
    logger.info("✅ Dependencies check passed")
    
    # Check configuration
    logger.info("🔍 Checking configuration...")
    check_configuration()
    
    # Create directories
    logger.info("📁 Creating directories...")
    create_directories()
    logger.info("✅ Directories created")
    
    # Import and start the unified interface
    try:
        logger.info("🔄 Importing unified interface...")
        from gui.unified_schwabot_interface import SchwabotUnifiedInterface
        
        logger.info("✅ Unified interface imported successfully")
        
        # Create interface instance
        interface = SchwabotUnifiedInterface()
        
        # Display startup information
        print(f"\n🎯 Schwabot Unified Interface Starting...")
        print(f"📍 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"🐛 Debug: {args.debug}")
        print(f"🌐 URL: http://{args.host}:{args.port}")
        print(f"📊 Dashboard: http://{args.host}:{args.port}/")
        print(f"📈 API Status: http://{args.host}:{args.port}/api/system/status")
        print("\n" + "="*60)
        
        logger.info(f"🚀 Starting interface on {args.host}:{args.port}")
        
        # Start the interface
        interface.start(host=args.host, port=args.port, debug=args.debug)
        
    except ImportError as e:
        logger.error(f"❌ Failed to import unified interface: {e}")
        print(f"\n❌ Error: {e}")
        print("Please ensure all required modules are available.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start interface: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 