#!/usr/bin/env python3
"""
Schwabot Altitude Dashboard Launcher
====================================
Launch script for the Schwabot altitude adjustment visualization dashboard.
Handles initialization, setup, and graceful startup/shutdown.
"""

import sys
import os
import logging
import subprocess
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/altitude_dashboard.log'),
            logging.StreamHandler()
        ]
    )

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"📦 Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    return True

def check_core_systems():
    """Check availability of core Schwabot systems"""
    print("🔍 Checking core system availability...")
    
    try:
        from core.sfsss_strategy_bundler import create_sfsss_bundler
        print("✅ Strategy Bundler: Available")
        bundler_available = True
    except ImportError as e:
        print(f"⚠️ Strategy Bundler: Not available ({e})")
        bundler_available = False
    
    try:
        from core.constraints import get_system_bounds
        print("✅ Constraints System: Available")
        constraints_available = True
    except ImportError as e:
        print(f"⚠️ Constraints System: Not available ({e})")
        constraints_available = False
    
    try:
        from core.enhanced_btc_integration_bridge import create_enhanced_bridge
        print("✅ BTC Integration Bridge: Available")
        bridge_available = True
    except ImportError as e:
        print(f"⚠️ BTC Integration Bridge: Not available ({e})")
        bridge_available = False
    
    try:
        from core.integrated_pathway_test_suite import create_integrated_pathway_test_suite
        print("✅ Pathway Test Suite: Available")
        test_suite_available = True
    except ImportError as e:
        print(f"⚠️ Pathway Test Suite: Not available ({e})")
        test_suite_available = False
    
    total_available = sum([bundler_available, constraints_available, bridge_available, test_suite_available])
    print(f"📊 Core Systems Status: {total_available}/4 Available")
    
    if total_available == 0:
        print("⚠️ No core systems available - dashboard will run in simulation mode")
    elif total_available < 4:
        print("🔶 Partial core system availability - some features may be limited")
    else:
        print("🟢 All core systems available - full functionality enabled")
    
    return total_available > 0

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'config', 'data']
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Schwabot Altitude Adjustment Dashboard...")
    print("📍 Dashboard will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        # Launch Streamlit with the dashboard
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'schwabot_altitude_adjustment_dashboard.py',
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false',
            '--theme.base=dark'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print("=" * 80)
    print("🛠️  SCHWABOT ALTITUDE ADJUSTMENT DASHBOARD LAUNCHER")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages manually.")
        sys.exit(1)
    
    # Check core systems
    core_systems_available = check_core_systems()
    
    if not core_systems_available:
        print("\n⚠️ WARNING: Core systems not available")
        print("Dashboard will run in simulation mode with limited functionality")
        response = input("Continue anyway? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            print("❌ Startup cancelled by user")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("🎯 ALTITUDE DASHBOARD FEATURES:")
    print("   ✈️ Real-time altitude navigation visualization")
    print("   🌍 STAM zone classification with physics calculations") 
    print("   🕸️ Multivector stability regulation display")
    print("   🛣️ Pathway health monitoring (NCCO, SFS, ALIF, GAN, UFS)")
    print("   👻 Ghost phase integrator visualization")
    print("   🔗 Hash correlation analysis")
    print("   📊 Integration with strategy bundler and constraints")
    print("=" * 80)
    
    # Launch dashboard
    try:
        launch_dashboard()
    except Exception as e:
        logger.error(f"Failed to launch dashboard: {e}")
        print(f"❌ Launch failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 