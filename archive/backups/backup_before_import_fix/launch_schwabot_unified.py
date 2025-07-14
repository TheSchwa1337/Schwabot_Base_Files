import argparse
import logging
import sys
from pathlib import Path

from schwabot_unified_launcher import SchwabotUnifiedLauncher

from core.unified_component_bridge import get_component_bridge

#!/usr/bin/env python3
"""Schwabot Unified Launcher - Quick Start Script."

This script launches the complete unified Schwabot platform with:
- Visual command center with Ferris wheel interface
- Component bridge for all Schwabot systems
- Historical data integration
- Resource monitoring
- Plugin/benchmark/device/processor/manager tabs
"""


# Ensure we can import all modules
sys.path.append(str(Path(__file__).parent))


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig()
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("schwabot_unified.log")],
    )


def check_requirements():
    """Check if all required packages are installed."""
    required_packages = []
        "tkinter",
        "pandas",
        "numpy",
        "yaml",
        "flask",
        "psutil",
        "matplotlib",
        "requests",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements_unified.txt")
        return False

    return True


def initialize_data_directories():
    """Create necessary data directories if they don't exist."""'
    directories = []
        "data/historical/btc_usdc",
        "data/historical/eth_usdc",
        "data/historical/xrp_usdc",
        "data/preprocessed",
        "data/live",
        "data/cache",
        "logs",
        "config",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("📁 Data directories initialized")


def check_schwabot_components():
    """Check if core Schwabot components are available."""
    core_files = []
        "core/live_execution_mapper.py",
        "core/risk_manager.py",
        "core/trade_executor.py",
        "core/speed_lattice_trading_integration.py",
        "hash_recollection/",
        "schwabot/",
    ]

    missing_files = []
    for file_path in core_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"⚠️ Missing core components: {', '.join(missing_files)}")
        print("Some features may not be available")
        return False

    return True


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Schwabot Unified Launcher")
    parser.add_argument()
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    parser.add_argument()
        "--simulation-mode",
        action="store_true",
        default=True,
        help="Run in simulation mode (default)",
    )
    parser.add_argument()
        "--live-mode",
        action="store_true",
        help="Run in live trading mode (requires API, keys)",
    )
    parser.add_argument()
        "--skip-checks",
        action="store_true",
        help="Skip requirement and component checks",
    )
    parser.add_argument()
        "--debug-ui", action="store_true", help="Enable UI debugging features"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    print("🚀 Schwabot Unified Launcher v0.5")
    print("=" * 50)

    # Pre-flight checks
    if not args.skip_checks:
        print("🔍 Running pre-flight checks...")

        if not check_requirements():
            print("❌ Requirements check failed")
            return 1

        if not check_schwabot_components():
            print("⚠️ Some components missing, but continuing...")

        print("✅ Pre-flight checks completed")

    # Initialize directories
    initialize_data_directories()

    try:
        # Import and initialize the unified launcher
        print("🎯 Initializing Schwabot Unified Launcher...")

        # Initialize component bridge
        bridge = get_component_bridge()
        print()
            f"🔗 Component bridge initialized with {len(bridge.components)} components"
        )

        # Create and configure launcher
        launcher = SchwabotUnifiedLauncher()

        # Set simulation/live mode
        if args.live_mode:
            print("📈 Live trading mode enabled")
            launcher.simulation_mode = False
        else:
            print("🎮 Simulation mode enabled")
            launcher.simulation_mode = True

        # Enable debug features if requested
        if args.debug_ui:
            launcher.debug_mode = True
            print("🐛 UI debugging enabled")

        print("✅ Launcher initialized successfully")
        print("\n" + "=" * 50)
        print("🎨 Starting Visual Command Center...")
        print("   • Use tabs to navigate different components")
        print("   • Click Ferris wheel quadrants for quick access")
        print("   • Monitor system resources in real-time")
        print("   • Control all Schwabot components from one interface")
        print("=" * 50)

        # Start the launcher
        launcher.run()

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        print(f"❌ Import error: {e}")
        print("Make sure all requirements are installed and paths are correct")
        return 1

    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        logger.info("Launcher shutdown requested")
        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        return 1

    finally:
        print("👋 Schwabot Unified Launcher shutdown complete")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
