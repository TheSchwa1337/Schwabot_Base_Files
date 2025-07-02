#!/usr/bin/env python3
"""
Schwabot Production Launcher
===========================

Cross-platform production launcher for the Schwabot trading system.
Supports Windows, macOS, and Linux with comprehensive error handling
and automated deployment capabilities.

Usage:
    python schwabot_production_launcher.py --mode [demo|simulation|live]
    python schwabot_production_launcher.py --install-deps
    python schwabot_production_launcher.py --validate-system
    python schwabot_production_launcher.py --dashboard
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('schwabot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class CrossPlatformLauncher:
    """Cross-platform launcher for Schwabot trading system."""

    def __init__(self):
        """Initialize the launcher."""
        self.platform = platform.system().lower()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent.absolute()
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"

        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # System status
        self.system_status = {
            "platform": self.platform,
            "python_version": str(self.python_version),
            "project_root": str(self.project_root),
            "dependencies_installed": False,
            "system_validated": False,
            "last_validation": None
        }

        logger.info(f"🚀 Schwabot Launcher initialized on {self.platform}")
        logger.info(f"📍 Project root: {self.project_root}")

    def print_banner(self):
        """Print the Schwabot banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                      SCHWABOT TRADING SYSTEM                 ║
║                   Production Launcher v2.0                   ║
║                                                              ║
║  🤖 Advanced AI Trading Bot with Mathematical Framework      ║
║  📈 Real-time Market Analysis & Portfolio Management        ║
║  🧮 Tensor Core Calculations & Drift Detection              ║
║  🔗 Cross-platform CLI & API Integration                    ║
║                                                              ║
║  Platform: {:<48} ║
║  Python:   {:<48} ║
╚══════════════════════════════════════════════════════════════╝
        """.format(
            f"{self.platform.title()} {platform.release()}",
            f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}"
        )
        print(banner)

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        required_packages = [
            "numpy",
            "pandas",
            "asyncio",
            "websockets",
            "requests",
            "pyyaml",
            "psutil"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.warning(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False

        logger.info("✅ All dependencies are installed")
        self.system_status["dependencies_installed"] = True
        return True

    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        logger.info("📦 Installing dependencies...")

        requirements = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "websockets>=10.0",
            "requests>=2.25.0",
            "pyyaml>=5.4.0",
            "psutil>=5.8.0",
            "aiofiles>=0.7.0"
        ]

        try:
            for requirement in requirements:
                logger.info(f"Installing {requirement}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", requirement],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"✅ {requirement} installed successfully")

            logger.info("🎉 All dependencies installed successfully!")
            self.system_status["dependencies_installed"] = True
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during installation: {e}")
            return False

    def validate_system(self) -> bool:
        """Validate the entire Schwabot system."""
        logger.info("🔍 Validating Schwabot system...")

        try:
            # Run the comprehensive validation
            result = subprocess.run(
                [sys.executable, "system_comprehensive_validation.py"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                logger.info("✅ System validation passed - FULLY OPERATIONAL")
                self.system_status["system_validated"] = True
                self.system_status["last_validation"] = time.time()
                return True
            elif result.returncode == 1:
                logger.warning("⚠️  System validation passed with warnings - MOSTLY OPERATIONAL")
                self.system_status["system_validated"] = True
                self.system_status["last_validation"] = time.time()
                return True
            else:
                logger.error(f"❌ System validation failed with exit code {result.returncode}")
                logger.error(f"Validation output: {result.stdout}")
                logger.error(f"Validation errors: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ System validation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error during system validation: {e}")
            return False

    def create_desktop_shortcut(self):
        """Create desktop shortcut for easy access."""
        try:
            if self.platform == "windows":
                self._create_windows_shortcut()
            elif self.platform == "darwin":  # macOS
                self._create_macos_app()
            elif self.platform == "linux":
                self._create_linux_desktop_entry()

            logger.info("✅ Desktop shortcut created successfully")

        except Exception as e:
            logger.error(f"❌ Failed to create desktop shortcut: {e}")

    def _create_windows_shortcut(self):
        """Create Windows shortcut."""
        import winshell
        from win32com.client import Dispatch

        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, "Schwabot Trading System.lnk")

        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{self.project_root / "schwabot_production_launcher.py"}" --mode demo'
        shortcut.WorkingDirectory = str(self.project_root)
        shortcut.IconLocation = sys.executable
        shortcut.save()

    def _create_macos_app(self):
        """Create macOS application bundle."""
        app_dir = Path.home() / "Applications" / "Schwabot.app"
        contents_dir = app_dir / "Contents"
        macos_dir = contents_dir / "MacOS"

        # Create directories
        macos_dir.mkdir(parents=True, exist_ok=True)

        # Create Info.plist
        info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>schwabot</string>
    <key>CFBundleIdentifier</key>
    <string>com.schwabot.trading</string>
    <key>CFBundleName</key>
    <string>Schwabot Trading System</string>
    <key>CFBundleVersion</key>
    <string>2.0</string>
</dict>
</plist>"""

        with open(contents_dir / "Info.plist", "w") as f:
            f.write(info_plist)

        # Create executable script
        executable_script = f"""#!/bin/bash
cd "{self.project_root}"
"{sys.executable}" schwabot_production_launcher.py --mode demo
"""

        executable_path = macos_dir / "schwabot"
        with open(executable_path, "w") as f:
            f.write(executable_script)

        # Make executable
        os.chmod(executable_path, 0o755)

    def _create_linux_desktop_entry(self):
        """Create Linux desktop entry."""
        desktop_dir = Path.home() / ".local" / "share" / "applications"
        desktop_dir.mkdir(parents=True, exist_ok=True)

        desktop_entry = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Schwabot Trading System
Comment=Advanced AI Trading Bot
Exec="{sys.executable}" "{self.project_root / 'schwabot_production_launcher.py'}" --mode demo
Path={self.project_root}
Icon=utilities-terminal
Terminal=true
Categories=Office;Finance;
"""

        with open(desktop_dir / "schwabot.desktop", "w") as f:
            f.write(desktop_entry)

        # Make executable
        os.chmod(desktop_dir / "schwabot.desktop", 0o755)

    async def start_demo_mode(self):
        """Start the system in demo mode."""
        logger.info("🎯 Starting Schwabot in DEMO mode...")

        try:
            # Import and start the launcher
            from launcher import main as launcher_main

            # Run the launcher in demo mode
            await launcher_main(mode="demo")

        except ImportError:
            logger.error("❌ launcher.py not found. Running basic demo...")
            await self._run_basic_demo()
        except Exception as e:
            logger.error(f"❌ Error starting demo mode: {e}")
            await self._run_basic_demo()

    async def start_simulation_mode(self):
        """Start the system in simulation mode."""
        logger.info("🧪 Starting Schwabot in SIMULATION mode...")

        try:
            from launcher import main as launcher_main
            await launcher_main(mode="simulation")
        except Exception as e:
            logger.error(f"❌ Error starting simulation mode: {e}")

    async def start_live_mode(self):
        """Start the system in live trading mode."""
        logger.warning("⚡ Starting Schwabot in LIVE TRADING mode...")
        logger.warning("🚨 This will use real money! Ensure you understand the risks!")

        # Additional confirmation for live mode
        if not self._confirm_live_trading():
            logger.info("❌ Live trading cancelled by user")
            return

        try:
            from launcher import main as launcher_main
            await launcher_main(mode="live")
        except Exception as e:
            logger.error(f"❌ Error starting live mode: {e}")

    def _confirm_live_trading(self) -> bool:
        """Confirm live trading with user."""
        try:
            confirmation = input("⚠️  Are you sure you want to start LIVE TRADING? Type 'YES' to confirm: ")
            return confirmation.strip().upper() == "YES"
        except KeyboardInterrupt:
            return False

    async def _run_basic_demo(self):
        """Run a basic demo when full system is not available."""
        logger.info("🎮 Running basic Schwabot demo...")

        # Simulate basic trading operations
        demo_data = {
            "portfolio_value": 100000.0,
            "btc_price": 50000.0,
            "current_position": 0.1
        }

        for i in range(10):
            # Simulate price changes
            price_change = (hash(str(time.time())) % 200) - 100
            demo_data["btc_price"] += price_change

            # Calculate portfolio value
            position_value = demo_data["current_position"] * demo_data["btc_price"]
            cash_value = demo_data["portfolio_value"] - (demo_data["current_position"] * 50000)
            total_value = position_value + cash_value

            logger.info(f"📊 Tick {i+1}: BTC ${demo_data['btc_price']:.2f} | Portfolio: ${total_value:.2f}")

            await asyncio.sleep(2)

        logger.info("🎉 Basic demo completed!")

    async def start_dashboard(self):
        """Start the web dashboard."""
        logger.info("🌐 Starting Schwabot web dashboard...")

        try:
            # Try to import and start the dashboard
            from dashboard_backend import run_dashboard
            await run_dashboard()
        except ImportError:
            logger.info("📡 Starting basic web server on port 8080...")
            await self._start_basic_server()
        except Exception as e:
            logger.error(f"❌ Error starting dashboard: {e}")

    async def _start_basic_server(self):
        """Start a basic web server."""
        import http.server
        import socketserver
        import threading

        def serve():
            with socketserver.TCPServer(("", 8080), http.server.SimpleHTTPRequestHandler) as httpd:
                logger.info("🌐 Server running at http://localhost:8080")
                httpd.serve_forever()

        server_thread = threading.Thread(target=serve, daemon=True)
        server_thread.start()

        logger.info("✅ Basic web server started. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Server stopped")

    def save_status(self):
        """Save system status to file."""
        status_file = self.project_root / "system_status.json"

        try:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(self.system_status, f, indent=2, default=str)
            logger.info(f"💾 System status saved to {status_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save system status: {e}")

    def load_status(self):
        """Load system status from file."""
        status_file = self.project_root / "system_status.json"

        try:
            if status_file.exists():
                with open(status_file, "r", encoding="utf-8") as f:
                    loaded_status = json.load(f)
                    self.system_status.update(loaded_status)
                logger.info(f"📂 System status loaded from {status_file}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load system status: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Schwabot Trading System - Production Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python schwabot_production_launcher.py --mode demo
  python schwabot_production_launcher.py --install-deps
  python schwabot_production_launcher.py --validate-system
  python schwabot_production_launcher.py --dashboard
        """
    )

    parser.add_argument(
        "--mode",
        choices=["demo", "simulation", "live"],
        default="demo",
        help="Operating mode (default: demo)"
    )

    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required dependencies"
    )

    parser.add_argument(
        "--validate-system",
        action="store_true",
        help="Validate system functionality"
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start web dashboard"
    )

    parser.add_argument(
        "--create-shortcut",
        action="store_true",
        help="Create desktop shortcut"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize launcher
    launcher = CrossPlatformLauncher()
    launcher.load_status()
    launcher.print_banner()

    try:
        # Handle command line options
        if args.install_deps:
            success = launcher.install_dependencies()
            if not success:
                return 1

        if args.validate_system:
            success = launcher.validate_system()
            if not success:
                return 1

        if args.create_shortcut:
            launcher.create_desktop_shortcut()

        if args.dashboard:
            await launcher.start_dashboard()
            return 0

        # Check dependencies if not installing them
        if not args.install_deps and not launcher.check_dependencies():
            logger.error("❌ Dependencies not installed. Run with --install-deps first.")
            return 1

        # Start the requested mode
        if args.mode == "demo":
            await launcher.start_demo_mode()
        elif args.mode == "simulation":
            await launcher.start_simulation_mode()
        elif args.mode == "live":
            await launcher.start_live_mode()

        return 0

    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
        return 0
    except Exception as e:
        logger.error(f"🚨 Critical error: {e}")
        return 1
    finally:
        launcher.save_status()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"🚨 Fatal error: {e}")
        sys.exit(1)
