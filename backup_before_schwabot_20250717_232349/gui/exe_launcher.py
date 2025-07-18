#!/usr/bin/env python3
"""
Schwabot EXE Launcher - Standalone Trading Application
Bundles Flask GUI with all core components for executable distribution.
"""

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.clean_unified_math import clean_unified_math
    from core.unified_trade_router import UnifiedTradeRouter
    from core.visual_execution_node import VisualExecutionNode
    from flask import Flask
    from gui.flask_app import app as flask_app

    print("✅ All core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install flask pyinstaller")
    sys.exit(1)


class SchwabotEXELauncher:
    """Main launcher for Schwabot executable."""

    def __init__(self):
        self.app = flask_app
        self.port = 5000
        self.host = '127.0.0.1'
        self.server_thread = None
        self.is_running = False

    def start_server(self):
        """Start Flask server in background thread."""
        try:
            print(f"🚀 Starting Schwabot server on http://{self.host}:{self.port}")
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,  # Disable debug for production
                use_reloader=False,
                threaded=True,
            )
        except Exception as e:
            print(f"❌ Server error: {e}")

    def open_browser(self):
        """Open browser to Schwabot dashboard."""
        time.sleep(2)  # Wait for server to start
        try:
            url = f"http://{self.host}:{self.port}"
            webbrowser.open(url)
            print(f"🌐 Opened browser to: {url}")
        except Exception as e:
            print(f"❌ Browser error: {e}")

    def run(self):
        """Main launcher execution."""
        print("=" * 60)
        print("🚀 SCHWABOT TRADING SYSTEM v1.0.0")
        print("=" * 60)
        print("🎯 Advanced Algorithmic Trading Intelligence")
        print("⚡ Quantum Mathematical Trading Engine")
        print("🧠 AI-Powered Strategic Decision Network")
        print("💰 Multi-Dimensional Profit Optimization")
        print("🛡️ Military-Grade Security Architecture")
        print("=" * 60)

        # Test core components
        print("\n🔧 Testing core components...")
        try:
            # Test VisualExecutionNode
            node = VisualExecutionNode("BTC/USDC", 60000.0)
            result = node.execute()
            print(f"✅ VisualExecutionNode: {result['hash'][:16]}...")

            # Test UnifiedTradeRouter
            router = UnifiedTradeRouter()
            print("✅ UnifiedTradeRouter initialized")

            # Test clean_unified_math
            math_result = clean_unified_math.integrate_all_systems(
                {'tensor': [[60000, 1000]], 'metadata': {'confidence': 0.5}}
            )
            print(f"✅ CleanUnifiedMath: score {math_result.get('combined_score', 0):.4f}")

        except Exception as e:
            print(f"❌ Component test failed: {e}")
            return False

        print("\n🎬 Starting Schwabot GUI...")

        # Start server in background thread
        self.server_thread = threading.Thread(target=self.start_server, daemon=True)
        self.server_thread.start()
        self.is_running = True

        # Open browser
        browser_thread = threading.Thread(target=self.open_browser, daemon=True)
        browser_thread.start()

        print("\n🎮 Schwabot is now running!")
        print("📊 Dashboard: http://127.0.0.1:5000")
        print("🔧 API Endpoints: http://127.0.0.1:5000/api/")
        print("💻 CLI Commands: Available in dashboard")
        print("\nPress Ctrl+C to stop Schwabot")

        try:
            # Keep main thread alive
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down Schwabot...")
            self.is_running = False
            print("✅ Schwabot stopped successfully")

        return True


def create_pyinstaller_spec():
    """Create PyInstaller spec file for bundling."""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['gui/exe_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('gui/templates', 'gui/templates'),
        ('core', 'core'),
        ('utils', 'utils'),
        ('core/api', 'core/api'),
        ('core/api/handlers', 'core/api/handlers'),
        ('core/strategy', 'core/strategy'),
        ('core/profit', 'core/profit'),
        ('core/immune', 'core/immune'),
        ('core/entropy', 'core/entropy'),
        ('core/data', 'core/data'),
        ('core/swarm', 'core/swarm'),
        ('core/math', 'core/math'),
    ],
    hiddenimports=[
        'flask',
        'core.visual_execution_node',
        'core.unified_trade_router',
        'core.clean_unified_math',
        'core.ccxt_integration',
        'core.unified_math_system',
        'utils.safe_print',
        'utils.math_utils',
        'utils.market_data_utils',
        'core.api.integration_manager',
        'core.api.exchange_connection',
        'core.api.handlers.coingecko',
        'core.api.handlers.glassnode',
        'core.api.handlers.whale_alert',
        'core.api.handlers.alt_fear_greed',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Schwabot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/schwabot_icon.ico' if os.path.exists('assets/schwabot_icon.ico') else None,
)
'''

    with open('schwabot.spec', 'w') as f:
        f.write(spec_content)

    print("✅ Created PyInstaller spec file: schwabot.spec")


def build_exe():
    """Build executable using PyInstaller."""
    print("🔨 Building Schwabot executable...")

    # Create spec file
    create_pyinstaller_spec()

    # Run PyInstaller
    import subprocess

    try:
        result = subprocess.run(
            ['pyinstaller', '--clean', '--onefile', '--name=Schwabot', 'gui/exe_launcher.py'],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Executable built successfully!")
            print("📁 Location: dist/Schwabot.exe")
        else:
            print(f"❌ Build failed: {result.stderr}")

    except FileNotFoundError:
        print("❌ PyInstaller not found. Install with: pip install pyinstaller")
    except Exception as e:
        print(f"❌ Build error: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == '--build':
        build_exe()
    else:
        launcher = SchwabotEXELauncher()
        launcher.run()


if __name__ == '__main__':
    main()
