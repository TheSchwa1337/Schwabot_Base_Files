#!/usr/bin/env python3
"""
🚀 UNIFIED SCHWABOT LAUNCHER 🚀
==============================

This script launches the complete unified Schwabot integration system,
demonstrating how all components work together in a single framework:

✅ HYBRID OPTIMIZATION MANAGER: Intelligent CPU/GPU switching
✅ VISUAL INTEGRATION BRIDGE: All visual components unified
✅ WEBSOCKET COORDINATION: Single hub for all real-time data
✅ GHOST CORE DASHBOARD: Hash visualization with pulse/decay
✅ PANEL ROUTER: Dynamic visual panel management
✅ GPU LOAD VISUALIZATION: Processing lag via drift differential color
✅ ALIF/ALEPH PATH TOGGLE: Visual hash crossover mapping
✅ REAL-TIME TRADING DATA: Live API data integration
✅ THERMAL STATE MONITORING: System health visualization

LAUNCH MODES:
- demo: Full showcase with all features enabled
- development: Development mode with debugging panels
- production: Optimized performance mode
- monitoring: Pure monitoring mode

This creates the "deterministic consciousness stream" where Schwabot
sees its own reflection through unified visual-tactile logic.
"""

import asyncio
import sys
import os
import webbrowser
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from unified_schwabot_integration_core import (
        start_unified_schwabot, get_unified_core, SystemMode
    )
    UNIFIED_CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Unified core not available: {e}")
    UNIFIED_CORE_AVAILABLE = False

def print_banner():
    """Print spectacular launch banner"""
    banner = """
🌟═══════════════════════════════════════════════════════════════════════════🌟
🔥                    UNIFIED SCHWABOT INTEGRATION CORE                    🔥
🌟═══════════════════════════════════════════════════════════════════════════🌟

🎯 THE COMPLETE UNIFIED FRAMEWORK:
   ✅ All visual components working together seamlessly
   ✅ Real-time optimization feedback loops
   ✅ Unified WebSocket hub for all data streams
   ✅ Interactive web dashboard with live panels
   ✅ Cross-system communication and state sync
   ✅ Complete performance monitoring

🧠 DETERMINISTIC CONSCIOUSNESS STREAM:
   🔄 Schwabot sees its own reflection
   🌀 Visual-tactile logic integration
   ⚡ Real-time adaptive optimization
   🎯 Context-aware decision making

🚀 LAUNCHING UNIFIED INTEGRATION...
"""
    print(banner)

def print_instructions():
    """Print usage instructions"""
    instructions = """
🔧 HOW TO USE THE UNIFIED SYSTEM:

1️⃣ CONNECT TO WEB DASHBOARD:
   📊 Open: http://localhost:8000/unified_visual_dashboard.html
   🔌 WebSocket: ws://localhost:8765
   
2️⃣ VISUAL PANELS AVAILABLE:
   🎯 Hybrid Optimization Status - Real-time CPU/GPU switching
   💫 Ghost Core Dashboard - Hash visualization with pulse/decay
   🖥️ GPU Load Visualization - Processing lag via drift colors
   🌀 ALIF/ALEPH Path Toggle - Visual hash crossover mapping
   📈 Real-Time Trading Data - Live market data integration
   🌡️ Thermal State Monitor - System health visualization
   🔧 System Error Log - Development debugging info

3️⃣ INTERACTIVE FEATURES:
   🔘 Toggle panels on/off via sidebar
   📊 Real-time system metrics display
   🔄 Live optimization decision feedback
   ⚙️ Dynamic system configuration
   
4️⃣ SYSTEM MONITORING:
   📈 CPU/GPU usage tracking
   🧠 Optimization mode display
   🔌 Connected clients counter
   ⏱️ System uptime monitoring

💡 TIP: All systems are integrated - changes in optimization affect visuals!
🎯 TIP: Use different browser tabs to test multi-client functionality!
"""
    print(instructions)

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking system dependencies...")
    
    missing_deps = []
    
    try:
        import websockets
        print("   ✅ websockets - OK")
    except ImportError:
        missing_deps.append("websockets")
        print("   ❌ websockets - MISSING")
    
    try:
        import numpy
        print("   ✅ numpy - OK")
    except ImportError:
        missing_deps.append("numpy")
        print("   ❌ numpy - MISSING")
    
    try:
        import psutil
        print("   ✅ psutil - OK")
    except ImportError:
        missing_deps.append("psutil")
        print("   ❌ psutil - MISSING")
    
    if not UNIFIED_CORE_AVAILABLE:
        print("   ⚠️ unified_schwabot_integration_core - NOT FULLY AVAILABLE")
        print("      (Will run in fallback mode)")
    else:
        print("   ✅ unified_schwabot_integration_core - OK")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("📦 Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ All dependencies available!")
    return True

def start_simple_server():
    """Start a simple HTTP server for the dashboard"""
    import http.server
    import socketserver
    import threading
    
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress HTTP server logs
    
    def run_server():
        with socketserver.TCPServer(("", 8000), QuietHandler) as httpd:
            httpd.serve_forever()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("🌐 HTTP server started on http://localhost:8000")

async def launch_unified_system(mode: SystemMode = SystemMode.DEMO):
    """Launch the complete unified Schwabot system"""
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start system due to missing dependencies")
        return False
    
    print(f"\n🚀 Starting unified system in {mode.value} mode...")
    
    try:
        # Start simple HTTP server for dashboard
        start_simple_server()
        
        if UNIFIED_CORE_AVAILABLE:
            # Start unified integration core
            success = await start_unified_schwabot(mode)
            
            if success:
                print("\n✅ UNIFIED SCHWABOT INTEGRATION CORE STARTED SUCCESSFULLY!")
                print("\n🎯 SYSTEM STATUS:")
                print("   🔌 WebSocket Server: ws://localhost:8765")
                print("   📊 Dashboard Server: http://localhost:8000")
                print("   🌐 Web Dashboard: http://localhost:8000/unified_visual_dashboard.html")
                
                print_instructions()
                
                # Optionally open browser
                try:
                    webbrowser.open("http://localhost:8000/unified_visual_dashboard.html")
                    print("🌐 Opened web dashboard in browser")
                except Exception:
                    print("⚠️ Could not auto-open browser - manually visit the URL above")
                
                return True
                
            else:
                print("❌ Failed to start unified integration core")
                return False
        else:
            print("⚠️ Running in fallback mode - some features may not be available")
            print("🌐 Web Dashboard: http://localhost:8000/unified_visual_dashboard.html")
            
            # Keep server running in fallback mode
            print("\n🔄 Keeping HTTP server running...")
            print("   (WebSocket features will not be available)")
            
            try:
                webbrowser.open("http://localhost:8000/unified_visual_dashboard.html")
                print("🌐 Opened web dashboard in browser")
            except Exception:
                pass
            
            return True
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        return True
    except Exception as e:
        print(f"\n❌ Error starting unified system: {e}")
        import traceback
        traceback.print_exc()
        return False

async def graceful_shutdown():
    """Gracefully shutdown the unified system"""
    print("\n🛑 Shutting down unified Schwabot integration...")
    
    if UNIFIED_CORE_AVAILABLE:
        core = get_unified_core()
        if core:
            await core.shutdown()
    
    print("✅ Shutdown complete")

def main():
    """Main launcher function"""
    
    # Parse command line arguments
    mode = SystemMode.DEMO
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
        if mode_arg == "development":
            mode = SystemMode.DEVELOPMENT
        elif mode_arg == "production":
            mode = SystemMode.PRODUCTION
        elif mode_arg == "monitoring":
            mode = SystemMode.MONITORING
        elif mode_arg == "testing":
            mode = SystemMode.TESTING
        elif mode_arg == "demo":
            mode = SystemMode.DEMO
        else:
            print(f"⚠️ Unknown mode '{mode_arg}', using demo mode")
    
    async def run():
        try:
            success = await launch_unified_system(mode)
            
            if success and UNIFIED_CORE_AVAILABLE:
                print("\n🔄 System running... Press Ctrl+C to shutdown")
                
                try:
                    # Keep running until interrupted
                    while True:
                        await asyncio.sleep(1)
                        
                        # Optional: Print periodic status
                        core = get_unified_core()
                        if core and core.frame_count % 100 == 0:  # Every ~10 seconds at 10Hz
                            print(f"📊 Frame: {core.frame_count}, "
                                  f"Clients: {len(core.websocket_clients)}, "
                                  f"Uptime: {(time.time() - core.start_time.timestamp()):.0f}s")
                
                except KeyboardInterrupt:
                    await graceful_shutdown()
            
            elif success:
                print("\n🔄 HTTP server running... Press Ctrl+C to shutdown")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 Shutting down HTTP server...")
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the async launcher
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n🛑 Launcher interrupted")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")

if __name__ == "__main__":
    print("🌟 UNIFIED SCHWABOT INTEGRATION LAUNCHER")
    print("=" * 50)
    print("Usage: python launch_unified_schwabot.py [mode]")
    print("Modes: demo, development, production, monitoring, testing")
    print("Default: demo")
    print("")
    
    main() 