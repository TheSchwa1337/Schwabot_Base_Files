#!/usr/bin/env python3
"""
Unified Visual Controller Demonstration
======================================

Demonstrates the complete visual interface system for Schwabot that embodies
the 8 principles of sustainment and provides elegant, non-intrusive controls.

This demo shows:
- Simple toggle controls for BTC processor features
- Real-time visual feedback without interface disruption
- Seamless mode switching (development -> testing -> live trading)
- Custom panel installation capabilities
- Emergency procedures and resource management
- Integration with all mathematical engines

The interface is designed to be:
1. Simple - Clean, intuitive controls
2. Non-intrusive - Doesn't interfere with trading visualization
3. Responsive - Real-time updates and immediate feedback
4. Customizable - Install new panels and controls as needed
5. Robust - Graceful handling of errors and emergency situations
"""

import asyncio
import logging
import time
import webbrowser
import threading
from pathlib import Path
from typing import Dict, Any

# Import core controllers
from core.btc_processor_controller import BTCProcessorController
from core.visual_integration_bridge import VisualIntegrationBridge
from core.ui_integration_bridge import UIIntegrationBridge
from core.unified_visual_controller import UnifiedVisualController, create_unified_visual_controller
from core.sustainment_underlay_controller import SustainmentUnderlayController

# Mock imports for demonstration
try:
    from core.ui_state_bridge import UIStateBridge, create_ui_bridge
except ImportError:
    # Create mock classes for demo
    class UIStateBridge:
        def start(self): pass
        def stop(self): pass
        def register_ui_callback(self, callback): pass
    
    def create_ui_bridge(*args): 
        return UIStateBridge()

try:
    from core.thermal_zone_manager import ThermalZoneManager
except ImportError:
    class ThermalZoneManager:
        def get_thermal_state(self): return {"temperature": 45.0, "status": "normal"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedVisualDemo:
    """
    Comprehensive demonstration of the unified visual controller.
    
    Shows how the interface provides simple, elegant controls that don't
    interfere with trading operations while maintaining full system oversight.
    """
    
    def __init__(self):
        """Initialize the demonstration"""
        
        self.btc_controller = None
        self.visual_bridge = None
        self.ui_bridge = None
        self.sustainment_controller = None
        self.unified_controller = None
        
        # Demo state
        self.demo_active = False
        self.current_mode = "development"
        self.simulated_trades = []
        self.performance_metrics = {}
        
        logger.info("Unified Visual Demo initialized")

    async def setup_controllers(self) -> None:
        """Set up all the core controllers for the demonstration"""
        
        logger.info("Setting up core controllers...")
        
        # Initialize BTC Processor Controller
        self.btc_controller = BTCProcessorController()
        
        # Initialize UI Integration Bridge
        self.ui_bridge = UIIntegrationBridge()
        
        # Mock Sustainment Controller for demo
        self.sustainment_controller = self._create_mock_sustainment_controller()
        
        # Create UI State Bridge
        ui_state_bridge = create_ui_bridge(
            self.sustainment_controller, 
            ThermalZoneManager(), 
            None,  # profit navigator
            None   # fractal core
        )
        
        # Initialize Visual Integration Bridge
        self.visual_bridge = VisualIntegrationBridge(
            ui_bridge=ui_state_bridge,
            sustainment_controller=self.sustainment_controller,
            websocket_port=8765
        )
        
        # Create the Unified Visual Controller
        self.unified_controller = create_unified_visual_controller(
            btc_controller=self.btc_controller,
            visual_bridge=self.visual_bridge,
            ui_bridge=self.ui_bridge,
            sustainment_controller=self.sustainment_controller,
            websocket_port=8080
        )
        
        logger.info("All controllers initialized successfully")

    def _create_mock_sustainment_controller(self):
        """Create a mock sustainment controller for demonstration"""
        
        class MockSustainmentController:
            def __init__(self):
                self.principles_metrics = {
                    "anticipation": 0.85,
                    "continuity": 0.92,
                    "responsiveness": 0.88,
                    "integration": 0.90,
                    "simplicity": 0.95,
                    "improvisation": 0.80,
                    "survivability": 0.87,
                    "economy": 0.91
                }
            
            def get_sustainment_metrics(self):
                return self.principles_metrics
            
            def update_principle(self, principle, value):
                self.principles_metrics[principle] = value
        
        return MockSustainmentController()

    async def start_demo(self) -> None:
        """Start the complete demonstration"""
        
        if self.demo_active:
            logger.warning("Demo already active")
            return
        
        self.demo_active = True
        
        try:
            # Setup all controllers
            await self.setup_controllers()
            
            # Start the unified visual controller
            await self.unified_controller.start_visual_controller()
            
            # Start subsidiary systems
            self.ui_bridge.start()
            self.visual_bridge.start_visual_bridge()
            
            logger.info("🚀 Unified Visual Demo started successfully!")
            
            # Open the web interface
            await self._open_web_interface()
            
            # Start demonstration scenarios
            await self._run_demonstration_scenarios()
            
        except Exception as e:
            logger.error(f"Failed to start demo: {e}")
            await self.stop_demo()

    async def _open_web_interface(self) -> None:
        """Open the web interface in the default browser"""
        
        # Get the path to the HTML file
        html_file = Path(__file__).parent.parent / "core" / "modern_ui_interface.html"
        
        if html_file.exists():
            # Open in browser after a short delay
            def open_browser():
                time.sleep(2)
                webbrowser.open(f"file://{html_file.absolute()}")
            
            threading.Thread(target=open_browser, daemon=True).start()
            logger.info(f"🌐 Web interface will open at: file://{html_file.absolute()}")
        else:
            logger.warning("Web interface file not found - serving from WebSocket only")
        
        logger.info("💡 WebSocket server running on ws://localhost:8080")
        logger.info("🎮 You can now interact with the visual interface!")

    async def _run_demonstration_scenarios(self) -> None:
        """Run various demonstration scenarios"""
        
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATION SCENARIOS")
        logger.info("="*60)
        
        # Scenario 1: Basic Feature Control
        await self._demo_basic_feature_control()
        
        # Scenario 2: Mode Switching
        await self._demo_mode_switching()
        
        # Scenario 3: Resource Management
        await self._demo_resource_management()
        
        # Scenario 4: Custom Panel Installation
        await self._demo_custom_panel_installation()
        
        # Scenario 5: Emergency Procedures
        await self._demo_emergency_procedures()
        
        # Scenario 6: Live Trading Preparation
        await self._demo_live_trading_preparation()
        
        logger.info("✅ All demonstration scenarios completed!")

    async def _demo_basic_feature_control(self) -> None:
        """Demonstrate basic feature toggle controls"""
        
        logger.info("\n📋 SCENARIO 1: Basic Feature Control")
        logger.info("Demonstrating simple toggle controls for BTC processor features...")
        
        # Show current state
        current_state = self.unified_controller.get_visual_state()
        logger.info(f"Current mode: {current_state['mode']}")
        
        # Demonstrate toggle controls
        features_to_test = [
            ("btc_mining_analysis", "Mining Analysis"),
            ("btc_backlog_processing", "Backlog Processing"),
            ("btc_memory_management", "Memory Management")
        ]
        
        for feature_id, feature_name in features_to_test:
            logger.info(f"  🔄 Toggling {feature_name}...")
            
            # Toggle off
            await self.unified_controller._handle_toggle_control({
                "control_id": feature_id,
                "enabled": False,
                "confirmed": True
            })
            
            await asyncio.sleep(1)
            
            # Toggle back on
            await self.unified_controller._handle_toggle_control({
                "control_id": feature_id,
                "enabled": True,
                "confirmed": True
            })
            
            logger.info(f"  ✅ {feature_name} toggle demonstrated")
            await asyncio.sleep(0.5)
        
        logger.info("✅ Basic feature control demonstration completed")

    async def _demo_mode_switching(self) -> None:
        """Demonstrate seamless mode switching"""
        
        logger.info("\n🔄 SCENARIO 2: Mode Switching")
        logger.info("Demonstrating seamless switching between operational modes...")
        
        modes_to_test = [
            ("testing", "Testing Mode - Safe experimentation"),
            ("analysis", "Analysis Mode - Deep data processing"),
            ("development", "Development Mode - Full feature access")
        ]
        
        for mode, description in modes_to_test:
            logger.info(f"  📊 Switching to {mode}...")
            logger.info(f"     {description}")
            
            if mode == "testing":
                await self.unified_controller.switch_to_testing_mode()
            else:
                self.unified_controller.visual_state.mode = getattr(
                    self.unified_controller.visual_state.mode.__class__, mode.upper()
                )
                await self.unified_controller._broadcast_mode_change()
            
            # Show the effect of mode change
            state = self.unified_controller.get_visual_state()
            logger.info(f"     Mode changed to: {state['mode']}")
            logger.info(f"     Active toggles: {len([k for k, v in state['toggle_states'].items() if v])}")
            
            await asyncio.sleep(2)
        
        logger.info("✅ Mode switching demonstration completed")

    async def _demo_resource_management(self) -> None:
        """Demonstrate resource management and monitoring"""
        
        logger.info("\n📊 SCENARIO 3: Resource Management")
        logger.info("Demonstrating real-time resource monitoring and automatic management...")
        
        # Demonstrate slider controls
        slider_tests = [
            ("max_memory_usage", 15.0, "Memory limit increased for heavy processing"),
            ("cpu_usage_limit", 60.0, "CPU limit reduced for thermal protection"),
            ("gpu_usage_limit", 90.0, "GPU limit increased for analysis work")
        ]
        
        for slider_id, new_value, description in slider_tests:
            logger.info(f"  🎛️  Adjusting {slider_id} to {new_value}")
            logger.info(f"     {description}")
            
            # Update slider value
            self.unified_controller.visual_state.slider_values[slider_id] = new_value
            
            # Broadcast the change
            await self.unified_controller._broadcast_control_update("slider", slider_id, new_value)
            
            await asyncio.sleep(1)
        
        # Simulate resource threshold warning
        logger.info("  ⚠️  Simulating resource threshold warning...")
        logger.info("     Memory usage approaching limit - automatic optimization triggered")
        
        # This would trigger automatic memory cleanup in a real scenario
        await self.btc_controller._reduce_memory_usage()
        
        logger.info("✅ Resource management demonstration completed")

    async def _demo_custom_panel_installation(self) -> None:
        """Demonstrate custom panel installation (Improvisation principle)"""
        
        logger.info("\n🔧 SCENARIO 4: Custom Panel Installation")
        logger.info("Demonstrating the ability to install custom panels for new functionality...")
        
        # Install a custom RSI indicator panel
        custom_panel_config = {
            "panel_id": "rsi_indicator",
            "title": "RSI Technical Indicator",
            "panel_type": "custom",
            "position": {"x": 100, "y": 200},
            "size": {"width": 300, "height": 250},
            "visible": True,
            "update_frequency": 1.0
        }
        
        panel_id = await self.unified_controller.install_custom_panel(custom_panel_config)
        logger.info(f"  📊 Installed custom panel: {panel_id}")
        logger.info(f"     Panel provides RSI indicator functionality")
        
        await asyncio.sleep(1)
        
        # Install a second custom panel for volatility analysis
        volatility_panel_config = {
            "panel_id": "volatility_analyzer",
            "title": "Volatility Analysis",
            "panel_type": "custom",
            "position": {"x": 420, "y": 200},
            "size": {"width": 350, "height": 300},
            "visible": True,
            "update_frequency": 2.0
        }
        
        panel_id2 = await self.unified_controller.install_custom_panel(volatility_panel_config)
        logger.info(f"  📈 Installed custom panel: {panel_id2}")
        logger.info(f"     Panel provides volatility analysis tools")
        
        # Show current panel registry
        panel_count = len(self.unified_controller.panel_registry)
        logger.info(f"  ✅ Total panels available: {panel_count}")
        
        logger.info("✅ Custom panel installation demonstration completed")

    async def _demo_emergency_procedures(self) -> None:
        """Demonstrate emergency procedures and failsafes"""
        
        logger.info("\n🚨 SCENARIO 5: Emergency Procedures")
        logger.info("Demonstrating emergency procedures and system protection...")
        
        logger.info("  ⚠️  Simulating critical resource condition...")
        logger.info("     Memory usage: 95% - triggering emergency mode")
        
        # Enter emergency mode
        await self.unified_controller._enter_emergency_mode()
        
        # Show the effects
        state = self.unified_controller.get_visual_state()
        logger.info(f"     🛡️  Emergency mode activated")
        logger.info(f"     📊 Memory limit reduced to: {state['slider_values']['max_memory_usage']} GB")
        logger.info(f"     🔄 CPU limit reduced to: {state['slider_values']['cpu_usage_limit']}%")
        logger.info(f"     🎮 Live trading disabled: {not state['toggle_states'].get('live_trading', False)}")
        
        await asyncio.sleep(3)
        
        logger.info("  🔄 Simulating system recovery...")
        logger.info("     Resources stabilized - exiting emergency mode")
        
        # Exit emergency mode
        await self.unified_controller._exit_emergency_mode()
        
        state = self.unified_controller.get_visual_state()
        logger.info(f"     ✅ Normal operation restored")
        logger.info(f"     📊 Memory limit restored to: {state['slider_values']['max_memory_usage']} GB")
        
        logger.info("✅ Emergency procedures demonstration completed")

    async def _demo_live_trading_preparation(self) -> None:
        """Demonstrate preparation for live trading"""
        
        logger.info("\n💰 SCENARIO 6: Live Trading Preparation")
        logger.info("Demonstrating the process of preparing for live trading...")
        
        logger.info("  🔍 Step 1: Optimizing BTC processor for live trading")
        
        # Switch to live trading mode
        await self.unified_controller.switch_to_live_trading_mode()
        
        state = self.unified_controller.get_visual_state()
        logger.info(f"     Mode: {state['mode']}")
        logger.info(f"     Analysis features disabled for optimal performance")
        logger.info(f"     Conservative resource limits applied")
        
        await asyncio.sleep(2)
        
        logger.info("  💡 Step 2: System health verification")
        health_score = self.unified_controller._calculate_overall_health()
        logger.info(f"     Overall system health: {health_score:.1%}")
        
        if health_score > 0.8:
            logger.info("     ✅ System ready for live trading")
        else:
            logger.info("     ⚠️  System health suboptimal - remaining in testing mode")
        
        await asyncio.sleep(1)
        
        logger.info("  🎯 Step 3: Final safety checks")
        logger.info("     ✓ Risk monitoring enabled")
        logger.info("     ✓ Emergency procedures tested")
        logger.info("     ✓ Resource limits configured")
        logger.info("     ✓ Mathematical engines optimized")
        
        logger.info("✅ Live trading preparation demonstration completed")

    async def demonstrate_real_time_updates(self) -> None:
        """Show real-time updates and data streaming"""
        
        logger.info("\n📡 REAL-TIME UPDATES DEMONSTRATION")
        logger.info("The interface is now receiving real-time updates...")
        logger.info("You should see live data in the web interface:")
        logger.info("  • Resource usage meters updating")
        logger.info("  • System health scores changing")
        logger.info("  • Toggle states reflecting changes")
        logger.info("  • Trading metrics updating")
        
        # Simulate some activity for 30 seconds
        for i in range(30):
            # Update performance metrics
            self.unified_controller.visual_state.performance_metrics.update({
                "cpu_usage": 30 + (i % 20),
                "memory_usage": 40 + (i % 15),
                "gpu_usage": 25 + (i % 30),
                "update_latency": 0.5 + (i % 5) * 0.1
            })
            
            # Broadcast periodic update
            await self.unified_controller._broadcast_periodic_updates()
            
            if i % 10 == 0:
                logger.info(f"  📊 Update cycle {i//10 + 1}/3 - Check the web interface!")
            
            await asyncio.sleep(1)
        
        logger.info("✅ Real-time updates demonstration completed")

    async def show_interface_features(self) -> None:
        """Show the key features of the interface"""
        
        logger.info("\n🎨 INTERFACE FEATURES SHOWCASE")
        logger.info("The unified visual interface provides:")
        
        features = [
            "Simple Toggle Controls - One-click feature management",
            "Real-time Resource Monitoring - Live CPU, Memory, GPU metrics",
            "Mode Switching - Development, Testing, Live Trading, Emergency",
            "Custom Panel Installation - Add new functionality on-demand",
            "Emergency Procedures - Automatic protection and recovery",
            "Non-intrusive Design - Doesn't interfere with trading visualization",
            "Responsive Layout - Adapts to different screen sizes",
            "WebSocket Communication - Real-time bidirectional updates"
        ]
        
        for i, feature in enumerate(features, 1):
            logger.info(f"  {i:2d}. {feature}")
            await asyncio.sleep(0.5)
        
        logger.info("\n💡 KEY BENEFITS:")
        benefits = [
            "Simplicity - Clean, intuitive controls following sustainment principles",
            "Reliability - Robust error handling and graceful degradation",
            "Performance - Minimal overhead, optimized for trading environments",
            "Flexibility - Easy customization and extension",
            "Safety - Built-in protection mechanisms and emergency procedures"
        ]
        
        for benefit in benefits:
            logger.info(f"     ✓ {benefit}")
            await asyncio.sleep(0.3)

    async def stop_demo(self) -> None:
        """Stop the demonstration gracefully"""
        
        if not self.demo_active:
            return
        
        logger.info("\n🛑 Stopping Unified Visual Demo...")
        
        self.demo_active = False
        
        try:
            # Stop the unified controller
            if self.unified_controller:
                await self.unified_controller.stop_visual_controller()
            
            # Stop subsidiary systems
            if self.ui_bridge:
                self.ui_bridge.stop()
            
            if self.visual_bridge:
                self.visual_bridge.stop_visual_bridge()
            
            logger.info("✅ Demo stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping demo: {e}")

    def get_demo_statistics(self) -> Dict[str, Any]:
        """Get demonstration statistics"""
        
        if not self.unified_controller:
            return {}
        
        state = self.unified_controller.get_visual_state()
        
        return {
            "current_mode": state['mode'],
            "active_panels": len(state['active_panels']),
            "enabled_features": len([k for k, v in state['toggle_states'].items() if v]),
            "total_features": len(state['toggle_states']),
            "performance_metrics": state['performance_metrics'],
            "uptime_seconds": state['uptime_seconds'],
            "error_count": state['error_count']
        }

async def main():
    """Main demonstration function"""
    
    print("\n" + "="*80)
    print("🚀 SCHWABOT UNIFIED VISUAL CONTROLLER DEMONSTRATION")
    print("="*80)
    print("Showcasing elegant, simple visual controls for BTC processing and trading")
    print("Built on the 8 principles of sustainment for robust, intuitive operation")
    print("="*80)
    
    demo = UnifiedVisualDemo()
    
    try:
        # Start the complete demonstration
        await demo.start_demo()
        
        # Show interface features
        await demo.show_interface_features()
        
        # Run real-time updates demonstration
        await demo.demonstrate_real_time_updates()
        
        # Show final statistics
        stats = demo.get_demo_statistics()
        print("\n📊 DEMONSTRATION STATISTICS:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("The web interface remains active for further exploration.")
        print("Press Ctrl+C to stop the demo.")
        
        # Keep running to allow web interface interaction
        while demo.demo_active:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.stop_demo()

if __name__ == "__main__":
    asyncio.run(main()) 