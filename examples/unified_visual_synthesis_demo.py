#!/usr/bin/env python3
"""
Unified Visual Synthesis System Demonstration
===========================================

Demonstrates the complete integration of all Schwabot visual components into
the unified interface you described - like a task manager for Schwabot where
all panels work together and can be controlled dynamically.

This demo shows:
- BTC Processor UI integration
- Ghost Architecture profit handoff visualization
- Edge Vector Field pattern detection
- Drift Exit Detector analysis
- Future Hooks evaluation system
- Error Handling Pipeline status
- Sustainment Underlay metrics
- Real-time visual synthesis

The interface implements the 8 principles of sustainment and provides the
integrated visual experience where toggle controls dynamically change how
core functionality is displayed and integrated.
"""

import asyncio
import logging
import time
import webbrowser
import threading
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the unified visual synthesis controller
try:
    from core.unified_visual_synthesis_controller import (
        UnifiedVisualSynthesisController, 
        create_unified_visual_synthesis
    )
    from core.error_handling_pipeline import safe_print, convert_emojis
    SYNTHESIS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import synthesis controller: {e}")
    SYNTHESIS_AVAILABLE = False

class UnifiedVisualSynthesisDemo:
    """
    Comprehensive demonstration of the unified visual synthesis system.
    
    This shows how all your core files integrate into the single visual
    interface you described - the task manager-like interface for Schwabot.
    """
    
    def __init__(self):
        self.synthesis_controller = None
        self.demo_active = False
        self.web_server_port = 8080
        self.websocket_port = 8081
        
        # Demo configuration
        self.demo_config = {
            "show_all_panels": True,
            "auto_start_components": True,
            "enable_real_time_updates": True,
            "demo_duration": 300,  # 5 minutes
            "update_frequency": 2.0  # 2 Hz
        }
        
        safe_print("🚀 Unified Visual Synthesis Demo initialized", "INFO")
    
    async def setup_synthesis_system(self):
        """Setup the complete unified visual synthesis system"""
        
        if not SYNTHESIS_AVAILABLE:
            safe_print("❌ Synthesis system not available - check imports", "ERROR")
            return False
        
        try:
            safe_print("🔧 Setting up unified visual synthesis system...", "INFO")
            
            # Create the unified visual synthesis controller
            # This integrates all your core files into one interface
            self.synthesis_controller = create_unified_visual_synthesis(
                websocket_port=self.websocket_port
            )
            
            safe_print("✅ Unified Visual Synthesis Controller created", "SUCCESS")
            
            # Display the integrated components
            self._display_integrated_components()
            
            return True
            
        except Exception as e:
            safe_print(f"❌ Failed to setup synthesis system: {e}", "ERROR")
            return False
    
    def _display_integrated_components(self):
        """Display all the integrated core components"""
        
        safe_print("\n📋 INTEGRATED CORE COMPONENTS", "INFO")
        safe_print("=" * 50, "INFO")
        
        components = [
            ("🔧 BTC Processor UI", "btc_processor_ui.py", "Web-based BTC processor control"),
            ("👻 Ghost Architecture", "ghost_architecture_btc_profit_handoff.py", "Advanced ghost profit handoff"),
            ("🌊 Edge Vector Field", "edge_vector_field.py", "Edge-case pattern detection"),
            ("🌪️ Drift Exit Detector", "drift_exit_detector.py", "Drift entropy detection"),
            ("🔮 Future Hooks", "future_hooks.py", "Hook evaluation system"),
            ("🛡️ Error Handling", "error_handling_pipeline.py", "Windows CLI compatibility"),
            ("⚖️ Sustainment Metrics", "sustainment_underlay_controller.py", "8 principles framework"),
            ("🎛️ Visual Integration", "unified_visual_controller.py", "Master visual controller")
        ]
        
        for emoji_desc, filename, description in components:
            # Convert emojis for Windows compatibility
            safe_desc = convert_emojis(emoji_desc)
            safe_print(f"{safe_desc}: {description}", "INFO")
            safe_print(f"    File: {filename}", "INFO")
        
        safe_print("\n🎯 INTEGRATION FEATURES", "INFO")
        safe_print("=" * 30, "INFO")
        safe_print("[TOGGLE] Dynamic panel control - each toggle changes functionality", "INFO")
        safe_print("[VISUAL] Real-time synthesis of all mathematical cores", "INFO")
        safe_print("[STREAM] WebSocket data streaming to frontend", "INFO")
        safe_print("[MONITOR] Task manager-like interface for Schwabot", "INFO")
        safe_print("[SUSTAIN] 8 principles sustainment framework integration", "INFO")
    
    async def start_demo(self):
        """Start the complete demonstration"""
        
        if self.demo_active:
            safe_print("⚠️ Demo already active", "WARN")
            return
        
        # Setup the synthesis system
        setup_success = await self.setup_synthesis_system()
        if not setup_success:
            return
        
        self.demo_active = True
        
        try:
            safe_print("\n🚀 STARTING UNIFIED VISUAL SYNTHESIS DEMO", "INFO")
            safe_print("=" * 60, "INFO")
            
            # Start the synthesis controller
            await self.synthesis_controller.start_visual_synthesis()
            
            # Display connection information
            self._display_connection_info()
            
            # Run demonstration scenarios
            await self._run_demonstration_scenarios()
            
        except Exception as e:
            safe_print(f"❌ Demo error: {e}", "ERROR")
            await self.stop_demo()
    
    def _display_connection_info(self):
        """Display connection information for the interface"""
        
        safe_print("\n🌐 CONNECTION INFORMATION", "INFO")
        safe_print("=" * 40, "INFO")
        safe_print(f"WebSocket Server: ws://localhost:{self.websocket_port}", "INFO")
        safe_print(f"Real-time Updates: Active (2 Hz)", "INFO")
        safe_print(f"Panel Count: {len(self.synthesis_controller.synthesis_state.active_panels)}", "INFO")
        
        # Show active panels
        safe_print("\n📊 ACTIVE VISUAL PANELS", "INFO")
        safe_print("-" * 30, "INFO")
        
        for panel_id, panel_state in self.synthesis_controller.synthesis_state.active_panels.items():
            status = "[VISIBLE]" if panel_state.is_visible else "[HIDDEN]"
            panel_name = panel_state.panel_type.value.replace("_", " ").title()
            safe_print(f"{status} {panel_name} ({panel_id})", "INFO")
        
        safe_print("\n💡 INTERFACE FEATURES", "INFO")
        safe_print("-" * 25, "INFO")
        safe_print("[CONTROL] Toggle any panel on/off dynamically", "INFO")
        safe_print("[CONFIG] Resize and reposition panels", "INFO")
        safe_print("[DATA] Real-time data from all core systems", "INFO")
        safe_print("[COMMAND] System commands and emergency controls", "INFO")
        safe_print("[MODE] Switch between development/testing/live trading", "INFO")
    
    async def _run_demonstration_scenarios(self):
        """Run various demonstration scenarios"""
        
        scenarios = [
            ("Panel Toggle Demo", self._demo_panel_toggles),
            ("System Health Monitoring", self._demo_system_health),
            ("Ghost Architecture Handoffs", self._demo_ghost_handoffs),
            ("Edge Vector Detection", self._demo_edge_vectors),
            ("Sustainment Principles", self._demo_sustainment_principles),
            ("Error Handling Pipeline", self._demo_error_handling),
            ("Emergency Procedures", self._demo_emergency_procedures)
        ]
        
        safe_print("\n🎭 RUNNING DEMONSTRATION SCENARIOS", "INFO")
        safe_print("=" * 50, "INFO")
        
        for scenario_name, scenario_func in scenarios:
            try:
                safe_print(f"\n▶️ Starting: {scenario_name}", "INFO")
                await scenario_func()
                safe_print(f"✅ Completed: {scenario_name}", "SUCCESS")
                
                # Pause between scenarios
                await asyncio.sleep(2)
                
            except KeyboardInterrupt:
                safe_print("\n⏹️ Demo interrupted by user", "WARN")
                break
            except Exception as e:
                safe_print(f"❌ Scenario error in {scenario_name}: {e}", "ERROR")
        
        safe_print("\n🎯 All demonstration scenarios completed", "SUCCESS")
    
    async def _demo_panel_toggles(self):
        """Demonstrate dynamic panel toggling"""
        
        safe_print("Demonstrating panel toggle functionality...", "INFO")
        
        # Get panels to toggle
        panels = list(self.synthesis_controller.synthesis_state.active_panels.keys())
        
        for panel_id in panels[:3]:  # Demo first 3 panels
            # Hide panel
            await self.synthesis_controller._handle_toggle_panel({
                "panel_id": panel_id,
                "enabled": False
            })
            safe_print(f"  Hidden panel: {panel_id}", "INFO")
            await asyncio.sleep(1)
            
            # Show panel
            await self.synthesis_controller._handle_toggle_panel({
                "panel_id": panel_id,
                "enabled": True
            })
            safe_print(f"  Restored panel: {panel_id}", "INFO")
            await asyncio.sleep(1)
    
    async def _demo_system_health(self):
        """Demonstrate system health monitoring"""
        
        safe_print("Showing system health monitoring...", "INFO")
        
        # Get current status
        status = self.synthesis_controller.get_synthesis_status()
        
        safe_print(f"  System Health: {status['synthesis_state']['system_health']:.1%}", "INFO")
        safe_print(f"  Sustainment Index: {status['synthesis_state']['sustainment_index']:.3f}", "INFO")
        safe_print(f"  Active Panels: {status['active_panels']}", "INFO")
        safe_print(f"  Visible Panels: {status['visible_panels']}", "INFO")
        safe_print(f"  WebSocket Clients: {status['websocket_clients']}", "INFO")
        safe_print(f"  Uptime: {status['uptime']:.1f} seconds", "INFO")
    
    async def _demo_ghost_handoffs(self):
        """Demonstrate ghost architecture profit handoffs"""
        
        safe_print("Simulating ghost architecture profit handoffs...", "INFO")
        
        # Simulate handoff operations
        ghost_data = self.synthesis_controller._get_ghost_architecture_data()
        
        safe_print(f"  Active Phantoms: {ghost_data['active_phantoms']}", "INFO")
        safe_print(f"  Handoff Success Rate: {ghost_data['handoff_success_rate']:.1%}", "INFO")
        safe_print(f"  Available Strategies: {len(ghost_data['handoff_strategies'])}", "INFO")
        
        for strategy in ghost_data['handoff_strategies'][:3]:
            safe_print(f"    - {strategy.replace('_', ' ').title()}", "INFO")
    
    async def _demo_edge_vectors(self):
        """Demonstrate edge vector field detection"""
        
        safe_print("Analyzing edge vector field patterns...", "INFO")
        
        edge_data = self.synthesis_controller._get_edge_vector_field_data()
        
        safe_print(f"  Total Detections: {edge_data['stats']['total_detections']}", "INFO")
        safe_print(f"  Success Rate: {edge_data['stats']['success_rate']:.1%}", "INFO")
        
        safe_print("  Active Patterns:", "INFO")
        for pattern in edge_data['active_patterns']:
            safe_print(f"    - {pattern['name']}: {pattern['confidence']:.1%} confidence", "INFO")
    
    async def _demo_sustainment_principles(self):
        """Demonstrate 8 principles sustainment framework"""
        
        safe_print("Evaluating 8 principles sustainment framework...", "INFO")
        
        sustainment_data = self.synthesis_controller._get_sustainment_metrics_data()
        
        safe_print(f"  Sustainment Index: {sustainment_data['sustainment_index']:.3f}", "INFO")
        safe_print(f"  System Health: {sustainment_data['system_health']:.1%}", "INFO")
        safe_print(f"  Status: {sustainment_data['status']}", "INFO")
        
        safe_print("  Principle Scores:", "INFO")
        for principle, score in sustainment_data['principles'].items():
            safe_print(f"    - {principle.title()}: {score:.1%}", "INFO")
    
    async def _demo_error_handling(self):
        """Demonstrate error handling pipeline"""
        
        safe_print("Testing error handling pipeline...", "INFO")
        
        error_data = self.synthesis_controller._get_error_handling_data()
        
        safe_print(f"  Emojis Converted: {error_data['conversion_stats']['emojis_converted']}", "INFO")
        safe_print(f"  Errors Prevented: {error_data['conversion_stats']['errors_prevented']}", "INFO")
        safe_print(f"  Critical Failures Avoided: {error_data['critical_errors_prevented']}", "INFO")
        safe_print(f"  Windows Compatibility: {error_data['windows_compatibility']}", "INFO")
        
        safe_print("  Recent Conversions:", "INFO")
        for conversion in error_data['recent_conversions'][:2]:
            safe_print(f"    '{conversion['original']}' -> '{conversion['converted']}'", "INFO")
    
    async def _demo_emergency_procedures(self):
        """Demonstrate emergency procedures"""
        
        safe_print("Testing emergency procedures...", "INFO")
        
        # Simulate emergency cleanup
        await self.synthesis_controller._handle_system_command({
            "command": "emergency_cleanup",
            "parameters": {}
        })
        
        safe_print("  Emergency cleanup executed", "INFO")
        
        # Show system mode switching
        for mode in ["development", "testing"]:
            await self.synthesis_controller._handle_system_command({
                "command": "system_mode",
                "parameters": {"mode": mode}
            })
            safe_print(f"  System mode set to: {mode}", "INFO")
            await asyncio.sleep(0.5)
    
    async def _monitor_real_time_data(self):
        """Monitor and display real-time data updates"""
        
        safe_print("\n📊 REAL-TIME DATA MONITORING", "INFO")
        safe_print("Press Ctrl+C to stop monitoring...", "INFO")
        
        try:
            while self.demo_active:
                # Get current synthesis status
                status = self.synthesis_controller.get_synthesis_status()
                
                # Display key metrics
                safe_print(f"\rSynthesis Count: {status['synthesis_count']} | "
                         f"Updates: {status['update_count']} | "
                         f"Clients: {status['websocket_clients']} | "
                         f"Health: {status['synthesis_state']['system_health']:.1%}", "INFO")
                
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            safe_print("\n⏹️ Real-time monitoring stopped", "WARN")
    
    async def stop_demo(self):
        """Stop the demonstration"""
        
        if not self.demo_active:
            return
        
        self.demo_active = False
        
        try:
            safe_print("\n🛑 Stopping Unified Visual Synthesis Demo...", "INFO")
            
            if self.synthesis_controller:
                await self.synthesis_controller.stop_visual_synthesis()
            
            safe_print("✅ Demo stopped successfully", "SUCCESS")
            
        except Exception as e:
            safe_print(f"❌ Error stopping demo: {e}", "ERROR")
    
    def get_demo_summary(self) -> Dict[str, Any]:
        """Get demonstration summary"""
        
        if not self.synthesis_controller:
            return {"status": "not_initialized"}
        
        status = self.synthesis_controller.get_synthesis_status()
        
        return {
            "demo_active": self.demo_active,
            "synthesis_status": status,
            "integrated_components": [
                "btc_processor_ui",
                "ghost_architecture_btc_profit_handoff", 
                "edge_vector_field",
                "drift_exit_detector",
                "future_hooks",
                "error_handling_pipeline",
                "sustainment_underlay_controller"
            ],
            "key_features": [
                "Dynamic panel control",
                "Real-time visual synthesis",
                "8 principles sustainment",
                "WebSocket data streaming",
                "Emergency procedures",
                "System health monitoring"
            ]
        }

async def run_quick_demo():
    """Run a quick demonstration of the system"""
    
    safe_print("🚀 QUICK UNIFIED VISUAL SYNTHESIS DEMO", "INFO")
    safe_print("=" * 50, "INFO")
    
    demo = UnifiedVisualSynthesisDemo()
    
    try:
        # Start the demo
        await demo.start_demo()
        
        # Monitor for a short time
        safe_print("\n📊 Monitoring system for 30 seconds...", "INFO")
        for i in range(30):
            if demo.synthesis_controller:
                status = demo.synthesis_controller.get_synthesis_status()
                safe_print(f"[{i+1:2d}/30] Updates: {status['update_count']} | "
                         f"Health: {status['synthesis_state']['system_health']:.1%}", "INFO")
            await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        safe_print("\n⏹️ Demo interrupted by user", "WARN")
    
    finally:
        # Stop the demo
        await demo.stop_demo()
        
        # Show summary
        summary = demo.get_demo_summary()
        safe_print("\n📋 DEMO SUMMARY", "INFO")
        safe_print("=" * 20, "INFO")
        safe_print(f"Components Integrated: {len(summary.get('integrated_components', []))}", "INFO")
        safe_print(f"Key Features: {len(summary.get('key_features', []))}", "INFO")
        safe_print("Demo Status: Completed", "SUCCESS")

async def run_interactive_demo():
    """Run an interactive demonstration"""
    
    safe_print("🎮 INTERACTIVE UNIFIED VISUAL SYNTHESIS DEMO", "INFO")
    safe_print("=" * 60, "INFO")
    
    demo = UnifiedVisualSynthesisDemo()
    
    try:
        # Start the demo
        await demo.start_demo()
        
        # Interactive menu
        while demo.demo_active:
            safe_print("\n🎛️ DEMO CONTROLS", "INFO")
            safe_print("-" * 20, "INFO")
            safe_print("1. Show system status", "INFO")
            safe_print("2. Toggle random panel", "INFO")
            safe_print("3. Emergency cleanup", "INFO")
            safe_print("4. Show sustainment metrics", "INFO")
            safe_print("5. Monitor real-time data", "INFO")
            safe_print("6. Stop demo", "INFO")
            
            try:
                choice = input("\nEnter choice (1-6): ").strip()
                
                if choice == "1":
                    status = demo.synthesis_controller.get_synthesis_status()
                    safe_print(f"System Status: {json.dumps(status['synthesis_state'], indent=2)}", "INFO")
                
                elif choice == "2":
                    panels = list(demo.synthesis_controller.synthesis_state.active_panels.keys())
                    if panels:
                        panel = panels[0]
                        current_state = demo.synthesis_controller.synthesis_state.active_panels[panel].is_visible
                        await demo.synthesis_controller._handle_toggle_panel({
                            "panel_id": panel,
                            "enabled": not current_state
                        })
                        safe_print(f"Toggled panel '{panel}' to {'visible' if not current_state else 'hidden'}", "INFO")
                
                elif choice == "3":
                    await demo.synthesis_controller._emergency_cleanup()
                    safe_print("Emergency cleanup completed", "SUCCESS")
                
                elif choice == "4":
                    sustainment_data = demo.synthesis_controller._get_sustainment_metrics_data()
                    safe_print(f"Sustainment Metrics: {json.dumps(sustainment_data, indent=2)}", "INFO")
                
                elif choice == "5":
                    await demo._monitor_real_time_data()
                
                elif choice == "6":
                    break
                
                else:
                    safe_print("Invalid choice", "WARN")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                safe_print(f"Error: {e}", "ERROR")
    
    finally:
        await demo.stop_demo()

def main():
    """Main demonstration entry point"""
    
    safe_print("🌟 UNIFIED VISUAL SYNTHESIS SYSTEM DEMONSTRATION", "INFO")
    safe_print("=" * 70, "INFO")
    safe_print("This demonstrates the integration of ALL your core Schwabot files", "INFO")
    safe_print("into the unified visual interface you described.", "INFO")
    safe_print("=" * 70, "INFO")
    
    if not SYNTHESIS_AVAILABLE:
        safe_print("❌ Synthesis system not available", "ERROR")
        safe_print("Please ensure all core files are in the correct locations:", "ERROR")
        safe_print("  - core/unified_visual_synthesis_controller.py", "ERROR")
        safe_print("  - core/ghost_architecture_btc_profit_handoff.py", "ERROR") 
        safe_print("  - core/edge_vector_field.py", "ERROR")
        safe_print("  - core/drift_exit_detector.py", "ERROR")
        safe_print("  - core/future_hooks.py", "ERROR")
        safe_print("  - core/error_handling_pipeline.py", "ERROR")
        return
    
    # Choose demo type
    safe_print("\n🎯 DEMO OPTIONS", "INFO")
    safe_print("-" * 15, "INFO")
    safe_print("1. Quick Demo (30 seconds)", "INFO")
    safe_print("2. Interactive Demo (full control)", "INFO")
    
    try:
        choice = input("\nSelect demo type (1-2): ").strip()
        
        if choice == "1":
            asyncio.run(run_quick_demo())
        elif choice == "2":
            asyncio.run(run_interactive_demo())
        else:
            safe_print("Invalid choice, running quick demo", "WARN")
            asyncio.run(run_quick_demo())
            
    except KeyboardInterrupt:
        safe_print("\n👋 Demo cancelled by user", "INFO")
    except Exception as e:
        safe_print(f"❌ Demo error: {e}", "ERROR")

if __name__ == "__main__":
    main() 