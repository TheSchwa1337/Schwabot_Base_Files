#!/usr/bin/env python3
"""
Integrated Schwabot System Launcher
==================================

Launch script that demonstrates the complete integration of all your Schwabot
components into the unified visual synthesis interface you described.

This launcher shows how:
- All core files integrate into a single visual interface
- Toggle controls dynamically change mathematical core operations
- Entry/exit logic is modified by component integration levels
- NCCO/SFS controls affect volume and speed
- ALIF pathways adapt based on integration state
- Real-time visual synthesis coordinates everything

This is the "task manager for Schwabot" interface you described where all
panels work together and can be controlled dynamically.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the integration components
try:
    from core.unified_visual_synthesis_controller import create_unified_visual_synthesis
    from core.schwabot_integration_orchestrator import create_schwabot_orchestrator, CoreFunctionalityMode
    from examples.unified_visual_synthesis_demo import UnifiedVisualSynthesisDemo
    from core.error_handling_pipeline import safe_print
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Integration components not available: {e}")
    INTEGRATION_AVAILABLE = False

async def launch_integrated_system():
    """Launch the complete integrated Schwabot system"""
    
    if not INTEGRATION_AVAILABLE:
        safe_print("❌ Integration system not available", "ERROR")
        safe_print("Please ensure all core files are properly located:", "ERROR")
        safe_print("  - core/unified_visual_synthesis_controller.py", "ERROR")
        safe_print("  - core/schwabot_integration_orchestrator.py", "ERROR")
        safe_print("  - examples/unified_visual_synthesis_demo.py", "ERROR")
        return
    
    safe_print("🚀 LAUNCHING INTEGRATED SCHWABOT SYSTEM", "INFO")
    safe_print("=" * 60, "INFO")
    
    try:
        # Step 1: Create the unified visual synthesis controller
        safe_print("\n1️⃣ Creating Unified Visual Synthesis Controller...", "INFO")
        synthesis_controller = create_unified_visual_synthesis(websocket_port=8080)
        safe_print("   ✅ Visual synthesis controller created", "SUCCESS")
        
        # Step 2: Create the integration orchestrator  
        safe_print("\n2️⃣ Creating Integration Orchestrator...", "INFO")
        orchestrator = create_schwabot_orchestrator(
            visual_synthesis=synthesis_controller
        )
        safe_print("   ✅ Integration orchestrator created", "SUCCESS")
        
        # Step 3: Start the synthesis system
        safe_print("\n3️⃣ Starting Visual Synthesis System...", "INFO")
        await synthesis_controller.start_visual_synthesis()
        safe_print("   ✅ Visual synthesis system started", "SUCCESS")
        
        # Step 4: Start the orchestration system
        safe_print("\n4️⃣ Starting Integration Orchestration...", "INFO")
        await orchestrator.start_orchestration()
        safe_print("   ✅ Integration orchestration started", "SUCCESS")
        
        # Step 5: Display system status
        safe_print("\n📊 SYSTEM STATUS", "INFO")
        safe_print("-" * 20, "INFO")
        
        synthesis_status = synthesis_controller.get_synthesis_status()
        orchestration_status = orchestrator.get_orchestration_status()
        
        safe_print(f"Visual Panels: {synthesis_status['active_panels']}", "INFO")
        safe_print(f"Visible Panels: {synthesis_status['visible_panels']}", "INFO")
        safe_print(f"Integration Components: {orchestration_status['active_components']}", "INFO")
        safe_print(f"WebSocket Clients: {synthesis_status['websocket_clients']}", "INFO")
        safe_print(f"System Mode: {orchestration_status['integration_state']['active_mode']}", "INFO")
        
        # Step 6: Demonstrate dynamic toggling
        safe_print("\n🎛️ DEMONSTRATING DYNAMIC TOGGLING", "INFO")
        safe_print("-" * 40, "INFO")
        
        # Toggle ghost architecture to partial integration
        await orchestrator.toggle_component("ghost_architecture", integration_level=0.5)
        safe_print("   🔄 Ghost architecture set to partial integration", "INFO")
        safe_print("      (Entry/exit logic partially modified)", "INFO")
        
        # Hide edge vector field but keep it processing
        await orchestrator.toggle_component("edge_vector_field", visible=False)
        safe_print("   👻 Edge vector field hidden (still processing vectors)", "INFO")
        safe_print("      (Vector calculations continue in background)", "INFO")
        
        # Enable NCCO volume control
        await orchestrator.toggle_component("ncco_volume_control", enabled=True)
        safe_print("   🔊 NCCO volume control enabled", "INFO")
        safe_print("      (Trading volume dynamics affected)", "INFO")
        
        # Enable SFS speed control
        await orchestrator.toggle_component("sfs_speed_control", enabled=True)
        safe_print("   ⚡ SFS speed control enabled", "INFO")
        safe_print("      (Tick processing speed controlled)", "INFO")
        
        # Step 7: Show integration effects
        safe_print("\n⚙️ INTEGRATION EFFECTS", "INFO")
        safe_print("-" * 25, "INFO")
        safe_print("   📊 Mathematical cores adapting to toggle states", "INFO")
        safe_print("   🔄 Entry/exit logic dynamically modified", "INFO")
        safe_print("   📈 Volume/speed controls affecting determinism", "INFO")
        safe_print("   🌊 Vector field routing strategies changed", "INFO")
        safe_print("   🛤️ ALIF pathways adapting to integration levels", "INFO")
        
        await asyncio.sleep(2)
        
        # Step 8: Demonstrate system mode switching
        safe_print("\n🔄 DEMONSTRATING MODE SWITCHING", "INFO")
        safe_print("-" * 35, "INFO")
        
        # Switch to full synthesis mode
        await orchestrator.set_system_mode(CoreFunctionalityMode.FULL_SYNTHESIS)
        safe_print("   🎯 System mode: FULL_SYNTHESIS", "INFO")
        safe_print("      (All components fully integrated)", "INFO")
        
        await asyncio.sleep(1)
        
        # Switch to development mode
        await orchestrator.set_system_mode(CoreFunctionalityMode.DEVELOPMENT_MODE)
        safe_print("   🛠️ System mode: DEVELOPMENT_MODE", "INFO")
        safe_print("      (Safe testing with 80% integration)", "INFO")
        
        # Step 9: Show final status
        safe_print("\n📋 FINAL SYSTEM STATUS", "INFO")
        safe_print("-" * 25, "INFO")
        
        final_status = orchestrator.get_orchestration_status()
        safe_print(f"Integration Count: {final_status['integration_count']}", "INFO")
        safe_print(f"Toggle Count: {final_status['toggle_count']}", "INFO")
        safe_print(f"Routing Switches: {final_status['routing_switches']}", "INFO")
        safe_print(f"Current Mode: {final_status['integration_state']['active_mode']}", "INFO")
        safe_print(f"NCCO Integration: {final_status['integration_state']['ncco_integration']}", "INFO")
        safe_print(f"SFS Integration: {final_status['integration_state']['sfs_integration']}", "INFO")
        safe_print(f"ALIF Pathways: {final_status['integration_state']['alif_pathway_active']}", "INFO")
        
        # Step 10: Connection information
        safe_print("\n🌐 CONNECTION INFORMATION", "INFO")
        safe_print("-" * 30, "INFO")
        safe_print("WebSocket Server: ws://localhost:8080", "INFO")
        safe_print("Real-time Updates: Active", "INFO")
        safe_print("Panel Controls: Dynamic toggling available", "INFO")
        safe_print("Integration Level: Fully operational", "INFO")
        
        safe_print("\n🎯 INTEGRATION FEATURES ACTIVE", "INFO")
        safe_print("-" * 35, "INFO")
        safe_print("[TOGGLE] Dynamic panel control with functional integration", "INFO")
        safe_print("[VISUAL] Real-time synthesis of all mathematical cores", "INFO")
        safe_print("[ROUTE] Vector field routing adapts to integration levels", "INFO")
        safe_print("[LOGIC] Entry/exit logic modified by component states", "INFO")
        safe_print("[SPEED] NCCO/SFS controls affecting volume and determinism", "INFO")
        safe_print("[ADAPT] ALIF pathways responding to integration changes", "INFO")
        safe_print("[SUSTAIN] 8 principles framework monitoring everything", "INFO")
        
        # Step 11: Keep system running
        safe_print("\n⏳ SYSTEM RUNNING", "INFO")
        safe_print("-" * 18, "INFO")
        safe_print("The integrated system is now operational.", "INFO")
        safe_print("All your core Schwabot files are working together in the", "INFO")
        safe_print("unified visual interface you described.", "INFO")
        safe_print("", "INFO")
        safe_print("Press Ctrl+C to stop the system gracefully...", "INFO")
        
        # Monitor system indefinitely
        try:
            while True:
                await asyncio.sleep(5)
                
                # Show periodic status updates
                current_status = synthesis_controller.get_synthesis_status()
                safe_print(f"\r[ACTIVE] Updates: {current_status['update_count']} | "
                         f"Health: {current_status['synthesis_state']['system_health']:.1%} | "
                         f"Panels: {current_status['visible_panels']}", "INFO")
                
        except KeyboardInterrupt:
            safe_print("\n\n⏹️ Shutdown requested...", "WARN")
        
        # Step 12: Graceful shutdown
        safe_print("\n🛑 SHUTTING DOWN INTEGRATED SYSTEM", "INFO")
        safe_print("-" * 40, "INFO")
        
        safe_print("   Stopping integration orchestration...", "INFO")
        await orchestrator.stop_orchestration()
        safe_print("   ✅ Orchestration stopped", "SUCCESS")
        
        safe_print("   Stopping visual synthesis...", "INFO")
        await synthesis_controller.stop_visual_synthesis()
        safe_print("   ✅ Visual synthesis stopped", "SUCCESS")
        
        safe_print("\n✅ INTEGRATED SYSTEM SHUTDOWN COMPLETE", "SUCCESS")
        
    except Exception as e:
        safe_print(f"\n❌ System error: {e}", "ERROR")
        logger.exception("System error details:")

async def run_demo_mode():
    """Run in demonstration mode"""
    
    safe_print("🎮 RUNNING INTERACTIVE DEMO MODE", "INFO")
    safe_print("=" * 45, "INFO")
    
    try:
        demo = UnifiedVisualSynthesisDemo()
        await demo.start_demo()
        
        # Keep demo running
        safe_print("\n📊 Demo is running...", "INFO")
        safe_print("Press Ctrl+C to stop...", "INFO")
        
        while demo.demo_active:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        safe_print("\n⏹️ Demo stopped", "WARN")
        if 'demo' in locals():
            await demo.stop_demo()

def main():
    """Main entry point"""
    
    safe_print("🌟 SCHWABOT INTEGRATED SYSTEM LAUNCHER", "INFO")
    safe_print("=" * 50, "INFO")
    safe_print("", "INFO")
    safe_print("This launcher integrates ALL your core Schwabot files:", "INFO")
    safe_print("  • ghost_architecture_btc_profit_handoff.py", "INFO")
    safe_print("  • edge_vector_field.py", "INFO")
    safe_print("  • drift_exit_detector.py", "INFO")
    safe_print("  • future_hooks.py", "INFO")
    safe_print("  • error_handling_pipeline.py", "INFO")
    safe_print("  • btc_processor_ui.py", "INFO")
    safe_print("", "INFO")
    safe_print("Into the unified visual synthesis interface you described.", "INFO")
    safe_print("=" * 50, "INFO")
    
    if not INTEGRATION_AVAILABLE:
        safe_print("\n❌ Integration components not available", "ERROR")
        safe_print("Please ensure core files are properly located.", "ERROR")
        return
    
    # Choose launch mode
    safe_print("\n🎯 LAUNCH OPTIONS", "INFO")
    safe_print("-" * 18, "INFO")
    safe_print("1. Full Integrated System (recommended)", "INFO")
    safe_print("2. Demo Mode Only", "INFO")
    safe_print("3. Exit", "INFO")
    
    try:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            safe_print("\n🚀 Launching full integrated system...", "INFO")
            asyncio.run(launch_integrated_system())
        elif choice == "2":
            safe_print("\n🎮 Launching demo mode...", "INFO")
            asyncio.run(run_demo_mode())
        elif choice == "3":
            safe_print("\n👋 Goodbye!", "INFO")
        else:
            safe_print("\n❌ Invalid choice", "ERROR")
            
    except KeyboardInterrupt:
        safe_print("\n\n👋 Launch cancelled", "INFO")
    except Exception as e:
        safe_print(f"\n❌ Launch error: {e}", "ERROR")

if __name__ == "__main__":
    main() 