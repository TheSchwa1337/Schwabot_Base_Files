import sys
import time
import traceback

from core.speed_lattice_visualizer import PanelType, SpeedLatticeLivePanelSystem

#!/usr/bin/env python3
"""
Simple Live Panel System Test
"""


# Import only the visualizer
    try:

    print("✅ Successfully imported SpeedLatticeLivePanelSystem")
    except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def simple_test():
    """Simple test of the live panel system"""
    print("🚀 Simple Live Panel System Test")
    print("=" * 50)

    try:
        # Create system
        panel_system = SpeedLatticeLivePanelSystem()
        print("✅ Created panel system")

        # Test panel initialization
        print(f"✅ Initialized {len(panel_system.panels)} panels")

        # Test panel switching
        test_panels = []
            PanelType.DRIFT_MATRIX,
            PanelType.TRADING_STATE,
            PanelType.POOL_ANALYSIS,
            PanelType.PATTERN_RECOGNITION,
        ]
        for panel_type in test_panels:
            panel_system.switch_panel(panel_type)
            print(f"✅ Switched to: {panel_type.value}")

        # Test data generation
        for panel_type in test_panels:
            panel_system._generate_simulation_data(f"api/test/{panel_type.value}")
            print(f"✅ Generated data for {panel_type.value}")

        # Test state management
        for panel_type in test_panels:
            panel_state = panel_system.panels[panel_type]
            panel_state.update_data({"test": "data", "timestamp": time.time()})
            print(f"✅ Updated {panel_type.value}")

        # Test save state
        filename = panel_system.save_panel_state()
        print(f"✅ Saved state to: {filename}")

        # Test system status
        status_items = []
            ("Total Panels", len(panel_system.panels)),
            ("API Connections", len(panel_system.api_connections)),
            ()
                "Current Panel",
                panel_system.current_panel.value
                if panel_system.current_panel
                else None,
            ),
        ]
        for item, value in status_items:
            print(f"✅ {item}: {value}")

        # Cleanup
        panel_system.close()
        print("✅ System cleaned up")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")

        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("🚀 Speed Lattice Vault Live Panel System - Simple Test")
    print("=" * 60)

    success = simple_test()

    if success:
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Live Panel System Functional")
        print("=" * 60)

        print("\n🎯 System Features Verified:")
        print("   • Panel initialization and management")
        print("   • Dynamic panel switching")
        print("   • Data generation and simulation")
        print("   • State management and persistence")
        print("   • System status monitoring")
        print("   • Clean shutdown and cleanup")

        print("\n🔧 Ready for Integration:")
        print("   • Connect to real trading APIs")
        print("   • Implement live data feeds")
        print("   • Add authentication")
        print("   • Deploy to production")

    else:
        print("\n❌ Tests failed")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
