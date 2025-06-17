#!/usr/bin/env python3
"""
BTC Processor Control Demo
==========================

Demonstrates how to use the BTC processor control features to manage system load
and prevent overload during live testing and hash processing.

This demo shows:
- Starting and stopping analysis features
- Monitoring system resources
- Automatic load management
- Emergency cleanup procedures
- Configuration management
"""

import asyncio
import time
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.btc_processor_controller import BTCProcessorController
from core.btc_data_processor import BTCDataProcessor
from core.btc_processor_ui import BTCProcessorUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BTCProcessorControlDemo:
    """Demonstrates BTC processor control capabilities"""
    
    def __init__(self):
        self.processor = None
        self.controller = None
        self.ui = None
        
    async def setup(self):
        """Initialize the demo environment"""
        try:
            logger.info("🚀 Setting up BTC Processor Control Demo...")
            
            # Initialize BTC processor
            self.processor = BTCDataProcessor("config/btc_processor_config.yaml")
            
            # Initialize controller
            self.controller = BTCProcessorController(self.processor)
            
            # Initialize UI (but don't start the web server)
            self.ui = BTCProcessorUI(self.processor)
            
            logger.info("✅ Demo setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Setup error: {e}")
            raise
            
    async def demonstrate_basic_controls(self):
        """Demonstrate basic feature control"""
        try:
            logger.info("\n📋 === Basic Feature Control Demo ===")
            
            # Show current status
            status = self.controller.get_current_status()
            logger.info(f"Initial status: {len(status.get('feature_states', {}))} features available")
            
            # Disable some analysis features to reduce load
            logger.info("⏹️ Disabling mining analysis to reduce CPU load...")
            await self.controller.disable_feature('mining_analysis')
            
            logger.info("⏹️ Disabling nonce sequence analysis...")
            await self.controller.disable_feature('nonce_sequences')
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Show updated status
            status = self.controller.get_current_status()
            disabled_features = [f for f, enabled in status.get('feature_states', {}).items() if not enabled]
            logger.info(f"✅ Disabled features: {disabled_features}")
            
            # Re-enable features
            logger.info("🔄 Re-enabling analysis features...")
            await self.controller.enable_feature('mining_analysis')
            await self.controller.enable_feature('nonce_sequences')
            
            logger.info("✅ Basic controls demo complete")
            
        except Exception as e:
            logger.error(f"❌ Basic controls demo error: {e}")
            
    async def demonstrate_bulk_controls(self):
        """Demonstrate bulk feature control"""
        try:
            logger.info("\n📋 === Bulk Feature Control Demo ===")
            
            # Disable all analysis features at once (useful for live testing)
            logger.info("⏹️ Disabling ALL analysis features for live testing...")
            await self.controller.disable_all_analysis_features()
            
            # Check system resources after disabling analysis
            await asyncio.sleep(3)
            
            status = self.controller.get_current_status()
            metrics = status.get('system_metrics', {})
            logger.info(f"📊 System metrics after disabling analysis:")
            logger.info(f"   CPU: {metrics.get('cpu_usage', 0):.1f}%")
            logger.info(f"   Memory: {metrics.get('memory_usage_gb', 0):.1f} GB")
            
            # Simulate live testing period
            logger.info("⏳ Simulating live testing period (5 seconds)...")
            await asyncio.sleep(5)
            
            # Re-enable all analysis features
            logger.info("🔄 Re-enabling ALL analysis features...")
            await self.controller.enable_all_analysis_features()
            
            logger.info("✅ Bulk controls demo complete")
            
        except Exception as e:
            logger.error(f"❌ Bulk controls demo error: {e}")
            
    async def demonstrate_threshold_monitoring(self):
        """Demonstrate system threshold monitoring"""
        try:
            logger.info("\n📋 === System Threshold Monitoring Demo ===")
            
            # Set conservative thresholds for demo
            demo_thresholds = {
                'memory_warning': 4.0,   # 4 GB warning
                'memory_critical': 6.0,  # 6 GB critical
                'cpu_warning': 50.0,     # 50% CPU warning
                'cpu_critical': 70.0     # 70% CPU critical
            }
            
            logger.info("⚙️ Setting demo thresholds...")
            self.controller.update_thresholds(demo_thresholds)
            
            for threshold, value in demo_thresholds.items():
                logger.info(f"   {threshold}: {value}")
                
            # Start monitoring
            logger.info("📡 Starting system monitoring...")
            
            # Start monitoring in background
            import threading
            def run_monitoring():
                asyncio.run(self.controller.start_monitoring())
                
            monitor_thread = threading.Thread(target=run_monitoring)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Monitor for a period
            logger.info("⏳ Monitoring system for 10 seconds...")
            for i in range(10):
                status = self.controller.get_current_status()
                metrics = status.get('system_metrics', {})
                
                logger.info(f"📊 [{i+1:2d}/10] CPU: {metrics.get('cpu_usage', 0):5.1f}% | "
                          f"Memory: {metrics.get('memory_usage_gb', 0):5.1f}GB")
                
                await asyncio.sleep(1)
                
            # Stop monitoring
            await self.controller.stop_monitoring()
            logger.info("✅ Threshold monitoring demo complete")
            
        except Exception as e:
            logger.error(f"❌ Threshold monitoring demo error: {e}")
            
    async def demonstrate_emergency_procedures(self):
        """Demonstrate emergency cleanup procedures"""
        try:
            logger.info("\n📋 === Emergency Procedures Demo ===")
            
            # Simulate high memory usage scenario
            logger.info("🚨 Simulating high memory usage scenario...")
            
            # Check current memory usage
            status = self.controller.get_current_status()
            initial_memory = status.get('system_metrics', {}).get('memory_usage_gb', 0)
            logger.info(f"📊 Initial memory usage: {initial_memory:.1f} GB")
            
            # Trigger emergency cleanup
            logger.info("🚨 Triggering emergency memory cleanup...")
            await self.controller._emergency_memory_cleanup()
            
            # Check memory usage after cleanup
            await asyncio.sleep(2)
            status = self.controller.get_current_status()
            final_memory = status.get('system_metrics', {}).get('memory_usage_gb', 0)
            logger.info(f"📊 Memory usage after cleanup: {final_memory:.1f} GB")
            
            # Check which features were disabled
            disabled_features = [f for f, enabled in status.get('feature_states', {}).items() if not enabled]
            logger.info(f"⏹️ Features disabled during cleanup: {disabled_features}")
            
            # Restore features after emergency
            logger.info("🔄 Restoring features after emergency...")
            await self.controller.enable_all_analysis_features()
            
            logger.info("✅ Emergency procedures demo complete")
            
        except Exception as e:
            logger.error(f"❌ Emergency procedures demo error: {e}")
            
    async def demonstrate_configuration_management(self):
        """Demonstrate configuration management"""
        try:
            logger.info("\n📋 === Configuration Management Demo ===")
            
            # Show current configuration
            status = self.controller.get_current_status()
            config = status.get('configuration', {})
            logger.info("📋 Current configuration:")
            for key, value in config.items():
                logger.info(f"   {key}: {value}")
                
            # Update configuration for live testing
            live_testing_config = {
                'max_memory_usage_gb': 8.0,    # Limit to 8GB
                'max_cpu_usage_percent': 60.0,  # Limit to 60% CPU
                'max_gpu_usage_percent': 70.0   # Limit to 70% GPU
            }
            
            logger.info("⚙️ Updating configuration for live testing...")
            self.controller.update_configuration(live_testing_config)
            
            # Show updated configuration
            status = self.controller.get_current_status()
            config = status.get('configuration', {})
            logger.info("📋 Updated configuration:")
            for key, value in config.items():
                logger.info(f"   {key}: {value}")
                
            # Save configuration
            logger.info("💾 Saving configuration...")
            self.controller.save_configuration("demo_config.json")
            
            # Restore original configuration
            logger.info("🔄 Restoring original configuration...")
            original_config = {
                'max_memory_usage_gb': 10.0,
                'max_cpu_usage_percent': 80.0,
                'max_gpu_usage_percent': 85.0
            }
            self.controller.update_configuration(original_config)
            
            logger.info("✅ Configuration management demo complete")
            
        except Exception as e:
            logger.error(f"❌ Configuration management demo error: {e}")
            
    async def demonstrate_live_testing_scenario(self):
        """Demonstrate a complete live testing scenario"""
        try:
            logger.info("\n📋 === Live Testing Scenario Demo ===")
            logger.info("🎯 Scenario: Preparing system for intensive hash processing")
            
            # Step 1: Prepare for live testing
            logger.info("\n📋 Step 1: Preparing for live testing...")
            
            # Disable non-essential analysis
            await self.controller.disable_feature('mining_analysis')
            await self.controller.disable_feature('block_timing')
            await self.controller.disable_feature('difficulty_tracking')
            
            # Keep only essential features
            logger.info("✅ Disabled non-essential analysis features")
            logger.info("✅ Keeping hash generation and load balancing active")
            
            # Step 2: Set conservative thresholds
            logger.info("\n📋 Step 2: Setting conservative resource thresholds...")
            
            testing_thresholds = {
                'memory_warning': 6.0,
                'memory_critical': 8.0,
                'cpu_warning': 60.0,
                'cpu_critical': 80.0
            }
            self.controller.update_thresholds(testing_thresholds)
            logger.info("✅ Conservative thresholds set")
            
            # Step 3: Start monitoring
            logger.info("\n📋 Step 3: Starting intensive monitoring...")
            
            # Start monitoring in background
            import threading
            def run_monitoring():
                asyncio.run(self.controller.start_monitoring())
                
            monitor_thread = threading.Thread(target=run_monitoring)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Step 4: Simulate live testing period
            logger.info("\n📋 Step 4: Simulating live testing period...")
            logger.info("⏳ Running for 15 seconds with real-time monitoring...")
            
            for i in range(15):
                status = self.controller.get_current_status()
                metrics = status.get('system_metrics', {})
                emergency = status.get('emergency_shutdown', False)
                
                status_icon = "🚨" if emergency else "✅"
                logger.info(f"{status_icon} [{i+1:2d}/15] CPU: {metrics.get('cpu_usage', 0):5.1f}% | "
                          f"Memory: {metrics.get('memory_usage_gb', 0):5.1f}GB | "
                          f"GPU: {metrics.get('gpu_usage', 0):5.1f}%")
                
                if emergency:
                    logger.warning("🚨 Emergency shutdown detected!")
                    break
                    
                await asyncio.sleep(1)
                
            # Step 5: Restore normal operations
            logger.info("\n📋 Step 5: Restoring normal operations...")
            
            # Stop monitoring
            await self.controller.stop_monitoring()
            
            # Re-enable all features
            await self.controller.enable_all_analysis_features()
            
            # Restore normal thresholds
            normal_thresholds = {
                'memory_warning': 8.0,
                'memory_critical': 12.0,
                'cpu_warning': 70.0,
                'cpu_critical': 90.0
            }
            self.controller.update_thresholds(normal_thresholds)
            
            logger.info("✅ Live testing scenario complete")
            logger.info("🎯 System successfully managed load during intensive processing")
            
        except Exception as e:
            logger.error(f"❌ Live testing scenario error: {e}")
            
    async def demonstrate_web_ui_integration(self):
        """Demonstrate web UI integration"""
        try:
            logger.info("\n📋 === Web UI Integration Demo ===")
            
            # Create UI instance
            logger.info("🌐 Web UI is available for real-time control")
            logger.info("🔗 To start the web interface, run:")
            logger.info("   python -m core.btc_processor_ui")
            logger.info("🌐 Then open: http://localhost:5000")
            
            logger.info("\n📋 Web UI Features:")
            logger.info("   • Real-time system metrics display")
            logger.info("   • Toggle switches for all features")
            logger.info("   • Emergency cleanup buttons")
            logger.info("   • Threshold configuration")
            logger.info("   • Live status logging")
            logger.info("   • Configuration save/load")
            
            # Show how to get status for web UI
            status = self.controller.get_current_status()
            logger.info(f"\n📊 Current status (as displayed in UI):")
            logger.info(f"   Monitoring: {'Active' if status.get('monitoring_active') else 'Inactive'}")
            logger.info(f"   Emergency: {'YES' if status.get('emergency_shutdown') else 'No'}")
            logger.info(f"   Features: {sum(status.get('feature_states', {}).values())}/{len(status.get('feature_states', {}))}")
            
            logger.info("✅ Web UI integration demo complete")
            
        except Exception as e:
            logger.error(f"❌ Web UI integration demo error: {e}")
            
    async def show_cli_examples(self):
        """Show CLI usage examples"""
        try:
            logger.info("\n📋 === CLI Usage Examples ===")
            
            logger.info("🖥️ Command-line interface examples:")
            logger.info("")
            
            logger.info("📊 Check system status:")
            logger.info("   python tools/btc_processor_cli.py status")
            logger.info("")
            
            logger.info("⏹️ Disable all analysis for live testing:")
            logger.info("   python tools/btc_processor_cli.py disable-all")
            logger.info("")
            
            logger.info("✅ Enable specific feature:")
            logger.info("   python tools/btc_processor_cli.py enable mining_analysis")
            logger.info("")
            
            logger.info("🚨 Emergency cleanup:")
            logger.info("   python tools/btc_processor_cli.py emergency-cleanup")
            logger.info("")
            
            logger.info("⚙️ Update thresholds:")
            logger.info("   python tools/btc_processor_cli.py set-thresholds --memory-warning 6 --cpu-warning 50")
            logger.info("")
            
            logger.info("📡 Monitor system:")
            logger.info("   python tools/btc_processor_cli.py monitor --duration 30")
            logger.info("")
            
            logger.info("💾 Save configuration:")
            logger.info("   python tools/btc_processor_cli.py save-config --file my_config.json")
            
            logger.info("✅ CLI examples complete")
            
        except Exception as e:
            logger.error(f"❌ CLI examples error: {e}")
            
    async def run_complete_demo(self):
        """Run the complete control demonstration"""
        try:
            logger.info("🚀 Starting Complete BTC Processor Control Demo")
            logger.info("=" * 60)
            
            # Setup
            await self.setup()
            
            # Run all demonstrations
            await self.demonstrate_basic_controls()
            await self.demonstrate_bulk_controls()
            await self.demonstrate_threshold_monitoring()
            await self.demonstrate_emergency_procedures()
            await self.demonstrate_configuration_management()
            await self.demonstrate_live_testing_scenario()
            await self.demonstrate_web_ui_integration()
            await self.show_cli_examples()
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ Complete Control Demo Finished Successfully!")
            logger.info("")
            logger.info("🎯 Key Takeaways:")
            logger.info("   • Features can be toggled individually or in bulk")
            logger.info("   • System automatically monitors resource usage")
            logger.info("   • Emergency procedures protect against overload")
            logger.info("   • Configuration can be saved and restored")
            logger.info("   • Web UI provides real-time control interface")
            logger.info("   • CLI offers quick command-line access")
            logger.info("")
            logger.info("🔧 This system ensures your BTC processor stays within")
            logger.info("   the 10-50 GB information synthesis limit while")
            logger.info("   providing maximum processing efficiency!")
            
        except Exception as e:
            logger.error(f"❌ Demo execution error: {e}")
            
    async def cleanup(self):
        """Cleanup demo resources"""
        try:
            if self.controller and self.controller.is_monitoring:
                await self.controller.stop_monitoring()
                
            # Clean up any demo files
            demo_config = Path("demo_config.json")
            if demo_config.exists():
                demo_config.unlink()
                logger.info("🧹 Cleaned up demo configuration file")
                
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Main demo function"""
    demo = BTCProcessorControlDemo()
    
    try:
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 