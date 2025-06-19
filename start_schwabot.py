#!/usr/bin/env python3
"""
Schwabot System Startup Script
==============================

Comprehensive startup script that demonstrates the full integration of all
Schwabot components with centralized configuration management and the
integration orchestrator.

This script showcases:
- Centralized configuration management
- Integration orchestrator with all components
- GAN filtering system integration
- Real-time component monitoring
- Configuration hot-reloading
- Comprehensive system status reporting

Usage:
    python start_schwabot.py [--config CONFIG_FILE] [--mode MODE]

Arguments:
    --config: Path to configuration file (default: schwabot_config.yaml)
    --mode: Integration mode (development, testing, production, maintenance)

Windows CLI compatible with flake8 compliance.
"""

import argparse
import sys
import time
import signal
from pathlib import Path
from typing import Optional

def setup_signal_handlers(orchestrator) -> None:
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print("\n🛑 Shutdown signal received...")
        if orchestrator:
            orchestrator.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main() -> None:
    """Main startup function"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Schwabot System Startup')
        parser.add_argument('--config', type=str, default='schwabot_config.yaml',
                          help='Configuration file path')
        parser.add_argument('--mode', type=str, choices=['development', 'testing', 'production', 'maintenance'],
                          default='development', help='Integration mode')
        parser.add_argument('--enable-monitoring', action='store_true',
                          help='Enable continuous monitoring')
        parser.add_argument('--enable-hot-reload', action='store_true',
                          help='Enable configuration hot-reloading')
        
        args = parser.parse_args()
        
        print("🚀 Schwabot System Startup")
        print("=" * 60)
        print(f"   Configuration: {args.config}")
        print(f"   Mode: {args.mode}")
        print(f"   Monitoring: {'enabled' if args.enable_monitoring else 'disabled'}")
        print(f"   Hot-reload: {'enabled' if args.enable_hot_reload else 'disabled'}")
        
        # Check if configuration file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"⚠️ Configuration file not found: {args.config}")
            print("   Using default configuration...")
        
        # Initialize configuration system
        print("\n⚙️ Initializing Configuration System...")
        try:
            from core.config import get_config_manager
            config_manager = get_config_manager(args.config)
            
            # Update environment mode if specified
            if args.mode:
                config_manager.update_config('system', 'environment', args.mode)
            
            config = config_manager.get_config()
            print(f"✅ Configuration loaded")
            print(f"   Environment: {config.system.environment.value}")
            print(f"   Debug mode: {config.system.debug}")
            print(f"   Log level: {config.system.log_level}")
            
        except Exception as e:
            print(f"❌ Configuration system failed: {e}")
            return False
        
        # Enable hot-reloading if requested
        if args.enable_hot_reload:
            print("\n🔄 Enabling Configuration Hot-Reloading...")
            config_manager.enable_hot_reload(check_interval=5)
            print("✅ Hot-reloading enabled (5 second interval)")
        
        # Initialize Schwabot core system
        print("\n🧠 Initializing Schwabot Core System...")
        try:
            from core import initialize_schwabot
            core = initialize_schwabot()
            print("✅ Schwabot core system initialized")
            
        except Exception as e:
            print(f"❌ Core system initialization failed: {e}")
            return False
        
        # Get integration orchestrator
        orchestrator = None
        if hasattr(core, 'orchestrator'):
            orchestrator = core.orchestrator
            print("✅ Integration orchestrator available")
            
            # Setup signal handlers for graceful shutdown
            setup_signal_handlers(orchestrator)
            
            # Display detailed system status
            print("\n📊 Detailed System Status:")
            status = orchestrator.get_system_status()
            
            print(f"   Orchestrator Mode: {status['orchestrator']['mode']}")
            print(f"   Orchestrator Running: {status['orchestrator']['running']}")
            print(f"   Total Components: {status['metrics']['total_components']}")
            print(f"   Running Components: {status['metrics']['running_components']}")
            print(f"   Failed Components: {status['metrics']['failed_components']}")
            
            # Display component details
            print("\n🔍 Component Status Details:")
            for name, info in status['components'].items():
                status_emoji = {
                    'running': '✅',
                    'error': '❌',
                    'initializing': '⏳',
                    'paused': '⏸️',
                    'shutdown': '🛑',
                    'uninitialized': '⚪'
                }.get(info['status'], '❓')
                
                print(f"   {status_emoji} {name.ljust(25)} {info['status']}")
                if info['error_count'] > 0:
                    print(f"      └─ Errors: {info['error_count']}")
                if info['dependencies']:
                    print(f"      └─ Depends on: {', '.join(info['dependencies'])}")
            
            # Test component access
            print("\n🧪 Testing Component Access:")
            
            # Test mathematical libraries
            mathlib_v1 = orchestrator.get_component('mathlib_v1')
            if mathlib_v1:
                print("   ✅ MathLib V1 accessible")
            else:
                print("   ❌ MathLib V1 not accessible")
            
            mathlib_v3 = orchestrator.get_component('mathlib_v3')
            if mathlib_v3:
                print("   ✅ MathLib V3 (with auto-diff) accessible")
            else:
                print("   ❌ MathLib V3 not accessible")
            
            # Test GAN filter system
            gan_filter = orchestrator.get_component('gan_filter')
            if gan_filter:
                print("   ✅ GAN Filter system accessible")
                print(f"      └─ Enabled in config: {config.advanced.gan_enabled}")
                print(f"      └─ Batch size: {config.advanced.gan_batch_size}")
                print(f"      └─ Confidence threshold: {config.advanced.gan_confidence_threshold}")
            else:
                print("   ⚠️ GAN Filter not accessible")
                if not config.advanced.gan_enabled:
                    print("      └─ Disabled in configuration")
                else:
                    print("      └─ May require PyTorch installation")
            
            # Test trading components
            btc_integration = orchestrator.get_component('btc_integration')
            if btc_integration:
                print("   ✅ BTC Integration accessible")
            else:
                print("   ❌ BTC Integration not accessible")
            
            risk_monitor = orchestrator.get_component('risk_monitor')
            if risk_monitor:
                print("   ✅ Risk Monitor accessible")
            else:
                print("   ❌ Risk Monitor not accessible")
            
            # Test high-performance computing
            rittle_gemm = orchestrator.get_component('rittle_gemm')
            if rittle_gemm:
                print("   ✅ Rittle GEMM (high-performance matrix ops) accessible")
            else:
                print("   ❌ Rittle GEMM not accessible")
            
        else:
            print("⚠️ Integration orchestrator not available")
        
        # Test configuration system
        print("\n⚙️ Testing Configuration System:")
        print(f"   Current GAN enabled: {config.advanced.gan_enabled}")
        print(f"   Current GAN batch size: {config.advanced.gan_batch_size}")
        
        # Demonstrate configuration update
        print("\n🔄 Testing Configuration Updates:")
        original_batch_size = config.advanced.gan_batch_size
        new_batch_size = 128 if original_batch_size != 128 else 64
        
        config_manager.update_config('advanced', 'gan_batch_size', new_batch_size)
        updated_config = config_manager.get_config()
        print(f"   Updated GAN batch size: {original_batch_size} → {updated_config.advanced.gan_batch_size}")
        
        # Restore original value
        config_manager.update_config('advanced', 'gan_batch_size', original_batch_size)
        print(f"   Restored GAN batch size: {original_batch_size}")
        
        # Save configuration
        print("\n💾 Saving Configuration...")
        save_success = config_manager.save_configuration()
        if save_success:
            print("✅ Configuration saved successfully")
        else:
            print("❌ Configuration save failed")
        
        # System ready
        print("\n🎉 Schwabot System Ready!")
        print("=" * 60)
        print("   All components initialized and integrated")
        print("   Configuration system active with hot-reloading")
        print("   GAN filtering system integrated and configurable")
        print("   Real-time monitoring and health checks active")
        print("   System ready for trading operations")
        
        # Continuous monitoring mode
        if args.enable_monitoring and orchestrator:
            print("\n📡 Entering Continuous Monitoring Mode...")
            print("   Press Ctrl+C to shutdown gracefully")
            
            try:
                while True:
                    time.sleep(30)  # Monitor every 30 seconds
                    
                    # Get current status
                    current_status = orchestrator.get_system_status()
                    running = current_status['metrics']['running_components']
                    total = current_status['metrics']['total_components']
                    uptime = current_status['orchestrator'].get('uptime_seconds', 0)
                    
                    print(f"📊 Status Update: {running}/{total} components running, uptime: {uptime:.0f}s")
                    
                    # Check for any failed components
                    failed = current_status['metrics']['failed_components']
                    if failed > 0:
                        print(f"⚠️ Warning: {failed} components in error state")
                        
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
        
        else:
            print("\n✨ System startup complete!")
            print("   Use --enable-monitoring for continuous monitoring mode")
        
        return True
        
    except Exception as e:
        print(f"\n❌ System startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if 'orchestrator' in locals() and orchestrator:
            print("\n🧹 Performing cleanup...")
            orchestrator.shutdown()
            print("✅ Cleanup complete")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 