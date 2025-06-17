#!/usr/bin/env python3
"""
BTC Processor CLI
=================

Command-line interface for controlling BTC processor features and managing system resources.
Provides quick access to feature toggles and system monitoring.
"""

import argparse
import asyncio
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.btc_processor_controller import BTCProcessorController
from core.btc_data_processor import BTCDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BTCProcessorCLI:
    """Command-line interface for BTC processor control"""
    
    def __init__(self):
        self.controller = BTCProcessorController()
        self.processor = None
        
    async def setup_processor(self, config_path: str = None):
        """Setup the BTC processor"""
        try:
            if config_path:
                self.processor = BTCDataProcessor(config_path)
            else:
                self.processor = BTCDataProcessor()
            self.controller.processor = self.processor
            logger.info("✅ BTC processor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize processor: {e}")
            
    async def show_status(self):
        """Show current system status"""
        try:
            status = self.controller.get_current_status()
            
            print("\n🚀 BTC Processor Status")
            print("=" * 50)
            
            # System metrics
            metrics = status.get('system_metrics', {})
            print(f"\n📊 System Metrics:")
            print(f"  CPU Usage:    {metrics.get('cpu_usage', 0):.1f}%")
            print(f"  Memory Usage: {metrics.get('memory_usage_gb', 0):.1f} GB")
            print(f"  GPU Usage:    {metrics.get('gpu_usage', 0):.1f}%")
            print(f"  Disk Usage:   {metrics.get('disk_usage_gb', 0):.1f} GB")
            
            # Monitoring status
            monitoring = "🟢 Active" if status.get('monitoring_active', False) else "🔴 Inactive"
            emergency = "🚨 EMERGENCY" if status.get('emergency_shutdown', False) else "✅ Normal"
            print(f"\n📡 Monitoring:  {monitoring}")
            print(f"🚨 Status:      {emergency}")
            
            # Feature states
            features = status.get('feature_states', {})
            print(f"\n⚙️ Feature States:")
            for feature, enabled in features.items():
                state = "🟢 ON " if enabled else "🔴 OFF"
                print(f"  {feature:<20} {state}")
                
            # Configuration
            config = status.get('configuration', {})
            print(f"\n🔧 Configuration:")
            print(f"  Max Memory:   {config.get('max_memory_gb', 0)} GB")
            print(f"  Max CPU:      {config.get('max_cpu_percent', 0)}%")
            print(f"  Max GPU:      {config.get('max_gpu_percent', 0)}%")
            
            # Thresholds
            thresholds = status.get('thresholds', {})
            print(f"\n⚠️ Thresholds:")
            print(f"  Memory Warn:  {thresholds.get('memory_warning', 0)} GB")
            print(f"  Memory Crit:  {thresholds.get('memory_critical', 0)} GB")
            print(f"  CPU Warn:     {thresholds.get('cpu_warning', 0)}%")
            print(f"  CPU Crit:     {thresholds.get('cpu_critical', 0)}%")
            
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            
    async def enable_feature(self, feature_name: str):
        """Enable a specific feature"""
        try:
            await self.controller.enable_feature(feature_name)
            print(f"✅ Enabled {feature_name}")
        except Exception as e:
            logger.error(f"❌ Error enabling {feature_name}: {e}")
            
    async def disable_feature(self, feature_name: str):
        """Disable a specific feature"""
        try:
            await self.controller.disable_feature(feature_name)
            print(f"⏹️ Disabled {feature_name}")
        except Exception as e:
            logger.error(f"❌ Error disabling {feature_name}: {e}")
            
    async def enable_all_analysis(self):
        """Enable all analysis features"""
        try:
            await self.controller.enable_all_analysis_features()
            print("✅ Enabled all analysis features")
        except Exception as e:
            logger.error(f"❌ Error enabling all analysis: {e}")
            
    async def disable_all_analysis(self):
        """Disable all analysis features"""
        try:
            await self.controller.disable_all_analysis_features()
            print("⏹️ Disabled all analysis features")
        except Exception as e:
            logger.error(f"❌ Error disabling all analysis: {e}")
            
    async def emergency_cleanup(self):
        """Perform emergency cleanup"""
        try:
            await self.controller._emergency_memory_cleanup()
            print("🚨 Emergency cleanup completed")
        except Exception as e:
            logger.error(f"❌ Emergency cleanup error: {e}")
            
    async def start_monitoring(self):
        """Start system monitoring"""
        try:
            # Start monitoring in background
            import threading
            def run_monitoring():
                asyncio.run(self.controller.start_monitoring())
            
            thread = threading.Thread(target=run_monitoring)
            thread.daemon = True
            thread.start()
            
            print("▶️ System monitoring started")
        except Exception as e:
            logger.error(f"❌ Error starting monitoring: {e}")
            
    async def stop_monitoring(self):
        """Stop system monitoring"""
        try:
            await self.controller.stop_monitoring()
            print("⏹️ System monitoring stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring: {e}")
            
    def list_features(self):
        """List available features"""
        features = [
            'mining_analysis',
            'block_timing',
            'nonce_sequences', 
            'difficulty_tracking',
            'memory_management',
            'hash_generation',
            'load_balancing',
            'storage',
            'monitoring'
        ]
        
        print("\n⚙️ Available Features:")
        print("=" * 30)
        for i, feature in enumerate(features, 1):
            print(f"{i:2d}. {feature}")
            
    async def update_thresholds(self, memory_warning: float = None, memory_critical: float = None,
                              cpu_warning: float = None, cpu_critical: float = None,
                              gpu_warning: float = None, gpu_critical: float = None):
        """Update system thresholds"""
        try:
            thresholds = {}
            if memory_warning is not None:
                thresholds['memory_warning'] = memory_warning
            if memory_critical is not None:
                thresholds['memory_critical'] = memory_critical
            if cpu_warning is not None:
                thresholds['cpu_warning'] = cpu_warning
            if cpu_critical is not None:
                thresholds['cpu_critical'] = cpu_critical
            if gpu_warning is not None:
                thresholds['gpu_warning'] = gpu_warning
            if gpu_critical is not None:
                thresholds['gpu_critical'] = gpu_critical
                
            if thresholds:
                self.controller.update_thresholds(thresholds)
                print("✅ Thresholds updated")
                
                # Show updated thresholds
                for key, value in thresholds.items():
                    print(f"  {key}: {value}")
            else:
                print("⚠️ No thresholds specified")
                
        except Exception as e:
            logger.error(f"❌ Error updating thresholds: {e}")
            
    async def update_config(self, max_memory: float = None, max_cpu: float = None, max_gpu: float = None):
        """Update processor configuration"""
        try:
            config = {}
            if max_memory is not None:
                config['max_memory_usage_gb'] = max_memory
            if max_cpu is not None:
                config['max_cpu_usage_percent'] = max_cpu
            if max_gpu is not None:
                config['max_gpu_usage_percent'] = max_gpu
                
            if config:
                self.controller.update_configuration(config)
                print("✅ Configuration updated")
                
                # Show updated config
                for key, value in config.items():
                    print(f"  {key}: {value}")
            else:
                print("⚠️ No configuration specified")
                
        except Exception as e:
            logger.error(f"❌ Error updating configuration: {e}")
            
    def save_config(self, file_path: str = None):
        """Save current configuration"""
        try:
            if file_path:
                self.controller.save_configuration(file_path)
            else:
                self.controller.save_configuration()
            print("💾 Configuration saved")
        except Exception as e:
            logger.error(f"❌ Error saving configuration: {e}")
            
    def load_config(self, file_path: str = None):
        """Load configuration from file"""
        try:
            if file_path:
                self.controller.load_configuration(file_path)
            else:
                self.controller.load_configuration()
            print("📁 Configuration loaded")
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            
    async def monitor_system(self, duration: int = 60):
        """Monitor system for specified duration"""
        try:
            print(f"📊 Monitoring system for {duration} seconds...")
            print("Press Ctrl+C to stop early")
            
            await self.start_monitoring()
            
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    status = self.controller.get_current_status()
                    metrics = status.get('system_metrics', {})
                    
                    # Clear line and show metrics
                    print(f"\r🖥️ CPU: {metrics.get('cpu_usage', 0):5.1f}% | "
                          f"💾 Memory: {metrics.get('memory_usage_gb', 0):5.1f}GB | "
                          f"🎮 GPU: {metrics.get('gpu_usage', 0):5.1f}% | "
                          f"Time: {int(time.time() - start_time):3d}s", end='', flush=True)
                    
                    await asyncio.sleep(2)
                    
                except KeyboardInterrupt:
                    break
                    
            print("\n⏹️ Monitoring stopped")
            await self.stop_monitoring()
            
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='BTC Processor Control CLI')
    parser.add_argument('--config', '-c', help='Path to processor config file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show current status')
    
    # Feature control commands
    enable_parser = subparsers.add_parser('enable', help='Enable a feature')
    enable_parser.add_argument('feature', help='Feature name to enable')
    
    disable_parser = subparsers.add_parser('disable', help='Disable a feature')
    disable_parser.add_argument('feature', help='Feature name to disable')
    
    subparsers.add_parser('enable-all', help='Enable all analysis features')
    subparsers.add_parser('disable-all', help='Disable all analysis features')
    subparsers.add_parser('list-features', help='List available features')
    
    # System control commands
    subparsers.add_parser('start-monitoring', help='Start system monitoring')
    subparsers.add_parser('stop-monitoring', help='Stop system monitoring')
    subparsers.add_parser('emergency-cleanup', help='Perform emergency cleanup')
    
    # Configuration commands
    thresholds_parser = subparsers.add_parser('set-thresholds', help='Update system thresholds')
    thresholds_parser.add_argument('--memory-warning', type=float, help='Memory warning threshold (GB)')
    thresholds_parser.add_argument('--memory-critical', type=float, help='Memory critical threshold (GB)')
    thresholds_parser.add_argument('--cpu-warning', type=float, help='CPU warning threshold (%)')
    thresholds_parser.add_argument('--cpu-critical', type=float, help='CPU critical threshold (%)')
    thresholds_parser.add_argument('--gpu-warning', type=float, help='GPU warning threshold (%)')
    thresholds_parser.add_argument('--gpu-critical', type=float, help='GPU critical threshold (%)')
    
    config_parser = subparsers.add_parser('set-config', help='Update processor configuration')
    config_parser.add_argument('--max-memory', type=float, help='Maximum memory usage (GB)')
    config_parser.add_argument('--max-cpu', type=float, help='Maximum CPU usage (%)')
    config_parser.add_argument('--max-gpu', type=float, help='Maximum GPU usage (%)')
    
    save_parser = subparsers.add_parser('save-config', help='Save configuration to file')
    save_parser.add_argument('--file', help='Configuration file path')
    
    load_parser = subparsers.add_parser('load-config', help='Load configuration from file')
    load_parser.add_argument('--file', help='Configuration file path')
    
    # Monitoring command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor system for specified duration')
    monitor_parser.add_argument('--duration', '-d', type=int, default=60, help='Monitoring duration in seconds')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Create CLI instance
    cli = BTCProcessorCLI()
    
    async def run_command():
        """Run the specified command"""
        try:
            if args.command in ['status', 'enable', 'disable', 'enable-all', 'disable-all', 
                              'start-monitoring', 'stop-monitoring', 'emergency-cleanup', 
                              'set-thresholds', 'set-config', 'monitor']:
                await cli.setup_processor(args.config)
                
            if args.command == 'status':
                await cli.show_status()
                
            elif args.command == 'enable':
                await cli.enable_feature(args.feature)
                
            elif args.command == 'disable':
                await cli.disable_feature(args.feature)
                
            elif args.command == 'enable-all':
                await cli.enable_all_analysis()
                
            elif args.command == 'disable-all':
                await cli.disable_all_analysis()
                
            elif args.command == 'list-features':
                cli.list_features()
                
            elif args.command == 'start-monitoring':
                await cli.start_monitoring()
                
            elif args.command == 'stop-monitoring':
                await cli.stop_monitoring()
                
            elif args.command == 'emergency-cleanup':
                await cli.emergency_cleanup()
                
            elif args.command == 'set-thresholds':
                await cli.update_thresholds(
                    args.memory_warning, args.memory_critical,
                    args.cpu_warning, args.cpu_critical,
                    args.gpu_warning, args.gpu_critical
                )
                
            elif args.command == 'set-config':
                await cli.update_config(args.max_memory, args.max_cpu, args.max_gpu)
                
            elif args.command == 'save-config':
                cli.save_config(args.file)
                
            elif args.command == 'load-config':
                cli.load_config(args.file)
                
            elif args.command == 'monitor':
                await cli.monitor_system(args.duration)
                
        except KeyboardInterrupt:
            print("\n⏹️ Operation cancelled by user")
        except Exception as e:
            logger.error(f"❌ Command failed: {e}")
            
    # Run the command
    asyncio.run(run_command())

if __name__ == '__main__':
    main() 