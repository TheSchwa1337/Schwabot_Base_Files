#!/usr/bin/env python3
"""
System Monitor CLI - Comprehensive Trading System Monitoring

Provides real-time monitoring and control of the entire Schwabot
trading system across Windows, macOS, and Linux.

Features:
- Real-time system performance monitoring
- CPU/GPU utilization tracking
- Memory and disk usage monitoring
- Trading performance metrics
- Network and API status
- System health diagnostics
- Cross-platform compatibility
"""

import argparse
import asyncio
import json
import sys
import time
import psutil
import platform
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.system.dual_state_router import DualStateRouter
from core.advanced_tensor_algebra import AdvancedTensorAlgebra
from core.unified_profit_vectorization_system import UnifiedProfitVectorizationSystem
from utils.safe_print import safe_print, info, warn, error, success


class SystemMonitorCLI:
    """CLI interface for comprehensive system monitoring."""
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.router = None
        self.tensor_algebra = None
        self.profit_system = None
        self.is_initialized = False
        self.monitoring_data = []
        self.start_time = None
        
    async def initialize_system(self):
        """Initialize the system monitoring components."""
        try:
            info("Initializing System Monitor...")
            
            # Initialize core components
            self.router = DualStateRouter()
            self.tensor_algebra = AdvancedTensorAlgebra()
            self.profit_system = UnifiedProfitVectorizationSystem()
            
            # Test system components
            await self._test_system()
            
            self.is_initialized = True
            self.start_time = time.time()
            success("✅ System Monitor initialized successfully")
            
        except Exception as e:
            error(f"❌ Failed to initialize system monitor: {e}")
            return False
            
        return True
    
    async def _test_system(self):
        """Test the system monitoring components."""
        info("Testing system monitoring components...")
        
        # Test system info
        system_info = self._get_system_info()
        info(f"System: {system_info['system']} {system_info['release']}")
        info(f"Python: {system_info['python_version']}")
        
        # Test monitoring capabilities
        cpu_info = self._get_cpu_info()
        info(f"CPU: {cpu_info['cpu_count']} cores, {cpu_info['cpu_freq']:.1f} GHz")
        
        # Test memory info
        memory_info = self._get_memory_info()
        info(f"Memory: {memory_info['total']:.1f} GB total")
        
        # Test disk info
        disk_info = self._get_disk_info()
        info(f"Disk: {disk_info['total']:.1f} GB total")
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "platform": sys.platform
        }
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        try:
            cpu_freq = psutil.cpu_freq()
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": cpu_freq.current if cpu_freq else 0,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "cpu_load": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except:
            return {"cpu_count": 0, "cpu_freq": 0, "cpu_percent": 0, "cpu_load": [0, 0, 0]}
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total / (1024**3),  # GB
                "available": memory.available / (1024**3),  # GB
                "used": memory.used / (1024**3),  # GB
                "percent": memory.percent
            }
        except:
            return {"total": 0, "available": 0, "used": 0, "percent": 0}
    
    def _get_disk_info(self) -> Dict[str, float]:
        """Get disk information."""
        try:
            disk = psutil.disk_usage('/')
            return {
                "total": disk.total / (1024**3),  # GB
                "used": disk.used / (1024**3),  # GB
                "free": disk.free / (1024**3),  # GB
                "percent": (disk.used / disk.total) * 100
            }
        except:
            return {"total": 0, "used": 0, "free": 0, "percent": 0}
    
    def _get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        try:
            network = psutil.net_io_counters()
            return {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        except:
            return {"bytes_sent": 0, "bytes_recv": 0, "packets_sent": 0, "packets_recv": 0}
    
    async def show_system_status(self):
        """Display comprehensive system status."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info("🖥️  SYSTEM STATUS OVERVIEW")
        info("=" * 40)
        
        # System information
        system_info = self._get_system_info()
        info(f"💻 System: {system_info['system']} {system_info['release']}")
        info(f"🐍 Python: {system_info['python_version']}")
        info(f"🏗️  Architecture: {system_info['machine']}")
        
        # CPU information
        cpu_info = self._get_cpu_info()
        info(f"\n⚡ CPU Information:")
        info(f"  Cores: {cpu_info['cpu_count']}")
        info(f"  Frequency: {cpu_info['cpu_freq']:.1f} GHz")
        info(f"  Usage: {cpu_info['cpu_percent']:.1f}%")
        if cpu_info['cpu_load'][0] > 0:
            info(f"  Load Average: {cpu_info['cpu_load'][0]:.2f}")
        
        # Memory information
        memory_info = self._get_memory_info()
        info(f"\n💾 Memory Information:")
        info(f"  Total: {memory_info['total']:.1f} GB")
        info(f"  Used: {memory_info['used']:.1f} GB ({memory_info['percent']:.1f}%)")
        info(f"  Available: {memory_info['available']:.1f} GB")
        
        # Disk information
        disk_info = self._get_disk_info()
        info(f"\n💿 Disk Information:")
        info(f"  Total: {disk_info['total']:.1f} GB")
        info(f"  Used: {disk_info['used']:.1f} GB ({disk_info['percent']:.1f}%)")
        info(f"  Free: {disk_info['free']:.1f} GB")
        
        # Network information
        network_info = self._get_network_info()
        info(f"\n🌐 Network Information:")
        info(f"  Bytes Sent: {network_info['bytes_sent'] / (1024**2):.1f} MB")
        info(f"  Bytes Received: {network_info['bytes_recv'] / (1024**2):.1f} MB")
        info(f"  Packets Sent: {network_info['packets_sent']:,}")
        info(f"  Packets Received: {network_info['packets_recv']:,}")
        
        # Trading system status
        if self.router:
            router_stats = self.router.get_statistics()
            info(f"\n🔄 Trading System Status:")
            info(f"  Total Tasks: {router_stats.get('total_tasks', 0)}")
            info(f"  CPU Tasks: {router_stats.get('cpu_tasks', 0)}")
            info(f"  GPU Tasks: {router_stats.get('gpu_tasks', 0)}")
            info(f"  Average Response Time: {router_stats.get('avg_response_time', 0):.3f}s")
    
    async def show_performance_metrics(self):
        """Display detailed performance metrics."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info("📊 DETAILED PERFORMANCE METRICS")
        info("=" * 40)
        
        # System performance
        cpu_info = self._get_cpu_info()
        memory_info = self._get_memory_info()
        disk_info = self._get_disk_info()
        
        info(f"⚡ System Performance:")
        info(f"  CPU Utilization: {cpu_info['cpu_percent']:.1f}%")
        info(f"  Memory Utilization: {memory_info['percent']:.1f}%")
        info(f"  Disk Utilization: {disk_info['percent']:.1f}%")
        
        # Trading system performance
        if self.router:
            router_perf = self.router.get_performance_metrics()
            info(f"\n🔄 Trading System Performance:")
            info(f"  CPU Utilization: {router_perf.get('cpu_utilization', 0):.1f}%")
            info(f"  GPU Utilization: {router_perf.get('gpu_utilization', 0):.1f}%")
            info(f"  Memory Usage: {router_perf.get('memory_usage', 0):.1f}MB")
            info(f"  Response Time: {router_perf.get('avg_response_time', 0):.3f}s")
        
        # Tensor system performance
        if self.tensor_algebra:
            tensor_perf = self.tensor_algebra.get_performance_metrics()
            info(f"\n🧮 Tensor System Performance:")
            info(f"  Total Operations: {tensor_perf.get('total_operations', 0)}")
            info(f"  Average Operation Time: {tensor_perf.get('avg_operation_time', 0):.3f}s")
            info(f"  Cache Hit Rate: {tensor_perf.get('cache_hit_rate', 0):.1f}%")
            info(f"  Memory Efficiency: {tensor_perf.get('memory_efficiency', 0):.1f}%")
        
        # Profit system performance
        if self.profit_system:
            profit_perf = self.profit_system.get_performance_summary()
            info(f"\n💰 Profit System Performance:")
            info(f"  Total Calculations: {profit_perf.get('total_calculations', 0)}")
            info(f"  Average Calculation Time: {profit_perf.get('average_calculation_time', 0):.3f}s")
            info(f"  Average Profit: {profit_perf.get('average_profit', 0):.6f}")
            info(f"  Profit Standard Deviation: {profit_perf.get('profit_std', 0):.6f}")
    
    async def show_trading_metrics(self):
        """Display trading-specific metrics."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info("📈 TRADING METRICS")
        info("=" * 25)
        
        # Router trading metrics
        if self.router:
            router_stats = self.router.get_statistics()
            registry_stats = self.router.get_profit_registry_stats()
            
            info(f"🔄 Router Metrics:")
            info(f"  Total Tasks Processed: {router_stats.get('total_tasks', 0)}")
            info(f"  CPU Task Ratio: {router_stats.get('cpu_tasks', 0) / max(router_stats.get('total_tasks', 1), 1) * 100:.1f}%")
            info(f"  GPU Task Ratio: {router_stats.get('gpu_tasks', 0) / max(router_stats.get('total_tasks', 1), 1) * 100:.1f}%")
            info(f"  Average Response Time: {router_stats.get('avg_response_time', 0):.3f}s")
            
            info(f"\n💰 Profit Registry Metrics:")
            info(f"  Total Strategies: {registry_stats.get('total_strategies', 0)}")
            info(f"  Short-term Strategies: {registry_stats.get('short_term', 0)}")
            info(f"  Mid-term Strategies: {registry_stats.get('mid_term', 0)}")
            info(f"  Long-term Strategies: {registry_stats.get('long_term', 0)}")
        
        # Strategy performance
        if self.router:
            all_performance = self.router.get_all_strategy_performance()
            if all_performance:
                info(f"\n📊 Strategy Performance:")
                for strategy, metrics in list(all_performance.items())[:5]:  # Top 5
                    info(f"  {strategy}:")
                    info(f"    Executions: {metrics.get('total_executions', 0)}")
                    info(f"    Success Rate: {metrics.get('success_rate', 0):.1f}%")
                    info(f"    Avg Profit: {metrics.get('avg_profit', 0):.3f}")
    
    async def show_health_diagnostics(self):
        """Display system health diagnostics."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info("🏥 SYSTEM HEALTH DIAGNOSTICS")
        info("=" * 35)
        
        # System health checks
        health_status = {}
        
        # CPU health
        cpu_info = self._get_cpu_info()
        if cpu_info['cpu_percent'] > 90:
            health_status['cpu'] = "CRITICAL"
        elif cpu_info['cpu_percent'] > 70:
            health_status['cpu'] = "WARNING"
        else:
            health_status['cpu'] = "HEALTHY"
        
        # Memory health
        memory_info = self._get_memory_info()
        if memory_info['percent'] > 90:
            health_status['memory'] = "CRITICAL"
        elif memory_info['percent'] > 80:
            health_status['memory'] = "WARNING"
        else:
            health_status['memory'] = "HEALTHY"
        
        # Disk health
        disk_info = self._get_disk_info()
        if disk_info['percent'] > 90:
            health_status['disk'] = "CRITICAL"
        elif disk_info['percent'] > 80:
            health_status['disk'] = "WARNING"
        else:
            health_status['disk'] = "HEALTHY"
        
        # Display health status
        info(f"⚡ CPU: {health_status['cpu']} ({cpu_info['cpu_percent']:.1f}%)")
        info(f"💾 Memory: {health_status['memory']} ({memory_info['percent']:.1f}%)")
        info(f"💿 Disk: {health_status['disk']} ({disk_info['percent']:.1f}%)")
        
        # Overall health
        critical_count = sum(1 for status in health_status.values() if status == "CRITICAL")
        warning_count = sum(1 for status in health_status.values() if status == "WARNING")
        
        if critical_count > 0:
            overall_health = "CRITICAL"
        elif warning_count > 0:
            overall_health = "WARNING"
        else:
            overall_health = "HEALTHY"
        
        info(f"\n🎯 Overall System Health: {overall_health}")
        
        # Recommendations
        if overall_health != "HEALTHY":
            info(f"\n💡 Recommendations:")
            if health_status['cpu'] != "HEALTHY":
                info(f"  - Consider reducing CPU load or upgrading hardware")
            if health_status['memory'] != "HEALTHY":
                info(f"  - Consider clearing caches or adding more RAM")
            if health_status['disk'] != "HEALTHY":
                info(f"  - Consider cleaning up disk space")
    
    async def start_real_time_monitoring(self, interval: int = 5):
        """Start real-time monitoring with specified interval."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info(f"📡 STARTING REAL-TIME MONITORING (Interval: {interval}s)")
        info("=" * 50)
        info("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Clear screen (platform-specific)
                if platform.system() == "Windows":
                    os.system('cls')
                else:
                    os.system('clear')
                
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                info(f"🕐 {timestamp}")
                info("=" * 50)
                
                # System metrics
                cpu_info = self._get_cpu_info()
                memory_info = self._get_memory_info()
                disk_info = self._get_disk_info()
                
                info(f"⚡ CPU: {cpu_info['cpu_percent']:.1f}% | 💾 Memory: {memory_info['percent']:.1f}% | 💿 Disk: {disk_info['percent']:.1f}%")
                
                # Trading system metrics
                if self.router:
                    router_stats = self.router.get_statistics()
                    info(f"🔄 Tasks: {router_stats.get('total_tasks', 0)} | CPU: {router_stats.get('cpu_tasks', 0)} | GPU: {router_stats.get('gpu_tasks', 0)}")
                
                # Network metrics
                network_info = self._get_network_info()
                info(f"🌐 Network: ↑{network_info['bytes_sent'] / (1024**2):.1f}MB ↓{network_info['bytes_recv'] / (1024**2):.1f}MB")
                
                # Wait for next update
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            info("\n🛑 Real-time monitoring stopped")
    
    async def export_system_report(self, file_path: str):
        """Export comprehensive system report to file."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info(f"📤 EXPORTING SYSTEM REPORT")
        info("=" * 30)
        
        try:
            # Collect all system data
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "system_info": self._get_system_info(),
                "cpu_info": self._get_cpu_info(),
                "memory_info": self._get_memory_info(),
                "disk_info": self._get_disk_info(),
                "network_info": self._get_network_info(),
                "uptime": time.time() - self.start_time if self.start_time else 0
            }
            
            # Add trading system data
            if self.router:
                report_data["trading_system"] = {
                    "router_stats": self.router.get_statistics(),
                    "registry_stats": self.router.get_profit_registry_stats(),
                    "performance_metrics": self.router.get_performance_metrics()
                }
            
            # Add tensor system data
            if self.tensor_algebra:
                report_data["tensor_system"] = {
                    "performance_metrics": self.tensor_algebra.get_performance_metrics(),
                    "statistics": self.tensor_algebra.get_statistics()
                }
            
            # Add profit system data
            if self.profit_system:
                report_data["profit_system"] = self.profit_system.get_performance_summary()
            
            # Export to file
            file_path = Path(file_path)
            
            if file_path.suffix == '.json':
                with open(file_path, 'w') as f:
                    json.dump(report_data, f, indent=2)
                info(f"Saved as JSON: {file_path}")
            elif file_path.suffix == '.txt':
                with open(file_path, 'w') as f:
                    f.write("SCHWABOT SYSTEM REPORT\n")
                    f.write("=" * 30 + "\n\n")
                    f.write(f"Generated: {report_data['timestamp']}\n")
                    f.write(f"Uptime: {report_data['uptime']:.1f} seconds\n\n")
                    
                    f.write("SYSTEM INFORMATION:\n")
                    for key, value in report_data['system_info'].items():
                        f.write(f"  {key}: {value}\n")
                    
                    f.write(f"\nCPU: {report_data['cpu_info']['cpu_percent']:.1f}%\n")
                    f.write(f"Memory: {report_data['memory_info']['percent']:.1f}%\n")
                    f.write(f"Disk: {report_data['disk_info']['percent']:.1f}%\n")
                
                info(f"Saved as text: {file_path}")
            else:
                error(f"Unsupported file format: {file_path.suffix}")
                return
            
            success(f"✅ System report exported successfully to {file_path}")
            
        except Exception as e:
            error(f"❌ Export failed: {e}")
    
    async def run_interactive_mode(self):
        """Run interactive CLI mode."""
        info("🎮 INTERACTIVE SYSTEM MONITOR CLI")
        info("=" * 40)
        info("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\n📊 monitor> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    info("👋 Goodbye!")
                    break
                elif command == 'help':
                    self._show_help()
                elif command == 'status':
                    await self.show_system_status()
                elif command == 'performance':
                    await self.show_performance_metrics()
                elif command == 'trading':
                    await self.show_trading_metrics()
                elif command == 'health':
                    await self.show_health_diagnostics()
                elif command.startswith('monitor '):
                    parts = command.split()
                    interval = int(parts[1]) if len(parts) > 1 else 5
                    await self.start_real_time_monitoring(interval)
                elif command.startswith('export '):
                    file_path = command.split(' ', 1)[1]
                    await self.export_system_report(file_path)
                else:
                    warn(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                info("\n👋 Goodbye!")
                break
            except Exception as e:
                error(f"Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        info("📖 AVAILABLE COMMANDS:")
        info("  status                    - Show system status")
        info("  performance               - Show performance metrics")
        info("  trading                   - Show trading metrics")
        info("  health                    - Show health diagnostics")
        info("  monitor [interval]        - Start real-time monitoring")
        info("  export <file_path>        - Export system report")
        info("  quit/exit                 - Exit CLI")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="System Monitor CLI - Comprehensive Trading System Monitoring")
    parser.add_argument("--init", action="store_true", help="Initialize the system")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--performance", action="store_true", help="Show performance metrics")
    parser.add_argument("--trading", action="store_true", help="Show trading metrics")
    parser.add_argument("--health", action="store_true", help="Show health diagnostics")
    parser.add_argument("--monitor", type=int, metavar="INTERVAL", help="Start real-time monitoring")
    parser.add_argument("--export", metavar="FILE", help="Export system report")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    
    args = parser.parse_args()
    
    cli = SystemMonitorCLI()
    
    # Initialize if requested or if any command needs it
    if args.init or any([args.status, args.performance, args.trading, args.health, args.monitor, args.export, args.interactive]):
        if not await cli.initialize_system():
            return 1
    
    # Execute commands
    if args.status:
        await cli.show_system_status()
    elif args.performance:
        await cli.show_performance_metrics()
    elif args.trading:
        await cli.show_trading_metrics()
    elif args.health:
        await cli.show_health_diagnostics()
    elif args.monitor is not None:
        await cli.start_real_time_monitoring(args.monitor)
    elif args.export:
        await cli.export_system_report(args.export)
    elif args.interactive:
        await cli.run_interactive_mode()
    elif not args.init:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 