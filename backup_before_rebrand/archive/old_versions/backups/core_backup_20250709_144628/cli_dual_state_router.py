"""Module for Schwabot trading system."""


import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.cpu_handlers import CPUHandlers
from core.gpu_handlers import GPUHandlers
from core.system.dual_state_router import DualStateRouter

#!/usr/bin/env python3
"""
Dual State Router CLI - CPU/GPU Orchestration Control

Provides command-line interface for controlling the profit-tiered
dualistic compute orchestration system (ZPE/ZBE, routing).

    Features:
    - Monitor CPU/GPU routing decisions
    - View profit registry and strategy metadata
    - Control routing parameters
    - Real-time performance monitoring
    - Strategy tier management
    """

    # Add project root to path
    sys.path.append(str(Path(__file__).parent.parent))


        class DualStateRouterCLI:
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """CLI interface for dual-state router operations."""

            def __init__(self) -> None:
            """Initialize the CLI interface."""
            self.router = None
            self.cpu_handlers = None
            self.gpu_handlers = None
            self.is_initialized = False

                async def initialize_system(self):
                """Initialize the dual-state router system."""
                    try:
                    info("Initializing Dual State Router System...")

                    # Initialize handlers
                    self.cpu_handlers = CPUHandlers()
                    self.gpu_handlers = GPUHandlers()

                    # Initialize router
                    self.router = DualStateRouter()

                    # Test system
                    await self._test_system()

                    self.is_initialized = True
                    success("✅ Dual State Router System initialized successfully")

                        except Exception as e:
                        error("❌ Failed to initialize system: {0}".format(e))
                    return False

                return True

                    async def _test_system(self):
                    """Test the dual-state router system."""
                    info("Testing system components...")

                    # Test CPU handlers
                    test_data = {"price": 50000.0, "volume": 1000.0, "volatility": 0.15}
                    cpu_result = await self.cpu_handlers.process_market_data(test_data)
                    info("CPU test result: {0}".format(cpu_result.get('status', 'unknown')))

                    # Test GPU handlers (with, fallback)
                    gpu_result = await self.gpu_handlers.process_market_data(test_data)
                    info("GPU test result: {0}".format(gpu_result.get('status', 'unknown')))

                    # Test router
                    router_result = await self.router.route_task(
                    task_type="market_analysis",
                    strategy_metadata={
                    "strategy_tier": "short",
                    "profit_density": 0.75,
                    "historical_performance": 0.8,
                    },
                    )
                    info("Router test result: {0}".format(router_result.get('compute_mode', 'unknown')))

                        async def show_system_status(self):
                        """Display comprehensive system status."""
                            if not self.is_initialized:
                            error("System not initialized. Run 'init' first.")
                        return

                        info("🔍 DUAL STATE ROUTER SYSTEM STATUS")
                        info("=" * 50)

                        # Router status
                        router_stats = self.router.get_statistics()
                        info(f"📊 Router Statistics:")
                        info("  Total Tasks: {0}".format(router_stats.get('total_tasks', 0)))
                        info("  CPU Tasks: {0}".format(router_stats.get('cpu_tasks', 0)))
                        info("  GPU Tasks: {0}".format(router_stats.get('gpu_tasks', 0)))
                        info("  Average Response Time: {0}s".format(router_stats.get('avg_response_time', 0)))

                        # Profit registry status
                        registry_stats = self.router.get_profit_registry_stats()
                        info(f"💰 Profit Registry:")
                        info("  Total Strategies: {0}".format(registry_stats.get('total_strategies', 0)))
                        info("  Short-term Strategies: {0}".format(registry_stats.get('short_term', 0)))
                        info("  Mid-term Strategies: {0}".format(registry_stats.get('mid_term', 0)))
                        info("  Long-term Strategies: {0}".format(registry_stats.get('long_term', 0)))

                        # Performance metrics
                        performance = self.router.get_performance_metrics()
                        info(f"⚡ Performance Metrics:")
                        info("  CPU Utilization: {0}%".format(performance.get('cpu_utilization', 0)))
                        info("  GPU Utilization: {0}%".format(performance.get('gpu_utilization', 0)))
                        info("  Memory Usage: {0}MB".format(performance.get('memory_usage', 0)))

                        # System load
                        system_load = self.router.get_system_load()
                        info(f"🖥️  System Load:")
                        info("  CPU Load: {0}".format(system_load.get('cpu_load', 0)))
                        info("  GPU Load: {0}".format(system_load.get('gpu_load', 0)))
                        info("  Memory Load: {0}".format(system_load.get('memory_load', 0)))

                            async def show_profit_registry(self, limit: int = 10):
                            """Display profit registry entries."""
                                if not self.is_initialized:
                                error("System not initialized. Run 'init' first.")
                            return

                            info("💰 PROFIT REGISTRY (Top {0})".format(limit))
                            info("=" * 40)

                            registry_entries = self.router.get_profit_registry_entries(limit=limit)

                                if not registry_entries:
                                warn("No profit registry entries found.")
                            return

                                for i, entry in enumerate(registry_entries, 1):
                                info("{0}. Strategy: {1}".format(i, entry.get('strategy_name', 'Unknown')))
                                info("   Tier: {0}".format(entry.get('strategy_tier', 'Unknown')))
                                info("   Profit Density: {0}".format(entry.get('profit_density', 0)))
                                info("   Success Rate: {0}%".format(entry.get('success_rate', 0)))
                                info("   Preferred Mode: {0}".format(entry.get('preferred_compute_mode', 'Unknown')))
                                info("   Last Updated: {0}".format(entry.get('last_updated', 'Unknown')))
                                info("")

                                    async def route_test_task(self, task_type: str, strategy_tier: str, profit_density: float):
                                    """Test routing a task through the system."""
                                        if not self.is_initialized:
                                        error("System not initialized. Run 'init' first.")
                                    return

                                    info(f"🧪 TESTING TASK ROUTING")
                                    info("=" * 30)

                                    strategy_metadata = {
                                    "strategy_tier": strategy_tier,
                                    "profit_density": profit_density,
                                    "historical_performance": 0.8,
                                    "compute_time": 0.1,
                                    }

                                    info("Task Type: {0}".format(task_type))
                                    info("Strategy Tier: {0}".format(strategy_tier))
                                    info("Profit Density: {0}".format(profit_density))

                                    # Route the task
                                    start_time = time.time()
                                    result = await self.router.route_task(task_type, strategy_metadata)
                                    routing_time = time.time() - start_time

                                    info("Routing Decision: {0}".format(result.get('compute_mode', 'Unknown')))
                                    info("Routing Time: {0}s".format(routing_time))
                                    info("Confidence: {0}".format(result.get('confidence', 0)))
                                    info("Reason: {0}".format(result.get('reason', 'No reason provided')))

                                        async def process_btc_price(self, price: float, volume: float, volatility: float):
                                        """Process BTC price data through the dual-state system."""
                                            if not self.is_initialized:
                                            error("System not initialized. Run 'init' first.")
                                        return

                                        info(f"₿ PROCESSING BTC PRICE DATA")
                                        info("=" * 35)

                                        market_data = {
                                        "price": price,
                                        "volume": volume,
                                        "volatility": volatility,
                                        "timestamp": time.time(),
                                        }

                                        info("BTC Price: ${0}".format(price))
                                        info("Volume: {0}".format(volume))
                                        info("Volatility: {0}".format(volatility))

                                        # Process through different strategies
                                        strategies = ["short", "mid", "long"]

                                            for strategy in strategies:
                                            info("\n📈 Processing {0}-term strategy...".format(strategy.upper()))

                                            strategy_metadata = {
                                            "strategy_tier": strategy,
                                            "profit_density": 0.7 if strategy == "short" else 0.5,
                                            "historical_performance": 0.8,
                                            "compute_time": 0.5,
                                            }

                                            # Route and process
                                            routing_result = await self.router.route_task("btc_price_analysis", strategy_metadata)

                                                if routing_result.get("compute_mode") == "ZPE":
                                                result = await self.cpu_handlers.process_market_data(market_data)
                                                    else:
                                                    result = await self.gpu_handlers.process_market_data(market_data)

                                                    info("  Compute Mode: {0}".format(routing_result.get('compute_mode', 'Unknown')))
                                                    info("  Processing Time: {0}s".format(result.get('processing_time', 0)))
                                                    info("  Signal Strength: {0}".format(result.get('signal_strength', 0)))
                                                    info("  Confidence: {0}".format(result.get('confidence', 0)))

                                                        async def show_strategy_performance(self, strategy_name: Optional[str] = None):
                                                        """Display strategy performance metrics."""
                                                            if not self.is_initialized:
                                                            error("System not initialized. Run 'init' first.")
                                                        return

                                                        info("📊 STRATEGY PERFORMANCE METRICS")
                                                        info("=" * 40)

                                                            if strategy_name:
                                                            # Show specific strategy
                                                            performance = self.router.get_strategy_performance(strategy_name)
                                                                if performance:
                                                                info("Strategy: {0}".format(strategy_name))
                                                                info("Total Executions: {0}".format(performance.get('total_executions', 0)))
                                                                info("Success Rate: {0}%".format(performance.get('success_rate', 0)))
                                                                info("Average Profit: {0}".format(performance.get('avg_profit', 0)))
                                                                info("Preferred Compute Mode: {0}".format(performance.get('preferred_compute_mode', 'Unknown')))
                                                                info("Last Execution: {0}".format(performance.get('last_execution', 'Unknown')))
                                                                    else:
                                                                    warn("Strategy '{0}' not found.".format(strategy_name))
                                                                        else:
                                                                        # Show all strategies
                                                                        all_performance = self.router.get_all_strategy_performance()

                                                                            if not all_performance:
                                                                            warn("No strategy performance data available.")
                                                                        return

                                                                            for strategy, metrics in all_performance.items():
                                                                            info("\n📈 {0}:".format(strategy))
                                                                            info("  Executions: {0}".format(metrics.get('total_executions', 0)))
                                                                            info("  Success Rate: {0}%".format(metrics.get('success_rate', 0)))
                                                                            info("  Avg Profit: {0}".format(metrics.get('avg_profit', 0)))
                                                                            info("  Compute Mode: {0}".format(metrics.get('preferred_compute_mode', 'Unknown')))

                                                                                async def reset_system(self):
                                                                                """Reset the dual-state router system."""
                                                                                    if not self.is_initialized:
                                                                                    error("System not initialized.")
                                                                                return

                                                                                info("🔄 Resetting Dual State Router System...")

                                                                                # Reset router
                                                                                self.router.reset_statistics()
                                                                                self.router.clear_profit_registry()

                                                                                # Reset handlers
                                                                                    if self.cpu_handlers:
                                                                                    self.cpu_handlers.reset()
                                                                                        if self.gpu_handlers:
                                                                                        self.gpu_handlers.reset()

                                                                                        success("✅ System reset completed")

                                                                                            async def run_interactive_mode(self):
                                                                                            """Run interactive CLI mode."""
                                                                                            info("🎮 INTERACTIVE DUAL STATE ROUTER CLI")
                                                                                            info("=" * 40)
                                                                                            info("Type 'help' for commands, 'quit' to exit")

                                                                                                while True:
                                                                                                    try:
                                                                                                    command = input("\n🔄 dual-state> ").strip().lower()

                                                                                                        if command == "quit" or command == "exit":
                                                                                                        info("👋 Goodbye!")
                                                                                                    break
                                                                                                        elif command == "help":
                                                                                                        self._show_help()
                                                                                                            elif command == "status":
                                                                                                            await self.show_system_status()
                                                                                                                elif command == "registry":
                                                                                                                await self.show_profit_registry()
                                                                                                                    elif command == "reset":
                                                                                                                    await self.reset_system()
                                                                                                                        elif command.startswith("test "):
                                                                                                                        parts = command.split()
                                                                                                                            if len(parts) >= 4:
                                                                                                                            await self.route_test_task(parts[1], parts[2], float(parts[3]))
                                                                                                                                else:
                                                                                                                                error("Usage: test <task_type> <strategy_tier> <profit_density>")
                                                                                                                                    elif command.startswith("btc "):
                                                                                                                                    parts = command.split()
                                                                                                                                        if len(parts) >= 4:
                                                                                                                                        await self.process_btc_price(float(parts[1]), float(parts[2]), float(parts[3]))
                                                                                                                                            else:
                                                                                                                                            error("Usage: btc <price> <volume> <volatility>")
                                                                                                                                                elif command.startswith("performance"):
                                                                                                                                                parts = command.split()
                                                                                                                                                strategy = parts[1] if len(parts) > 1 else None
                                                                                                                                                await self.show_strategy_performance(strategy)
                                                                                                                                                    else:
                                                                                                                                                    warn("Unknown command: {0}".format(command))

                                                                                                                                                        except KeyboardInterrupt:
                                                                                                                                                        info("\n👋 Goodbye!")
                                                                                                                                                    break
                                                                                                                                                        except Exception as e:
                                                                                                                                                        error("Error: {0}".format(e))

                                                                                                                                                            def _show_help(self) -> None:
                                                                                                                                                            """Show help information."""
                                                                                                                                                            info("📖 AVAILABLE COMMANDS:")
                                                                                                                                                            info("  status                    - Show system status")
                                                                                                                                                            info("  registry                  - Show profit registry")
                                                                                                                                                            info("  test <type> <tier> <density> - Test task routing")
                                                                                                                                                            info("  btc <price> <volume> <vol> - Process BTC price data")
                                                                                                                                                            info("  performance [strategy]    - Show performance metrics")
                                                                                                                                                            info("  reset                     - Reset system")
                                                                                                                                                            info("  quit/exit                 - Exit CLI")


                                                                                                                                                                async def main():
                                                                                                                                                                """Main CLI entry point."""
                                                                                                                                                                parser = argparse.ArgumentParser(description="Dual State Router CLI - CPU/GPU Orchestration Control")
                                                                                                                                                                parser.add_argument("--init", action="store_true", help="Initialize the system")
                                                                                                                                                                parser.add_argument("--status", action="store_true", help="Show system status")
                                                                                                                                                                parser.add_argument("--registry", action="store_true", help="Show profit registry")
                                                                                                                                                                parser.add_argument("--test", nargs=3, metavar=("TYPE", "TIER", "DENSITY"), help="Test task routing")
                                                                                                                                                                parser.add_argument(
                                                                                                                                                                "--btc",
                                                                                                                                                                nargs=3,
                                                                                                                                                                metavar=("PRICE", "VOLUME", "VOLATILITY"),
                                                                                                                                                                type=float,
                                                                                                                                                                help="Process BTC price data",
                                                                                                                                                                )
                                                                                                                                                                parser.add_argument("--performance", nargs="?", metavar="STRATEGY", help="Show performance metrics")
                                                                                                                                                                parser.add_argument("--reset", action="store_true", help="Reset system")
                                                                                                                                                                parser.add_argument("--interactive", action="store_true", help="Run interactive mode")

                                                                                                                                                                args = parser.parse_args()

                                                                                                                                                                cli = DualStateRouterCLI()

                                                                                                                                                                # Initialize if requested or if any command needs it
                                                                                                                                                                if args.init or any(
                                                                                                                                                                [
                                                                                                                                                                args.status,
                                                                                                                                                                args.registry,
                                                                                                                                                                args.test,
                                                                                                                                                                args.btc,
                                                                                                                                                                args.performance,
                                                                                                                                                                args.reset,
                                                                                                                                                                args.interactive,
                                                                                                                                                                ]
                                                                                                                                                                    ):
                                                                                                                                                                        if not await cli.initialize_system():
                                                                                                                                                                    return 1

                                                                                                                                                                    # Execute commands
                                                                                                                                                                        if args.status:
                                                                                                                                                                        await cli.show_system_status()
                                                                                                                                                                            elif args.registry:
                                                                                                                                                                            await cli.show_profit_registry()
                                                                                                                                                                                elif args.test:
                                                                                                                                                                                await cli.route_test_task(args.test[0], args.test[1], float(args.test[2]))
                                                                                                                                                                                    elif args.btc:
                                                                                                                                                                                    await cli.process_btc_price(args.btc[0], args.btc[1], args.btc[2])
                                                                                                                                                                                        elif args.performance is not None:
                                                                                                                                                                                        await cli.show_strategy_performance(args.performance)
                                                                                                                                                                                            elif args.reset:
                                                                                                                                                                                            await cli.reset_system()
                                                                                                                                                                                                elif args.interactive:
                                                                                                                                                                                                await cli.run_interactive_mode()
                                                                                                                                                                                                    elif not args.init:
                                                                                                                                                                                                    parser.print_help()

                                                                                                                                                                                                return 0


                                                                                                                                                                                                    if __name__ == "__main__":
                                                                                                                                                                                                    sys.exit(asyncio.run(main()))
