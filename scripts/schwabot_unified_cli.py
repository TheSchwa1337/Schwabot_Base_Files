#!/usr/bin/env python3
"""
Schwabot Unified CLI - Master Command Line Interface

Provides unified access to all Schwabot CLI tools and functionality
across Windows, macOS, and Linux platforms.

Available CLI Tools:
- Dual State Router CLI (CPU/GPU, orchestration)
- Tensor State Manager CLI (tensor, operations)
- System Monitor CLI (system, monitoring)
- Live Trading CLI (real, trading)
- Strategy CLI (strategy, management)
- Digest Time Mapper CLI (phase wheel & temporal, socketing)
- Cross-platform validation

Usage:
    python schwabot_unified_cli.py [tool] [command] [options]
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.safe_print import error, info, safe_print, success, warn


class SchwabotUnifiedCLI:
    """Unified CLI launcher for all Schwabot tools."""

    def __init__(self):
        """Initialize the unified CLI."""
        self.available_tools = {}
            "dual-state": {}
                "description": "CPU/GPU Orchestration Control",
                "module": "core.cli_dual_state_router",
                "help": "Control profit-tiered dualistic compute orchestration"
            },
            "tensor": {}
                "description": "Tensor State Management",
                "module": "core.cli_tensor_state_manager",
                "help": "Manage tensor states and BTC price processing"
            },
            "monitor": {}
                "description": "System Monitoring",
                "module": "core.cli_system_monitor",
                "help": "Monitor system performance and health"
            },
            "live": {}
                "description": "Live Trading",
                "module": "core.cli_live_entry",
                "help": "Execute live trades through API connections"
            },
            "strategy": {}
                "description": "Strategy Management",
                "module": "schwabot_cli",
                "help": "Manage trading strategies and bit mapping"
            },
            "digest": {}
                "description": "Digest Time Mapper",
                "module": "cli.digest_time_mapper_cli",
                "help": "Phase wheel & temporal socketing for millisecond tick processing"
            },
            "profit": {}
                "description": "Pure Profit Calculator",
                "module": "cli.pure_profit_calculator_cli",
                "help": "Mathematical profit calculation with full introspection"
            },
            "validate": {}
                "description": "Cross-Platform Validation",
                "module": "cross_platform_cli_validator",
                "help": "Validate CLI functionality across platforms"
            },
            "orbit": {}
                "description": "Orbital Profit Control",
                "module": "core.cli_orbital_profit_control",
                "help": "Control the Orbital Profit Control System"
            },
            "entropy": {}
                "description": "Entropy Risk Manager",
                "module": "core.cli_entropy_manager",
                "help": "CLI for Entropy-Driven Risk Management"
            },
            "pipeline": {}
                "description": "Automated Trading Pipeline",
                "module": "cli.automated_trading_pipeline_cli",
                "help": "Unified decision engine connecting all core systems"
            },
            "exchange": {}
                "description": "Secure Exchange Manager",
                "module": "cli.secure_exchange_cli",
                "help": "Professional API key & exchange management with security"
            },
            "deploy": {}
                "description": "Production Deployment Manager",
                "module": "cli.production_deployment_cli",
                "help": "Enterprise-grade production deployment with environment validation"
            },
            "fill": {}
                "description": "Fill Handler Management",
                "module": "cli.fill_handler_cli",
                "help": "Advanced crypto trading fill management with partial fills and retries"
            }
        }

    def show_banner(self):
        """Display the Schwabot CLI banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🚀 SCHWABOT UNIFIED CLI 🚀                ║
║                                                              ║
║  Advanced Trading System with CPU/GPU Orchestration         ║
║  Cross-Platform Command Line Interface                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        safe_print(banner)

    def show_available_tools(self):
        """Display all available CLI tools."""
        info("📋 AVAILABLE CLI TOOLS")
        info("=" * 30)

        for tool_name, tool_info in self.available_tools.items():
            info(f"🔧 {tool_name:12} - {tool_info['description']}")
            info(f"    {tool_info['help']}")
            info("")

    def show_tool_help(self, tool_name: str):
        """Show help for a specific tool."""
        if tool_name not in self.available_tools:
            error(f"Unknown tool: {tool_name}")
            return

        tool_info = self.available_tools[tool_name]
        info(f"📖 HELP FOR: {tool_name.upper()}")
        info("=" * 30)
        info(f"Description: {tool_info['description']}")
        info(f"Purpose: {tool_info['help']}")
        info(f"Module: {tool_info['module']}")
        info("")

        # Show tool-specific help
        try:
            if tool_name == "dual-state":
                info("Usage: python schwabot_unified_cli.py dual-state [options]")
                info("Options:")
                info("  --init                    Initialize the system")
                info("  --status                  Show system status")
                info("  --registry                Show profit registry")
                info("  --test TYPE TIER DENSITY  Test task routing")
                info("  --btc PRICE VOLUME VOL    Process BTC price data")
                info("  --performance [STRATEGY]  Show performance metrics")
                info("  --reset                   Reset system")
                info("  --interactive             Run interactive mode")
            elif tool_name == "tensor":
                info("Usage: python schwabot_unified_cli.py tensor [options]")
                info("Options:")
                info("  --init                    Initialize the system")
                info("  --status                  Show system status")
                info("  --cache                   Show tensor cache")
                info("  --clear-cache             Clear tensor cache")
                info("  --performance             Show performance metrics")
                info("  --inspect TENSOR          Inspect specific tensor")
                info("  --btc PRICE VOLUME VOL    Process BTC through tensor pipeline")
                info("  --export TENSOR FILE      Export tensor data")
                info("  --interactive             Run interactive mode")
            elif tool_name == "monitor":
                info("Usage: python schwabot_unified_cli.py monitor [options]")
                info("Options:")
                info("  --init                    Initialize the system")
                info("  --status                  Show system status")
                info("  --performance             Show performance metrics")
                info("  --trading                 Show trading metrics")
                info("  --health                  Show health diagnostics")
                info("  --monitor INTERVAL        Start real-time monitoring")
                info("  --export FILE             Export system report")
                info("  --interactive             Run interactive mode")
            elif tool_name == "live":
                info("Usage: python schwabot_unified_cli.py live [options]")
                info("Options:")
                info("  --mode MODE               Trading operation mode")
                info("  --config FILE             Trading bot configuration file")
                info("  --symbol SYMBOL           Trading symbol")
                info("  --interval SECONDS        Trading interval in seconds")
                info("  --force-refresh           Force refresh market data")
                info("  --safe-mode               Run in safe mode")
            elif tool_name == "strategy":
                info("Usage: python schwabot_unified_cli.py strategy [command] [options]")
                info("Commands:")
                info("  fit                       Run Schwafit on a price series")
                info("  test                      Run a mock Schwafit test")
                info("  status                    Show Schwafit memory state")
                info("  select-strategy           Select strategy using Schwafit")
                info("  match-matrix              Match hash to matrix")
                info("  live-status               Show live handler status")
                info("  ferris-spin               Spin Ferris wheel")
                info("  live-tick                 Simulate a live tick")
                info("  entry-exit                Calculate entry/exit")
                info("  ghost-trade               Simulate ghost trade")
            elif tool_name == "digest":
                info("Usage: python schwabot_unified_cli.py digest [command] [options]")
                info("Commands:")
                info("  init [--frame-window MS] [--phase-period MS]  Initialize mapper")
                info("  process-tick <price> [--volume] [--bid] [--ask]  Process price tick")
                info("  generate-digest [--frame-count N]  Generate SHA digest from frames")
                info("  ferris-wheel <duration>  Run continuous processing loop")
                info("  analyze <digest>  Analyze temporal patterns in digest")
                info("  status  Show mapper statistics")
                info("  stop  Stop processing loop")
            elif tool_name == "profit":
                info("Usage: python schwabot_unified_cli.py profit [command] [options]")
                info("Commands:")
                info("  init [--risk-tolerance] [--profit-target] [--position-size] [--mode]  Initialize calculator")
                info("  flash  Display flash screen with current state")
                info("  calculate <btc_price> [--eth-price] [--usdc-volume] [--volatility] [--momentum] [--volume-profile]  Perform profit calculation")
                info("  explain [--full]  Explain last calculation (summary or, full)")
                info("  metrics  Show calculation metrics and performance")
                info("  test  Run comprehensive test suite")
                info("  validate  Validate profit calculation purity")
                info("  errors  Show error summary and recent errors")
                info("  reset  Reset error log and metrics")
            elif tool_name == "validate":
                info("Usage: python schwabot_unified_cli.py validate [options]")
                info("Options:")
                info("  --all                     Run all validation tests")
                info("  --platform                Test platform detection")
                info("  --cli                     Test CLI functionality")
                info("  --network                 Test network functionality")
                info("  --math                    Test mathematical operations")
                info("  --report                  Generate validation report")
            elif tool_name == "orbit":
                info("Usage: python schwabot_unified_cli.py orbit [command] [options]")
                info("Commands:")
                info("  init                      Initialise and start the orbital system")
                info("  status                    Show status snapshot")
                info("  optimize [--market-data]  Run one optimise cycle with optional data")
                info("  stop                      Gracefully stop the system")
            elif tool_name == "entropy":
                info("Usage: python schwabot_unified_cli.py entropy [command] [options]")
                info("Commands:")
                info("  init [--active]           Initialise (and optionally, start) manager")
                info("  status                    Show status snapshot")
                info("  process [--market-data]   Single risk-management pass")
                info("  stop                      Stop manager (placeholder)")
            elif tool_name == "pipeline":
                info("Usage: python schwabot_unified_cli.py pipeline [command] [options]")
                info("Commands:")
                info("  init [--risk-tolerance] [--profit-target] [--position-size] [--mode] [--registry-path]  Initialize pipeline")
                info("  process-tick <price> [--volume] [--bid] [--ask]  Process single price tick")
                info("  run-stream <duration> [--max-decisions]  Run continuous pipeline on simulated stream")
                info("  explain  Explain last trading decision")
                info("  metrics  Show pipeline performance metrics")
                info("  decisions [--count]  Show recent trading decisions")
                info("  stop  Stop continuous pipeline")
            elif tool_name == "exchange":
                info("Usage: python schwabot_unified_cli.py exchange [command] [options]")
                info("Commands:")
                info("  init                      Initialize secure exchange manager")
                info("  status                    Show current API key status")
                info("  add-key <api_key> <secret>  Add new API key")
                info("  remove-key <api_key>      Remove API key")
                info("  list-keys                 List all API keys")
                info("  test-connection           Test API key connection")
                info("  interactive               Run interactive mode")
        except Exception as e:
            warn(f"Could not load specific help for {tool_name}: {e}")

    async def launch_tool(self, tool_name: str, args: List[str]):
        """Launch a specific CLI tool."""
        if tool_name not in self.available_tools:
            error(f"Unknown tool: {tool_name}")
            return 1

        tool_info = self.available_tools[tool_name]
        module_name = tool_info['module']

        try:
            info(f"🚀 Launching {tool_name} CLI...")

            # Import and run the tool
            if module_name == "core.cli_dual_state_router":
                from core.cli_dual_state_router import main as tool_main
            elif module_name == "core.cli_tensor_state_manager":
                from core.cli_tensor_state_manager import main as tool_main
            elif module_name == "core.cli_system_monitor":
                from core.cli_system_monitor import main as tool_main
            elif module_name == "core.cli_live_entry":
                from core.cli_live_entry import main as tool_main
            elif module_name == "schwabot_cli":
                from schwabot_cli import main as tool_main
            elif module_name == "cross_platform_cli_validator":
                from cross_platform_cli_validator import main as tool_main
            elif module_name == "core.cli_orbital_profit_control":
                 from core.cli_orbital_profit_control import main as tool_main
            elif module_name == "core.cli_entropy_manager":
                 from core.cli_entropy_manager import main as tool_main
            elif module_name == "cli.digest_time_mapper_cli":
                 from cli.digest_time_mapper_cli import main as tool_main
            elif module_name == "cli.pure_profit_calculator_cli":
                 from cli.pure_profit_calculator_cli import main as tool_main
            elif module_name == "cli.automated_trading_pipeline_cli":
                 from cli.automated_trading_pipeline_cli import main as tool_main
            elif module_name == "cli.secure_exchange_cli":
                 from cli.secure_exchange_cli import main as tool_main
            elif module_name == "cli.production_deployment_cli":
                 from cli.production_deployment_cli import main as tool_main
            else:
                error(f"Unknown module: {module_name}")
                return 1

            # Set up sys.argv for the tool
            original_argv = sys.argv
            sys.argv = [f"schwabot_{tool_name}"] + args

            try:
                # Run the tool
                if asyncio.iscoroutinefunction(tool_main):
                    result = await tool_main()
                else:
                    result = tool_main()
                return result if result is not None else 0
            finally:
                # Restore original argv
                sys.argv = original_argv

        except ImportError as e:
            error(f"Failed to import {module_name}: {e}")
            return 1
        except Exception as e:
            error(f"Failed to launch {tool_name}: {e}")
            return 1

    def show_quick_start(self):
        """Show quick start guide."""
        info("🚀 QUICK START GUIDE")
        info("=" * 25)
        info("1. Initialize the system:")
        info("   python schwabot_unified_cli.py dual-state --init")
        info("")
        info("2. Check system status:")
        info("   python schwabot_unified_cli.py monitor --status")
        info("")
        info("3. Process BTC price data:")
        info("   python schwabot_unified_cli.py dual-state --btc 50000 1000 0.15")
        info("")
        info("4. Start real-time monitoring:")
        info("   python schwabot_unified_cli.py monitor --monitor 5")
        info("")
        info("5. Run interactive mode:")
        info("   python schwabot_unified_cli.py dual-state --interactive")
        info("")
        info("For more help on any tool:")
        info("   python schwabot_unified_cli.py [tool] --help")

    def show_system_info(self):
        """Show system information."""
        import platform

        info("💻 SYSTEM INFORMATION")
        info("=" * 25)
        info(f"Operating System: {platform.system()} {platform.release()}")
        info(f"Architecture: {platform.machine()}")
        info(f"Python Version: {platform.python_version()}")
        info(f"Platform: {sys.platform}")
        info(f"Working Directory: {os.getcwd()}")
        info(f"Python Executable: {sys.executable}")

    async def run_interactive_mode(self):
        """Run interactive unified CLI mode."""
        self.show_banner()
        info("🎮 INTERACTIVE UNIFIED CLI MODE")
        info("=" * 35)
        info("Type 'help' for commands, 'tools' for available tools, 'quit' to exit")

        while True:
            try:
                command = input("\n🚀 schwabot> ").strip().lower()

                if command == 'quit' or command == 'exit':
                    info("👋 Goodbye!")
                    break
                elif command == 'help':
                    self._show_interactive_help()
                elif command == 'tools':
                    self.show_available_tools()
                elif command == 'quickstart':
                    self.show_quick_start()
                elif command == 'system':
                    self.show_system_info()
                elif command.startswith('launch '):
                    parts = command.split()
                    if len(parts) >= 2:
                        tool_name = parts[1]
                        tool_args = parts[2:] if len(parts) > 2 else []
                        await self.launch_tool(tool_name, tool_args)
                    else:
                        error("Usage: launch <tool_name> [args...]")
                elif command.startswith('help '):
                    tool_name = command.split(' ', 1)[1]
                    self.show_tool_help(tool_name)
                else:
                    warn(f"Unknown command: {command}")
                    info("Type 'help' for available commands")

            except KeyboardInterrupt:
                info("\n👋 Goodbye!")
                break
            except Exception as e:
                error(f"Error: {e}")

    def _show_interactive_help(self):
        """Show interactive mode help."""
        info("📖 INTERACTIVE MODE COMMANDS:")
        info("  help                       - Show this help")
        info("  tools                      - Show available tools")
        info("  quickstart                 - Show quick start guide")
        info("  system                     - Show system information")
        info("  launch <tool> [args...]    - Launch a specific tool")
        info("  help <tool>                - Show help for specific tool")
        info("  quit/exit                  - Exit interactive mode")
        info("")
        info("📋 AVAILABLE TOOLS:")
        for tool_name, tool_info in self.available_tools.items():
            info(f"  {tool_name:12} - {tool_info['description']}")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser()
        description="Schwabot Unified CLI - Master Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python schwabot_unified_cli.py dual-state --init
  python schwabot_unified_cli.py monitor --status
  python schwabot_unified_cli.py tensor --btc 50000 1000 0.15
  python schwabot_unified_cli.py --interactive
        """
    )

    parser.add_argument("tool", nargs="?", help="CLI tool to launch")
    parser.add_argument("tool_args", nargs="*", help="Arguments for the tool")
    parser.add_argument("--help-tool", metavar="TOOL", help="Show help for specific tool")
    parser.add_argument("--tools", action="store_true", help="Show all available tools")
    parser.add_argument("--quickstart", action="store_true", help="Show quick start guide")
    parser.add_argument("--system", action="store_true", help="Show system information")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--version", action="version", version="Schwabot Unified CLI v1.0.0")

    args = parser.parse_args()

    cli = SchwabotUnifiedCLI()

    # Show banner
    cli.show_banner()

    # Handle special commands
    if args.help_tool:
        cli.show_tool_help(args.help_tool)
        return 0
    elif args.tools:
        cli.show_available_tools()
        return 0
    elif args.quickstart:
        cli.show_quick_start()
        return 0
    elif args.system:
        cli.show_system_info()
        return 0
    elif args.interactive:
        await cli.run_interactive_mode()
        return 0
    elif args.tool:
        # Launch specific tool
        return await cli.launch_tool(args.tool, args.tool_args)
    else:
        # No tool specified, show help
        parser.print_help()
        info("")
        cli.show_quick_start()
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 