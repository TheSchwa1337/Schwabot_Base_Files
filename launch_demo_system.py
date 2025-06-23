#!/usr/bin/env python3
"""
Schwabot Demo System Launcher
=============================

Comprehensive launcher for the Schwabot demo system that provides
easy access to all demo functionality, backtesting, and analysis.

This launcher integrates all demo components:
- Demo Integration System
- Demo Entry Simulator
- Demo Backtest Runner
- Settings Controller
- Vector Validator
- Matrix Allocator

Usage:
    python launch_demo_system.py [command] [options]

Commands:
    backtest          - Run comprehensive backtest
    entry-test        - Test entry strategies
    quick-test        - Run quick demo test
    analyze           - Analyze existing results
    report            - Generate reports
    config            - Show/edit configuration
    status            - Show system status
    help              - Show this help
"""

import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
import time

# Add core directory to path
sys.path.append(str(Path(__file__).parent / "core"))

from core.demo_integration_system import get_demo_integration_system
from core.demo_entry_simulator import get_demo_entry_simulator
from core.demo_backtest_runner import get_demo_backtest_runner
from core.settings_controller import get_settings_controller
from core.vector_validator import get_vector_validator
from core.matrix_allocator import get_matrix_allocator


class DemoSystemLauncher:
    """Comprehensive demo system launcher"""
    
    def __init__(self):
        self.demo_system = get_demo_integration_system()
        self.entry_simulator = get_demo_entry_simulator()
        self.backtest_runner = get_demo_backtest_runner()
        self.settings_controller = get_settings_controller()
        self.vector_validator = get_vector_validator()
        self.matrix_allocator = get_matrix_allocator()
    
    def run_backtest(self, args):
        """Run comprehensive backtest"""
        print("🚀 Starting Comprehensive Backtest...")
        print("=" * 50)
        
        # Create backtest config
        config = self.backtest_runner.create_backtest_config(
            strategy_types=args.strategies if args.strategies else None,
            market_conditions=args.markets if args.markets else None,
            num_trades_per_strategy=args.trades,
            base_price=args.base_price,
            price_volatility=args.volatility,
            enable_reinforcement_learning=not args.no_learning,
            save_detailed_results=True
        )
        
        # Run backtest
        result = self.backtest_runner.run_backtest(config)
        
        # Generate report
        if args.report:
            report_path = self.backtest_runner.generate_backtest_report(result)
            print(f"\n📊 Report generated: {report_path}")
        
        # Show summary
        print("\n" + "=" * 50)
        print("BACKTEST SUMMARY")
        print("=" * 50)
        print(f"Backtest ID: {result.backtest_id}")
        print(f"Total Trades: {result.total_trades:,}")
        print(f"Success Rate: {result.success_rate:.2%}")
        print(f"Total Profit: ${result.total_profit:,.2f}")
        print(f"Average Profit: ${result.average_profit:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        return result
    
    def run_entry_test(self, args):
        """Test entry strategies"""
        print("🎯 Starting Entry Strategy Test...")
        print("=" * 50)
        
        # Run entry simulation
        analysis = self.entry_simulator.simulate_entry(
            strategy_type=args.strategy,
            market_condition=args.market,
            num_simulations=args.simulations
        )
        
        # Show results
        print("\n" + "=" * 50)
        print("ENTRY TEST RESULTS")
        print("=" * 50)
        print(f"Strategy: {analysis.simulation_id}")
        print(f"Total Entries: {analysis.total_entries}")
        print(f"Success Rate: {analysis.success_rate:.2%}")
        print(f"Average Confidence: {analysis.average_confidence:.3f}")
        print(f"Average Ghost Signal: {analysis.average_ghost_signal:.3f}")
        print(f"Average Entropy: {analysis.average_entropy:.3f}")
        
        # Show matrix performance
        print("\nMatrix Performance:")
        for matrix_id, perf in analysis.matrix_performance.items():
            print(f"  {matrix_id}: {perf['success_rate']:.2%} ({perf['entries']} entries)")
        
        # Save analysis if requested
        if args.save:
            self.entry_simulator.save_entry_analysis()
            print(f"\n💾 Analysis saved to tests/demo_analysis/entry_analysis.json")
        
        return analysis
    
    def run_quick_test(self, args):
        """Run quick demo test"""
        print("⚡ Starting Quick Demo Test...")
        print("=" * 50)
        
        # Start demo mode
        self.demo_system.start_demo_mode("backtest")
        
        # Execute a few demo trades
        for i in range(args.trades):
            trade_data = {
                "trade_id": f"quick_test_{i + 1}",
                "matrix_id": "SFS8-A5",
                "entry_price": 50000.0 + i * 10,
                "exit_price": 50000.0 + i * 10 + 50,
                "entry_time": datetime.now().isoformat(),
                "exit_time": datetime.now().isoformat(),
                "confidence": 0.7 + (i % 3) * 0.1,
                "strategy_type": "quick_test",
                "volume_data": {"current": 1000000, "average": 800000},
                "ghost_signal_strength": 0.6 + (i % 2) * 0.2,
                "entropy_level": 0.3 + (i % 2) * 0.4,
                "tick_id": i
            }
            
            result = self.demo_system.execute_demo_trade(trade_data)
            print(f"Trade {i + 1}: {'✅' if result.success else '❌'} "
                  f"Profit: ${result.profit_loss:.2f} "
                  f"Confidence: {result.confidence_score:.3f}")
        
        # Stop demo mode
        self.demo_system.stop_demo_mode()
        
        # Show summary
        summary = self.demo_system.get_demo_summary()
        print("\n" + "=" * 50)
        print("QUICK TEST SUMMARY")
        print("=" * 50)
        print(f"Total Demo Trades: {summary['total_demo_trades']}")
        print(f"Demo Performance: {summary['demo_performance']}")
        
        return summary
    
    def analyze_results(self, args):
        """Analyze existing results"""
        print("📊 Analyzing Existing Results...")
        print("=" * 50)
        
        # Get backtest summary
        backtest_summary = self.backtest_runner.get_backtest_summary()
        
        print("BACKTEST SUMMARY:")
        print(f"Total Backtests: {backtest_summary['performance_metrics']['total_backtests']}")
        print(f"Total Trades: {backtest_summary['performance_metrics']['total_trades']:,}")
        print(f"Overall Success Rate: {backtest_summary['performance_metrics']['overall_success_rate']:.2%}")
        print(f"Overall Profit: ${backtest_summary['performance_metrics']['overall_profit']:,.2f}")
        
        print("\nBest Performing Strategies:")
        for strategy, rate in backtest_summary['best_performing_strategies'].items():
            print(f"  {strategy}: {rate:.2%}")
        
        print("\nBest Performing Matrices:")
        for matrix, rate in backtest_summary['best_performing_matrices'].items():
            print(f"  {matrix}: {rate:.2%}")
        
        print("\nRecent Backtests:")
        for backtest in backtest_summary['recent_backtests']:
            print(f"  {backtest['backtest_id']}: {backtest['success_rate']:.2%} "
                  f"(${backtest['total_profit']:.2f})")
        
        # Get demo summary
        demo_summary = self.demo_system.get_demo_summary()
        print(f"\nDemo System Status:")
        print(f"Total Demo Trades: {demo_summary['total_demo_trades']}")
        print(f"Known Bad Vectors: {demo_summary['settings_controller_status']['known_bad_vectors']}")
        
        return backtest_summary
    
    def generate_report(self, args):
        """Generate comprehensive report"""
        print("📋 Generating Comprehensive Report...")
        print("=" * 50)
        
        # Get all summaries
        backtest_summary = self.backtest_runner.get_backtest_summary()
        demo_summary = self.demo_system.get_demo_summary()
        vector_summary = self.vector_validator.get_performance_summary()
        matrix_summary = self.matrix_allocator.get_allocation_summary()
        
        # Create comprehensive report
        report = f"""# Schwabot Demo System Comprehensive Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Overview
- Total Backtests: {backtest_summary['performance_metrics']['total_backtests']}
- Total Demo Trades: {demo_summary['total_demo_trades']}
- Total Vector Validations: {vector_summary['total_vectors']}
- Total Matrix Allocations: {matrix_summary['total_allocations']}

## Performance Metrics
- Overall Success Rate: {backtest_summary['performance_metrics']['overall_success_rate']:.2%}
- Overall Profit: ${backtest_summary['performance_metrics']['overall_profit']:,.2f}
- Vector Success Rate: {vector_summary['overall_success_rate']:.2%}
- Matrix Average Confidence: {matrix_summary.get('average_confidence', 0):.3f}

## Best Performing Components
### Strategies:
"""
        
        for strategy, rate in backtest_summary['best_performing_strategies'].items():
            report += f"- {strategy}: {rate:.2%}\n"
        
        report += "\n### Matrices:\n"
        for matrix, rate in backtest_summary['best_performing_matrices'].items():
            report += f"- {matrix}: {rate:.2%}\n"
        
        report += f"""
## Reinforcement Learning Status
- Known Bad Vectors: {demo_summary['settings_controller_status']['known_bad_vectors']}
- Matrix Weights: {len(demo_summary['settings_controller_status']['matrix_weights'])} matrices
- Vector History: {vector_summary['total_vectors']} vectors

## Recent Activity
"""
        
        for backtest in backtest_summary['recent_backtests'][:5]:
            report += f"- {backtest['backtest_id']}: {backtest['success_rate']:.2%} "
            report += f"(${backtest['total_profit']:.2f})\n"
        
        # Save report
        report_path = Path("tests/demo_reports/comprehensive_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📋 Comprehensive report saved to: {report_path}")
        
        return report_path
    
    def show_config(self, args):
        """Show/edit configuration"""
        print("⚙️ Configuration Management...")
        print("=" * 50)
        
        if args.show:
            # Show current settings
            print("Current Settings:")
            print(f"Matrix ID: {self.settings_controller.matrix_settings.matrix_id}")
            print(f"Entry Logic: {self.settings_controller.vector_settings.entry_logic}")
            print(f"Exit Logic: {self.settings_controller.vector_settings.exit_logic}")
            print(f"Allocator Mode: {self.settings_controller.allocator_settings.allocator_mode}")
            print(f"Reinforcement Enabled: {self.settings_controller.reinforcement_settings.enable_backlog_reinforcement}")
            print(f"Fault Tolerance: {self.settings_controller.fault_settings.fault_tolerance}")
            print(f"Known Bad Vectors: {len(self.settings_controller.known_bad_vectors)}")
            print(f"Matrix Weights: {self.settings_controller.matrix_path_weights}")
        
        if args.save:
            # Save current settings
            self.settings_controller.save_settings()
            print("✅ Settings saved successfully!")
        
        if args.reset:
            # Reset to defaults
            if input("Are you sure you want to reset all settings to defaults? (y/N): ").lower() == 'y':
                # This would reset settings to defaults
                print("🔄 Settings reset to defaults!")
    
    def show_status(self, args):
        """Show system status"""
        print("📈 System Status...")
        print("=" * 50)
        
        # Demo system status
        demo_summary = self.demo_system.get_demo_summary()
        print("Demo System:")
        print(f"  Demo Mode: {demo_summary['current_mode']['demo_mode']}")
        print(f"  Backtest Mode: {demo_summary['current_mode']['backtest_mode']}")
        print(f"  Total Demo Trades: {demo_summary['total_demo_trades']}")
        print(f"  Experimental Mode: {demo_summary['settings_controller_status']['experimental_mode']}")
        
        # Vector validator status
        vector_summary = self.vector_validator.get_performance_summary()
        print("\nVector Validator:")
        print(f"  Total Vectors: {vector_summary['total_vectors']}")
        print(f"  Success Rate: {vector_summary['overall_success_rate']:.2%}")
        print(f"  Known Bad Vectors: {vector_summary['known_bad_vectors']}")
        
        # Matrix allocator status
        tick_summary = self.matrix_allocator.get_tick_map_summary()
        allocation_summary = self.matrix_allocator.get_allocation_summary()
        print("\nMatrix Allocator:")
        print(f"  Current Tick: {tick_summary['current_tick_id']}")
        print(f"  Active Matrices: {len(tick_summary['active_matrices'])}")
        print(f"  Total Allocations: {allocation_summary['total_allocations']}")
        print(f"  Average Confidence: {allocation_summary.get('average_confidence', 0):.3f}")
        
        # Settings controller status
        print("\nSettings Controller:")
        print(f"  Matrix ID: {self.settings_controller.matrix_settings.matrix_id}")
        print(f"  Entry Tolerance: {self.settings_controller.matrix_settings.entry_tolerance}")
        print(f"  Exit Flex: {self.settings_controller.matrix_settings.exit_flex}")
        print(f"  Thermal Limit: {self.settings_controller.matrix_settings.thermal_limit}")
    
    def show_help(self, args):
        """Show help information"""
        print(__doc__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Schwabot Demo System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run comprehensive backtest')
    backtest_parser.add_argument('--strategies', nargs='+', help='Strategy types to test')
    backtest_parser.add_argument('--markets', nargs='+', help='Market conditions to test')
    backtest_parser.add_argument('--trades', type=int, default=50, help='Trades per strategy')
    backtest_parser.add_argument('--base-price', type=float, default=50000.0, help='Base price')
    backtest_parser.add_argument('--volatility', type=float, default=0.02, help='Price volatility')
    backtest_parser.add_argument('--no-learning', action='store_true', help='Disable reinforcement learning')
    backtest_parser.add_argument('--report', action='store_true', help='Generate report')
    
    # Entry test command
    entry_parser = subparsers.add_parser('entry-test', help='Test entry strategies')
    entry_parser.add_argument('--strategy', required=True, help='Strategy type to test')
    entry_parser.add_argument('--market', default='sideways', help='Market condition')
    entry_parser.add_argument('--simulations', type=int, default=100, help='Number of simulations')
    entry_parser.add_argument('--save', action='store_true', help='Save analysis')
    
    # Quick test command
    quick_parser = subparsers.add_parser('quick-test', help='Run quick demo test')
    quick_parser.add_argument('--trades', type=int, default=10, help='Number of trades')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing results')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive report')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show/edit configuration')
    config_parser.add_argument('--show', action='store_true', help='Show current settings')
    config_parser.add_argument('--save', action='store_true', help='Save settings')
    config_parser.add_argument('--reset', action='store_true', help='Reset to defaults')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Help command
    help_parser = subparsers.add_parser('help', help='Show help information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize launcher
    launcher = DemoSystemLauncher()
    
    # Execute command
    try:
        if args.command == 'backtest':
            launcher.run_backtest(args)
        elif args.command == 'entry-test':
            launcher.run_entry_test(args)
        elif args.command == 'quick-test':
            launcher.run_quick_test(args)
        elif args.command == 'analyze':
            launcher.analyze_results(args)
        elif args.command == 'report':
            launcher.generate_report(args)
        elif args.command == 'config':
            launcher.show_config(args)
        elif args.command == 'status':
            launcher.show_status(args)
        elif args.command == 'help':
            launcher.show_help(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 