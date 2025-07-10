#!/usr/bin/env python3
"""
Pure Profit Calculator CLI - Mathematical Profit Calculation Interface
Provides CLI commands for pure profit calculations with full introspection
and mathematical transparency.

Commands:
  * init                    – initialize profit calculator
  * flash                   – display flash screen with current state
  * calculate <btc_price>   – perform profit calculation with sample data
  * explain [--full]        – explain last calculation (summary or full)
  * metrics                 – show calculation metrics and performance
  * test                    – run comprehensive test suite
  * validate                – validate profit calculation purity
  * errors                  – show error summary and recent errors
  * reset                   – reset error log and metrics
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

from core.pure_profit_calculator import (
    HistoryState,
    MarketData,
    ProcessingMode,
    PureProfitCalculator,
    StrategyParameters,
    assert_zpe_isolation,
    create_pure_profit_calculator,
    create_sample_market_data,
)

logger = logging.getLogger(__name__)


class PureProfitCalculatorCLI:
    """CLI interface for Pure Profit Calculator operations."""

    def __init__(self):
        self.calculator: Optional[PureProfitCalculator] = None

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def init(
        self,
        risk_tolerance: float = 0.02,
        profit_target: float = 0.05,
        position_size: float = 0.1,
        processing_mode: str = "hybrid",
    ) -> bool:
        """Initialize the pure profit calculator."""
        try:
            logger.info("Initializing Pure Profit Calculator...")

            # Create strategy parameters
            strategy_params = StrategyParameters(
                risk_tolerance=risk_tolerance,
                profit_target=profit_target,
                position_size=position_size,
            )

            # Parse processing mode
            mode_map = {
                "gpu": ProcessingMode.GPU_ACCELERATED,
                "cpu": ProcessingMode.CPU_FALLBACK,
                "hybrid": ProcessingMode.HYBRID,
                "safe": ProcessingMode.SAFE_MODE,
            }
            mode = mode_map.get(processing_mode.lower(), ProcessingMode.HYBRID)

            # Create calculator
            self.calculator = create_pure_profit_calculator(strategy_params=strategy_params, processing_mode=mode)

            # Verify ZPE/ZBE isolation
            assert_zpe_isolation()

            logger.info("✅ Pure Profit Calculator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Pure Profit Calculator: {e}")
            return False

    def flash(self) -> str:
        """Display flash screen with current calculator state."""
        if not self.calculator:
            return "❌ Calculator not initialized. Run 'init' first."

        try:
            # Capture flash screen output
            import contextlib
            import io

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                self.calculator.flash_screen()

            return f.getvalue()

        except Exception as e:
            logger.error(f"❌ Error displaying flash screen: {e}")
            return f"❌ Flash screen error: {e}"

    def calculate(
        self,
        btc_price: float,
        eth_price: float = 3000.0,
        usdc_volume: float = 1000000.0,
        volatility: float = 0.02,
        momentum: float = 0.01,
        volume_profile: float = 0.8,
    ) -> str:
        """Perform a profit calculation with specified market data."""
        if not self.calculator:
            return "❌ Calculator not initialized. Run 'init' first."

        try:
            # Create market data
            market_data = MarketData(
                timestamp=time.time(),
                btc_price=btc_price,
                eth_price=eth_price,
                usdc_volume=usdc_volume,
                volatility=volatility,
                momentum=momentum,
                volume_profile=volume_profile,
                on_chain_signals={"whale_activity": 0.3, "network_health": 0.9},
            )

            # Create history state
            history_state = HistoryState(timestamp=time.time())

            # Perform calculation
            result = self.calculator.calculate_profit(market_data, history_state)

            return (
                f"✅ Profit calculation completed:\n"
                f"   Total score: {result.total_profit_score:.6f}\n"
                f"   Confidence: {result.confidence_score:.4f}\n"
                f"   Processing: {result.processing_mode.value}"
            )

        except Exception as e:
            logger.error(f"❌ Error in profit calculation: {e}")
            return f"❌ Calculation error: {e}"

    def explain(self, detail_level: str = "summary") -> str:
        """Explain the last profit calculation."""
        if not self.calculator:
            return "❌ Calculator not initialized. Run 'init' first."

        try:
            return self.calculator.explain_last_calculation(detail_level)

        except Exception as e:
            logger.error(f"❌ Error explaining calculation: {e}")
            return f"❌ Explanation error: {e}"

    def metrics(self) -> str:
        """Show calculation metrics and performance data."""
        if not self.calculator:
            return "❌ Calculator not initialized. Run 'init' first."

        try:
            metrics = self.calculator.get_calculation_metrics()

            result = "📊 CALCULATION METRICS\n"
            result += "=" * 40 + "\n"
            result += f"Total calculations    : {metrics['total_calculations']}\n"
            result += f"Total time           : {metrics['total_calculation_time']:.4f}s\n"
            result += f"Average time         : {metrics['avg_calculation_time']:.6f}s\n"
            result += f"Processing mode      : {metrics['processing_mode']}\n"
            result += f"Backend              : {metrics['backend']}\n"
            result += f"Error count          : {metrics['error_count']}\n"
            result += "\nPERFORMANCE METRICS:\n"
            result += f"  GPU operations      : {metrics['performance_metrics']['gpu_operations']}\n"
            result += f"  CPU operations      : {metrics['performance_metrics']['cpu_operations']}\n"
            result += f"  Fallback operations : {metrics['performance_metrics']['fallback_operations']}\n"
            result += "\nSTRATEGY PARAMETERS:\n"
            result += f"  Risk tolerance      : {metrics['strategy_params']['risk_tolerance']}\n"
            result += f"  Profit target       : {metrics['strategy_params']['profit_target']}\n"
            result += f"  Position size       : {metrics['strategy_params']['position_size']}\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error getting metrics: {e}")
            return f"❌ Metrics error: {e}"

    def test(self) -> str:
        """Run comprehensive test suite."""
        if not self.calculator:
            return "❌ Calculator not initialized. Run 'init' first."

        try:
            results = []

            # Test 1: Basic calculation
            market_data = create_sample_market_data()
            history_state = HistoryState(timestamp=time.time())
            result = self.calculator.calculate_profit(market_data, history_state)
            results.append(f"✅ Basic calculation: {result.total_profit_score:.6f}")

            # Test 2: Different calculation modes
            for mode_name in ["conservative", "balanced", "aggressive", "tensor_optimized"]:
                try:
                    from core.pure_profit_calculator import ProfitCalculationMode

                    mode = ProfitCalculationMode(mode_name)
                    result = self.calculator.calculate_profit(market_data, history_state, mode)
                    results.append(f"✅ {mode_name} mode: {result.total_profit_score:.6f}")
                except Exception as e:
                    results.append(f"❌ {mode_name} mode: {e}")

            # Test 3: Purity validation
            is_pure = self.calculator.validate_profit_purity(market_data, history_state)
            results.append(f"{'✅' if is_pure else '❌'} Purity validation: {'PASS' if is_pure else 'FAIL'}")

            # Test 4: Error handling (force CPU fallback)
            try:
                result = self.calculator.calculate_profit(market_data, history_state, force_cpu=True)
                results.append(f"✅ CPU fallback: {result.total_profit_score:.6f}")
            except Exception as e:
                results.append(f"❌ CPU fallback: {e}")

            return "🧪 TEST RESULTS\n" + "=" * 30 + "\n" + "\n".join(results)

        except Exception as e:
            logger.error(f"❌ Error in test suite: {e}")
            return f"❌ Test error: {e}"

    def validate(self) -> str:
        """Validate profit calculation purity."""
        if not self.calculator:
            return "❌ Calculator not initialized. Run 'init' first."

        try:
            # Test with sample data
            market_data = create_sample_market_data()
            history_state = HistoryState(timestamp=time.time())

            is_pure = self.calculator.validate_profit_purity(market_data, history_state)

            if is_pure:
                return (
                    "✅ Profit calculation is mathematically pure\n"
                    "   - No ZPE/ZBE systems detected\n"
                    "   - All calculations are deterministic\n"
                    "   - Results are within valid bounds"
                )
            else:
                return (
                    "❌ Profit calculation purity validation failed\n"
                    "   - Check for ZPE/ZBE system imports\n"
                    "   - Verify mathematical bounds\n"
                    "   - Review calculation logic"
                )

        except Exception as e:
            logger.error(f"❌ Error in validation: {e}")
            return f"❌ Validation error: {e}"

    def errors(self) -> str:
        """Show error summary and recent errors."""
        if not self.calculator:
            return "❌ Calculator not initialized. Run 'init' first."

        try:
            error_summary = self.calculator.get_error_summary()

            result = "🚨 ERROR SUMMARY\n"
            result += "=" * 30 + "\n"
            result += f"Total errors         : {error_summary['total_errors']}\n"
            result += f"Fallback usage       : {error_summary['fallback_usage']}\n"

            if error_summary['error_types']:
                result += "\nERROR TYPES:\n"
                for error_type, count in error_summary['error_types'].items():
                    result += f"  {error_type}: {count}\n"

            if error_summary['recent_errors']:
                result += "\nRECENT ERRORS:\n"
                for error in error_summary['recent_errors'][-5:]:  # Last 5
                    result += f"  {error.timestamp}: {error.error_type} - {error.error_message}\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error getting error summary: {e}")
            return f"❌ Error summary error: {e}"

    def reset(self) -> str:
        """Reset error log and metrics."""
        if not self.calculator:
            return "❌ Calculator not initialized. Run 'init' first."

        try:
            self.calculator.reset_error_log()
            return "✅ Error log and metrics reset"

        except Exception as e:
            logger.error(f"❌ Error resetting: {e}")
            return f"❌ Reset error: {e}"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pure Profit Calculator CLI - Mathematical Profit Calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                           # Initialize calculator
  %(prog)s flash                          # Display flash screen
  %(prog)s calculate 50000                # Calculate profit for BTC at $50k
  %(prog)s explain --full                 # Show detailed explanation
  %(prog)s metrics                        # Show performance metrics
  %(prog)s test                           # Run test suite
  %(prog)s validate                       # Validate calculation purity
  %(prog)s errors                         # Show error summary
  %(prog)s reset                          # Reset error log
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize profit calculator')
    init_parser.add_argument('--risk-tolerance', type=float, default=0.02, help='Risk tolerance (default: 0.02)')
    init_parser.add_argument('--profit-target', type=float, default=0.05, help='Profit target (default: 0.05)')
    init_parser.add_argument('--position-size', type=float, default=0.1, help='Position size (default: 0.1)')
    init_parser.add_argument(
        '--mode',
        choices=['gpu', 'cpu', 'hybrid', 'safe'],
        default='hybrid',
        help='Processing mode (default: hybrid)',
    )

    # Flash command
    subparsers.add_parser('flash', help='Display flash screen')

    # Calculate command
    calc_parser = subparsers.add_parser('calculate', help='Perform profit calculation')
    calc_parser.add_argument('btc_price', type=float, help='BTC price')
    calc_parser.add_argument('--eth-price', type=float, default=3000.0, help='ETH price')
    calc_parser.add_argument('--usdc-volume', type=float, default=1000000.0, help='USDC volume')
    calc_parser.add_argument('--volatility', type=float, default=0.02, help='Volatility')
    calc_parser.add_argument('--momentum', type=float, default=0.01, help='Momentum')
    calc_parser.add_argument('--volume-profile', type=float, default=0.8, help='Volume profile')

    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Explain last calculation')
    explain_parser.add_argument('--full', action='store_true', help='Show full details')

    # Metrics command
    subparsers.add_parser('metrics', help='Show calculation metrics')

    # Test command
    subparsers.add_parser('test', help='Run test suite')

    # Validate command
    subparsers.add_parser('validate', help='Validate calculation purity')

    # Errors command
    subparsers.add_parser('errors', help='Show error summary')

    # Reset command
    subparsers.add_parser('reset', help='Reset error log')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create CLI instance
    cli = PureProfitCalculatorCLI()

    try:
        if args.command == 'init':
            success = cli.init(args.risk_tolerance, args.profit_target, args.position_size, args.mode)
            sys.exit(0 if success else 1)

        elif args.command == 'flash':
            print(cli.flash())

        elif args.command == 'calculate':
            result = cli.calculate(
                args.btc_price,
                args.eth_price,
                args.usdc_volume,
                args.volatility,
                args.momentum,
                args.volume_profile,
            )
            print(result)

        elif args.command == 'explain':
            detail_level = "full" if args.full else "summary"
            print(cli.explain(detail_level))

        elif args.command == 'metrics':
            print(cli.metrics())

        elif args.command == 'test':
            print(cli.test())

        elif args.command == 'validate':
            print(cli.validate())

        elif args.command == 'errors':
            print(cli.errors())

        elif args.command == 'reset':
            print(cli.reset())

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
