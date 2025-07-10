#!/usr/bin/env python3
"""
Fill Handler CLI - Advanced Crypto Trading Fill Management Interface

Provides command-line interface for managing fill handling operations:
- Fill event processing and monitoring
- Partial fill handling and retry management
- Order state inspection and management
- Fill statistics and performance analysis
- State persistence and recovery
- Integration with secure exchange manager

Usage:
    python cli/fill_handler_cli.py [command] [options]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.fill_handler import FillEvent, FillHandler, FillStatus, OrderState, create_fill_handler
from core.secure_exchange_manager import ExchangeType, SecureExchangeManager
from utils.safe_print import error, info, safe_print, success, warn

logger = logging.getLogger(__name__)


class FillHandlerCLI:
    """Command-line interface for fill handler operations."""

    def __init__(self):
        """Initialize the CLI."""
        self.fill_handler: Optional[FillHandler] = None
        self.exchange_manager: Optional[SecureExchangeManager] = None

    async def initialize(self):
        """Initialize the fill handler and exchange manager."""
        try:
            # Initialize fill handler
            self.fill_handler = await create_fill_handler(
                {
                    'retry_config': {
                        'max_retries': 3,
                        'base_delay': 1.0,
                        'max_delay': 30.0,
                        'exponential_base': 2.0,
                        'jitter_factor': 0.1,
                    }
                }
            )

            # Initialize exchange manager
            self.exchange_manager = SecureExchangeManager()

            success("✅ Fill Handler CLI initialized")
            return True

        except Exception as e:
            error(f"❌ Initialization failed: {e}")
            return False

    def show_banner(self):
        """Display the CLI banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                🔧 FILL HANDLER CLI 🔧                        ║
║                                                              ║
║  Advanced Crypto Trading Fill Management                    ║
║  Partial Fills • Retries • Order State Management           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        safe_print(banner)

    def show_help(self):
        """Show help information."""
        help_text = """
📖 FILL HANDLER CLI COMMANDS

🔍 MONITORING:
  status                    Show fill handler status and statistics
  orders                    List all active orders
  order <order_id>          Show detailed order information
  fills                     Show recent fill events
  stats                     Show detailed statistics

⚙️ OPERATIONS:
  process-fill <file>       Process fill event from JSON file
  handle-partial <order_id> Handle partial fill for specific order
  retry <order_id>          Manually retry an order
  cancel <order_id>         Cancel an order

💾 PERSISTENCE:
  export <file>             Export fill handler state to file
  import <file>             Import fill handler state from file
  clear-completed           Clear completed orders older than 24h

🔗 INTEGRATION:
  exchange-status           Show exchange manager status
  test-integration          Test fill handler with exchange manager
  simulate-fill             Simulate fill event processing

📊 ANALYSIS:
  performance               Show performance analysis
  slippage-analysis         Analyze slippage patterns
  fee-analysis              Analyze fee patterns

❓ HELP:
  help                      Show this help message
  version                   Show version information

Examples:
  python cli/fill_handler_cli.py status
  python cli/fill_handler_cli.py process-fill sample_fill.json
  python cli/fill_handler_cli.py order 123456789
  python cli/fill_handler_cli.py export state_backup.json
        """
        safe_print(help_text)

    async def cmd_status(self):
        """Show fill handler status."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        info("📊 FILL HANDLER STATUS")
        info("=" * 40)

        stats = self.fill_handler.get_fill_statistics()

        info(f"Total Fills Processed: {stats['total_fills_processed']}")
        info(f"Total Retries: {stats['total_retries']}")
        info(f"Total Fees: {stats['total_fees']}")
        info(f"Active Orders: {stats['active_orders']}")
        info(f"Completed Orders: {stats['completed_orders']}")
        info(f"Partial Orders: {stats['partial_orders']}")
        info(f"Completion Rate: {stats['completion_rate']:.2f}%")

        # Show recent activity
        if self.fill_handler.fill_history:
            info("\n🕒 RECENT ACTIVITY:")
            recent_fills = self.fill_handler.fill_history[-5:]
            for fill in recent_fills:
                info(f"  {fill.symbol} {fill.amount} @ {fill.price} ({fill.timestamp})")

    async def cmd_orders(self):
        """List all active orders."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        info("📋 ACTIVE ORDERS")
        info("=" * 40)

        if not self.fill_handler.active_orders:
            info("No active orders")
            return

        for order_id, order in self.fill_handler.active_orders.items():
            status_icon = (
                "✅"
                if order.status == FillStatus.COMPLETE
                else (
                    "🔄" if order.status == FillStatus.PARTIAL else "⏳" if order.status == FillStatus.PENDING else "❌"
                )
            )

            info(f"{status_icon} {order_id}")
            info(f"  Symbol: {order.symbol}")
            info(f"  Side: {order.side}")
            info(f"  Type: {order.order_type}")
            info(f"  Progress: {order.fill_percentage:.1f}% ({order.filled_amount}/{order.original_amount})")
            info(f"  Average Price: {order.average_price}")
            info(f"  Status: {order.status.value}")
            info("")

    async def cmd_order(self, order_id: str):
        """Show detailed order information."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        order = self.fill_handler.get_order_state(order_id)
        if not order:
            error(f"❌ Order {order_id} not found")
            return

        info(f"📋 ORDER DETAILS: {order_id}")
        info("=" * 40)

        info(f"Symbol: {order.symbol}")
        info(f"Side: {order.side}")
        info(f"Type: {order.order_type}")
        info(f"Original Amount: {order.original_amount}")
        info(f"Filled Amount: {order.filled_amount}")
        info(f"Remaining Amount: {order.remaining_amount}")
        info(f"Average Price: {order.average_price}")
        info(f"Total Cost: {order.total_cost}")
        info(f"Total Fee: {order.total_fee}")
        info(f"Status: {order.status.value}")
        info(f"Fill Percentage: {order.fill_percentage:.2f}%")
        info(f"Retry Count: {order.retry_count}")
        info(f"Created: {order.created_at}")
        info(f"Updated: {order.updated_at}")

        if order.fills:
            info(f"\n🔍 FILL EVENTS ({len(order.fills)}):")
            for i, fill in enumerate(order.fills, 1):
                info(f"  {i}. {fill.trade_id}: {fill.amount} @ {fill.price} (fee: {fill.fee} {fill.fee_currency})")

    async def cmd_fills(self, limit: int = 10):
        """Show recent fill events."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        info(f"🕒 RECENT FILL EVENTS (last {limit})")
        info("=" * 40)

        if not self.fill_handler.fill_history:
            info("No fill events recorded")
            return

        recent_fills = self.fill_handler.fill_history[-limit:]
        for fill in recent_fills:
            info(f"📈 {fill.symbol} {fill.side.upper()} {fill.amount} @ {fill.price}")
            info(f"   Trade ID: {fill.trade_id}")
            info(f"   Order ID: {fill.order_id}")
            info(f"   Fee: {fill.fee} {fill.fee_currency}")
            info(f"   Time: {fill.timestamp}")
            info("")

    async def cmd_stats(self):
        """Show detailed statistics."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        info("📈 DETAILED STATISTICS")
        info("=" * 40)

        stats = self.fill_handler.get_fill_statistics()

        # Performance metrics
        info("🎯 PERFORMANCE METRICS:")
        info(f"  Completion Rate: {stats['completion_rate']:.2f}%")
        info(f"  Total Fills: {stats['total_fills_processed']}")
        info(f"  Total Retries: {stats['total_retries']}")

        # Financial metrics
        info("\n💰 FINANCIAL METRICS:")
        info(f"  Total Fees: {stats['total_fees']}")
        info(f"  Total Slippage: {stats['total_slippage']}")

        # Order metrics
        info("\n📊 ORDER METRICS:")
        info(f"  Active Orders: {stats['active_orders']}")
        info(f"  Completed Orders: {stats['completed_orders']}")
        info(f"  Partial Orders: {stats['partial_orders']}")

        # Retry analysis
        if self.fill_handler.retry_history:
            info("\n🔄 RETRY ANALYSIS:")
            total_retries = sum(len(retries) for retries in self.fill_handler.retry_history.values())
            info(f"  Total Retry Events: {total_retries}")
            info(f"  Orders with Retries: {len(self.fill_handler.retry_history)}")

    async def cmd_process_fill(self, file_path: str):
        """Process fill event from JSON file."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        try:
            with open(file_path, 'r') as f:
                fill_data = json.load(f)

            info(f"🔄 Processing fill from {file_path}")

            fill_event = await self.fill_handler.process_fill_event(fill_data)

            success(f"✅ Fill processed successfully")
            info(f"Trade ID: {fill_event.trade_id}")
            info(f"Amount: {fill_event.amount}")
            info(f"Price: {fill_event.price}")
            info(f"Fee: {fill_event.fee} {fill_event.fee_currency}")

        except FileNotFoundError:
            error(f"❌ File not found: {file_path}")
        except json.JSONDecodeError:
            error(f"❌ Invalid JSON in file: {file_path}")
        except Exception as e:
            error(f"❌ Error processing fill: {e}")

    async def cmd_handle_partial(self, order_id: str):
        """Handle partial fill for specific order."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        order = self.fill_handler.get_order_state(order_id)
        if not order:
            error(f"❌ Order {order_id} not found")
            return

        if not order.is_partial:
            error(f"❌ Order {order_id} is not partially filled")
            return

        info(f"🔄 Handling partial fill for order {order_id}")
        info(f"Current fill: {order.fill_percentage:.1f}%")

        # Simulate partial fill handling
        result = await self.fill_handler.handle_partial_fill(order_id, {"orderId": order_id, "status": "partial"})

        info(f"Result: {result['status']}")
        if "remaining_amount" in result:
            info(f"Remaining: {result['remaining_amount']}")

    async def cmd_retry(self, order_id: str):
        """Manually retry an order."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        order = self.fill_handler.get_order_state(order_id)
        if not order:
            error(f"❌ Order {order_id} not found")
            return

        if order.retry_count >= order.max_retries:
            error(f"❌ Order {order_id} has reached maximum retries")
            return

        info(f"🔄 Manually retrying order {order_id}")
        info(f"Current retry count: {order.retry_count}")

        # Increment retry count
        order.retry_count += 1
        order.updated_at = int(asyncio.get_event_loop().time() * 1000)

        success(f"✅ Order {order_id} retry count updated to {order.retry_count}")

    async def cmd_cancel(self, order_id: str):
        """Cancel an order."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        order = self.fill_handler.get_order_state(order_id)
        if not order:
            error(f"❌ Order {order_id} not found")
            return

        if order.status in [FillStatus.COMPLETE, FillStatus.CANCELLED, FillStatus.FAILED]:
            error(f"❌ Order {order_id} cannot be cancelled (status: {order.status.value})")
            return

        info(f"❌ Cancelling order {order_id}")

        order.status = FillStatus.CANCELLED
        order.updated_at = int(asyncio.get_event_loop().time() * 1000)

        success(f"✅ Order {order_id} cancelled")

    async def cmd_export(self, file_path: str):
        """Export fill handler state to file."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        try:
            state_data = self.fill_handler.export_state()

            with open(file_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            success(f"✅ State exported to {file_path}")
            info(f"Active orders: {len(state_data['active_orders'])}")
            info(f"Fill history: {len(state_data['fill_history'])}")

        except Exception as e:
            error(f"❌ Error exporting state: {e}")

    async def cmd_import(self, file_path: str):
        """Import fill handler state from file."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)

            self.fill_handler.import_state(state_data)

            success(f"✅ State imported from {file_path}")
            info(f"Active orders: {len(state_data['active_orders'])}")
            info(f"Fill history: {len(state_data['fill_history'])}")

        except FileNotFoundError:
            error(f"❌ File not found: {file_path}")
        except json.JSONDecodeError:
            error(f"❌ Invalid JSON in file: {file_path}")
        except Exception as e:
            error(f"❌ Error importing state: {e}")

    async def cmd_clear_completed(self):
        """Clear completed orders older than 24h."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        initial_count = len(self.fill_handler.active_orders)

        self.fill_handler.clear_completed_orders(max_age_hours=24)

        final_count = len(self.fill_handler.active_orders)
        cleared_count = initial_count - final_count

        success(f"✅ Cleared {cleared_count} completed orders")
        info(f"Remaining active orders: {final_count}")

    async def cmd_exchange_status(self):
        """Show exchange manager status."""
        if not self.exchange_manager:
            error("❌ Exchange manager not initialized")
            return

        info("🔗 EXCHANGE MANAGER STATUS")
        info("=" * 40)

        summary = self.exchange_manager.get_secure_summary()

        info(f"Exchanges Configured: {summary['exchanges_configured']}")
        info(f"Exchanges Connected: {summary['exchanges_connected']}")
        info(f"Exchanges Authenticated: {summary['exchanges_authenticated']}")
        info(f"Fill Handler Available: {summary['fill_handler_available']}")
        info(f"Secure Storage Available: {summary['secure_storage_available']}")
        info(f"CCXT Available: {summary['ccxt_available']}")

        if 'fill_statistics' in summary:
            info("\n📊 FILL HANDLER STATISTICS:")
            stats = summary['fill_statistics']
            info(f"  Total Fills: {stats['total_fills_processed']}")
            info(f"  Total Retries: {stats['total_retries']}")
            info(f"  Completion Rate: {stats['completion_rate']:.2f}%")

    async def cmd_test_integration(self):
        """Test fill handler with exchange manager."""
        if not self.fill_handler or not self.exchange_manager:
            error("❌ Fill handler or exchange manager not initialized")
            return

        info("🧪 TESTING INTEGRATION")
        info("=" * 40)

        try:
            # Test fill handler initialization in exchange manager
            await self.exchange_manager._initialize_fill_handler()

            if self.exchange_manager.fill_handler:
                success("✅ Fill handler successfully integrated with exchange manager")

                # Test fill statistics
                stats = await self.exchange_manager.get_fill_statistics()
                if "total_fills_processed" in stats:
                    success("✅ Fill statistics accessible through exchange manager")
                else:
                    warn("⚠️ Fill statistics not available through exchange manager")
            else:
                warn("⚠️ Fill handler not available in exchange manager")

        except Exception as e:
            error(f"❌ Integration test failed: {e}")

    async def cmd_simulate_fill(self):
        """Simulate fill event processing."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        info("🎭 SIMULATING FILL EVENT")
        info("=" * 40)

        # Create sample fill data
        sample_fill = {
            "orderId": "sim_test_123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "fills": [
                {
                    "tradeId": "sim_trade_456",
                    "qty": "0.001",
                    "price": "50000.00",
                    "commission": "0.000001",
                    "commissionAsset": "BTC",
                    "takerOrMaker": "taker",
                }
            ],
        }

        try:
            fill_event = await self.fill_handler.process_fill_event(sample_fill)

            success("✅ Fill simulation successful")
            info(f"Trade ID: {fill_event.trade_id}")
            info(f"Amount: {fill_event.amount}")
            info(f"Price: {fill_event.price}")
            info(f"Fee: {fill_event.fee} {fill_event.fee_currency}")

        except Exception as e:
            error(f"❌ Fill simulation failed: {e}")

    async def cmd_performance(self):
        """Show performance analysis."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        info("📈 PERFORMANCE ANALYSIS")
        info("=" * 40)

        stats = self.fill_handler.get_fill_statistics()

        # Calculate performance metrics
        total_orders = stats['active_orders'] + stats['completed_orders']
        if total_orders > 0:
            success_rate = (stats['completed_orders'] / total_orders) * 100
            info(f"Success Rate: {success_rate:.2f}%")

        # Retry efficiency
        if stats['total_fills_processed'] > 0:
            retry_rate = (stats['total_retries'] / stats['total_fills_processed']) * 100
            info(f"Retry Rate: {retry_rate:.2f}%")

        # Fee analysis
        if stats['total_fees'] != '0':
            info(f"Average Fee per Fill: {float(stats['total_fees']) / stats['total_fills_processed']:.6f}")

    async def cmd_slippage_analysis(self):
        """Analyze slippage patterns."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        info("📊 SLIPPAGE ANALYSIS")
        info("=" * 40)

        # This would require implementing slippage calculation
        # For now, show basic information
        info("Slippage analysis requires implementing slippage calculation logic")
        info("This would compare fill prices against expected prices")
        info("Total slippage tracked: " + self.fill_handler.get_fill_statistics()['total_slippage'])

    async def cmd_fee_analysis(self):
        """Analyze fee patterns."""
        if not self.fill_handler:
            error("❌ Fill handler not initialized")
            return

        info("💰 FEE ANALYSIS")
        info("=" * 40)

        stats = self.fill_handler.get_fill_statistics()
        total_fees = float(stats['total_fees'])
        total_fills = stats['total_fills_processed']

        if total_fills > 0:
            avg_fee = total_fees / total_fills
            info(f"Total Fees: {total_fees}")
            info(f"Total Fills: {total_fills}")
            info(f"Average Fee per Fill: {avg_fee:.6f}")

            # Analyze by currency if available
            if self.fill_handler.fill_history:
                fee_by_currency = {}
                for fill in self.fill_handler.fill_history:
                    currency = fill.fee_currency
                    if currency not in fee_by_currency:
                        fee_by_currency[currency] = 0
                    fee_by_currency[currency] += float(fill.fee)

                info("\nFees by Currency:")
                for currency, amount in fee_by_currency.items():
                    info(f"  {currency}: {amount}")
        else:
            info("No fills processed yet")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Fill Handler CLI - Advanced Crypto Trading Fill Management")
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("args", nargs="*", help="Command arguments")

    args = parser.parse_args()

    cli = FillHandlerCLI()
    cli.show_banner()

    # Initialize
    if not await cli.initialize():
        sys.exit(1)

    # Execute command
    try:
        if args.command == "status":
            await cli.cmd_status()
        elif args.command == "orders":
            await cli.cmd_orders()
        elif args.command == "order":
            if not args.args:
                error("❌ Order ID required")
                return
            await cli.cmd_order(args.args[0])
        elif args.command == "fills":
            limit = int(args.args[0]) if args.args else 10
            await cli.cmd_fills(limit)
        elif args.command == "stats":
            await cli.cmd_stats()
        elif args.command == "process-fill":
            if not args.args:
                error("❌ File path required")
                return
            await cli.cmd_process_fill(args.args[0])
        elif args.command == "handle-partial":
            if not args.args:
                error("❌ Order ID required")
                return
            await cli.cmd_handle_partial(args.args[0])
        elif args.command == "retry":
            if not args.args:
                error("❌ Order ID required")
                return
            await cli.cmd_retry(args.args[0])
        elif args.command == "cancel":
            if not args.args:
                error("❌ Order ID required")
                return
            await cli.cmd_cancel(args.args[0])
        elif args.command == "export":
            if not args.args:
                error("❌ File path required")
                return
            await cli.cmd_export(args.args[0])
        elif args.command == "import":
            if not args.args:
                error("❌ File path required")
                return
            await cli.cmd_import(args.args[0])
        elif args.command == "clear-completed":
            await cli.cmd_clear_completed()
        elif args.command == "exchange-status":
            await cli.cmd_exchange_status()
        elif args.command == "test-integration":
            await cli.cmd_test_integration()
        elif args.command == "simulate-fill":
            await cli.cmd_simulate_fill()
        elif args.command == "performance":
            await cli.cmd_performance()
        elif args.command == "slippage-analysis":
            await cli.cmd_slippage_analysis()
        elif args.command == "fee-analysis":
            await cli.cmd_fee_analysis()
        elif args.command == "help":
            cli.show_help()
        elif args.command == "version":
            info("Fill Handler CLI v1.0.0")
        else:
            error(f"❌ Unknown command: {args.command}")
            cli.show_help()

    except KeyboardInterrupt:
        info("\n👋 Goodbye!")
    except Exception as e:
        error(f"❌ Error executing command: {e}")


if __name__ == "__main__":
    asyncio.run(main())
