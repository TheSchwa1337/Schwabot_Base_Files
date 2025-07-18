#!/usr/bin/env python3
"""
Secure Exchange CLI - Professional API Key & Exchange Management

Provides secure CLI commands for:
- Setting up exchange credentials
- Validating connections and authentication
- Checking trading readiness
- Executing test trades
- Managing multiple exchanges

Security Features:
- Never displays actual secret keys
- Validates credentials before use
- Clear labeling of public vs private keys
- Environment variable support
- Encrypted local storage

Usage:
    python cli/secure_exchange_cli.py setup binance
    python cli/secure_exchange_cli.py status
    python cli/secure_exchange_cli.py validate
    python cli/secure_exchange_cli.py test-trade BTC/USDT 0.001
"""

import argparse
import getpass
import logging
import sys
from pathlib import Path
from typing import Optional

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

from core.secure_exchange_manager import ExchangeType, SecureExchangeManager, get_exchange_manager

logger = logging.getLogger(__name__)


class SecureExchangeCLI:
    """CLI interface for secure exchange management."""

    def __init__(self):
        self.manager = get_exchange_manager()

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def setup_exchange(
        self,
        exchange_name: str,
        api_key: str = None,
        secret: str = None,
        passphrase: str = None,
        sandbox: bool = True,
    ) -> str:
        """Setup exchange credentials securely."""
        try:
            # Validate exchange name
            try:
                exchange = ExchangeType(exchange_name.lower())
            except ValueError:
                return f"❌ Unsupported exchange: {exchange_name}. Supported: {[e.value for e in ExchangeType]}"

            # Check if already configured
            if exchange in self.manager.exchanges:
                return f"⚠️ {exchange.value} already configured. Use 'reconfigure' to update."

            # Get credentials if not provided
            if not api_key:
                print(f"\n🔧 Setting up {exchange.value} exchange...")
                print("📋 Enter your API key (public - safe to display):")
                api_key = input("API Key: ").strip()

            if not secret:
                print("🔐 Enter your secret key (private - will be masked):")
                secret = getpass.getpass("Secret: ").strip()

            # Get passphrase for exchanges that need it
            if exchange.value in ['coinbase', 'okx'] and not passphrase:
                print("🔐 Enter your passphrase (if required):")
                passphrase = getpass.getpass("Passphrase (optional): ").strip()
                if not passphrase:
                    passphrase = None

            # Validate inputs
            if not api_key or not secret:
                return "❌ API key and secret are required"

            # Setup exchange
            success = self.manager.setup_exchange(
                exchange=exchange,
                api_key=api_key,
                secret=secret,
                passphrase=passphrase,
                sandbox=sandbox,
            )

            if success:
                return f"✅ {exchange.value} setup successful and connected"
            else:
                return f"❌ {exchange.value} setup failed - check credentials and try again"

        except Exception as e:
            logger.error(f"❌ Error setting up exchange: {e}")
            return f"❌ Setup error: {e}"

    def setup_from_env(self, exchange_name: str) -> str:
        """Setup exchange from environment variables."""
        try:
            success = self.manager._load_from_environment(ExchangeType(exchange_name.lower()))
            if success:
                return f"✅ {exchange_name} loaded from environment variables"
            else:
                return f"❌ {exchange_name} not found in environment variables"
        except Exception as e:
            return f"❌ Error loading from environment: {e}"

    def status(self) -> str:
        """Show exchange status and configuration."""
        try:
            summary = self.manager.get_secure_summary()

            result = "🔐 SECURE EXCHANGE STATUS\n"
            result += "=" * 40 + "\n"
            result += f"Total exchanges configured: {summary['total_exchanges']}\n"
            result += f"Connected exchanges: {summary['connected_exchanges']}\n"
            result += f"Ready for trading: {summary['trading_ready']}\n\n"

            if summary['exchanges']:
                result += "📊 EXCHANGE DETAILS:\n"
                for exchange_name, status in summary['exchanges'].items():
                    result += f"\n{exchange_name.upper()}:\n"
                    for key, value in status.items():
                        if key == 'error' and value:
                            result += f"  ❌ {key}: {value}\n"
                        elif key in [
                            'connected',
                            'authenticated',
                            'trading_enabled',
                            'balance_available',
                        ]:
                            status_icon = "✅" if value else "❌"
                            result += f"  {status_icon} {key}: {value}\n"
                        else:
                            result += f"  📋 {key}: {value}\n"
            else:
                result += "⚠️ No exchanges configured\n"
                result += "\nTo setup an exchange:\n"
                result += "  python cli/secure_exchange_cli.py setup <exchange_name>\n"
                result += "  or set environment variables:\n"
                result += "  export BINANCE_API_KEY=\"your_key\"\n"
                result += "  export BINANCE_API_SECRET=\"your_secret\"\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return f"❌ Status error: {e}"

    def validate(self) -> str:
        """Validate trading system readiness."""
        try:
            is_ready, issues = self.manager.validate_trading_ready()

            result = "🔍 TRADING SYSTEM VALIDATION\n"
            result += "=" * 40 + "\n"

            if is_ready:
                result += "✅ Trading system is ready!\n"
                available_exchanges = self.manager.get_available_exchanges()
                result += f"📊 Available exchanges: {[e.value for e in available_exchanges]}\n"
            else:
                result += "❌ Trading system is not ready\n"
                result += "📋 Issues found:\n"
                for issue in issues:
                    result += f"  • {issue}\n"

                result += "\n🔧 To fix issues:\n"
                result += "  1. Setup exchange credentials\n"
                result += "  2. Ensure CCXT is installed: pip install ccxt\n"
                result += "  3. Check network connectivity\n"
                result += "  4. Verify API permissions\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error validating system: {e}")
            return f"❌ Validation error: {e}"

    def test_trade(self, symbol: str, amount: float, exchange_name: str = "binance") -> str:
        """Execute a test trade (sandbox only)."""
        try:
            # Validate exchange
            try:
                exchange = ExchangeType(exchange_name.lower())
            except ValueError:
                return f"❌ Unsupported exchange: {exchange_name}"

            # Check if exchange is ready
            if exchange not in self.manager.get_available_exchanges():
                return f"❌ {exchange.value} not ready for trading. Run 'validate' to check status."

            # Execute test trade
            result = self.manager.execute_trade(
                exchange=exchange, symbol=symbol, side="buy", amount=amount, order_type="market"
            )

            if result.get("success"):
                result_str = f"✅ Test trade executed successfully!\n"
                result_str += f"📋 Order ID: {result.get('order_id')}\n"
                result_str += f"📊 Symbol: {result.get('symbol')}\n"
                result_str += f"💰 Amount: {result.get('amount')}\n"
                result_str += f"📈 Status: {result.get('status')}\n"
                if result.get('cost'):
                    result_str += f"💵 Cost: {result.get('cost')}\n"
                if result.get('fee'):
                    result_str += f"💸 Fee: {result.get('fee')}\n"
                return result_str
            else:
                return f"❌ Test trade failed: {result.get('error')}"

        except Exception as e:
            logger.error(f"❌ Error executing test trade: {e}")
            return f"❌ Test trade error: {e}"

    def get_balance(self, exchange_name: str = "binance", currency: str = None) -> str:
        """Get account balance."""
        try:
            exchange = ExchangeType(exchange_name.lower())
            balance = self.manager.get_balance(exchange, currency)

            if "error" in balance:
                return f"❌ Balance error: {balance['error']}"

            result = f"💰 {exchange.value.upper()} BALANCE\n"
            result += "=" * 30 + "\n"

            if currency:
                result += f"Currency: {balance['currency']}\n"
                result += f"Free: {balance['free']}\n"
                result += f"Used: {balance['used']}\n"
                result += f"Total: {balance['total']}\n"
            else:
                balances = balance.get('balances', {})
                if balances:
                    for curr, amount in balances.items():
                        result += f"{curr}: {amount}\n"
                else:
                    result += "No balances found\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error getting balance: {e}")
            return f"❌ Balance error: {e}"

    def list_exchanges(self) -> str:
        """List all supported exchanges."""
        result = "📋 SUPPORTED EXCHANGES\n"
        result += "=" * 25 + "\n"

        for exchange in ExchangeType:
            result += f"• {exchange.value}\n"

        result += "\n🔧 To setup an exchange:\n"
        result += "  python cli/secure_exchange_cli.py setup <exchange_name>\n"
        result += "\n🔐 Environment variables:\n"
        result += "  <EXCHANGE>_API_KEY=your_public_key\n"
        result += "  <EXCHANGE>_API_SECRET=your_secret_key\n"
        result += "  <EXCHANGE>_PASSPHRASE=your_passphrase (if required)\n"

        return result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Secure Exchange CLI - Professional API Key & Exchange Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup binance                    # Interactive setup
  %(prog)s setup-from-env binance           # Setup from environment variables
  %(prog)s status                           # Show exchange status
  %(prog)s validate                         # Validate trading readiness
  %(prog)s test-trade BTC/USDT 0.001        # Execute test trade
  %(prog)s balance binance                  # Get account balance
  %(prog)s list-exchanges                   # List supported exchanges

Environment Variables:
  BINANCE_API_KEY=your_public_key
  BINANCE_API_SECRET=your_secret_key
  COINBASE_API_KEY=your_public_key
  COINBASE_API_SECRET=your_secret_key
  KRAKEN_API_KEY=your_public_key
  KRAKEN_API_SECRET=your_secret_key
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup exchange credentials')
    setup_parser.add_argument('exchange', help='Exchange name (e.g., binance, coinbase)')
    setup_parser.add_argument('--api-key', help='API key (optional - will prompt if not provided)')
    setup_parser.add_argument('--secret', help='Secret key (optional - will prompt if not provided)')
    setup_parser.add_argument('--passphrase', help='Passphrase (optional)')
    setup_parser.add_argument('--live', action='store_true', help='Use live trading (default: sandbox)')

    # Setup from env command
    env_parser = subparsers.add_parser('setup-from-env', help='Setup from environment variables')
    env_parser.add_argument('exchange', help='Exchange name')

    # Status command
    subparsers.add_parser('status', help='Show exchange status')

    # Validate command
    subparsers.add_parser('validate', help='Validate trading system readiness')

    # Test trade command
    trade_parser = subparsers.add_parser('test-trade', help='Execute test trade')
    trade_parser.add_argument('symbol', help='Trading symbol (e.g., BTC/USDT)')
    trade_parser.add_argument('amount', type=float, help='Trade amount')
    trade_parser.add_argument('--exchange', default='binance', help='Exchange to use')

    # Balance command
    balance_parser = subparsers.add_parser('balance', help='Get account balance')
    balance_parser.add_argument('--exchange', default='binance', help='Exchange name')
    balance_parser.add_argument('--currency', help='Specific currency (optional)')

    # List exchanges command
    subparsers.add_parser('list-exchanges', help='List supported exchanges')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create CLI instance
    cli = SecureExchangeCLI()

    try:
        if args.command == 'setup':
            result = cli.setup_exchange(
                args.exchange, args.api_key, args.secret, args.passphrase, sandbox=not args.live
            )
            print(result)

        elif args.command == 'setup-from-env':
            result = cli.setup_from_env(args.exchange)
            print(result)

        elif args.command == 'status':
            print(cli.status())

        elif args.command == 'validate':
            print(cli.validate())

        elif args.command == 'test-trade':
            result = cli.test_trade(args.symbol, args.amount, args.exchange)
            print(result)

        elif args.command == 'balance':
            result = cli.get_balance(args.exchange, args.currency)
            print(result)

        elif args.command == 'list-exchanges':
            print(cli.list_exchanges())

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
