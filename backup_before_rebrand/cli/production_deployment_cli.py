#!/usr/bin/env python3
"""
Production Deployment CLI - Enterprise Deployment Management

Provides CLI commands for:
- Environment variable validation
- System health checks
- Deployment readiness verification
- Production deployment execution
- Security configuration validation
- Monitoring and alerting setup

Security Features:
- Validates all required environment variables
- Checks for proper security configurations
- Ensures production-safe settings
- Comprehensive deployment reporting

Usage:
    python cli/production_deployment_cli.py validate
    python cli/production_deployment_cli.py health
    python cli/production_deployment_cli.py deploy
    python cli/production_deployment_cli.py check-all
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

from core.production_deployment_manager import ProductionDeploymentManager, get_production_manager

logger = logging.getLogger(__name__)


class ProductionDeploymentCLI:
    """CLI interface for production deployment management."""

    def __init__(self):
        self.manager = get_production_manager()

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def validate_environment(self) -> str:
        """Validate production environment configuration."""
        try:
            validation = self.manager.validate_environment()

            result = "🔍 PRODUCTION ENVIRONMENT VALIDATION\n"
            result += "=" * 50 + "\n"

            if validation.is_valid:
                result += "✅ Environment validation PASSED\n\n"
            else:
                result += "❌ Environment validation FAILED\n\n"

            # Show missing variables
            if validation.missing_vars:
                result += "📋 MISSING ENVIRONMENT VARIABLES:\n"
                for var in validation.missing_vars:
                    result += f"  • {var}\n"
                result += "\n"

            # Show security issues
            if validation.security_issues:
                result += "🚨 SECURITY ISSUES:\n"
                for issue in validation.security_issues:
                    result += f"  • {issue}\n"
                result += "\n"

            # Show warnings
            if validation.warnings:
                result += "⚠️ WARNINGS:\n"
                for warning in validation.warnings:
                    result += f"  • {warning}\n"
                result += "\n"

            # Show recommendations
            if validation.recommendations:
                result += "💡 RECOMMENDATIONS:\n"
                for rec in validation.recommendations:
                    result += f"  • {rec}\n"
                result += "\n"

            # Environment info
            result += f"🌍 Environment: {self.manager.environment.value}\n"
            result += f"🔒 Security Level: {self.manager.config.security_level.value}\n"
            result += f"📝 Log Level: {self.manager.config.log_level}\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error validating environment: {e}")
            return f"❌ Validation error: {e}"

    def check_health(self) -> str:
        """Check system health and resources."""
        try:
            health = self.manager.check_system_health()

            result = "🏥 SYSTEM HEALTH CHECK\n"
            result += "=" * 30 + "\n"

            # Overall health
            health_icon = (
                "🟢" if health.overall_health == "healthy" else "🟡" if health.overall_health == "degraded" else "🔴"
            )
            result += f"{health_icon} Overall Health: {health.overall_health.upper()}\n\n"

            # Resource usage
            result += "📊 RESOURCE USAGE:\n"
            result += f"  CPU: {health.cpu_usage:.1f}%\n"
            result += f"  Memory: {health.memory_usage:.1f}%\n"
            result += f"  Disk: {health.disk_usage:.1f}%\n"
            result += f"  Network: {health.network_status}\n\n"

            # Service status
            result += "🔧 SERVICE STATUS:\n"
            for service, status in health.services_status.items():
                status_icon = "✅" if status in ["available", "connected", "running"] else "❌"
                result += f"  {status_icon} {service}: {status}\n"
            result += "\n"

            # Issues
            if health.issues:
                result += "⚠️ ISSUES DETECTED:\n"
                for issue in health.issues:
                    result += f"  • {issue}\n"
                result += "\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error checking health: {e}")
            return f"❌ Health check error: {e}"

    def validate_exchanges(self) -> str:
        """Validate exchange API credentials."""
        try:
            exchange_validation = self.manager.validate_exchange_credentials()

            result = "🔐 EXCHANGE CREDENTIALS VALIDATION\n"
            result += "=" * 40 + "\n"

            all_valid = True
            for exchange, is_valid in exchange_validation.items():
                status_icon = "✅" if is_valid else "❌"
                result += f"{status_icon} {exchange.upper()}: {'Valid' if is_valid else 'Invalid'}\n"
                if not is_valid:
                    all_valid = False

            result += "\n"

            if all_valid:
                result += "✅ All configured exchanges are valid\n"
            else:
                result += "❌ Some exchanges have validation issues\n"
                result += "💡 Run 'exchange status' for detailed information\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error validating exchanges: {e}")
            return f"❌ Exchange validation error: {e}"

    def run_deployment_checks(self) -> str:
        """Run comprehensive deployment readiness checks."""
        try:
            results = self.manager.run_deployment_checks()

            result = "🚀 DEPLOYMENT READINESS CHECK\n"
            result += "=" * 40 + "\n"

            # Overall status
            if results["deployment_ready"]:
                result += "✅ DEPLOYMENT READY\n\n"
            else:
                result += "❌ DEPLOYMENT NOT READY\n\n"

            # Environment
            env_check = results["checks"]["environment"]
            result += "🌍 ENVIRONMENT:\n"
            result += f"  Status: {'✅ PASSED' if env_check['passed'] else '❌ FAILED'}\n"
            if env_check['missing_vars']:
                result += f"  Missing Variables: {len(env_check['missing_vars'])}\n"
            if env_check['security_issues']:
                result += f"  Security Issues: {len(env_check['security_issues'])}\n"
            if env_check['warnings']:
                result += f"  Warnings: {len(env_check['warnings'])}\n"
            result += "\n"

            # System Health
            health_check = results["checks"]["system_health"]
            result += "🏥 SYSTEM HEALTH:\n"
            result += f"  Status: {health_check['overall_health'].upper()}\n"
            result += f"  CPU: {health_check['cpu_usage']:.1f}%\n"
            result += f"  Memory: {health_check['memory_usage']:.1f}%\n"
            result += f"  Disk: {health_check['disk_usage']:.1f}%\n"
            result += f"  Network: {health_check['network_status']}\n"
            if health_check['issues']:
                result += f"  Issues: {len(health_check['issues'])}\n"
            result += "\n"

            # Exchanges
            exchange_check = results["checks"]["exchanges"]
            result += "🔐 EXCHANGES:\n"
            valid_exchanges = sum(1 for valid in exchange_check.values() if valid)
            result += f"  Valid: {valid_exchanges}/{len(exchange_check)}\n"
            for exchange, is_valid in exchange_check.items():
                status_icon = "✅" if is_valid else "❌"
                result += f"    {status_icon} {exchange.upper()}\n"
            result += "\n"

            # Recommendations
            if env_check['recommendations']:
                result += "💡 RECOMMENDATIONS:\n"
                for rec in env_check['recommendations']:
                    result += f"  • {rec}\n"
                result += "\n"

            # Report location
            result += f"📊 Detailed report saved to: logs/deployment_report_{int(results['timestamp'])}.json\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error running deployment checks: {e}")
            return f"❌ Deployment check error: {e}"

    def deploy_to_production(self) -> str:
        """Deploy Schwabot to production environment."""
        try:
            result = "🚀 STARTING PRODUCTION DEPLOYMENT\n"
            result += "=" * 40 + "\n"

            # Run deployment checks first
            result += "🔍 Running deployment checks...\n"
            checks = self.manager.run_deployment_checks()

            if not checks["deployment_ready"]:
                result += "❌ Deployment checks failed - cannot proceed\n"
                result += "💡 Fix issues and run 'check-all' again\n"
                return result

            result += "✅ Deployment checks passed\n\n"

            # Execute deployment
            result += "🚀 Executing production deployment...\n"
            success = self.manager.deploy_to_production()

            if success:
                result += "✅ Production deployment completed successfully!\n\n"
                result += "🎉 Schwabot is now running in production mode\n"
                result += "📊 Monitor logs at: logs/schwabot.log\n"
                result += "🔍 Check status with: python schwabot_unified_cli.py monitor --status\n"
            else:
                result += "❌ Production deployment failed\n"
                result += "📋 Check logs for detailed error information\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error during deployment: {e}")
            return f"❌ Deployment error: {e}"

    def show_environment_info(self) -> str:
        """Show current environment information."""
        try:
            result = "🌍 ENVIRONMENT INFORMATION\n"
            result += "=" * 30 + "\n"

            result += f"Environment: {self.manager.environment.value}\n"
            result += f"Security Level: {self.manager.config.security_level.value}\n"
            result += f"Log Level: {self.manager.config.log_level}\n"
            result += f"Max Concurrent Trades: {self.manager.config.max_concurrent_trades}\n"
            result += f"Data Retention Days: {self.manager.config.data_retention_days}\n"
            result += f"Enable Monitoring: {self.manager.config.enable_monitoring}\n"
            result += f"Enable Backups: {self.manager.config.enable_backups}\n"
            result += f"Enable SSL: {self.manager.config.enable_ssl}\n"
            result += f"Enable Rate Limiting: {self.manager.config.enable_rate_limiting}\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error showing environment info: {e}")
            return f"❌ Environment info error: {e}"

    def generate_deployment_script(self) -> str:
        """Generate deployment script for the current environment."""
        try:
            script_content = self._create_deployment_script()

            script_file = "deploy_to_production.sh"
            with open(script_file, 'w') as f:
                f.write(script_content)

            result = f"📜 DEPLOYMENT SCRIPT GENERATED\n"
            result += "=" * 35 + "\n"
            result += f"Script saved to: {script_file}\n"
            result += f"Make it executable: chmod +x {script_file}\n"
            result += f"Run deployment: ./{script_file}\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error generating deployment script: {e}")
            return f"❌ Script generation error: {e}"

    def _create_deployment_script(self) -> str:
        """Create deployment script content."""
        return f"""#!/bin/bash
# Schwabot Production Deployment Script
# Generated on: $(date)

set -e  # Exit on any error

echo "🚀 Starting Schwabot Production Deployment..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "💡 Copy config/production.env.template to .env and configure it"
    exit 1
fi

# Load environment variables
source .env

# Validate environment
echo "🔍 Validating environment..."
python cli/production_deployment_cli.py validate

# Check system health
echo "🏥 Checking system health..."
python cli/production_deployment_cli.py health

# Run deployment checks
echo "🚀 Running deployment checks..."
python cli/production_deployment_cli.py check-all

# Deploy to production
echo "🚀 Deploying to production..."
python cli/production_deployment_cli.py deploy

echo "✅ Deployment completed successfully!"
echo "📊 Monitor logs: tail -f logs/schwabot.log"
echo "🔍 Check status: python schwabot_unified_cli.py monitor --status"
"""


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Production Deployment CLI - Enterprise Deployment Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate                    # Validate environment configuration
  %(prog)s health                      # Check system health
  %(prog)s check-all                   # Run all deployment checks
  %(prog)s deploy                      # Deploy to production
  %(prog)s info                        # Show environment information
  %(prog)s generate-script             # Generate deployment script

Environment Setup:
  1. Copy config/production.env.template to .env
  2. Configure your environment variables in .env
  3. Run: %(prog)s validate
  4. Run: %(prog)s check-all
  5. Run: %(prog)s deploy
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Validate command
    subparsers.add_parser('validate', help='Validate environment configuration')

    # Health command
    subparsers.add_parser('health', help='Check system health and resources')

    # Validate exchanges command
    subparsers.add_parser('exchanges', help='Validate exchange API credentials')

    # Check all command
    subparsers.add_parser('check-all', help='Run comprehensive deployment checks')

    # Deploy command
    subparsers.add_parser('deploy', help='Deploy to production environment')

    # Info command
    subparsers.add_parser('info', help='Show environment information')

    # Generate script command
    subparsers.add_parser('generate-script', help='Generate deployment script')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create CLI instance
    cli = ProductionDeploymentCLI()

    try:
        if args.command == 'validate':
            print(cli.validate_environment())

        elif args.command == 'health':
            print(cli.check_health())

        elif args.command == 'exchanges':
            print(cli.validate_exchanges())

        elif args.command == 'check-all':
            print(cli.run_deployment_checks())

        elif args.command == 'deploy':
            print(cli.deploy_to_production())

        elif args.command == 'info':
            print(cli.show_environment_info())

        elif args.command == 'generate-script':
            print(cli.generate_deployment_script())

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
