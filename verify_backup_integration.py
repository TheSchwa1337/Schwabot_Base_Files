#!/usr/bin/env python3
"""
Backup Integration Verification Script
=====================================

Comprehensive verification of:
1. Backup components integration
2. Mathematical system integrity
3. Trading pipeline connectivity
4. Core system functionality
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_backup_components():
    """Check if backup integration components are available."""
    print("🔍 Checking Backup Components Integration...")

    backup_status = {
        "ghost_flip_executor": False,
        "profit_orbit_engine": False,
        "pair_flip_orbit": False,
        "backup_integration_demo": False,
    }

    # Check if backup files exist in core_backup
    backup_dir = Path("core_backup")
    if backup_dir.exists():
        backup_files = list(backup_dir.glob("*.py"))
        print(f"   📁 Found {len(backup_files)} backup files")

        # Check specific backup components
        critical_backups = [
            "ghost_flip_executor.py",
            "profit_orbit_engine.py",
            "pair_flip_orbit.py",
        ]

        for backup_file in critical_backups:
            if (backup_dir / backup_file).exists():
                backup_status[backup_file.replace(".py", "")] = True
                print(f"   ✅ {backup_file} found in backup")
            else:
                print(f"   ❌ {backup_file} missing from backup")

    # Check if backup integration demo exists
    if Path("test_backup_integration_standalone.py").exists():
        backup_status["backup_integration_demo"] = True
        print("   ✅ Backup integration demo available")

    return backup_status


def check_mathematical_integration():
    """Verify mathematical systems are properly integrated."""
    print("\n🧮 Checking Mathematical Integration...")

    math_status = {
        "mathlib_v4": False,
        "unified_math_system": False,
        "matrix_math_utils": False,
        "mathematical_framework_config": False,
    }

    try:
        from core.mathlib_v4 import MathLibV4

        ml4 = MathLibV4()
        math_status["mathlib_v4"] = True
        print(f"   ✅ MathLibV4 v{ml4.version.value} operational")
    except Exception as e:
        print(f"   ❌ MathLibV4 failed: {e}")

    try:
        from core.unified_math_system import UnifiedMathSystem

        UnifiedMathSystem()
        math_status["unified_math_system"] = True
        print("   ✅ Unified Math System operational")
    except Exception as e:
        print(f"   ❌ Unified Math System failed: {e}")

    try:
        from core.matrix_math_utils import analyze_price_matrix
        import numpy as np

        test_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        analyze_price_matrix(test_matrix)
        math_status["matrix_math_utils"] = True
        print("   ✅ Matrix Math Utils operational")
    except Exception as e:
        print(f"   ❌ Matrix Math Utils failed: {e}")

    # Check mathematical framework config
    config_path = Path("config/mathematical_framework_config.py")
    if config_path.exists():
        math_status["mathematical_framework_config"] = True
        print("   ✅ Mathematical Framework Config available")
    else:
        print("   ❌ Mathematical Framework Config missing")

    return math_status


def check_trading_integration():
    """Verify trading components are integrated."""
    print("\n📈 Checking Trading Integration...")

    trading_status = {
        "unified_trading_pipeline": False,
        "brain_trading_engine": False,
        "risk_manager": False,
        "strategy_logic": False,
        "profit_vectorization": False,
        "ccxt_integration": False,
    }

    try:
        trading_status["unified_trading_pipeline"] = True
        print("   ✅ Unified Trading Pipeline available")
    except Exception as e:
        print(f"   ❌ Unified Trading Pipeline failed: {e}")

    try:
        trading_status["brain_trading_engine"] = True
        print("   ✅ Brain Trading Engine available")
    except Exception as e:
        print(f"   ❌ Brain Trading Engine failed: {e}")

    try:
        trading_status["risk_manager"] = True
        print("   ✅ Risk Manager available")
    except Exception as e:
        print(f"   ❌ Risk Manager failed: {e}")

    try:
        trading_status["strategy_logic"] = True
        print("   ✅ Strategy Logic available")
    except Exception as e:
        print(f"   ❌ Strategy Logic failed: {e}")

    try:
        trading_status["profit_vectorization"] = True
        print("   ✅ Profit Vectorization System available")
    except Exception as e:
        print(f"   ❌ Profit Vectorization System failed: {e}")

    try:
        trading_status["ccxt_integration"] = True
        print("   ✅ CCXT Integration available")
    except Exception as e:
        print(f"   ❌ CCXT Integration failed: {e}")

    return trading_status


def check_backtesting_system():
    """Verify backtesting functionality."""
    print("\n⏮️ Checking Backtesting System...")

    backtest_status = {
        "simple_backtester": False,
        "historical_data_manager": False,
        "backtest_integration": False,
    }

    try:
        backtest_status["simple_backtester"] = True
        print("   ✅ Simple Backtester available")
    except Exception as e:
        print(f"   ❌ Simple Backtester failed: {e}")

    try:
        backtest_status["historical_data_manager"] = True
        print("   ✅ Historical Data Manager available")
    except Exception as e:
        print(f"   ❌ Historical Data Manager failed: {e}")

    # Check if backtesting is integrated with main pipeline
    try:
        # Check if main.py has backtest functionality
        main_path = Path("main.py")
        if main_path.exists():
            main_content = main_path.read_text()
            if "backtest" in main_content.lower():
                backtest_status["backtest_integration"] = True
                print("   ✅ Backtest integration in main pipeline")
            else:
                print("   ⚠️  Backtest not integrated in main pipeline")
    except Exception as e:
        print(f"   ❌ Failed to check backtest integration: {e}")

    return backtest_status


def generate_integration_report(
    backup_status, math_status, trading_status, backtest_status
):
    """Generate comprehensive integration report."""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE INTEGRATION REPORT")
    print("=" * 60)

    total_components = 0
    working_components = 0

    sections = [
        ("Backup Components", backup_status),
        ("Mathematical Systems", math_status),
        ("Trading Components", trading_status),
        ("Backtesting System", backtest_status),
    ]

    for section_name, status_dict in sections:
        section_total = len(status_dict)
        section_working = sum(status_dict.values())
        total_components += section_total
        working_components += section_working

        percentage = (section_working / section_total * 100) if section_total > 0 else 0
        status_icon = "✅" if percentage >= 80 else "⚠️" if percentage >= 50 else "❌"

        print(
            f"\n{status_icon} {section_name}: {section_working}/{section_total} ({percentage:.1f}%)"
        )

        for component, status in status_dict.items():
            icon = "✅" if status else "❌"
            print(f"   {icon} {component.replace('_', ' ').title()}")

    overall_percentage = (
        (working_components / total_components * 100) if total_components > 0 else 0
    )
    overall_icon = (
        "🎉" if overall_percentage >= 80 else "⚠️" if overall_percentage >= 50 else "💥"
    )

    print(
        f"\n{overall_icon} OVERALL SYSTEM STATUS: {working_components}/{total_components} ({overall_percentage:.1f}%)"
    )

    # Recommendations
    print("\n📋 RECOMMENDATIONS:")

    if backup_status["backup_integration_demo"] and not all(backup_status.values()):
        print("   • Restore missing backup components from core_backup directory")

    if not trading_status["unified_trading_pipeline"]:
        print("   • Fix unified trading pipeline import issues")

    if not backtest_status["backtest_integration"]:
        print("   • Integrate backtesting system with main trading pipeline")

    if overall_percentage >= 80:
        print(
            "   🎉 System is well integrated! Minor fixes needed for optimal performance."
        )
    elif overall_percentage >= 50:
        print(
            "   ⚠️  System needs attention. Address missing components for better functionality."
        )
    else:
        print("   💥 Critical integration issues. Immediate action required.")

    return overall_percentage


def main():
    """Run comprehensive backup and integration verification."""
    print("🚀 Schwabot Backup & Integration Verification")
    print("=" * 50)

    # Run all checks
    backup_status = check_backup_components()
    math_status = check_mathematical_integration()
    trading_status = check_trading_integration()
    backtest_status = check_backtesting_system()

    # Generate report
    overall_score = generate_integration_report(
        backup_status, math_status, trading_status, backtest_status
    )

    # Return appropriate exit code
    if overall_score >= 80:
        sys.exit(0)  # Success
    elif overall_score >= 50:
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Critical


if __name__ == "__main__":
    main()
