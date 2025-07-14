#!/usr/bin/env python3
"""
Simple Schwabot Launcher
Uses existing functional systems without removing anything

This launcher preserves all existing functionality while fixing import issues.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add core directory to path
sys.path.append(str(Path(__file__).parent / "core"))
sys.path.append(str(Path(__file__).parent / "utils"))

# Setup logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[]
        logging.FileHandler('schwabot_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for simple Schwabot launcher."""
    print("🚀 Schwabot Simple Launcher")
    print("=" * 50)
    print("Using existing functional systems")
    print("=" * 50)

    try:
        # Import existing functional components
        from core.automated_strategy_engine import AutomatedStrategyEngine
        from core.enhanced_ccxt_trading_engine import EnhancedCCXTTradingEngine
        from core.profit_optimization_engine import ProfitOptimizationEngine
        from core.qsc_enhanced_profit_allocator import QSCEnhancedProfitAllocator
        from core.soulprint_registry import SoulprintRegistry

        print("✅ Core components imported successfully")

        # Initialize components
        soulprint_registry = SoulprintRegistry()
        qsc_allocator = QSCEnhancedProfitAllocator()
        profit_optimizer = ProfitOptimizationEngine()
        trading_engine = EnhancedCCXTTradingEngine()
        strategy_engine = AutomatedStrategyEngine()

        print("✅ Components initialized successfully")

        # Start the existing dashboard if available
        try:
            from schwabot_dashboard_integration import main as start_dashboard
            print("🌐 Starting dashboard...")
            start_dashboard()
        except ImportError:
            print("⚠️ Dashboard not available, starting core only")

            # Keep core running
            print("🔄 Core system running...")
            while True:
                time.sleep(10)
                print("💼 System active - Press Ctrl+C to stop")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Please check that all core components are available")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    main() 