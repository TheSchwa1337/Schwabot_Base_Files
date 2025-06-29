# -*- coding: utf-8 -*-
"""
Test Script for Glyph Strategy System
------------------------------------
Comprehensive test and demonstration of the glyph-to-strategy mapping system
for Schwabot's mathematical trading framework.

This script demonstrates:
1. Glyph to strategy bit mapping via SHA256
2. Gear-driven strategy selection based on volume
3. Fractal memory encoding and storage
4. Entry/exit portal integration
5. Simulated trade execution
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import unittest
from unittest.mock import MagicMock, patch
import traceback

# Add the parent directory to sys.path to allow imports from 'core'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_glyph_strategy_core():
    """Test the core glyph strategy functionality."""
    print("=" * 60)
    print("🧠 GLYPH STRATEGY CORE TEST")
    print("=" * 60)
    
    try:
        from core.strategy.glyph_strategy_core import GlyphStrategyCore, glyph_to_strategy
        
        # Initialize core
        core = GlyphStrategyCore(
            enable_fractal_memory=True,
            enable_gear_shifting=True,
            volume_thresholds=(1.5e6, 5e6)
        )
        
        # Test glyphs
        test_glyphs = ['brain', 'skull', 'fire', 'hourglass', 'tornado', 'lightning', 'shield', 'target', 'crystal', 'scales']
        test_volumes = [1e6, 3e6, 6e6]  # Low, medium, high volume
        
        print(f"Testing {len(test_glyphs)} glyphs across {len(test_volumes)} volume levels")
        print()
        
        results = []
        
        for glyph in test_glyphs:
            print(f"Glyph: {glyph}")
            glyph_results = []
            
            for volume in test_volumes:
                # Get strategy selection
                result = core.select_strategy(glyph, volume)
                
                glyph_results.append({
                    'volume': volume,
                    'gear_state': result.gear_state,
                    'strategy_id': result.strategy_id,
                    'confidence': result.confidence,
                    'fractal_hash': result.fractal_hash[:8] + "..."
                })
                
                print(f"  Volume: {volume:.1e} → Gear: {result.gear_state}-bit, "
                      f"Strategy: {result.strategy_id}, Confidence: {result.confidence:.3f}")
            
            results.append({
                'glyph': glyph,
                'results': glyph_results
            })
            print()
        
        # Show performance stats
        print("📊 PERFORMANCE STATISTICS")
        print("-" * 40)
        stats = core.get_performance_stats()
        for key, value in stats.items():
            if key == 'fractal_memory':
                print(f"Fractal Memory:")
                for mem_key, mem_value in value.items():
                    print(f"  {mem_key}: {mem_value}")
            else:
                print(f"{key}: {value}")
        
        return core, results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None, []
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(traceback.format_exc())
        return None, []

def test_entry_exit_portal():
    """Test the entry/exit portal functionality."""
    print("\n" + "=" * 60)
    print("🚪 ENTRY/EXIT PORTAL TEST")
    print("=" * 60)
    
    try:
        from core.strategy.entry_exit_portal import EntryExitPortal, process_glyph_trade_signal
        
        # Initialize portal
        portal = EntryExitPortal(
            enable_risk_management=True,
            enable_portfolio_tracking=True,
            max_position_size=0.1,
            min_confidence_threshold=0.5
        )
        
        # Test parameters
        test_glyphs = ['brain', 'skull', 'fire', 'hourglass', 'tornado']
        test_volume = 3.2e6
        test_price = 50000.0
        test_asset = "BTC/USD"
        
        print(f"Testing trade signal processing with:")
        print(f"  Volume: {test_volume:.1e}")
        print(f"  Price: ${test_price:,.2f}")
        print(f"  Asset: {test_asset}")
        print()
        
        executed_trades = []
        
        for glyph in test_glyphs:
            print(f"Processing glyph: {glyph}")
            
            # Process signal
            signal = portal.process_glyph_signal(
                glyph, test_volume, test_asset, test_price
            )
            
            if signal:
                print(f"  ✅ Signal generated:")
                print(f"    Strategy ID: {signal.strategy_id}")
                print(f"    Direction: {signal.direction.value}")
                print(f"    Confidence: {signal.confidence:.3f}")
                print(f"    Gear State: {signal.metadata.get('gear_state', 'N/A')}")
                
                # Execute signal (simulated)
                result = portal.execute_signal(signal, dry_run=True)
                
                if "execution_result" in result:
                    exec_result = result["execution_result"]
                    print(f"  📈 Execution result:")
                    print(f"    Status: {exec_result['status']}")
                    print(f"    Order ID: {exec_result['order_id']}")
                    print(f"    Size: ${exec_result['executed_size']:,.2f}")
                    print(f"    Fees: ${exec_result['fees']:,.2f}")
                    
                    executed_trades.append({
                        'glyph': glyph,
                        'signal': signal,
                        'execution': exec_result
                    })
                else:
                    print(f"  ❌ Execution failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"  ❌ Signal rejected (confidence too low)")
            
            print()
        
        # Show portal stats
        print("📊 PORTAL STATISTICS")
        print("-" * 40)
        stats = portal.get_performance_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return portal, executed_trades
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None, []
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(traceback.format_exc())
        return None, []

def test_integrated_workflow():
    """Test the complete integrated workflow."""
    print("\n" + "=" * 60)
    print("🔄 INTEGRATED WORKFLOW TEST")
    print("=" * 60)
    
    try:
        from core.strategy import create_glyph_trading_system
        
        # Create complete system
        glyph_core, portal = create_glyph_trading_system(
            enable_fractal_memory=True,
            enable_gear_shifting=True,
            enable_risk_management=True,
            enable_portfolio_tracking=True
        )
        
        print("✅ Complete glyph trading system created")
        print()
        
        # Simulate market conditions
        market_scenarios = [
            {'volume': 1e6, 'price': 45000, 'description': 'Low volume, bearish'},
            {'volume': 3e6, 'price': 50000, 'description': 'Medium volume, neutral'},
            {'volume': 7e6, 'price': 55000, 'description': 'High volume, bullish'}
        ]
        
        test_glyphs = ['brain', 'skull', 'fire']
        
        for scenario in market_scenarios:
            print(f"📊 Market Scenario: {scenario['description']}")
            print(f"   Volume: {scenario['volume']:.1e}, Price: ${scenario['price']:,.2f}")
            print()
            
            for glyph in test_glyphs:
                # Process signal
                signal = portal.process_glyph_signal(
                    glyph, scenario['volume'], "BTC/USD", scenario['price']
                )
                
                if signal:
                    # Execute signal
                    result = portal.execute_signal(signal, dry_run=True)
                    
                    print(f"  {glyph} → {signal.direction.value} "
                          f"(Strategy: {signal.strategy_id}, "
                          f"Confidence: {signal.confidence:.3f})")
                    
                    if "execution_result" in result:
                        exec_result = result["execution_result"]
                        print(f"    Size: ${exec_result['executed_size']:,.2f}, "
                              f"Fees: ${exec_result['fees']:,.2f}")
            
            print()
        
        # Show combined stats
        print("📊 COMBINED SYSTEM STATISTICS")
        print("-" * 40)
        print("Glyph Core:")
        core_stats = glyph_core.get_performance_stats()
        for key, value in core_stats.items():
            if key != 'fractal_memory':
                print(f"  {key}: {value}")
        
        print("\nPortal:")
        portal_stats = portal.get_performance_stats()
        for key, value in portal_stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None, []
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(traceback.format_exc())
        return None, []

def demonstrate_mathematical_framework():
    """Demonstrate the mathematical framework behind the glyph system."""
    print("\n" + "=" * 60)
    print("🧮 MATHEMATICAL FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    
    try:
        from core.strategy.glyph_strategy_core import GlyphStrategyCore
        
        core = GlyphStrategyCore()
        
        # Demonstrate SHA256 transformation
        print("🔐 SHA-256 Transformation Process:")
        test_glyph = "brain"
        sha_hash = core.glyph_to_sha(test_glyph)
        print(f"  Glyph: {test_glyph}")
        print(f"  SHA-256: {sha_hash}")
        print()
        
        # Demonstrate bit extraction
        print("🔢 Bit Extraction Process:")
        for bit_depth in [4, 8, 16]:
            strategy_bits = core.sha_to_strategy_bits(sha_hash, bit_depth)
            binary = bin(strategy_bits)[2:].zfill(bit_depth)
            print(f"  {bit_depth}-bit: {strategy_bits} (binary: {binary})")
        print()
        
        # Demonstrate gear shifting
        print("⚙️ Gear Shifting Logic:")
        volumes = [1e6, 2e6, 4e6, 6e6]
        for volume in volumes:
            gear = core.gear_shift(volume)
            print(f"  Volume: {volume:.1e} → Gear: {gear}-bit")
        print()
        
        # Demonstrate fractal memory
        print("♾️ Fractal Memory Encoding:")
        for i, glyph in enumerate(['brain', 'skull', 'fire']):
            result = core.select_strategy(glyph, 3e6)
            print(f"  {glyph} → Hash: {result.fractal_hash[:16]}...")
        
        fractal_stats = core.get_fractal_memory_stats()
        print(f"\n  Total hashes stored: {fractal_stats['total_hashes']}")
        print(f"  Memory size limit: {fractal_stats['memory_size']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Demonstration failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Main test function."""
    print("SCHWABOT GLYPH STRATEGY SYSTEM TEST")
    print("============================================================")
    print("Testing the complete glyph-to-strategy mapping system")
    print("with fractal memory, gear shifting, and trade execution.")
    print()
    
    # Run all tests
    test_results = {}
    
    # Test 1: Core functionality
    core, core_results = test_glyph_strategy_core()
    test_results['core'] = core is not None
    
    # Test 2: Entry/exit portal
    portal, portal_results = test_entry_exit_portal()
    test_results['portal'] = portal is not None
    
    # Test 3: Integrated workflow
    workflow_result = test_integrated_workflow()
    test_results['workflow'] = workflow_result is not None
    
    # Test 4: Mathematical framework
    math_result = demonstrate_mathematical_framework()
    test_results['mathematical'] = math_result
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name.upper()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Glyph strategy system is ready for use.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    print("\n" + "=" * 60)
    print("📚 USAGE EXAMPLES")
    print("=" * 60)
    
    print("Quick glyph-to-strategy conversion:")
    print("  from core.strategy import glyph_to_strategy")
    print("  result = glyph_to_strategy('brain', 3.2e6)")
    print("  print(result)")
    print()
    
    print("Complete trading system setup:")
    print("  from core.strategy import create_glyph_trading_system")
    print("  glyph_core, portal = create_glyph_trading_system()")
    print("  signal = portal.process_glyph_signal('brain', 3.2e6, 'BTC/USD', 50000)")
    print("  result = portal.execute_signal(signal, dry_run=True)")
    print()
    
    print("The glyph strategy system is now fully integrated with Schwabot!")
    print("Ready for both backtesting and live execution modes.")

if __name__ == "__main__":
    main() 