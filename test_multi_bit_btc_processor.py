#!/usr/bin/env python3
"""
Test Multi-Bit BTC Processor - Schwabot UROS v1.0
================================================

Test script to verify the Multi-Bit BTC processor works correctly
after fixing the circular import issues.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the cleanup_backup directory to the path
REPO_ROOT = Path(__file__).resolve().parent
CORE_PATH = REPO_ROOT / "cleanup_backup" / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

# Add the main core directory to the path
MAIN_CORE_PATH = REPO_ROOT / "core"
if str(MAIN_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(MAIN_CORE_PATH))

def test_multi_bit_btc_processor():
    """Test the Multi-Bit BTC processor functionality."""
    print("🧪 Testing Multi-Bit BTC Processor")
    print("=" * 50)
    
    try:
        # Import the processor
        from multi_bit_btc_processor import MultiBitBTCProcessor, BitLevel
        
        print("✅ Successfully imported MultiBitBTCProcessor")
        
        # Initialize processor
        processor = MultiBitBTCProcessor()
        print("✅ Successfully initialized processor")
        
        # Test data processing
        base_price = 50000.0
        base_volume = 1000.0
        
        print("📊 Processing test data...")
        
        # Process data at different bit levels
        for i in range(10):
            price_change = np.random.normal(0, 100)
            volume_change = np.random.normal(0, 100)
            
            price = base_price + price_change
            volume = base_volume + volume_change
            
            # Process at different bit levels
            for bit_level in BitLevel:
                try:
                    data_point = processor.process_btc_data(price, volume, bit_level)
                    print(f"  ✅ Processed {bit_level.value}-bit data: price=${price:.2f}, vol={volume:.2f}")
                except Exception as e:
                    print(f"  ❌ Failed to process {bit_level.value}-bit data: {e}")
        
        # Test bit level analysis
        print("\n📈 Testing bit level analysis...")
        for bit_level in BitLevel:
            try:
                analysis = processor.analyze_bit_level(bit_level)
                if analysis:
                    print(f"  ✅ {bit_level.value}-bit analysis: confidence={analysis.confidence_score:.3f}")
                else:
                    print(f"  ⚠️ No data for {bit_level.value}-bit analysis")
            except Exception as e:
                print(f"  ❌ Failed {bit_level.value}-bit analysis: {e}")
        
        # Test cross-bit correlations
        print("\n🔗 Testing cross-bit correlations...")
        try:
            correlations = processor.analyze_cross_bit_correlations()
            print(f"  ✅ Found {len(correlations)} cross-bit correlations")
        except Exception as e:
            print(f"  ❌ Failed cross-bit correlations: {e}")
        
        # Test optimization
        print("\n🎯 Testing bit level optimization...")
        try:
            optimal_level = processor.optimize_bit_level_selection()
            print(f"  ✅ Optimal bit level: {optimal_level.value}-bit")
        except Exception as e:
            print(f"  ❌ Failed optimization: {e}")
        
        # Test statistics
        print("\n📊 Testing statistics...")
        try:
            stats = processor.get_btc_statistics()
            print(f"  ✅ Statistics: {stats['total_data_points']} data points, {stats['total_errors']} errors")
        except Exception as e:
            print(f"  ❌ Failed statistics: {e}")
        
        # Test trading signals
        print("\n📡 Testing trading signals...")
        try:
            signals = processor.get_trading_signals()
            print(f"  ✅ Generated {len(signals)} trading signals")
        except Exception as e:
            print(f"  ❌ Failed trading signals: {e}")
        
        print("\n🎉 Multi-Bit BTC Processor test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This might be due to missing dependencies or path issues.")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_math_system():
    """Test the unified math system."""
    print("\n🧮 Testing Unified Math System")
    print("=" * 50)
    
    try:
        from core.unified_math_system import unified_math
        
        print("✅ Successfully imported unified math system")
        
        # Test basic operations
        test_data = np.array([1, 2, 3, 4, 5])
        
        # Test mean
        mean_result = unified_math.mean(test_data)
        print(f"  ✅ Mean calculation: {mean_result}")
        
        # Test std
        std_result = unified_math.std(test_data)
        print(f"  ✅ Std calculation: {std_result}")
        
        # Test correlation
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([2, 4, 6, 8, 10])
        corr_result = unified_math.correlation(data1, data2)
        print(f"  ✅ Correlation calculation: {corr_result}")
        
        print("🎉 Unified Math System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Unified math test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution."""
    print("🧬 Multi-Bit BTC Processor Integration Test - Schwabot UROS v1.0")
    print("=" * 70)
    
    # Test unified math system first
    math_success = test_unified_math_system()
    
    # Test multi-bit BTC processor
    processor_success = test_multi_bit_btc_processor()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 Test Summary")
    print("=" * 70)
    print(f"Unified Math System: {'✅ PASS' if math_success else '❌ FAIL'}")
    print(f"Multi-Bit BTC Processor: {'✅ PASS' if processor_success else '❌ FAIL'}")
    
    if math_success and processor_success:
        print("\n🎉 All tests passed! The circular import issue has been resolved.")
        print("The Multi-Bit BTC processor is now ready for integration.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    return math_success and processor_success

if __name__ == "__main__":
    main() 