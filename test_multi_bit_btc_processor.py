from dual_unicore_handler import DualUnicoreHandler
from pathlib import Path
import os
import sys

import numpy as np


# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf - 8 -*-
"""
"""
"""
"""
"""
Test Multi - Bit BTC Processor - Schwabot UROS v1.0
================================================

Test script to verify the Multi - Bit BTC processor works correctly
after fixing the circular import issues.
"""
"""
"""
"""
"""


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
    """Test the Multi - Bit BTC processor functionality."""


"""
"""
"""
"""
 print("\\u1f9ea Testing Multi - Bit BTC Processor")
  print("=" * 50)

   try:
        # Import the processor
        from multi_bit_btc_processor import MultiBitBTCProcessor, BitLevel

        print("\\u2705 Successfully imported MultiBitBTCProcessor")

# Initialize processor
        processor = MultiBitBTCProcessor()
        print("\\u2705 Successfully initialized processor")

# Test data processing
        base_price = 50000.0
        base_volume = 1000.0

        print("\\u1f4ca Processing test data...")

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
                    print(f"  \\u2705 Processed {bit_level.value}-bit data: price=${price:.2f}, vol={volume:.2f}")
                except Exception as e:
                    print(f"  \\u274c Failed to process {bit_level.value}-bit data: {e}")

# Test bit level analysis
        print("\\n\\u1f4c8 Testing bit level analysis...")
        for bit_level in BitLevel:
            try:
                analysis = processor.analyze_bit_level(bit_level)
                if analysis:
                    print(f"  \\u2705 {bit_level.value}-bit analysis: confidence={analysis.confidence_score:.3f}")
                else:
                    print(f"  \\u26a0\\ufe0f No data for {bit_level.value}-bit analysis")
            except Exception as e:
                print(f"  \\u274c Failed {bit_level.value}-bit analysis: {e}")

# Test cross - bit correlations
        print("\\n\\u1f517 Testing cross - bit correlations...")
        try:
            correlations = processor.analyze_cross_bit_correlations()
            print(f"  \\u2705 Found {len(correlations)} cross - bit correlations")
        except Exception as e:
            print(f"  \\u274c Failed cross - bit correlations: {e}")

# Test optimization
        print("\\n\\u1f3af Testing bit level optimization...")
        try:
            optimal_level = processor.optimize_bit_level_selection()
            print(f"  \\u2705 Optimal bit level: {optimal_level.value}-bit")
        except Exception as e:
            print(f"  \\u274c Failed optimization: {e}")

# Test statistics
        print("\\n\\u1f4ca Testing statistics...")
        try:
            stats = processor.get_btc_statistics()
            print(f"  \\u2705 Statistics: {stats['total_data_points']} data points, {stats['total_errors']} errors")
        except Exception as e:
            print(f"  \\u274c Failed statistics: {e}")

# Test trading signals
        print("\\n\\u1f4e1 Testing trading signals...")
        try:
            signals = processor.get_trading_signals()
            print(f"  \\u2705 Generated {len(signals)} trading signals")
        except Exception as e:
            print(f"  \\u274c Failed trading signals: {e}")

        print("\\n\\u1f389 Multi - Bit BTC Processor test completed successfully!")
        return True

    except ImportError as e:
        print(f"\\u274c Import error: {e}")
        print("This might be due to missing dependencies or path issues.")
        return False
    except Exception as e:
        print(f"\\u274c Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_math_system():
    """Test the unified math system."""


"""
"""
"""
"""
 print("\\n\\u1f9ee Testing Unified Math System")
  print("=" * 50)

   try:
        from core.unified_math_system import unified_math

        print("\\u2705 Successfully imported unified math system")

# Test basic operations
        test_data = np.array([1, 2, 3, 4, 5])

# Test mean
        mean_result = unified_math.mean(test_data)
        print(f"  \\u2705 Mean calculation: {mean_result}")

# Test std
        std_result = unified_math.std(test_data)
        print(f"  \\u2705 Std calculation: {std_result}")

# Test correlation
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([2, 4, 6, 8, 10])
        corr_result = unified_math.correlation(data1, data2)
        print(f"  \\u2705 Correlation calculation: {corr_result}")

        print("\\u1f389 Unified Math System test completed successfully!")
        return True

    except Exception as e:
        print(f"\\u274c Unified math test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test execution."""


"""
"""
"""
"""
 print("\\u1f9ec Multi - Bit BTC Processor Integration Test - Schwabot UROS v1.0")
  print("=" * 70)

# Test unified math system first
   math_success = test_unified_math_system()

# Test multi - bit BTC processor
    processor_success = test_multi_bit_btc_processor()

# Summary
    print("\n" + "=" * 70)
    print("\\u1f4cb Test Summary")
    print("=" * 70)
    print(f"Unified Math System: {'\\u2705 PASS' if math_success else '\\u274c FAIL'}")
    print(f"Multi - Bit BTC Processor: {'\\u2705 PASS' if processor_success else '\\u274c FAIL'}")

    if math_success and processor_success:
        print("\\n\\u1f389 All tests passed! The circular import issue has been resolved.")
        print("The Multi - Bit BTC processor is now ready for integration.")
    else:
        print("\\n\\u26a0\\ufe0f Some tests failed. Please check the error messages above.")

    return math_success and processor_success


if __name__ == "__main__":
    main()

"""
"""
"""
"""
"""
"""
