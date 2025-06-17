"""
BTC Processor Test Runner
=======================

Script to run BTC data processor tests with various configurations.
"""

import argparse
import sys
import unittest
import logging
from pathlib import Path

def setup_logging(verbose=False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run BTC processor tests')
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--gpu-only',
        action='store_true',
        help='Run only GPU-specific tests'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Run only CPU-specific tests'
    )
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # Import test module
    from tests.test_btc_processor import TestBTCDataProcessor
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests based on configuration
    if args.gpu_only:
        suite.addTest(TestBTCDataProcessor('test_gpu_setup'))
        suite.addTest(TestBTCDataProcessor('test_gpu_stream_management'))
    elif args.cpu_only:
        suite.addTest(TestBTCDataProcessor('test_hash_generation'))
        suite.addTest(TestBTCDataProcessor('test_buffer_management'))
    else:
        # Add all tests
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBTCDataProcessor))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    main() 