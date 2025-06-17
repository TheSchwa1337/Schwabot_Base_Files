#!/usr/bin/env python3
"""
Validation Runner
===============

Script to run the validation process and generate reports.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from core.validation_manager import ValidationManager

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run system validation')
    parser.add_argument('--config', type=str, default='config/validation_config.yaml',
                      help='Path to validation config file')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to output validation report')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        # Initialize validation manager
        manager = ValidationManager(config_path=args.config)
        
        # Run validation suite
        logger.info("Starting validation suite...")
        results = manager.run_validation_suite()
        
        # Generate report
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"validation_report_{timestamp}.txt"
            
        manager.save_validation_report(output_path)
        logger.info(f"Validation report saved to {output_path}")
        
        # Check for validation failures
        failed_categories = []
        for category, results in manager.validation_results.items():
            if isinstance(results, dict):
                for subcategory, subresults in results.items():
                    if isinstance(subresults, dict) and not subresults.get('passed', True):
                        failed_categories.append(f"{category}.{subcategory}")
            elif not results:
                failed_categories.append(category)
                
        if failed_categories:
            logger.error("Validation failed in the following categories:")
            for category in failed_categories:
                logger.error(f"  - {category}")
            sys.exit(1)
        else:
            logger.info("All validation checks passed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 