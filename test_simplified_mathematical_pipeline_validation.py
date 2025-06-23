#!/usr/bin/env python3
"""
Test Simplified Mathematical Pipeline Validation - Schwabot UROS v1.0
====================================================================

Simple test script to run the simplified mathematical pipeline validation.
This validates core components without circular imports.

This validates all components before going live with Schwabot UROS v1.0.
"""

import asyncio
import logging
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging with ASCII-safe format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Run the simplified mathematical pipeline validation."""
    try:
        print("Starting Schwabot UROS v1.0 Simplified Mathematical Pipeline Validation")
        print("=" * 80)
        
        # Import and run the simplified validator
        from core.mathematical_pipeline_validator_simple import run_simplified_mathematical_pipeline_validation
        
        # Run simplified validation
        report = await run_simplified_mathematical_pipeline_validation()
        
        # Print comprehensive results
        print(f"\n{'='*80}")
        print(f"SCHWABOT UROS v1.0 SIMPLIFIED VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Status: {report.overall_status}")
        print(f"Production Readiness Score: {report.production_readiness_score:.3f}")
        print(f"Average Confidence: {report.average_confidence:.3f}")
        print(f"Total Execution Time: {report.total_execution_time:.2f}ms")
        
        print(f"\nComponent Results:")
        print(f"  PASSED: {report.passed_components}")
        print(f"  FAILED: {report.failed_components}")
        print(f"  WARNINGS: {report.warning_components}")
        print(f"  Total Components: {report.total_components}")
        
        # Print detailed component results
        print(f"\nDetailed Component Analysis:")
        for component_name, result in report.component_results.items():
            status_symbol = "PASS" if result.validation_status == "PASS" else "FAIL" if result.validation_status == "FAIL" else "WARN"
            print(f"  {status_symbol} {component_name}: {result.validation_status} (Confidence: {result.confidence_score:.3f})")
        
        if report.critical_issues:
            print(f"\nCritical Issues:")
            for issue in report.critical_issues:
                print(f"  ERROR: {issue}")
        
        if report.optimization_recommendations:
            print(f"\nOptimization Recommendations:")
            for rec in report.optimization_recommendations:
                print(f"  WARNING: {rec}")
        
        # Final assessment
        print(f"\n{'='*80}")
        if report.overall_status == "PASS":
            print("SCHWABOT UROS v1.0 IS READY FOR PRODUCTION!")
            print("All critical components validated successfully")
            print("Mathematical pipeline integrity confirmed")
            print("Ready to execute live trading operations")
        elif report.overall_status == "WARN":
            print("SCHWABOT UROS v1.0 HAS WARNINGS - REVIEW RECOMMENDATIONS")
            print("Some components have warnings but no critical failures")
            print("Consider addressing optimization recommendations")
        else:
            print("SCHWABOT UROS v1.0 HAS CRITICAL ISSUES - DO NOT PROCEED")
            print("Critical components failed validation")
            print("Must resolve all critical issues before going live")
        
        print(f"{'='*80}")
        
        # Return appropriate exit code
        if report.overall_status == "PASS":
            return 0
        elif report.overall_status == "WARN":
            return 1
        else:
            return 2
            
    except Exception as e:
        print(f"Simplified validation failed with error: {e}")
        logging.error(f"Simplified validation error: {e}", exc_info=True)
        return 3

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 