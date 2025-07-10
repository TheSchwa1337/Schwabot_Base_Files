#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Fix Runner for Schwabot Trading System (Simple Version)

This script runs all comprehensive fixes in the correct sequence:
1. Syntax error fixes
2. Import fixes
3. Requirements updates
4. Platform-specific fixes
5. Final validation

Usage:
    python run_comprehensive_fixes_simple.py
"""

# Configure logging
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/comprehensive_fix.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ComprehensiveFixRunner:
    """Master script to run all comprehensive fixes."""

    def __init__(self):
        """Initialize the comprehensive fix runner."""
        self.start_time = time.time()
        self.fix_results = {}
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)

    def run_script(self, script_name: str, description: str) -> bool:
        """Run a Python script and capture the result."""
        logger.info(f"Running {description}...")

        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_name], capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"SUCCESS: {description} completed successfully")
                self.fix_results[script_name] = {'success': True, 'stdout': result.stdout, 'stderr': result.stderr}
                return True
            else:
                logger.error(f"FAILED: {description} failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                self.fix_results[script_name] = {
                    'success': False,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode,
                }
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"FAILED: {description} timed out after 5 minutes")
            self.fix_results[script_name] = {'success': False, 'error': 'Timeout'}
            return False
        except Exception as e:
            logger.error(f"FAILED: {description} failed with exception: {e}")
            self.fix_results[script_name] = {'success': False, 'error': str(e)}
            return False

    def run_syntax_fixes(self) -> bool:
        """Run syntax error fixes."""
        return self.run_script('fix_syntax_errors_comprehensive.py', 'Syntax Error Fixes')

    def run_import_fixes(self) -> bool:
        """Run import fixes."""
        return self.run_script('fix_imports_comprehensive.py', 'Import Fixes')

    def run_requirements_update(self) -> bool:
        """Run requirements update."""
        return self.run_script('update_requirements_comprehensive.py', 'Requirements Update')

    def run_flake8_check(self) -> bool:
        """Run Flake8 to check remaining issues."""
        logger.info("Running Flake8 check...")

        try:
            result = subprocess.run(
                ['flake8', 'core/', '--max-line-length=88', '--extend-ignore=E203,W503'],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                logger.info("SUCCESS: Flake8 check passed - no issues found")
                return True
            else:
                logger.warning(f"WARNING: Flake8 found {result.returncode} issues")
                logger.info("STDOUT: " + result.stdout)
                logger.info("STDERR: " + result.stderr)

                # Save Flake8 output
                with open('flake8_final_report.txt', 'w') as f:
                    f.write(result.stdout)
                    f.write(result.stderr)

                return False

        except Exception as e:
            logger.error(f"FAILED: Flake8 check failed: {e}")
            return False

    def create_test_script(self) -> None:
        """Create a test script to verify the fixes."""
        test_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify comprehensive fixes.
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all critical modules can be imported."""
    critical_modules = [
        'core.strategy_bit_mapper',
        'core.matrix_mapper',
        'core.trading_strategy_executor',
        'core.schwabot_rheology_integration',
        'core.orbital_shell_brain_system',
        'core.zpe_core',
        'core.zbe_core',
    ]
    
    failed_imports = []
    
    for module_name in critical_modules:
        try:
            importlib.import_module(module_name)
            print(f"SUCCESS: {module_name}")
        except Exception as e:
            print(f"FAILED: {module_name}: {e}")
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0

def test_syntax():
    """Test that all Python files have valid syntax."""
    import ast
    
    failed_files = []
    
    for py_file in Path('core').rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            print(f"SUCCESS: {py_file}")
        except Exception as e:
            print(f"FAILED: {py_file}: {e}")
            failed_files.append(str(py_file))
    
    return len(failed_files) == 0

def main():
    """Run all tests."""
    print("Testing comprehensive fixes...")
    print("=" * 50)
    
    import_success = test_imports()
    syntax_success = test_syntax()
    
    print("=" * 50)
    print(f"Import tests: {'SUCCESS' if import_success else 'FAILED'}")
    print(f"Syntax tests: {'SUCCESS' if syntax_success else 'FAILED'}")
    
    if import_success and syntax_success:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''

        with open('test_comprehensive_fixes.py', 'w') as f:
            f.write(test_script)

        logger.info("Created test script: test_comprehensive_fixes.py")

    def run_tests(self) -> bool:
        """Run the test script."""
        return self.run_script('test_comprehensive_fixes.py', 'Comprehensive Tests')

    def generate_final_report(self) -> None:
        """Generate a comprehensive final report."""
        report_path = Path('comprehensive_fix_final_report.md')

        with open(report_path, 'w') as f:
            f.write("# Comprehensive Fix Final Report\n\n")

            # Summary
            f.write("## Summary\n")
            total_time = time.time() - self.start_time
            f.write(f"- Total execution time: {total_time:.2f} seconds\n")
            f.write(f"- Scripts run: {len(self.fix_results)}\n")
            successful_scripts = sum(1 for result in self.fix_results.values() if result.get('success', False))
            f.write(f"- Successful scripts: {successful_scripts}\n")
            f.write(f"- Failed scripts: {len(self.fix_results) - successful_scripts}\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            for script_name, result in self.fix_results.items():
                status = "SUCCESS" if result.get('success', False) else "FAILED"
                f.write(f"### {script_name}: {status}\n")

                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")

                if 'stdout' in result and result['stdout']:
                    f.write("STDOUT:\n```\n")
                    f.write(result['stdout'][:1000])  # Limit output
                    if len(result['stdout']) > 1000:
                        f.write("\n... (truncated)")
                    f.write("\n```\n")

                if 'stderr' in result and result['stderr']:
                    f.write("STDERR:\n```\n")
                    f.write(result['stderr'][:1000])  # Limit output
                    if len(result['stderr']) > 1000:
                        f.write("\n... (truncated)")
                    f.write("\n```\n")

                f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            if successful_scripts == len(self.fix_results):
                f.write("ALL FIXES COMPLETED SUCCESSFULLY!\n")
                f.write("- The codebase should now be free of critical syntax and import errors\n")
                f.write("- Run `python test_comprehensive_fixes.py` to verify functionality\n")
                f.write("- Consider running `flake8` for additional code quality checks\n")
            else:
                f.write("SOME FIXES FAILED. Please review the detailed results above.\n")
                f.write("- Check the logs in the `logs/` directory for more details\n")
                f.write("- Manual intervention may be required for failed scripts\n")
                f.write("- Consider running individual fix scripts to isolate issues\n")

            f.write("\n## Next Steps\n\n")
            f.write("1. Review the generated reports\n")
            f.write("2. Test the system functionality\n")
            f.write("3. Run additional code quality checks if needed\n")
            f.write("4. Deploy the fixed system\n")

        logger.info(f"Final report generated: {report_path}")

    def run_all_fixes(self) -> bool:
        """Run all comprehensive fixes in sequence."""
        logger.info("Starting comprehensive fix process...")
        logger.info("=" * 60)

        # Phase 1: Syntax fixes
        logger.info("PHASE 1: Syntax Error Fixes")
        if not self.run_syntax_fixes():
            logger.warning("WARNING: Syntax fixes failed, but continuing...")

        # Phase 2: Import fixes
        logger.info("PHASE 2: Import Fixes")
        if not self.run_import_fixes():
            logger.warning("WARNING: Import fixes failed, but continuing...")

        # Phase 3: Requirements update
        logger.info("PHASE 3: Requirements Update")
        if not self.run_requirements_update():
            logger.warning("WARNING: Requirements update failed, but continuing...")

        # Phase 4: Create test script
        logger.info("PHASE 4: Creating Test Script")
        self.create_test_script()

        # Phase 5: Run tests
        logger.info("PHASE 5: Running Tests")
        if not self.run_tests():
            logger.warning("WARNING: Tests failed, but continuing...")

        # Phase 6: Final Flake8 check
        logger.info("PHASE 6: Final Flake8 Check")
        self.run_flake8_check()

        # Generate final report
        logger.info("PHASE 7: Generating Final Report")
        self.generate_final_report()

        # Summary
        total_time = time.time() - self.start_time
        successful_scripts = sum(1 for result in self.fix_results.values() if result.get('success', False))

        logger.info("=" * 60)
        logger.info("COMPREHENSIVE FIX PROCESS COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Scripts run: {len(self.fix_results)}")
        logger.info(f"Successful: {successful_scripts}")
        logger.info(f"Failed: {len(self.fix_results) - successful_scripts}")

        if successful_scripts == len(self.fix_results):
            logger.info("ALL FIXES COMPLETED SUCCESSFULLY!")
            return True
        else:
            logger.warning("SOME FIXES FAILED. Check the final report for details.")
            return False


def main():
    """Main function to run all comprehensive fixes."""
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)

    # Create and run the comprehensive fix runner
    runner = ComprehensiveFixRunner()
    success = runner.run_all_fixes()

    if success:
        print("\nALL FIXES COMPLETED SUCCESSFULLY!")
        print("Check the final report: comprehensive_fix_final_report.md")
        print("Run tests: python test_comprehensive_fixes.py")
        return 0
    else:
        print("\nSOME FIXES FAILED.")
        print("Check the final report: comprehensive_fix_final_report.md")
        print("Review logs in the logs/ directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
