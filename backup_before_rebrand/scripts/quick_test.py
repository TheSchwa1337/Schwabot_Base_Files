#!/usr/bin/env python3
"""
Quick Component Tester - Windows Console Compatible
Tests components without Unicode characters
"""

import importlib
import os
import sys

# Add core to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_component(filename):
    """Test a single component"""
    module_name = filename[:-3]  # Remove .py
    module_path = f"core.{module_name}"

    try:
        module = importlib.import_module(module_path)
        return True, "SUCCESS"
    except Exception as e:
        return False, str(e)[:50]


def main():
    """Test all components and show results"""
    print("SCHWABOT COMPONENT TEST")
    print("=" * 50)

    # Get all Python files in core
    core_files = [f for f in os.listdir("core")]
                  if f.endswith('.py') and f != '__init__.py']

    successful = 0
    failed = 0

    for i, file in enumerate(core_files, 1):
        success, message = test_component(file)
        status = "SUCCESS" if success else "FAILED"
        print(f"[{i:2d}/{len(core_files)}] {file:<45} {status}")

        if success:
            successful += 1
        else:
            failed += 1

    total = successful + failed
    success_rate = (successful / total * 100) if total > 0 else 0

    print("=" * 50)
    print(f"RESULTS:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Target: {total}/75 = 100% (need to fix {failed} components)")

    return success_rate


if __name__ == "__main__":
    main()
