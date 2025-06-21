#!/usr/bin/env python3
"""Show flake8 progress summary."""

import subprocess
import sys


def main():
    """Show progress summary."""
    print("FLAKE8 PROGRESS SUMMARY")
    print("=" * 50)
    
    try:
        # Get statistics
        result = subprocess.run(
            ["flake8", "--count", "--statistics", "."],
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.strip().split('\n')
        
        # Show statistics (usually at the end)
        stats_started = False
        for line in lines:
            if line.strip() and (line[0].isdigit() or stats_started):
                stats_started = True
                print(line)
        
        # Count critical errors
        critical_result = subprocess.run(
            ["flake8", "--select=E999,F821,F401,F405", "--count", "."],
            capture_output=True,
            text=True
        )
        
        critical_count = critical_result.stdout.strip()
        print(f"\nCritical runtime-blocking errors: {critical_count}")
        
        # Count our new modules
        new_result = subprocess.run(
            ["flake8", "--count", "core/ghost", "core/phantom", "core/lantern", 
             "core/matrix", "core/profit", "core/glyph"],
            capture_output=True,
            text=True
        )
        
        new_count = new_result.stdout.strip()
        print(f"Errors in new ghost pipeline modules: {new_count}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 