#!/usr/bin/env python3
"""
Launch Comprehensive Architecture Fix
====================================

Simple launcher script that runs the comprehensive architecture fix
following WINDOWS_CLI_COMPATIBILITY.md standards.

This script ensures proper execution order and handles any dependencies.

Usage:
    python launch_comprehensive_architecture_fix.py
    python launch_comprehensive_architecture_fix.py --dry-run
"""

import os
import sys
import subprocess
import platform
from typing import List
from windows_cli_compatibility import WindowsCliCompatibilityHandler

# Constants (Magic Number Replacements)
DEFAULT_RETRY_COUNT = 3


# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """Simple Windows CLI compatibility handler for the launcher"""
    
    @staticmethod
    def safe_print(message: str) -> str:
        """Print message safely with Windows CLI compatibility"""
        if platform.system() == "Windows":
            emoji_mapping = {
                '🚀': '[LAUNCH]', '✅': '[SUCCESS]', '❌': '[ERROR]',
                '⚠️': '[WARNING]', '🔧': '[PROCESSING]', '📊': '[INFO]',
                '🎉': '[COMPLETE]', '🔍': '[CHECKING]', '⚡': '[FAST]'
            }
            for emoji, replacement in emoji_mapping.items():
                message = message.replace(emoji, replacement)
        return message

def check_dependencies() -> bool:
    """Check if required files exist"""
    cli_handler = WindowsCliCompatibilityHandler()
    
    required_files = [
        'windows_cli_compliant_architecture_fixer.py',
        'apply_comprehensive_architecture_integration.py',
        'WINDOWS_CLI_COMPATIBILITY.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(cli_handler.safe_print("❌ Missing required files:"))
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print(cli_handler.safe_print("✅ All required files found"))
    return True

def run_architecture_fix(dry_run: bool = False) -> bool:
    """Run the comprehensive architecture fix"""
    cli_handler = WindowsCliCompatibilityHandler()
    
    print(cli_handler.safe_print("🚀 Starting Comprehensive Architecture Fix"))
    print("=" * 60)
    
    # Step 1: Run the architecture fixer
    print(cli_handler.safe_print("🔧 Step 1: Running architecture fixer..."))
    
    cmd = [sys.executable, 'windows_cli_compliant_architecture_fixer.py']
    if dry_run:
        cmd.append('--dry-run')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(cli_handler.safe_print("✅ Architecture fixer completed"))
        else:
            print(cli_handler.safe_print(f"⚠️ Architecture fixer warnings: {result.stderr}"))
    except subprocess.TimeoutExpired:
        print(cli_handler.safe_print("⚠️ Architecture fixer timed out"))
    except FileNotFoundError:
        print(cli_handler.safe_print("❌ Architecture fixer script not found"))
        return False
    except Exception as e:
        print(cli_handler.safe_print(f"❌ Error running architecture fixer: {e}"))
        return False
    
    # Step 2: Run the comprehensive integration
    print(cli_handler.safe_print("🔧 Step 2: Running comprehensive integration..."))
    
    cmd = [sys.executable, 'apply_comprehensive_architecture_integration.py']
    if dry_run:
        cmd.append('--dry-run')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(cli_handler.safe_print("✅ Comprehensive integration completed"))
        else:
            print(cli_handler.safe_print(f"⚠️ Integration warnings: {result.stderr}"))
    except subprocess.TimeoutExpired:
        print(cli_handler.safe_print("⚠️ Integration timed out"))
    except FileNotFoundError:
        print(cli_handler.safe_print("❌ Integration script not found"))
        return False
    except Exception as e:
        print(cli_handler.safe_print(f"❌ Error running integration: {e}"))
        return False
    
    # Step 3: Run flake8 fixes if available
    print(cli_handler.safe_print("🔧 Step 3: Running flake8 fixes..."))
    
    if os.path.exists('master_flake8_comprehensive_fixer.py'):
        cmd = [sys.executable, 'master_flake8_comprehensive_fixer.py']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(cli_handler.safe_print("✅ Flake8 fixes completed"))
            else:
                print(cli_handler.safe_print(f"⚠️ Flake8 fixer warnings: {result.stderr}"))
        except subprocess.TimeoutExpired:
            print(cli_handler.safe_print("⚠️ Flake8 fixer timed out"))
        except Exception as e:
            print(cli_handler.safe_print(f"❌ Error running flake8 fixer: {e}"))
    else:
        print(cli_handler.safe_print("📊 Flake8 fixer not found, skipping"))
    
    print(cli_handler.safe_print("🎉 Comprehensive architecture fix complete!"))
    return True

def main() -> None:
    """Main entry point"""
    cli_handler = WindowsCliCompatibilityHandler()
    
    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv
    
    if dry_run:
        print(cli_handler.safe_print("🔍 DRY RUN MODE - No files will be modified"))
    
    # Check dependencies
    if not check_dependencies():
        print(cli_handler.safe_print("❌ Dependency check failed"))
        sys.exit(1)
    
    # Run the fix
    success = run_architecture_fix(dry_run)
    
    if success:
        print(cli_handler.safe_print("🌟 All architecture fixes applied successfully!"))
        print(cli_handler.safe_print("📊 Check the generated reports for details"))
        sys.exit(0)
    else:
        print(cli_handler.safe_print("❌ Architecture fix failed"))
        sys.exit(1)

if __name__ == "__main__":
    main() 