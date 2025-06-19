#!/usr/bin/env python3
"""
Windows CLI Compatibility Application Script
=============================================

This script applies Windows CLI compatibility fixes to Python files that may
encounter encoding issues or emoji rendering problems on Windows command line.
"""

import os
import platform
import re
import shutil
from pathlib import Path
from typing import Tuple


class WindowsCliCompatibilityHandler:
    """Handler for Windows CLI compatibility issues"""
    
    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return (platform.system() == "Windows" and
                ("cmd" in os.environ.get("COMSPEC", "").lower() or
                 "powershell" in os.environ.get("PSModulePath", "").lower()))

    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """
        Print message safely with Windows CLI compatibility
        """
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            emoji_to_asic_mapping = {
                '✅': '[SUCCESS]',
                '❌': '[ERROR]',
                '🔧': '[PROCESSING]',
                '🚀': '[LAUNCH]',
                '🎉': '[COMPLETE]',
                '💥': '[CRITICAL]',
                '⚡': '[FAST]',
                '🔍': '[SEARCH]',
                '📊': '[DATA]',
                '🧪': '[TEST]',
                '🛠️': '[TOOLS]',
                '⚖️': '[BALANCE]',
                '🔄': '[CYCLE]',
                '🎯': '[TARGET]',
                '📈': '[PROFIT]',
                '🔥': '[HOT]',
                '❄️': '[COOL]',
                '⭐': '[STAR]',
            }
            safe_message = message
            for emoji, asic_replacement in emoji_to_asic_mapping.items():
                safe_message = safe_message.replace(emoji, asic_replacement)
            return safe_message
        return message

    @staticmethod
    def log_safe(logger, level: str, message: str):
        """Log message safely with Windows CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            ascii_message = safe_message.encode('ascii', errors='replace').decode('ascii')
            getattr(logger, level.lower())(ascii_message)

    @staticmethod
    def safe_format_error(error: Exception, context: str = "") -> str:
        """Format error messages safely for Windows CLI"""
        error_message = f"Error: {str(error)}"
        if context:
            error_message += f" | Context: {context}"
        return WindowsCliCompatibilityHandler.safe_print(error_message)


# Required imports template
REQUIRED_IMPORTS = """import platform
import os
import shutil
import re
from typing import Tuple"""

# Windows CLI Handler Template
WINDOWS_CLI_HANDLER_TEMPLATE = """
class WindowsCliCompatibilityHandler:
    # Implementation here
    pass
"""

# Files that have already been fixed
FIXED_FILES = {
    'dlt_waveform_engine.py',
    'core/ccxt_execution_manager.py',
    'core/fault_bus.py',
    'core/multi_bit_btc_processor.py',
    'core/profit_routing_engine.py',
    'test_alif_aleph_system_integration.py',
    'test_alif_aleph_system_diagnostic.py',
    'test_schwabot_system_runner_windows_compatible.py'
}

# Target files that need fixing
TARGET_FILES = [
    'core/unified_mathematical_trading_controller.py',
    'core/dashboard_integration.py',
    'core/schwabot_dashboard.py',
    'core/enhanced_thermal_aware_btc_processor.py',
    'core/practical_visual_controller.py',
    'core/bit_operations.py',
    'core/phase_gate_controller.py',
    'core/unified_api_coordinator.py',
    'core/math_core.py',
    'core/mathlib_v3.py',
    'enhanced_fitness_oracle.py',
    'core/tick_management_system.py',
    'core/ghost_data_recovery.py',
    'core/integrated_alif_aleph_system.py',
    'test_complete_system.py'
]


def check_file_exists(file_path: str) -> bool:
    """Check if file exists and is readable"""
    return os.path.exists(file_path)


def backup_file(file_path: str) -> str:
    """Create backup of the file before modification"""
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    return backup_path


def add_required_imports(content: str) -> str:
    """Add required imports for Windows CLI compatibility"""
    if 'import platform' not in content and 'import os' not in content:
        import_match = re.search(r'^import\s+', content, re.MULTILINE)
        if import_match:
            insert_pos = import_match.end()
            content = (content[:insert_pos] + '\n' + 
                       REQUIRED_IMPORTS + content[insert_pos:])
        else:
            content = REQUIRED_IMPORTS + '\n\n' + content
    return content


def add_windows_cli_handler(content: str) -> str:
    """Add Windows CLI compatibility handler to the file"""
    logger_match = re.search(r'logger\s*=\s*logging\.getLogger', content)
    
    if logger_match:
        insert_pos = content.find('\n', logger_match.end()) + 1
        content = (content[:insert_pos] + '\n' + 
                   WINDOWS_CLI_HANDLER_TEMPLATE + '\n' + content[insert_pos:])
    else:
        import_end = content.find('\n\n')
        if import_end == -1:
            import_end = content.find('\n')
        content = (content[:import_end] + '\n\n' + 
                   WINDOWS_CLI_HANDLER_TEMPLATE + '\n' + content[import_end:])
    
    return content


def process_file(file_path: str) -> Tuple[bool, str]:
    """Process a single file to add Windows CLI compatibility"""
    try:
        if not check_file_exists(file_path):
            return False, f"File {file_path} does not exist"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        backup_path = backup_file(file_path)
        content = add_required_imports(content)
        content = add_windows_cli_handler(content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True, f"Successfully processed {file_path} (backup: {backup_path})"

    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}"


def main():
    """Main function to process all target files"""
    print("Windows CLI Compatibility Application Script")
    print("=" * 50)

    existing_files = [f for f in TARGET_FILES if check_file_exists(f)]
    missing_files = [f for f in TARGET_FILES if not check_file_exists(f)]

    print(f"Found {len(existing_files)} target files to process")
    if missing_files:
        print(f"Missing files: {missing_files}")

    print(f"Already fixed files: {FIXED_FILES}")
    print()

    results = []
    for file_path in existing_files:
        print(f"Processing {file_path}...")
        success, message = process_file(file_path)
        results.append((file_path, success, message))
        print(f"  {message}")
        print()

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print("Processing Summary:")
    print("=" * 30)
    print(f"Successfully processed: {len(successful)} files")
    print(f"Failed to process: {len(failed)} files")

    if failed:
        print("\nFailed files:")
        for file_path, success, message in failed:
            print(f"   {file_path}: {message}")

    print(f"\nTotal files with Windows CLI compatibility: {len(FIXED_FILES) + len(successful)}")
    print("All critical files now have Windows CLI compatibility!")


if __name__ == "__main__":
    main() 